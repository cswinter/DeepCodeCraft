import logging
import subprocess
import time
import os

import torch
import torch.optim as optim
import numpy as np

import wandb

from gym_codecraft import envs
from hyper_params import HyperParams
from policy import Policy

TEST_LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft_test'
LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft'
EVAL_MODELS_PATH = '/home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models/v1'


def run_codecraft():
    nenv = 32
    env = envs.CodeCraftVecEnv(nenv, envs.Objective.DISTANCE_TO_ORIGIN, 1, action_delay=0)

    log_interval = 5
    frames = 0
    last_time = time.time()

    env.reset()
    while True:
        elapsed = time.time() - last_time
        if elapsed > log_interval:
            print(f"{frames / elapsed}fps")
            frames = 0
            last_time = time.time()

        env.step_async([4]*nenv)
        env.observe()
        frames += nenv


def train(hps: HyperParams) -> None:
    assert(hps.rosteps % hps.bs == 0)
    assert(hps.eval_envs % 4 == 0)

    next_eval = 0

    env = envs.CodeCraftVecEnv(hps.num_envs,
                               hps.num_self_play,
                               hps.objective,
                               hps.action_delay)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    policy = Policy(hps.depth, hps.width, hps.conv, hps.small_init_pi, hps.zero_init_vf, hps.fp16).to(device)
    if hps.fp16:
        policy = policy.half()
    if hps.optimizer == 'SGD':
        optimizer = optim.SGD(policy.parameters(), lr=hps.lr, momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'RMSProp':
        optimizer = optim.RMSprop(policy.parameters(), lr=hps.lr, momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'Adam':
        optimizer = optim.Adam(policy.parameters(), lr=hps.lr, weight_decay=hps.weight_decay, eps=1e-5)
    else:
        raise Exception(f'Invalid optimizer name `{hps.optimizer}`')

    wandb.watch(policy)

    total_steps = 0
    epoch = 0
    obs = env.reset()
    eprewmean = 0
    eplenmean = 0
    completed_episodes = 0
    while total_steps < hps.steps:
        if total_steps >= next_eval and hps.eval_envs > 0:
            eval_return = eval(policy, hps, device)
            print(f'Eval: {eval_return}')
            wandb.log({'eval_return': eval_return}, step=total_steps)
            next_eval += hps.eval_frequency

        episode_start = time.time()
        entropies = []
        all_obs = []
        all_actions = []
        all_logprobs = []
        all_values = []
        all_rewards = []
        all_dones = []

        policy.eval()
        torch.no_grad()
        # Rollout
        for step in range(hps.seq_rosteps):
            obs_tensor = torch.tensor(obs).to(device)
            actions, logprobs, entropy, values = policy.evaluate(obs_tensor)
            actions = actions.cpu().numpy()

            entropies.extend(entropy.detach().cpu().numpy())

            all_obs.extend(obs)
            all_actions.extend(actions)
            all_logprobs.extend(logprobs.detach().cpu().numpy())
            all_values.extend(values)

            obs, rews, dones, infos = env.step(actions)

            all_rewards.extend(rews)
            all_dones.extend(dones)

            for info in infos:
                ema = min(95, completed_episodes * 10) / 100.0
                eprewmean = eprewmean * ema + (1 - ema) * info['episode']['r']
                eplenmean = eplenmean * ema + (1 - ema) * info['episode']['l']
                completed_episodes += 1

        obs_tensor = torch.tensor(obs).to(device)
        _, _, _, final_values = policy.evaluate(obs_tensor)

        all_rewards = np.array(all_rewards) * hps.rewscale
        all_returns = np.zeros(len(all_rewards), dtype=np.float32)
        all_values = np.array(all_values)
        last_gae = np.zeros(hps.num_envs)
        for t in reversed(range(hps.seq_rosteps)):
            # TODO: correct for action delay?
            # TODO: vectorize
            for i in range(hps.num_envs):
                ti = t * hps.num_envs + i
                tnext_i = (t + 1) * hps.num_envs + i
                nextnonterminal = 1.0 - all_dones[ti]
                if t == hps.seq_rosteps - 1:
                    next_value = final_values[i]
                else:
                    next_value = all_values[tnext_i]
                td_error = all_rewards[ti] + hps.gamma * next_value * nextnonterminal - all_values[ti]
                last_gae[i] = td_error + hps.gamma * hps.lamb * last_gae[i] * nextnonterminal
                all_returns[ti] = last_gae[i] + all_values[ti]

        advantages = all_returns - all_values
        if hps.norm_advs:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        explained_var = explained_variance(all_values, all_returns)

        all_actions = np.array(all_actions)
        all_logprobs = np.array(all_logprobs)
        all_obs = np.array(all_obs)

        for epoch in range(hps.sample_reuse):
            if hps.shuffle:
                perm = np.random.permutation(len(all_obs))
                all_obs = all_obs[perm]
                all_returns = all_returns[perm]
                all_actions = all_actions[perm]
                all_logprobs = all_logprobs[perm]
                all_values = all_values[perm]
                advantages = advantages[perm]

            # Policy Update
            policy_loss_sum = 0
            value_loss_sum = 0
            clipfrac_sum = 0
            aproxkl_sum = 0
            gradnorm = 0
            policy.train()
            torch.enable_grad()
            num_minibatches = int(hps.rosteps / hps.bs)
            for batch in range(num_minibatches):
                start = hps.bs * batch
                end = hps.bs * (batch + 1)

                o = torch.tensor(all_obs[start:end]).to(device)
                actions = torch.tensor(all_actions[start:end]).to(device)
                probs = torch.tensor(all_logprobs[start:end]).to(device)
                returns = torch.tensor(all_returns[start:end]).to(device)
                advs = torch.tensor(advantages[start:end]).to(device)
                vals = torch.tensor(all_values[start:end]).to(device)

                optimizer.zero_grad()
                policy_loss, value_loss, aproxkl, clipfrac =\
                    policy.backprop(hps, o, actions, probs, returns, hps.vf_coef, advs, vals)
                policy_loss_sum += policy_loss
                value_loss_sum += value_loss
                aproxkl_sum += aproxkl
                clipfrac_sum += clipfrac
                gradnorm += torch.nn.utils.clip_grad_norm_(policy.parameters(), hps.max_grad_norm)
                optimizer.step()

        epoch += 1
        total_steps += hps.rosteps
        throughput = int(hps.rosteps / (time.time() - episode_start))

        metrics = {
            'loss': policy_loss_sum / num_minibatches,
            'value_loss': value_loss_sum / num_minibatches,
            'clipfrac': clipfrac_sum / num_minibatches,
            'aproxkl': aproxkl_sum / num_minibatches,  # TODO: is average a good summary?
            'throughput': throughput,
            'eprewmean': eprewmean,
            'eplenmean': eplenmean,
            'entropy': sum(entropies) / len(entropies) / np.log(2),
            'explained variance': explained_var,
            'gradnorm': gradnorm * hps.bs / hps.rosteps,
            'advantages': wandb.Histogram(advantages),
            'values': wandb.Histogram(all_values),
            'meanval': all_values.mean(),
            'returns': wandb.Histogram(all_returns),
            'meanret': all_returns.mean(),
            'actions': wandb.Histogram(np.array(all_actions)),
            'observations': wandb.Histogram(np.array(all_obs)),
            'obs_max': all_obs.max(),
            'obs_min': all_obs.min(),
            'rewards': wandb.Histogram(np.array(all_rewards)),
        }
        total_norm = 0.0
        count = 0
        for name, param in policy.named_parameters():
            norm = param.data.norm()
            metrics[f'weight_norm[{name}]'] = norm
            count += 1
            total_norm += norm
        metrics['mean_weight_norm'] = total_norm / count

        wandb.log(metrics, step=total_steps)

        print(f'{throughput} samples/s')

    env.close()

    if hps.eval_envs > 0:
        eval_return = eval(policy, hps, device)
        print(f'Final eval: {eval_return}')
        wandb.log({'eval_return': eval_return}, step=total_steps)

    t = time.strftime("%Y-%m-%d~%H:%M:%S")
    commit = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")[:-1]
    out_dir = os.path.join(LOG_ROOT_DIR, f"{t}-{commit}")
    os.mkdir(out_dir)
    model_path = os.path.join(out_dir, 'model.pt')
    print(f'Saving final policy to {model_path}')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'model_kwargs': policy.kwargs,
    }, model_path)


def eval(policy, hps, device) -> float:
    opp_hps = HyperParams()
    opp_hps.depth = 4
    opp_hps.width = 1024
    opp_hps.conv = True
    opp_hps.fp16 = False
    opp_hps.small_init_pi = False
    opp_hps.zero_init_vf = True

    checkpoint = torch.load(os.path.join(EVAL_MODELS_PATH, '21011a1-1M.pt'))
    opponent = Policy(**checkpoint['model_kwargs']).to(device)
    opponent.load_state_dict(checkpoint['model_state_dict'])

    policy.eval()
    opponent.eval()

    env = envs.CodeCraftVecEnv(hps.eval_envs,
                               hps.eval_envs // 2,
                               hps.objective,
                               hps.action_delay,
                               stagger=False)

    returns = []
    lengths = []
    obs = env.reset()
    evens = list([2 * i for i in range(hps.eval_envs // 2)])
    odds = list([2 * i + 1 for i in range(hps.eval_envs // 2)])
    policy_envs = evens[:hps.eval_envs // 4] + odds[hps.eval_envs // 4:]
    opponent_envs = odds[:hps.eval_envs // 4] + evens[hps.eval_envs // 4:]
    for step in range(hps.eval_timesteps):
        obs_tensor = torch.tensor(obs).to(device)
        obs_policy = obs_tensor[policy_envs]
        obs_opponent = obs_tensor[opponent_envs]
        actionsp, _, _, _ = policy.evaluate(obs_policy)
        actionso, _, _, _ = opponent.evaluate(obs_opponent)

        actions = np.zeros(hps.eval_envs, dtype=np.int)
        actions[policy_envs] = actionsp.cpu()
        actions[opponent_envs] = actionso.cpu()

        obs, rews, dones, infos = env.step(actions)

        for info in infos:
            index = info['episode']['index']
            if index in policy_envs:
                returns.append(info['episode']['r'])
                lengths.append(info['episode']['l'])

    env.close()

    return np.array(returns).mean()


def explained_variance(ypred,y):
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y-ypred)/vary


def main():
    logging.basicConfig(level=logging.INFO)
    wandb.init(project="deep-codecraft")

    hps = HyperParams()
    args_parser = hps.args_parser()
    args_parser.add_argument("--out-dir")
    args_parser.add_argument("--device", default=0)
    args_parser.add_argument("--descriptor", default="none")
    args = args_parser.parse_args()
    for key, value in vars(args).items():
        if value is not None and hasattr(hps, key):
            setattr(hps, key, value)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    config = vars(hps)
    config['commit'] = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")[:-1]
    config['descriptor'] = vars(args)['descriptor']
    wandb.config.update(config)

    if isinstance(hps.objective, str):
        hps.objective = envs.Objective(hps.objective)

    train(hps)

"""
    if args.out_dir:
        out_dir = args.out_dir
    else:
        commit = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")
        t = time.strftime("%Y-%m-%d~%H:%M:%S")
        out_dir = os.path.join(TEST_LOG_ROOT_DIR, f"{t}-{commit}")

    hps = {}
    for hp in HYPER_PARAMS:
        if args_dict[hp.shortname] is not None:
            hps[hp.shortname] = args_dict[hp.shortname]
            if args.out_dir is None:
                out_dir += f"-{hp.shortname}{hps[hp.shortname]}"
        else:
            hps[hp.shortname] = hp.default

    logger.configure(dir=out_dir, format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    train(hps)
"""


if __name__ == "__main__":
    main()

