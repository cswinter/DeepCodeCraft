import logging
import subprocess
import time
import os
from collections import defaultdict

import torch
import torch.optim as optim
import numpy as np

import wandb

from gym_codecraft import envs
from hyper_params import HyperParams
from policy import Policy

TEST_LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft_test'
LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft'
EVAL_MODELS_PATH = '/home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models'


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


def train(hps: HyperParams, out_dir: str) -> None:
    assert(hps.rosteps % hps.bs == 0)
    assert(hps.eval_envs % 4 == 0)

    next_eval = 0
    next_model_save = hps.model_save_frequency

    env = envs.CodeCraftVecEnv(hps.num_envs,
                               hps.num_self_play,
                               hps.objective,
                               hps.action_delay,
                               randomize=True,
                               use_action_masks=hps.use_action_masks)
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
    obs, action_masks = env.reset()
    eprewmean = 0
    eplenmean = 0
    completed_episodes = 0
    while total_steps < hps.steps:
        if total_steps >= next_eval and hps.eval_envs > 0:
            eval(policy, hps, device, total_steps)
            next_eval += hps.eval_frequency
            next_model_save -= 1
            if next_model_save == 0:
                next_model_save = hps.model_save_frequency
                save_policy(policy, out_dir, total_steps)

        episode_start = time.time()
        entropies = []
        all_obs = []
        all_actions = []
        all_probs = []
        all_logprobs = []
        all_values = []
        all_rewards = []
        all_dones = []
        all_action_masks = []

        policy.eval()
        torch.no_grad()
        # Rollout
        for step in range(hps.seq_rosteps):
            obs_tensor = torch.tensor(obs).to(device)
            action_masks_tensor = torch.tensor(action_masks).to(device)
            actions, logprobs, entropy, values, probs = policy.evaluate(obs_tensor, action_masks_tensor)
            actions = actions.cpu().numpy()

            entropies.extend(entropy.detach().cpu().numpy())

            all_action_masks.extend(action_masks)
            all_obs.extend(obs)
            all_actions.extend(actions)
            all_logprobs.extend(logprobs.detach().cpu().numpy())
            all_values.extend(values)
            all_probs.extend(probs)

            obs, rews, dones, infos, action_masks = env.step(actions)

            all_rewards.extend(rews)
            all_dones.extend(dones)

            for info in infos:
                ema = min(95, completed_episodes * 10) / 100.0
                eprewmean = eprewmean * ema + (1 - ema) * info['episode']['r']
                eplenmean = eplenmean * ema + (1 - ema) * info['episode']['l']
                completed_episodes += 1

        obs_tensor = torch.tensor(obs).to(device)
        action_masks_tensor = torch.tensor(action_masks).to(device)
        _, _, _, final_values, final_probs = policy.evaluate(obs_tensor, action_masks_tensor)

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
        all_action_masks = np.array(all_action_masks)
        all_probs = np.array(all_probs)

        for epoch in range(hps.sample_reuse):
            if hps.shuffle:
                perm = np.random.permutation(len(all_obs))
                all_obs = all_obs[perm]
                all_returns = all_returns[perm]
                all_actions = all_actions[perm]
                all_logprobs = all_logprobs[perm]
                all_values = all_values[perm]
                advantages = advantages[perm]
                all_action_masks = all_action_masks[perm]
                all_probs = all_probs[perm]

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
                amasks = torch.tensor(all_action_masks[start:end]).to(device)
                actual_probs = torch.tensor(all_probs[start:end]).to(device)

                optimizer.zero_grad()
                policy_loss, value_loss, aproxkl, clipfrac =\
                    policy.backprop(hps, o, actions, probs, returns, hps.vf_coef, advs, vals, amasks, actual_probs)
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
            'masked_actions': all_action_masks.mean(),
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
        eval(policy, hps, device, total_steps)
    save_policy(policy, out_dir, total_steps)


def save_policy(policy, out_dir, total_steps):
    model_path = os.path.join(out_dir, f'model-{total_steps}.pt')
    print(f'Saving policy to {model_path}')
    torch.save({
        'model_state_dict': policy.state_dict(),
        'model_kwargs': policy.kwargs,
    }, model_path)


def eval(policy, hps, device, total_steps):

    if hps.objective == envs.Objective.ARENA_TINY:
        opponents = {
            '10m': {'model_file': 'v1/25909ee-10M.pt'},
            '1m': {'model_file': 'v1/21011a1-1M.pt'},
        }
    elif hps.objective == envs.Objective.ARENA_TINY_2V2:
        opponents = {
            'random': {'model_file': 'v3/random.pt'},
            'easy': {'model_file': 'v3/helpful-glade-10M.pt'},
        }
    else:
        raise Exception(f'No eval opponents configured for {hps.objective}')

    policy.eval()
    env = envs.CodeCraftVecEnv(hps.eval_envs,
                               hps.eval_envs // 2,
                               hps.objective,
                               hps.action_delay,
                               stagger=False,
                               fair=True,
                               use_action_masks=hps.use_action_masks)

    scores = []
    scores_by_opp = defaultdict(list)
    lengths = []
    obs, action_masks = env.reset()
    evens = list([2 * i for i in range(hps.eval_envs // 2)])
    odds = list([2 * i + 1 for i in range(hps.eval_envs // 2)])
    policy_envs = evens

    i = 0
    for name, opp in opponents.items():
        opp['policy'] = load_policy(opp['model_file']).to(device)
        opp['envs'] = odds[i * len(odds) // len(opponents):(i+1) * len(odds) // len(opponents)]
        i += 1

    for step in range(hps.eval_timesteps):
        actions = np.zeros((hps.eval_envs, 2), dtype=np.int)
        obs_tensor = torch.tensor(obs).to(device)
        action_masks_tensor = torch.tensor(action_masks).to(device)
        obs_policy = obs_tensor[policy_envs]
        action_masks_policy = action_masks_tensor[policy_envs]
        actionsp, _, _, _, _ = policy.evaluate(obs_policy, action_masks_policy)
        actions[policy_envs] = actionsp.cpu()

        for _, opp in opponents.items():
            obs = obs_tensor[opp['envs']]
            action_masks_opp = action_masks_tensor[opp['envs']]
            actions_opp, _, _, _, _ = opp['policy'].evaluate(obs, action_masks_opp)
            actions[opp['envs']] = actions_opp.cpu()

        obs, rews, dones, infos, action_masks = env.step(actions)

        for info in infos:
            index = info['episode']['index']
            if index in policy_envs:
                score = info['episode']['score']
                length = info['episode']['l']
                scores.append(score)
                lengths.append(length)
                for name, opp in opponents.items():
                    if index + 1 in opp['envs']:
                        scores_by_opp[name].append(score)
                        break

    scores = np.array(scores)
    wandb.log({
        'eval_mean_score': scores.mean(),
        'eval_max_score': scores.max(),
        'eval_min_score': scores.min(),
    }, step=total_steps)
    for opp_name, scores in scores_by_opp.items():
        scores = np.array(scores)
        wandb.log({f'eval_mean_score_vs_{opp_name}': scores.mean()}, step=total_steps)
    print(f'Eval: {scores.mean()}')

    env.close()


def load_policy(name):
    checkpoint = torch.load(os.path.join(EVAL_MODELS_PATH, name))
    policy = Policy(**checkpoint['model_kwargs'])
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    return policy


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

    if isinstance(hps.objective, str):
        hps.objective = envs.Objective(hps.objective)

    wandb_project = 'deep-codecraft-vs' if hps.objective.vs() else 'deep-codecraft'
    wandb.init(project=wandb_project)
    wandb.config.update(config)

    if not args.out_dir:
        t = time.strftime("%Y-%m-%d~%H:%M:%S")
        commit = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")[:-1]
        out_dir = os.path.join(LOG_ROOT_DIR, f"{t}-{commit}")
        os.mkdir(out_dir)
    else:
        out_dir = args.out_dir

    train(hps, out_dir)

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

