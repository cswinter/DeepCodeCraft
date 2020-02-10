import logging
import subprocess
import time
import os
from collections import defaultdict

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

import wandb

from gym_codecraft import envs
from gym_codecraft.envs.codecraft_vec_env import ObsConfig
from hyper_params import HyperParams
from policy_t import TransformerPolicy
from policy_t2 import TransformerPolicy2, InputNorm
from policy_mem import PolicyTMem
import policy_t_old

logger = logging.getLogger(__name__)

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


def warmup_lr_schedule(warmup_steps: int):
    def lr(step):
        return (step + 1) / warmup_steps if step < warmup_steps else 1.0
    return lr


def train(hps: HyperParams, out_dir: str) -> None:
    assert(hps.rosteps % (hps.bs * hps.batches_per_update) == 0)
    assert(hps.eval_envs % 4 == 0)
    assert(hps.seq_rosteps % hps.tbptt_seq_len == 0)
    assert(hps.bs % hps.tbptt_seq_len == 0)

    next_model_save = hps.model_save_frequency

    obs_config = ObsConfig(
        allies=hps.obs_allies,
        drones=hps.obs_allies + hps.obs_enemies,
        minerals=hps.obs_minerals,
        global_drones=hps.obs_enemies if hps.use_privileged else 0,
        relative_positions=False,
        v2=True,
        obs_last_action=hps.obs_last_action,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    if hps.optimizer == 'SGD':
        optimizer_fn = optim.SGD
        optimizer_kwargs = dict(lr=hps.lr, momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'RMSProp':
        optimizer_fn = optim.RMSprop
        optimizer_kwargs = dict(lr=hps.lr, momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'Adam':
        optimizer_fn = optim.Adam
        optimizer_kwargs = dict(lr=hps.lr, weight_decay=hps.weight_decay, eps=1e-5)
    else:
        raise Exception(f'Invalid optimizer name `{hps.optimizer}`')

    resume_steps = 0
    if hps.resume_from == '':
        policy = PolicyTMem(
            hps.d_agent,
            hps.d_item,
            hps.dff_ratio,
            hps.nhead,
            hps.dropout,
            hps.small_init_pi,
            hps.zero_init_vf,
            hps.fp16,
            agents=hps.agents,
            nally=hps.nally,
            nenemy=hps.nenemy,
            nmineral=hps.nmineral,
            obs_config=obs_config,
            norm=hps.norm,
            use_privileged=hps.use_privileged,
            nearby_map=hps.nearby_map,
            ring_width=hps.nm_ring_width,
            nrays=hps.nm_nrays,
            nrings=hps.nm_nrings,
            map_conv=hps.map_conv,
            map_embed_offset=hps.map_embed_offset,
            keep_abspos=hps.obs_keep_abspos,
            ally_enemy_same=hps.ally_enemy_same,
            naction=hps.objective.naction(),
            memory=hps.tbptt_seq_len > 1,
            obs_last_action=hps.obs_last_action,
        ).to(device)
        optimizer = optimizer_fn(policy.parameters(), **optimizer_kwargs)
    else:
        policy, optimizer, resume_steps = load_policy(hps.resume_from, device, optimizer_fn, optimizer_kwargs, hps)

    lr_scheduler = None
    if hps.warmup > 0:
        warmup_steps = hps.sample_reuse * hps.warmup // hps.bs
        lr_scheduler = LambdaLR(optimizer, lr_lambda=[
            warmup_lr_schedule(warmup_steps),
            warmup_lr_schedule(warmup_steps),
            warmup_lr_schedule(warmup_steps),
        ])

    if hps.fp16:
        policy = policy.half()
        for layer in policy.modules():
            if isinstance(layer, InputNorm):
                layer.enable_fp16()

    wandb.watch(policy)

    total_steps = resume_steps
    next_eval = total_steps
    epoch = 0
    eprewmean = 0
    eplenmean = 0
    completed_episodes = 0
    env = None
    num_self_play_schedule = hps.get_num_self_play_schedule()
    hidden_state = policy.initial_hidden_state(hps.num_envs, device)
    while total_steps < hps.steps + resume_steps:
        if len(num_self_play_schedule) > 0 and num_self_play_schedule[-1][0] <= total_steps:
            _, num_self_play = num_self_play_schedule.pop()
            hps.num_self_play = num_self_play
            if env is not None:
                env.close()
                env = None
        if env is None:
            env = envs.CodeCraftVecEnv(hps.num_envs,
                                       hps.num_self_play,
                                       hps.objective,
                                       hps.action_delay,
                                       randomize=hps.task_randomize,
                                       use_action_masks=hps.use_action_masks,
                                       obs_config=obs_config,
                                       symmetric=hps.symmetric_map,
                                       hardness=hps.task_hardness)
            obs, action_masks = env.reset()
            action_masks = torch.tensor(action_masks).to(device)
            obs = torch.tensor(obs).to(device)

        if total_steps >= next_eval and hps.eval_envs > 0:
            eval(policy=policy,
                 num_envs=hps.eval_envs,
                 device=device,
                 objective=hps.objective,
                 eval_steps=hps.eval_timesteps,
                 curr_step=total_steps,
                 symmetric=hps.eval_symmetric)
            next_eval += hps.eval_frequency
            next_model_save -= 1
            if next_model_save == 0:
                next_model_save = hps.model_save_frequency
                save_policy(policy, out_dir, total_steps, optimizer)

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
        all_hidden_states = []
        all_cell_states = []

        policy.eval()
        with torch.no_grad():
            # Rollout
            for step in range(hps.seq_rosteps):
                if hps.tbptt_seq_len > 1 and step % hps.tbptt_seq_len == 0:
                    all_hidden_states.append(hidden_state[0].detach())
                    all_cell_states.append(hidden_state[1].detach())

                actions, logprobs, entropy, values, probs, hidden_state =\
                    policy.evaluate(obs, action_masks, hidden_state)

                entropies.extend(entropy.cpu().numpy())
                all_action_masks.append(action_masks)
                all_obs.append(obs)
                all_actions.append(actions)
                all_logprobs.append(logprobs.detach())
                all_probs.append(probs)

                all_values.extend(values.cpu().numpy())

                obs, rews, dones, infos, action_masks = env.step(actions.cpu().numpy())
                obs = torch.tensor(obs).to(device)
                action_masks = torch.tensor(action_masks).to(device)

                all_rewards.extend(rews)
                all_dones.extend(dones)

                for info in infos:
                    ema = min(95, completed_episodes * 10) / 100.0
                    eprewmean = eprewmean * ema + (1 - ema) * info['episode']['r']
                    eplenmean = eplenmean * ema + (1 - ema) * info['episode']['l']
                    completed_episodes += 1

            _, _, _, final_values, final_probs, hidden_state =\
                policy.evaluate(obs, action_masks, hidden_state)

        final_values = final_values.cpu().numpy()
        all_rewards = np.array(all_rewards) * hps.rewscale
        all_returns = np.zeros(len(all_rewards), dtype=np.float32)
        all_values = np.array(all_values)
        last_gae = np.zeros(hps.num_envs)
        for t in reversed(range(hps.seq_rosteps)):
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

        dtime = hps.tbptt_seq_len
        dbatch = (hps.seq_rosteps * hps.num_envs) // dtime

        def timeslice(tensor):
            return tensor.view(hps.seq_rosteps, hps.num_envs, -1).permute(1, 0, 2)\
                .reshape(dbatch, dtime, -1).permute(1, 0, 2)
        all_obs = timeslice(torch.cat(all_obs, dim=0))
        all_returns = timeslice(torch.tensor(all_returns).to(device))
        all_actions = timeslice(torch.cat(all_actions, dim=0))
        all_logprobs = timeslice(torch.cat(all_logprobs, dim=0))
        all_values = timeslice(torch.tensor(all_values).to(device))
        advantages = timeslice(torch.tensor(advantages).to(device))
        all_action_masks = timeslice(torch.cat(all_action_masks, dim=0)[:, :hps.agents, :]).view(dtime, dbatch, hps.agents, -1)
        all_probs = timeslice(torch.cat(all_probs, dim=0))
        if hps.tbptt_seq_len > 1:
            all_hidden_states = torch.cat(all_hidden_states, dim=0).unsqueeze(1).view(dbatch, hps.d_agent)
            all_cell_states = torch.cat(all_cell_states, dim=0).unsqueeze(1).view(dbatch, hps.d_agent)

        for epoch in range(hps.sample_reuse):
            if hps.shuffle:
                perm = np.random.permutation(dbatch)
                all_obs = all_obs[:, perm, :]
                all_returns = all_returns[:, perm, :]
                all_actions = all_actions[:, perm, :]
                all_logprobs = all_logprobs[:, perm, :]
                all_values = all_values[:, perm, :]
                advantages = advantages[:, perm, :]
                all_action_masks = all_action_masks[:, perm, :, :]
                all_probs = all_probs[:, perm, :]
                all_hidden_states = all_hidden_states[perm, :]
                all_cell_states = all_cell_states[perm, :]

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
                if batch % hps.batches_per_update == 0:
                    optimizer.zero_grad()

                start = hps.bs * batch // dtime
                end = hps.bs * (batch + 1) // dtime

                hidden = None
                if hps.tbptt_seq_len > 1:
                    hidden = (
                        all_hidden_states[start:end, :].unsqueeze(0),
                        all_cell_states[start:end, :].unsqueeze(0),
                    )
                policy_loss, value_loss, aproxkl, clipfrac = policy.backprop(
                    hps,
                    all_obs[:, start:end, :],
                    all_actions[:, start:end, :].reshape(-1),
                    all_logprobs[:, start:end, :].reshape(-1),
                    all_returns[:, start:end, :].reshape(-1),
                    hps.vf_coef,
                    advantages[:, start:end, :].reshape(-1),
                    all_values[:, start:end, :].reshape(-1),
                    all_action_masks[:, start:end, :, :].reshape(dtime * (end - start), hps.agents, -1),
                    all_probs[:, start:end, :].reshape(dtime * (end - start), -1),
                    hps.split_reward,
                    hidden_state=hidden
                )
                policy_loss_sum += policy_loss
                value_loss_sum += value_loss
                aproxkl_sum += aproxkl
                clipfrac_sum += clipfrac
                gradnorm += torch.nn.utils.clip_grad_norm_(policy.parameters(), hps.max_grad_norm)

                if (batch + 1) % hps.batches_per_update == 0:
                    optimizer.step()
                    if lr_scheduler:
                        lr_scheduler.step()

        epoch += 1
        total_steps += hps.rosteps
        throughput = int(hps.rosteps / (time.time() - episode_start))

        all_agent_masks = all_action_masks.sum(3) != 0
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
            'advantages': wandb.Histogram(advantages.cpu()),
            'values': wandb.Histogram(all_values.cpu()),
            'meanval': all_values.mean(),
            'returns': wandb.Histogram(all_returns.cpu()),
            'meanret': all_returns.mean(),
            'actions': wandb.Histogram(np.array(all_actions[all_agent_masks].cpu())),
            'observations': wandb.Histogram(all_obs.cpu()),
            'obs_max': all_obs.max(),
            'obs_min': all_obs.min(),
            'rewards': wandb.Histogram(np.array(all_rewards)),
            'masked_actions': 1 - all_action_masks.mean(),
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
        eval(policy=policy,
             num_envs=hps.eval_envs,
             device=device,
             objective=hps.objective,
             eval_steps=hps.eval_timesteps,
             curr_step=total_steps,
             symmetric=hps.eval_symmetric)
    save_policy(policy, out_dir, total_steps, optimizer)


def eval(policy,
         num_envs,
         device,
         objective,
         eval_steps,
         curr_step=None,
         opponents=None,
         printerval=None,
         randomize=False,
         hardness=10,
         symmetric=True):
    with torch.no_grad():

        if printerval is None:
            printerval = eval_steps

        if not opponents:
            if objective == envs.Objective.ARENA_TINY:
                opponents = {
                    'easy': {'model_file': 'arena_tiny/t2_random.pt'},
                }
            elif objective == envs.Objective.ARENA_TINY_2V2:
                opponents = {
                    'easy': {'model_file': 'arena_tiny_2v2/fine-sky-10M.pt'},
                }
            elif objective == envs.Objective.ARENA_MEDIUM:
                opponents = {
                    # Scores -0.32 vs previous best, jumping-totem-100M
                    'easy': {'model_file': 'arena_medium/copper-snow-25M.pt'},
                }
            elif objective == envs.Objective.ARENA:
                opponents = {
                    'beta': {'model_file': 'arena/glad-breeze-25M.pt'},
                }
            else:
                raise Exception(f'No eval opponents configured for {objective}')

        policy.eval()
        env = envs.CodeCraftVecEnv(num_envs,
                                   num_envs // 2,
                                   objective,
                                   action_delay=0,
                                   stagger=False,
                                   fair=not symmetric,
                                   use_action_masks=True,
                                   obs_config=policy.obs_config,
                                   randomize=randomize,
                                   hardness=hardness,
                                   symmetric=symmetric)

        scores = []
        scores_by_opp = defaultdict(list)
        lengths = []
        evens = list([2 * i for i in range(num_envs // 2)])
        odds = list([2 * i + 1 for i in range(num_envs // 2)])
        policy_envs = evens

        partitions = [(policy_envs, policy.obs_config)]
        i = 0
        hidden_state_opps = []
        for name, opp in opponents.items():
            opp_policy, _, _ = load_policy(opp['model_file'], device)
            opp_policy.eval()
            opp['policy'] = opp_policy
            opp['envs'] = odds[i * len(odds) // len(opponents):(i+1) * len(odds) // len(opponents)]
            opp['obs_config'] = opp_policy.obs_config
            opp['i'] = i
            i += 1
            partitions.append((opp['envs'], opp_policy.obs_config))
            hidden_state_opps.append(opp_policy.initial_hidden_state(len(opp['envs']), device))

        initial_obs = env.reset(partitions)

        obs, action_masks = initial_obs[0]
        hidden_state = policy.initial_hidden_state(len(evens), device)
        obs_opps, action_masks_opps = ([], [])
        for o, a in initial_obs[1:]:
            obs_opps.append(o)
            action_masks_opps.append(a)


        for step in range(eval_steps):
            actionsp, _, _, _, _, hidden_state = policy.evaluate(
                torch.tensor(obs).to(device),
                torch.tensor(action_masks).to(device),
                hidden_state)
            env.step_async(actionsp.cpu(), policy_envs)

            for _, opp in opponents.items():
                i = opp['i']
                actions_opp, _, _, _, _, hidden_state_opps[i] = opp['policy'].evaluate(
                    torch.tensor(obs_opps[i]).to(device),
                    torch.tensor(action_masks_opps[i]).to(device),
                    hidden_state_opps[i])
                env.step_async(actions_opp.cpu(), opp['envs'])

            obs, _, _, infos, action_masks = env.observe(policy_envs)
            for _, opp in opponents.items():
                i = opp['i']
                obs_opps[i], _, _, _, action_masks_opps[i] = \
                    env.observe(opp['envs'], opp['obs_config'])

            for info in infos:
                index = info['episode']['index']
                score = info['episode']['score']
                length = info['episode']['l']
                scores.append(score)
                lengths.append(length)
                for name, opp in opponents.items():
                    if index + 1 in opp['envs']:
                        scores_by_opp[name].append(score)
                        break

            if (step + 1) % printerval == 0:
                print(f'Eval: {np.array(scores).mean()}')

        scores = np.array(scores)

        if curr_step is not None:
            wandb.log({
                'eval_mean_score': scores.mean(),
                'eval_max_score': scores.max(),
                'eval_min_score': scores.min(),
            }, step=curr_step)
            for opp_name, scores in scores_by_opp.items():
                scores = np.array(scores)
                wandb.log({f'eval_mean_score_vs_{opp_name}': scores.mean()}, step=curr_step)

        env.close()


def save_policy(policy, out_dir, total_steps, optimizer=None):
    model_path = os.path.join(out_dir, f'model-{total_steps}.pt')
    print(f'Saving policy to {model_path}')
    model = {
        'model_state_dict': policy.state_dict(),
        'model_kwargs': policy.kwargs,
        'total_steps': total_steps,
        'policy_version': policy.version,
    }
    if optimizer:
        model['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(model, model_path)


def load_policy(name, device, optimizer_fn=None, optimizer_kwargs=None, hps=None):
    checkpoint = torch.load(os.path.join(EVAL_MODELS_PATH, name), map_location=device)
    version = checkpoint.get('policy_version')
    kwargs = checkpoint['model_kwargs']
    if hps:
        kwargs['obs_config'] = ObsConfig(
            allies=hps.obs_allies,
            drones=hps.obs_allies + hps.obs_enemies,
            minerals=hps.obs_minerals,
            global_drones=hps.obs_enemies if hps.use_privileged else 0,
            relative_positions=False,
            v2=True,
        )
    if version == 'transformer_v1':
        # This model didn't have the params on it's Normalize layer saved - need to backfill manually :(
        if name.endswith("jumping-totem-100M.pt"):
            policy = policy_t_old.TransformerPolicy(**kwargs)
            policy.self_embedding.normalize.count = 382405392.0
            policy.self_embedding.normalize.mean = torch.tensor(
                [0.1220, -0.0122, -0.0332, -0.0351, -0.0493,  0.0661, -0.3715, -0.5018, 0.4990,  0.4910,
                 0.2545,  0.4910,  0.0000,  0.0000, -0.8230,  1.0000],
            ).to(device)
            policy.self_embedding.normalize.stddev = lambda: torch.tensor(
                [0.0890, 0.3567, 0.3637, 0.7138, 0.6977, 0.0892, 0.9284, 0.8650, 0.2667,
                 0.4999, 0.2500, 0.4999, 0.0000, 0.0000, 0.5680, 0.0000]
            ).to(device)
            policy.mineral_embedding.normalize.count = 1465606824.0
            policy.mineral_embedding.normalize.mean = torch.tensor([0.0146, 0.0236, 0.5068, 0.2037]).to(device)
            policy.mineral_embedding.normalize.stddev = lambda: torch.tensor([0.4094, 0.4130, 0.2866, 0.1520]).to(device)
            policy.drone_embedding.normalize.count = 889158346.0
            policy.drone_embedding.normalize.mean = torch.tensor(
                [-2.7168e-04, -1.1618e-02, -4.6683e-02, -2.0150e-02,  4.4679e-02,
                 -5.9865e-01, -7.7697e-01,  3.7921e-01,  2.8120e-01,  3.5940e-01,
                 2.8120e-01,  0.0000e+00,  0.0000e+00, -7.5037e-01,  1.2427e-01]
            ).to(device)
            policy.drone_embedding.normalize.stddev = lambda: torch.tensor(
                [0.3811, 0.3875, 0.7165, 0.6957, 0.0831, 0.8010, 0.6295, 0.2315, 0.4496,
                 0.2248, 0.4496, 0.0000, 0.0000, 0.6610, 0.9922]
            ).to(device)
        else:
            policy = TransformerPolicy(**kwargs)
    elif version == 'transformer_v2':
        policy = TransformerPolicy2(**kwargs)
    elif version == 'transformer_lstm':
        policy = PolicyTMem(**kwargs)
        # policy = TransformerPolicy2(**kwargs)
    else:
        raise Exception(f"Unknown policy version {version}")

    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.to(device)

    optimizer = None
    if optimizer_fn:
        optimizer = optimizer_fn(policy.parameters(), **optimizer_kwargs)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            logger.warning(f'Failed to restore optimizer state: No `optimizer_state_dict` in saved model.')

    return policy, optimizer, checkpoint.get('total_steps', 0)


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
    # torch.set_printoptions(threshold=25000)

    hps = HyperParams()
    args_parser = hps.args_parser()
    args_parser.add_argument("--out-dir")
    args_parser.add_argument("--device", default=0)
    args_parser.add_argument("--descriptor", default="none")
    args_parser.add_argument("--hpset", default="default")
    args = args_parser.parse_args()
    if args.hpset == 'allied_wealth':
        hps = HyperParams.allied_wealth()
    elif args.hpset == 'arena_tiny':
        hps = HyperParams.arena_tiny()
    elif args.hpset == 'arena_tiny_2v2':
        hps = HyperParams.arena_tiny_2v2()
    elif args.hpset == 'arena_medium':
        hps = HyperParams.arena_medium()
    elif args.hpset == 'arena':
        hps = HyperParams.arena()
    elif args.hpset != 'default':
        raise Exception(f"Unknown hpset `{args.hpset}`")
    for key, value in vars(args).items():
        if value is not None and hasattr(hps, key):
            setattr(hps, key, value)

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

