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

from adr import ADR
from gym_codecraft import envs
from gym_codecraft.envs.codecraft_vec_env import ObsConfig
from hyper_params import HyperParams
from policy_t2 import TransformerPolicy2, InputNorm
from policy_t3 import TransformerPolicy3, InputNorm
from policy_t4 import TransformerPolicy4, InputNorm
from policy_t5 import TransformerPolicy5, InputNorm
from policy_t6 import TransformerPolicy6, InputNorm

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

    next_model_save = hps.model_save_frequency

    obs_config = ObsConfig(
        allies=hps.obs_allies,
        drones=hps.obs_allies + hps.obs_enemies,
        minerals=hps.obs_minerals,
        tiles=hps.obs_map_tiles,
        global_drones=hps.obs_enemies if hps.use_privileged else 0,
        relative_positions=False,
        feat_last_seen=hps.feat_last_seen,
        feat_is_visible=hps.feat_is_visible,
        feat_map_size=hps.feat_map_size,
        feat_abstime=hps.feat_abstime,
        v2=True,
        feat_rule_msdm=hps.rule_rng_fraction > 0 or hps.adr,
        feat_rule_costs=hps.rule_cost_rng > 0 or hps.adr,
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
        policy = TransformerPolicy6(hps, obs_config).to(device)
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
    eliminationmean = 0
    buildmean = defaultdict(lambda: 0)
    completed_episodes = 0
    env = None
    num_self_play_schedule = hps.get_num_self_play_schedule()
    batches_per_update_schedule = hps.get_batches_per_update_schedule()
    entropy_bonus_schedule = hps.get_entropy_bonus_schedule()
    adr = ADR()
    rewmean = 0.0
    rewstd = 1.0
    average_cost_modifier = 1.0
    while total_steps < hps.steps + resume_steps:
        if len(num_self_play_schedule) > 0 and num_self_play_schedule[-1][0] <= total_steps:
            _, num_self_play = num_self_play_schedule.pop()
            hps.num_self_play = num_self_play
            if env is not None:
                env.close()
                env = None

        if len(batches_per_update_schedule) > 0 and batches_per_update_schedule[-1][0] <= total_steps:
            _, batches_per_update = batches_per_update_schedule.pop()
            hps.batches_per_update = batches_per_update
            assert(hps.rosteps % (hps.bs * hps.batches_per_update) == 0)

        if len(entropy_bonus_schedule) > 0 and entropy_bonus_schedule[-1][0] <= total_steps:
            _, entropy_bonus = entropy_bonus_schedule.pop()
            hps.entropy_bonus = entropy_bonus

        if env is None:
            env = envs.CodeCraftVecEnv(hps.num_envs,
                                       hps.num_self_play,
                                       hps.objective,
                                       hps.action_delay,
                                       randomize=hps.task_randomize,
                                       use_action_masks=hps.use_action_masks,
                                       obs_config=obs_config,
                                       symmetric=hps.symmetric_map,
                                       hardness=hps.task_hardness,
                                       mix_mp=hps.mix_mp,
                                       build_variety_bonus=hps.build_variety_bonus,
                                       win_bonus=hps.win_bonus,
                                       attac=hps.attac,
                                       protec=hps.protec,
                                       max_army_size_score=hps.max_army_size_score,
                                       max_enemy_army_size_score=hps.max_enemy_army_size_score,
                                       rule_rng_fraction=hps.rule_rng_fraction,
                                       rule_rng_amount=hps.rule_rng_amount,
                                       rule_cost_rng=hps.rule_cost_rng)
            obs, action_masks, privileged_obs = env.reset()

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
        all_privileged_obs = []

        policy.eval()
        buildtotal = defaultdict(lambda: 0)
        eliminations = []
        if hps.adr:
            env.rng_ruleset = adr.ruleset
            env.hardness = adr.hardness
        with torch.no_grad():
            # Rollout
            for step in range(hps.seq_rosteps):
                obs_tensor = torch.tensor(obs).to(device)
                privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
                action_masks_tensor = torch.tensor(action_masks).to(device)
                actions, logprobs, entropy, values, probs =\
                    policy.evaluate(obs_tensor, action_masks_tensor, privileged_obs_tensor)
                actions = actions.cpu().numpy()

                entropies.extend(entropy.detach().cpu().numpy())

                all_action_masks.extend(action_masks)
                all_obs.extend(obs)
                all_privileged_obs.extend(privileged_obs)
                all_actions.extend(actions)
                all_logprobs.extend(logprobs.detach().cpu().numpy())
                all_values.extend(values)
                all_probs.extend(probs)

                obs, rews, dones, infos, action_masks, privileged_obs = env.step(actions, action_masks=action_masks)

                rews -= hps.liveness_penalty
                all_rewards.extend(rews)
                all_dones.extend(dones)

                for info in infos:
                    ema = 0.95 * (1 - 1 / (completed_episodes + 1))

                    win_by_elimination = 1 if info['episode']['outcome'] in [-1, 1] else 0
                    eliminations.append(win_by_elimination)
                    eliminationmean = eliminationmean * ema + (1 - ema) * win_by_elimination

                    eprewmean = eprewmean * ema + (1 - ema) * info['episode']['r']
                    eplenmean = eplenmean * ema + (1 - ema) * info['episode']['l']

                    builds = info['episode']['builds']
                    for build in set().union(builds.keys(), buildmean.keys()):
                        count = builds[build]
                        buildmean[build] = buildmean[build] * ema + (1 - ema) * count
                        buildtotal[build] += count
                    completed_episodes += 1

        average_cost_modifier = adr.adjust(buildtotal, np.array(eliminations).mean() if len(eliminations) > 0 else None)

        obs_tensor = torch.tensor(obs).to(device)
        action_masks_tensor = torch.tensor(action_masks).to(device)
        privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
        _, _, _, final_values, final_probs =\
            policy.evaluate(obs_tensor, action_masks_tensor, privileged_obs_tensor)

        all_rewards = np.array(all_rewards) * hps.rewscale
        w = hps.rewnorm_emaw * (1 - 1 / (total_steps + 1))
        rewmean = all_rewards.mean() * (1 - w) + rewmean * w
        rewstd = all_rewards.std() * (1 - w) + rewstd * w
        if hps.rewnorm:
            all_rewards = all_rewards / rewstd - rewmean

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
        all_privileged_obs = np.array(all_privileged_obs)
        all_action_masks = np.array(all_action_masks)[:, :hps.agents, :]
        all_probs = np.array(all_probs)

        for epoch in range(hps.sample_reuse):
            if hps.shuffle:
                perm = np.random.permutation(len(all_obs))
                all_obs = all_obs[perm]
                all_privileged_obs = all_privileged_obs[perm]
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
                if batch % hps.batches_per_update == 0:
                    optimizer.zero_grad()

                start = hps.bs * batch
                end = hps.bs * (batch + 1)

                o = torch.tensor(all_obs[start:end]).to(device)
                op = torch.tensor(all_privileged_obs[start:end]).to(device)
                actions = torch.tensor(all_actions[start:end]).to(device)
                probs = torch.tensor(all_logprobs[start:end]).to(device)
                returns = torch.tensor(all_returns[start:end]).to(device)
                advs = torch.tensor(advantages[start:end]).to(device)
                vals = torch.tensor(all_values[start:end]).to(device)
                amasks = torch.tensor(all_action_masks[start:end]).to(device)
                actual_probs = torch.tensor(all_probs[start:end]).to(device)

                policy_loss, value_loss, aproxkl, clipfrac =\
                    policy.backprop(hps, o, actions, probs, returns, hps.vf_coef,
                                    advs, vals, amasks, actual_probs, op, hps.split_reward)
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

        all_agent_masks = all_action_masks.sum(2) != 0
        metrics = {
            'loss': policy_loss_sum / num_minibatches,
            'value_loss': value_loss_sum / num_minibatches,
            'clipfrac': clipfrac_sum / num_minibatches,
            'aproxkl': aproxkl_sum / num_minibatches,  # TODO: is average a good summary?
            'throughput': throughput,
            'eprewmean': eprewmean,
            'eplenmean': eplenmean,
            'eliminationmean': eliminationmean,
            'entropy': sum(entropies) / len(entropies) / np.log(2),
            'explained variance': explained_var,
            'gradnorm': gradnorm * hps.bs / hps.rosteps,
            'advantages': wandb.Histogram(advantages),
            'values': wandb.Histogram(all_values),
            'meanval': all_values.mean(),
            'returns': wandb.Histogram(all_returns),
            'meanret': all_returns.mean(),
            'actions': wandb.Histogram(np.array(all_actions[all_agent_masks])),
            'observations': wandb.Histogram(np.array(all_obs)),
            'obs_max': all_obs.max(),
            'obs_min': all_obs.min(),
            'rewards': wandb.Histogram(np.array(all_rewards)),
            'masked_actions': 1 - all_action_masks.mean(),
            'rewmean': rewmean,
            'rewstd': rewstd,
            'average_cost_modifier': average_cost_modifier,
            'hardness': adr.hardness,
        }
        for action, count in buildmean.items():
            metrics[f'build_{action}'] = count
        metrics.update(adr.metrics())
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
         symmetric=True,
         random_rules=0.0):
    if printerval is None:
        printerval = eval_steps

    if opponents is None:
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
        elif objective == envs.Objective.ARENA_MEDIUM_LARGE_MS:
            opponents = {
                'easy': {'model_file': 'arena_medium_large_ms/honest-violet-50M.pt'},
            }
        elif objective == envs.Objective.ARENA:
            opponents = {
                'beta': {'model_file': 'arena/glad-breeze-25M.pt'},
            }
        elif objective == envs.Objective.STANDARD:
            opponents = {
                'alpha': {'model_file': 'standard/curious-dust-35M.pt'},
            }
            randomize = True
            hardness = 4
        elif objective == envs.Objective.SMOL_STANDARD:
            opponents = {
                'alpha': {'model_file': 'standard/curious-dust-35M.pt'},
            }
            randomize = True
            hardness = 1
        elif objective == envs.Objective.MICRO_PRACTICE:
            opponents = {
                'beta': {'model_file': 'mp/ethereal-bee-40M.pt'},
            }
        else:
            raise Exception(f'No eval opponents configured for {objective}')

    policy.eval()
    env = envs.CodeCraftVecEnv(num_envs,
                               num_envs // 2 if len(opponents) > 0 else 0,
                               objective,
                               action_delay=0,
                               stagger=False,
                               fair=not symmetric,
                               use_action_masks=True,
                               obs_config=policy.obs_config,
                               randomize=randomize,
                               hardness=hardness,
                               symmetric=symmetric,
                               strong_scripted_opponent=True,
                               rule_rng_amount=random_rules,
                               rule_rng_fraction=1.0 if random_rules > 0 else 0.0)

    scores = []
    eliminations = []
    scores_by_opp = defaultdict(list)
    lengths = []
    evens = list([2 * i for i in range(num_envs // 2)])
    odds = list([2 * i + 1 for i in range(num_envs // 2)])
    policy_envs = evens

    partitions = [(policy_envs, policy.obs_config)]
    i = 0
    for name, opp in opponents.items():
        opp_policy, _, _ = load_policy(opp['model_file'], device)
        opp_policy.eval()
        opp['policy'] = opp_policy
        opp['envs'] = odds[i * len(odds) // len(opponents):(i+1) * len(odds) // len(opponents)]
        opp['obs_config'] = opp_policy.obs_config
        opp['i'] = i
        i += 1
        partitions.append((opp['envs'], opp_policy.obs_config))

    initial_obs = env.reset(partitions)

    obs, action_masks, privileged_obs = initial_obs[0]
    obs_opps, action_masks_opps, privileged_obs_opps = ([], [], [])
    for o, a, p in initial_obs[1:]:
        obs_opps.append(o)
        action_masks_opps.append(a)
        privileged_obs_opps.append(p)

    for step in range(eval_steps):
        obs_tensor = torch.tensor(obs).to(device)
        privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
        action_masks_tensor = torch.tensor(action_masks).to(device)
        actionsp, _, _, _, _ = policy.evaluate(obs_tensor, action_masks_tensor, privileged_obs_tensor)
        env.step_async(actionsp.cpu(), policy_envs)

        for _, opp in opponents.items():
            i = opp['i']
            obs_opp_tensor = torch.tensor(obs_opps[i]).to(device)
            privileged_obs_opp_tensor = torch.tensor(privileged_obs_opps[i]).to(device)
            action_masks_opp_tensor = torch.tensor(action_masks_opps[i]).to(device)
            actions_opp, _, _, _, _ = opp['policy'].evaluate(obs_opp_tensor,
                                                             action_masks_opp_tensor,
                                                             privileged_obs_opp_tensor)
            env.step_async(actions_opp.cpu(), opp['envs'])

        obs, _, _, infos, action_masks, privileged_obs = env.observe(policy_envs)
        for _, opp in opponents.items():
            i = opp['i']
            obs_opps[i], _, _, _, action_masks_opps[i], privileged_obs_opps[i] = \
                env.observe(opp['envs'], opp['obs_config'])

        for info in infos:
            index = info['episode']['index']
            score = info['episode']['score']
            length = info['episode']['l']
            elimination = info['episode']['elimination']
            scores.append(score)
            eliminations.append(elimination)
            lengths.append(length)
            for name, opp in opponents.items():
                if index + 1 in opp['envs']:
                    scores_by_opp[name].append(score)
                    break

        if (step + 1) % printerval == 0:
            print(f'Eval: {np.array(scores).mean()}')

    scores = np.array(scores)
    eliminations = np.array(eliminations)

    if curr_step is not None:
        wandb.log({
            'eval_mean_score': scores.mean(),
            'eval_max_score': scores.max(),
            'eval_min_score': scores.min(),
            'eval_games': len(scores),
            'eval_elimination_rate': eliminations.mean(),
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
            tiles=hps.obs_map_tiles,
            global_drones=hps.obs_enemies if hps.use_privileged else 0,
            relative_positions=False,
            feat_last_seen=hps.feat_last_seen,
            feat_is_visible=hps.feat_is_visible,
            feat_map_size=hps.feat_map_size,
            feat_abstime=hps.feat_abstime,
            v2=True,
        )
    if version == 'transformer_v2':
        kwargs['obs_config'].tiles = 0
        policy = TransformerPolicy2(**kwargs)
    elif version == 'transformer_v3':
        kwargs['obs_config'].tiles = 0
        policy = TransformerPolicy3(**kwargs)
    elif version == 'transformer_v4':
        kwargs['obs_config'].tiles = 0
        policy = TransformerPolicy4(**kwargs)
    elif version == 'transformer_v5':
        policy = TransformerPolicy5(**kwargs)
    elif version == 'transformer_v6':
        policy = TransformerPolicy6(**kwargs)
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
    elif args.hpset == 'arena_medium_large_ms':
        hps = HyperParams.arena_medium_large_ms()
    elif args.hpset == 'arena':
        hps = HyperParams.arena()
    elif args.hpset == 'standard':
        hps = HyperParams.standard()
    elif args.hpset == 'micro_practice':
        hps = HyperParams.micro_practice()
    elif args.hpset == 'scout':
        hps = HyperParams.scout()
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


if __name__ == "__main__":
    main()

