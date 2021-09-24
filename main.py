import logging
import subprocess
import time
import os
from collections import defaultdict
import dataclasses
from pathlib import Path
from typing import List, Optional, Any
from dataclasses import dataclass

import torch.distributed as dist
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from torch.optim.optimizer import Optimizer
from torch_ema import ExponentialMovingAverage

import wandb

import hyperstate
from hyperstate import (
    Config,
    ObsConfig as HSObsConfig,
    Blob,
    OptimizerConfig,
    HyperState,
    TaskConfig,
)

from adr import ADR, ADRState, normalize, spec_key
from gym_codecraft import envs
from gym_codecraft.envs.codecraft_vec_env import ObsConfig, Rules
from hyper_params import HyperParams, parse_schedule
from policy_t2 import TransformerPolicy2, InputNorm
from policy_t3 import TransformerPolicy3, InputNorm
from policy_t4 import TransformerPolicy4, InputNorm
from policy_t5 import TransformerPolicy5, InputNorm
from policy_t6 import TransformerPolicy6, InputNorm
from policy_t7 import TransformerPolicy7, InputNorm
from policy_t8 import TransformerPolicy8, InputNorm
from policy_t8_hs import TransformerPolicy8HS, InputNorm

logger = logging.getLogger(__name__)

LOG_ROOT_DIR = "/home/clemens/Dropbox/artifacts/DeepCodeCraft"

if "EVAL_MODELS_PATH" in os.environ:
    EVAL_MODELS_PATH = os.environ["EVAL_MODELS_PATH"]
elif "XPRUN_ID" in os.environ:
    EVAL_MODELS_PATH = "/mnt/xprun/common/DeepCodeCraft/golden-models"
else:
    EVAL_MODELS_PATH = "/home/clemens/Dropbox/artifacts/DeepCodeCraft/golden-models"


@dataclass
class State:
    step: int
    iteration: int
    epoch: int
    policy: Blob[Any]
    optimizer: Blob[Any]
    ema: List[Blob[Any]]
    adr: ADRState


def run_codecraft():
    nenv = 32
    env = envs.CodeCraftVecEnv(
        nenv, envs.Objective.DISTANCE_TO_ORIGIN, 1, action_delay=0
    )

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

        env.step_async([4] * nenv)
        env.observe()
        frames += nenv


def create_optimizer(
    policy: TransformerPolicy8HS, config: OptimizerConfig
) -> Optimizer:
    if config.optimizer_type == "SGD":
        optimizer_fn = optim.SGD
        optimizer_kwargs = dict(
            lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "RMSProp":
        optimizer_fn = optim.RMSprop
        optimizer_kwargs = dict(
            lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "Adam":
        optimizer_fn = optim.Adam
        optimizer_kwargs = dict(
            lr=config.lr, weight_decay=config.weight_decay, eps=1e-5
        )
    else:
        raise Exception(f"Invalid optimizer name `{config.optimizer_type}`")
    return optimizer_fn(policy.parameters(), **optimizer_kwargs)


def initial_state(config: Config) -> State:
    policy = TransformerPolicy8HS(
        config.policy,
        config.obs,
        config.task.objective.naction() + config.obs.extra_actions(),
    )
    optimizer = create_optimizer(policy, config.optimizer)
    adr = ADRState(
        hardness=config.adr.initial_hardness,
        ruleset=Rules(
            mothership_damage_multiplier=config.task.mothership_damage_scale,
            cost_modifiers={build: 1.0 for build in config.task.objective.builds()},
        ),
    )
    policy_emas = [
        ExponentialMovingAverage(policy.parameters(), decay=float(decay))
        for decay in config.optimizer.weights_ema
    ]

    return State(
        step=0,
        iteration=0,
        epoch=0,
        policy=Blob(policy.state_dict()),
        optimizer=Blob(optimizer.state_dict()),
        ema=policy_emas,
        adr=adr,
    )


class Trainer:
    def __init__(self, state_manager: HyperState[Config, State]):
        config = state_manager.config
        state = state_manager.state

        self.hyperstate = state_manager
        self.config = config
        self.state = state

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            print("Running on CPU")
            self.device = "cpu"

        self.policy = TransformerPolicy8HS(
            config.policy,
            config.obs,
            config.task.objective.naction() + config.obs.extra_actions(),
        )
        self.policy.load_state_dict(state.policy.get())
        self.policy.to(self.device)
        self.optimizer = create_optimizer(self.policy, config.optimizer)
        self.optimizer.load_state_dict(state.optimizer.get())
        self.ema = state.ema
        self.adr = ADR(config.adr, state.adr)

    def train(self, out_dir: str) -> None:
        config = self.config
        state = self.state
        device = self.device

        # TODO: hyperstate
        next_model_save = config.eval.model_save_frequency

        obs_config = obs_config_from(config)

        # self.hyperstate.checkpoint("test")
        # return

        # TODO: hyperstate
        # if hps.resume_from != "":
        #    policy, optimizer, resume_steps, adr, lr_scheduler = load_policy(
        #        hps.resume_from, device, optimizer_fn, optimizer_kwargs, hps, hps.verify
        #    )

        # TODO: xprun
        # if hps.parallelism > 1:
        #    sync_parameters(policy)

        # TODO: xprun
        # if hps.rank == 0:
        #    wandb.watch(policy)

        next_eval = state.step
        next_full_eval = 1
        eprewmean = 0
        eplenmean = 0
        eliminationmean = 0
        buildmean = defaultdict(lambda: 0)
        completed_episodes = 0
        env = None
        extra_checkpoint_steps = [
            step for step in config.eval.extra_checkpoint_steps if step > state.step
        ]
        rewmean = 0.0
        rewstd = 1.0
        while state.step < config.ppo.steps:
            # TODO: better solution
            self.config.task.build_variety_bonus = self.config.ppo.build_variety_bonus
            self.config.task.cost_variance = self.config.adr.cost_variance

            # TODO: step
            for g in self.optimizer.param_groups:
                g["lr"] = config.optimizer.lr
            assert config.rosteps % config.optimizer.batch_size == 0

            if env is None:
                env = envs.CodeCraftVecEnv(
                    config.ppo.num_envs,
                    config.ppo.num_self_play,
                    config.task.objective,
                    config.task.action_delay,
                    config=config.task,
                    randomize=config.task.randomize,
                    use_action_masks=config.task.use_action_masks,
                    obs_config=obs_config,
                    symmetric=config.task.symmetric_map,
                    hardness=config.task.task_hardness,
                    mix_mp=config.task.mix_mp,
                    win_bonus=config.ppo.win_bonus,
                    attac=config.ppo.attac,
                    protec=config.ppo.protec,
                    max_army_size_score=config.ppo.max_army_size_score,
                    max_enemy_army_size_score=config.ppo.max_enemy_army_size_score,
                    rule_rng_fraction=config.task.rule_rng_fraction,
                    rule_rng_amount=config.task.rule_rng_amount,
                    rule_cost_rng=config.task.rule_cost_rng,
                    scripted_opponents=[
                        ("destroyer", config.ppo.num_vs_destroyer),
                        ("replicator", config.ppo.num_vs_replicator),
                        ("aggressive_replicator", config.ppo.num_vs_aggro_replicator),
                    ],
                    max_game_length=None
                    if config.task.max_game_length == 0
                    else config.task.max_game_length,
                    # TODO: xprun
                    stagger_offset=0,  # hps.rank / hps.parallelism,
                    loss_penalty=config.ppo.loss_penalty,
                    partial_score=config.ppo.partial_score,
                )
                env.rng_ruleset = self.adr.state.ruleset
                env.hardness = self.adr.state.hardness
                obs, action_masks, privileged_obs = env.reset()

            if state.step >= next_eval:
                if config.eval.eval_envs > 0:
                    next_full_eval -= 1
                    if next_full_eval == 0:
                        emas = [None] + self.ema
                        next_full_eval = config.eval.full_eval_frequency
                    else:
                        emas = [None]
                    for policy_ema in emas:
                        eval(
                            policy=self.policy,
                            # TODO: xprun
                            num_envs=config.eval.eval_envs,  # // hps.parallelism,
                            device=device,
                            objective=config.task.objective,
                            eval_steps=config.eval.eval_timesteps,
                            curr_step=state.step,
                            symmetric=config.eval.eval_symmetric,
                            # TODO: xprun
                            rank=0,  # hps.rank,
                            parallelism=1,  # hps.parallelism,
                            policy_ema=policy_ema,
                        )
                next_eval += config.eval.eval_frequency
                next_model_save -= 1
                # TODO: xprun
                if next_model_save == 0:  # and hps.rank == 0:
                    # TODO: hyperstate
                    # hyperstate.checkpoint(state)
                    # next_model_save = config.eval.model_save_frequency
                    # save_policy(
                    #    state.policy,
                    #    out_dir,
                    #    total_steps,
                    #    optimizer,
                    #    adr,
                    #    lr_scheduler,
                    #    policy_emas,
                    # )
                    pass

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

            self.policy.eval()
            buildtotal = defaultdict(lambda: 0)
            eliminations = []
            if config.task.adr:
                env.rng_ruleset = state.adr.ruleset
                env.hardness = state.adr.hardness
            # TODO: hyperstate schedule
            if config.task.symmetry_increase > 0:
                env.symmetric = min(state.step * config.task.symmetry_increase, 1.0)
            with torch.no_grad():
                cost_sum = 0.0
                cost_weight = 0.0
                # Rollout
                for step in range(config.ppo.seq_rosteps):
                    obs_tensor = torch.tensor(obs).to(device)
                    privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
                    action_masks_tensor = torch.tensor(action_masks).to(device)
                    actions, logprobs, entropy, values, probs = self.policy.evaluate(
                        obs_tensor, action_masks_tensor, privileged_obs_tensor
                    )
                    actions = actions.cpu().numpy()

                    entropies.extend(entropy.detach().cpu().numpy())

                    all_action_masks.extend(action_masks)
                    all_obs.extend(obs)
                    all_privileged_obs.extend(privileged_obs)
                    all_actions.extend(actions)
                    all_logprobs.extend(logprobs.detach().cpu().numpy())
                    all_values.extend(values)
                    all_probs.extend(probs)

                    obs, rews, dones, infos, action_masks, privileged_obs = env.step(
                        actions, action_masks=action_masks
                    )

                    rews -= config.ppo.liveness_penalty
                    all_rewards.extend(rews)
                    all_dones.extend(dones)

                    for info in infos:
                        ema = 0.95 * (1 - 1 / (completed_episodes + 1))

                        decided_by_elimination = info["episode"]["elimination"]
                        eliminations.append(decided_by_elimination)
                        eliminationmean = (
                            eliminationmean * ema + (1 - ema) * decided_by_elimination
                        )

                        eprewmean = eprewmean * ema + (1 - ema) * info["episode"]["r"]
                        eplenmean = eplenmean * ema + (1 - ema) * info["episode"]["l"]

                        builds = info["episode"]["builds"]
                        for build in set().union(builds.keys(), buildmean.keys()):
                            count = builds[build]
                            buildmean[build] = (
                                buildmean[build] * ema + (1 - ema) * count
                            )
                            buildtotal[build] += count
                            cost = info["episode"]["ruleset"].cost_modifiers[build]
                            cost_sum += cost * sum(build) * count
                            cost_weight += sum(build) * count
                        completed_episodes += 1
                average_cost_modifier = (
                    cost_sum / cost_weight if cost_weight > 0 else 1.0
                )
                elimination_rate = (
                    np.array(eliminations).mean() if len(eliminations) > 0 else None
                )
                self.adr.adjust(
                    buildtotal,
                    average_cost_modifier,
                    elimination_rate,
                    eplenmean,
                    state.step,
                )

                obs_tensor = torch.tensor(obs).to(device)
                action_masks_tensor = torch.tensor(action_masks).to(device)
                privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
                _, _, _, final_values, final_probs = self.policy.evaluate(
                    obs_tensor, action_masks_tensor, privileged_obs_tensor
                )

                all_rewards = np.array(all_rewards) * config.ppo.rewscale
                w = config.ppo.rewnorm_emaw * (1 - 1 / (state.step + 1))
                rewmean = all_rewards.mean() * (1 - w) + rewmean * w
                rewstd = all_rewards.std() * (1 - w) + rewstd * w
                if config.ppo.rewnorm:
                    all_rewards = all_rewards / rewstd - rewmean

                all_returns = np.zeros(len(all_rewards), dtype=np.float32)
                all_values = np.array(all_values)
                last_gae = np.zeros(config.ppo.num_envs)
                gamma = config.ppo.gamma
                for t in reversed(range(config.ppo.seq_rosteps)):
                    for i in range(config.ppo.num_envs):
                        ti = t * config.ppo.num_envs + i
                        tnext_i = (t + 1) * config.ppo.num_envs + i
                        nextnonterminal = 1.0 - all_dones[ti]
                        if t == config.ppo.seq_rosteps - 1:
                            next_value = final_values[i]
                        else:
                            next_value = all_values[tnext_i]
                        td_error = (
                            all_rewards[ti]
                            + gamma * next_value * nextnonterminal
                            - all_values[ti]
                        )
                        last_gae[i] = (
                            td_error
                            + gamma * config.ppo.lamb * last_gae[i] * nextnonterminal
                        )
                        all_returns[ti] = last_gae[i] + all_values[ti]

                advantages = all_returns - all_values
                if config.ppo.norm_advs:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
                explained_var = explained_variance(all_values, all_returns)

                all_actions = np.array(all_actions)
                all_logprobs = np.array(all_logprobs)
                all_obs = np.array(all_obs)
                all_privileged_obs = np.array(all_privileged_obs)
                all_action_masks = np.array(all_action_masks)[
                    :, : config.policy.agents, :
                ]
                all_probs = np.array(all_probs)

            for epoch in range(config.optimizer.epochs):
                if config.optimizer.shuffle:
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
                entropy_loss_sum = 0
                gradnorm = 0
                self.policy.train()
                torch.enable_grad()
                num_micro_batches = int(
                    config.ppo.seq_rosteps
                    * config.ppo.num_envs
                    / config.optimizer.batch_size
                )
                for micro_batch in range(num_micro_batches):
                    if (
                        micro_batch
                        * config.optimizer.micro_batch_size
                        % config.optimizer.batch_size
                        == 0
                    ):
                        self.optimizer.zero_grad()

                    start = config.optimizer.micro_batch_size * micro_batch
                    end = config.optimizer.micro_batch_size * (micro_batch + 1)

                    o = torch.tensor(all_obs[start:end]).to(device)
                    op = torch.tensor(all_privileged_obs[start:end]).to(device)
                    actions = torch.tensor(all_actions[start:end]).to(device)
                    probs = torch.tensor(all_logprobs[start:end]).to(device)
                    returns = torch.tensor(all_returns[start:end]).to(device)
                    advs = torch.tensor(advantages[start:end]).to(device)
                    vals = torch.tensor(all_values[start:end]).to(device)
                    amasks = torch.tensor(all_action_masks[start:end]).to(device)
                    actual_probs = torch.tensor(all_probs[start:end]).to(device)

                    (
                        policy_loss,
                        value_loss,
                        entropy_loss,
                        aproxkl,
                        clipfrac,
                    ) = self.policy.backprop(
                        config,
                        o,
                        actions,
                        probs,
                        returns,
                        config.optimizer.vf_coef,
                        advs,
                        vals,
                        amasks,
                        actual_probs,
                        op,
                        config.ppo.split_reward,
                    )

                    policy_loss_sum += policy_loss
                    entropy_loss_sum += entropy_loss
                    value_loss_sum += value_loss
                    aproxkl_sum += aproxkl
                    clipfrac_sum += clipfrac
                    gradnorm += torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), config.optimizer.max_grad_norm
                    )

                    if (
                        (micro_batch + 1)
                        * config.optimizer.micro_batch_size
                        % config.optimizer.batch_size
                        == 0
                    ):
                        # TODO: xprun
                        # if hps.parallelism > 1:
                        #    gradient_allreduce(policy)
                        self.optimizer.step()
                        for ema in self.ema:
                            ema.update(self.policy.parameters())
            torch.cuda.empty_cache()

            state.epoch += 1
            # TODO: xprun
            state.step += config.rosteps  # * hps.parallelism
            state.iteration += 1
            # TODO: xprun
            throughput = int(
                config.rosteps / (time.time() - episode_start)
            )  # * hps.parallelism

            all_agent_masks = all_action_masks.sum(2) > 1
            # TODO: xprun if rank == 0
            if config.optimizer.epochs > 0:
                if config.wandb:
                    # TODO: hyperstate metrics
                    metrics = {
                        "policy_loss": policy_loss_sum / num_micro_batches,
                        "value_loss": value_loss_sum / num_micro_batches,
                        "entropy_loss": entropy_loss_sum / num_micro_batches,
                        "clipfrac": clipfrac_sum / num_micro_batches,
                        "aproxkl": aproxkl_sum / num_micro_batches,
                        "throughput": throughput,
                        "eprewmean": eprewmean,
                        "eplenmean": eplenmean,
                        "target_eplenmean": self.adr.target_eplenmean(),
                        "eliminationmean": eliminationmean,
                        "entropy": sum(entropies) / len(entropies) / np.log(2),
                        "explained variance": explained_var,
                        "gradnorm": gradnorm
                        * config.optimizer.batch_size
                        / config.rosteps,
                        "advantages": wandb.Histogram(advantages),
                        "values": wandb.Histogram(all_values),
                        "meanval": all_values.mean(),
                        "returns": wandb.Histogram(all_returns),
                        "meanret": all_returns.mean(),
                        "actions": wandb.Histogram(
                            np.array(all_actions[all_agent_masks])
                        ),
                        "active_agents": all_agent_masks.sum() / all_agent_masks.size,
                        "observations": wandb.Histogram(np.array(all_obs)),
                        "obs_max": all_obs.max(),
                        "obs_min": all_obs.min(),
                        "rewards": wandb.Histogram(np.array(all_rewards)),
                        "masked_actions": 1 - all_action_masks.mean(),
                        "rewmean": rewmean,
                        "rewstd": rewstd,
                        "average_cost_modifier": average_cost_modifier,
                        "hardness": self.adr.state.hardness,
                        "iteration": state.iteration,
                    }
                    metrics.update(hyperstate.asdict(config))
                    for action, count in buildmean.items():
                        metrics[f"build_{spec_key(action)}"] = count
                    for action, fraction in normalize(buildmean).items():
                        metrics[f"frac_{spec_key(action)}"] = fraction

                    metrics.update(self.adr.metrics())
                    total_norm = 0.0
                    count = 0
                    for name, param in self.policy.named_parameters():
                        norm = param.data.norm()
                        metrics[f"weight_norm[{name}]"] = norm
                        count += 1
                        total_norm += norm
                    metrics["mean_weight_norm"] = total_norm / count
                    # TODO: steps
                    wandb.log(metrics, step=state.step)

                # TODO: some way to avoid doing this on every iteration?
                state.policy.set(self.policy.state_dict())
                state.optimizer.set(self.optimizer.state_dict())
                self.hyperstate.step()

            print(f"{throughput} samples/s", flush=True)

        env.close()

        if config.eval.eval_envs > 0:
            for policy_ema in [None] + self.ema:
                eval(
                    policy=self.policy,
                    # TODO: xprun
                    num_envs=config.eval.eval_envs,  # // hps.parallelism,
                    device=device,
                    objective=config.task.objective,
                    eval_steps=5 * config.eval.eval_timesteps,
                    curr_step=state.step,
                    symmetric=config.eval.eval_symmetric,
                    printerval=config.eval.eval_timesteps,
                    # TODO: xprun
                    rank=0,  # hps.rank,
                    parallelism=1,  # parallelism,
                    policy_ema=policy_ema,
                )
        # TODO: xprun
        # if hps.rank == 0:
        # TODO: hyperstate
        # save_policy(
        #    policy, out_dir, total_steps, optimizer, adr, lr_scheduler, policy_emas
        # )


def eval(
    policy,
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
    random_rules=0.0,
    rank=0,
    parallelism=1,
    policy_ema=None,
    create_game_delay=0.0,
):
    start_time = time.time()

    if printerval is None:
        printerval = eval_steps

    scripted_opponents = []
    if opponents is None:
        if objective == envs.Objective.ARENA_TINY:
            opponents = {
                "easy": {"model_file": "arena_tiny/t2_random.pt"},
            }
        elif objective == envs.Objective.ARENA_TINY_2V2:
            opponents = {
                "easy": {"model_file": "arena_tiny_2v2/fine-sky-10M.pt"},
            }
        elif objective == envs.Objective.ARENA_MEDIUM:
            opponents = {
                "10m": {"model_file": "arena_medium/arena_medium-5f06842-0-10m"}
            }
        elif objective == envs.Objective.ARENA_MEDIUM_LARGE_MS:
            opponents = {
                "easy": {"model_file": "arena_medium_large_ms/honest-violet-50M.pt"},
            }
        elif objective == envs.Objective.ARENA:
            opponents = {
                "beta": {"model_file": "arena/glad-breeze-25M.pt"},
            }
        elif objective == envs.Objective.STANDARD:
            opponents = {
                "noble-sky-145": {"model_file": "standard/noble-sky-145M.pt"},
                "radiant-sun-35": {"model_file": "standard/radiant-sun-35M.pt"},
            }
            scripted_opponents = ["destroyer", "replicator"]
            hardness = 5
        elif objective == envs.Objective.ENHANCED:
            opponents = {
                "logical-dust-250": {"model_file": "enhanced/logical-dust-250m.pt"},
                "youthful-vortex-500": {
                    "model_file": "enhanced/youthful-vortex-500m.pt"
                },
            }
            hardness = 150
        elif objective == envs.Objective.SMOL_STANDARD:
            opponents = {
                "alpha": {"model_file": "standard/curious-dust-35M.pt"},
            }
            randomize = True
            hardness = 1
        elif objective == envs.Objective.MICRO_PRACTICE:
            opponents = {
                "beta": {"model_file": "mp/ethereal-bee-40M.pt"},
            }
        else:
            raise Exception(f"No eval opponents configured for {objective}")

    if policy_ema is not None:
        policy_ema.store(policy.parameters())
        policy_ema.copy_to(policy.parameters())
        postfix = "-ema" + str(policy_ema.decay).replace(".", "")
    else:
        postfix = ""

    policy.eval()

    n_opponent = len(opponents) + len(scripted_opponents)
    n_scripted = len(scripted_opponents)
    if n_opponent == 0:
        self_play_envs = 0
    else:
        assert num_envs * n_scripted % n_opponent == 0
        non_self_play_envs = num_envs * n_scripted // n_opponent
        assert (num_envs - non_self_play_envs) % 2 == 0
        self_play_envs = (num_envs - non_self_play_envs) // 2

    env = envs.CodeCraftVecEnv(
        num_envs,
        self_play_envs,
        objective,
        action_delay=0,
        config=TaskConfig(
            objective=objective,
            mothership_damage_scale=0.0,
            rule_rng_amount=random_rules,
            rule_rng_fraction=1.0 if random_rules > 0 else 0.0,
            symmetric_map=1.0 if symmetric else 0.0,
            task_hardness=hardness,
            randomize=randomize,
        ),
        stagger=False,
        fair=not symmetric,
        use_action_masks=True,
        obs_config=policy.obs_config,
        randomize=randomize,
        hardness=hardness,
        symmetric=1.0 if symmetric else 0.0,
        scripted_opponents=[(o, num_envs // n_opponent) for o in scripted_opponents],
        rule_rng_amount=random_rules,
        rule_rng_fraction=1.0 if random_rules > 0 else 0.0,
        create_game_delay=create_game_delay,
    )

    scores = []
    eliminations = []
    scores_by_opp = defaultdict(list)
    eliminations_by_opp = defaultdict(list)
    lengths = []
    evens = list([2 * i for i in range(self_play_envs)])
    odds = list([2 * i + 1 for i in range(self_play_envs)])
    policy_envs = evens + list(range(2 * self_play_envs, num_envs))

    partitions = [(policy_envs, policy.obs_config)]
    i = 0
    for name, opp in opponents.items():
        if opp["model_file"].endswith(".pt"):
            opp_policy, _, _, _ = load_policy(opp["model_file"], device)
        else:
            opp_policy = load_hs_policy(
                Path(EVAL_MODELS_PATH) / opp["model_file"], device
            )
        opp_policy.eval()
        opp["policy"] = opp_policy
        opp["envs"] = odds[
            i * len(odds) // len(opponents) : (i + 1) * len(odds) // len(opponents)
        ]
        opp["obs_config"] = opp_policy.obs_config
        opp["i"] = i
        i += 1
        partitions.append((opp["envs"], opp_policy.obs_config))

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
        actionsp, _, _, _, _ = policy.evaluate(
            obs_tensor, action_masks_tensor, privileged_obs_tensor
        )
        env.step_async(actionsp.cpu(), policy_envs)

        for _, opp in opponents.items():
            i = opp["i"]
            obs_opp_tensor = torch.tensor(obs_opps[i]).to(device)
            privileged_obs_opp_tensor = torch.tensor(privileged_obs_opps[i]).to(device)
            action_masks_opp_tensor = torch.tensor(action_masks_opps[i]).to(device)
            actions_opp, _, _, _, _ = opp["policy"].evaluate(
                obs_opp_tensor, action_masks_opp_tensor, privileged_obs_opp_tensor
            )
            env.step_async(actions_opp.cpu(), opp["envs"])

        obs, _, _, infos, action_masks, privileged_obs = env.observe(policy_envs)
        for _, opp in opponents.items():
            i = opp["i"]
            (
                obs_opps[i],
                _,
                _,
                _,
                action_masks_opps[i],
                privileged_obs_opps[i],
            ) = env.observe(opp["envs"], opp["obs_config"])

        for info in infos:
            index = info["episode"]["index"]
            score = info["episode"]["score"]
            length = info["episode"]["l"]
            elimination_win = 1 if info["episode"]["outcome"] == 1 else 0
            scores.append(score)
            eliminations.append(elimination_win)
            lengths.append(length)
            if index >= 2 * self_play_envs:
                name = info["episode"]["opponent"]
                scores_by_opp[name].append(score)
                eliminations_by_opp[name].append(elimination_win)
            else:
                for name, opp in opponents.items():
                    if index + 1 in opp["envs"]:
                        scores_by_opp[name].append(score)
                        eliminations_by_opp[name].append(elimination_win)
                        break

        if (step + 1) % printerval == 0:
            print(
                f"Eval{postfix}: {np.array(scores).mean():6.3f}  {sum(eliminations)}/{len(scores)}  (total)"
            )
            for name, _scores in sorted(scores_by_opp.items()):
                print(
                    f"      {np.array(_scores).mean():6.3f}  {sum(eliminations_by_opp[name])}/{len(_scores)}  ({name})"
                )

    scores = torch.FloatTensor(scores)
    eliminations = torch.FloatTensor(eliminations)

    if curr_step is not None:
        if parallelism > 1:
            scores = allcat(scores, rank, parallelism)
            eliminations = allcat(eliminations, rank, parallelism)
        if rank == 0:
            wandb.log(
                {
                    f"eval_mean_score{postfix}": scores.mean().item(),
                    f"eval_max_score{postfix}": scores.max().item(),
                    f"eval_min_score{postfix}": scores.min().item(),
                    f"eval_games{postfix}": len(scores),
                    f"eval_elimination_rate{postfix}": eliminations.mean().item(),
                    f"evalu_duration_secs{postfix}": time.time() - start_time,
                },
                step=curr_step,
            )
        for opp_name, scores in sorted(scores_by_opp.items()):
            scores = torch.Tensor(scores)
            eliminations = torch.Tensor(eliminations_by_opp[opp_name])
            if parallelism > 1:
                scores = allcat(scores, rank, parallelism)
                eliminations = allcat(eliminations, rank, parallelism)
            if rank == 0:
                wandb.log(
                    {
                        f"eval_mean_score_vs_{opp_name}{postfix}": scores.mean().item(),
                        f"eval_games_vs_{opp_name}{postfix}": len(scores),
                        f"eval_elimination_rate_vs_{opp_name}{postfix}": eliminations.mean().item(),
                    },
                    step=curr_step,
                )

    if policy_ema is not None:
        policy_ema.restore(policy.parameters())

    env.close()


def obs_config_from(config: Config) -> ObsConfig:
    oc = config.obs
    return ObsConfig(
        allies=oc.allies,
        drones=oc.allies + oc.obs_enemies,
        minerals=oc.obs_minerals,
        tiles=oc.obs_map_tiles,
        num_builds=len(config.task.objective.builds()),
        global_drones=oc.global_drones,
        relative_positions=False,
        feat_last_seen=oc.feat_last_seen,
        feat_is_visible=oc.feat_is_visible,
        feat_map_size=oc.feat_map_size,
        feat_abstime=oc.feat_abstime,
        v2=True,
        feat_rule_msdm=config.task.rule_rng_fraction > 0 or config.task.adr,
        feat_rule_costs=config.task.rule_cost_rng > 0 or config.task.adr,
        feat_mineral_claims=oc.feat_mineral_claims,
        harvest_action=oc.harvest_action,
        lock_build_action=oc.lock_build_action,
        feat_dist_to_wall=oc.feat_dist_to_wall,
        unit_count=oc.feat_unit_count,
        construction_progress=oc.feat_construction_progress,
    )


def save_policy(
    policy, out_dir, total_steps, optimizer=None, adr=None, policy_emas=None,
):
    if policy_emas is None:
        policy_emas = []
    for policy_ema in [None] + policy_emas:
        postfix = ""
        if policy_ema is not None:
            policy_ema.store(policy.parameters())
            policy_ema.copy_to(policy.parameters())
            postfix = "-ema" + str(policy_ema.decay).replace(".", "")
        model_path = os.path.join(out_dir, f"model-{total_steps}{postfix}.pt")
        print(f"Saving policy to {model_path}")
        model = {
            "model_state_dict": policy.state_dict(),
            "model_kwargs": policy.kwargs,
            "total_steps": total_steps,
            "policy_version": policy.version,
        }
        if optimizer:
            model["optimizer_state_dict"] = optimizer.state_dict()
        if adr:
            model["adr_state_dict"] = {
                "hardness": adr.hardness,
                "rules": dataclasses.asdict(adr.ruleset),
                "max_hardness": adr.max_hardness,
                "linear_hardness": adr.linear_hardness,
                "hardness_offset": adr.hardness_offset,
                "step": adr.step,
            }
        torch.save(model, model_path)
        if policy_ema is not None:
            policy_ema.restore(policy.parameters())


def load_policy(
    name, device, optimizer_fn=None, optimizer_kwargs=None, hps=None, rawpath=False
):
    if rawpath:
        checkpoint = torch.load(name, map_location=device)
    else:
        checkpoint = torch.load(
            os.path.join(EVAL_MODELS_PATH, name), map_location=device
        )
    version = checkpoint.get("policy_version")
    kwargs = checkpoint["model_kwargs"]
    if hps:
        kwargs["obs_config"] = obs_config_from(hps)
    if version == "transformer_v2":
        kwargs["obs_config"].tiles = 0
        policy = TransformerPolicy2(**kwargs)
    elif version == "transformer_v3":
        kwargs["obs_config"].tiles = 0
        policy = TransformerPolicy3(**kwargs)
    elif version == "transformer_v4":
        kwargs["obs_config"].tiles = 0
        policy = TransformerPolicy4(**kwargs)
    elif version == "transformer_v5":
        policy = TransformerPolicy5(**kwargs)
    elif version == "transformer_v6":
        policy = TransformerPolicy6(**kwargs)
    elif version == "transformer_v7":
        policy = TransformerPolicy7(**kwargs)
    elif version == "transformer_v8":
        policy = TransformerPolicy8(**kwargs)
    else:
        raise Exception(f"Unknown policy version {version}")

    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)

    optimizer = None
    if optimizer_fn:
        optimizer = optimizer_fn(policy.parameters(), **optimizer_kwargs)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        else:
            logger.warning(
                f"Failed to restore optimizer state: No `optimizer_state_dict` in saved model."
            )

    adr = None
    if hps is not None:
        hardness = 0.0
        ruleset = None
        linear_hardness = False
        max_hardness = 200
        hardness_offset = 0.0
        step = 0
        if "adr_state_dict" in checkpoint:
            adr_state = checkpoint["adr_state_dict"]
            hardness = adr_state["hardness"]
            if "rules" in adr_state:
                ruleset = Rules(**adr_state["rules"])
            if "linear_hardness" in adr_state:
                linear_hardness = adr_state["linear_hardness"]
            if "max_hardness" in adr_state:
                max_hardness = adr_state["max_hardness"]
            if "hardness_offset" in adr_state:
                hardness_offset = adr_state["hardness_offset"]
            if "step" in adr_state:
                step = adr_state["step"]
        adr = ADR(
            hstepsize=hps.adr_hstepsize,
            initial_hardness=hardness,
            ruleset=ruleset,
            linear_hardness=linear_hardness,
            max_hardness=max_hardness,
            hardness_offset=hardness_offset,
            step=step,
        )

    return policy, optimizer, checkpoint.get("total_steps", 0), adr


def load_hs_policy(path, device):
    hs = HyperState.load(Config, State, lambda _: None, path)
    config = hs.config
    state = hs.state
    policy = TransformerPolicy8HS(
        config.policy,
        config.obs,
        config.task.objective.naction() + config.obs.extra_actions(),
    )
    policy.load_state_dict(state.policy.get())
    policy.to(device)
    return policy


def explained_variance(ypred, y):
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
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def load_samples_from_disk():
    return (
        np.load("verify/obs.npy"),
        np.load("verify/privileged_obs.npy"),
        np.load("verify/returns.npy"),
        np.load("verify/actions.npy"),
        np.load("verify/logprobs.npy"),
        np.load("verify/values.npy"),
        np.load("verify/advantages.npy"),
        np.load("verify/action_masks.npy"),
        np.load("verify/probs.npy"),
    )


def write_samples_to_disk(
    all_obs,
    all_privileged_obs,
    all_returns,
    all_actions,
    all_logprobs,
    all_values,
    advantages,
    all_action_masks,
    all_probs,
):
    Path(f"verify").mkdir(parents=True, exist_ok=True)
    np.save("verify/obs", all_obs)
    np.save("verify/privileged_obs", all_privileged_obs)
    np.save("verify/returns", all_returns)
    np.save("verify/actions", all_actions)
    np.save("verify/logprobs", all_logprobs)
    np.save("verify/values", all_values)
    np.save("verify/advantages", advantages)
    np.save("verify/action_masks", all_action_masks)
    np.save("verify/probs", all_probs)


def write_gradients_to_disk(policy, epoch, batch):
    Path(f"verify/grad/{epoch}/{batch}").mkdir(parents=True, exist_ok=True)
    for name, param in policy.named_parameters():
        if param.grad is not None:
            np.save(f"verify/grad/{epoch}/{batch}/{name}", param.grad.cpu().numpy())
    print(f"Stored gradients for epoch {epoch}, batch {batch}")


def verify_gradients(policy, epoch, batch) -> bool:
    print(f"Verifying gradients for epoch {epoch} batch {batch}")
    errors = False
    expected_grads = {}
    for file in os.listdir(f"verify/grad/{epoch}/{batch}"):
        expected_grads[file[:-4]] = np.load(f"verify/grad/{epoch}/{batch}/{file}")
    remaining = set(expected_grads.keys())
    for name, param in policy.named_parameters():
        if name not in expected_grads:
            print(f"WARNING: no expected gradient found for {name}")
            continue
        if param.grad is None:
            print(f"WARNING: {name} has no gradient")
            continue
        remaining.remove(name)
        error = np.linalg.norm((expected_grads[name] - param.grad.cpu().numpy()))
        maxnorm = max(
            np.linalg.norm(expected_grads[name]),
            np.linalg.norm(param.grad.cpu().numpy()),
        )
        if maxnorm == 0:
            relerror = 0.0
        else:
            relerror = error / maxnorm

        if relerror >= 1e-2:
            print(f"ERROR: mismatch for {name}, abs {error:.2g}, rel {relerror:.2g}")
            errors = True
        else:
            print(f"OK: {name}, {relerror:.2g} < 1e-4")
    for name in remaining:
        print(f"WARNING: no gradient for {name}")
    return errors


def sync_parameters(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def gradient_allreduce(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def allcat(tensor: torch.Tensor, rank: int, parallelism: int) -> Optional[torch.Tensor]:
    if rank == 0:
        alltensors = [tensor]
        for sender in range(1, parallelism):
            size = torch.zeros(1)
            dist.recv(size, src=sender)
            tensor = torch.zeros(int(size.item()))
            dist.recv(tensor, src=sender)
            alltensors.append(tensor)
        return torch.cat(alltensors, dim=0)
    else:
        dist.send(torch.FloatTensor([len(tensor)]), dst=0)
        dist.send(tensor, dst=0)
        return None


def profile_fp(hps: HyperParams) -> None:
    import torchprof  # type: ignore

    start_time = time.time()
    device = torch.device("cuda:0")
    obs_config = obs_config_from(hps)
    env = envs.CodeCraftVecEnv(
        hps.num_envs,
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
        rule_cost_rng=hps.rule_cost_rng,
        scripted_opponents=[
            ("destroyer", hps.num_vs_destroyer),
            ("replicator", hps.num_vs_replicator),
            ("aggressive_replicator", hps.num_vs_aggro_replicator),
        ],
        max_game_length=None if hps.max_game_length == 0 else hps.max_game_length,
        stagger_offset=hps.rank / hps.parallelism,
        mothership_damage_scale=hps.mothership_damage_scale,
    )
    policy = TransformerPolicy8(hps, obs_config).to(device)
    obs, action_masks, privileged_obs = env.reset()

    with torchprof.Profile(policy, use_cuda=True) as prof:
        for _ in range(0, hps.seq_rosteps):
            obs_tensor = torch.tensor(obs).to(device)
            privileged_obs_tensor = torch.tensor(privileged_obs).to(device)
            action_masks_tensor = torch.tensor(action_masks).to(device)
            actions, logprobs, entropy, values, probs = policy.evaluate(
                obs_tensor, action_masks_tensor, privileged_obs_tensor
            )
            actions = actions.cpu().numpy()
            obs, _, _, _, action_masks, privileged_obs = env.step(
                actions, action_masks=action_masks
            )
    elapsed = time.time() - start_time
    print(
        f"Collected {hps.seq_rosteps * hps.num_envs} frames in {int(elapsed)}s ({int(hps.seq_rosteps * hps.num_envs / elapsed)}fps)"
    )
    print(prof.display(show_events=False))


def init_process(backend="gloo"):
    """ Initialize the distributed environment. """
    rank = int(os.environ["XPRUN_RANK"])
    world_size = int(os.environ["XPRUN_REPLICAS"])
    replica_name = os.environ["XPRUN_REPLICA_NAME"]
    xprun_id = os.environ["XPRUN_ID"]
    os.environ["MASTER_ADDR"] = f"xprun.{xprun_id}.{replica_name}-0"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def main():
    logging.basicConfig(level=logging.INFO)
    # torch.set_printoptions(threshold=25000)

    hps = HyperParams()
    args_parser = hps.args_parser()
    args_parser.add_argument("--out-dir")
    args_parser.add_argument("--config")
    args_parser.add_argument("--device", default=0)
    args_parser.add_argument("--descriptor", default="none")
    args_parser.add_argument("--profile", action="store_true")
    args = args_parser.parse_args()

    if not args.out_dir:
        if "XPRUN_ID" in os.environ:
            out_dir = os.path.join(
                "/mnt/xprun",
                os.environ["XPRUN_PROJECT"],
                os.environ["XPRUN_SANITIZED_NAME"] + "-" + os.environ["XPRUN_ID"],
            )
        else:
            t = time.strftime("%Y-%m-%d~%H:%M:%S")
            commit = subprocess.check_output(
                ["git", "describe", "--tags", "--always", "--dirty"]
            ).decode("UTF-8")[:-1]
            out_dir = os.path.join(LOG_ROOT_DIR, f"{t}-{commit}")
    else:
        out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_dir = Path(out_dir) / "checkpoints"
    checkpoint = hyperstate.find_latest_checkpoint(checkpoint_dir)
    if checkpoint is not None:
        args.config = checkpoint

    # TODO: xprun
    """
    if hps.parallelism > 1:
        if "XPRUN_ID" not in os.environ:
            raise Exception("Data parallel training only supported with xprun")
        else:
            init_process()
            hps.rank = int(os.environ["XPRUN_RANK"])
    """

    hs = HyperState.load(Config, State, initial_state, args.config, checkpoint_dir,)
    config = hs.config

    # TODO: xprun
    if config.wandb:  # hps.rank == 0:
        wandb_project = (
            "deep-codecraft-vs" if config.task.objective.vs() else "deep-codecraft"
        )
        if "XPRUN_NAME" in os.environ:
            wandb.init(
                project=wandb_project,
                name=os.environ["XPRUN_NAME"],
                id=os.environ["XPRUN_ID"],
            )
        else:
            wandb.init(project=wandb_project)
        cfg = hyperstate.asdict(config)
        cfg["commit"] = subprocess.check_output(
            ["git", "describe", "--tags", "--always", "--dirty"]
        ).decode("UTF-8")[:-1]
        cfg["descriptor"] = vars(args)["descriptor"]
        if "XPRUN_NAME" in os.environ:
            cfg["xp_name"] = os.environ["XPRUN_NAME"]
        wandb.config.update(cfg)

    trainer = Trainer(hs)
    trainer.train(out_dir=out_dir)


if __name__ == "__main__":
    main()

