import logging
import subprocess
import time
import os

from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.pg.ppo import PPO
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging.context import logger_context

import torch
import torch.optim as optim
import numpy as np

import wandb

from gym_codecraft import envs
from hyper_params import HyperParams
from policy import Policy, CodeCraftAgent

TEST_LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft_test'


def run_codecraft():
    nenv = 32
    env = envs.CodeCraftVecEnv(nenv, 3 * 60 * 60, envs.Objective.DISTANCE_TO_ORIGIN, 1)

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

    num_envs = hps.rosteps // hps.seq_rosteps
    env = envs.CodeCraftVecEnv(num_envs, hps.game_length, hps.objective, hps.action_delay)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    policy = Policy(hps.depth, hps.width, hps.conv, hps).to(device)
    if hps.fp16:
        policy = policy.half()
    if hps.optimizer == 'SGD':
        optimizer = optim.SGD
        optim_kwargs = dict(momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'RMSProp':
        optimizer = optim.RMSprop
        optim_kwargs = dict(momentum=hps.momentum, weight_decay=hps.weight_decay)
    elif hps.optimizer == 'Adam':
        optimizer = optim.Adam
        optim_kwargs = dict(weight_decay=hps.weight_decay, eps=1e-5)

    wandb.watch(policy)

    env = envs.SharedBatchingEnv(num_envs=num_envs,
                                 game_length=hps.game_length,
                                 objective=hps.objective,
                                 action_delay=0)

        #dict()
    sampler = GpuSampler(
        EnvCls=env.create_env,
        env_kwargs={},
        #eval_env_kwargs={},
        batch_T=hps.seq_rosteps,  # Four time-steps per sampler iteration.
        batch_B=num_envs,
        max_decorrelation_steps=0,
        #eval_n_envs=0,
        #eval_max_steps=int(10e3),
        #eval_max_trajectories=2,
    )
    minibatches = hps.rosteps // hps.bs
    algo = PPO(
        discount=hps.gamma,
        learning_rate=hps.lr,
        value_loss_coeff=hps.vf_coef,
        entropy_loss_coeff=0.001,
        OptimCls=optimizer,
        optim_kwargs=optim_kwargs,
        clip_grad_norm=hps.max_grad_norm,
        initial_optim_state_dict=None,
        gae_lambda=hps.lamb,
        minibatches=minibatches,
        epochs=hps.sample_reuse,
        ratio_clip=hps.cliprange,
        linear_lr_schedule=True,  # TODO
        normalize_advantage=hps.norm_advs,
    )
    # agent = AtariDqnAgent()
    agent = CodeCraftAgent(hps)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=hps.steps,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=0, workers_cpus=list(range(num_envs))),
    )
    config = dict(game="pong")
    name = "dqn_pong"
    log_dir = "example_1"
    with logger_context(log_dir, "run_id", name, config, snapshot_mode="last"):
        runner.train()


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
    wandb.init(project="deep-codecraft-rlpyt")

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
    config['commit'] = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")
    config['descriptor'] = vars(args)['descriptor']
    wandb.config.update(config)

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

