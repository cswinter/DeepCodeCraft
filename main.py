import argparse
import logging
import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np

from baselines.ppo2 import ppo2
from baselines import logger
import tensorflow as tf
import tensorflow.contrib.layers as layers

import codecraft
from gym_codecraft import envs

TEST_LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft_test'


def run_codecraft():
    games = []
    for i in range(5):
        game_id = codecraft.create_game()
        print("Starting game:", game_id)
        games.append(game_id)

    log_interval = 5
    frames = 0
    last_time = time.time()

    policy = Policy()

    while True:
        elapsed = time.time() - last_time
        if elapsed > log_interval:
            logging.info(f"{frames / elapsed}fps")
            frames = 0
            last_time = time.time()

        for i in range(len(games)):
            game_id = games[i]
            observation = codecraft.observe(game_id)
            if len(observation['winner']) > 0:
                print(f'Game {game_id} won by {observation["winner"][0]}')
                game_id = codecraft.create_game()
                print("Starting game:", game_id)
                games[i] = game_id
            else:
                obs_np = codecraft.observation_to_np(observation)
                action = codecraft.one_hot_to_action(policy.evaluate(obs_np))
                codecraft.act(game_id, action)
            frames += 1


def train(rollout_steps: int = 256, batch_size: int = 64 * 128) -> None:
    games = []
    num_envs = batch_size // rollout_steps
    env = envs.CodeCraftVecEnv(num_envs, 3 * 60 * 60)
    policy = Policy()

    obs = env.reset()
    while True:
        for step in range(rollout_steps):
            actions = policy.evaluate(obs)
            obs, rews, dones, infos = env.step(actions)



class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.dense1 = nn.Linear(47, 100)
        self.dense_final = nn.Linear(100, 8)

    def evaluate(self, observation):
        observation = torch.tensor(observation)
        probs = self.forward(observation)
        actions = []
        probs.detach_()
        for i in range(probs.size()[0]):
            actions.append(np.random.choice(8, 1, p=probs[i].numpy()))
        return actions

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.softmax(self.dense_final(x), dim=1)
        return x


class Hyperparam:
    def __init__(self, name, shortname, default):
        self.name = name
        self.shortname = shortname
        self.default = default

    def add_argument(self, parser):
        parser.add_argument(f"--{self.shortname}", f"--{self.name}", type=type(self.default))


HYPER_PARAMS = [
    Hyperparam("learning-rate", "lr", 1e-4),
    Hyperparam("num-layers", "nl", 4),
    Hyperparam("num-hidden", "nh", 1024),
    Hyperparam("total-timesteps", "steps", 1e7),
    Hyperparam("sequential-rollout-steps", "rosteps", 256),
]


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    for hp in HYPER_PARAMS:
        hp.add_argument(parser)
    return parser


def main():
    train()

    args = args_parser().parse_args()
    args_dict = vars(args)

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
    logging.basicConfig(level=logging.INFO)
    train(hps)


if __name__ == "__main__":
    main()
