import argparse
import logging
import os
import subprocess
import time

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

  while True:
    elapsed = time.time() - last_time
    if elapsed > log_interval:
      logging.info(f"{frames/elapsed}fps")
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
        codecraft.act(game_id)
      frames += 1


def train(hps):
  env = envs.CodeCraftVecEnv(64)
  ppo2.learn(
          network=lambda it: network(hps, it),
    env=env,
    gamma=0.9,
    nsteps=hps["rosteps"],
    total_timesteps=hps["steps"],
    log_interval=1,
    lr=hps["lr"])

def network(hps, input_tensor):
    #with tf.variable_scope(scope, reuse=reuse):
    out = input_tensor
    for _ in range(hps["nl"]):
        out = layers.fully_connected(out, num_outputs=hps["nh"], activation_fn=None)
        out = tf.nn.relu(out)
    q_out = out
    q_out = layers.fully_connected(out, num_outputs=6, activation_fn=None)
    return q_out


class Hyperparam:
    def __init__(self, name, shortname, default):
        self.name = name
        self.shortname = shortname
        self.default = default

    def add_argument(self, parser):
        parser.add_argument(f"--{self.shortname}", f"--{self.name}", type=type(self.default))


HYPER_PARAMS = [
    Hyperparam("learning-rate", "lr", 3e-4),
    Hyperparam("num-layers", "nl", 3),
    Hyperparam("num-hidden", "nh", 1024),
    Hyperparam("total-timesteps", "steps", 2e7),
    Hyperparam("sequential-rollout-steps", "rosteps", 256),
]


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    for hp in HYPER_PARAMS:
        hp.add_argument(parser)
    return parser

def main():
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


if __name__== "__main__":
  main()

