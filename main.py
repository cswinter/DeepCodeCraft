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


LOG_ROOT_DIR = '/home/clemens/Dropbox/artifacts/DeepCodeCraft'


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


def train():
  env = envs.CodeCraftVecEnv(64)
  ppo2.learn(
    network=network,
    env=env,
    gamma=0.9,
    nsteps=256,
    total_timesteps=1e8,
    log_interval=1,
    num_hidden=1024,
    num_layers=3,
    lr=3e-5)

def network(input_tensor):
    #with tf.variable_scope(scope, reuse=reuse):
    out = input_tensor
    for hidden in [1024, 1024, 1024]:
        out = layers.fully_connected(out, num_outputs=hidden, activation_fn=None)
        out = tf.nn.relu(out)
    q_out = out
    q_out = layers.fully_connected(out, num_outputs=6, activation_fn=None)
    return q_out

def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("UTF-8")[:8]
    t = time.strftime("%Y-%m-%d~%H:%M:%S")
    logger.configure(dir=os.path.join(LOG_ROOT_DIR, f"{t}-{commit}"),
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])
    logging.basicConfig(level=logging.INFO)
    train()


if __name__== "__main__":
  main()

