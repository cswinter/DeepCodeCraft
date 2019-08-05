import argparse
import logging
import os
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import wandb
from baselines import logger

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


def train(sequential_rollout_steps: int = 256,
          total_rollout_steps: int = 256 * 64,
          optimizer_batch_size: int = 4096,
          gamma: int = 0.9) -> None:
    assert(total_rollout_steps % optimizer_batch_size == 0)

    lr = 0.1
    momentum = 0.9
    optimizer = 'SGD'
    game_length = 3 * 60 * 60
    objective = envs.Objective.DISTANCE_TO_ORIGIN
    commit = subprocess.check_output(["git", "describe", "--tags", "--always", "--dirty"]).decode("UTF-8")
    wandb.config.update({
        'sequential_rollout_steps': sequential_rollout_steps,
        'total_rollout_steps': total_rollout_steps,
        'optimizer_batch_size': optimizer_batch_size,
        'gamma': gamma,
        'lr': lr,
        'momentum': momentum,
        'optimizer': optimizer,
        'objective': objective,
        'commit': commit,
    })

    num_envs = total_rollout_steps // sequential_rollout_steps
    env = envs.CodeCraftVecEnv(num_envs, game_length, envs.Objective.DISTANCE_TO_ORIGIN)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = Policy(4, 1024)
    policy.to(device)
    optimizer = optim.SGD(policy.parameters(), lr=0.1, momentum=0.9)

    wandb.watch(policy)

    total_steps = 0
    epoch = 0
    last_obs = env.reset()
    eprewmean = 0
    eplenmean = 0
    while True:
        episode_start = time.time()
        all_obs = []
        all_actions = []
        all_rewards = []

        # Rollout
        for step in range(sequential_rollout_steps):
            obs = torch.tensor(last_obs).to(device)
            actions = policy.evaluate(obs)

            all_obs.extend(last_obs)
            all_actions.extend(actions)

            last_obs, rews, dones, infos = env.step(actions)
            for info in infos:
                eprewmean = eprewmean * 0.95 + 0.05 * info['episode']['r']
                eplenmean = eplenmean * 0.95 + 0.05 * info['episode']['l']

            all_rewards.extend(rews)

        # Policy Update
        # TODO: shuffle
        episode_loss = 0
        for batch in range(int(total_rollout_steps / optimizer_batch_size)):
            start = optimizer_batch_size * batch
            end = optimizer_batch_size * (batch + 1)

            obs = torch.tensor(all_obs[start:end]).to(device)
            actions = torch.tensor(all_actions[start:end]).to(device)
            returns = torch.tensor(all_rewards[start:end]).to(device)

            optimizer.zero_grad()
            episode_loss += policy.backprop(obs, actions, returns)
            optimizer.step()

        epoch += 1
        total_steps += total_rollout_steps
        throughput = int(total_rollout_steps / (time.time() - episode_start))

        wandb.log({
            'step': total_steps,
            'loss': episode_loss,
            'throughput': throughput,
            'eprewmean': eprewmean,
            'eplenmean': eplenmean,
        })

        print(f'{throughput} samples/s')
        print(episode_loss)


class Policy(nn.Module):
    def __init__(self, layers, nhidden):
        super(Policy, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(47, nhidden)])
        for layer in range(layers - 1):
            self.fc_layers.append(nn.Linear(nhidden, nhidden))
        # TODO: init to 0
        self.dense_final = nn.Linear(nhidden, 8)

    def evaluate(self, observation):
        probs = self.forward(observation)
        actions = []
        probs.detach_()
        for i in range(probs.size()[0]):
            actions.append(np.random.choice(8, 1, p=probs[i].cpu().numpy())[0])
        return actions

    def backprop(self, obs, actions, returns):
        logits = self.logits(obs)
        loss = torch.sum(returns * F.cross_entropy(logits, actions))
        loss.backward()
        return loss.data.tolist()

    def forward(self, x):
        return F.softmax(self.logits(x), dim=1)

    def logits(self, x):
        for fc in self.fc_layers:
            x = F.relu(fc(x))
        return self.dense_final(x)


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
    wandb.init(project="deep-codecraft")
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
