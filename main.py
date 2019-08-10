import logging
import subprocess
import time
import os

import torch
import torch.optim as optim
import numpy as np

import wandb

import codecraft
from gym_codecraft import envs
from hyper_params import HyperParams
from policy import Policy

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


def train(hps: HyperParams) -> None:
    assert(hps.rosteps % hps.bs == 0)

    num_envs = hps.rosteps // hps.seq_rosteps
    env = envs.CodeCraftVecEnv(num_envs, hps.game_length, hps.objective)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("Running on CPU")
        device = "cpu"

    policy = Policy(hps.depth, hps.width)
    policy.to(device)
    optimizer = optim.SGD(policy.parameters(), lr=hps.lr, momentum=hps.momentum)

    wandb.watch(policy)

    total_steps = 0
    epoch = 0
    obs = env.reset()
    eprewmean = 0
    eplenmean = 0
    completed_episodes = 0
    while total_steps < hps.steps:
        episode_start = time.time()
        entropies = []
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []

        # Rollout
        for step in range(hps.seq_rosteps):
            obs_tensor = torch.tensor(obs).to(device)
            actions, entropy = policy.evaluate(obs_tensor)

            entropies.append(entropy)

            all_obs.extend(obs)
            all_actions.extend(actions)

            obs, rews, dones, infos = env.step(actions)

            all_rewards.extend(rews)
            all_dones.extend(dones)

            for info in infos:
                ema = min(95, completed_episodes * 10) / 100.0
                eprewmean = eprewmean * ema + (1 - ema) * info['episode']['r']
                eplenmean = eplenmean * ema + (1 - ema) * info['episode']['l']
                completed_episodes += 1

        all_returns = np.zeros(len(all_rewards), dtype=np.float32)
        ret = np.zeros(num_envs)
        retscale = 1.0 - hps.gamma
        for t in reversed(range(hps.seq_rosteps)):
            # TODO: correction at end of rollout/episode
            for i in range(num_envs):
                ret[i] = hps.gamma * ret[i] + all_rewards[t * num_envs + i]
                all_returns[t * num_envs + i] = ret[i] * retscale
                if all_dones[t * num_envs + i] == 1:
                    ret[i] = 0

        # Policy Update
        # TODO: shuffle
        episode_loss = 0
        for batch in range(int(hps.rosteps / hps.bs)):
            start = hps.bs * batch
            end = hps.bs * (batch + 1)

            o = torch.tensor(all_obs[start:end]).to(device)
            actions = torch.tensor(all_actions[start:end]).to(device)
            returns = torch.tensor(all_returns[start:end]).to(device)

            optimizer.zero_grad()
            episode_loss += policy.backprop(o, actions, returns)
            optimizer.step()

        epoch += 1
        total_steps += hps.rosteps
        throughput = int(hps.rosteps / (time.time() - episode_start))

        wandb.log({
            'loss': episode_loss / hps.rosteps,
            'throughput': throughput,
            'eprewmean': eprewmean,
            'eplenmean': eplenmean,
            'entropy': sum(entropies) / len(entropies),
        }, step=total_steps)

        print(f'{throughput} samples/s')
        print(episode_loss)


def main():
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
    logging.basicConfig(level=logging.INFO)
    train(hps)
"""


if __name__ == "__main__":
    main()

