import gym
from baselines.common.vec_env import VecEnv
from gym.utils import seeding
from gym import spaces
import numpy as np

import codecraft


class CodeCraftVecEnv(VecEnv):
    def __init__(self, num_envs):
        super().__init__(
            num_envs,
            spaces.Box(
                low=np.array([-2.5, -2.5, 0]),
                high=np.array([2.5, 2.5, np.pi * 2]),
                dtype=np.float32),
            spaces.Discrete(6))
        self.games = []
        self.eplen = []
        self.eprew = []

    def reset(self):
        self.games = []
        self.eplen = []
        for _ in range(self.num_envs):
            game_id = codecraft.create_game()
            # print("Starting game:", game_id)
            self.games.append(game_id)
            self.eplen.append(1)
            self.eprew.append(1)
        return self.observe()[0]

    def step_async(self, actions):
        game_actions = []
        for (game_id, action) in zip(self.games, actions):
            move = False
            turn = 0
            if action == 0 or action == 1 or action == 2:
                move = True
            if action == 0 or action == 3:
                turn = -1
            if action == 2 or action == 5:
                turn = 1
            game_actions.append((game_id, move, turn))

        codecraft.act_batch(game_actions)

    def step_wait(self):
        return self.observe()

    def observe(self):
        obs = []
        rews = []
        dones = []
        infos = []
        for (i, observation) in enumerate(codecraft.observe_batch(self.games)):
            game_id = self.games[i]
            x = (observation['alliedDrones'][0]['xPos'] - 750) / 2000
            y = (observation['alliedDrones'][0]['yPos']) / 2000
            reward = 1 - np.sqrt((x * x + y * y))
            reward *= 0.1
            if len(observation['winner']) > 0:
                # print(f'Game {game_id} won by {observation["winner"][0]}')
                game_id = codecraft.create_game()
                # print("Starting game:", game_id)
                self.games[i] = game_id
                dones.append(1.0)
                infos.append({'episode': { 'r': self.eprew[i], 'l': self.eplen[i]}})
                self.eplen[i] = 1
                self.eprew[i] = reward
            else:
                self.eplen[i] += 1
                self.eprew[i] += reward
                dones.append(0.0)

            obs.append(np.array([
                float(observation['alliedDrones'][0]['xPos'] / 2000.0),
                float(observation['alliedDrones'][0]['yPos'] / 2000.0),
                float(observation['alliedDrones'][0]['orientation']),
            ]))
            rews.append(reward)

        return np.array(obs), np.array(rews), np.array(dones), infos
