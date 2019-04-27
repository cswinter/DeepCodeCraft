import gym
from baselines.common.vec_env import VecEnv
from gym.utils import seeding
from gym import spaces
import numpy as np

import codecraft



class CodeCraftVecEnv(VecEnv):
    def __init__(self, num_envs):
        observations_low = []
        observations_high = []
        # Drone x, y
        observations_low.extend([-3, -2])
        observations_high.extend([3, 2])
        # Drone orientation unit vector
        observations_low.extend([-1, -1])
        observations_high.extend([1, 1])
        # 10 closest minerals
        for _ in range(0, 10):
            # x, y
            observations_low.extend([-3, -2])
            observations_high.extend([3, 2])
            # size
            observations_low.append(0)
            observations_high.append(2)

        super().__init__(
            num_envs,
            spaces.Box(
                low=np.array(observations_low),
                high=np.array(observations_high),
                dtype=np.float32),
            spaces.Discrete(6))
        self.games = []
        self.eplen = []
        self.eprew = []
        self.score = []

    def reset(self):
        self.games = []
        self.eplen = []
        self.score = []
        for _ in range(self.num_envs):
            game_id = codecraft.create_game()
            # print("Starting game:", game_id)
            self.games.append(game_id)
            self.eplen.append(1)
            self.eprew.append(1)
            self.score.append(None)
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

            o = []
            x = float(observation['alliedDrones'][0]['xPos'])
            y = float(observation['alliedDrones'][0]['yPos'])
            o.append(x / 1000.0)
            o.append(x / 1000.0)
            o.append(np.sin(float(observation['alliedDrones'][0]['orientation'])))
            o.append(np.cos(float(observation['alliedDrones'][0]['orientation'])))
            minerals = sorted(observation['minerals'], key=lambda m: dist2(m['xPos'], m['yPos'], x, y))
            for m in range(0, 10):
                if m < len(minerals):
                    o.append(float(minerals[m]['xPos'] / 1000.0))
                    o.append(float(minerals[m]['yPos'] / 1000.0))
                    o.append(float(minerals[m]['size'] / 100.0))
                else:
                    o.extend([0.0, 0.0, 0.0])
            obs.append(np.array(o))

            game_id = self.games[i]
            score = 0
            if len(minerals) > 0:
                score = np.maximum(0.0, 10.0 - np.sqrt(dist2(minerals[0]['xPos'], minerals[0]['yPos'], x, y)) / 50.0)
            if self.score[i] is None:
                self.score[i] = 0
            reward = score - self.score[i]
            self.score[i] = score
            if len(observation['winner']) > 0:
                # print(f'Game {game_id} won by {observation["winner"][0]}')
                game_id = codecraft.create_game()
                # print("Starting game:", game_id)
                self.games[i] = game_id
                dones.append(1.0)
                infos.append({'episode': { 'r': self.eprew[i], 'l': self.eplen[i]}})
                self.eplen[i] = 1
                self.eprew[i] = reward
                self.score[i] = None
            else:
                self.eplen[i] += 1
                self.eprew[i] += reward
                dones.append(0.0)

            rews.append(reward)

        return np.array(obs), np.array(rews), np.array(dones), infos

def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy
