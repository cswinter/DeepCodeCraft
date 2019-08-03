import gym
from baselines.common.vec_env import VecEnv
from gym.utils import seeding
from gym import spaces
import numpy as np

import codecraft


class CodeCraftVecEnv(VecEnv):
    def __init__(self, num_envs, game_length):
        observations_low = []
        observations_high = []
        # Drone x, y
        observations_low.extend([-3, -2])
        observations_high.extend([3, 2])
        # Drone orientation unit vector
        observations_low.extend([-1, -1])
        observations_high.extend([1, 1])
        # Drone resources
        observations_low.append(0)
        observations_high.append(2)
        # Is constructing, is harvesting
        observations_low.extend([-1, -1])
        observations_high.extend([1, 1])
        # 10 closest minerals
        for _ in range(0, 10):
            # relative x, y and distance
            observations_low.extend([-3, -2, 0])
            observations_high.extend([3, 2, 8])
            # size
            observations_low.append(0)
            observations_high.append(2)
        super().__init__(
            num_envs,
            spaces.Box(
                low=np.array(observations_low),
                high=np.array(observations_high),
                dtype=np.float32),
            spaces.Discrete(8))

        self.games = []
        self.eplen = []
        self.eprew = []
        self.score = []
        self.game_length = game_length

    def reset(self):
        self.games = []
        self.eplen = []
        self.score = []
        for i in range(self.num_envs):
            # spread out initial game lengths to stagger start times
            game_id = codecraft.create_game((self.game_length * (i + 1) // self.num_envs))
            # print("Starting game:", game_id)
            self.games.append(game_id)
            self.eplen.append(1)
            self.eprew.append(1)
            self.score.append(None)
        return self.observe()[0]

    def step_async(self, actions):
        game_actions = []
        for (game_id, action) in zip(self.games, actions):
            # 0-5: turn/movement (4 is no turn, no movement)
            # 6: build [0,1,0,0,0] drone (if minerals > 5)
            # 7: harvest
            move = False
            harvest = False
            turn = 0
            build = []
            if action == 0 or action == 1 or action == 2:
                move = True
            if action == 0 or action == 3:
                turn = -1
            if action == 2 or action == 5:
                turn = 1
            if action == 6:
                build = [[0, 1, 0, 0, 0]]
            if action == 7:
                harvest = True
            game_actions.append((game_id, move, turn, build, harvest))

        codecraft.act_batch(game_actions)

    def step_wait(self):
        return self.observe()

    def observe(self):
        obs = []
        rews = []
        dones = []
        infos = []
        for (i, observation) in enumerate(codecraft.observe_batch(self.games)):
            obs.append(codecraft.observation_to_np(observation))

            score = float(observation['alliedScore']) * 0.1
            if self.score[i] is None:
                self.score[i] = score
            reward = score - self.score[i]
            self.score[i] = score

            if len(observation['winner']) > 0:
                # print(f'Game {game_id} won by {observation["winner"][0]}')
                game_id = codecraft.create_game()
                # print("Starting game:", game_id)
                self.games[i] = game_id
                dones.append(1.0)
                infos.append({'episode': {'r': self.eprew[i], 'l': self.eplen[i]}})
                self.eplen[i] = 1
                self.eprew[i] = reward
                self.score[i] = None
            else:
                self.eplen[i] += 1
                self.eprew[i] += reward
                dones.append(0.0)

            rews.append(reward)

        return np.array(obs, dtype=np.float32), np.array(rews), np.array(dones), infos


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)
