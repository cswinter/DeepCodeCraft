from collections import defaultdict

from baselines.common.vec_env import VecEnv
from enum import Enum
from gym import spaces
import numpy as np

import codecraft


def map_arena_tiny():
    return {
        'mapWidth': 1000,
        'mapHeight': 1000,
        'player1Drones': [
            {
                'xPos': np.random.randint(-450, 450),
                'yPos': np.random.randint(-450, 450),
                'resources': 0,
                'storageModules': 1,
                'missileBatteries': 0,
                'constructors': 1,
                'engines': 0,
                'shieldGenerators': 0,
            }
        ],
        'player2Drones': [
            {
                'xPos': np.random.randint(-450, 450),
                'yPos': np.random.randint(-450, 450),
                'resources': 0,
                'storageModules': 0,
                'missileBatteries': 1,
                'constructors': 0,
                'engines': 0,
                'shieldGenerators': 3,
            }
        ]
    }


class CodeCraftVecEnv(VecEnv):
    def __init__(self, num_envs, num_self_play, objective, action_delay, stagger=True):
        assert(num_envs >= 2 * num_self_play)
        self.objective = objective
        self.action_delay = action_delay
        self.num_self_play = num_self_play
        self.stagger = stagger
        self.game_length = 3 * 60 * 60
        self.custom_map = lambda: None
        if objective == Objective.ARENA_TINY:
            self.game_length = 1 * 60 * 60
            self.custom_map = map_arena_tiny

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

    def reset(self):
        self.games = []
        self.eplen = []
        self.score = []
        for i in range(self.num_envs - self.num_self_play):
            # spread out initial game lengths to stagger start times
            self_play = i < self.num_self_play
            game_length = self.game_length * (i + 1) // (self.num_envs - self.num_self_play) if self.stagger else self.game_length
            game_id = codecraft.create_game(
                game_length,
                self.action_delay,
                self_play,
                self.custom_map())
            # print("Starting game:", game_id)
            self.games.append((game_id, 0))
            self.eplen.append(1)
            self.eprew.append(0)
            self.score.append(None)
            if self_play:
                self.games.append((game_id, 1))
                self.eplen.append(1)
                self.eprew.append(0)
                self.score.append(None)
        return self.observe()[0]

    def step_async(self, actions):
        game_actions = []
        for ((game_id, player_id), action) in zip(self.games, actions):
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
            game_actions.append((game_id, player_id, move, turn, build, harvest))

        codecraft.act_batch(game_actions, disable_harvest=self.objective == Objective.DISTANCE_TO_CRYSTAL)

    def step_wait(self):
        return self.observe()

    def observe(self):
        rews = []
        dones = []
        infos = []
        obs = codecraft.observe_batch_raw(self.games)
        global_features = 1
        nonobs_features = 3
        dstride = 13
        mstride = 4
        stride = global_features + dstride + 10 * mstride + 10 * dstride
        for i in range(self.num_envs):
            x = obs[stride * i + global_features + 0]
            y = obs[stride * i + global_features + 1]
            if self.objective == Objective.ARENA_TINY:
                allied_score = obs[stride * self.num_envs + i * nonobs_features + 1]
                enemy_score = obs[stride * self.num_envs + i * nonobs_features + 2]
                score = 2 * allied_score / (allied_score + enemy_score + 1e-8)
            elif self.objective == Objective.ALLIED_WEALTH:
                score = obs[stride * self.num_envs + i * nonobs_features + 1] * 0.1
            elif self.objective == Objective.DISTANCE_TO_ORIGIN:
                score = -dist(x, y, 0.0, 0.0)
            elif self.objective == Objective.DISTANCE_TO_1000_500:
                score = -dist(x, y, 1.0, 0.5)
            elif self.objective == Objective.DISTANCE_TO_CRYSTAL:
                score = 0
                for j in range(10):
                    offset = stride * i + global_features + dstride + mstride * j
                    distance = obs[offset + 2]
                    size = obs[offset + 3]
                    if size == 0:
                        break
                    nearness = 0.5 - distance
                    score = max(score, 20 * nearness * size)
            else:
                raise Exception(f"Unknown objective {self.objective}")

            # TODO: this is a workaround for reward spikes most likely caused by minerals not beeing visible until first movement
            if self.eplen[i] < 3:
                self.score[i] = score
            # if self.score[i] is None:
            #    self.score[i] = score
            reward = score - self.score[i]
            self.score[i] = score

            if obs[stride * self.num_envs + i * nonobs_features] > 0:
                (game_id, pid) = self.games[i]
                if pid == 0:
                    self_play = i // 2 < self.num_self_play
                    game_id = codecraft.create_game(self.game_length,
                                                    self.action_delay,
                                                    self_play,
                                                    self.custom_map())
                    self.games[i] = (game_id, 0)
                    if self_play:
                        self.games[i + 1] = (game_id, 1)
                observation = codecraft.observe(game_id, pid)
                # TODO
                # obs[stride * i:stride * (i + 1)] = codecraft.observation_to_np(observation)

                dones.append(1.0)
                infos.append({'episode': {'r': self.eprew[i], 'l': self.eplen[i], 'index': i}})
                self.eplen[i] = 1
                self.eprew[i] = 0
                self.score[i] = None
            else:
                self.eplen[i] += 1
                self.eprew[i] += reward
                dones.append(0.0)

            rews.append(reward)

        return obs[:stride * self.num_envs].reshape(self.num_envs, -1),\
               np.array(rews),\
               np.array(dones),\
               infos

    def close(self):
        # Run all games to completion
        done = defaultdict(lambda: False)
        running = len(self.games)
        while running > 0:
            game_actions = []
            active_games = []
            for (game_id, player_id) in self.games:
                if not done[game_id]:
                    active_games.append((game_id, player_id))
                    game_actions.append((game_id, player_id, False, 0, [], False))
            codecraft.act_batch(game_actions)
            obs = codecraft.observe_batch(active_games)
            for o, (game_id, _) in zip(obs, active_games):
                if o['winner']:
                    done[game_id] = True
                    running -= 1


class Objective(Enum):
    ALLIED_WEALTH = 'ALLIED_WEALTH'
    DISTANCE_TO_CRYSTAL = 'DISTANCE_TO_CRYSTAL'
    DISTANCE_TO_ORIGIN = 'DISTANCE_TO_ORIGIN'
    DISTANCE_TO_1000_500 = 'DISTANCE_TO_1000_500'
    ARENA_TINY = 'ARENA_TINY'


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)
