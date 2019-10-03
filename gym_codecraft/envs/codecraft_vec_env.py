from multiprocessing import Queue, Pipe, Process, Value
from baselines.common.vec_env import VecEnv
from enum import Enum
from gym import spaces
import numpy as np

from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.envs.base import Env

import codecraft

def obs_low_high():
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
    return observations_low, observations_high


class CodeCraftRlpytEnv(Env):
    def __init__(self, action_queue, obs_pipe, index, just_sample, objective):
        self.objective = objective
        self._action_space = IntBox(low=0, high=8)
        obs_low, obs_high = obs_low_high()
        self._observation_space = FloatBox(low=obs_low, high=obs_high)

        self.ignore_reset = False
        self.require_example = just_sample
        self.action_queue = action_queue
        self.obs_pipe = obs_pipe
        self.index = index
        self.last_obs = None

    def step(self, action):
        # print(f'stepping env {self.index}')
        if self.index == 0 and self.require_example:
            self.require_example = False
            tmp_env = CodeCraftVecEnv(1, 1, self.objective, 0)
            tmp_env.reset()
            tmp_env.step_async([0])
            obs, rews, dones, infos = tmp_env.observe()
            return obs[0], rews[0], dones[0], ()
        self.action_queue.put((self.index, action))
        return self.obs()

    def reset(self):
        if self.ignore_reset:
            # print("IGNORE RESET")
            return self.last_obs
        print("RESETTING")
        self.ignore_reset = True
        print("rcv")
        o = self.obs_pipe.recv()
        print('success')
        return o  # self.env.reset()[0]

    def obs(self):
        obs = self.obs_pipe.recv()
        self.last_obs = obs[0]
        return obs

    @property
    def horizon(self):
        return None


class Counter(object):
    def __init__(self):
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n
            return self.val.value

    @property
    def value(self):
        return self.val.value


class SharedBatchingEnv:
    def __init__(self, num_envs, game_length, objective, action_delay):
        self.objective = objective
        self.num_envs = num_envs
        self.vec_env = CodeCraftVecEnv(num_envs, game_length, objective, action_delay)
        self.action_queue = Queue()
        self.pipes = list([Pipe() for _ in range(num_envs)])
        self.obs_pipe = list([self.pipes[i][0] for i in range(num_envs)])
        self.actions = list([None for _ in range(num_envs)])
        self.envs_created = Counter()

        obs = self.vec_env.reset()
        for i in range(self.num_envs):
            print(f'populating pipe {i}')
            self.obs_pipe[i].send(obs[i])
        self.obs_pipe[0].send(obs[0])

        p = Process(target=self.run)
        p.start()

    def run(self):
        print('STARTING RUN LOOP')
        while True:
            action_cnt = 0
            while action_cnt < self.num_envs:
                i, action = self.action_queue.get()
                #print(f'received action for {i}')
                assert self.actions[i] is None
                self.actions[i] = action
                action_cnt += 1
                #print(f'received {action_cnt} actions')

            self.vec_env.step_async(self.actions)
            obs, rews, dones, infos = self.vec_env.observe()
            for i in range(self.num_envs):
                self.actions[i] = None
                self.obs_pipe[i].send((obs[i], rews[i], dones[i], ()))

    def create_env(self):
        print("NOW")
        index = self.envs_created.increment() - 1
        just_sample = index == 0
        if index == self.num_envs:
            index = 0
        assert index < self.num_envs
        receiver = self.pipes[index][1]
        env = CodeCraftRlpytEnv(self.action_queue, receiver, index, just_sample, self.objective)
        print(f'Created env {index}/{self.num_envs}')

        return env


class CodeCraftVecEnv(VecEnv):
    def __init__(self, num_envs, game_length, objective, action_delay):
        self.objective = objective
        self.action_delay = action_delay

        observations_low, observations_high = obs_low_high()
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
        self.last_obs = None

    def reset(self):
        self.games = []
        self.eplen = []
        self.score = []
        for i in range(self.num_envs):
            # spread out initial game lengths to stagger start times
            game_id = codecraft.create_game(self.game_length * (i + 1) // self.num_envs, self.action_delay)
            # print("Starting game:", game_id)
            self.games.append(game_id)
            self.eplen.append(1)
            self.eprew.append(0)
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

        codecraft.act_batch(game_actions, disable_harvest=self.objective == Objective.DISTANCE_TO_CRYSTAL)

    def step_wait(self):
        return self.observe()

    def observe(self):
        rews = []
        dones = []
        infos = []
        obs = codecraft.observe_batch_raw(self.games)
        global_features = 2
        dstride = 7
        mstride = 4
        stride = global_features + dstride + 10 * mstride
        for i in range(self.num_envs):
            x = obs[stride * i + global_features + 0]
            y = obs[stride * i + global_features + 1]
            if self.objective == Objective.ALLIED_WEALTH:
                score = obs[stride * i + 1] * 0.1
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

            if obs[stride * self.num_envs + i] > 0:
                game_id = codecraft.create_game(self.game_length, self.action_delay)
                self.games[i] = game_id
                observation = codecraft.observe(game_id)
                # TODO
                # obs[stride * i:stride * (i + 1)] = codecraft.observation_to_np(observation)

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

        self.last_obs = obs[:stride * self.num_envs].reshape(self.num_envs, -1),\
               np.array(rews),\
               np.array(dones),\
               infos
        return self.last_obs


class Objective(Enum):
    ALLIED_WEALTH = 'ALLIED_WEALTH'
    DISTANCE_TO_CRYSTAL = 'DISTANCE_TO_CRYSTAL'
    DISTANCE_TO_ORIGIN = 'DISTANCE_TO_ORIGIN'
    DISTANCE_TO_1000_500 = 'DISTANCE_TO_1000_500'


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)
