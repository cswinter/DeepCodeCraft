from collections import defaultdict

from enum import Enum
from dataclasses import dataclass
import numpy as np

import codecraft


@dataclass
class ObsConfig:
    allies: int
    drones: int
    minerals: int


GLOBAL_FEATURES = 1
DSTRIDE = 15
MSTRIDE = 4
NONOBS_FEATURES = 3
DEFAULT_OBS_CONFIG = ObsConfig(allies=2, drones=4, minerals=2)


def drone_dict(x, y,
          storage_modules=0,
          missile_batteries=0,
          constructors=0,
          engines=0,
          shield_generators=0,
          resources=0):
    return {
        'xPos': x,
        'yPos': y,
        'resources': resources,
        'storageModules': storage_modules,
        'missileBatteries': missile_batteries,
        'constructors': constructors,
        'engines': engines,
        'shieldGenerators': shield_generators,
    }


def random_drone():
    modules = ['storageModules', 'constructors', 'missileBatteries', 'shieldGenerators', 'missileBatteries']
    drone = drone_dict(np.random.randint(-450, 450), np.random.randint(-450, 450))
    for _ in range(0, np.random.randint(2, 5)):
        module = modules[np.random.randint(0, len(modules))]
        drone[module] += 1
    return drone


def map_arena_tiny_random():
    return {
        'mapWidth': 1000,
        'mapHeight': 1000,
        'player1Drones': [random_drone()],
        'player2Drones': [random_drone()],
    }


def map_arena_tiny(randomize: bool):
    storage_modules = 1
    constructors = 1
    missiles_batteries = 1
    if randomize:
        storage_modules = np.random.randint(1, 3)
        constructors = np.random.randint(1, 3)
        missiles_batteries = np.random.randint(1, 3)
    return {
        'mapWidth': 1000,
        'mapHeight': 1000,
        'player1Drones': [
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       storage_modules=storage_modules,
                       constructors=constructors)
        ],
        'player2Drones': [
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       missile_batteries=missiles_batteries,
                       shield_generators=4 - missiles_batteries)
        ],
    }


def map_arena_tiny_2v2(randomize: bool):
    s1 = 1
    s2 = 1
    if randomize:
        s1 = np.random.randint(0, 2)
        s2 = np.random.randint(0, 2)
    return {
        'mapWidth': 1000,
        'mapHeight': 1000,
        'player1Drones': [
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       missile_batteries=1-s1,
                       shield_generators=s1),
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       missile_batteries=1),
        ],
        'player2Drones': [
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       missile_batteries=1-s2,
                       shield_generators=s2),
            drone_dict(np.random.randint(-450, 450),
                       np.random.randint(-450, 450),
                       missile_batteries=1),
        ],
    }


class CodeCraftVecEnv(object):
    def __init__(self,
                 num_envs,
                 num_self_play,
                 objective,
                 action_delay,
                 stagger=True,
                 fair=False,
                 randomize=False,
                 use_action_masks=True,
                 obs_config=DEFAULT_OBS_CONFIG):
        assert(num_envs >= 2 * num_self_play)
        self.num_envs = num_envs
        self.objective = objective
        self.action_delay = action_delay
        self.num_self_play = num_self_play
        self.stagger = stagger
        self.fair = fair
        self.game_length = 3 * 60 * 60
        self.custom_map = lambda _: None
        self.last_map = None
        self.randomize = randomize
        self.use_action_masks = use_action_masks
        self.obs_config = obs_config
        if objective == Objective.ARENA_TINY:
            self.game_length = 1 * 60 * 60
            self.custom_map = map_arena_tiny
        elif objective == Objective.ARENA_TINY_2V2:
            self.game_length = 1 * 30 * 60
            self.custom_map = map_arena_tiny_2v2

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
                self.next_map())
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
        obs, _, _, _, action_masks = self.observe()
        return obs, action_masks

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        game_actions = []
        for ((game_id, player_id), player_actions) in zip(self.games, actions):
            player_actions2 = []
            for action in player_actions:
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
                player_actions2.append((move, turn, build, harvest))
            game_actions.append((game_id, player_id, player_actions2))

        codecraft.act_batch(game_actions, disable_harvest=self.objective == Objective.DISTANCE_TO_CRYSTAL)

    def step_wait(self):
        return self.observe()

    def observe(self):
        rews = []
        dones = []
        infos = []
        obs = codecraft.observe_batch_raw(self.games,
                                          allies=self.obs_config.allies,
                                          drones=self.obs_config.drones,
                                          minerals=self.obs_config.minerals)
        stride = self.obs_config.allies * (GLOBAL_FEATURES +
                                           DSTRIDE +
                                           self.obs_config.minerals * MSTRIDE +
                                           self.obs_config.drones * DSTRIDE)
        for i in range(self.num_envs):
            x = obs[stride * i + GLOBAL_FEATURES + 0]
            y = obs[stride * i + GLOBAL_FEATURES + 1]
            if self.objective == Objective.ARENA_TINY or self.objective == Objective.ARENA_TINY_2V2:
                allied_score = obs[stride * self.num_envs + i * NONOBS_FEATURES + 1]
                enemy_score = obs[stride * self.num_envs + i * NONOBS_FEATURES + 2]
                score = 2 * allied_score / (allied_score + enemy_score + 1e-8) - 1
            elif self.objective == Objective.ALLIED_WEALTH:
                score = obs[stride * self.num_envs + i * NONOBS_FEATURES + 1] * 0.1
            elif self.objective == Objective.DISTANCE_TO_ORIGIN:
                score = -dist(x, y, 0.0, 0.0)
            elif self.objective == Objective.DISTANCE_TO_1000_500:
                score = -dist(x, y, 1.0, 0.5)
            elif self.objective == Objective.DISTANCE_TO_CRYSTAL:
                score = 0
                for j in range(10):
                    offset = stride * i + GLOBAL_FEATURES + DSTRIDE + MSTRIDE * j
                    distance = obs[offset + 2]
                    size = obs[offset + 3]
                    if size == 0:
                        break
                    nearness = 0.5 - distance
                    score = max(score, 20 * nearness * size)
            else:
                raise Exception(f"Unknown objective {self.objective}")

            if self.score[i] is None:
                self.score[i] = score
            reward = score - self.score[i]
            self.score[i] = score
            self.eprew[i] += reward

            if obs[stride * self.num_envs + i * NONOBS_FEATURES] > 0:
                (game_id, pid) = self.games[i]
                if pid == 0:
                    self_play = i // 2 < self.num_self_play
                    game_id = codecraft.create_game(self.game_length,
                                                    self.action_delay,
                                                    self_play,
                                                    self.next_map())
                    self.games[i] = (game_id, 0)
                    if self_play:
                        self.games[i + 1] = (game_id, 1)
                observation = codecraft.observe(game_id, pid)
                # TODO
                # obs[stride * i:stride * (i + 1)] = codecraft.observation_to_np(observation)

                dones.append(1.0)
                infos.append({'episode': {
                    'r': self.eprew[i],
                    'l': self.eplen[i],
                    'index': i,
                    'score': self.score[i],
                }})
                self.eplen[i] = 1
                self.eprew[i] = 0
                self.score[i] = None
            else:
                self.eplen[i] += 1
                dones.append(0.0)

            rews.append(reward)

        action_masks = obs[-8 * self.obs_config.allies * self.num_envs:].reshape(-1, self.obs_config.allies, 8)

        return obs[:stride * self.num_envs].reshape(self.num_envs, -1),\
               np.array(rews),\
               np.array(dones),\
               infos,\
               action_masks if self.use_action_masks\
                            else np.ones([self.num_envs, self.obs_config.allies, 8], dtype=np.float32)

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
                    game_actions.append((game_id, player_id, [(False, 0, [], False)]))
            codecraft.act_batch(game_actions)
            obs = codecraft.observe_batch(active_games)
            for o, (game_id, _) in zip(obs, active_games):
                if o['winner']:
                    done[game_id] = True
                    running -= 1

    def next_map(self):
        if self.fair:
            return self.fair_map()
        else:
            return self.custom_map(self.randomize)

    def fair_map(self):
        if self.last_map is None:
            self.last_map = self.custom_map(self.randomize)
            return self.last_map
        else:
            result = self.last_map
            self.last_map = None
            p1 = result['player1Drones']
            result['player1Drones'] = result['player2Drones']
            result['player2Drones'] = p1
            return result


class Objective(Enum):
    ALLIED_WEALTH = 'ALLIED_WEALTH'
    DISTANCE_TO_CRYSTAL = 'DISTANCE_TO_CRYSTAL'
    DISTANCE_TO_ORIGIN = 'DISTANCE_TO_ORIGIN'
    DISTANCE_TO_1000_500 = 'DISTANCE_TO_1000_500'
    ARENA_TINY = 'ARENA_TINY'
    ARENA_TINY_2V2 = 'ARENA_TINY_2V2'

    def vs(self):
        if self == Objective.ALLIED_WEALTH or\
           self == Objective.DISTANCE_TO_CRYSTAL or\
           self == Objective.DISTANCE_TO_ORIGIN or\
           self == Objective.DISTANCE_TO_1000_500:
           return False
        elif self == Objective.ARENA_TINY or\
            self == Objective.ARENA_TINY_2V2:
            return True
        else:
            raise Exception(f'Objective.vs not implemented for {self}')


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)
