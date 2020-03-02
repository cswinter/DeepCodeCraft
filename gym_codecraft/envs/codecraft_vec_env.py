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
    global_drones: int = 0
    relative_positions: bool = True
    feat_last_seen: bool = False
    feat_map_size: bool = False
    v2: bool = False

    def global_features(self):
        if self.feat_map_size:
            return 4
        else:
            return 2

    def dstride(self):
        if self.feat_last_seen:
            return 17
        else:
            return 15

    def mstride(self):
        return 3

    def nonobs_features(self):
        return 3

    def enemies(self):
        return self.drones - self.allies

    def total_drones(self):
        return 2 * self.drones - self.allies

    def stride(self):
        return self.global_features() + self.total_drones() * self.dstride() + self.minerals * self.mstride()

    def endglobals(self):
        return self.global_features()

    def endallies(self):
        return self.global_features() + self.dstride() * self.allies

    def endenemies(self):
        return self.global_features() + self.dstride() * self.drones

    def endmins(self):
        return self.endenemies() + self.mstride() * self.minerals

    def endallenemies(self):
        return self.endmins() + self.dstride() * self.enemies()


GLOBAL_FEATURES_V2 = 2
DSTRIDE_V2 = 15
MSTRIDE_V2 = 3
NONOBS_FEATURES_V2 = 3
DEFAULT_OBS_CONFIG = ObsConfig(allies=2, drones=4, minerals=2, global_drones=4)


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


def map_arena_tiny(randomize: bool, hardness: int):
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
        'minerals': [],
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


def map_arena_tiny_2v2(randomize: bool, hardness: int):
    s1 = 1
    s2 = 1
    if randomize:
        s1 = np.random.randint(0, 2)
        s2 = np.random.randint(0, 2)
    return {
        'mapWidth': 1000,
        'mapHeight': 1000,
        'minerals': [],
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


def map_arena_medium(randomize: bool, hardness: int):
    if randomize:
        hardness = np.random.randint(0, hardness+1)
    if hardness == 0:
        map_width = 1500
        map_height = 1500
        mineral_count = 5
    else:
        map_width = 2000
        map_height = 2000
        mineral_count = 8

    angle = 2 * np.pi * np.random.rand()
    spawn_x = (map_width // 2 - 100) * np.sin(angle)
    spawn_y = (map_height // 2 - 100) * np.cos(angle)
    return {
        'mapWidth': map_width,
        'mapHeight': map_height,
        'minerals': mineral_count * [(2, 25)],
        'player1Drones': [
            drone_dict(spawn_x,
                       spawn_y,
                       constructors=2,
                       storage_modules=2),
        ],
        'player2Drones': [
            drone_dict(-spawn_x,
                       -spawn_y,
                       constructors=2,
                       storage_modules=2),
        ],
    }


def map_arena(randomize: bool, hardness: int):
    if randomize:
        hardness = np.random.randint(0, hardness+1)
    if hardness == 0:
        map_width = 1500
        map_height = 1500
        mineral_count = 2
    elif hardness == 1:
        map_width = 2000
        map_height = 2000
        mineral_count = 4
    elif hardness == 2:
        map_width = 2500
        map_height = 2500
        mineral_count = 6
    else:
        map_width = 3500
        map_height = 2500
        mineral_count = 6

    angle = 2 * np.pi * np.random.rand()
    spawn_x = (map_width // 2 - 100) * np.sin(angle)
    spawn_y = (map_height // 2 - 100) * np.cos(angle)
    return {
        'mapWidth': map_width,
        'mapHeight': map_height,
        'minerals': mineral_count * [(1, 50)],
        'player1Drones': [
            drone_dict(spawn_x,
                       spawn_y,
                       constructors=1,
                       storage_modules=2,
                       missile_batteries=1)
        ],
        'player2Drones': [
            drone_dict(-spawn_x,
                       -spawn_y,
                       constructors=1,
                       storage_modules=2,
                       missile_batteries=1)
        ],
    }


def map_standard(randomize: bool, hardness: int):
    if randomize:
        hardness = np.random.randint(0, hardness+1)
    minerals = None
    if hardness == 0:
        map_width = 1000
        map_height = 1000
        mineral_count = 2
    elif hardness == 1:
        map_width = 2000
        map_height = 1500
        mineral_count = 3
    elif hardness == 2:
        map_width = 3000
        map_height = 2000
        mineral_count = 6
    elif hardness == 3:
        map_width = 4000
        map_height = 2500
        mineral_count = 8
    elif hardness == 4:
        map_width = 5000
        map_height = 3000
        mineral_count = 10
    else:
        map_width = 6000
        map_height = 4000
        minerals = [
            (10, 10),
            (10, 10),
            (7, 20),
            (7, 20),
            (5, 30),
            (5, 30),
            (5, 50),
            (5, 70),
            (5, 100),
        ]
    if minerals is None:
        minerals = mineral_count * [(1, 50)]

    angle = 2 * np.pi * np.random.rand()
    spawn_x = (map_width // 2.2) * np.sin(angle)
    spawn_y = (map_height // 2.2) * np.cos(angle)
    if randomize:
        d = np.random.randint(0, 7)
        res = 3 * np.random.randint(0, 3)
    else:
        d = 6
        res = 7
    if d == 0:
        drone = dict(
            constructors=1,
            storage_modules=1,
            resources=res)
    elif d == 1:
        drone = dict(
            constructors=2,
            storage_modules=2,
            resources=2*res)
    elif d == 2:
        drone = dict(
            constructors=1,
            storage_modules=2,
            missile_batteries=1,
            resources=2*res)
    elif d == 3:
        drone = dict(
            constructors=2,
            storage_modules=3,
            missile_batteries=1,
            engines=1,
            resources=2*res)
    elif d == 4:
        drone = dict(
            constructors=2,
            storage_modules=3,
            engines=2,
            resources=2*res)
    elif d == 5:
        drone = dict(
            constructors=2,
            storage_modules=3,
            missile_batteries=3,
            shield_generators=1,
            engines=1,
            resources=2*res)
    else:
        drone = dict(
            constructors=3,
            storage_modules=3,
            missile_batteries=3,
            shield_generators=1,
            resources=2*res)

    return {
        'mapWidth': map_width,
        'mapHeight': map_height,
        'minerals': minerals,
        'player1Drones': [drone_dict(spawn_x, spawn_y, **drone)],
        'player2Drones': [drone_dict(-spawn_x, -spawn_y, **drone)],
    }


def map_mp(randomize: bool, hardness: int):
    map_width = np.random.randint(2, 7) * 500
    map_height = np.random.randint(2, 7) * 500
    player1_drones = []
    player2_drones = []

    def randpos():
        return np.random.randint(-map_width//3, map_width//3), np.random.randint(-map_height//3, map_height//3)
    scenario = np.random.randint(0, 4)
    if scenario == 0:
        drone_count = np.random.randint(2, 11)
        for _ in range(drone_count):
            x1, y1 = randpos()
            player1_drones.append(drone_dict(x1, y1, missile_batteries=1))
            x2, y2 = randpos()
            player2_drones.append(drone_dict(x2, y2, missile_batteries=1))
    elif scenario == 1:
        p1_drone_count = np.random.randint(0, 2)
        p2_drone_count = np.random.randint(5, 11)
        xm, ym = randpos()
        player1_drones.append(drone_dict(xm, ym, constructors=3, missile_batteries=3, storage_modules=3, shield_generators=1))
        if np.random.randint(0, 3) == 0:
            engines = np.random.randint(0, 2)
            x, y = randpos()
            player2_drones.append(drone_dict(x, y, missile_batteries=2, shield_generators=2-engines, engines=engines))
            p2_drone_count -= 3
        nearby_count = np.random.randint(0, p2_drone_count+1)
        for i in range(p1_drone_count):
            x, y = randpos()
            player1_drones.append(drone_dict(x, y, missile_batteries=1))
        for i in range(p2_drone_count):
            if i < nearby_count:
                x, y = randpos()
            else:
                x = int(np.clip(xm + np.random.randint(-350, 350), -map_width//2, map_width//2))
                y = int(np.clip(ym + np.random.randint(-350, 350), -map_height//2, map_height//2))
            player2_drones.append(drone_dict(x, y, missile_batteries=1))
    elif scenario == 2:
        p1_drone_count = np.random.randint(0, 3)
        p2_drone_count = np.random.randint(3, 7)
        nearby_count = np.random.randint(0, p2_drone_count+1)
        xm, ym = randpos()
        engines = np.random.randint(0, 2)
        player1_drones.append(drone_dict(xm, ym, missile_batteries=2, shield_generators=2-engines, engines=engines))
        for i in range(p1_drone_count):
            x, y = randpos()
            player1_drones.append(drone_dict(x, y, missile_batteries=1))
        for i in range(p2_drone_count):
            if i < nearby_count:
                x, y = randpos()
            else:
                x = int(np.clip(xm + np.random.randint(-350, 350), -map_width//2, map_width//2))
                y = int(np.clip(ym + np.random.randint(-350, 350), -map_height//2, map_height//2))
            player2_drones.append(drone_dict(x, y, missile_batteries=1))
    elif scenario == 3:
        total = np.random.randint(4, 12)
        p1_large = np.random.randint(0, total//2)
        p2_large = np.random.randint(0, total//2)
        for i in range(total - 2 * p1_large):
            x, y = randpos()
            player1_drones.append(drone_dict(x, y, missile_batteries=1))
        for i in range(total - 2 * p2_large):
            x, y = randpos()
            player2_drones.append(drone_dict(x, y, missile_batteries=1))
        for i in range(p1_large):
            x, y = randpos()
            shields = np.random.randint(0, 2)
            player1_drones.append(drone_dict(x, y, missile_batteries=2-shields, shield_generators=shields))
        for i in range(p2_large):
            x, y = randpos()
            shields = np.random.randint(0, 2)
            player2_drones.append(drone_dict(x, y, missile_batteries=2-shields, shield_generators=shields))
    return {
        'mapWidth': map_width,
        'mapHeight': map_height,
        'minerals': 2 * [(1, 50)],
        'player1Drones': player1_drones,
        'player2Drones': player2_drones,
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
                 obs_config=DEFAULT_OBS_CONFIG,
                 hardness=0,
                 symmetric=False,
                 strong_scripted_opponent=False):
        assert(num_envs >= 2 * num_self_play)
        self.num_envs = num_envs
        self.objective = objective
        self.action_delay = action_delay
        self.num_self_play = num_self_play
        self.stagger = stagger
        self.fair = fair
        self.game_length = 3 * 60 * 60
        self.custom_map = lambda _1, _2: None
        self.last_map = None
        self.randomize = randomize
        self.use_action_masks = use_action_masks
        self.obs_config = obs_config
        self.hardness = hardness
        self.symmetric = symmetric
        self.builds = []
        self.strong_scripted_opponent = strong_scripted_opponent
        if objective == Objective.ARENA_TINY:
            self.game_length = 1 * 60 * 60
            self.custom_map = map_arena_tiny
        elif objective == Objective.ARENA_TINY_2V2:
            self.game_length = 1 * 30 * 60
            self.custom_map = map_arena_tiny_2v2
        elif objective == Objective.ARENA_MEDIUM:
            self.game_length = 3 * 60 * 60
            self.custom_map = map_arena_medium
        elif objective == Objective.ARENA:
            self.game_length = 3 * 60 * 60
            # [storageModules, missileBatteries, constructors, engines, shieldGenerators]
            self.builds = [
                [1, 0, 1, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 1, 0, 0, 1]
            ]
            self.custom_map = map_arena
        elif objective == Objective.STANDARD:
            self.game_length = 5 * 60 * 60
            self.builds = [
                [1, 0, 1, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 3, 0, 0, 1],
                [0, 2, 0, 0, 2],
                [2, 1, 1, 0, 0],
                [2, 0, 2, 0, 0],
                [2, 0, 1, 1, 0],
                [0, 2, 0, 1, 1],
                [1, 0, 0, 0, 0],
            ]
            self.custom_map = map_standard
        elif objective == Objective.MICRO_PRACTICE:
            self.game_length = 20 * 60
            self.custom_map = map_mp
        self.build_costs = [sum(modules) for modules in self.builds]
        self.naction = 8 + len(self.builds)

        self.games = []
        self.eplen = []
        self.eprew = []
        self.score = []

    def reset(self, partitioned_obs_config=None):
        if partitioned_obs_config:
            return list(self._reset(partitioned_obs_config))
        else:
            return next(self._reset())

    def _reset(self, partitioned_obs_config=None):
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
                self.next_map(),
                self.strong_scripted_opponent)
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

        if partitioned_obs_config:
            for envs, obs_config in partitioned_obs_config:
                obs, _, _, _, action_masks, privileged_obs = self.observe(envs, obs_config)
                yield obs, action_masks, privileged_obs
        else:
            obs, _, _, _, action_masks, privileged_obs = self.observe()
            yield obs, action_masks, privileged_obs

    def step(self, actions, env_subset=None, obs_config=None):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions, env_subset)
        return self.observe(env_subset, obs_config)

    def step_async(self, actions, env_subset=None):
        game_actions = []
        games = [self.games[env] for env in env_subset] if env_subset else self.games
        for ((game_id, player_id), player_actions) in zip(games, actions):
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
                if action >= 8:
                    build = [self.builds[action - 8]]
                player_actions2.append((move, turn, build, harvest))
            game_actions.append((game_id, player_id, player_actions2))

        codecraft.act_batch(game_actions, disable_harvest=self.objective == Objective.DISTANCE_TO_CRYSTAL)

    def observe(self, env_subset=None, obs_config=None):
        obs_config = obs_config or self.obs_config
        games = [self.games[env] for env in env_subset] if env_subset else self.games
        num_envs = len(games)

        rews = []
        dones = []
        infos = []
        obs = codecraft.observe_batch_raw(games,
                                          allies=obs_config.allies,
                                          drones=obs_config.drones,
                                          minerals=obs_config.minerals,
                                          global_drones=obs_config.global_drones,
                                          relative_positions=obs_config.relative_positions,
                                          v2=True,
                                          extra_build_costs=self.build_costs,
                                          map_size=obs_config.feat_map_size,
                                          last_seen=obs_config.feat_last_seen)
        stride = obs_config.stride()
        for i in range(num_envs):
            game = env_subset[i] if env_subset else i
            if self.objective.vs():
                allied_score = obs[stride * num_envs + i * obs_config.nonobs_features() + 1]
                enemy_score = obs[stride * num_envs + i * obs_config.nonobs_features() + 2]
                score = 2 * allied_score / (allied_score + enemy_score + 1e-8) - 1
            elif self.objective == Objective.ALLIED_WEALTH:
                score = obs[stride * num_envs + i * obs_config.nonobs_features() + 1] * 0.1
            elif self.objective in [Objective.DISTANCE_TO_CRYSTAL, Objective.DISTANCE_TO_1000_500, Objective.DISTANCE_TO_ORIGIN]:
                raise Exception(f"Deprecated objective {self.objective}")
            else:
                raise Exception(f"Unknown objective {self.objective}")

            if self.score[game] is None:
                self.score[game] = score
            reward = score - self.score[game]
            self.score[game] = score
            self.eprew[game] += reward

            winner = obs[stride * num_envs + i * obs_config.nonobs_features()]
            if winner > 0:
                (game_id, pid) = games[i]
                if pid == 0:
                    self_play = game // 2 < self.num_self_play
                    game_id = codecraft.create_game(self.game_length,
                                                    self.action_delay,
                                                    self_play,
                                                    self.next_map(),
                                                    self.strong_scripted_opponent)
                else:
                    game_id = self.games[game - 1][0]
                # print(f"COMPLETED {i} {game} {games[i]} == {self.games[game]} new={game_id}")
                self.games[game] = (game_id, pid)
                observation = codecraft.observe(game_id, pid)
                # TODO: use actual observation
                if not obs.flags['WRITEABLE']:
                    obs = obs.copy()
                obs[stride * i:stride * (i + 1)] = 0.0  # codecraft.observation_to_np(observation)

                dones.append(1.0)
                infos.append({'episode': {
                    'r': self.eprew[game],
                    'l': self.eplen[game],
                    'index': game,
                    'score': self.score[game],
                }})
                self.eplen[game] = 1
                self.eprew[game] = 0
                self.score[game] = None
            else:
                self.eplen[game] += 1
                dones.append(0.0)

            rews.append(reward)

        action_mask_elems = self.naction * obs_config.allies * num_envs
        action_masks = obs[-action_mask_elems:].reshape(-1, obs_config.allies, self.naction)

        # TODO: merged with other obs, remove completely
        privileged_obs = np.zeros([num_envs, 1])

        return obs[:stride * num_envs].reshape(num_envs, -1), \
               np.array(rews), \
               np.array(dones), \
               infos, \
               action_masks if self.use_action_masks \
                   else np.ones([num_envs, obs_config.allies, self.naction], dtype=np.float32), \
               privileged_obs

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
            map = self.fair_map()
        else:
            map = self.custom_map(self.randomize, self.hardness)
        if map:
            map['symmetric'] = self.symmetric
        return map

    def fair_map(self):
        if self.last_map is None:
            self.last_map = self.custom_map(self.randomize, self.hardness)
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
    ARENA_MEDIUM = 'ARENA_MEDIUM'
    ARENA = 'ARENA'
    STANDARD = 'STANDARD'
    MICRO_PRACTICE = 'MICRO_PRACTICE'

    def vs(self):
        if self == Objective.ALLIED_WEALTH or\
           self == Objective.DISTANCE_TO_CRYSTAL or\
           self == Objective.DISTANCE_TO_ORIGIN or\
           self == Objective.DISTANCE_TO_1000_500:
           return False
        elif self == Objective.ARENA_TINY or\
            self == Objective.ARENA_TINY_2V2 or\
            self == Objective.ARENA_MEDIUM or\
            self == Objective.ARENA or\
            self == Objective.STANDARD or\
            self == Objective.MICRO_PRACTICE:
            return True
        else:
            raise Exception(f'Objective.vs not implemented for {self}')

    def naction(self):
        if self == Objective.STANDARD:
            return 18
        elif self == Objective.ARENA:
            return 11
        else:
            return 8


def dist2(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + dy * dy


def dist(x1, y1, x2, y2):
    dx = x1 - x2
    dy = y1 - y2
    return np.sqrt(dx * dx + dy * dy)
