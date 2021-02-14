import argparse
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'Adam'  # Optimizer ("SGD" or "RMSProp" or "Adam")
        self.lr = 0.0003            # Learning rate
        self.final_lr = 0.0001      # Learning rate floor when using cosine schedule
        self.lr_schedule = 'none'   # Learning rate schedule ("none" or "cosine")
        self.momentum = 0.9         # Momentum
        self.weight_decay = 0.0001
        self.bs = 2048              # Batch size during optimization
        self.batches_per_update = 1 # Accumulate gradients over this many batches before applying gradients
        self.batches_per_update_schedule = ''
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 1.0          # Weighting of value function loss in optimization objective
        self.entropy_bonus = 0.0    # Weighting of  entropy bonus in loss function
        self.entropy_bonus_schedule = ''
        self.max_grad_norm = 20.0   # Maximum gradient norm for gradient clipping
        self.epochs = 2             # Number of optimizer passes over samples collected during rollout
        self.lr_ratios = 1.0        # Learning rate multiplier applied to earlier layers
        self.warmup = 0             # Learning rate is increased linearly from 0 during first n samples

        # Policy
        self.d_agent = 256
        self.d_item = 128
        self.dff_ratio = 2
        self.nhead = 8
        self.item_item_attn_layers = 0
        self.dropout = 0.0             # Try 0.1?
        self.nearby_map = True         # Construct map of nearby objects populated with scatter connections
        self.nm_ring_width = 60        # Width of circles on nearby map
        self.nm_nrays = 8              # Number of rays on nearby map
        self.nm_nrings = 8             # Number of rings on nearby map
        self.map_conv = False          # Whether to perform convolution on nearby map
        self.mc_kernel_size = 3        # Size of convolution kernel for nearby map
        self.map_embed_offset = False  # Whether the nearby map has 2 channels corresponding to the offset of objects within the tile
        self.item_ff = True            # Adds itemwise ff resblock after initial embedding before transformer
        self.agents = 1                # Max number of simultaneously controllable drones
        self.nally = 1                 # Max number of allies observed by each drone
        self.nenemy = 0                # Max number of enemies observed by each drone
        self.nmineral = 10             # Max number of minerals observed by each drone
        self.ntile = 0                 # Number of map tiles observed by each drone
        self.nconstant = 0             # Number learnable constant valued items observed by each drone
        self.ally_enemy_same = False   # Use same weights for processing ally and enemy drones
        self.norm = 'layernorm'        # Normalization layers ("none", "batchnorm", "layernorm")
        self.fp16 = False              # Whether to use half-precision floating point
        self.zero_init_vf = True       # Set all initial weights for value function head to zero
        self.small_init_pi = False     # Set initial weights for policy head to small values and biases to zero

        self.resume_from = ''       # Filepath to saved policy

        # Data parallel
        self.rank = 0
        self.parallelism = 1           # Number of data parallel processes. Must be set explicitly when using schedule.py, otherwise runner.py will just spawn a single process.

        # Observations
        self.obs_allies = 10            # Max number of allied drones returned by the env
        self.obs_enemies = 10           # Max number of enemy drones returned by the env
        self.obs_minerals = 10          # Max number of minerals returned by the env
        self.obs_map_tiles = 10         # Max number of map tiles returned by the env
        self.obs_keep_abspos = False    # Have features for both absolute and relative positions on each object
        self.use_privileged = True      # Whether value function has access to hidden information
        self.feat_map_size = True       # Global features for width/height of map
        self.feat_last_seen = False     # Remember last position/time each enemy was seen + missile cooldown feat
        self.feat_is_visible = True     # Feature for whether drone is currently visible
        self.feat_abstime = True        # Global features for absolute remaining/elapsed number of timesteps
        self.feat_mineral_claims = False  # Feature for whether another drone is currently harvesting a mineral
        self.harvest_action = False     # Harvest action that will freeze drone until one resource has been harvested
        self.lock_build_action = False  # Pair of actions to disable/enable all build actions
        self.feat_dist_to_wall = False  # Five features giving distance to closest wall in movement direction, and in movement direction offset by +-pi/2 and +-pi/4

        # Eval
        self.eval_envs = 256
        self.eval_timesteps = 360
        self.eval_frequency = 1e5
        self.model_save_frequency = 10
        self.eval_symmetric = True

        self.extra_checkpoint_steps = []

        # RL
        self.steps = 10e6           # Total number of timesteps
        self.num_envs = 64          # Number of environments
        self.num_self_play = 32     # Number of self-play environments (each provides two environments)
        self.num_vs_replicator = 0  # Number of environments played vs scripted replicator AI
        self.num_vs_aggro_replicator = 0  # Number of environments played vs scripted aggressive replicator AI
        self.num_vs_destroyer = 0   # Number of environments played vs scripted destroyer AI
        self.num_self_play_schedule = ''
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.gamma = 0.99           # Discount factor
        self.gamma_schedule = ''
        self.lamb = 0.95            # Generalized advantage estimation parameter lambda
        self.norm_advs = True       # Normalize advantage values
        self.rewscale = 1.0         # Scaling of reward values
        self.ppo = True             # Use PPO-clip instead of vanilla policy gradients objective
        self.cliprange = 0.2        # PPO cliprange
        self.clip_vf = True         # Use clipped value function objective
        self.split_reward = False   # Split reward evenly amongst all active agents.
        self.liveness_penalty = 0.0 # Negative reward applied at each timestep
        self.build_variety_bonus = 0.0  # Extra reward for building a drone type at least once during episode
        self.win_bonus = 0.0        # Reward received when winning game by eliminating opponent
        self.loss_penalty = 0.0     # Negative reward received when losing game by being eliminated
        self.partial_score = 1.0    # Instantaneous reward received from change in relative amount of resources under allied control
        self.attac = 0.0            # Fraction of shaped reward awarded for minimum health of enemy mothership during episode
        self.protec = 0.0           # Fraction of shaped reward awarded for maximum health of allied mothership during episode
        self.rewnorm = False        # Rescale reward values by ema of mean and variance
        self.rewnorm_emaw = 0.97
        self.max_army_size_score = 9999999
        self.max_enemy_army_size_score = 9999999

        # Task/Curriculum
        self.objective = envs.Objective.ARENA_TINY_2V2
        self.action_delay = 0
        self.use_action_masks = True
        self.task_hardness = 0
        self.max_game_length = 0       # Max length of games, or default game length for map if 0.
        self.max_hardness = 150        # Maxiumum map area
        self.hardness_offset = 1e6     # Number of timesteps steps after which hardness starts to increase
        self.task_randomize = True
        self.symmetric_map = 0.0       # Percentage of maps which are symmetric
        self.symmetry_increase = 2e-8  # Linearly increase env symmetry parameter with this slope for every step
        self.mix_mp = 0.0              # Fraction of maps that use MICRO_PRACTICE instead of the main objective
        self.rule_rng_fraction = 0.0   # Fraction of maps that use randomize ruleset
        self.rule_rng_amount = 1.0     # Amount of rule randomization
        self.rule_cost_rng = 0.0
        self.adr = False               # Automatically adjust environment rules
        self.adr_hstepsize = 2.0e-6    # Amount by which task difficulty/map size is increased for each processed frame
        self.linear_hardness = True    # Linearly increase task difficulty/map size
        self.mothership_damage_scale = 4.0
        self.mothership_damage_scale_schedule = 'lin 50e6:1.0,150e6:0.0'
        self.adr_average_cost_target = 1.0  # Target value for average module cost
        self.adr_avg_cost_schedule = ''
        self.adr_cost_variance = 0.5
        self.adr_cost_variance_schedule = 'lin 0:0.5,140e6:0.1'

        self.adr_variety = 0.8
        self.adr_variety_schedule = '60e6:0.5,120e6:0.4,140e6:0.3'

        # Testing
        self.verify_create_golden = False
        self.verify = False


    @staticmethod
    def micro_practice():
        hps = HyperParams()
        hps.objective = envs.Objective.MICRO_PRACTICE

        hps.steps = 40e6

        hps.agents = 8
        hps.nenemy = 7
        hps.nally = 7
        hps.nmineral = 5

        hps.batches_per_update = 2
        hps.bs = 1024
        hps.seq_rosteps = 256
        hps.num_envs = 64
        hps.num_self_play = 32

        hps.eval_envs = 256
        hps.eval_frequency = 1e6
        hps.eval_timesteps = 500

        hps.gamma = 0.997
        hps.entropy_bonus = 0.001

        hps.symmetric_map = 0.0
        hps.eval_symmetric = False

        return hps


    @staticmethod
    def standard():
        hps = HyperParams()
        hps.objective = envs.Objective.STANDARD

        hps.steps = 150e6

        hps.agents = 15
        hps.nenemy = 15
        hps.nally = 15
        hps.nmineral = 5
        hps.ntile = 5

        hps.obs_minerals = 5
        hps.obs_allies = 15
        hps.obs_map_tiles = 5
        hps.obs_enemies = 15
        hps.feat_last_seen = True
        hps.feat_mineral_claims = True
        hps.harvest_action = True
        hps.feat_dist_to_wall = True

        hps.lr = 0.0005
        hps.final_lr = 0.00005
        hps.lr_schedule = 'cosine'
        hps.win_bonus = 2.0
        hps.partial_score = 1.0
        hps.vf_coef = 1.0
        hps.rule_rng_fraction = 1.0
        hps.rule_rng_amount = 1.0
        hps.adr = True
        hps.gamma = 0.999
        hps.entropy_bonus = 0.2
        hps.entropy_bonus_schedule = 'lin 15e6:0.1,60e6:0.0'
        hps.mothership_damage_scale = 0.0
        hps.mothership_damage_scale_schedule = ''
        hps.adr_hstepsize = 3.0e-6
        hps.adr_variety = 0.4
        hps.adr_variety_schedule = '60e6:0.3,120e6:0.2,140e6:0.1'


        hps.batches_per_update = 32
        hps.bs = 512
        hps.seq_rosteps = 128
        hps.num_envs = 128
        hps.num_self_play = 64

        hps.model_save_frequency = 1
        hps.eval_envs = 128
        hps.eval_frequency = 5e6
        hps.eval_timesteps = 5000

        hps.extra_checkpoint_steps = [1e6, 2.5e6]

        return hps

    @staticmethod
    def enhanced():
        hps = HyperParams.standard()
        hps.max_hardness = 200
        hps.objective = envs.Objective.ENHANCED
        return hps

    # Equivalent to `standard` config when run dataparallel across 2 processes.
    @staticmethod
    def standard_2dataparallel():
        hps = HyperParams.standard()
        hps.batches_per_update //= 2
        hps.num_envs //= 2
        hps.num_self_play //= 2
        return hps

    @staticmethod
    def enhanced_2dataparallel():
        hps = HyperParams.standard_2dataparallel()
        hps.max_hardness = 200
        hps.objective = envs.Objective.ENHANCED
        return hps


    @staticmethod
    def standard_dataparallel():
        hps = HyperParams.standard()

        hps.steps = 300e6

        hps.batches_per_update = 16
        hps.num_envs = 64
        hps.num_self_play = 32

        hps.entropy_bonus_schedule = 'lin 30e6:0.1,120e6:0.0'
        hps.mothership_damage_scale_schedule = 'lin 100e6:0.0'
        hps.hardness_offset *= 2.0
        hps.adr_hstepsize *= 0.5
        hps.mothership_damage_scale_schedule = 'lin 100e6:1.0,300e6:0.0'
        hps.adr_variety_schedule = '120e6:0.5,240e6:0.4,280e6:0.3'

        return hps

    @staticmethod
    def arena():
        hps = HyperParams()
        hps.objective = envs.Objective.ARENA

        hps.steps = 25e6

        hps.agents = 6
        hps.nenemy = 5
        hps.nally = 5
        hps.nmineral = 5

        hps.batches_per_update = 2
        hps.bs = 1024
        hps.seq_rosteps = 256
        hps.num_envs = 64
        hps.num_self_play = 32

        hps.eval_envs = 256
        hps.eval_frequency = 5e5
        hps.eval_timesteps = 1100

        hps.gamma = 0.997
        hps.entropy_bonus = 0.001

        hps.symmetric_map = 1.0
        hps.task_hardness = 4

        return hps


    @staticmethod
    def arena_medium():
        hps = HyperParams()
        hps.objective = envs.Objective.ARENA_MEDIUM

        hps.steps = 50e6

        hps.agents = 4
        hps.nenemy = 5
        hps.nally = 5
        hps.nmineral = 5

        hps.batches_per_update = 1
        hps.batches_per_update_schedule = '15e6:2,30e6:4'
        hps.bs = 1024
        hps.seq_rosteps = 256
        hps.num_envs = 64
        hps.num_self_play = 32

        hps.model_save_frequency = 1
        hps.eval_envs = 512
        hps.eval_frequency = 5e6
        hps.eval_timesteps = 2000

        hps.gamma = 0.997
        hps.entropy_bonus = 0.002
        hps.entropy_bonus_schedule = '15e6:0.0005,30e6:0.0'

        hps.symmetric_map = 1.0
        hps.task_hardness = 0

        return hps

    @staticmethod
    def arena_medium_large_ms():
        hps = HyperParams.arena_medium()
        hps.objective = envs.Objective.ARENA_MEDIUM_LARGE_MS
        hps.task_hardness = 1
        hps.win_bonus = 2
        hps.vf_coef = 0.5
        hps.rule_rng_fraction = 1.0
        hps.rule_rng_amount = 1.0
        hps.agents = 7
        hps.gamma = 0.999
        hps.eval_envs = 256
        hps.nenemy = 7
        hps.nally = 7
        hps.obs_allies = 15
        hps.obs_enemies = 15
        hps.batches_per_update_schedule = '20e6:2,35e6:4,45e6:8'
        hps.entropy_bonus = 0.01
        hps.entropy_bonus_schedule = '15e6:0.003,40e6:0.001'
        return hps

    @staticmethod
    def arena_tiny_2v2():
        hps = HyperParams()
        hps.objective = envs.Objective.ARENA_TINY_2V2

        hps.steps = 25e6

        hps.entropy_bonus = 0.001

        hps.agents = 2
        hps.nally = 2
        hps.nenemy = 2
        hps.nmineral = 1
        hps.obs_allies = 2
        hps.obs_enemies = 2
        hps.obs_minerals = 1  # Could be 0, currently incompatible with ally_enemy_same=False

        hps.eval_envs = 256
        hps.eval_timesteps = 360
        hps.eval_frequency = 1e5
        hps.eval_symmetric = False

        return hps

    @staticmethod
    def arena_tiny():
        hps = HyperParams()
        hps.objective = envs.Objective.ARENA_TINY

        hps.steps = 2e6

        hps.d_agent = 128
        hps.d_item = 64

        hps.agents = 1
        hps.nally = 1
        hps.nenemy = 1
        hps.nmineral = 1
        hps.obs_allies = 1
        hps.obs_enemies = 1
        hps.obs_minerals = 1  # Could be 0, currently incompatible with ally_enemy_same=False

        hps.eval_envs = 256
        hps.eval_frequency = 1e5
        hps.eval_timesteps = 360

        hps.num_envs = 64
        hps.num_self_play = 32
        hps.seq_rosteps = 256
        hps.eval_symmetric = False

        return hps


    @staticmethod
    def scout():
        hps = HyperParams()
        hps.objective = envs.Objective.SCOUT

        hps.steps = 1e6

        hps.agents = 5
        hps.nenemy = 5
        hps.nally = 5
        hps.nmineral = 0
        hps.ntile = 5
        hps.obs_map_tiles = 10
        hps.use_privileged = False

        hps.batches_per_update = 1
        hps.bs = 256
        hps.seq_rosteps = 64
        hps.num_envs = 64
        hps.num_self_play = 0

        hps.eval_envs = 0

        hps.gamma = 0.99

        return hps


    @staticmethod
    def allied_wealth():
        hps = HyperParams()
        hps.clip_vf = True
        hps.steps = 1.5e6
        hps.dff_ratio = 2
        hps.eval_envs = 0
        hps.gamma = 0.99
        hps.lamb = 0.95
        hps.lr = 0.0003
        hps.max_grad_norm = 20.0
        hps.momentum = 0.9
        hps.norm = 'layernorm'
        hps.norm_advs = True
        hps.num_envs = 64
        hps.num_self_play = 0
        hps.objective = envs.Objective.ALLIED_WEALTH
        hps.nally = 1
        hps.nmineral = 10
        hps.obs_allies = 1
        hps.obs_map_tiles = 0
        hps.obs_enemies = 0
        hps.obs_global_drones = 0
        hps.optimizer = 'Adam'
        hps.epochs = 2
        hps.small_init_pi = False
        hps.transformer_layers = 1
        hps.use_action_masks = True
        hps.use_privileged = False
        hps.vf_coef = 1.0
        hps.weight_decay = 0.0001
        hps.zero_init_vf = True

        return hps


    @staticmethod
    def distance_to_origin():
        hps = HyperParams()
        hps.objective = envs.Objective.DISTANCE_TO_ORIGIN
        hps.num_self_play = 0
        hps.eval_envs = 0
        hps.agents = 1
        hps.obs_allies = 1
        hps.obs_enemies = 0
        hps.use_privileged = False

        return hps


    @staticmethod
    def distance_to_mineral():
        hps = HyperParams()
        hps.objective = envs.Objective.DISTANCE_TO_CRYSTAL
        hps.num_self_play = 0
        hps.eval_envs = 0
        hps.agents = 1
        hps.obs_allies = 1
        hps.obs_enemies = 0
        hps.use_privileged = False

        return hps


    @property
    def rosteps(self):
        return self.num_envs * self.seq_rosteps

    def get_num_self_play_schedule(self):
        return parse_int_schedule(self.num_self_play_schedule)

    def get_entropy_bonus_schedule(self):
        return parse_float_schedule(self.entropy_bonus_schedule)

    def get_batches_per_update_schedule(self):
        return parse_int_schedule(self.batches_per_update_schedule)

    def get_variety_schedule(self) -> List[Tuple[float, float]]:
        return parse_float_schedule(self.adr_variety_schedule)

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_const', const=False, dest=name)
                parser.add_argument(f"--{name}", action='store_const', const=True, dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


class HPSchedule(ABC):
    @abstractmethod
    def value_at(self, step: int) -> float:
        pass


class LinearHPSchedule(HPSchedule):
    def __init__(self, segments: List[Tuple[int, float]]):
        self.segments = segments

    def value_at(self, step: int) -> float:
        left, right = find_adjacent(self.segments, step)
        if right is None:
            return left[1]
        return left[1] + (step - left[0]) * (right[1] - left[1]) / (right[0] - left[0])


class CosineSchedule(HPSchedule):
    def __init__(self, initial_value: float, final_value: float, steps: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.steps = steps

    def value_at(self, step: int) -> float:
        return (self.initial_value - self.final_value) * 0.5 * (math.cos(math.pi * step / self.steps) + 1) \
               + self.final_value


class StepHPSchedule(HPSchedule):
    def __init__(self, segments: List[Tuple[int, float]]):
        self.segments = segments

    def value_at(self, step: int) -> float:
        left, _ = find_adjacent(self.segments, step)
        return left[1]


class ConstantSchedule(HPSchedule):
    def __init__(self, value):
        self.value = value

    def value_at(self, step) -> float:
        return self.value


def parse_schedule(schedule: str, initial_value: float, steps: int) -> HPSchedule:
    if schedule == '':
        return ConstantSchedule(initial_value)
    elif schedule.startswith('lin '):
        segments = [(0, initial_value)]
        for kv in schedule[len('lin '):].split(","):
            [k, v] = kv.split(":")
            segments.append((int(float(k)), float(v)))
        return LinearHPSchedule(segments)
    elif schedule.startswith('cos'):
        if schedule == 'cos':
            final_value = 0.0
        else:
            final_value = float(schedule[len('cos '):])
        return CosineSchedule(initial_value, final_value, steps)
    else:
        segments = [(0, initial_value)]
        for kv in schedule.split(","):
            [k, v] = kv.split(":")
            segments.append((int(float(k)), float(v)))
        return StepHPSchedule(segments)


def find_adjacent(segments: List[Tuple[int, float]], step: int) -> Tuple[Tuple[int, float], Optional[Tuple[int, float]]]:
    left = None
    right: Optional[Tuple[int, float]] = None
    for s, v in segments:
        if s <= step:
            left = (s, v)
        if step < s:
            right = (s, v)
            break
    assert left is not None, f"invalid inputs to find_adjacent: segments={segments}, step={step}"
    return left, right


def parse_int_schedule(schedule):
    if schedule == '':
        return []
    else:
        items = []
        for kv in schedule.split(","):
            [k, v] = kv.split(":")
            items.append((float(k), int(v)))
        return list(reversed(items))


def parse_float_schedule(schedule) -> List[Tuple[float, float]]:
    if schedule == '':
        return []
    else:
        items = []
        for kv in schedule.split(","):
            [k, v] = kv.split(":")
            items.append((float(k), float(v)))
        return list(reversed(items))
