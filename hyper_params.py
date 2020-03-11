import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'Adam'  # Optimizer ("SGD" or "RMSProp" or "Adam")
        self.lr = 0.0003            # Learning rate
        self.momentum = 0.9         # Momentum
        self.weight_decay = 0.0001
        self.bs = 2048              # Batch size during optimization
        self.batches_per_update = 1 # Accumulate gradients over this many batches before applying gradients
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 1.0          # Weighting of value function loss in optimization objective
        self.entropy_bonus = 0.0    # Weighting of  entropy bonus in loss function
        self.max_grad_norm = 20.0   # Maximum gradient norm for gradient clipping
        self.sample_reuse = 2       # Number of optimizer passes over samples collected during rollout
        self.lr_ratios = 1.0        # Learning rate multiplier applied to earlier layers
        self.warmup = 0             # Learning rate is increased linearly from 0 during first n samples

        # Policy (transformer)
        self.d_agent = 256
        self.d_item = 128
        self.dff_ratio = 2
        self.nhead = 8
        self.item_item_attn_layers = 0
        self.dropout = 0.0             # Try 0.1?
        self.nearby_map = True         # Construct map of nearby objects populated with scatter connections
        self.nm_ring_width = 30        # Width of circles on nearby map
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
        self.norm = 'layernorm'     # Normalization layers ("none", "batchnorm", "layernorm")
        self.fp16 = False           # Whether to use half-precision floating point
        self.zero_init_vf = True    # Set all initial weights for value function head to zero
        self.small_init_pi = False  # Set initial weights for policy head to small values and biases to zero

        self.resume_from = ''       # Filepath to saved policy

        # Observations
        self.obs_allies = 10          # Max number of allied drones returned by the env
        self.obs_enemies = 10         # Max number of enemy drones returned by the env
        self.obs_minerals = 10        # Max number of minerals returned by the env
        self.obs_map_tiles = 10       # Max number of map tiles returned by the env
        self.obs_keep_abspos = False  # Have features for both absolute and relative positions on each object
        self.use_privileged = True    # Whether value function has access to hidden information
        self.feat_map_size = True     # Global features for width/height of map
        self.feat_last_seen = True    # Remember last position/time each enemy was seen + missile cooldown feat
        self.feat_is_visible = True   # Feature for whether drone is currently visible
        self.feat_abstime = True      # Global features for absolute remaining/elapsed number of timesteps

        # Eval
        self.eval_envs = 256
        self.eval_timesteps = 360
        self.eval_frequency = 1e5
        self.model_save_frequency = 10
        self.eval_symmetric = True

        # RL
        self.steps = 10e6           # Total number of timesteps
        self.num_envs = 64          # Number of environments
        self.num_self_play = 32     # Number of self-play environments (each provides two environments)
        self.num_self_play_schedule = ''
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.gamma = 0.99           # Discount factor
        self.lamb = 0.95            # Generalized advantage estimation parameter lambda
        self.norm_advs = True       # Normalize advantage values
        self.rewscale = 1.0         # Scaling of reward values
        self.ppo = True             # Use PPO-clip instead of vanilla policy gradients objective
        self.cliprange = 0.2        # PPO cliprange
        self.clip_vf = True         # Use clipped value function objective
        self.split_reward = False   # Split reward evenly amongst all active agents
        self.liveness_penalty = 0.0 # Negative reward applied at each timestep

        # Task
        self.objective = envs.Objective.ARENA_TINY_2V2
        self.action_delay = 0
        self.use_action_masks = True
        self.task_hardness = 0
        self.task_randomize = True
        self.symmetric_map = False
        self.mix_mp = 0.0       # Fraction of maps that use MICRO_PRACTICE instead of the main objective


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

        hps.symmetric_map = False
        hps.eval_symmetric = False

        return hps


    @staticmethod
    def standard():
        hps = HyperParams()
        hps.objective = envs.Objective.STANDARD

        hps.steps = 40e6

        hps.agents = 10
        hps.nenemy = 10
        hps.nally = 10
        hps.nmineral = 5
        hps.ntile = 5

        hps.obs_allies = 15
        hps.obs_enemies = 15

        hps.batches_per_update = 1
        hps.bs = 1024
        hps.seq_rosteps = 256
        hps.num_envs = 64
        hps.num_self_play = 32

        hps.eval_envs = 256
        hps.eval_frequency = 5e6
        hps.eval_timesteps = 3000
        hps.model_save_frequency = 1

        hps.gamma = 0.997
        hps.entropy_bonus = 0.001

        hps.symmetric_map = True
        hps.task_hardness = 4

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

        hps.symmetric_map = True
        hps.task_hardness = 4

        return hps


    @staticmethod
    def arena_medium():
        hps = HyperParams()
        hps.objective = envs.Objective.ARENA_MEDIUM

        hps.steps = 25e6

        hps.agents = 4
        hps.nenemy = 5
        hps.nally = 5
        hps.nmineral = 5

        hps.batches_per_update = 2
        hps.bs = 1024
        hps.seq_rosteps = 128
        hps.num_envs = 128
        hps.num_self_play = 64

        hps.eval_envs = 512
        hps.eval_frequency = 5e6
        hps.eval_timesteps = 2000

        hps.gamma = 0.997
        hps.entropy_bonus = 0.001

        hps.symmetric_map = True
        hps.task_hardness = 0

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
        hps.obs_global_drones = 0
        hps.optimizer = 'Adam'
        hps.sample_reuse = 2
        hps.small_init_pi = False
        hps.transformer_layers = 1
        hps.use_action_masks = True
        hps.use_privileged = False
        hps.vf_coef = 1.0
        hps.weight_decay = 0.0001
        hps.zero_init_vf = True

        return hps

    @property
    def rosteps(self):
        return self.num_envs * self.seq_rosteps

    def get_num_self_play_schedule(self):
        if self.num_self_play_schedule == '':
            return []
        else:
            items = []
            for kv in self.num_self_play_schedule.split(","):
                [k, v] = kv.split(":")
                items.append((float(k), int(v)))
            return list(reversed(items))

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_const', const=False, dest=name)
                parser.add_argument(f"--{name}", action='store_const', const=True, dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


