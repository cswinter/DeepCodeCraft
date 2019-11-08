import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'RMSProp'  # Optimizer ("SGD" or "RMSProp" or "Adam")
        self.lr = 0.0001            # Learning rate
        self.momentum = 0.9         # Momentum
        self.weight_decay = 0.0001
        self.bs = 2048              # Batch size during optimization
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 1.0          # Weighting of value function loss in optimization objective
        self.max_grad_norm = 1.0    # Maximum gradient norm for gradient clipping
        self.sample_reuse = 3       # Number of optimizer passes over samples collected during rollout
        self.lr_ratios = 1.0        # Learning rate multiplier applied to earlier layers
        # TODO: weak evidence this might help, enables higher lr
        self.warmup = 0             # Learning rate is increased linearly from 0 during first n samples

        # Policy
        self.depth = 4              # Number of hidden layers
        self.width = 2048           # Number of activations on each hidden layer
        self.fp16 = False           # Whether to use half-precision floating point
        self.zero_init_vf = True    # Set all initial weights for value function head to zero
        self.small_init_pi = False  # Set initial weights for policy head to small values and biases to zero
        self.resume_from = ''       # Filepath to saved policy
        self.obs_allies = 1         # Max number of controllable allies per player
        self.obs_drones = 2         # Max number of drones observed by each drone
        self.obs_minerals = 6       # Max number of minerals observed by each drone
        self.obs_global_drones = 2  # Max number of (possibly hidden) drones observed by value function
        self.mconv_pooling = 'max'  # Pooling layer after mineral convolutions ('max', 'avg' or 'both')
        self.dconv_pooling = 'max'  # Pooling layer after drone convolutions ('max', 'avg' or 'both')
        self.norm = 'batchnorm'     # Normalization layers ("none", "batchnorm", "layernorm")

        # Eval
        self.eval_envs = 0
        self.eval_timesteps = 360
        self.eval_frequency = 1e5
        self.model_save_frequency = 10

        # RL
        self.steps = 10e6           # Total number of timesteps
        self.num_envs = 64          # Number of environments
        self.num_self_play = 0      # Number of self-play environments (each provides two environments)
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.gamma = 0.99           # Discount factor
        self.lamb = 0.95            # Generalized advantage estimation parameter lambda
        self.norm_advs = True       # Normalize advantage values
        self.rewscale = 1.0         # Scaling of reward values
        self.ppo = True             # Use PPO-clip instead of vanilla policy gradients objective
        self.cliprange = 0.2        # PPO cliprange
        self.clip_vf = False        # Use clipped value function objective

        self.rosteps = self.num_envs * self.seq_rosteps

        # Task
        self.objective = envs.Objective.ALLIED_WEALTH
        self.action_delay = 0
        self.use_action_masks = True

    @staticmethod
    def allied_wealth():
        hps = HyperParams()
        hps.clip_vf = True
        hps.depth = 4
        hps.eval_envs = 0
        hps.gamma = 0.99
        hps.lamb = 0.95
        hps.max_grad_norm = 5.0
        hps.momentum = 0.9
        hps.norm_advs = True
        hps.num_envs = 64
        hps.num_self_play = 0
        hps.obs_allies = 1
        hps.obs_drones = 0
        hps.obs_global_drones = 0
        hps.obs_minerals = 10
        hps.sample_reuse = 2
        hps.small_init_pi = False
        hps.use_action_masks = True
        hps.use_privileged = False
        hps.vf_coef = 1.0
        hps.weight_decay = 0.0001
        hps.width = 2048
        hps.zero_init_vf = True
        return hps

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_const', const=False, dest=name)
                parser.add_argument(f"--{name}", action='store_const', const=True, dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


