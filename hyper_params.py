import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'RMSProp'  # Optimizer
        self.lr = 0.0001            # Learning rate
        self.momentum = 0.9         # Momentum
        self.weight_decay = 0.0001
        self.bs = 2048              # Batch size during optimization
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 1.0          # Weighting of value function loss in optimization objective
        self.max_grad_norm = 1.0    # Maximum gradient norm for gradient clipping
        self.sample_reuse = 3       # Number of optimizer passes over samples collected during rollout

        # Policy
        self.depth = 4              # Number of hidden layers
        self.width = 2048           # Number of activations on each hidden layer
        self.conv = True            # Use convolution to share weights on objects
        self.fp16 = False           # Whether to use half-precision floating point
        self.zero_init_vf = True    # Set all initial weights for value function head to zero
        self.small_init_pi = False  # Set initial weights for policy head to small values and biases to zero
        self.resume_from = ''       # Filepath to saved policy

        # Eval
        self.eval_envs = 256
        self.eval_timesteps = 360
        self.eval_frequency = 1e5
        self.model_save_frequency = 10

        # RL
        self.steps = 10e6           # Total number of timesteps
        self.num_envs = 64          # Number of environments
        self.num_self_play = 32     # Number of self-play environments (each provides two environments)
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
        self.objective = envs.Objective.ARENA_TINY_2V2
        self.action_delay = 0
        self.use_action_masks = True


    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_const', const=False, dest=name)
                parser.add_argument(f"--{name}", action='store_const', const=True, dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


