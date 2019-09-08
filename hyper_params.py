import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'Adam'     # Optimizer
        self.lr = 0.0001            # Learning rate
        self.momentum = 0.9         # Momentum
        self.weight_decay = 0
        self.bs = 2048              # Batch size during optimization
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 0.5          # Weighting of value function loss in optimization objective
        self.max_grad_norm = 1      # Maximum gradient norm for gradient clipping

        # Policy
        self.depth = 4              # Number of hidden layers
        self.width = 1024           # Number of activations on each hidden layer
        self.conv = False           # Use convolution to share weights on objects
        self.fp16 = False           # Whether to use half-precision floating point

        # RL
        self.steps = 20e6           # Total number of timesteps
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.rosteps = 256 * 32     # Number of total rollout steps
        self.gamma = 0.99           # Discount factor
        self.lamb = 0.95            # Generalized advantage estimation parameter lambda
        self.norm_advs = True       # Normalize advantage values
        self.rewscale = 1.0         # Scaling of reward values
        self.ppo = True             # Use PPO-clip instead of vanilla policy gradients objective
        self.cliprange = 0.2        # PPO cliprange

        # Task
        self.objective = envs.Objective.ALLIED_WEALTH
        self.game_length = 3 * 60 * 60
        self.action_delay = 0

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_const', const=False, dest=name)
                parser.add_argument(f"--{name}", action='store_const', const=True, dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


