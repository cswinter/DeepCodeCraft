import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'SGD'      # Optimizer
        self.lr = 0.001             # Learning rate
        self.momentum = 0.9         # Momentum
        self.bs = 128               # Batch size during optimization
        self.shuffle = False        # Shuffle samples collected during rollout before optimization

        # Policy
        self.depth = 3              # Number of hidden layers
        self.width = 1024           # Number of activations on each hidden layer

        # RL
        self.steps = 1e7            # Total number of timesteps
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.rosteps = 256 * 32     # Number of total rollout steps
        self.gamma = 0.99           # Discount factor

        # Task
        self.objective = envs.Objective.ALLIED_WEALTH
        self.game_length = 3 * 60 * 60
        self.action_delay = 0

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            if isinstance(value, bool):
                parser.add_argument(f"--no-{name}", action='store_false', dest=name)
                parser.add_argument(f"--{name}", action='store_true', dest=name)
            else:
                parser.add_argument(f"--{name}", type=type(value))
        return parser


