import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'SGD'      # Optimizer
        self.lr = 0.1               # Learning rate
        self.momentum = 0.9         # Momentum
        self.bs = 128               # Batch size during optimization

        # Policy
        self.depth = 3              # Number of hidden layers
        self.width = 1024           # Number of activations on each hidden layer

        # RL
        self.steps = 1e7            # Total number of timesteps
        self.seq_rosteps = 256      # Number of sequential steps per rollout
        self.rosteps = 256 * 64     # Number of total rollout steps
        self.gamma = 0.9            # Discount factor

        # Task
        self.objective = envs.Objective.DISTANCE_TO_ORIGIN
        self.game_length = 3 * 60 * 60

    def args_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        for name, value in vars(self).items():
            parser.add_argument(f"--{name}", type=type(value))
        return parser


