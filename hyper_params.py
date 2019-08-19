import argparse
from gym_codecraft import envs


class HyperParams:
    def __init__(self):
        # Optimizer
        self.optimizer = 'SGD'      # Optimizer
        self.lr = 0.001             # Learning rate
        self.momentum = 0.8         # Momentum
        self.bs = 128               # Batch size during optimization
        self.shuffle = True         # Shuffle samples collected during rollout before optimization
        self.vf_coef = 2            # Weighting of value function loss in optimization objective
        self.max_grad_norm = 0.5    # Maximum gradient norm for gradient clipping

        # Policy
        self.depth = 3              # Number of hidden layers
        self.width = 1024           # Number of activations on each hidden layer

        # RL
        self.steps = 5e6            # Total number of timesteps
        self.seq_rosteps = 64       # Number of sequential steps per rollout
        self.rosteps = 64 * 32      # Number of total rollout steps
        self.gamma = 0.8            # Discount factor
        self.norm_advs = True       # Normalize advantage values
        self.rewscale = 1.0         # Scaling of reward values

        # Task
        self.objective = envs.Objective.DISTANCE_TO_CRYSTAL
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


