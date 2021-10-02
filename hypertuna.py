import optuna
import xprun
import random
import wandb
from copy import deepcopy


class HyperOptimizer:
    def __init__(self):
        self.config = xprun.build_xpdef(
            "xprun.ron", ignore_dirty=False, include_dirty=True, verbose=False,
        )
        self.config.containers[0].command += [
            "--config",
            "configs/allied_wealth.yaml",
            "--hps",
            "ppo.steps=1e6",
        ]
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.xp_name = f"optuna-{random.randint(0, 0xffffff):06x}"

    def objective(self, trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
        final_lr_mult = trial.suggest_loguniform("final_lr_mult", 1e-2, 1e1)
        # batch_size = 2 ** trial.suggest_int("batch_size", 8, 14)
        xp = deepcopy(self.config)
        xp.base_name = self.xp_name
        xp.name = f"{self.xp_name}-lr={lr:.4f}-final_lr={lr*final_lr_mult:.4f}-{self.trial}"
        xp.containers[0].command += [
            f"optimizer.lr=step: {lr}@0 cos {lr*final_lr_mult}@1e6"
        ]
        self.trial += 1
        self.xprun.run_to_completion(xp, wait=True, priority=3, user="clemens")
        run = list(
            self.wandb.runs("cswinter/deep-codecraft", {"config.xp_name": xp.name})
        )[0]
        print(run)
        return run.summary["eprewmean"]

    def optimize(self):
        study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        study.optimize(self.objective, n_trials=25, n_jobs=6)
        print(study.best_params)


if __name__ == "__main__":
    HyperOptimizer().optimize()
