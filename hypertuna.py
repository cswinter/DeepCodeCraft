import optuna
import xprun
import random
import wandb
from copy import deepcopy


class HyperOptimizer:
    def __init__(self):
        self.config = xprun.build_xpdef(
            "xprun.ron", ignore_dirty=False, include_dirty=False, verbose=False,
        )
        self.config.containers[0].command += [
            "--config",
            "configs/allied_wealth.yaml",
            "--hps",
            "ppo.steps=500e3",
        ]
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.xp_name = f"optuna-{random.randint(0, 0xffffff):06x}"

    def objective(self, trial: optuna.trial.Trial):
        lr = trial.suggest_loguniform("lr", 1e-6, 1)
        final_lr_mult = trial.suggest_loguniform("final_lr_mult", 1e-2, 1)
        lg_seq_rosteps = trial.suggest_int("lg_seq_rosteps", 4, 8)
        lg_num_envs = trial.suggest_int("lg_num_envs", 4, 10)
        batch_size = 2 ** trial.suggest_int(
            "lg_batch_size", 8, min(14, lg_seq_rosteps + lg_num_envs)
        )
        seq_rosteps = 2 ** lg_seq_rosteps
        num_envs = 2 ** lg_num_envs
        micro_batch_size = min(2048, batch_size)
        xp = deepcopy(self.config)
        xp.base_name = self.xp_name
        xp.name = f"{self.xp_name}-{self.trial}"
        xp.containers[0].command += [
            f"optimizer.lr=step: {lr}@0 cos {lr*final_lr_mult}@500e3",
            f"optimizer.batch_size={batch_size}",
            f"optimizer.micro_batch_size={micro_batch_size}",
            f"ppo.seq_rosteps={seq_rosteps}",
            f"ppo.num_envs={num_envs}",
            f"--descriptor=lr={lr:.4f},final_lr={lr*final_lr_mult:.4f},batch_size={batch_size},seq_rosteps={seq_rosteps},num_envs={num_envs}",
        ]
        self.trial += 1
        self.xprun.run_to_completion(xp, wait=True, priority=3, user="clemens")
        run = list(
            self.wandb.runs("cswinter/deep-codecraft", {"config.xp_name": xp.name})
        )[0]
        print(run)
        return run.summary.get("eprewmean", 0)

    def optimize(self):
        study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        study.optimize(self.objective, n_trials=200, n_jobs=4)
        print(study.best_params)


if __name__ == "__main__":
    HyperOptimizer().optimize()
