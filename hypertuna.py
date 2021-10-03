import optuna
import xprun
import random
import wandb
from copy import deepcopy


class HyperOptimizer:
    def __init__(self):
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.xp_name = f"optuna-{random.randint(0, 0xffffff):06x}"
        xp = xprun.build_xpdef(
            "xprun.ron", ignore_dirty=False, include_dirty=False, verbose=False,
        )
        xp.base_name = self.xp_name
        self.config = xp

    def objective(self, trial: optuna.trial.Trial):
        # return self.allied_wealth(trial)
        return self.arena_medium(trial)

    def arena_medium(self, trial: optuna.trial.Trial):
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
        final_lr_mult = trial.suggest_loguniform("final_lr_mult", 1e-2, 1)
        lg_seq_rosteps = trial.suggest_int("lg_seq_rosteps", 3, 8)
        seq_rosteps = 2 ** lg_seq_rosteps
        lg_num_envs = trial.suggest_int("lg_num_envs", 7, 10)
        num_envs = 2 ** lg_num_envs
        batch_size = 2 ** trial.suggest_int(
            "lg_batch_size", 9, min(14, lg_seq_rosteps + lg_num_envs)
        )
        ommomentum = trial.suggest_loguniform("ommomentum", 0.01, 1)
        momentum = 1 - ommomentum
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        epochs = trial.suggest_int("epochs", 1, 3)
        omgamma = trial.suggest_loguniform("omgamma", 0.0001, 1)
        gamma = 1 - omgamma
        cliprange = trial.suggest_uniform("cliprange", 0.05, 0.5)
        micro_batch_size = min(2048, batch_size)
        xp = deepcopy(self.config)
        xp.containers[0].command += [
            "--config",
            "configs/arena_medium.yaml",
            "--hps",
            "ppo.steps=5e6",
            f"optimizer.lr=step: {lr}@0 cos {lr*final_lr_mult}@5e6",
            f"optimizer.batch_size={batch_size}",
            f"optimizer.micro_batch_size={micro_batch_size}",
            f"optimizer.momentum={momentum}",
            f"optimizer.weight_decay={weight_decay}",
            f"optimizer.epochs={epochs}",
            f"ppo.cliprange={cliprange}",
            f"ppo.gamma={gamma}",
            f"ppo.seq_rosteps={seq_rosteps}",
            f"ppo.num_envs={num_envs}",
            f"--descriptor=lr={lr:.2e},final_lr={lr*final_lr_mult:.2e},batch_size={batch_size},seq_rosteps={seq_rosteps},num_envs={num_envs},momentum={momentum:.2e},weight_decay={weight_decay:.2e},epochs={epochs},cliprange={cliprange:.2e},gamma={gamma:.2e}",
        ]
        xp.name = f"{self.xp_name}-{self.trial}"
        self.trial += 1
        self.xprun.run_to_completion(xp, wait=True, priority=3, user="clemens")
        run = list(
            self.wandb.runs("cswinter/deep-codecraft-vs", {"config.xp_name": xp.name})
        )[0]
        return run.summary.get("eval_mean_score", 0)

    def allied_wealth(self, trial: optuna.trial.Trial):
        lr = trial.suggest_loguniform("lr", 1e-6, 1)
        final_lr_mult = trial.suggest_loguniform("final_lr_mult", 1e-2, 1)
        lg_seq_rosteps = trial.suggest_int("lg_seq_rosteps", 3, 8)
        seq_rosteps = 2 ** lg_seq_rosteps
        lg_num_envs = trial.suggest_int("lg_num_envs", 6, 10)
        num_envs = 2 ** lg_num_envs
        batch_size = 2 ** trial.suggest_int(
            "lg_batch_size", 8, min(14, lg_seq_rosteps + lg_num_envs)
        )
        ommomentum = trial.suggest_loguniform("ommomentum", 0.01, 1)
        momentum = 1 - ommomentum
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
        epochs = trial.suggest_int("epochs", 1, 3)
        omgamma = trial.suggest_loguniform("omgamma", 0.001, 1)
        gamma = 1 - omgamma
        omlambda = trial.suggest_loguniform("omlambda", 0.01, 1)
        lamb = 1 - omlambda
        cliprange = trial.suggest_uniform("cliprange", 0.05, 0.5)
        micro_batch_size = min(2048, batch_size)
        xp = deepcopy(self.config)
        xp.containers[0].requests[0].resources["memory"] = int(3.5e9)
        xp.containers[0].command += [
            "--config",
            "configs/allied_wealth.yaml",
            "--hps",
            "ppo.steps=250e3",
            f"optimizer.lr=step: {lr}@0 cos {lr*final_lr_mult}@250e3",
            f"optimizer.batch_size={batch_size}",
            f"optimizer.micro_batch_size={micro_batch_size}",
            f"optimizer.momentum={momentum}",
            f"optimizer.weight_decay={weight_decay}",
            f"optimizer.epochs={epochs}",
            f"ppo.cliprange={cliprange}",
            f"ppo.gamma={gamma}",
            f"ppo.lamb={lamb}",
            f"ppo.seq_rosteps={seq_rosteps}",
            f"ppo.num_envs={num_envs}",
            f"--descriptor=lr={lr:.2e},final_lr={lr*final_lr_mult:.2e},batch_size={batch_size},seq_rosteps={seq_rosteps},num_envs={num_envs},momentum={momentum:.2e},weight_decay={weight_decay:.2e},epochs={epochs},cliprange={cliprange:.2e},gamma={gamma:.2e},lamb={lamb:.2e}",
        ]
        xp.name = f"{self.xp_name}-{self.trial}"
        self.trial += 1
        self.xprun.run_to_completion(xp, wait=True, priority=3, user="clemens")
        run = list(
            self.wandb.runs("cswinter/deep-codecraft", {"config.xp_name": xp.name})
        )[0]
        return run.summary.get("eprewmean", 0)

    def optimize(self):
        study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        study.optimize(self.objective, n_trials=50, n_jobs=6)
        print(study.best_params)


if __name__ == "__main__":
    HyperOptimizer().optimize()
