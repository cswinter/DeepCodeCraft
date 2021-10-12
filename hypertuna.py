from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import argparse
import optuna
import xprun
import math
import time
import random
import wandb
from copy import deepcopy
import threading

from enum import Enum

from hyperstate import Config, _load_file_and_schedules


class SamplingStrategy(Enum):
    OMINUS = 0
    POWER_OF_TWO = 1
    LOGUNIFORM = 2
    UNIFORM = 3
    SCHEDULE = 4


@dataclass
class HyperParam:
    path: str
    sampling_strategy: SamplingStrategy
    min_value: float = -float("inf")
    max_value: float = float("inf")

    def suggest(self, trial, center, range, steps) -> Tuple[str, float]:
        if self.sampling_strategy == SamplingStrategy.OMINUS:
            min_value = max((1 - center) / range, self.min_value)
            max_value = min((1 - center) * range, self.max_value)
            value = 1 - trial.suggest_uniform(self.optuna_name(), min_value, max_value)
        elif self.sampling_strategy == SamplingStrategy.POWER_OF_TWO:
            min_value = int(max(center / range, self.min_value))
            max_value = int(min(center * range, self.max_value))
            value = 2 ** trial.suggest_int(
                self.optuna_name(), int(math.log2(min_value)), int(math.log2(max_value))
            )
        elif self.sampling_strategy == SamplingStrategy.LOGUNIFORM:
            min_value = max(center / range, self.min_value)
            max_value = min(center * range, self.max_value)
            value = trial.suggest_loguniform(self.optuna_name(), min_value, max_value)
        elif self.sampling_strategy == SamplingStrategy.UNIFORM:
            min_value = max(center - range, self.min_value)
            max_value = min(center + range, self.max_value)
            value = trial.suggest_uniform(f"{self.path}", min_value, max_value)
        elif self.sampling_strategy == SamplingStrategy.SCHEDULE:
            min_value = max(center / range, self.min_value)
            max_value = min(center * range, self.max_value)
            value = trial.suggest_loguniform(f"{self.path}", min_value, max_value)
            return f"{self.path}=step: {value}@0 lin 0@{steps}", value
        return f"{self.path}={value}", value

    def optuna_name(self):
        if self.sampling_strategy == SamplingStrategy.OMINUS:
            return f"om_{self.path}"
        elif self.sampling_strategy == SamplingStrategy.POWER_OF_TWO:
            return f"lg_{self.path}"
        else:
            return f"{self.path}"


hyper_params = {
    "lr": HyperParam(path="optimizer.lr", sampling_strategy=SamplingStrategy.SCHEDULE,),
    "momentum": HyperParam(
        path="optimizer.momentum", sampling_strategy=SamplingStrategy.OMINUS,
    ),
    "weight_decay": HyperParam(
        path="optimizer.weight_decay", sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
    "entropy_bonus": HyperParam(
        path="optimizer.entropy_bonus", sampling_strategy=SamplingStrategy.SCHEDULE,
    ),
    "batch_size": HyperParam(
        path="optimizer.batch_size",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        min_value=512,
    ),
    "cliprange": HyperParam(
        path="ppo.cliprange", sampling_strategy=SamplingStrategy.UNIFORM,
    ),
    "gamma": HyperParam(path="ppo.gamma", sampling_strategy=SamplingStrategy.OMINUS,),
    "seq_rosteps": HyperParam(
        path="ppo.seq_rosteps", sampling_strategy=SamplingStrategy.POWER_OF_TWO,
    ),
    "num_envs": HyperParam(
        path="ppo.num_envs",
        sampling_strategy=SamplingStrategy.POWER_OF_TWO,
        min_value=128,
    ),
}


class HyperOptimizer:
    def __init__(
        self,
        base_config_path: str,
        params: List[Tuple[str, float]],
        parallelism: int = 6,
    ):
        self.xprun = xprun.Client()
        self.wandb = wandb.Api()
        self.trial = 0
        self.xp_name = f"optuna-{random.randint(0, 0xffffff):06x}"
        xp = xprun.build_xpdef(
            "xprun.ron", ignore_dirty=False, include_dirty=True, verbose=False,
        )
        xp.base_name = self.xp_name
        self.config = xp
        self.study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        self.lock = threading.Lock()
        self.cvar = threading.Condition(self.lock)
        self.running_xps = 0
        self.parallelism = parallelism

        hpconfig, schedules = _load_file_and_schedules(Config, base_config_path, [])
        params_with_center = []
        for name, range in params:
            path = hyper_params[name].path
            segments = path.split(".")
            _hpconfig = hpconfig
            while len(segments) > 1:
                _hpconfig = _hpconfig.__getattribute__(segments.pop(0))
            center = _hpconfig.__getattribute__(segments[0])
            params_with_center.append((name, center, range,))
        self.params = params_with_center
        self.steps = hpconfig.ppo.steps

    def base_xp_config(self, trial: int) -> Config:
        xp = deepcopy(self.config)
        xp.containers[0].command += [
            "--config",
            "configs/arena_medium.yaml",
            "--hps",
            f"ppo.steps={self.steps}",
        ]
        xp.name = f"{self.xp_name}-{trial}"
        return xp

    def sample_xp(self, trial):
        xp = self.base_xp_config(self.trial)
        for path, center, range in self.params:
            arg, value = hyper_params[path].suggest(trial, center, range, self.steps)
            xp.containers[0].command.append(arg)
            if path == "batch_size":
                xp.containers[0].command.append(
                    f"optimizer.micro_batch_size={min(value, 2048)}"
                )
            if path == "num_envs":
                xp.containers[0].command.append(f"ppo.num_self_play={value // 2}",)
        self.trial += 1
        return xp

    def run(self, n_trials: int):
        default_params = {}
        for name, center, _ in self.params:
            if hyper_params[name].sampling_strategy == SamplingStrategy.POWER_OF_TWO:
                center = int(math.log2(center))
            elif hyper_params[name].sampling_strategy == SamplingStrategy.OMINUS:
                center = 1 - center
            oname = hyper_params[name].optuna_name()
            default_params[oname] = center
        self.study.enqueue_trial(default_params)
        for _ in range(n_trials):
            # Wait until we have a free slot
            with self.lock:
                while self.running_xps >= self.parallelism:
                    self.cvar.wait()
                self.running_xps += 1
            trial = self.study.ask()
            xp = self.sample_xp(trial)
            threading.Thread(target=self.run_xp, args=(xp, trial,),).start()

    def run_xp(self, xp, trial):
        for retry in range(10):
            try:
                self.xprun.run(xp, wait=True, priority=3, user="clemens")
            except Exception as e:
                print(f"Failed to run {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue
            break
        else:
            print(f"Failed to run {xp.name}")
            return 0
        while True:
            try:
                self.xprun.block_until_completed(xp.name)
                break
            except Exception as e:
                print(f"Failed to block_until_completed {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue
        run = list(
            self.wandb.runs("cswinter/deep-codecraft-vs", {"config.xp_name": xp.name})
        )[0]
        result = run.summary.get("eval_mean_score", -1)
        self.study.tell(trial.number, result)
        with self.lock:
            self.running_xps -= 1
            self.cvar.notify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_path", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--params", type=str, nargs="+", default=[])
    parser.add_argument("--parallelism", type=int, default=6)
    args = parser.parse_args()
    params = []
    for param in args.params:
        path, r = param.split("=")
        params.append((path, float(r),))
    HyperOptimizer(args.base_config_path, params, args.parallelism).run(args.n_trials)


class LegacyHyperOptimizer:
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
        steps = int(5e6)
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
        entropy_bonus1 = trial.suggest_loguniform("entropy_bonus1", 0.1, 10)
        entropy_bonus2 = trial.suggest_loguniform("entropy_bonus2", 0.01, 1)
        micro_batch_size = min(2048, batch_size)
        xp = deepcopy(self.config)
        xp.containers[0].command += [
            "--config",
            "configs/arena_medium.yaml",
            "--hps",
            f"ppo.steps={steps}",
            f"optimizer.lr=step: {lr}@0 cos {lr*final_lr_mult}@{steps}",
            f"optimizer.batch_size={batch_size}",
            f"optimizer.micro_batch_size={micro_batch_size}",
            f"optimizer.momentum={momentum}",
            f"optimizer.weight_decay={weight_decay}",
            f"optimizer.epochs={epochs}",
            f"optimizer.entropy_bonus=step: {entropy_bonus1}@0 lin {entropy_bonus2}@{steps // 2} 0.0@{steps}",
            f"ppo.cliprange={cliprange}",
            f"ppo.gamma={gamma}",
            f"ppo.seq_rosteps={seq_rosteps}",
            f"ppo.num_envs={num_envs}",
            f"ppo.num_self_play={num_envs // 2}",
            f"--descriptor=lr={lr:.2e},final_lr={lr*final_lr_mult:.2e},batch_size={batch_size},seq_rosteps={seq_rosteps},num_envs={num_envs},momentum={momentum:.2e},weight_decay={weight_decay:.2e},epochs={epochs},cliprange={cliprange:.2e},gamma={gamma:.2e}",
        ]
        xp.name = f"{self.xp_name}-{self.trial}"

        self.trial += 1
        self.run_xp(xp)
        run = list(
            self.wandb.runs("cswinter/deep-codecraft-vs", {"config.xp_name": xp.name})
        )[0]
        return run.summary.get("eval_mean_score", -1)

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
        self.run_xp(xp)
        run = list(
            self.wandb.runs("cswinter/deep-codecraft", {"config.xp_name": xp.name})
        )[0]
        return run.summary.get("eprewmean", 0)

    def run_xp(self, xp):
        for retry in range(10):
            try:
                self.xprun.run(xp, wait=True, priority=3, user="clemens")
            except Exception as e:
                print(f"Failed to run {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue
            break
        else:
            print(f"Failed to run {xp.name}")
            return 0
        while True:
            try:
                self.xprun.block_until_completed(xp.name)
                break
            except Exception as e:
                print(f"Failed to block_until_completed {xp.name}: {e}")
                print(f"Retrying in 60 seconds... ({retry})")
                time.sleep(60)
                continue

    def optimize(self):
        study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
        study.optimize(self.objective, n_trials=500, n_jobs=6)
        print(study.best_params)


# python hypertuna.py --base_config_path=configs/arena_medium_10m.yaml --params lr=10 gamma=10 batch_size=4 weight_decay=10 entropy_bonus=10 cliprange=0.1 seq_rosteps=4 num_envs=4
