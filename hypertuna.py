from collections import defaultdict
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
import numpy as np

from enum import Enum

from config import Config
from hyperstate.hyperstate import _load_file_and_schedules


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
    "vf_coef": HyperParam(
        path="optimizer.vf_coef", sampling_strategy=SamplingStrategy.LOGUNIFORM,
    ),
}


class HyperOptimizer:
    def __init__(
        self,
        base_config_path: str,
        params: List[Tuple[str, float]],
        parallelism: int = 6,
        steps: Optional[int] = None,
        xps_per_trial: int = 1,
        priority: int = 3,
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
        self.outstanding_xps = 0
        self.parallelism = parallelism
        self.base_config_path = base_config_path
        self.steps = steps
        self.xps_per_trial = xps_per_trial
        self.trial_results = defaultdict(list)
        self.best_result = None
        self.best_config = None
        self.priority = priority

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
        self.steps = steps or hpconfig.ppo.steps

    def base_xp_config(self, trial: int) -> Config:
        xp = deepcopy(self.config)
        xp.containers[0].command += [
            "--config",
            self.base_config_path,
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
        threads = []
        for trial_id in range(n_trials):
            # Wait until we have a free slot
            with self.lock:
                while self.running_xps >= self.parallelism or self.outstanding_xps > 0:
                    self.cvar.wait()
            trial = self.study.ask()
            xp = self.sample_xp(trial)
            self.outstanding_xps += self.xps_per_trial
            thread = threading.Thread(
                target=self.run_trial, args=(xp, trial, trial_id,),
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        print(f"Best result: {self.best_result}")
        print(f"Best config: {self.best_config}")

    def run_trial(self, xp, trial, trial_id):
        threads = []
        for i in range(self.xps_per_trial):
            with self.lock:
                while self.running_xps >= self.parallelism:
                    self.cvar.wait()
                self.running_xps += 1
                self.outstanding_xps -= 1
                _xp = deepcopy(xp)
                if self.xps_per_trial > 1:
                    _xp.containers[0].command.append(f"trial={trial_id}")
                    _xp.name = f"{_xp.name}-{i}"
                thread = threading.Thread(target=self.run_xp, args=(_xp, trial,))
                thread.start()
                threads.append(thread)
                self.cvar.notify()
        for thread in threads:
            thread.join()
        result = np.array(self.trial_results[trial]).mean()
        with self.lock:
            self.study.tell(trial, result)
            print(f"Trial {trial_id}: {result}")
            if self.best_result is None or result > self.best_result:
                self.best_result = result
                self.best_config = xp.containers[0].command
                print(f"New best config: {self.best_config}")

    def run_xp(self, xp, trial):
        for retry in range(10):
            try:
                self.xprun.run(xp, wait=True, priority=self.priority, user="clemens")
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
        with self.lock:
            self.running_xps -= 1
            self.trial_results[trial].append(result)
            self.cvar.notify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config_path", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--params", type=str, nargs="+", default=[])
    parser.add_argument("--parallelism", type=int, default=6)
    parser.add_argument("--steps", type=float)
    parser.add_argument("--xps_per_trial", type=int, default=1)
    parser.add_argument("--priority", type=int, default=3)
    args = parser.parse_args()
    params = []
    for param in args.params:
        path, r = param.split("=")
        params.append((path, float(r),))
    HyperOptimizer(
        args.base_config_path,
        params,
        args.parallelism,
        float(args.steps) if args.steps is not None else None,
        xps_per_trial=args.xps_per_trial,
        priority=args.priority,
    ).run(args.n_trials)

# python hypertuna.py --base_config_path=configs/arena_medium_10m.yaml --params lr=10 gamma=10 batch_size=4 weight_decay=10 entropy_bonus=10 cliprange=0.1 seq_rosteps=4 num_envs=4
