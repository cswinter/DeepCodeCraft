from config import AdrConfig
import math
from collections import defaultdict
from gym_codecraft.envs.codecraft_vec_env import Rules
from typing import List, Dict, Tuple

from dataclasses import dataclass, field


@dataclass
class ADRState:
    hardness: float
    ruleset: Rules
    target_elimination_rate: float = 0.97
    w_ema: float = 0.5
    counts: Dict[Tuple[int, int, int, int, int], float] = field(default_factory=dict)
    step: int = 0


@dataclass
class ADR:
    config: AdrConfig
    state: ADRState

    def target_eplenmean(self):
        if self.state.hardness < 25:
            return 250 + 6 * self.state.hardness
        elif self.state.hardness < 50:
            return 400 + 4 * (self.state.hardness - 25)
        elif self.state.hardness < 100:
            return 500 + 2 * (self.state.hardness - 50)
        else:
            return 600

    def adjust(
        self, counts, average_modifier, elimination_rate, eplenmean, step
    ) -> float:
        state = self.state

        state.step += 1
        stepsize = self.config.stepsize * min(1.0, state.step / self.config.warmup)
        for build, bfraction in counts.items():
            if build not in state.counts:
                state.counts[build] = 0.0
            state.counts[build] = (
                1 - state.w_ema
            ) * bfraction + state.w_ema * state.counts[build]

        target_fraction = 1.0 / len(state.counts) if len(state.counts) > 0 else 1
        gradient = defaultdict(lambda: 0.0)
        for build, bfraction in normalize(state.counts).items():
            if bfraction == 0:
                loss = -100
            else:
                loss = -self.config.variety * math.log(target_fraction / bfraction)
            gradient[build] += loss

        modifier_decay = 1 - self.config.variety
        for spec, modifier in state.ruleset.cost_modifiers.items():
            gradient[spec] += modifier_decay * math.log(
                self.config.average_cost_target / modifier
            )

        if average_modifier == 0:
            return 0

        average_cost_grad = 10 * math.log(
            self.config.average_cost_target / average_modifier
        )
        for spec, grad in gradient.items():
            exponent = stepsize * min(10.0, max(-10.0, grad + average_cost_grad))
            multiplier = math.exp(exponent)
            state.ruleset.cost_modifiers[spec] *= multiplier

        if step > self.config.hardness_offset:
            if self.config.linear_hardness:
                state.hardness = min(
                    (step - self.config.hardness_offset) * self.config.hstepsize,
                    self.config.max_hardness,
                )
            else:
                if eplenmean is not None:
                    state.hardness += self.config.hstepsize * (
                        self.target_eplenmean() - eplenmean
                    )
                    state.hardness = max(0.0, state.hardness)

        return average_modifier

    def metrics(self):
        return {
            f"adr_{spec_key(spec)}_cost": cost
            for spec, cost in self.state.ruleset.cost_modifiers.items()
        }


def spec_key(module_counts: List[int]):
    key = ""
    [s, m, c, e, p, l] = module_counts
    if s > 0:
        key += f"{s}s"
    if m > 0:
        key += f"{m}m"
    if c > 0:
        key += f"{c}c"
    if e > 0:
        key += f"{e}e"
    if p > 0:
        key += f"{p}p"
    if l > 0:
        key += f"{l}l"
    return key


def size(build):
    return sum(build)


def normalize(weights):
    total = sum(weights.values())
    if total == 0:
        total = 1
    return {key: weight / total for key, weight in weights.items()}

