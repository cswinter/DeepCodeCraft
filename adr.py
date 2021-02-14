import math
from collections import defaultdict
from gym_codecraft.envs.codecraft_vec_env import Rules
from typing import List


class ADR:
    def __init__(self,
                 hstepsize,
                 stepsize=0.002,
                 warmup=10,
                 initial_hardness=0.0,
                 ruleset: Rules = None,
                 linear_hardness: bool = False,
                 max_hardness: float = 200,
                 hardness_offset: float = 0,
                 variety: float = 0.7,
                 step: int = 0,
                 average_cost_target: float = 0.8):
        if ruleset is None:
            ruleset = Rules()
        self.ruleset = ruleset
        self.variety = variety

        self.target_modifier = average_cost_target
        self.stepsize = stepsize
        self.warmup = warmup
        self.step = step

        self.w_ema = 0.5
        self.counts = defaultdict(lambda: 0.0)

        self.hardness = initial_hardness
        self.max_hardness = max_hardness
        self.linear_hardness = linear_hardness
        self.hardness_offset = hardness_offset
        self.stepsize_hardness = hstepsize
        self.target_elimination_rate = 0.97

    def target_eplenmean(self):
        if self.hardness < 25:
            return 250 + 6 * self.hardness
        elif self.hardness < 50:
            return 400 + 4 * (self.hardness - 25)
        elif self.hardness < 100:
            return 500 + 2 * (self.hardness - 50)
        else:
            return 600

    def adjust(self, counts, elimination_rate, eplenmean, step) -> float:
        self.step += 1
        stepsize = self.stepsize * min(1.0, self.step / self.warmup)
        for build, bfraction in counts.items():
            self.counts[build] = (1 - self.w_ema) * bfraction + self.w_ema * self.counts[build]

        target_fraction = 1.0 / len(self.counts) if len(self.counts) > 0 else 1
        gradient = defaultdict(lambda: 0.0)
        for build, bfraction in normalize(self.counts).items():
            if bfraction == 0:
                loss = -100
            else:
                loss = -self.variety * math.log(target_fraction / bfraction)
            gradient[build] += loss

        modifier_decay = 1 - self.variety
        for spec, modifier in self.ruleset.cost_modifiers.items():
            gradient[spec] += modifier_decay * math.log(self.target_modifier / modifier)

        size_weighted_counts = normalize({build: count * size(build) for build, count in self.counts.items()})
        average_modifier = sum([self.ruleset.cost_modifiers[build] * bfraction
                                for build, bfraction in size_weighted_counts.items()])

        if average_modifier == 0:
            return 0

        average_cost_grad = 10 * math.log(self.target_modifier / average_modifier)
        for spec, grad in gradient.items():
            exponent = stepsize * min(10.0, max(-10.0, grad + average_cost_grad))
            multiplier = math.exp(exponent)
            self.ruleset.cost_modifiers[spec] *= multiplier

        if step > self.hardness_offset:
            if self.linear_hardness:
                self.hardness = min((step - self.hardness_offset) * self.stepsize_hardness, self.max_hardness)
            else:
                if eplenmean is not None:
                    self.hardness += self.stepsize_hardness * (self.target_eplenmean() - eplenmean)
                    self.hardness = max(0.0, self.hardness)

        return average_modifier

    def metrics(self):
        return {
            f'adr_{spec_key(spec)}_cost': cost
            for spec, cost in self.ruleset.cost_modifiers.items()
        }


def spec_key(module_counts: List[int]):
    key = ''
    [s, m, c, e, p] = module_counts
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
    return key


def size(build):
    return sum(build)


def normalize(weights):
    total = sum(weights.values())
    if total == 0:
        total = 1
    return {key: weight / total for key, weight in weights.items()}

