import math
from collections import defaultdict
from gym_codecraft.envs.codecraft_vec_env import Rules


class ADR:
    def __init__(self,
                 hstepsize,
                 stepsize=0.02,
                 warmup=10,
                 initial_hardness=0.0,
                 ruleset: Rules = None,
                 linear_hardness: bool = False,
                 max_hardness: float = 200,
                 hardness_offset: float = 0,
                 variety: float = 0.7,
                 step: int = 0):
        if ruleset is None:
            ruleset = Rules(cost_modifier_size=[1.2, 0.6, 0.6, 0.4])
        self.ruleset = ruleset
        self.variety = variety
        self.target_fractions = normalize({b: 1.0 for b in [
            '1m', '1s', '1m1p', '2m', '1s1c', '2m1e1p', '3m1p', '2m2p', '2s2c', '2s1c1e', '2s1m1c'
        ]})

        self.target_modifier = 0.8
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

        gradient = defaultdict(lambda: 0.0)
        weight = defaultdict(lambda: 0.0)
        for build, bfraction in normalize(self.counts).items():
            if bfraction == 0:
                loss = -100
            else:
                loss = -self.variety * math.log(self.target_fractions[build] / bfraction)

            for module, mfraction in module_norm(build).items():
                gradient[module] += mfraction * loss
                weight[module] += mfraction
            size_key = f'size{size(build)}'
            gradient[size_key] += 0.3 * loss
            weight[size_key] += 1
        for key in gradient.keys():
            gradient[key] /= weight[key]

        modifier_decay = 1 - self.variety
        gradient['m'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_missiles)
        gradient['s'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_storage)
        gradient['p'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_shields)
        gradient['c'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_constructor)
        gradient['e'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_engines)
        gradient['size1'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_size[0])
        gradient['size2'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_size[1])
        gradient['size4'] += modifier_decay * math.log(self.target_modifier / self.ruleset.cost_modifier_size[3])

        size_weighted_counts = normalize({build: count * size(build) for build, count in self.counts.items()})
        average_modifier = 0.0
        for build, bfraction in size_weighted_counts.items():
            modifier = 0.0
            for module, mfraction in module_norm(build).items():
                if module == 'm':
                    modifier += self.ruleset.cost_modifier_missiles * mfraction
                if module == 's':
                    modifier += self.ruleset.cost_modifier_storage * mfraction
                if module == 'p':
                    modifier += self.ruleset.cost_modifier_shields * mfraction
                if module == 'c':
                    modifier += self.ruleset.cost_modifier_constructor * mfraction
                if module == 'e':
                    modifier += self.ruleset.cost_modifier_engines * mfraction
            size_modifier = self.ruleset.cost_modifier_size[size(build) - 1]
            average_modifier += modifier * size_modifier * bfraction

        if average_modifier == 0:
            return

        average_cost_grad = 10 * math.log(self.target_modifier / average_modifier)
        for key, grad in gradient.items():
            exponent = stepsize * min(10.0, max(-10.0, grad + average_cost_grad))
            multiplier = math.exp(exponent)
            if key == 'm':
                self.ruleset.cost_modifier_missiles *= multiplier
            if key == 's':
                self.ruleset.cost_modifier_storage *= multiplier
            if key == 'p':
                self.ruleset.cost_modifier_shields *= multiplier
            if key == 'c':
                self.ruleset.cost_modifier_constructor *= multiplier
            if key == 'e':
                self.ruleset.cost_modifier_engines *= multiplier
            if key == 'size1':
                self.ruleset.cost_modifier_size[0] *= multiplier
            if key == 'size2':
                self.ruleset.cost_modifier_size[1] *= multiplier
            if key == 'size3':
                self.ruleset.cost_modifier_size[2] *= multiplier
            if key == 'size4':
                self.ruleset.cost_modifier_size[3] *= multiplier

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
            'adr_missile_cost': self.ruleset.cost_modifier_missiles,
            'adr_storage_cost': self.ruleset.cost_modifier_storage,
            'adr_constructor_cost': self.ruleset.cost_modifier_constructor,
            'adr_engine_cost': self.ruleset.cost_modifier_engines,
            'adr_shield_cost': self.ruleset.cost_modifier_shields,
            'adr_size1_cost': self.ruleset.cost_modifier_size[0],
            'adr_size2_cost': self.ruleset.cost_modifier_size[1],
            'adr_size4_cost': self.ruleset.cost_modifier_size[3],
        }


def size(build):
    modules = 0
    for module in [build[i:i+2] for i in range(0, len(build), 2)]:
        modules += int(module[:1])
    return modules


def module_norm(build):
    weights = defaultdict(lambda: 0.0)
    for module in [build[i:i+2] for i in range(0, len(build), 2)]:
        weights[module[1:]] = float(module[:1])
    return normalize(weights)


def normalize(weights):
    total = sum(weights.values())
    if total == 0:
        total = 1
    return {key: weight / total for key, weight in weights.items()}

