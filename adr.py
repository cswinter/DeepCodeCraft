from collections import defaultdict
from gym_codecraft.envs.codecraft_vec_env import Rules


class ADR:
    def __init__(self, hstepsize, stepsize=0.05, warmup=100, initial_hardness=0.0, ruleset: Rules = None):
        if ruleset is None:
            ruleset = Rules(
                cost_modifier_size=[1.2, 0.8, 0.8, 0.6],
                cost_modifier_engines=0.7,
                cost_modifier_constructor=0.5,
            )
        self.ruleset = ruleset
        self.target_fractions = normalize({
            '1m': 15.0,
            '1s': 3.0,
            '1m1p': 8.0,
            '2m': 1.0,
            '1s1c': 3.0,
            '2m1e1p': 2.0,
            '3m1p': 1.0,
            '2m2p': 1.0,
            '2s2c': 1.0,
            '2s1c1e': 1.0,
            '2s1m1c': 1.0,
        })
        self.target_modifier = 0.8
        self.stepsize = stepsize
        self.warmup = warmup
        self.step = 0

        self.hardness = initial_hardness
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

    def adjust(self, counts, elimination_rate, eplenmean) -> float:
        self.step += 1
        stepsize = self.stepsize * min(1.0, self.step / self.warmup)
        gradient = defaultdict(lambda: 0.0)
        for build, bfraction in normalize(counts).items():
            loss = self.target_fractions[build] - bfraction
            for module, mfraction in module_norm(build).items():
                gradient[module] += mfraction * loss * stepsize
            # Size is less important predictor of utility than modules, adjust by 0.3
            gradient[f'size{size(build)}'] += 0.3 * loss * stepsize

        size_weighted_counts = normalize({build: count * size(build) for build, count in counts.items()})
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

        average_cost_grad = (self.target_modifier - average_modifier) * stepsize
        for key, grad in gradient.items():
            multiplier = (1.0 - grad) * (1.0 + average_cost_grad)
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
    return {key: weight / total for key, weight in weights.items()}

