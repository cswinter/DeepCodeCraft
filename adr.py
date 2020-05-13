from collections import defaultdict
from gym_codecraft.envs.codecraft_vec_env import Rules


class ADR:
    def __init__(self, stepsize=0.05):
        self.ruleset = Rules()
        self.ruleset.cost_modifier_storage = 0.7
        self.ruleset.cost_modifier_constructor = 0.5
        self.ruleset.cost_modifier_engines = 0.5
        self.ruleset.cost_modifier_size[2] = 0.8
        self.ruleset.cost_modifier_size[3] = 0.6
        self.target_fractions = normalize({
            '1m': 10.0,
            '1s': 5.0,
            '1m1p': 4.0,
            '2m': 2.0,
            '1s1c': 2.0,
            '2m1e1p': 2.0,
            '3m1p': 2.0,
            '2m2p': 2.0,
            '2s2c': 2.0,
            '2s1c1e': 2.0,
            '2s1m1c': 2.0,
        })
        self.stepsize = stepsize

    def step(self, counts):
        gradient = defaultdict(lambda: 0.0)
        for build, bfraction in normalize(counts).items():
            loss = self.target_fractions[build] - bfraction
            for module, mfraction in module_norm(build).items():
                gradient[module] += mfraction * loss * self.stepsize
            gradient[f'size{size(build)}'] += loss * self.stepsize
        for key, grad in gradient.items():
            if key == 'm':
                self.ruleset.cost_modifier_missiles -= grad
            if key == 's':
                self.ruleset.cost_modifier_storage -= grad
            if key == 'p':
                self.ruleset.cost_modifier_shields -= grad
            if key == 'c':
                self.ruleset.cost_modifier_constructor -= grad
            if key == 'e':
                self.ruleset.cost_modifier_engines -= grad
            if key == 'size1':
                self.ruleset.cost_modifier_size[0] -= grad
            if key == 'size2':
                self.ruleset.cost_modifier_size[1] -= grad
            if key == 'size3':
                self.ruleset.cost_modifier_size[2] -= grad
            if key == 'size4':
                self.ruleset.cost_modifier_size[3] -= grad

        high = max(
            self.ruleset.cost_modifier_storage,
            self.ruleset.cost_modifier_engines,
            self.ruleset.cost_modifier_shields,
            self.ruleset.cost_modifier_missiles,
            self.ruleset.cost_modifier_constructor,
            *self.ruleset.cost_modifier_size
        )

        def clip(val):
            return max(0.5, val / high)
        self.ruleset.cost_modifier_storage = clip(self.ruleset.cost_modifier_storage)
        self.ruleset.cost_modifier_engines = clip(self.ruleset.cost_modifier_engines)
        self.ruleset.cost_modifier_shields = clip(self.ruleset.cost_modifier_shields)
        self.ruleset.cost_modifier_missiles = clip(self.ruleset.cost_modifier_missiles)
        self.ruleset.cost_modifier_constructor = clip(self.ruleset.cost_modifier_constructor)
        for i in range(4):
            self.ruleset.cost_modifier_size[i] = clip(self.ruleset.cost_modifier_size[i])

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

