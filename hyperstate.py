from abc import ABC, abstractclassmethod, abstractmethod
from enum import Enum, EnumMeta
from torch import optim
from torch.optim.optimizer import Optimizer
from policy_t8 import TransformerPolicy8
from typing import Generic, List, Any, Type, TypeVar, Dict, Tuple
from dataclasses import dataclass, field, is_dataclass
from gym_codecraft import envs
import yaml


def qualified_name(clz):
    if clz.__module__ == "builtin":
        return clz.__name__
    else:
        return f"{clz.__module__}.{clz.__name__}"


def _typecheck(name, value, typ):
    if not isinstance(value, typ):
        raise TypeError(
            f"{name} has type {qualified_name(typ)}, but received value {value} of type {qualified_name(value.__class__)}."
        )


T = TypeVar("T")


def load_file(clz: Type[T], path: str) -> T:
    file = open(path)
    values = yaml.safe_load(file)
    if not is_dataclass(clz):
        raise TypeError(f"{clz.__module__}.{clz.__name__} must be a dataclass")
    return _parse(clz, values)


def _parse(clz: Type[T], values: Dict[str, Any]) -> T:
    print(values)
    kwargs = {}
    remaining_fields = set(clz.__annotations__.keys())
    for field_name, value in values.items():
        if field_name not in remaining_fields:
            raise TypeError(
                f"{clz.__module__}.{clz.__name__} has no attribute {field_name}."
            )
        else:
            remaining_fields.remove(field_name)
        field_clz = clz.__annotations__[field_name]
        if field_clz == float:
            if isinstance(value, float):
                value = float(value)
            _typecheck(field_name, value, float)
        elif field_clz == int:
            if isinstance(value, str):
                parsed = _parse_int(value)
                if parsed is not None:
                    value = parsed
            _typecheck(field_name, value, int)
        elif field_clz == bool:
            _typecheck(field_name, value, bool)
        elif field_clz == str:
            _typecheck(field_name, value, str)
        elif hasattr(field_clz, "__args__") and field_clz == List[field_clz.__args__]:
            _typecheck(field_name, value, list)
            # TODO: nested types, dataclasses etc.
            for i, item in enumerate(value):
                _typecheck(f"{field_name}[{i}]", item, field_clz.__args__[0])
        elif is_dataclass(field_clz):
            value = _parse(field_clz, value)
        elif isinstance(field_clz, EnumMeta):
            value = field_clz(value)
        else:
            __import__("ipdb").set_trace()
            raise TypeError(
                f"Field {clz.__module__}.{clz.__name__}.{field_name} has unsupported type {field_clz.__module__}.{field_clz.__name__}."
            )
        kwargs[field_name] = value

    try:
        return clz(**kwargs)
    except TypeError as e:
        raise TypeError(f"Failed to initialize {clz.__module__}.{clz.__name__}: {e}")


# TODO: gah
def _parse_int(s: str):
    try:
        f = float(s)
        i = int(f)

        return i if f == i else None
    except:
        return None


# Initial call:
# - path to config
# - extra params for override
# - config, state = State.load(config, overrides)
#
# From checkpoint:
# - path to folder with config + state
# - extra params for override
# - config, state = State.load(config, overrides)


class Opaque(Generic[T]):
    pass


@dataclass
class OptimizerConfig:
    # Optimizer ("SGD" or "RMSProp" or "Adam")
    optimizer_type: str = "Adam"
    # Learning rate
    lr: float = 0.0003
    # Learning rate floor when using cosine schedule
    final_lr: float = 0.0001
    # Learning rate schedule ("none" or "cosine")
    lr_schedule: str = "none"
    # Momentum
    momentum: float = 0.9
    # Weight decay
    weight_decay: float = 0.0001
    # Batch size during optimization
    bs: int = 2048
    # Accumulate gradients over this many batches before applying gradients
    batches_per_update: int = 1
    # Shuffle samples collected during rollout before optimization
    shuffle: bool = True
    # Weighting of value function loss in optimization objective
    vf_coef: float = 1.0
    # Weighting of  entropy bonus in loss function
    entropy_bonus: float = 0.0
    # Maximum gradient norm for gradient clipping
    max_grad_norm: float = 20.0
    # Number of optimizer passes over samples collected during rollout
    epochs: int = 2
    # Learning rate is increased linearly from 0 during first n samples
    warmup: int = 0
    # Exponentially moving averages of model weights
    weights_ema: List[float] = field(default_factory=list)
    # [0.99, 0.997, 0.999, 0.9997, 0.9999]


@dataclass
class PolicyConfig:
    d_agent: int = 256
    d_item: int = 128
    dff_ratio: int = 2
    nhead: int = 8
    item_item_attn_layers: int = 0
    dropout: float = 0.0  # Try 0.1?
    # Construct map of nearby objects populated with scatter connections
    nearby_map: bool = False
    # Width of circles on nearby map
    nm_ring_width = 60
    # Number of rays on nearby map
    nm_nrays = 8
    # Number of rings on nearby map
    nm_nrings = 8
    # Whether to perform convolution on nearby map
    map_conv = False
    # Size of convolution kernel for nearby map
    mc_kernel_size: int = 3
    # Whether the nearby map has 2 channels corresponding to the offset of objects within the tile
    map_embed_offset: bool = False
    # Adds itemwise ff resblock after initial embedding before transformer
    item_ff: bool = True
    # Max number of simultaneously controllable drones
    agents: int = 1
    # Max number of allies observed by each drone
    nally: int = 1
    # Max number of enemies observed by each drone
    nenemy: int = 0
    # Max number of minerals observed by each drone
    nmineral: int = 10
    # Number of map tiles observed by each drone
    ntile: int = 0
    # Number learnable constant valued items observed by each drone
    nconstant: int = 0
    # Use same weights for processing ally and enemy drones
    ally_enemy_same: bool = False
    # Normalization layers ("none", "batchnorm", "layernorm")
    norm: str = "layernorm"
    # Set all initial weights for value function head to zero
    zero_init_vf: bool = True
    # Set initial weights for policy head to small values and biases to zero
    small_init_pi: bool = False


@dataclass
class ObsConfig:
    # Max number of allied drones returned by the env
    allies: int = 10
    # Max number of enemy drones returned by the env
    obs_enemies: int = 10
    # Max number of minerals returned by the env
    obs_minerals: int = 10
    # Max number of map tiles returned by the env
    obs_map_tiles: int = 10
    # Have features for both absolute and relative positions on each object
    obs_keep_abspos: bool = False
    # Whether value function has access to hidden information
    use_privileged: bool = True
    # Global features for width/height of map
    feat_map_size: bool = True
    # Remember last position/time each enemy was seen + missile cooldown feat
    feat_last_seen: bool = False
    # Feature for whether drone is currently visible
    feat_is_visible: bool = True
    # Global features for absolute remaining/elapsed number of timesteps
    feat_abstime: bool = True
    # Feature for whether another drone is currently harvesting a mineral
    feat_mineral_claims: bool = False
    # Harvest action that will freeze drone until one resource has been harvested
    harvest_action: bool = False
    # Pair of actions to disable/enable all build actions
    lock_build_action: bool = False
    # Five features giving distance to closest wall in movement direction, and in movement direction offset by +-pi/2 and +-pi/4
    feat_dist_to_wall: bool = False
    feat_unit_count: bool = True
    feat_construction_progress: bool = True

    # TODO: hack
    feat_rule_msdm = None
    feat_rule_costs = None
    num_builds = None

    @property
    def drones(self):
        return self.allies + self.obs_enemies

    def global_features(self):
        gf = 2
        if self.feat_map_size:
            gf += 2
        if self.feat_abstime:
            gf += 2
        if self.feat_rule_msdm:
            gf += 1
        if self.feat_rule_costs:
            gf += self.num_builds
        if self.feat_unit_count:
            gf += 1
        return gf

    def dstride(self):
        ds = 17
        if self.feat_last_seen:
            ds += 2
        if self.feat_is_visible:
            ds += 1
        if self.lock_build_action:
            ds += 1
        if self.feat_dist_to_wall:
            ds += 5
        if self.feat_construction_progress:
            ds += self.num_builds + 2
        return ds

    def mstride(self):
        return 4 if self.feat_mineral_claims else 3

    def tstride(self):
        return 4

    def nonobs_features(self):
        return 5

    def enemies(self):
        return self.drones - self.allies

    def total_drones(self):
        return 2 * self.drones - self.allies

    def stride(self):
        return (
            self.global_features()
            + self.total_drones() * self.dstride()
            + self.obs_minerals * self.mstride()
            + self.obs_map_tiles * self.tstride()
        )

    def endglobals(self):
        return self.global_features()

    def endallies(self):
        return self.global_features() + self.dstride() * self.allies

    def endenemies(self):
        return self.global_features() + self.dstride() * self.drones

    def endmins(self):
        return self.endenemies() + self.mstride() * self.obs_minerals

    def endtiles(self):
        return self.endmins() + self.tstride() * self.obs_map_tiles

    def endallenemies(self):
        return self.endtiles() + self.dstride() * self.enemies()

    def extra_actions(self):
        if self.lock_build_action:
            return 2
        else:
            return 0

    @property
    def global_drones(self):
        return self.obs_enemies if self.use_privileged else 0

    @property
    def unit_count(self):
        return self.feat_unit_count

    @property
    def construction_progress(self):
        return self.feat_construction_progress


@dataclass
class EvalConfig:
    eval_envs: int = 256
    eval_timesteps: int = 360
    eval_frequency: int = 1e5
    model_save_frequency: int = 10
    eval_symmetric: bool = True
    full_eval_frequency: int = 5
    extra_checkpoint_steps: List[int] = field(default_factory=list)


@dataclass
class PPOConfig:
    # Total number of timesteps
    steps: int = 10e6
    # Number of environments
    num_envs: int = 64
    # Number of self-play environments (each provides two environments)
    num_self_play: int = 32
    # Number of environments played vs scripted replicator AI
    num_vs_replicator: int = 0
    # Number of environments played vs scripted aggressive replicator AI
    num_vs_aggro_replicator: int = 0
    # Number of environments played vs scripted destroyer AI
    num_vs_destroyer: int = 0
    # Number of sequential steps per rollout
    seq_rosteps: int = 256
    # Discount factor
    gamma: float = 0.99
    # Generalized advantage estimation parameter lambda
    lamb: float = 0.95
    # Normalize advantage values
    norm_advs: bool = True
    # Scaling of reward values
    rewscale: float = 1.0
    # Use PPO-clip instead of vanilla policy gradients objective
    ppo: bool = True
    # PPO cliprange
    cliprange: float = 0.2
    # Use clipped value function objective
    clip_vf: bool = True
    # Split reward evenly amongst all active agents.
    split_reward: bool = False
    # Negative reward applied at each timestep
    liveness_penalty: float = 0.0
    # Extra reward for building a drone type at least once during episode
    build_variety_bonus: float = 0.0
    # Reward received when winning game by eliminating opponent
    win_bonus: float = 0.0
    # Negative reward received when losing game by being eliminated
    loss_penalty: float = 0.0
    # Instantaneous reward received from change in relative amount of resources under allied control
    partial_score: float = 1.0
    # Fraction of shaped reward awarded for minimum health of enemy mothership during episode
    attac: float = 0.0
    # Fraction of shaped reward awarded for maximum health of allied mothership during episode
    protec: float = 0.0
    # Rescale reward values by ema of mean and variance
    rewnorm: bool = False
    rewnorm_emaw: float = 0.97
    max_army_size_score: float = 9999999
    max_enemy_army_size_score: float = 9999999


@dataclass
class TaskConfig:
    objective: envs.Objective = envs.Objective.ARENA_TINY_2V2
    action_delay: int = 0
    use_action_masks: bool = True
    task_hardness: int = 0
    # Max length of games, or default game length for map if 0.
    max_game_length: int = 0
    # Maxiumum map area
    max_hardness: int = 150
    # Number of timesteps steps after which hardness starts to increase
    hardness_offset: int = 1e6
    randomize: bool = True
    # Percentage of maps which are symmetric
    symmetric_map: float = 0.0
    # Linearly increase env symmetry parameter with this slope for every step
    symmetry_increase: float = 2e-8
    # Fraction of maps that use MICRO_PRACTICE instead of the main objective
    mix_mp: float = 0.0
    # Fraction of maps that use randomize ruleset
    rule_rng_fraction: float = 0.0
    # Amount of rule randomization
    rule_rng_amount: float = 1.0
    rule_cost_rng: float = 0.0
    # Automatically adjust environment rules
    adr: bool = False
    # Amount by which task difficulty/map size is increased for each processed frame
    adr_hstepsize: float = 2.0e-6
    # Linearly increase task difficulty/map size
    linear_hardness: bool = True
    mothership_damage_scale: float = 4.0
    enforce_unit_cap: bool = False
    unit_cap: int = 0


@dataclass
class AdrConfig:
    # Target value for average module cost
    hstepsize: float = 3.0e-6
    average_cost_target: float = 1.0
    cost_variance: float = 0.5
    variety: float = 0.8
    stepsize: float = 0.003
    warmup: int = 100
    initial_hardness: float = 0.0
    linear_hardness: bool = False
    max_hardness: float = 200
    hardness_offset: float = 0
    variety: float = 0.7
    average_cost_target: float = 0.8


@dataclass
class Config:
    optimizer: OptimizerConfig
    eval: EvalConfig
    ppo: PPOConfig
    task: TaskConfig
    adr: AdrConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)

    @property
    def rosteps(self):
        return self.ppo.num_envs * self.ppo.seq_rosteps

    def validate(self):
        assert (
            self.ppo.seq_rosteps
            % (self.optimizer.bs * self.optimizer.batches_per_update)
            == 0
        )
        assert self.eval.eval_envs % 4 == 0


class Serializer(ABC, Generic[T]):
    @abstractclassmethod
    def serialize(t: T) -> Tuple[bytes, str]:
        raise NotImplementedError("Method `serialize` not implemented.")

    @abstractclassmethod
    def deserialize(serialized: bytes, tag: str) -> T:
        raise NotImplementedError("Method `deserialize` not implemented.")


class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> Tuple[bytes, str]:
        raise NotImplementedError("Method `serialize` not implemented.")

    @abstractclassmethod
    def deserialize(clz: Type[T], serialized: bytes, tag: str) -> T:
        raise NotImplementedError("Method `deserialize` not implemented.")


TSerializer = TypeVar("TSerializer", bound=Serializer)


@dataclass
class Serializable(Generic[T, TSerializer]):
    inner: T


class OptimizerSerializer(Serializer[Optimizer]):
    pass


class Policy:
    pass


# TODO:
# - separate state and config
# - first: load config
# - second: initialize or load state
#   - config passed in as param
#   - sets _state field on config that allows `property` to work
# - state and config stored in separate files (config.yaml, state.yaml)
# - support schedules by initializing field with `property` functions that take _state as argument
#
# NEW IDEAS:
# - add `Serializable` class to support arbitrary types (or just duck type `serialize`, `deserialize` methods)
# - don't need `Opaque` or `Lazy`: when loading, can just get lazy loading with property
# - for now, keep it simple: require separate load calls for state and config and explicit init call for config (actually, can just be normal constructor). can refactor later.
# - xprun can call with `config-path` and (optional) `state-path`
# - on xprun lib (or hyperstate) call "set-checkpoint-path" to save checkpoint, get `state-path` arg on subsequent resume calls
#
# QUESTION: how to support choice between different datatypes? e.g. different policies, different optimizers.
# => abstract base class has a `serialize` class method that also takes a tag, can choose which concrete instance to return
# => the `deserialize` class method also returns a tag
# => incorporate tag into filename, so serialized object is still deserializable without hyperstate tag parsing
# => if no tag supplied, use "default" tag. so then you can decide to add tagging later, still keep old things working.
# QUESTION: how to handle initialization args/dependencies/extra stuff: e.g. sending to device (determine from config, ez), optimizer initialized from network params
