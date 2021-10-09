import os
import shutil

from enum import Enum, EnumMeta
import math
from pathlib import Path
import tempfile
from typing import (
    Callable,
    Generic,
    List,
    Any,
    Optional,
    Type,
    TypeVar,
    Dict,
    Tuple,
    Union,
)
from dataclasses import dataclass, field, is_dataclass
import yaml

C = TypeVar("C")
S = TypeVar("S")
T = TypeVar("T")


@dataclass
class HyperState(Generic[C, S]):
    config: C
    state: S
    checkpoint_dir: Optional[Path]
    checkpoint_key: str
    config_clz: Type[C]
    state_clz: Type[S]
    last_checkpoint: Optional[Path] = None
    schedules: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        clz,
        config_clz: Type[C],
        state_clz: Type[S],
        initial_state: Callable[[C], S],
        path: str,
        checkpoint_dir: Optional[str] = None,
        checkpoint_key: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> "HyperState[C, S]":
        """
        Loads a HyperState from a checkpoint (if exists) or initializes a new one.

        :param config_clz: The type of the config object.
        :param state_clz: The type of the state object.
        :param initial_state: A function that takes a config object and returns an initial state object.
        :param path: The path to the checkpoint.
        :param checkpoint_dir: The directory to store checkpoints.
        :param checkpoint_key: The key to use for the checkpoint. This must be a field of the state object (e.g. a field holding current iteration).
        :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
        """
        if checkpoint_key is None:
            checkpoint_key = "step"
        if overrides is None:
            overrides = []

        path = Path(path)
        if os.path.isdir(path):
            config_path = path / "config.yaml"
            state_path = path / "state.yaml"
        else:
            config_path = path
            state_path = None

        config, schedules = _load_file_and_schedules(config_clz, config_path, overrides)
        # TODO: hack
        config.obs.feat_rule_msdm = config.task.rule_rng_fraction > 0 or config.task.adr
        config.obs.feat_rule_costs = config.task.rule_cost_rng > 0 or config.task.adr
        config.obs.num_builds = len(config.task.objective.builds())

        if state_path is None:
            state = initial_state(config)
        else:
            state = load_file(state_clz, state_path, overrides=[])

        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
        hs = HyperState(
            config,
            state,
            checkpoint_dir,
            checkpoint_key,
            config_clz,
            state_clz,
            schedules=schedules,
        )
        apply_schedules(state, config, hs.schedules)
        return hs

    def checkpoint(self, target_dir: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "checkpoint"
            p.mkdir()
            with open(p / "config.yaml", "w") as f:
                yaml.dump(asdict(self.config, schedules=self.schedules), f)
            checkpoint(self.state, p)
            shutil.move(str(p), target_dir)

    def step(self):
        apply_schedules(self.state, self.config, self.schedules)
        if self.checkpoint_dir is not None:
            val = getattr(self.state, self.checkpoint_key)
            assert isinstance(
                val, int
            ), f"checkpoint key `{self.checkpoint_key}` must be an integer, but found value `{val}` of type `{type(val)}`"
            checkpoint_dir = (
                self.checkpoint_dir / f"latest-{self.checkpoint_key}{val:012}"
            )
            self.checkpoint(str(checkpoint_dir))
            if self.last_checkpoint is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.move(str(self.last_checkpoint), tmpdir)
            self.last_checkpoint = checkpoint_dir
            # TODO: persistent checkpoints


def apply_schedules(state, config, schedules):
    for field_name, schedule in schedules.items():
        if isinstance(schedule, Schedule):
            schedule.update_value(config, state)
        else:
            assert isinstance(schedule, dict)
            apply_schedules(state, getattr(config, field_name), schedule)


def qualified_name(clz):
    if clz.__module__ == "builtin":
        return clz.__name__
    elif not hasattr(clz, "__module__") or not hasattr(clz, "__name__"):
        return repr(clz)
    else:
        return f"{clz.__module__}.{clz.__name__}"


def _typecheck(name, value, typ):
    if not isinstance(value, typ):
        raise TypeError(
            f"{name} has type {qualified_name(typ)}, but received value {value} of type {qualified_name(value.__class__)}."
        )


def checkpoint(state, target_path: Path):
    builder, blobs = _checkpoint(state, target_path)
    with open(target_path / "state.yaml", "w") as f:
        yaml.dump(builder, f)
    for path, blob in blobs.items():
        with open(target_path / path, "wb") as f:
            f.write(blob)


def find_latest_checkpoint(dir: Path) -> Optional[Path]:
    # TODO: error handling
    # Check that dir exists
    if not dir.exists():
        return None
    latest = None
    latest_dir = None
    for d in dir.iterdir():
        if d.is_dir() and len(d.name) >= 12:
            if latest is None or int(d.name[-12:]) > latest:
                latest = int(d.name[-12:])
                latest_dir = d
    return latest_dir


def _checkpoint(state, target_path) -> Tuple[Any, Dict[str, bytes]]:
    builder = {}
    blobs = {}
    for field_name, field_clz in state.__annotations__.items():
        value = getattr(state, field_name)
        if is_dataclass(field_clz):
            value, _blobs = _checkpoint(value, target_path)
            for path, blob in _blobs:
                blobs[os.path.join(field_name, path)] = blob
        elif field_clz in [int, float, str, bool]:
            pass
        elif hasattr(field_clz, "__args__") and (
            (len(field_clz.__args__) == 1 and field_clz == List[field_clz.__args__])
            or (len(field_clz.__args__) == 2 and field_clz == Dict[field_clz.__args__])
        ):
            # TODO: recurse
            pass
        elif isinstance(field_clz, EnumMeta):
            value = value.name
        elif (
            hasattr(field_clz, "__args__")
            and len(field_clz.__args__) == 1
            and field_clz == Blob[field_clz.__args__]
        ):
            # TODO: use sane serialization library
            import dill

            data = dill.dumps(value._inner)
            blobs[field_name] = data
            value = "<BLOB>"
        else:
            raise TypeError(f"Unexpected type {field_clz}")
        builder[field_name] = value
    return builder, blobs


def asdict(x, schedules: Optional[Dict[str, Any]] = None):
    if schedules is None:
        schedules = {}
    result = {}
    for field_name, field_clz in x.__annotations__.items():
        if field_name in schedules and isinstance(schedules[field_name], Schedule):
            result[field_name] = schedules[field_name].unparsed
            continue
        value = getattr(x, field_name)
        if is_dataclass(field_clz):
            result[field_name] = asdict(value, schedules.get(field_name))
        elif field_clz in [int, float, str, bool]:
            result[field_name] = value
        elif hasattr(field_clz, "__args__") and (
            field_clz == List[field_clz.__args__]
            or field_clz == Dict[field_clz.__args__]
        ):
            # TODO: recurse
            result[field_name] = value
        elif isinstance(field_clz, EnumMeta):
            result[field_name] = value.name
        else:
            raise TypeError(f"Unexpected type {field_clz}")
    return result


def load_file(clz: Type[T], path: str, overrides: List[str]) -> T:
    return _load_file_and_schedules(clz, path, overrides)[0]


def _load_file_and_schedules(clz: Type[T], path: str, overrides: List[str]) -> T:
    path = Path(path)
    if not is_dataclass(clz):
        raise TypeError(f"{clz.__module__}.{clz.__name__} must be a dataclass")
    file = open(path)
    values = yaml.full_load(file)
    for override in overrides:
        key, str_val = override.split("=")
        fpath = key.split(".")
        _values = values
        _clz = clz
        for segment in fpath[:-1]:
            _values = _values[segment]
            _clz = _clz.__annotations__[segment]
        # TODO: missing types
        if (_clz == int or _clz == float) and "@" in str_val:
            val = str_val
        elif _clz == int:
            val = _parse_int(str_val)
        elif clz == float or _clz == bool:
            val = _clz(str_val)
        else:
            val = str_val
        _values[fpath[-1]] = val
    return _parse(clz, values, path.absolute().parent)


def _parse(
    clz: Type[T], values: Dict[str, Any], path: Path
) -> Tuple[T, Dict[str, Any]]:
    kwargs = {}
    remaining_fields = set(clz.__annotations__.keys())
    schedules = {}
    for field_name, value in values.items():
        if field_name not in remaining_fields:
            raise TypeError(
                f"{clz.__module__}.{clz.__name__} has no attribute {field_name}."
            )
        else:
            remaining_fields.remove(field_name)
        field_clz = clz.__annotations__[field_name]
        if field_clz == float:
            if isinstance(value, str):
                if "@" in value:
                    schedule = _parse_schedule(value)

                    # TODO: surely there's a better way?
                    def _capture(field_name, schedule):
                        def update(self, state):
                            x = getattr(state, schedule.xname)
                            value = schedule.get_value(x)
                            setattr(self, field_name, value)

                        return update

                    schedules[field_name] = Schedule(
                        _capture(field_name, schedule), value
                    )
                    value = schedule.get_value(0.0)
                else:
                    value = float(value)
            if isinstance(value, int):
                value = float(value)
            _typecheck(field_name, value, float)
        elif field_clz == int:
            if isinstance(value, str):
                if "@" in value:
                    schedule = _parse_schedule(value)

                    def _capture(field_name, schedule):
                        def update(self, state):
                            x = getattr(state, schedule.xname)
                            value = int(schedule.get_value(x))
                            setattr(self, field_name, value)

                        return update

                    schedules[field_name] = Schedule(
                        _capture(field_name, schedule), value
                    )
                    value = int(schedule.get_value(0))
                else:
                    parsed = _parse_int(value)
                    if parsed is not None:
                        value = parsed
            _typecheck(field_name, value, int)
        elif field_clz == bool:
            _typecheck(field_name, value, bool)
        elif field_clz == str:
            _typecheck(field_name, value, str)
        elif (
            hasattr(field_clz, "__args__")
            and len(field_clz.__args__) == 1
            and field_clz == List[field_clz.__args__]
        ):
            _typecheck(field_name, value, list)
            # TODO: nested types, dataclasses etc.
            for i, item in enumerate(value):
                _typecheck(f"{field_name}[{i}]", item, field_clz.__args__[0])
        elif (
            hasattr(field_clz, "__args__")
            and len(field_clz.__args__) == 2
            and field_clz == Dict[field_clz.__args__]
        ):
            _typecheck(field_name, value, dict)
            # TODO: recurse
        elif (
            hasattr(field_clz, "__args__")
            and len(field_clz.__args__) == 1
            and field_clz == Blob[field_clz.__args__]
        ):
            assert value == "<BLOB>", f"{value} != <BLOB>"
            value = Blob(path / field_name)
        elif is_dataclass(field_clz):
            value, inner_schedules = _parse(field_clz, value, path / field_name)
            schedules[field_name] = inner_schedules
        elif isinstance(field_clz, EnumMeta):
            value = field_clz(value)
        else:
            raise TypeError(
                f"Field {clz.__module__}.{clz.__name__}.{field_name} has unsupported type {qualified_name(field_clz)}."
            )
        kwargs[field_name] = value

    try:
        instance = clz(**kwargs)
        return instance, schedules
    except TypeError as e:
        raise TypeError(f"Failed to initialize {clz.__module__}.{clz.__name__}: {e}")


@dataclass
class Schedule:
    update_value: Callable[[Any], None]
    unparsed: str


@dataclass
class ScheduleSegment:
    start: float
    end: float
    initial_value: float
    final_value: float
    interpolator: Callable[[float], float]


@dataclass
class PiecewiseFunction:
    segments: List[ScheduleSegment]
    xname: str

    def get_value(self, x: float) -> float:
        segment = None
        for s in self.segments:
            segment = s
            if x < s.end:
                break
        rescaled = (x - segment.start) / (segment.end - segment.start)
        if rescaled < 0:
            rescaled = 0
        elif rescaled > 1:
            rescaled = 1

        w = segment.interpolator(rescaled)
        return w * segment.initial_value + (1 - w) * segment.final_value


INTERPOLATORS = {
    "lin": lambda x: 1 - x,
    "cos": lambda x: math.cos(x * math.pi) * 0.5 + 0.5,
    "step": lambda _: 1,
}


def _parse_schedule(schedule: str) -> Callable[[float], float]:
    # Grammar
    #
    # rule := ident ": " point [join point]
    # join := " " | " " ident " "
    # point := num "@" num
    # ident := "lin" | "cos" | "exp" | "step" | "quad" | "poly(" num ")"
    # num := `float` | `int` | `path`
    # path := ident [ "." path]
    #
    # Example:
    # "0.3@step=0 lin 0.15@150e6 0.1@200e6 0.01@250e6"

    # TODO: lots of checks and helpful error messages
    parts = schedule.split(" ")
    interpolator = INTERPOLATORS["lin"]
    last_x, last_y = None, None
    xname = parts[0][:-1]
    segments = []
    for part in parts[1:]:
        if "@" in part:
            y, x = part.split("@")
            y, x = float(y), float(x)
            if last_x is not None:
                segments.append(
                    ScheduleSegment(
                        start=last_x,
                        initial_value=last_y,
                        end=x,
                        final_value=y,
                        interpolator=interpolator,
                    )
                )
            last_x, last_y = x, y
        else:
            interpolator = INTERPOLATORS[part]
    return PiecewiseFunction(segments, xname)


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


class Objective(Enum):
    ALLIED_WEALTH = "ALLIED_WEALTH"
    DISTANCE_TO_CRYSTAL = "DISTANCE_TO_CRYSTAL"
    DISTANCE_TO_ORIGIN = "DISTANCE_TO_ORIGIN"
    DISTANCE_TO_1000_500 = "DISTANCE_TO_1000_500"
    ARENA_TINY = "ARENA_TINY"
    ARENA_TINY_2V2 = "ARENA_TINY_2V2"
    ARENA_MEDIUM = "ARENA_MEDIUM"
    ARENA_MEDIUM_LARGE_MS = "ARENA_MEDIUM_LARGE_MS"
    ARENA = "ARENA"
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"
    SMOL_STANDARD = "SMOL_STANDARD"
    MICRO_PRACTICE = "MICRO_PRACTICE"
    SCOUT = "SCOUT"

    def vs(self):
        if (
            self == Objective.ALLIED_WEALTH
            or self == Objective.DISTANCE_TO_CRYSTAL
            or self == Objective.DISTANCE_TO_ORIGIN
            or self == Objective.DISTANCE_TO_1000_500
            or self == Objective.SCOUT
        ):
            return False
        elif (
            self == Objective.ARENA_TINY
            or self == Objective.ARENA_TINY_2V2
            or self == Objective.ARENA_MEDIUM
            or self == Objective.ARENA
            or self == Objective.STANDARD
            or self == Objective.ENHANCED
            or self == Objective.SMOL_STANDARD
            or self == Objective.MICRO_PRACTICE
            or self == Objective.ARENA_MEDIUM_LARGE_MS
        ):
            return True
        else:
            raise Exception(f"Objective.vs not implemented for {self}")

    def naction(self):
        return 8 + len(self.extra_builds())

    def builds(self):
        b = self.extra_builds()
        b.append((0, 1, 0, 0, 0, 0))
        return b

    def extra_builds(self):
        # [storageModules, missileBatteries, constructors, engines, shieldGenerators]
        if self == Objective.ARENA:
            return [(1, 0, 1, 0, 0), (0, 2, 0, 0, 0), (0, 1, 0, 0, 1)]
        elif self == Objective.SMOL_STANDARD or self == Objective.STANDARD:
            return [
                (1, 0, 1, 0, 0),
                (0, 2, 0, 0, 0),
                (0, 1, 0, 0, 1),
                (0, 3, 0, 0, 1),
                (0, 2, 0, 0, 2),
                (2, 1, 1, 0, 0),
                (2, 0, 2, 0, 0),
                (2, 0, 1, 1, 0),
                (0, 2, 0, 1, 1),
                (1, 0, 0, 0, 0),
            ]
        elif self == Objective.ENHANCED:
            return [
                # [s, m, c, e, p, l]
                (1, 0, 0, 0, 0, 0),  # 1s
                (1, 0, 1, 0, 0, 0),  # 1s1c
                (0, 1, 0, 0, 1, 0),  # 1m1p
                (0, 0, 0, 0, 0, 2),  # 2l
                (0, 2, 0, 2, 0, 0),  # 2m2e
                (0, 1, 0, 2, 1, 0),  # 1m1p2e
                (0, 2, 0, 1, 1, 0),  # 2m1e1p
                (0, 0, 0, 1, 0, 3),  # 1e3l
                (2, 0, 1, 1, 0, 0),  # 2s1c1e
                (0, 4, 0, 3, 3, 0),  # 4m3e3p
                (0, 0, 0, 4, 1, 5),  # 4e1p5l
            ]
        else:
            return []


@dataclass
class Blob(Generic[T]):
    _inner: Union[T, Path]

    def get(self) -> T:
        if isinstance(self._inner, Path):
            import pickle

            with open(self._inner, "rb") as f:
                self._inner = pickle.load(f)
        return self._inner

    def set(self, value: T):
        self._inner = value


@dataclass
class OptimizerConfig:
    # Optimizer ("SGD" or "RMSProp" or "Adam")
    optimizer_type: str = "Adam"
    # Learning rate
    lr: float = 0.0003
    # Momentum
    momentum: float = 0.9
    # Weight decay
    weight_decay: float = 0.0001
    # Batch size during optimization
    batch_size: int = 2048
    # Micro batch size for gradient accumulation
    micro_batch_size: int = 2048
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
    # Exponentially moving averages of model weights
    weights_ema: List[float] = field(default_factory=list)
    # [0.99, 0.997, 0.999, 0.9997, 0.9999]

    # TODO: hack to load old checkpoints, find good solution for backwards compatibility
    batches_per_update: int = -1
    bs: int = -1


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
    eval_frequency: int = int(1e5)
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
    objective: Objective = Objective.ARENA_TINY_2V2
    action_delay: int = 0
    use_action_masks: bool = True
    task_hardness: float = 0
    # Max length of games, or default game length for map if 0.
    max_game_length: int = 0
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
    mothership_damage_scale: float = 4.0
    enforce_unit_cap: bool = False
    unit_cap: int = 20

    # Set by ppo
    build_variety_bonus: float = 0.0
    # Set by adr
    cost_variance: float = 0.0


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
    # Linearly increase task difficulty/map size
    linear_hardness: bool = False
    # Maxiumum map area
    max_hardness: float = 150
    # Number of timesteps steps after which hardness starts to increase
    hardness_offset: float = 1e6
    variety: float = 0.7


@dataclass
class Config:
    optimizer: OptimizerConfig
    eval: EvalConfig
    ppo: PPOConfig
    task: TaskConfig
    adr: AdrConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    wandb: bool = True

    @property
    def rosteps(self):
        return self.ppo.num_envs * self.ppo.seq_rosteps

    def validate(self):
        assert self.rosteps % self.optimizer.batch_size == 0
        assert self.eval.eval_envs % 4 == 0
