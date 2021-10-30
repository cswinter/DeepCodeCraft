import os
import shutil

from collections import namedtuple
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
import pyron
from hyperstate.schedule import Schedule, _parse_schedule

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
            config_path = path / "config.ron"
            state_path = path / "state.ron"
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
            with open(p / "config.ron", "w") as f:
                serialized = pyron.to_string(
                    asdict(self.config, schedules=self.schedules, named_tuples=True)
                )
                f.write(serialized)
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


def apply_schedules(state, config, schedules: Dict[str, Any]):
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
    with open(target_path / "state.ron", "w") as f:
        serialized = pyron.to_string(builder)
        f.write(serialized)
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
        if is_optional(field_clz):
            if value is None:
                builder[field_name] = value
                continue
            field_clz = field_clz.__args__[0]
        if is_dataclass(field_clz):
            value, _blobs = _checkpoint(value, target_path)
            value = namedtuple(field_clz.__name__, value.keys())(**value)
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


def asdict(x, schedules: Optional[Dict[str, Any]] = None, named_tuples: bool = False):
    if schedules is None:
        schedules = {}
    result = {}
    for field_name, field_clz in x.__annotations__.items():
        if field_name in schedules and isinstance(schedules[field_name], Schedule):
            result[field_name] = schedules[field_name].unparsed
            continue
        value = getattr(x, field_name)

        if is_optional(field_clz):
            if value is None:
                result[field_name] = value
                continue
            field_clz = field_clz.__args__[0]
        if is_dataclass(field_clz):
            attrs = asdict(value, schedules.get(field_name), named_tuples)
            if named_tuples:
                result[field_name] = namedtuple(field_clz.__name__, attrs.keys())(
                    **attrs
                )
            else:
                result[field_name] = attrs
        elif field_clz in [int, float, str, bool]:
            result[field_name] = value
        elif hasattr(field_clz, "__origin__") and (
            (field_clz.__origin__ is list and field_clz == List[field_clz.__args__])
            or (field_clz.__origin__ is dict and field_clz == Dict[field_clz.__args__])
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
    with open(path, "r") as f:
        content = f.read()
        values = pyron.load(content)
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
        is_opt = is_optional(field_clz)
        if is_opt and value is not None:
            field_clz = field_clz.__args__[0]

        if is_opt and value is None:
            pass
        elif field_clz == float:
            if isinstance(value, str):
                if "@" in value:
                    schedule = _parse_schedule(value)

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
                            value = schedule.get_value(x)
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
            if isinstance(value, float) and value == int(value):
                value = int(value)
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


# TODO: gah
def _parse_int(s: str):
    try:
        f = float(s)
        i = int(f)

        return i if f == i else None
    except:
        return None


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


def is_optional(clz):
    return (
        hasattr(clz, "__origin__")
        and clz.__origin__ is Union  # type: ignore
        and clz.__args__.__len__() == 2
        and clz.__args__[1] is type(None)
    )
