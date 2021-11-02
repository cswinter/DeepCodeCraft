from collections import namedtuple
import os
import shutil
from abc import ABC, abstractmethod
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
import inspect
from dataclasses import dataclass, field, is_dataclass

from hyperstate.schema.versioned import Versioned
from hyperstate.serde import (
    Deserializer,
    Serializer,
    _asdict,
    from_dict,
)
from .lazy import LazyField

import torch
import pyron

from hyperstate.schedule import Schedule, _parse_schedule

C = TypeVar("C")
S = TypeVar("S")
T = TypeVar("T")


class HyperState(ABC, Generic[C, S]):
    def __init__(
        self,
        config_clz: Type[C],
        state_clz: Type[S],
        initial_config: str,
        checkpoint_dir: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> None:
        """
        :param config_clz: The type of the config object.
        :param state_clz: The type of the state object.
        :param initial_config: Path to a config file or checkpoint.
        :param checkpoint_dir: Directory to store checkpoints. If the directory contains a valid checkpoint, the latest checkpoint will be loaded and `initial_config` will be ignored.
        :param overrides: A list of overrides to apply to the config. (Example: ["optimizer.lr=0.1"])
        """
        self.config_clz = config_clz
        self.state_clz = state_clz
        self._last_checkpoint: Optional[Path] = None

        if overrides is None:
            overrides = []

        checkpoint = None
        if checkpoint_dir is not None:
            checkpoint = find_latest_checkpoint(checkpoint_dir)
            if checkpoint is not None:
                print(f"Resuming from checkpoint {checkpoint}")
                initial_config = checkpoint

        path = Path(initial_config)
        if os.path.isdir(path):
            config_path = path / "config.ron"
            state_path = path / "state.ron"
        else:
            config_path = path
            state_path = None

        try:
            config, schedules = _load_file_and_schedules(
                config_clz,
                config_path,
                overrides=overrides,
                allow_missing_version=state_path is not None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e
        self.config: C = config
        self.schedules = schedules

        # TODO: hack
        config = self.config
        config.obs.feat_rule_msdm = config.task.rule_rng_fraction > 0 or config.task.adr
        config.obs.feat_rule_costs = config.task.rule_cost_rng > 0 or config.task.adr
        config.obs.num_builds = len(config.task.objective.builds())

        if state_path is None:
            self.state = self.initial_state()
        else:
            self.state, _ = _load_file_and_schedules(
                state_clz, state_path, overrides=[], config=config,
            )

        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = None

        _apply_schedules(self.state, config, self.schedules)

    @abstractmethod
    def initial_state(self, config: C) -> S:
        pass

    def checkpoint_key(self):
        return "step"

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
        _apply_schedules(self.state, self.config, self.schedules)
        if self.checkpoint_dir is not None:
            val = getattr(self.state, self.checkpoint_key())
            assert isinstance(
                val, int
            ), f"checkpoint key `{self.checkpoint_key()}` must be an integer, but found value `{val}` of type `{type(val)}`"
            checkpoint_dir = (
                self.checkpoint_dir / f"latest-{self.checkpoint_key()}{val:012}"
            )
            self.checkpoint(str(checkpoint_dir))
            if self._last_checkpoint is not None:
                with tempfile.TemporaryDirectory() as tmpdir:
                    shutil.move(str(self._last_checkpoint), tmpdir)
            self._last_checkpoint = checkpoint_dir
            # TODO: persistent checkpoints


def _apply_schedules(state, config, schedules: Dict[str, Any]):
    for field_name, schedule in schedules.items():
        if isinstance(schedule, Schedule):
            schedule.update_value(config, state)
        else:
            assert isinstance(schedule, dict)
            _apply_schedules(state, getattr(config, field_name), schedule)


def checkpoint(state, target_path: Path):
    builder, blobs = _checkpoint(state, target_path)
    with open(target_path / "state.ron", "w") as f:
        serialized = pyron.to_string(builder)
        f.write(serialized)
    for path, blob in blobs.items():
        with open(target_path / path, "wb") as f:
            f.write(blob)


def _load_file_and_schedules(
    clz: Type[T],
    path: str,
    overrides: List[str],
    config: Optional[Any] = None,
    allow_missing_version: bool = False,
) -> T:
    path = Path(path)
    if not is_dataclass(clz):
        raise TypeError(f"{clz.__module__}.{clz.__name__} must be a dataclass")
    with open(path, "r") as f:
        content = f.read()
        state_dict = pyron.load(content)
    for override in overrides:
        key, str_val = override.split("=")
        fpath = key.split(".")
        _values = state_dict
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
    return _parse(
        clz, state_dict, path.absolute().parent, config, allow_missing_version
    )


def _parse(
    clz: Type[T],
    values: Dict[str, Any],
    path: Path,
    config: Optional[Any] = None,
    allow_missing_version: bool = False,
) -> Tuple[T, Dict[str, Any]]:
    schedules = ScheduleDeserializer()
    lazy = LazyDeserializer(config, path)
    if issubclass(clz, Versioned):
        version = values.pop("version", None)
        if version is None:
            if allow_missing_version:
                version = 0
            else:
                raise ValueError(f"Config files are required to set `version` field.")
        values = clz._apply_upgrades(values, version)
    value = from_dict(clz, values, [schedules, lazy])
    if len(lazy.lazy_fields) > 0:
        value._unloaded_lazy_fields = lazy.lazy_fields
    return value, schedules.schedules


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


class ScheduleDeserializer(Deserializer):
    def __init__(self):
        self.schedules = {}

    def deserialize(self, clz: Type[T], value: Any, path: str) -> Tuple[T, bool]:
        if (clz == int or clz == float) and isinstance(value, str) and "@" in value:
            schedule = _parse_schedule(value)
            field_name = path.split(".")[-1]

            def update(self, state):
                x = getattr(state, schedule.xname)
                value = schedule.get_value(x)
                setattr(self, field_name, clz(value))

            self.schedules[path] = Schedule(update, value)
            value = schedule.get_value(0.0)
            return clz(value), True
        return None, False


@dataclass
class LazyDeserializer(Deserializer, Generic[C]):
    config: C
    path: Path
    lazy_fields: Dict[str, Tuple[C, str, bool]] = field(default_factory=dict)

    def deserialize(self, clz: Type[T], value: Any, path: str) -> Tuple[T, bool]:
        if inspect.isclass(clz) and issubclass(clz, LazyField):
            assert value == "<BLOB>" or value == "<blob:msgpack>"
            filepath = path.replace(".", "/").replace("[", "/").replace("]", "")
            self.lazy_fields[path] = (
                self.config,
                self.path / filepath,
                value == "<BLOB>",
            )
            return None, True
        return None, False


@dataclass
class VersionedSerializer(Serializer):
    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
        recurse: Callable[[Any, str], Any],
    ) -> Tuple[Any, bool]:
        if isinstance(value, Versioned):
            attrs = {
                field_name: recurse(
                    getattr(value, field_name),
                    path if path == "" else f"{path}.{field_name}",
                )
                for field_name in value.__dataclass_fields__
            }
            attrs["version"] = value.__class__.version()
            if named_tuples:
                return namedtuple(value.__class__.__name__, attrs.keys())(**attrs), True
            else:
                return attrs, True
        return None, False


@dataclass
class ScheduleSerializer(Serializer):
    schedules: Dict[str, Schedule]

    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
        recurse: Callable[[Any, str], Any],
    ) -> Tuple[Any, bool]:
        if path in self.schedules:
            return self.schedules[path].unparsed, True
        return None, False


def asdict(
    x,
    schedules: Optional[Dict[str, Any]] = None,
    named_tuples: bool = False,
    path: str = "",
) -> Any:
    return _asdict(
        x, named_tuples, [ScheduleSerializer(schedules or {}), VersionedSerializer()]
    )


def _dict_to_cpu(x: Any) -> Dict[str, Any]:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, dict):
        return {k: _dict_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_dict_to_cpu(v) for v in x]
    else:
        return x


@dataclass
class LazySerializer(Serializer):
    blobs: Dict[str, bytes] = field(default_factory=dict)

    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
        recurse: Callable[[Any, str], Any],
    ) -> Tuple[Any, bool]:
        if isinstance(value, LazyField):
            import dill

            self.blobs[path] = dill.dumps(value.state_dict())
            return "<BLOB>", True
            # TODO: make msgpack work with pytorch tensors
            # state_dict = _dict_to_cpu(value.state_dict())
            # blobs[field_name] = msgpack.packb(state_dict, default=msgpack_numpy.encode)
            # value = "<blob:msgpack>"
        return None, False


def _checkpoint(state, target_path) -> Tuple[Any, Dict[str, bytes]]:
    lazy_serializer = LazySerializer()
    state_dict = _asdict(state, named_tuples=True, serializers=[lazy_serializer])
    return state_dict, lazy_serializer.blobs


def _parse_int(s: str):
    try:
        f = float(s)
        i = int(f)

        return i if f == i else None
    except:
        return None
