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
from dataclasses import dataclass

from hyperstate.schema.versioned import (
    VersionedSerializer,
    VersionedDeserializer,
)
from hyperstate.serde import (
    Deserializer,
    Serializer,
    asdict,
    dump,
    load,
)
from .lazy import LazyDeserializer, LazySerializer

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
        initial_config: Union[str, Path],
        checkpoint_dir: Optional[Union[str, Path]] = None,
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
        if isinstance(initial_config, str):
            initial_config = Path(initial_config)
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)

        checkpoint = None
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            checkpoint = find_latest_checkpoint(checkpoint_dir)
            if checkpoint is not None:
                print(f"Resuming from checkpoint {checkpoint}")
                initial_config = checkpoint
        else:
            self.checkpoint_dir = None

        if os.path.isdir(initial_config):
            config_path = initial_config / "config.ron"
            state_path = initial_config / "state.ron"
        else:
            config_path = initial_config
            state_path = None

        try:
            self.config, self.schedules = _typed_load(
                config_clz,
                config_path,
                overrides=overrides or [],
                allow_missing_version=state_path is not None,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_path}: {e}") from e
        if state_path is None:
            self.state = self.initial_state()
        else:
            try:
                self.state = _typed_load(state_clz, state_path, config=self.config)[0]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load state from {state_path}: {e}"
                ) from e

    @abstractmethod
    def initial_state(self) -> S:
        pass

    def checkpoint_key(self):
        return "step"

    def checkpoint(self, target_dir: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "checkpoint"
            p.mkdir()
            _typed_dump(self.config, p / "config.ron", self.schedules)
            _typed_dump(self.state, p / "state.ron")
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

    def config_dict(self):
        return asdict(self.config, serializers=[VersionedSerializer()])


def _apply_schedules(state, config, schedules: Dict[str, Any]):
    for field_name, schedule in schedules.items():
        if isinstance(schedule, Schedule):
            schedule.update_value(config, state)
        else:
            assert isinstance(schedule, dict)
            _apply_schedules(state, getattr(config, field_name), schedule)


def _typed_dump(
    obj, path: Optional[Path], schedules: Optional[Dict[str, Any]] = None
) -> None:
    serializers = []
    lazy_serializer = LazySerializer()
    serializers = [lazy_serializer, VersionedSerializer()]
    if schedules is not None:
        serializers.append(ScheduleSerializer(schedules))
    result = dump(obj, path, serializers=serializers)
    if path is not None:
        for blobpath, blob in lazy_serializer.blobs.items():
            with open(path.parent / blobpath, "wb") as f:
                f.write(blob)
    return result


def typed_dump(obj, path: Optional[Path] = None) -> Union[None, str]:
    return _typed_dump(obj, path)


def _typed_load(
    clz: Type[T],
    source: Union[str, Path],
    overrides: Optional[List[str]] = None,
    config: Optional[Any] = None,
    allow_missing_version: bool = False,
) -> Tuple[T, Dict[str, Any]]:
    if overrides is not None:
        deserializers = [OverridesDeserializer(overrides)]
    else:
        deserializers = []
    schedules = ScheduleDeserializer()
    deserializers.append(schedules)
    deserializers.append(VersionedDeserializer(allow_missing_version))
    lazy = None
    if isinstance(source, Path):
        lazy = LazyDeserializer(config, source.absolute().parent)
        deserializers.append(lazy)
    value = load(clz, source, deserializers=deserializers)
    if lazy is not None and len(lazy.lazy_fields) > 0:
        value._unloaded_lazy_fields = lazy.lazy_fields
    return value, schedules.schedules


def typed_load(
    clz: Type[T], source: Union[str, Path], overrides: Optional[List[str]] = None,
) -> T:
    return _typed_load(clz, source, overrides)[0]


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


@dataclass
class OverridesDeserializer(Deserializer):
    overrides: List[str]
    applied_overrides: bool = False

    def deserialize(self, clz: Type[T], value: Any, path: str,) -> Tuple[T, bool, bool]:
        if self.applied_overrides:
            return None, False, False
        for override in self.overrides:
            key, str_val = override.split("=")
            val = pyron.load(str_val)
            fpath = key.split(".")
            _value = value
            for segment in fpath[:-1]:
                if segment not in _value:
                    _value[segment] = {}
                _value = _value[segment]
            _value[fpath[-1]] = val
        self.applied_overrides = True
        return value, True, False


class ScheduleDeserializer(Deserializer):
    def __init__(self):
        self.schedules = {}

    def deserialize(self, clz: Type[T], value: Any, path: str,) -> Tuple[T, bool, bool]:
        if (clz == int or clz == float) and isinstance(value, str) and "@" in value:
            schedule = _parse_schedule(value)
            field_name = path.split(".")[-1]

            def update(self, state):
                x = getattr(state, schedule.xname)
                value = schedule.get_value(x)
                setattr(self, field_name, clz(value))

            self.schedules[path] = Schedule(update, value)
            value = schedule.get_value(0.0)
            return clz(value), True, False
        return None, False, False


@dataclass
class ScheduleSerializer(Serializer):
    schedules: Dict[str, Schedule]

    def serialize(self, value: Any, path: str, namedtuples: bool) -> Tuple[Any, bool]:
        if path in self.schedules:
            return self.schedules[path].unparsed, True
        return None, False


def _dict_to_cpu(x: Any) -> Dict[str, Any]:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    elif isinstance(x, dict):
        return {k: _dict_to_cpu(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_dict_to_cpu(v) for v in x]
    else:
        return x
