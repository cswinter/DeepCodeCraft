from abc import ABC, abstractclassmethod, abstractmethod
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    Tuple,
    Type,
    Generic,
    TypeVar,
)
from dataclasses import dataclass, field
from pathlib import Path
import msgpack
import msgpack_numpy

from hyperstate.serde import Serializer, Deserializer

T = TypeVar("T")
C = TypeVar("C")

# TODO: blob, lazy, and serializable should be orthogonal
class Serializable(ABC):
    @abstractmethod
    def serialize(self) -> Any:
        pass

    @abstractclassmethod
    def deserialize(clz: Type[T], state_dict: Any, config: Any, state: Any) -> T:
        pass


class Lazy:
    def __init__(self):
        self._unloaded_lazy_fields = {}

    def __getattribute__(self, name: str) -> Any:
        try:
            unloaded = super(Lazy, self).__getattribute__("_unloaded_lazy_fields")
            if name in unloaded:
                ser_clz, config, path, legacy_pickle = unloaded[name]
                # clz = self.__annotations__[name]
                with open(path, "rb") as f:
                    # TODO: deprecate
                    if legacy_pickle:
                        import pickle

                        state_dict = pickle.load(f)
                    else:
                        # TODO: this doesn't work for tensors :( need custom encoder/decoder that converts numpy arrays back into tensors?
                        state_dict = msgpack.unpack(f, object_hook=msgpack_numpy.decode)
                # TODO: recursion check
                value = ser_clz.deserialize(state_dict, config, self)
                self.__setattr__(name, value)
                del unloaded[name]
        except AttributeError:
            pass
        return super(Lazy, self).__getattribute__(name)


@dataclass
class LazyDeserializer(Deserializer, Generic[C]):
    config: C
    path: Path
    lazy_fields: Dict[str, Tuple[C, str, bool]] = field(default_factory=dict)

    def deserialize(self, clz: Type[T], value: Any, path: str,) -> Tuple[T, bool, bool]:
        if inspect.isclass(clz) and issubclass(clz, Serializable):
            assert value == "<BLOB>" or value == "<blob:msgpack>"
            filepath = path.replace(".", "/").replace("[", "/").replace("]", "")
            self.lazy_fields[path] = (
                clz,
                self.config,
                self.path / filepath,
                value == "<BLOB>",
            )
            return None, True, True
        return None, False, False


@dataclass
class LazySerializer(Serializer):
    blobs: Dict[str, bytes] = field(default_factory=dict)

    def serialize(self, value: Any, path: str, named_tuples: bool,) -> Tuple[Any, bool]:
        if isinstance(value, Serializable):
            import dill

            self.blobs[path] = dill.dumps(value.serialize())
            return "<BLOB>", True
            # TODO: make msgpack work with pytorch tensors
            # state_dict = _dict_to_cpu(value.state_dict())
            # blobs[field_name] = msgpack.packb(state_dict, default=msgpack_numpy.encode)
            # value = "<blob:msgpack>"
        return None, False


def blob(clz: Type[T], mixin: Type[Serializable]) -> Type[T]:
    class Blob(mixin, clz):
        pass

    return Blob
