from abc import ABC
from typing import (
    Any,
    Type,
    TypeVar,
)
import msgpack
import msgpack_numpy


class Lazy:
    def __init__(self):
        self._unloaded_lazy_fields = {}

    def __getattribute__(self, name: str) -> Any:
        try:
            unloaded = super(Lazy, self).__getattribute__("_unloaded_lazy_fields")
            if name in unloaded:
                config, path, legacy_pickle = unloaded[name]
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
                value = super(Lazy, self).__getattribute__(f"_load_{name}")(
                    config, state_dict
                )
                self.__setattr__(name, value)
                del unloaded[name]
        except AttributeError:
            pass
        return super(Lazy, self).__getattribute__(name)


class LazyField(ABC):
    pass


T = TypeVar("T")


def lazy(clz: Type[T]) -> Type[T]:
    class _LazyField(clz, LazyField):
        pass

    return _LazyField
