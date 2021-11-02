from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum, EnumMeta
from typing import (
    Callable,
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
from dataclasses import is_dataclass

T = TypeVar("T")


class Serializer(ABC):
    @abstractmethod
    def serialize(
        self,
        value: Any,
        path: str,
        named_tuples: bool,
        recurse: Callable[[Any, str], Any],
    ) -> Tuple[Any, bool]:
        pass


def _asdict(
    value,
    named_tuples: bool = False,
    serializers: Optional[List[Serializer]] = None,
    path: str = "",
) -> Any:
    if serializers is None:
        serializers = []

    def __asdict(value: Any, path: str):
        for serializer in serializers:
            result, ok = serializer.serialize(value, path, named_tuples, __asdict)
            if ok:
                return result
        if is_dataclass(value):
            attrs = {
                field_name: __asdict(
                    getattr(value, field_name),
                    path if path == "" else f"{path}.{field_name}",
                )
                for field_name in value.__dataclass_fields__
            }
            if named_tuples:
                return namedtuple(value.__class__.__name__, attrs.keys())(**attrs)
            else:
                return attrs
        elif isinstance(value, dict) or isinstance(value, list):
            # TODO: recurse
            return value
        elif isinstance(value, Enum):
            return value.name
        elif (
            isinstance(value, int)
            or isinstance(value, float)
            or isinstance(value, str)
            or isinstance(value, bool)
        ):
            return value
        else:
            raise TypeError(f"Unexpected value {value} of type {type(value)}")

    return __asdict(value, path)


class Deserializer(ABC):
    @abstractmethod
    def deserialize(self, clz: Type[T], value: Any, path: str) -> Tuple[T, bool]:
        pass


def from_dict(
    clz: Type[T],
    value: Any,
    deserializers: Optional[List[Deserializer]] = None,
    path: str = "",
) -> T:
    if deserializers is None:
        deserializers = []
    if is_optional(clz):
        if value is None:
            return None
        else:
            clz = clz.__args__[0]
    for deserializer in deserializers:
        _value, ok = deserializer.deserialize(clz, value, path)
        if ok:
            return _value
    if inspect.isclass(clz) and isinstance(value, clz):
        return value
    elif clz == float and isinstance(value, int):
        return int(value)
    elif clz == int and isinstance(value, float) and int(value) == value:
        return int(value)
    elif clz == float and isinstance(value, str):
        return float(value)
    elif clz == int and isinstance(value, str):
        f = float(value)
        if int(f) == f:
            return int(f)
        else:
            raise ValueError(f"Expected {path} to be an int, got {value}")
    elif (
        hasattr(clz, "__args__")
        and len(clz.__args__) == 1
        and clz == List[clz.__args__]
        and isinstance(value, list)
    ):
        # TODO: recurse
        return value
    elif (
        hasattr(clz, "__args__")
        and len(clz.__args__) == 2
        and clz == Dict[clz.__args__]
        and isinstance(value, dict)
    ):
        # TODO: recurse
        return value
    elif is_dataclass(clz):
        # TODO: better error
        assert isinstance(value, dict), f"{value} is not a dict"
        kwargs = {}
        for field_name, v in value.items():
            field = clz.__dataclass_fields__.get(field_name)
            if field is None:
                raise TypeError(
                    f"{clz.__module__}.{clz.__name__} has no attribute {field_name}."
                )
            kwargs[field_name] = from_dict(
                field.type,
                v,
                deserializers,
                f"{path}.{field_name}" if path else field_name,
            )
        try:
            instance = clz(**kwargs)
            return instance
        except TypeError as e:
            raise TypeError(f"Failed to initialize {path}: {e}")
    elif isinstance(clz, EnumMeta):
        return clz(value)
    raise TypeError(
        f"Failed to deserialize {path}: {value} is not a {_qualified_name(clz)}"
    )

def is_optional(clz):
    return (
        hasattr(clz, "__origin__")
        and clz.__origin__ is Union  # type: ignore
        and clz.__args__.__len__() == 2
        and clz.__args__[1] is type(None)
    )


def _qualified_name(clz):
    if clz.__module__ == "builtin":
        return clz.__name__
    elif not hasattr(clz, "__module__") or not hasattr(clz, "__name__"):
        return repr(clz)
    else:
        return f"{clz.__module__}.{clz.__name__}"
