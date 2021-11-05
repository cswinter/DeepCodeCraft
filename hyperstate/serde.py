from abc import ABC, abstractmethod
from collections import namedtuple
from enum import Enum, EnumMeta
from os import name
from pathlib import Path
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
from dataclasses import MISSING, is_dataclass

import pyron

T = TypeVar("T")


class Serializer(ABC):
    @abstractmethod
    def serialize(self, value: Any, path: str, named_tuples: bool,) -> Tuple[Any, bool]:
        pass

    def modify_dataclass_attrs(self, value: Any, attrs: Dict[str, Any], path: str):
        pass


class Deserializer(ABC):
    @abstractmethod
    def deserialize(self, clz: Type[T], value: Any, path: str,) -> Tuple[T, bool, bool]:
        pass


def asdict(
    value,
    named_tuples: bool = False,
    serializers: Optional[List[Serializer]] = None,
    path: str = "",
) -> Any:
    if serializers is None:
        serializers = []

    for serializer in serializers:
        _value, _ok = serializer.serialize(value, path, named_tuples)
        if _ok:
            value = _value
    if is_dataclass(value):
        attrs = {
            field_name: asdict(
                value=getattr(value, field_name),
                named_tuples=named_tuples,
                serializers=serializers,
                path=field_name if path == "" else f"{path}.{field_name}",
            )
            for field_name in value.__dataclass_fields__
        }
        for serializer in serializers:
            serializer.modify_dataclass_attrs(value, attrs, path)
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
        or value is None
    ):
        return value
    else:
        raise TypeError(f"Can't serialize value {value} of type {type(value)}")


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

    ret = False
    for deserializer in deserializers:
        _value, ok, _ret = deserializer.deserialize(clz, value, path)
        if ok:
            value = _value
        ret = ret or _ret
    if ret:
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
        remaining_fields = set(clz.__dataclass_fields__.keys())
        for field_name, v in value.items():
            field = clz.__dataclass_fields__.get(field_name)
            if field is None:
                raise TypeError(
                    f"{clz.__module__}.{clz.__name__} has no attribute {field_name}."
                )
            remaining_fields.remove(field_name)
            kwargs[field_name] = from_dict(
                clz=field.type,
                value=v,
                deserializers=deserializers,
                path=f"{path}.{field_name}" if path else field_name,
            )
        for field_name in remaining_fields:
            field = clz.__dataclass_fields__.get(field_name)
            if (
                field.default is MISSING
                and field.default_factory is MISSING
                and is_dataclass(field.type)
            ):
                kwargs[field_name] = field.type()
        try:
            instance = clz(**kwargs)
            return instance
        except TypeError as e:
            raise TypeError(f"Failed to initialize {path}: {e}")
    elif isinstance(clz, EnumMeta):
        return clz(value)
    # elif isinstance(value, clz):
    #    return value
    raise TypeError(
        f"Failed to deserialize {path}: {value} is not a {_qualified_name(clz)}"
    )


def load(
    clz: Type[T],
    source: Union[str, Path],
    deserializers: Optional[List[Deserializer]] = None,
) -> Tuple[T, Dict[str, Any]]:
    if deserializers is None:
        deserializers = []
    if isinstance(source, str):
        state_dict = pyron.load(source)
    elif isinstance(source, Path):
        with open(source, "r") as f:
            state_dict = pyron.load(f.read())
    else:
        raise ValueError(f"source must be a `str` or `Path`, but found {source}")
    return from_dict(clz, state_dict, deserializers)


def dump(
    obj, path: Optional[Path] = None, serializers: Optional[List[Serializer]] = None
) -> None:
    if serializers is None:
        serializers = []
    state_dict = asdict(obj, named_tuples=True, serializers=serializers)
    serialized = pyron.to_string(state_dict)
    if path:
        with open(path, "w") as f:
            f.write(serialized)
    else:
        return serialized


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
