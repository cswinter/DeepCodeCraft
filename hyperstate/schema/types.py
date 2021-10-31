from typing import Type, TypeVar, Any
from enum import EnumMeta
import enum
import typing
from abc import ABC
from dataclasses import dataclass, is_dataclass
import dataclasses

import pyron

T = TypeVar("T")


class Type(ABC):
    pass


@dataclass(eq=True, frozen=True)
class Primitive(Type):
    type: str

    def __repr__(self) -> str:
        return self.type


@dataclass(eq=True, frozen=True)
class List(Type):
    inner: Type

    def __repr__(self) -> str:
        return f"List[{self.inner}]"


@dataclass(eq=True, frozen=True)
class Field:
    name: str
    type: Type
    default: Any
    has_default: bool


@dataclass(eq=True, frozen=True)
class Enum(Type):
    name: str
    variants: typing.Dict[str, typing.Union[str, int]]


# TODO: allow name to differ in equality
@dataclass(eq=True, frozen=True)
class Struct(Type):
    name: str
    fields: typing.Dict[str, Field]

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.fields.items())})"


@dataclass(eq=True, frozen=True)
class Option:
    type: Type

    def __repr__(self) -> str:
        return f"Optional[{self.type}]"


def materialize_type(clz: typing.Type[Any]) -> Type:
    if clz == int:
        return Primitive(type="int")
    elif clz == str:
        return Primitive(type="str")
    elif clz == bool:
        return Primitive(type="bool")
    elif clz == float:
        return Primitive(type="float")
    elif hasattr(clz, "__origin__") and clz.__origin__ == list:
        return List(materialize_type(clz.__args__[0]))
    elif is_dataclass(clz):
        fields = {}
        for name, field in clz.__dataclass_fields__.items():
            if field.default is not dataclasses.MISSING:
                has_default = True
                default = field.default
            elif field.default_factory is not dataclasses.MISSING:
                has_default = True
                default = field.default_factory()
            else:
                has_default = False
                default = None
            if isinstance(default, enum.Enum):
                default = default.value
            fields[name] = Field(
                name, materialize_type(field.type), default, has_default
            )
        return Struct(clz.__name__, fields)
    elif is_optional(clz):
        return Option(materialize_type(clz.__args__[0]))
    elif isinstance(clz, EnumMeta):
        variants = {}
        for name, value in clz.__members__.items():
            variants[name] = value.value
        return Enum(clz.__name__, variants)
    else:
        raise ValueError(f"Unsupported type: {clz}")


def is_optional(clz):
    return (
        hasattr(clz, "__origin__")
        and clz.__origin__ is typing.Union  # type: ignore
        and clz.__args__.__len__() == 2
        and clz.__args__[1] is type(None)
    )


def schema_from_namedtuple(schema: Any) -> Type:
    clz_name = schema.__class__.__name__
    if clz_name == "Primitive":
        return Primitive(schema.type)
    elif clz_name == "List":
        return List(schema_from_namedtuple(schema.inner))
    elif clz_name == "Struct":
        fields = {}
        for name, field in schema.fields.items():
            fields[name] = Field(
                name,
                schema_from_namedtuple(field.type),
                field.default,
                field.has_default,
            )
        return Struct(schema.name, fields)
    elif clz_name == "Option":
        return Option(schema_from_namedtuple(schema.type))
    elif clz_name == "Enum":
        variants = {}
        for name, value in schema.variants.items():
            variants[name] = value
        return Enum(schema.name, variants)
    else:
        raise ValueError(f"Unsupported type: {clz_name}")


def load_schema(path: str) -> Type:
    with open(path, "r") as f:
        schema = pyron.load(f.read(), preserve_structs=True)
    return schema_from_namedtuple(schema)
