from typing import Generic, Type, TypeVar, Any
from enum import EnumMeta
import enum
import typing
from abc import ABC, abstractmethod, abstractclassmethod
from dataclasses import dataclass, field, is_dataclass
import dataclasses

import pyron
import click

T = TypeVar("T")


class Type(ABC):
    pass


@dataclass
class Primitive(Type):
    type: str


@dataclass
class List(Type):
    item_type: Type


@dataclass
class Field:
    name: str
    type: Type
    default: Any
    has_default: bool


@dataclass
class Enum(Type):
    name: str
    variants: typing.Dict[str, typing.Union[str, int]]


@dataclass
class Struct(Type):
    name: str
    fields: typing.Dict[str, Field]


@dataclass
class Option:
    type: Type


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


CONFIG_CLZ: typing.Type[Any] = None


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def dump_schema(filename: str):
    type = materialize_type(CONFIG_CLZ)
    serialized = pyron.to_string(type)
    with open(filename, "w") as f:
        f.write(serialized)


def schema_evolution_cli(config_clz: typing.Type[Any]):
    global CONFIG_CLZ
    CONFIG_CLZ = config_clz
    cli()
