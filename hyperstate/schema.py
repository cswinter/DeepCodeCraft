from typing import Type, TypeVar, Any
from enum import EnumMeta
import enum
import typing
from abc import ABC
from dataclasses import dataclass, is_dataclass
import dataclasses

from .versioning import Versioned

import pyron
import click

T = TypeVar("T")


class Type(ABC):
    pass


@dataclass(eq=True, frozen=True)
class Primitive(Type):
    type: str


@dataclass(eq=True, frozen=True)
class List(Type):
    item_type: Type


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
    version: typing.Optional[int]


@dataclass(eq=True, frozen=True)
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
        if issubclass(clz, Versioned):
            version = clz.version()
        else:
            version = None
        return Struct(clz.__name__, fields, version)
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
        return List(schema_from_namedtuple(schema.item_type))
    elif clz_name == "Struct":
        fields = {}
        for name, field in schema.fields.items():
            fields[name] = Field(
                name,
                schema_from_namedtuple(field.type),
                field.default,
                field.has_default,
            )
        return Struct(schema.name, fields, schema.version)
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


def compare_schemas(old: Type, new: Type, path: typing.List[str]):
    if old.__class__ != new.__class__:
        # ALLOWED
        # T -> Optional[T]
        # POTENTIALLY FIXABLE
        # Optional[T] -> T => Add Map None -> new default
        # T -> List[T] => Add Map x => [x]
        # ERROR
        # Any other type change
        if isinstance(new, Option):  # and new.type == old:
            print(f"INFO: {'.'.join(path)}: {old} -> {new}")
        elif isinstance(old, Option) and old.type == new:
            print(f"WARN: {'.'.join(path)}: {old} -> {new}")
        elif isinstance(new, List) and new.item_type == old:
            print(f"WARN: {'.'.join(path)}: {old} -> {new}")
        else:
            print(f"ERROR: {'.'.join(path)}: {old} -> {new}")
    elif isinstance(old, Primitive):
        # ALLOWED
        # int -> float
        # ERROR
        # Any other type change
        if old.type == "int" and new.type == "float":
            print(f"INFO: {'.'.join(path)}: {old.type} -> {new.type}")
        elif old.type != new.type:
            print(f"ERROR: {'.'.join(path)}: {old.type} -> {new.type}")
    elif isinstance(old, List):
        compare_schemas(old.item_type, new.item_type, path + ["[]"])
    elif isinstance(old, Struct):
        # ALLOWED
        # new field with default value
        # default value added
        # POTENTIALLY FIXABLE
        # new field without default value
        # removed field
        # field default value changed
        # ERROR
        # default value removed

        # Check for new fields and field type changes
        for name, field in new.fields.items():
            if name not in old.fields:
                if field.has_default:
                    print(f"INFO: Added {'.'.join(path)}.{name}")
                else:
                    print(f"WARN: Added {'.'.join(path)}.{name} without default")
            else:
                compare_schemas(old.fields[name].type, field.type, path + [name])
                if old.fields[name].has_default != field.has_default:
                    if field.has_default:
                        print(f"INFO: Added default value to {'.'.join(path)}.{name}")
                    else:
                        print(
                            f"WARN: Removed default value from {'.'.join(path)}.{name}"
                        )
                elif (
                    old.fields[name].has_default
                    and old.fields[name].default != field.default
                ):
                    if is_dataclass(field.default):
                        # TODO: perform comparison against namedtuple
                        pass
                    else:
                        print(
                            f"WARN: Changed default value of {'.'.join(path)}.{name} from {old.fields[name].default} to {field.default}"
                        )
        # Check for removed fields
        for name, field in old.fields.items():
            if name not in new.fields:
                print(f"WARN: {'.'.join(path)}.{name} removed")
    elif isinstance(old, Option):
        compare_schemas(old.type, new.type, path + ["?"])
    elif isinstance(old, Enum):
        # ALLOWED
        # new variant
        # POTENTIALLY FIXABLE
        # variant removed
        # variant value changed
        for name, value in new.variants.items():
            if name not in old.variants:
                print(f"INFO: Added {'.'.join(path)}.{name}")
            else:
                if old.variants[name] != value:
                    print(
                        f"WARN: Changed value of {name} variant of {old.name} enum ({'.'.join(path)})"
                    )
        for name, value in old.variants.items():
            if name not in new.variants:
                print(
                    f"ERROR: Removed {name} variant of {old.name} enum ({'.'.join(path)})"
                )
    else:
        raise ValueError(f"Unsupported type: {old}")


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


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def check_schema(filename: str):
    global CONFIG_CLZ
    old = load_schema(filename)
    new = materialize_type(CONFIG_CLZ)
    compare_schemas(old, new, [])


def schema_evolution_cli(config_clz: typing.Type[Any]):
    global CONFIG_CLZ
    CONFIG_CLZ = config_clz
    cli()
