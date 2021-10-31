from typing import Type, TypeVar, Any
from enum import EnumMeta
import enum
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
import dataclasses
import logging

from .versioning import (
    RewriteRule,
    Versioned,
    DeleteField,
    RenameField,
    MapFieldValue,
    ChangeDefault,
    AddDefault,
)

import pyron
import click

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
    version: typing.Optional[int]

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


@dataclass
class SchemaChange(ABC):
    field: typing.List[str]

    @abstractmethod
    def diagnostic(self) -> str:
        pass

    def severity(self) -> int:
        if self.proposed_fix() is not None:
            return logging.WARNING
        else:
            return logging.ERROR

    def proposed_fix(self) -> typing.Optional[RewriteRule]:
        return None

    @property
    def field_name(self) -> str:
        return ".".join(self.field)

    def emit_diagnostic(self) -> None:
        severity = self.severity()
        if severity == logging.INFO:
            styled_severity = click.style("INFO ", fg="white")
        elif severity == logging.WARNING:
            styled_severity = click.style("WARN ", fg="yellow")
        else:
            styled_severity = click.style("ERROR", fg="red")

        print(
            f"{styled_severity} {self.diagnostic()}: {click.style(self.field_name, fg='cyan')}"
        )


@dataclass
class FieldAdded(SchemaChange):
    type: Type

    def diagnostic(self) -> str:
        return "new field without default value"


@dataclass
class FieldRemoved(SchemaChange):
    def diagnostic(self) -> str:
        return "field removed"

    def proposed_fix(self) -> RewriteRule:
        return DeleteField(self.field_name)


@dataclass
class DefaultValueChanged(SchemaChange):
    old: Any
    new: Any

    def diagnostic(self) -> str:
        return f"default value changed from {self.old} to {self.new}"

    def proposed_fix(self) -> RewriteRule:
        return ChangeDefault(self.field_name, self.new)


@dataclass
class DefaultValueRemoved(SchemaChange):
    old: Any

    def diagnostic(self) -> str:
        return f"default value removed: {self.old}"

    def proposed_fix(self) -> RewriteRule:
        return AddDefault(self.field_name, self.old)


@dataclass
class TypeChanged(SchemaChange):
    old: Type
    new: Type

    def severity(self) -> int:
        if isinstance(self.new, Option) and self.new.type == self.old:
            return logging.INFO
        elif (
            isinstance(self.old, Primitive)
            and isinstance(self.new, Primitive)
            and self.old.type == "int"
            and self.new.type == "float"
        ):
            return logging.INFO
        return super().severity()

    def diagnostic(self) -> str:
        return f"type changed from {self.old} to {self.new}"

    def proposed_fix(self) -> RewriteRule:
        if isinstance(self.new, List) and self.new.inner == self.old:
            return MapFieldValue(
                self.field_name, lambda x: [x], rendered="lambda x: [x]"
            )
        elif (
            isinstance(self.old, Primitive)
            and isinstance(self.new, Primitive)
            and self.old == "float"
            and self.new == "int"
        ):
            return MapFieldValue(
                self.field_name, lambda x: int(x), rendered="lambda x: int(x)"
            )


@dataclass
class EnumVariantValueChanged(SchemaChange):
    enum_name: str
    variant: str
    old_value: typing.Union[str, int]
    new_value: typing.Union[str, int]

    def diagnostic(self) -> str:
        return f"value of {self.enum_name}.{self.variant} changed from {self.old_value} to {self.new_value}"

    def proposed_fix(self) -> RewriteRule:
        return MapFieldValue(
            self.field_name,
            lambda x: x if x != self.old_value else self.new_value,
            rendered=f"lambda x: x if x != {self.old_value} else {self.new_value}",
        )


@dataclass
class EnumVariantRemoved(SchemaChange):
    enum_name: str
    variant: str

    def diagnostic(self) -> str:
        return f"variant {self.variant} of {self.enum_name} removed"


class SchemaChecker:
    def __init__(self, old: Type, new: Type):
        self.old = old
        self.new = new
        self.changes = []
        self._find_changes(old, new, [])

    def _find_changes(self, old: Type, new: Type, path: typing.List[str]):
        if old.__class__ != new.__class__:
            self.changes.append(TypeChanged(list(path), old, new))
        elif isinstance(old, Primitive):
            if old != new:
                self.changes.append(TypeChanged(list(path), old, new))
        elif isinstance(old, List):
            self._find_changes(old.inner, new.inner, path + ["[]"])
        elif isinstance(old, Struct):
            for name, field in new.fields.items():
                if name not in old.fields:
                    if not field.has_default:
                        self.changes.append(
                            FieldAdded(
                                list(path + [name]),
                                field.type,
                                field.default,
                                field.has_default,
                            )
                        )
                else:
                    self._find_changes(old.fields[name].type, field.type, path + [name])
                    if old.fields[name].has_default != field.has_default:
                        if not field.has_default:
                            self.changes.append(
                                DefaultValueRemoved(
                                    list(path + [name]), old.fields[name].default
                                )
                            )
                    elif (
                        old.fields[name].has_default
                        and old.fields[name].default != field.default
                    ):
                        if is_dataclass(field.default):
                            # TODO: perform comparison against namedtuple
                            pass
                        else:
                            self.changes.append(
                                DefaultValueChanged(
                                    list(path + [name]),
                                    old.fields[name].default,
                                    field.default,
                                )
                            )
            # Check for removed fields
            for name, field in old.fields.items():
                if name not in new.fields:
                    self.changes.append(FieldRemoved(list(path + [name])))

        elif isinstance(old, Option):
            self._find_changes(old.type, new.type, path + ["?"])
        elif isinstance(old, Enum):
            for name, value in new.variants.items():
                if name not in old.variants:
                    pass
                else:
                    if old.variants[name] != value:
                        self.changes.append(
                            EnumVariantValueChanged(
                                list(path), old.variants[name], value,
                            )
                        )
            for name, value in old.variants.items():
                if name not in new.variants:
                    self.changes.append(EnumVariantRemoved(list(path), value))
        else:
            raise ValueError(f"Unsupported type: {old}")

    def print_report(self):
        for change in self.changes:
            change.emit_diagnostic()

        print()
        click.echo(click.style("Proposed mitigations", fg="white", bold=True,))
        all_rewrite_rules = []
        for change in self.changes:
            proposed_fix = change.proposed_fix()
            if proposed_fix is not None:
                all_rewrite_rules.append(proposed_fix)
        if all_rewrite_rules:
            print("0: [")
            for mitigation in all_rewrite_rules:
                print(f"    {mitigation},")
            print("]")


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
    SchemaChecker(old, new).print_report()


def schema_evolution_cli(config_clz: typing.Type[Any]):
    global CONFIG_CLZ
    CONFIG_CLZ = config_clz
    cli()
