from pickle import load
from typing import List, Optional, Type, Any
from enum import Enum
import typing
from dataclasses import Field, is_dataclass

from hyperstate.schema.schema_change import (
    DefaultValueChanged,
    DefaultValueRemoved,
    EnumVariantRemoved,
    EnumVariantValueChanged,
    FieldAdded,
    FieldRemoved,
    FieldRenamed,
    SchemaChange,
    Severity,
    TypeChanged,
)
from hyperstate.schema.versioned import Versioned

from .types import T, Type, load_schema, materialize_type
from . import types

import pyron
import click


class SchemaChecker:
    def __init__(self, old: Type, config_clz: typing.Type[Versioned]):
        self.config_clz = config_clz
        self.new = materialize_type(config_clz)
        config_clz._apply_schema_upgrades(old)
        self.old = old
        self.changes: List[SchemaChange] = []
        self.proposed_fixes = []
        self._find_changes(old, self.new, [])
        self._find_renames()
        for change in self.changes:
            proposed_fix = change.proposed_fix()
            if proposed_fix is not None:
                self.proposed_fixes.append(proposed_fix)

    def severity(self) -> Severity:
        max_severity = Severity.INFO
        for change in self.changes:
            if change.severity() > max_severity:
                max_severity = change.severity()
        return max_severity

    def print_report(self):
        for change in self.changes:
            change.emit_diagnostic()
        if self.severity() > Severity.INFO and self.old.version == self.new.version:
            print(
                click.style("WARN", fg="yellow")
                + "  schema changed but version identical"
            )

        if self.severity() == Severity.INFO:
            click.secho("Schema compatible", fg="green")
        else:
            click.secho("Schema incompatible", fg="red")
            print()
            click.secho("Proposed mitigations", fg="white", bold=True)
            if self.proposed_fixes:
                click.secho("- add upgrade rules:", fg="white", bold=True)
                print(f"    {self.old.version}: [")
                for mitigation in self.proposed_fixes:
                    print(f"        {mitigation},")
                print("    ],")
            if self.severity() > Severity.INFO and self.old.version == self.new.version:
                click.secho(
                    f"- bump version to {self.old.version + 1}", fg="white", bold=True
                )

    def _find_changes(self, old: Type, new: Type, path: typing.List[str]):
        if old.__class__ != new.__class__:
            self.changes.append(TypeChanged(tuple(path), old, new))
        elif isinstance(old, types.Primitive):
            if old != new:
                self.changes.append(TypeChanged(tuple(path), old, new))
        elif isinstance(old, types.List):
            self._find_changes(old.inner, new.inner, path + ["[]"])
        elif isinstance(old, types.Struct):
            for name, field in new.fields.items():
                if name not in old.fields:
                    self.changes.append(
                        FieldAdded(
                            tuple(path + [name]),
                            field.type,
                            field.has_default,
                            field.default,
                        )
                    )
                else:
                    self._find_changes(old.fields[name].type, field.type, path + [name])
                    if old.fields[name].has_default != field.has_default:
                        if not field.has_default:
                            self.changes.append(
                                DefaultValueRemoved(
                                    tuple(path + [name]), old.fields[name].default
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
                                    tuple(path + [name]),
                                    old.fields[name].default,
                                    field.default,
                                )
                            )
            # Check for removed fields
            for name, field in old.fields.items():
                if name not in new.fields:
                    self.changes.append(
                        FieldRemoved(
                            tuple(path + [name],),
                            field.type,
                            field.has_default,
                            field.default,
                        )
                    )

        elif isinstance(old, types.Option):
            self._find_changes(old.type, new.type, path + ["?"])
        elif isinstance(old, types.Enum):
            for name, value in new.variants.items():
                if name not in old.variants:
                    pass
                else:
                    if old.variants[name] != value:
                        self.changes.append(
                            EnumVariantValueChanged(
                                tuple(path), old.variants[name], value,
                            )
                        )
            for name, value in old.variants.items():
                if name not in new.variants:
                    self.changes.append(EnumVariantRemoved(tuple(path), value))
        else:
            raise ValueError(f"Unsupported type: {old}")

    def _find_renames(self):
        threshold = 0.1
        removeds = [
            change for change in self.changes if isinstance(change, FieldRemoved)
        ]
        addeds = {change for change in self.changes if isinstance(change, FieldAdded)}
        for removed in removeds:
            best_similarity = threshold
            best_match: Optional[FieldAdded] = None
            for added in addeds:
                if (
                    removed.type == added.type
                    and removed.has_default == added.has_default
                    and removed.default == added.default
                ):
                    similarity = field_name_similarity(
                        removed.field[-1], added.field[-1]
                    )
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = added
            if best_match is not None:
                self.changes.remove(removed)
                self.changes.remove(best_match)
                addeds.remove(best_match)
                self.changes.append(FieldRenamed(removed.field, best_match.field))


def field_name_similarity(field1: str, field2: str):
    # Special cases
    if field1 == field2:
        return 1.1
    if field1.replace("_", "") == field2.replace("_", ""):
        return 1.0
    if field1 == "".join(
        [word[0:1] for word in field2.split("_")]
    ) or field2 == "".join([word[0:1] for word in field1.split("_")]):
        return 1.0
    return levenshtein(field1, field2) / max(len(field1), len(field2))


def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = (
                previous_row[j + 1] + 1
            )  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _dump_schema(filename: str, type: typing.Type[Versioned]):
    serialized = pyron.to_string(materialize_type(type))
    with open(filename, "w") as f:
        f.write(serialized)


def _upgrade_schema(filename: str, config_clz: typing.Type[Versioned]):
    schema = load_schema(filename)
    checker = SchemaChecker(schema, config_clz)
    if checker.severity() >= Severity.WARN:
        checker.print_report()
    else:
        _dump_schema(filename, config_clz)
        click.secho("Schema updated", fg="green")


CONFIG_CLZ: typing.Type[Any] = None


@click.group()
def cli():
    pass


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def dump_schema(filename: str):
    _dump_schema(filename, CONFIG_CLZ)


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def upgrade_schema(filename: str):
    global CONFIG_CLZ
    _upgrade_schema(filename, CONFIG_CLZ)


@cli.command()
@click.argument("filename", default="config-schema.ron", type=click.Path())
def check_schema(filename: str):
    global CONFIG_CLZ
    old = load_schema(filename)
    SchemaChecker(old, CONFIG_CLZ).print_report()


def schema_evolution_cli(config_clz: typing.Type[Any]):
    global CONFIG_CLZ
    CONFIG_CLZ = config_clz
    cli()
