from dataclasses import dataclass
from typing import Any, List, Optional
import tempfile
import pytest
from hyperstate.hyperstate import asdict, from_dict
from hyperstate.schema.rewrite_rule import (
    AddDefault,
    ChangeDefault,
    RenameField,
    RewriteRule,
)
from hyperstate.schema.schema_change import (
    DefaultValueChanged,
    FieldAdded,
    FieldRenamed,
    SchemaChange,
)

from hyperstate.schema.schema_checker import (
    SchemaChecker,
    Severity,
    _dump_schema,
)
from hyperstate.schema.types import materialize_type, Type, Primitive, Option


@dataclass
class ConfigV1:
    steps: int
    learning_rate: float
    batch_size: int
    epochs: int


@dataclass
class ConfigV2Error(ConfigV1):
    optimizer: str


@dataclass
class ConfigV2Warn(ConfigV1):
    optimizer: Optional[str]


@dataclass
class ConfigV2Info(ConfigV1):
    optimizer: str = "sgd"


@dataclass
class ConfigV3:
    steps: int
    lr: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"


def test_config_v1_to_v2():
    check_schema(ConfigV1, ConfigV2Info, [], [], Severity.INFO)
    check_schema(
        ConfigV1,
        ConfigV2Warn,
        [
            FieldAdded(
                ("optimizer",),
                type=Option(Primitive(type="str")),
                has_default=False,
                default=None,
            )
        ],
        [AddDefault(field=("optimizer",), default=None)],
        Severity.WARN,
    )
    check_schema(
        ConfigV1,
        ConfigV2Error,
        [FieldAdded(("optimizer",), type=Primitive(type="str"),)],
        [],
        Severity.ERROR,
    )
    automatic_upgrade(
        ConfigV1(steps=1, learning_rate=0.1, batch_size=32, epochs=10,),
        ConfigV2Warn(
            steps=1, learning_rate=0.1, batch_size=32, epochs=10, optimizer=None,
        ),
    )
    automatic_upgrade(
        ConfigV1(steps=1, learning_rate=0.1, batch_size=32, epochs=10,),
        ConfigV2Info(
            steps=1, learning_rate=0.1, batch_size=32, epochs=10, optimizer="sgd",
        ),
    )


def test_config_v2_to_v3():
    check_schema(
        ConfigV2Info,
        ConfigV3,
        [
            DefaultValueChanged(field=("optimizer",), old="sgd", new="adam"),
            FieldRenamed(field=("learning_rate",), new_name=("lr",)),
        ],
        [
            ChangeDefault(field=("optimizer",), new_default="adam"),
            RenameField(old_field=("learning_rate",), new_field=("lr",)),
        ],
        Severity.WARN,
    )
    automatic_upgrade(
        ConfigV2Info(steps=1, learning_rate=0.1, batch_size=32, epochs=10,),
        ConfigV3(steps=1, lr=0.1, batch_size=32, epochs=10, optimizer="sgd",),
    )


def check_schema(
    old: Type,
    new: Type,
    expected_changes: List[SchemaChange],
    expected_fixes: List[RewriteRule],
    expected_severity: Severity = Severity.ERROR,
    print_report: bool = False,
):
    with tempfile.TemporaryFile() as f:
        checker = SchemaChecker(materialize_type(old), materialize_type(new))
        if print_report:
            checker.print_report()
        assert checker.changes == expected_changes
        assert checker.proposed_fixes == expected_fixes
        assert checker.severity() == expected_severity


def automatic_upgrade(old: Any, new: Any):
    autofixes = SchemaChecker(
        materialize_type(old.__class__), materialize_type(new.__class__)
    ).proposed_fixes
    old_state_dict = asdict(old)
    print(old_state_dict)
    for fix in autofixes:
        print(fix)
        old_state_dict = fix.apply(old_state_dict)
        print(old_state_dict)
    assert from_dict(new.__class__, old_state_dict) == new

