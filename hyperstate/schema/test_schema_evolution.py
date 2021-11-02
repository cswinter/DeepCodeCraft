from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile
import pyron
from hyperstate.hyperstate import _parse, asdict, from_dict
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
)
from hyperstate.schema.types import materialize_type, Type, Primitive, Option
from hyperstate.schema.versioned import Versioned


@dataclass
class ConfigV1(Versioned):
    steps: int
    learning_rate: float
    batch_size: int
    epochs: int

    @classmethod
    def version(clz) -> int:
        return 1


@dataclass
class ConfigV2Error(ConfigV1):
    optimizer: str


@dataclass
class ConfigV2Warn(ConfigV1):
    optimizer: Optional[str]


@dataclass
class ConfigV2Info(ConfigV1):
    optimizer: str = "sgd"

    @classmethod
    def version(clz) -> int:
        return 2


@dataclass
class ConfigV3(Versioned):
    steps: int
    lr: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"

    @classmethod
    def version(clz) -> int:
        return 3

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        return {
            2: [
                ChangeDefault(field=("optimizer",), new_default="adam"),
                RenameField(old_field=("learning_rate",), new_field=("lr",)),
            ],
        }


def test_config_v1_to_v2():
    check_schema(
        ConfigV1,
        ConfigV2Info,
        [
            FieldAdded(
                field=("optimizer",),
                type=Primitive("str"),
                has_default=True,
                default="sgd",
            )
        ],
        [],
        Severity.INFO,
    )
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


def test_serde_upgrade():
    config_v2 = ConfigV2Info(steps=1, learning_rate=0.1, batch_size=32, epochs=10)
    serialized = pyron.to_string(asdict(config_v2, named_tuples=True))
    state_dict = pyron.load(serialized)
    config_v3, _ = _parse(ConfigV3, state_dict, Path())
    assert config_v3 == ConfigV3(
        steps=1, lr=0.1, batch_size=32, epochs=10, optimizer="sgd"
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
        checker = SchemaChecker(materialize_type(old), new, perform_upgrade=False)
        if print_report:
            checker.print_report()
        assert checker.changes == expected_changes
        assert checker.proposed_fixes == expected_fixes
        assert checker.severity() == expected_severity


def automatic_upgrade(old: Any, new: Any):
    autofixes = SchemaChecker(
        materialize_type(old.__class__), new.__class__, perform_upgrade=False
    ).proposed_fixes
    old_state_dict = asdict(old)
    for fix in autofixes:
        old_state_dict = fix.apply(old_state_dict)
    del old_state_dict["version"]
    assert from_dict(new.__class__, old_state_dict) == new

