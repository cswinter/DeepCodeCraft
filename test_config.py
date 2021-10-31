from config import Config
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import materialize_type, load_schema


def test_config():
    old = load_schema("config-schema.ron")
    new = materialize_type(Config)
    checker = SchemaChecker(old, new)
    if checker.severity() >= Severity.WARN:
        print(checker.print_report())
    assert checker.severity() == Severity.INFO
