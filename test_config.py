from pathlib import Path

from config import Config
from hyperstate.schema.schema_change import Severity
from hyperstate.schema.schema_checker import SchemaChecker
from hyperstate.schema.types import materialize_type, load_schema
from main import Trainer


def test_config():
    old = load_schema("config-schema.ron")
    checker = SchemaChecker(old, Config)
    if checker.severity() >= Severity.WARN:
        print(checker.print_report())
    assert checker.severity() == Severity.INFO


def test_load_old_checkpoint():
    path = Path(
        "/mnt/a/Dropbox/artifacts/xprun/dcc/micro_practice-7a06c09-3-0b4a0c6097d242b7b62e80496332c736/checkpoints/latest-step000040009728/"
    )
    if path.exists():
        trainer = Trainer(path)
        assert trainer.state.policy is not None
        assert trainer.state.optimizer is not None
