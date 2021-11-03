from abc import ABC, abstractclassmethod
from dataclasses import dataclass, MISSING
from typing import Any, Dict, List
from hyperstate.schema import types

from hyperstate.schema.rewrite_rule import RewriteRule


@dataclass
class Versioned(ABC):
    @abstractclassmethod
    def version(clz) -> int:
        raise NotImplementedError(f"{clz.__name__}.version() not implemented")

    @classmethod
    def minimum_version(clz) -> int:
        return 0

    @classmethod
    def upgrade_rules(clz) -> Dict[int, List[RewriteRule]]:
        """
        Returns a list of rewrite rules that can be applied to the given version
        to make it compatible with the next version.
        """
        return {}

    @classmethod
    def _apply_upgrades(clz, state_dict: Any, version: int) -> Any:
        for version in clz.upgrade_rules().keys():
            assert (
                version < clz.version()
            ), f"{clz.__name__}.upgrade_rules() keys must be less than {clz.__name__}.version()"
        for i in range(version, clz.version()):
            for rule in clz.upgrade_rules().get(i, []):
                state_dict = rule.apply(state_dict)
        return state_dict

    @classmethod
    def _apply_schema_upgrades(clz, schema: types.Struct):
        for i in range(schema.version, clz.version()):
            for rule in clz.upgrade_rules().get(i, []):
                rule.apply_to_schema(schema)
