from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from typing import Any, Dict, List
from hyperstate.schema import types

from hyperstate.schema.rewrite_rule import RewriteRule


@dataclass
class Versioned(ABC):
    version: int

    @abstractclassmethod
    def latest_version(clz) -> int:
        raise NotImplementedError()

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
        for i in range(version, clz.latest_version()):
            for rule in clz.upgrade_rules().get(i, []):
                state_dict = rule.apply(state_dict)
        state_dict["version"] = clz.latest_version()
        return state_dict

    @classmethod
    def _apply_schema_upgrades(clz, schema: types.Struct):
        for i in range(schema.version, clz.latest_version()):
            for rule in clz.upgrade_rules().get(i, []):
                rule.apply_to_schema(schema)
