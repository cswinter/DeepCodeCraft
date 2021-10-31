from abc import ABC, abstractclassmethod
from typing import List, Any, Dict

from hyperstate.schema.rewrite_rule import RewriteRule


class Versioned(ABC):
    @abstractclassmethod
    def version(clz) -> int:
        pass

    @classmethod
    def minimum_version(clz) -> int:
        return 0

    @classmethod
    def upgrade_rules(clz) -> Dict[int, RewriteRule]:
        """
        Returns a list of rewrite rules that can be applied to the given version
        to make it compatible with the next version.
        """
        return []

    @classmethod
    def _apply_upgrades(clz, state_dict: Any, version: int) -> Any:
        for i in range(version, clz.version()):
            for rule in clz.upgrade_rules.get(i, []):
                state_dict = rule.apply(state_dict)
        return state_dict
