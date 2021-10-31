from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, Callable


class RewriteRule(ABC):
    @abstractmethod
    def apply(self, state_dict: Any) -> Any:
        pass


@dataclass
class RenameField(RewriteRule):
    old_field: str
    new_field: str

    def apply(self, state_dict: Any) -> Any:
        # TODO: handle lists
        old_path = self.old_field.split(".")
        value, present = _remove(state_dict, old_path)
        if present:
            new_path = self.new_field.split(".")
            _insert(state_dict, new_path, value)
        return state_dict


@dataclass
class DeleteField(RewriteRule):
    field: str

    def apply(self, state_dict: Any) -> Any:
        _remove(state_dict, self.field.split("."))


@dataclass
class MapFieldValue(RewriteRule):
    field: str
    map_fn: Callable[[Any], Any]
    rendered: Optional[str] = None

    def apply(self, state_dict: Any) -> Any:
        path = self.field.split(".")
        value, present = _remove(state_dict, path)
        if present:
            new_value = self.map_fn(value)
            _insert(state_dict, path, new_value)
        return state_dict


@dataclass
class ChangeDefault(RewriteRule):
    field: str
    new_default: Any

    def apply(self, state_dict: Any) -> Any:
        path = self.field.split(".")
        _, present = _remove(state_dict, path)
        if not present:
            _insert(state_dict, path, self.default)
        return state_dict


@dataclass
class AddDefault(RewriteRule):
    field: str
    default: Any

    def apply(self, state_dict: Any) -> Any:
        path = self.field.split(".")
        value, present = _remove(state_dict, path)
        _insert(state_dict, path, value if present else self.default)
        return state_dict


def _remove(state_dict: Dict[str, Any], path: List[str]) -> Tuple[Any, bool]:
    assert len(path) > 0
    for field in path[:-1]:
        if field not in state_dict:
            return None, False
        state_dict = state_dict[field]
    if path[-1] not in state_dict:
        return None, False
    value = state_dict[path[-1]]
    del state_dict[path[-1]]
    return value, True


def _insert(state_dict: Dict[str, Any], path: List[str], value: Any) -> None:
    assert len(path) > 0
    for field in path[:-1]:
        if field not in state_dict:
            state_dict[field] = {}
        state_dict = state_dict[field]
    state_dict[path[-1]] = value
