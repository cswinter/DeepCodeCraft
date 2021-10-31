from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Any, Callable


class RewriteRule(ABC):
    @abstractmethod
    def apply(self, state_dict: Any) -> Any:
        pass


@dataclass
class RenameField(RewriteRule):
    old_field: Sequence[str]
    new_field: Sequence[str]

    def apply(self, state_dict: Any) -> Any:
        # TODO: handle lists
        value, ok = _remove(state_dict, self.old_field)
        if ok:
            _insert(state_dict, self.new_field, value)
        return state_dict


@dataclass
class DeleteField(RewriteRule):
    field: Sequence[str]

    def apply(self, state_dict: Any) -> Any:
        _remove(state_dict, self.field)


@dataclass
class MapFieldValue(RewriteRule):
    field: Sequence[str]
    map_fn: Callable[[Any], Any]
    rendered: Optional[str] = None

    def apply(self, state_dict: Any) -> Any:
        path = self.field
        value, present = _remove(state_dict, path)
        if present:
            new_value = self.map_fn(value)
            _insert(state_dict, path, new_value)
        return state_dict


@dataclass
class ChangeDefault(RewriteRule):
    field: Sequence[str]
    new_default: Any

    def apply(self, state_dict: Any) -> Any:
        existing_value, ok = _remove(state_dict, self.field)
        if not ok:
            _insert(state_dict, self.field, self.new_default)
        else:
            _insert(state_dict, self.field, existing_value)
        return state_dict


@dataclass
class AddDefault(RewriteRule):
    field: Sequence[str]
    default: Any

    def apply(self, state_dict: Any) -> Any:
        value, present = _remove(state_dict, self.field)
        _insert(state_dict, self.field, value if present else self.default)
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
