from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple, Dict, Any, Callable

from hyperstate.schema import types


class RewriteRule(ABC):
    @abstractmethod
    def apply(self, state_dict: Any) -> Any:
        pass

    @abstractmethod
    def apply_to_schema(self, schema: types.Type) -> None:
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

    def apply_to_schema(self, schema: types.Type) -> Dict[str, Any]:
        field = _remove_schema(schema, self.old_field)
        _insert_schema(schema, self.new_field, field)


@dataclass
class DeleteField(RewriteRule):
    field: Sequence[str]

    def apply(self, state_dict: Any) -> Any:
        _remove(state_dict, self.field)
        return state_dict

    def apply_to_schema(self, schema: types.Type) -> Dict[str, Any]:
        _remove_schema(schema, self.field)


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

    def apply_to_schema(self, schema: types.Type) -> Dict[str, Any]:
        raise NotImplementedError


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

    def apply_to_schema(self, schema: types.Type) -> Dict[str, Any]:
        field = _remove_schema(schema, self.field)
        _insert_schema(schema, self.field, replace(field, default=self.new_default))


@dataclass
class AddDefault(RewriteRule):
    field: Sequence[str]
    default: Any

    def apply(self, state_dict: Any) -> Any:
        value, present = _remove(state_dict, self.field)
        _insert(state_dict, self.field, value if present else self.default)
        return state_dict

    def apply_to_schema(self, schema: types.Type) -> Dict[str, Any]:
        field = _remove_schema(schema, self.field)
        _insert_schema(schema, self.field, replace(field, default=self.default))


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


def _remove_schema(schema: types.Type, path: List[str]) -> Optional[types.Field]:
    assert len(path) > 0
    for field in path[:-1]:
        if not isinstance(schema, types.Struct):
            return None
        if field not in schema.fields:
            return None
        schema = schema.fields[field].type
    if not isinstance(schema, types.Struct):
        return None
    if path[-1] not in schema.fields:
        return None
    field = schema.fields[path[-1]]
    del schema.fields[path[-1]]
    return field.type


def _insert_schema(schema: types.Type, path: List[str], field: types.Field) -> None:
    assert len(path) > 0
    for field_name in path[:-1]:
        if not isinstance(schema, types.Struct):
            return
        schema = schema.fields[field_name].type
    if not isinstance(schema, types.Struct):
        return
    schema.fields[path[-1]] = field
