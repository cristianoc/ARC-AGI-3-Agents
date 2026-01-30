"""Compatibility shim for types moved to arcengine.

This module re-exports types from arcengine and adds custom methods
needed by the navigator module.
"""

from typing import Any

# Re-export base types from arcengine
from arcengine import FrameData, GameAction, GameState

__all__ = ["FrameData", "GameAction", "GameState"]


# --- Custom GameAction methods (monkey-patched onto the arcengine enum) ---


def _clone(self: GameAction) -> GameAction:
    """Create a new instance with the same action type but fresh action_data."""
    new_instance = object.__new__(self.__class__)
    new_instance._name_ = self._name_
    new_instance._value_ = self._value_
    new_instance.action_type = self.action_type
    new_instance.action_data = self.action_type()
    new_instance.reasoning = None
    return new_instance


def _custom_str(self: GameAction) -> str:
    """Custom string representation including x,y for ACTION6."""
    if self._name_ == "ACTION6":
        data = getattr(self, "action_data", None)
        if data is not None and hasattr(data, "x") and hasattr(data, "y"):
            return f"{self.name}(x={data.x}, y={data.y})"
    return self.name


def _custom_repr(self: GameAction) -> str:
    return _custom_str(self)


def _custom_hash(self: GameAction) -> int:
    """Hash based on identity for dict/set operations, allowing distinct instances."""
    return id(self)


def _custom_eq(self: GameAction, other: object) -> bool:
    """Compare by identity, so cloned instances are distinct."""
    if not isinstance(other, GameAction):
        return False
    return self is other


# Apply monkey-patches
GameAction.clone = _clone  # type: ignore[attr-defined]
GameAction.__str__ = _custom_str  # type: ignore[method-assign]
GameAction.__repr__ = _custom_repr  # type: ignore[method-assign]
GameAction.__hash__ = _custom_hash  # type: ignore[method-assign]
GameAction.__eq__ = _custom_eq  # type: ignore[method-assign]
