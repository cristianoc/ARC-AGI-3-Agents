"""Shared type definitions and helpers for navigator modules."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Mapping, NamedTuple, NewType, Optional, Sequence, Set, Tuple

from ..structs import GameAction

logger = logging.getLogger(__name__)

FrameHash = NewType("FrameHash", str)

Frame = list[list[int]] # 64x64

def normalize_actions(actions: Sequence[int]) -> list[GameAction]:
    """Convert a list of action IDs (from arcengine) to GameAction instances."""
    return [GameAction.from_id(a) for a in actions]


def transition_key_from_action(action: GameAction) -> str:
    """Serialize an action (including coordinates for ACTION6) into a key."""

    if action.name == "ACTION6":
        data = getattr(action, "action_data", None)
        x = getattr(data, "x", None)
        y = getattr(data, "y", None)
        if x is not None and y is not None:
            return f"{action.name}@{int(x)},{int(y)}"
    return str(action.name)


def action_from_transition_key(key: str) -> GameAction:
    """Reconstruct a GameAction (with coords when present) from a key."""

    if "@" not in key:
        return GameAction[key]

    action_name, coords = key.split("@", 1)
    base_action = GameAction[action_name]
    if action_name != "ACTION6":
        return base_action

    try:
        x_str, y_str = coords.split(",", 1)
        x = int(x_str)
        y = int(y_str)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"Invalid transition key: {key}") from exc

    action = base_action.clone()
    action.set_data({"x": x, "y": y})
    return action


class Color(IntEnum):
    """16-color palette for ARC-AGI-3 frames."""

    WHITE = 0
    OFF_WHITE = 1
    NEUTRAL_LIGHT = 2
    NEUTRAL = 3
    OFF_BLACK = 4
    BLACK = 5
    MAGENTA = 6
    MAGENTA_LIGHT = 7  # Pink - used for energy HUD in vc33
    RED = 8
    BLUE = 9
    BLUE_LIGHT = 10
    YELLOW = 11
    ORANGE = 12
    MAROON = 13
    GREEN = 14
    PURPLE = 15


# 16-color palette as RGB tuples (derived from multimodal._PALETTE, stripping alpha).
from ..templates.multimodal import _PALETTE as _RGBA_PALETTE

PALETTE: list[tuple[int, int, int]] = [(r, g, b) for r, g, b, _ in _RGBA_PALETTE]

class MaskRect(NamedTuple):
    """Inclusive rectangle specified as (y0, y1, x0, x1)."""

    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class EnergyHudMeasurement:
    """Structured representation of an energy HUD value.

    The energy is represented as a non-negative integer `value` with an
    the HUD mask geometry. If the HUD spans disjoint regions, include every rectangle that belongs to the HUD.
    """

    value: int
    mask: Sequence["MaskRect"]
    """Rectangles describing every pixel that belongs to the energy HUD.

    This mask is considered the canonical HUD geometry for the current frame and
    is reused for frame hashing so that HUD redraws do not affect state identity.
    """


    # Backwards compatibility: expose `filled_blocks` as an alias for `value`.
    @property
    def filled_blocks(self) -> int:
        return self.value


@dataclass
class StateRecord:
    """Observed information about a specific frame hash."""

    transitions: Dict[str, FrameHash] = field(default_factory=dict)
    level: Optional[int] = None
    energy: Optional[int] = None
    is_initial: bool = False
    is_terminal: bool = False
    is_game_over: bool = False

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "transitions": {key: str(target) for key, target in self.transitions.items()}
        }
        if self.level is not None:
            payload["level"] = self.level
        if self.energy is not None:
            payload["energy"] = self.energy
        if self.is_initial:
            payload["is_initial"] = True
        if self.is_terminal:
            payload["is_terminal"] = True
        if self.is_game_over:
            payload["is_game_over"] = True
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateRecord":
        transitions_payload = payload.get("transitions", {})
        transitions: Dict[str, FrameHash] = {}
        if isinstance(transitions_payload, Mapping):
            for action_key, raw in transitions_payload.items():
                if not isinstance(action_key, str):
                    logger.warning("Skipping non-string action key in memory payload: %s", action_key)
                    continue
                try:
                    action_from_transition_key(action_key)
                except (KeyError, ValueError):
                    logger.warning("Skipping unknown action in memory payload: %s", action_key)
                    continue
                try:
                    transitions[action_key] = FrameHash(str(raw))
                except (TypeError, ValueError):
                    logger.warning(
                        "Skipping transition for action %s due to invalid target %s",
                        action_key,
                        raw,
                    )
                    continue

        raw_level = payload.get("level")
        level: Optional[int] = None
        if isinstance(raw_level, int):
            level = raw_level
        elif raw_level is not None:
            logger.warning("Skipping invalid level entry %s", raw_level)
        is_initial = bool(payload.get("is_initial", False))
        is_terminal = bool(payload.get("is_terminal", False))
        is_game_over = bool(payload.get("is_game_over", False))
        energy: Optional[int] = None
        raw_energy = payload.get("energy")
        if isinstance(raw_energy, int):
            energy = raw_energy
        elif raw_energy is not None:
            logger.warning("Skipping invalid energy entry %s", raw_energy)
        return cls(
            transitions=transitions,
            level=level,
            energy=energy,
            is_initial=is_initial,
            is_terminal=is_terminal,
            is_game_over=is_game_over,
        )


STATE_GRAPH = Dict[FrameHash, StateRecord]


@dataclass
class Memory:
    """Persistent navigation memory retained across runs."""

    state_graph: STATE_GRAPH = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "state_graph": {
                str(state_hash): record.to_dict()
                for state_hash, record in self.state_graph.items()
            }
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "Memory":
        memory = cls()
        state_graph_payload = payload.get("state_graph", payload)
        if isinstance(state_graph_payload, Mapping):
            for hash_str, transitions in state_graph_payload.items():
                try:
                    state_hash = FrameHash(str(hash_str))
                except (TypeError, ValueError):
                    logger.warning("Skipping invalid state id in memory payload: %s", hash_str)
                    continue
                if not isinstance(transitions, Mapping):
                    logger.warning(
                        "Skipping state %s because record payload is missing or malformed",
                        hash_str,
                    )
                    continue
                memory.state_graph[state_hash] = StateRecord.from_dict(transitions)
        return memory

    def ensure_state(self, frame_hash: FrameHash) -> StateRecord:
        record = self.state_graph.get(frame_hash)
        if record is None:
            record = StateRecord()
            self.state_graph[frame_hash] = record
        return record

    def mark_game_over(self, frame_hash: FrameHash) -> None:
        self.ensure_state(frame_hash).is_game_over = True

    def mark_terminal(self, frame_hash: FrameHash, level: int) -> None:
        for record in self.state_graph.values():
            if record.level == level and record.is_terminal:
                record.is_terminal = False
        record = self.ensure_state(frame_hash)
        record.level = level
        record.is_terminal = True

    def mark_initial(self, frame_hash: FrameHash, level: int) -> None:
        for state_hash, record in self.state_graph.items():
            if record.level == level and record.is_initial:
                if state_hash != frame_hash:
                    raise ValueError(
                        f"Initial state for level {level} already recorded as {state_hash}; "
                        f"cannot reassign to {frame_hash}"
                    )
                return
        record = self.ensure_state(frame_hash)
        record.level = level
        record.is_initial = True

    def record_level(self, frame_hash: FrameHash, level: int) -> None:
        record = self.ensure_state(frame_hash)
        if record.level is not None and record.level != level:
            logger.error(
                "memory: level changed for %s from %d to %d",
                frame_hash,
                record.level,
                level,
            )
        record.level = level

    def terminal_for_level(self, level: int) -> Optional[FrameHash]:
        for state_hash, record in self.state_graph.items():
            if record.level == level and record.is_terminal:
                return state_hash
        return None

    def initial_for_level(self, level: int) -> Optional[FrameHash]:
        for state_hash, record in self.state_graph.items():
            if record.level == level and record.is_initial:
                return state_hash
        return None

    def game_over_hashes(self) -> Set[FrameHash]:
        return {state_hash for state_hash, record in self.state_graph.items() if record.is_game_over}


def load_memory(path: Path, *, logger_prefix: Optional[str] = None) -> Memory:
    if not path.exists():
        return Memory()
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning(
            "%s could not load memory file %s; starting fresh",
            logger_prefix or "navigator",
            path,
        )
        return Memory()
    if isinstance(raw, dict):
        return Memory.from_dict(raw)
    logger.warning(
        "%s memory file %s was not a dict; starting fresh",
        logger_prefix or "navigator",
        path,
    )
    return Memory()


def save_memory(memory: Memory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory.to_dict(), indent=2))
