from __future__ import annotations

"""BaseAbstractionNavigator: generic infrastructure for abstraction-driven exploration.

This module contains only game-agnostic logic. Provide game-specific pieces
through constructor arguments from a thin wrapper (see `abstraction_navigator.py`).
"""

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..agent import Agent
from ..structs import FrameData, GameAction, GameState
from .abstractions import FrameAbstraction, AbstractionDetector
from .frame_viewer import save_png
from .grid_hash import FrameMask, hash_frame
from .nfr_planner import NearFrontierPlanner
from .types import (
    EnergyHudMeasurement,
    Frame,
    FrameHash,
    Memory,
    action_from_transition_key,
    load_memory,
    normalize_actions,
    save_memory,
    transition_key_from_action,
)

logger = logging.getLogger()

MEMORY_PATH = Path(__file__).resolve().parent / "memory" / "memory.json"


@dataclass(frozen=True)
class NavigatorSnapshot:
    """Immutable view of a single observation step."""

    frame: FrameData
    abstraction: FrameAbstraction
    frame_hash: FrameHash
    score: int
    level: int
    energy_measurement: Optional[EnergyHudMeasurement]
    level_start_state: FrameHash
    available_actions: list[GameAction]
    game_state: GameState


class BaseAbstractionNavigator(Agent):
    """Exploration-focused agent with pluggable, game-specific abstractions.

    Provide the following game-specific hooks when constructing:
      - user_abstractions: list[(name, detector)] where detector(frame) -> Any | None
      - measure_energy: callable (frame) -> EnergyHudMeasurement | None

    Notes on energy measurement:
      - By default, wrappers wire a concrete HUD extractor. If your game does not
        expose an energy HUD, pass a no-op function that returns None.
    """

    MAX_ACTIONS = 60
    ARROW_ACTIONS = [
        GameAction.ACTION1,  # Up
        GameAction.ACTION2,  # Down
        GameAction.ACTION3,  # Left
        GameAction.ACTION4,  # Right,
    ]
    APPARENT_RESTART_PIXEL_CHANGE_THRESHOLD = 0.4

    def __init__(
        self,
        *args: Any,
        user_abstractions: Sequence[tuple[str, AbstractionDetector]],
        measure_energy: Callable[[Frame], Optional[EnergyHudMeasurement]],
        **kwargs: Any,
    ) -> None:
        # initialise known attributes for type-checker; real values set in Agent.__init__
        self.game_id = getattr(self, "game_id", "")
        self.action_counter = getattr(self, "action_counter", 0)
        super().__init__(*args, **kwargs)
        seed = int(time.time() * 1_000_000) ^ hash(self.game_id)
        self.rng = random.Random(seed)
        self.last_action: Optional[GameAction] = None

        self._user_abstractions = list(user_abstractions)
        self._measure_energy = measure_energy

        self.memory: Memory = load_memory(MEMORY_PATH, logger_prefix=self.game_id)
        self._nfr_planner = NearFrontierPlanner(
            state_graph=self.memory.state_graph,
        )
        self._snapshots: deque[NavigatorSnapshot] = deque(maxlen=3)
        self._exploration_exhausted = False

    # Hints for the type checker; values are initialised in Agent.__init__
    game_id: str
    action_counter: int

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    @property
    def states_explored(self) -> int:
        return len(self.memory.state_graph)

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return any(
            [
                latest_frame.state is GameState.WIN,
                self.action_counter >= self.MAX_ACTIONS,
                self._exploration_exhausted,
            ]
        )

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state is GameState.NOT_PLAYED:
            self._reset_tracking()
            action = GameAction.RESET
            action.reasoning = "resetting before exploration"
            return action

        # Package raw frame into a snapshot with derived abstractions/state info.
        snapshot = self._create_navigator_snapshot(latest_frame)

        prev_snapshot = self._snapshots[-2] if len(self._snapshots) >= 2 else None

        if prev_snapshot is None:
            self.memory.mark_initial(snapshot.frame_hash, snapshot.level)

        self.memory.record_level(snapshot.frame_hash, snapshot.level)

        # Compute candidate actions early so we can record them in state graph.
        # For click-based games, these are the detected clickable squares.
        click_actions = _click_actions_for_snapshot(snapshot, self.game_id)
        candidate_actions: list[GameAction] = (
            click_actions if click_actions is not None else snapshot.available_actions
        )

        # Diagnostic: detect if we visit a state marked game-over but game says it's playable
        existing_record = self.memory.state_graph.get(snapshot.frame_hash)
        if existing_record and existing_record.is_game_over and snapshot.game_state is not GameState.GAME_OVER:
            logger.warning(
                "%s MISMATCH: state %s marked game-over in memory but game says %s (energy=%s, level=%d); clearing game-over flag",
                self.game_id,
                snapshot.frame_hash,
                snapshot.game_state.name,
                snapshot.energy_measurement.value if snapshot.energy_measurement else None,
                snapshot.level,
            )
            existing_record.is_game_over = False

        if snapshot.game_state is GameState.GAME_OVER:
            self._track_state_graph(prev_snapshot, snapshot, candidate_actions)
            prev_hash = prev_snapshot.frame_hash if prev_snapshot else None
            prev_energy = prev_snapshot.energy_measurement.value if prev_snapshot and prev_snapshot.energy_measurement else None
            action_name = self.last_action.name if self.last_action else None
            logger.warning(
                "%s GAME_OVER: hash=%s energy=%s prev_hash=%s prev_energy=%s action=%s",
                self.game_id,
                snapshot.frame_hash,
                snapshot.energy_measurement.value if snapshot.energy_measurement else None,
                prev_hash,
                prev_energy,
                action_name,
            )
            self.memory.mark_game_over(snapshot.frame_hash)
            self._reset_tracking()
            action = GameAction.RESET
            action.reasoning = "game-over-reset"
            return action

        if prev_snapshot is not None and snapshot.score > prev_snapshot.score:
            self._handle_level_change(prev_snapshot, snapshot)
        elif self._should_reset_for_apparent_restart(prev_snapshot, snapshot):
            logger.info(
                "%s apparent restart screen detected: state=%s score=%s",
                self.game_id,
                snapshot.game_state.name,
                snapshot.score,
            )
            action = GameAction.RESET
            action.reasoning = "apparent-restart-reset"
            self.last_action = None
            return action
        self._track_state_graph(prev_snapshot, snapshot, candidate_actions)

        state_record = self.memory.state_graph.get(snapshot.frame_hash)
        if state_record is not None and state_record.is_terminal:
            next_level = snapshot.level + 1
            next_level_start = self.memory.initial_for_level(next_level)
            if next_level_start is not None:
                for action_key, target_hash in state_record.transitions.items():
                    if target_hash == next_level_start:
                        action = action_from_transition_key(action_key)
                        logger.info(
                            "%s terminal transition reused: action=%s next_level=%d target=%s",
                            self.game_id,
                            action.name,
                            next_level,
                            next_level_start,
                        )
                        action.reasoning = "terminal-transition"
                        if action in self.ARROW_ACTIONS:
                            self.last_action = action
                        else:
                            self.last_action = None
                        return action

        terminal_target = self.memory.terminal_for_level(snapshot.level)
        nfr_action = self._nfr_planner.next_action(
            current_state=snapshot.frame_hash,
            available_actions=candidate_actions,
            level_start_state=snapshot.level_start_state,
            target_state=terminal_target,
            current_level=snapshot.level,
        )

        if nfr_action is None:
            logger.error(
                "%s exploration exhausted: no frontiers remaining at level %d with %d states explored",
                self.game_id,
                snapshot.level,
                len(self.memory.state_graph),
            )
            self._exploration_exhausted = True
            action = GameAction.RESET
            action.reasoning = "exploration-exhausted"
            self.last_action = None
            return action

        if nfr_action in self.ARROW_ACTIONS:
            self.last_action = nfr_action
        elif nfr_action is GameAction.RESET:
            self.last_action = None
        else:
            self.last_action = nfr_action
        return nfr_action

    def _reset_tracking(self) -> None:
        self.last_action = None
        self._snapshots.clear()

    def _create_navigator_snapshot(self, frame_data: FrameData) -> NavigatorSnapshot:

        prev_snapshot = self._snapshots[-1] if self._snapshots else None
        frame = frame_data.frame[-1]

        energy_measurement = self._measure_energy(frame)
        mask: FrameMask = tuple(energy_measurement.mask) if energy_measurement else ()
        frame_hash = hash_frame(frame, mask=mask)
        abstraction = FrameAbstraction(frame_hash=frame_hash, frame=frame)
        if energy_measurement is not None:
            abstraction.add("energy", energy_measurement)

        for name, detector in self._user_abstractions:
            try:
                result = detector(frame)
            except Exception:
                logger.exception(
                    "%s abstraction %s failed",
                    self.game_id,
                    getattr(detector, "__name__", repr(detector)),
                )
                continue
            if result is not None:
                abstraction.add(name, result)

        level, level_start_state = self._infer_level(
            prev_snapshot,
            score=frame_data.levels_completed,
            frame_hash=frame_hash,
        )

        snapshot = NavigatorSnapshot(
            frame=frame_data,
            abstraction=abstraction,
            frame_hash=frame_hash,
            score=frame_data.levels_completed,
            level=level,
            energy_measurement=energy_measurement,
            level_start_state=level_start_state,
            available_actions=normalize_actions(frame_data.available_actions),
            game_state=frame_data.state,
        )

        self._snapshots.append(snapshot)
        return snapshot

    def _should_reset_for_apparent_restart(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        snapshot: NavigatorSnapshot,
    ) -> bool:
        if not self._looks_like_transition_screen(snapshot):
            return False

        change_ratio = self._frame_internal_change_ratio(snapshot)
        if change_ratio < self.APPARENT_RESTART_PIXEL_CHANGE_THRESHOLD:
            return False

        confirmed_progress = (
            prev_snapshot is not None and snapshot.score > prev_snapshot.score
        )
        if confirmed_progress:
            return False

        return True

    def _looks_like_transition_screen(self, snapshot: NavigatorSnapshot) -> bool:
        frame_layers = getattr(snapshot.frame, "frame", None)
        if not isinstance(frame_layers, list):
            return False

        layer_count = len(frame_layers)
        if layer_count <= 1:
            return False

        return True

    def _frame_internal_change_ratio(self, snapshot: NavigatorSnapshot) -> float:
        layers = snapshot.frame.frame
        if not isinstance(layers, list) or len(layers) < 2:
            return 0.0

        return max(
            self._layer_difference_ratio(lhs, rhs)
            for lhs, rhs in zip(layers, layers[1:])
        )

    @staticmethod
    def _layer_difference_ratio(layer_a: Frame, layer_b: Frame) -> float:
        arr_a = np.asarray(layer_a)
        arr_b = np.asarray(layer_b)
        diff = np.count_nonzero(arr_a != arr_b)
        return float(diff) / float(arr_a.size)

    def cleanup(self, scorecard: Optional[Any] = None) -> None:
        known_states_total = len(self.memory.state_graph)
        logger.info(
            "%s known states total=%d",
            self.game_id,
            known_states_total,
        )
        save_memory(self.memory, MEMORY_PATH)
        super().cleanup(scorecard)

    def _track_state_graph(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        snapshot: NavigatorSnapshot,
        available_actions: list[GameAction],
    ) -> None:
        self._record_state_visit(snapshot, available_actions)
        self.memory.record_level(snapshot.frame_hash, snapshot.level)

        previous_state_hash = prev_snapshot.frame_hash if prev_snapshot else None
        if (
            previous_state_hash is None
            or not self.last_action
            or self.last_action is GameAction.RESET
        ):
            return
        self._record_state_transition(
            previous_state_hash,
            self.last_action,
            snapshot.frame_hash,
        )

    def _record_state_visit(
        self, snapshot: NavigatorSnapshot, available_actions: list[GameAction]
    ) -> None:
        record = self.memory.ensure_state(snapshot.frame_hash)

        # Diagnostic: warn if recording transitions on a game-over state
        if record.is_game_over and available_actions:
            logger.warning(
                "%s recording %d actions on game-over state %s",
                self.game_id,
                len(available_actions),
                snapshot.frame_hash,
            )

        # Record all available actions as unexplored (None) if not already known
        for action in available_actions:
            key = transition_key_from_action(action)
            if key not in record.transitions:
                record.transitions[key] = None

        energy_measurement = snapshot.energy_measurement
        if energy_measurement is None:
            return
        old_energy = record.energy
        new_energy = energy_measurement.value
        if old_energy is None:
            record.energy = new_energy
        else:
            if old_energy < energy_measurement.value:
                logger.info(
                    "memory: energy increased for %s from %d to %d; resetting %d transitions to unexplored",
                    snapshot.frame_hash,
                    old_energy,
                    energy_measurement.value,
                    len(record.transitions),
                )
                record.energy = new_energy
                for key in record.transitions:
                    record.transitions[key] = None

    def _record_state_transition(
        self,
        previous_hash: FrameHash,
        action: GameAction,
        next_hash: FrameHash,
    ) -> None:
        previous_record = self.memory.ensure_state(previous_hash)
        self.memory.ensure_state(next_hash)
        key = transition_key_from_action(action)
        existing = previous_record.transitions.get(key)
        # Set transition if not recorded or was unexplored (None)
        if key not in previous_record.transitions or existing is None:
            previous_record.transitions[key] = next_hash
            return
        if existing != next_hash:
            logger.warning(
                "Non-deterministic transition: state=%s, action=%s, existing_target=%s, new_target=%s",
                previous_hash,
                action.name,
                existing,
                next_hash,
            )

    def _handle_level_change(
        self, prev_snapshot: NavigatorSnapshot, snapshot: NavigatorSnapshot
    ) -> None:
        level_completed = prev_snapshot.level
        terminal_hash = prev_snapshot.frame_hash
        self.memory.mark_terminal(terminal_hash, level_completed)
        self.memory.mark_initial(snapshot.level_start_state, snapshot.level)
        logger.info(
            "%s level advanced to %d at step %d; start hash=%s; recorded terminal state for level %d=%s",
            self.game_id,
            snapshot.level,
            self.action_counter,
            snapshot.level_start_state,
            level_completed,
            terminal_hash,
        )

    def _infer_level(
        self,
        prev_snapshot: Optional[NavigatorSnapshot],
        *,
        score: int,
        frame_hash: FrameHash,
    ) -> tuple[int, FrameHash]:
        level = score + 1
        if prev_snapshot is None or level != prev_snapshot.level:
            return level, frame_hash
        return level, prev_snapshot.level_start_state


def _click_actions_for_snapshot(
    snapshot: NavigatorSnapshot, game_id: str
) -> Optional[list[GameAction]]:
    clickable_data = snapshot.abstraction.get("clickable")
    if not clickable_data:
        return None

    actions = _generate_click_actions(clickable_data, game_id)
    return actions


def _generate_click_actions(
    clickable_data: Sequence[tuple[int, int]], game_id: str
) -> list[GameAction]:
    """Convert abstraction-provided coordinates into actionable clicks."""

    actions: list[GameAction] = []
    seen: set[tuple[int, int]] = set()

    for x, y in clickable_data:
        if (x, y) in seen:
            continue
        seen.add((x, y))
        action = GameAction.ACTION6.clone()
        action.set_data({"game_id": game_id, "x": x, "y": y})
        actions.append(action)

    return actions
