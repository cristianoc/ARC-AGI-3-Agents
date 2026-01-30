"""Planner utilities for Near-Frontier with Reset (NFR) exploration."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..structs import GameAction
from .types import (
    FrameHash,
    STATE_GRAPH,
    action_from_transition_key,
    transition_key_from_action,
)


logger = logging.getLogger(__name__)

class NearFrontierPlanner:
    """Compute plans that minimise distance to the nearest unexplored frontier."""

    def __init__(
        self,
        *,
        state_graph: STATE_GRAPH,
    ) -> None:
        self._state_graph = state_graph

    def next_action(
        self,
        *,
        current_state: FrameHash,
        available_actions: Sequence[GameAction],
        level_start_state: FrameHash,
        target_state: Optional[FrameHash] = None,
    ) -> Optional[GameAction]:

        adj = self._build_adj()
        s0 = level_start_state
        if target_state is not None and self._is_blocked(target_state):
            target_state = None

        dist_c, prev_c = self._bfs(adj, current_state)
        dist_s0, prev_s0 = self._bfs(adj, s0)

        if target_state is not None and target_state in dist_c:
            terminal_path = self._recover_path(prev_c, current_state, target_state)
            if terminal_path:
                action = terminal_path[0]
                action.reasoning = f"nfr-terminal:{action.name.lower()}"
                return action

        INF = 10**9
        best: Optional[Tuple[int, int, FrameHash]] = None
        for state in self._frontier_states():
            d_current = dist_c.get(state, INF)
            d_reset = dist_s0.get(state, INF)
            frontier_cost = min(d_current, 1 + d_reset)
            if frontier_cost >= INF:
                continue
            key = (frontier_cost, d_reset, state)
            if best is None or key < best:
                best = key

        if best is None:
            return None

        # best stores (cost, reset_distance, frontier_state)
        _, d_reset, target_state = best
        d_current = dist_c.get(target_state, INF)

        navigation: List[GameAction] = []
        if d_current <= 1 + d_reset:
            navigation = self._recover_path(prev_c, current_state, target_state)
        else:
            logger.info(
                "nfr-reset decision: current=%s s0=%s target=%s d_current=%s d_reset=%s available=%s",
                current_state,
                s0,
                target_state,
                d_current,
                d_reset,
                [action.name for action in available_actions],
            )
            action = GameAction.RESET
            action.reasoning = "nfr-reset"
            return action

        if navigation:
            action = navigation[0]
            action.reasoning = f"nfr-nav:{action.name.lower()}"
            return action

        # Already at the chosen frontier: probe the first unseen arrow action that is available.
        for action in available_actions:
            if self._is_action_known(current_state, action):
                continue
            action.reasoning = f"nfr-probe:{action.name.lower()}"
            return action

        return None

    def _discovered_states(self) -> set[FrameHash]:
        states: set[FrameHash] = set()
        for state, record in self._state_graph.items():
            if record.is_game_over:
                continue
            states.add(state)
            for target in record.transitions.values():
                # Skip unexplored transitions (None) and blocked states
                if target is None or self._is_blocked(target):
                    continue
                states.add(target)
        return states

    def _build_adj(self) -> Dict[FrameHash, List[Tuple[FrameHash, GameAction]]]:
        """Build adjacency graph using ALL known transitions.

        For click-based games, different states have different available actions
        (different clickable positions per level). To enable cross-level navigation,
        we include all known edges regardless of current available_actions.
        """
        adjacency: Dict[FrameHash, List[Tuple[FrameHash, GameAction]]] = {}
        for state, record in self._state_graph.items():
            if record.is_game_over:
                continue
            for action_key, target in record.transitions.items():
                # Skip unexplored transitions (None) - can't navigate through unknown edges
                if target is None or self._is_blocked(target):
                    continue
                action = action_from_transition_key(action_key)
                adjacency.setdefault(state, []).append((target, action))
        return adjacency

    def _bfs(
        self,
        adjacency: Dict[FrameHash, List[Tuple[FrameHash, GameAction]]],
        start: FrameHash,
    ) -> Tuple[Dict[FrameHash, int], Dict[FrameHash, Tuple[FrameHash, GameAction]]]:
        distances: Dict[FrameHash, int] = {start: 0}
        predecessors: Dict[FrameHash, Tuple[FrameHash, GameAction]] = {}
        queue: deque[FrameHash] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor, action in adjacency.get(node, []):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[node] + 1
                predecessors[neighbor] = (node, action)
                queue.append(neighbor)
        return distances, predecessors

    def _recover_path(
        self,
        predecessors: Dict[FrameHash, Tuple[FrameHash, GameAction]],
        start: FrameHash,
        goal: FrameHash,
    ) -> List[GameAction]:
        actions: List[GameAction] = []
        node = goal
        while node != start:
            previous = predecessors.get(node)
            if previous is None:
                return []
            parent, action = previous
            actions.append(action)
            node = parent
        actions.reverse()
        return actions

    def _frontier_states(self) -> Iterable[FrameHash]:
        """Yield states that have unexplored (None) transitions."""
        for state in self._discovered_states():
            if self._is_blocked(state):
                continue
            record = self._state_graph.get(state)
            if record is None:
                # State discovered as transition target but never visited - it's a frontier
                yield state
                continue
            # Check if any recorded transition is unexplored (None)
            if any(target is None for target in record.transitions.values()):
                yield state

    def _is_action_known(self, state: FrameHash, action: GameAction) -> bool:
        """Check if an action has been explored (has a non-None target)."""
        record = self._state_graph.get(state)
        if record is None:
            return False
        key = transition_key_from_action(action)
        target = record.transitions.get(key)
        # Action is "known" only if it exists AND has been explored (not None)
        return target is not None

    def _is_blocked(self, state: FrameHash) -> bool:
        record = self._state_graph.get(state)
        return bool(record and record.is_game_over)
