import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

if "agents" not in sys.modules:
    pkg = types.ModuleType("agents")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2])]
    pkg.__spec__ = importlib.machinery.ModuleSpec(
        name="agents", loader=None, is_package=True
    )
    sys.modules["agents"] = pkg

from ..nfr_planner import NearFrontierPlanner
from ..types import FrameHash, StateRecord
from ...structs import GameAction


@pytest.mark.unit
def test_frontier_skips_blocked_states():
    safe_state = FrameHash("safe_state")
    game_over_state = FrameHash("game_over_state")
    state_graph = {
        safe_state: StateRecord(transitions={"ACTION1": game_over_state}),
        game_over_state: StateRecord(is_game_over=True),
    }

    planner = NearFrontierPlanner(
        state_graph=state_graph,
    )

    frontier = set(planner._frontier_states([GameAction.ACTION1]))

    assert safe_state in frontier
    assert game_over_state not in frontier
