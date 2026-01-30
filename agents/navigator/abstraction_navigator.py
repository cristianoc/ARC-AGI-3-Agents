"""AbstractionNavigator (game-specific wrapper).

This module is your starting point to configure per-game abstractions for the
abstraction navigator. Follow the steps below to implement energy measurement,
define the frame hashing mask, and register any additional detectors. These
instructions are self-contained so you can complete the setup without
referencing other files.

You must provide:
  1) Energy HUD measurement (required by default)
     - Implement `measure_energy(frame) -> EnergyHudMeasurement | None`.
     - It should detect the energy UI, returning:
         * `value`: current energy as a non-negative integer
         * `mask`: a sequence of rectangles `(y0, y1, x0, x1)` that cover
           the HUD area(s). Use multiple rectangles if the HUD is disjoint.
     - Be resilient to transient frames. When no HUD is visible, return None.

  2) Optional user abstractions
     - Extend `USER_ABSTRACTIONS` with `(name, detector)` pairs.
     - Each `detector(frame)` returns a structured value or `None`. Non-None
       values are attached to the `FrameAbstraction` under `name`.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from ..agent import Agent
from .abstractions import USER_ABSTRACTIONS, AbstractionDetector
from .base_navigator import BaseAbstractionNavigator
from .types import Color, EnergyHudMeasurement, Frame, MaskRect

# ---------------------------------------------------------------------------
# Game-specific abstractions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundingBox:
    min_y: int
    max_y: int
    min_x: int
    max_x: int

    @property
    def height(self) -> int:
        return self.max_y - self.min_y + 1

    @property
    def width(self) -> int:
        return self.max_x - self.min_x + 1


@dataclass(frozen=True)
class PlayerDetection:
    """Detect the player avatar sprite footprint."""

    center: tuple[float, float]
    pixel_count: int
    bbox: BoundingBox


def detect_clickable_squares_vc33(frame: Frame) -> list[tuple[int, int]]:
    """Detect blue clickable squares, returning their top-left corners."""
    arr = np.asarray(frame, dtype=np.int16)
    blue = arr == int(Color.BLUE)
    blue[:2, :] = False  # exclude HUD

    # Top-left corner: blue pixel with no blue above and no blue to the left
    above = np.pad(blue, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
    left = np.pad(blue, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
    corners = blue & ~above & ~left

    ys, xs = np.where(corners)
    return [(int(x), int(y)) for x, y in zip(xs, ys)]


def detect_player(frame_cells: Frame) -> Optional[PlayerDetection]:
    positions: list[tuple[int, int]] = []
    for y, row in enumerate(frame_cells):
        for x, cell in enumerate(row):
            if cell == Color.ORANGE:
                positions.append((y, x))

    if not positions or len(positions) > 256:
        return None

    min_y = min(y for y, _ in positions)
    max_y = max(y for y, _ in positions)
    min_x = min(x for _, x in positions)
    max_x = max(x for _, x in positions)

    height = max_y - min_y + 1
    width = max_x - min_x + 1
    if height > 16 or width > 16:
        return None

    total_y = sum(y for y, _ in positions)
    total_x = sum(x for _, x in positions)
    count = len(positions)
    center = (total_y / count, total_x / count)

    bbox = BoundingBox(
        min_y=min_y,
        max_y=max_y,
        min_x=min_x,
        max_x=max_x,
    )
    return PlayerDetection(center=center, pixel_count=count, bbox=bbox)


LS20_ENERGY_HUD_MASK: tuple[MaskRect, ...] = (MaskRect(y0=1, y1=2, x0=2, x1=45),)
AS66_ENERGY_HUD_MASK: tuple[MaskRect, ...] = (
    MaskRect(y0=0, y1=0, x0=0, x1=63),   # top HUD row
    MaskRect(y0=0, y1=63, x0=0, x1=0),   # left border column
    MaskRect(y0=0, y1=63, x0=63, x1=63), # right border column
)
VC33_ENERGY_HUD_MASK: tuple[MaskRect, ...] = (MaskRect(y0=0, y1=0, x0=0, x1=63),)


def _measure_energy_ls20(frame: Frame) -> Optional[EnergyHudMeasurement]:
    """Extract current energy HUD state for this game.

    Contract for the per-game implementation:
      - Input: `frame` is a 2D grid of cell values.
      - Output (when the HUD is visible): `EnergyHudMeasurement(value, mask)`
        with `mask` covering every `(y0, y1, x0, x1)` rectangle of the HUD.
    """
    row_index = 2
    if len(frame) <= row_index or not frame[row_index]:
        return None

    row = frame[row_index]
    _, _, x0, x1 = LS20_ENERGY_HUD_MASK[0]
    row_len = len(row)
    if row_len <= x0:
        return None
    upper_x = min(row_len - 1, x1)

    blocks: list[int] = []
    for x in range(x0, upper_x + 1, 2):
        value = row[x]
        if value in (Color.NEUTRAL, Color.PURPLE):
            blocks.append(value)
        elif blocks:
            break

    if any(v not in (Color.NEUTRAL, Color.PURPLE) for v in blocks):
        return None

    filled = sum(1 for v in blocks if v == Color.PURPLE)
    return EnergyHudMeasurement(value=filled, mask=LS20_ENERGY_HUD_MASK)


def _measure_energy_as66(frame: Frame) -> Optional[EnergyHudMeasurement]:
    """Detect energy by tallying orange tiles within the HUD mask corridors."""

    value = 0
    for rect in AS66_ENERGY_HUD_MASK:
        for y in range(rect.y0, rect.y1 + 1):
            for x in range(rect.x0, rect.x1 + 1):
                if frame[y][x] == Color.ORANGE:
                    value += 1
    return EnergyHudMeasurement(value=value, mask=AS66_ENERGY_HUD_MASK)


def _measure_energy_vc33(frame: Frame) -> Optional[EnergyHudMeasurement]:
    """Energy is the count of pink pixels in the top row."""

    top_row = frame[0]
    pink_pixels = sum(1 for value in top_row if value == Color.MAGENTA_LIGHT)
    return EnergyHudMeasurement(value=pink_pixels, mask=VC33_ENERGY_HUD_MASK)


def _measure_energy_for_game(game_id: str) -> Callable[[Frame], Optional[EnergyHudMeasurement]]:
    if game_id.startswith("ls20"):
        return _measure_energy_ls20
    if game_id.startswith("as66"):
        return _measure_energy_as66
    if game_id.startswith("vc33"):
        return _measure_energy_vc33
    return lambda frame: None


USER_ABSTRACTIONS.extend(
    [
        # Add new abstractions here: provide a (name, detector) pair. Detectors must
        # accept the frame and return either a structured result or None.
        ("player", detect_player),
    ]
)


def _user_abstractions_for_game(
    game_id: str,
) -> list[tuple[str, AbstractionDetector]]:
    abstractions = list(USER_ABSTRACTIONS)
    if game_id.startswith("vc33"):
        abstractions.append(("clickable", detect_clickable_squares_vc33))
    return abstractions


class AbstractionNavigator(BaseAbstractionNavigator, Agent):
    """Concrete navigator that wires up game-specific pieces.

    You can extend `USER_ABSTRACTIONS` and/or adjust masks and measurements
    above to customize behaviour for this game.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        game_id = str(kwargs.get("game_id", ""))

        measure_energy = _measure_energy_for_game(game_id)
        user_abstractions = _user_abstractions_for_game(game_id)

        super().__init__(
            *args,
            user_abstractions=user_abstractions,
            measure_energy=measure_energy,
            **kwargs,
        )


class AbstractionNavigatorNoEnergy(BaseAbstractionNavigator, Agent):
    """Variant of the navigator that disables energy measurement entirely."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        game_id = str(kwargs.get("game_id", ""))
        user_abstractions = _user_abstractions_for_game(game_id)
        super().__init__(
            *args,
            user_abstractions=user_abstractions,
            measure_energy=lambda frame: None,
            **kwargs,
        )
