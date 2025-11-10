"""
Utility helpers to preview or export ARC-AGI-3 navigator frames.

The implementation is adapted from the standalone `arc3_previewer.py` script,
exposed here so navigators can programmatically render frames for debugging.
All helpers operate on 64×64 integer grids (values 0..15).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, TYPE_CHECKING, cast

from .types import Frame

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from PIL.Image import Image as PILImage
else:
    PILImage = Any

import PIL.Image as pil_image_module
import PIL.ImageDraw as pil_image_draw_module

# 16-colour palette as RGB tuples.
PALETTE: list[tuple[int, int, int]] = [
    (240, 240, 240),   # 0 - bright white
    (96, 224, 64),     # 1 - bright green
    (120, 196, 228),   # 2 - light blue
    (101, 102, 101),   # 3 - medium grey
    (50, 51, 50),      # 4 - dark grey
    (40, 40, 40),      # 5 - very dark grey
    (210, 63, 159),    # 6 - magenta/pink
    (255, 220, 64),    # 7 - bright yellow
    (229, 77, 61),     # 8 - coral red
    (71, 144, 248),    # 9 - medium blue
    (154, 214, 238),   # 10 - sky blue
    (248, 221, 74),    # 11 - yellow
    (255, 132, 0),     # 12 - orange
    (134, 33, 51),     # 13 - dark red/maroon
    (214, 214, 214),   # 14 - light grey
    (153, 90, 208),    # 15 - purple
]

def _assert_frame_shape(frame: Frame) -> None:
    if len(frame) != 64:
        raise ValueError("Frame must have 64 rows")
    for row in frame:
        if len(row) != 64:
            raise ValueError("Frame rows must have 64 columns")
        for value in row:
            if not isinstance(value, int):
                raise ValueError("Frame values must be integers")
            if not 0 <= value <= 15:
                raise ValueError(f"Frame value {value} outside ARC-AGI-3 range 0..15")


def frame_to_image(
    frame: Frame,
    palette: Sequence[tuple[int, int, int]] = PALETTE,
    *,
    scale: int = 8,
    grid: int | None = None,
    grid_color: tuple[int, int, int] = (32, 32, 32),
) -> PILImage:
    """Convert a single frame into a PIL image (or None when Pillow is unavailable)."""

    _assert_frame_shape(frame)
    assert pil_image_module is not None
    assert pil_image_draw_module is not None

    image = pil_image_module.new("P", (64, 64))
    palette_values: list[int] = []
    for r, g, b in palette:
        palette_values.extend([r, g, b])
    palette_values.extend([0, 0, 0] * (256 - len(palette_values) // 3))
    image.putpalette(palette_values)
    image.putdata([value for row in frame for value in row])

    if scale != 1:
        resample = getattr(pil_image_module, "NEAREST", None)
        image = image.resize((64 * scale, 64 * scale), resample=resample)

    if grid and grid > 0:
        rgb = image.convert("RGB")
        draw = pil_image_draw_module.Draw(rgb)
        width, height = rgb.size
        step = grid * scale
        for x in range(step, width, step):
            draw.line([(x, 0), (x, height)], fill=grid_color)
        for y in range(step, height, step):
            draw.line([(0, y), (width, y)], fill=grid_color)
        image = rgb.convert("P", colors=256)

    return cast("PILImage", image)


def save_png(
    frame: Frame,
    out_path: Path | str,
    palette: Sequence[tuple[int, int, int]] = PALETTE,
    *,
    scale: int = 8,
    grid: int | None = None,
    grid_color: tuple[int, int, int] = (32, 32, 32),
) -> Path:
    """Save a single frame to PNG."""

    out = Path(out_path)
    image = frame_to_image(frame, palette, scale=scale, grid=grid, grid_color=grid_color)
    if image is None:
        raise RuntimeError("Pillow unexpectedly unavailable during PNG export.")
    image.save(out)
    return out


def iter_frames_from_record(record_path: Path | str) -> Iterator[Frame]:
    """Yield frames from a navigator JSONL recording file."""

    path = Path(record_path)
    with path.open() as fh:
        import json

        for line in fh:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:  # pragma: no cover - diagnostic
                continue
            frame_layers = data.get("data", {}).get("frame")
            if not frame_layers:
                continue
            frame = frame_layers[0]
            _assert_frame_shape(frame)
            yield frame
