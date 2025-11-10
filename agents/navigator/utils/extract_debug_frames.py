#!/usr/bin/env python3
"""
extract_debug_frames.py – export selected ARC‑AGI‑3 frames as PNGs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Iterable, Optional
from types import ModuleType

RECORDINGS_DIR = Path("recordings")
OUTPUT_DIR = Path("debug_frames")
FRAME_RANGE: Optional[range] = None
SCALE = 6


def find_latest_recording() -> Path:
    candidates = list(RECORDINGS_DIR.glob("*.recording.jsonl"))
    if not candidates:
        raise RuntimeError(f"no recordings found in {RECORDINGS_DIR}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_frame_viewer() -> ModuleType:
    """Import agents.navigator.frame_viewer without pulling in optional deps."""
    if "agents" not in sys.modules:
        pkg = types.ModuleType("agents")
        pkg.__path__ = [str(Path("agents").resolve())]
        sys.modules["agents"] = pkg
    if "agents.navigator" not in sys.modules:
        sub = types.ModuleType("agents.navigator")
        sub.__path__ = [str(Path("agents/navigator").resolve())]
        sys.modules["agents.navigator"] = sub

    module_path = Path("agents/navigator/frame_viewer.py").resolve()
    spec = importlib.util.spec_from_file_location(
        "agents.navigator.frame_viewer", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load frame_viewer module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agents.navigator.frame_viewer"] = module
    spec.loader.exec_module(module)

    return module


def iter_frames_with_layers(record_path: Path) -> Iterable[list[list[list[int]]]]:
    with record_path.open() as fh:
        for raw_line in fh:
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            frame_layers = data.get("data", {}).get("frame")
            if not frame_layers:
                continue
            yield frame_layers


def main() -> None:
    viewer = _load_frame_viewer()
    record_path = find_latest_recording()
    print(f"using recording: {record_path}")
    frames = list(iter_frames_with_layers(record_path))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = FRAME_RANGE if FRAME_RANGE is not None else range(len(frames))
    for idx in indices:
        if idx >= len(frames):
            print(f"frame {idx} out of range (only {len(frames)} frames available); stopping")
            break
        frame_layers = frames[idx]
        out_path = OUTPUT_DIR / f"frame_{idx:04d}.png"
        viewer.save_png(frame_layers[-1], out_path, scale=SCALE)
        print(out_path)


if __name__ == "__main__":
    main()
