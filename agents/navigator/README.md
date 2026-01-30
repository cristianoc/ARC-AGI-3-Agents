# Navigator Agent

An exploration-focused agent that uses abstraction-driven state space navigation to solve ARC-AGI-3 games.

## Architecture

```
navigator/
├── abstraction_navigator.py  # Game-specific wrapper with energy detection & detectors
├── base_navigator.py         # Core navigation logic (game-agnostic)
├── nfr_planner.py           # Near-Frontier Planner for action selection
├── types.py                 # Type definitions, Color palette, Memory persistence
├── grid_hash.py             # Frame hashing with HUD masking
├── frame_viewer.py          # PNG export for debugging
├── abstractions.py          # User abstraction registry
├── memory/                  # Persistent state graph storage
├── utils/
│   ├── extract_debug_frames.py  # Export frames as PNGs
│   └── memory_to_mermaid.py     # Generate state graph diagrams
├── tests/
└── docs/
    └── nfr_planner.pdf      # Algorithm documentation
```

## Key Concepts

### State Graph
The navigator builds a graph of game states by hashing frames (with energy HUD masked out). Each state records:
- Transitions to other states via actions
- Level number and energy value
- Terminal/initial state markers

### Near-Frontier Planner (NFR)
Selects actions by finding paths to unexplored "frontier" states—states with untried actions. See `docs/nfr_planner.pdf` for the algorithm.

### Energy Detection
Games with energy HUDs require a `measure_energy(frame)` function that returns:
- `value`: current energy as an integer
- `mask`: rectangles covering the HUD (excluded from frame hashing)

Currently supported games: `ls20`, `as66`, `vc33`

## Usage

```bash
# Run locally (default, fast, no API key needed)
uv run main.py --agent=abstractionnavigator --game=vc33

# Run for more steps (default: 60)
uv run main.py --agent=abstractionnavigator --game=vc33 --steps=1000

# Run online (leaderboard, shareable replays)
uv run main.py --agent=abstractionnavigator --game=vc33 --mode=online

# Run without energy tracking
uv run main.py --agent=abstractionnavigatornoenergy --game=ls20
```

## Debugging Tools

```bash
# Export frames from latest recording as PNGs
uv run python agents/navigator/utils/extract_debug_frames.py

# Generate Mermaid state diagram from memory
uv run python agents/navigator/utils/memory_to_mermaid.py
```

## Adding a New Game

1. Edit `abstraction_navigator.py`:
   - Add energy HUD mask constant (e.g., `NEWGAME_ENERGY_HUD_MASK`)
   - Implement `_measure_energy_newgame(frame)` function
   - Register in `_measure_energy_for_game()` dispatcher

2. Optionally add game-specific detectors to `USER_ABSTRACTIONS` in `abstractions.py`

