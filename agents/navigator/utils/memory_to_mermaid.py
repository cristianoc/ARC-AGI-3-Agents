from __future__ import annotations

"""Utility to convert navigator memory into a Mermaid state graph."""

import json
import textwrap
import webbrowser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_MEMORY_PATH = Path(__file__).resolve().parents[1] / "memory" / "memory.json"
DEFAULT_HTML_PATH = Path("memory.html")


def _sanitize_node_id(state_hash: str) -> str:
    """Return a Mermaid-safe node identifier."""
    safe = "".join(char if char.isalnum() else "_" for char in state_hash)
    return f"n_{safe}"


def _format_level(level: Any) -> str:
    if level is None:
        return "?"
    return str(level)


def _sorted_levels(levels: Dict[Any, List[str]]) -> Iterable[Tuple[Any, List[str]]]:
    def key(item: Tuple[Any, List[str]]) -> Tuple[int, Any]:
        level, _ = item
        if isinstance(level, int):
            return (0, level)
        try:
            numeric = int(level)
            return (0, numeric)
        except (TypeError, ValueError):
            return (1, str(level))

    return sorted(levels.items(), key=key)


def _escape_label(label: str) -> str:
    return label.replace('"', "&quot;")


def _format_node_label(state: str, record: Dict[str, Any]) -> str:
    energy = record.get("energy")
    energy_text = f"E={energy}"
    # Count unexplored (null) transitions
    transitions = record.get("transitions", {})
    unexplored = sum(1 for t in transitions.values() if t is None)
    if unexplored > 0:
        return f"{state}<br/>{energy_text} 🔍{unexplored}"
    return f"{state}<br/>{energy_text}"


def memory_to_mermaid(memory_payload: Dict[str, Any]) -> str:
    """Generate a Mermaid graph from a memory payload."""
    state_graph: Dict[str, Dict[str, Any]] = memory_payload.get("state_graph", {})
    initial_states = {
        state for state, record in state_graph.items() if record.get("is_initial")
    }
    terminal_states = {
        state for state, record in state_graph.items() if record.get("is_terminal")
    }
    start_levels = {
        record.get("level")
        for record in state_graph.values()
        if record.get("is_initial")
    }
    final_levels = {
        record.get("level")
        for record in state_graph.values()
        if record.get("is_terminal")
    }

    levels: Dict[Any, List[str]] = {}
    for state, record in state_graph.items():
        level = record.get("level", "unknown")
        levels.setdefault(level, []).append(state)

    mermaid_lines: List[str] = [
        "graph LR",
        "    classDef defaultLevel fill:#f2f2f2,stroke:#999,stroke-width:1px;",
        "    classDef initial fill:#d5f5e3,stroke:#239b56,stroke-width:2px;",
        "    classDef terminal fill:#fbe5e5,stroke:#c0392b,stroke-width:2px;",
    ]

    for level, nodes in _sorted_levels(levels):
        label = f"Level {level}" if level != "unknown" else "Unknown Level"
        mermaid_lines.append(f"    subgraph {label}")
        for state in sorted(nodes):
            node_id = _sanitize_node_id(state)
            record = state_graph.get(state, {})
            label = _escape_label(_format_node_label(state, record))
            if state in initial_states:
                mermaid_lines.append(f"        {node_id}(\"{label}\")")
            elif state in terminal_states:
                mermaid_lines.append(f"        {node_id}((\"{label}\"))")
            else:
                mermaid_lines.append(f"        {node_id}[\"{label}\"]")

            classes: list[str] = ["defaultLevel"]
            if state in initial_states:
                classes.append("initial")
            if state in terminal_states:
                classes.append("terminal")
            for cls in classes:
                mermaid_lines.append(f"        class {node_id} {cls};")
        mermaid_lines.append("    end")

    # Include targets that do not yet have a record.
    known_states = set(state_graph.keys())
    referenced_states: set[str] = set()
    for state, record in state_graph.items():
        transitions = record.get("transitions", {})
        # Skip None (unexplored) targets
        referenced_states.update(str(target) for target in transitions.values() if target is not None)

    missing_states = referenced_states - known_states
    if missing_states:
        mermaid_lines.append("    subgraph Unknown Level")
        for state in sorted(missing_states):
            node_id = _sanitize_node_id(state)
            if state in initial_states:
                mermaid_lines.append(f"        {node_id}(\"{_escape_label(state)}\")")
            elif state in terminal_states:
                mermaid_lines.append(f"        {node_id}((\"{_escape_label(state)}\"))")
            else:
                mermaid_lines.append(f"        {node_id}[\"{_escape_label(state)}\"]")
            classes = ["defaultLevel"]
            if state in initial_states:
                classes.append("initial")
            if state in terminal_states:
                classes.append("terminal")
            for cls in classes:
                mermaid_lines.append(f"        class {node_id} {cls};")
        mermaid_lines.append("    end")

    def state_level(state: str) -> Any:
        return state_graph.get(state, {}).get("level", "unknown")

    for state in sorted(state_graph.keys()):
        record = state_graph[state]
        src_id = _sanitize_node_id(state)
        src_level = state_level(state)
        transitions = record.get("transitions", {})
        for action, target in sorted(transitions.items()):
            # Skip unexplored (null) transitions - they're shown in node annotation
            if target is None:
                continue
            dst_id = _sanitize_node_id(target)
            mermaid_lines.append(f"    {src_id} -- {action} --> {dst_id}")
            dst_level = state_level(target)
            if src_level != dst_level:
                mermaid_lines.append(
                    f"    %% cross-level: {state}({_format_level(src_level)}) -> "
                    f"{target}({_format_level(dst_level)})"
                )

    return "\n".join(mermaid_lines)


def load_memory(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"memory file not found: {path}")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to decode memory JSON {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"memory payload at {path} was not a JSON object")
    return raw


def build_html(mermaid_graph: str) -> str:
    return textwrap.dedent(
        f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <title>Navigator Memory Graph</title>
            <script type="module">
                import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
                mermaid.initialize({{ startOnLoad: true }});
                const graph = `{mermaid_graph}`;
                document.addEventListener("DOMContentLoaded", () => {{
                    const container = document.getElementById("graph");
                    container.innerHTML = `<div class="mermaid">${{graph}}</div>`;
                    mermaid.run();
                }});
            </script>
            <style>
                :root {{
                    color-scheme: dark light;
                }}
                body {{
                    margin: 0;
                    font-family: system-ui, sans-serif;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                }}
                #graph {{
                    flex: 1;
                    overflow: auto;
                    padding: 1rem;
                }}
                details {{
                    padding: 1rem;
                    background: #f5f5f5;
                    border-top: 1px solid #ddd;
                }}
                pre {{
                    white-space: pre-wrap;
                    word-break: break-word;
                }}
            </style>
        </head>
        <body>
            <div id="graph"></div>
            <details>
                <summary>Mermaid source</summary>
                <pre>{mermaid_graph}</pre>
            </details>
        </body>
        </html>
        """
    )


def main() -> None:
    memory_payload = load_memory(DEFAULT_MEMORY_PATH)
    mermaid = memory_to_mermaid(memory_payload)
    print(mermaid)

    html_content = build_html(mermaid)
    html_path = DEFAULT_HTML_PATH.resolve()
    html_path.write_text(html_content)
    webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()

