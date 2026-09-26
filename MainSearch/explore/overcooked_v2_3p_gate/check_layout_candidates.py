"""Static reachability check for candidate 3-player V2 adaptations."""

import json
from pathlib import Path

from jaxmarl.environments.overcooked_v2.common import StaticObject
from jaxmarl.environments.overcooked_v2.layouts import Layout, cramped_room_v2, grounded_coord_ring, test_time_wide


def component(grid, start):
    height, width = grid.shape
    visited = {start}
    pending = [start]
    while pending:
        x, y = pending.pop()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] == StaticObject.EMPTY:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    pending.append((nx, ny))
    return visited


def main():
    candidates = {
        "ring3": grounded_coord_ring.replace("W       W", "W   A   W", 1),
        "cramped3": cramped_room_v2.replace("W   R", "W A R", 1),
        "wide3": test_time_wide.replace("1    1", "1  A 1", 1),
    }
    results = {}
    for name, text in candidates.items():
        layout = Layout.from_string(text, possible_recipes=[[0, 0, 0], [1, 1, 1]])
        assert len(layout.agent_positions) == 3
        components = [component(layout.static_objects, start) for start in layout.agent_positions]
        results[name] = {
            "agent_starts": [list(start) for start in layout.agent_positions],
            "reachable_floor_cells": [len(cells) for cells in components],
            "all_share_component": all(cells == components[0] for cells in components),
        }
        print(f"{name}: {results[name]}", flush=True)
    output = Path(__file__).resolve().parent / "results" / "layout_candidates.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
