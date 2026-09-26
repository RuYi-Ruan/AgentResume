"""Static checker for three-player OvercookedV2 layouts (no training).

Why: on small maps one agent can complete the whole loop alone (that is exactly the flaw that
killed `ring3`), so a three-player layout only makes partner capability matter when *travel cost
dominates* - i.e. one agent needs so many steps per soup that three agents clearly out-produce it.

What it computes, per layout:
  * walkable mask (cells with `StaticObject.EMPTY`, as the environment defines movement),
  * connected components of walkable cells and which objects each agent can reach,
  * interaction step cost between every pair of "key cells" (agent starts, ingredient piles, pots,
    plate piles, goal, recipe indicator/button) using Dijkstra over (cell, facing) with
    move cost 1 and turn cost 1, plus 1 for the interact action,
  * a single-agent soup estimate: 3 ingredient trips + plate trip + cook time + delivery,
  * a rough three-agent pipeline estimate (three agents splitting the same legs),
  * crowding: walkable cells per agent.

Usage:
  python check_layout_3p.py --layout kitchen3 --out results/layout_check_kitchen3.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))

from jaxmarl.environments.overcooked_v2.common import StaticObject  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402

DIRS = {"right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}
DIR_INDEX = {name: i for i, name in enumerate(DIRS)}
TURN_LEFT = {0: 3, 1: 0, 2: 1, 3: 2}
TURN_RIGHT = {0: 1, 1: 2, 2: 3, 3: 0}


def walkable(layout: Layout) -> np.ndarray:
    return layout.static_objects == StaticObject.EMPTY


def components(mask: np.ndarray) -> np.ndarray:
    """Label connected components (4-neighbour) of True cells; 0 means not walkable."""
    labels = np.zeros(mask.shape, dtype=int)
    current = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if not mask[y, x] or labels[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
    return labels


def dijkstra(mask: np.ndarray, start: tuple[int, int], start_dir: int = 1):
    """Step cost (move=1, turn=1) from a start cell/facing to every (cell, facing)."""
    inf = float("inf")
    dist = {(*start, start_dir): 0}
    heap = [(0, start[0], start[1], start_dir)]
    while heap:
        d, y, x, dir_idx = heapq.heappop(heap)
        if d > dist.get((y, x, dir_idx), inf):
            continue
        for ndir in (TURN_LEFT[dir_idx], TURN_RIGHT[dir_idx]):
            key = (y, x, ndir)
            if d + 1 < dist.get(key, inf):
                dist[key] = d + 1
                heapq.heappush(heap, (d + 1, y, x, ndir))
        dy, dx = list(DIRS.values())[dir_idx]
        ny, nx = y + dy, x + dx
        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]:
            key = (ny, nx, dir_idx)
            if d + 1 < dist.get(key, inf):
                dist[key] = d + 1
                heapq.heappush(heap, (d + 1, ny, nx, dir_idx))
    return dist


def interact_cost(mask: np.ndarray, dist: dict, target: tuple[int, int]) -> float:
    """Min steps to stand next to `target` facing it and issue one interact action."""
    best = float("inf")
    tx, ty = target  # objects are stored as (x, y)
    for name, (dy, dx) in DIRS.items():
        # to face the target, the agent stands on the opposite side and faces towards it
        y, x = ty - dy, tx - dx
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]) or not mask[y, x]:
            continue
        want_dir = DIR_INDEX[name]
        cost = dist.get((y, x, want_dir), float("inf"))
        best = min(best, cost + 1)  # +1 for the interact action itself
    return best


def find_objects(layout: Layout):
    grid = layout.static_objects
    found: dict[str, list[tuple[int, int]]] = {}
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            cell = int(grid[y, x])
            name = None
            if StaticObject.is_ingredient_pile(cell):
                name = f"ingredient{cell - StaticObject.INGREDIENT_PILE_BASE}"
            elif cell == StaticObject.POT:
                name = "pot"
            elif cell == StaticObject.PLATE_PILE:
                name = "plate_pile"
            elif cell == StaticObject.GOAL:
                name = "goal"
            elif cell == StaticObject.RECIPE_INDICATOR:
                name = "recipe_indicator"
            elif cell == StaticObject.BUTTON_RECIPE_INDICATOR:
                name = "button"
            if name:
                found.setdefault(name, []).append((x, y))
    return found


def object_interact_states(mask: np.ndarray, target: tuple[int, int]):
    """States from which one interact action hits `target` (agent adjacent, facing it)."""
    states = []
    tx, ty = target
    for name, (dy, dx) in DIRS.items():
        y, x = ty - dy, tx - dx
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
            states.append((y, x, DIR_INDEX[name]))
    return states


def pairwise_object_costs(mask: np.ndarray, objects: dict) -> dict:
    """cost[(src, src_pos)][(dst, dst_pos)] = steps from interacting with src to interacting with dst."""
    costs: dict = {}
    for src_name, positions in objects.items():
        for src_pos in positions:
            best = {}
            for state in object_interact_states(mask, src_pos):
                dist = dijkstra(mask, (state[0], state[1]), state[2])
                for dst_name, dst_positions in objects.items():
                    for dst_pos in dst_positions:
                        if (dst_name, tuple(dst_pos)) == (src_name, tuple(src_pos)):
                            continue
                        key = (dst_name, tuple(dst_pos))
                        cand = min(
                            dist.get(s, float("inf")) + 1 for s in object_interact_states(mask, dst_pos)
                        )
                        best[key] = min(best.get(key, float("inf")), cand)
            costs[(src_name, tuple(src_pos))] = best
    return costs



def analyse(layout: Layout, cook_time: int = 20, max_steps: int = 400) -> dict:
    mask = walkable(layout)
    labels = components(mask)
    objects = find_objects(layout)
    starts = list(layout.agent_positions)

    regions = {int(r): int((labels == r).sum()) for r in np.unique(labels) if r}
    agent_regions = [int(labels[y, x]) for x, y in starts]
    per_agent = []
    for (ax, ay) in starts:
        dist = dijkstra(mask, (ay, ax))
        costs = {name: [interact_cost(mask, dist, pos) for pos in positions]
                 for name, positions in objects.items()}
        per_agent.append({"start": [ax, ay], "interact_costs": costs})

    ingredient_names = sorted(n for n in objects if n.startswith("ingredient"))
    recipe_names = ingredient_names[:3]
    report = {
        "walkable_cells": int(mask.sum()),
        "walkable_per_agent": round(float(mask.sum()) / max(len(starts), 1), 1),
        "regions": regions,
        "agent_regions": agent_regions,
        "agent_positions": [list(p) for p in starts],
        "objects": {k: [list(p) for p in v] for k, v in objects.items()},
        "object_reachable_by_agent": [
            {name: [bool(c < float("inf")) for c in costs] for name, costs in a["interact_costs"].items()}
            for a in per_agent
        ],
        "per_agent_interact_costs": [a["interact_costs"] for a in per_agent],
    }

    # single-agent soup estimate: 3 round trips pile->pot, plate->pot twice, cook, deliver
    pair = pairwise_object_costs(mask, objects)
    solo_costs = {}
    for idx, a in enumerate(per_agent):
        start_costs = a["interact_costs"]

        def best(keys, from_key=None):
            if from_key is None:
                candidates = [min(start_costs.get(n, [float("inf")])) for n in keys]
            else:
                table = pair.get(from_key, {})
                candidates = [
                    table.get((n, tuple(pos)), float("inf"))
                    for n in keys for pos in objects.get(n, [])
                ]
            return min(candidates) if candidates else float("inf")

        pile_cost = min(
            pair.get((n, tuple(pos)), {}).get((pn, tuple(pp)), float("inf"))
            for n in recipe_names for pos in objects.get(n, [])
            for pn in ["pot"] for pp in objects.get(pn, [])
        )
        pot_to_plate = min(
            pair.get(("pot", tuple(pp)), {}).get((bn, tuple(bp)), float("inf"))
            for pp in objects.get("pot", []) for bn in ["plate_pile"] for bp in objects.get(bn, [])
        )
        pot_to_goal = min(
            pair.get(("pot", tuple(pp)), {}).get(("goal", tuple(gp)), float("inf"))
            for pp in objects.get("pot", []) for gp in objects.get("goal", [])
        )
        start_to_pile = best(recipe_names)
        if any(c == float("inf") for c in [pile_cost, pot_to_plate, pot_to_goal, start_to_pile]):
            solo_costs[f"agent{idx}"] = None
            continue
        soup_cost = start_to_pile + 6 * pile_cost + 2 * pot_to_plate + cook_time + pot_to_goal
        solo_costs[f"agent{idx}"] = {
            "start_to_pile": round(start_to_pile, 1),
            "pile_to_pot": round(pile_cost, 1),
            "pot_to_plate": round(pot_to_plate, 1),
            "pot_to_goal": round(pot_to_goal, 1),
            "first_soup_actions": round(soup_cost, 1),
            "soups_per_episode_if_alone": round(max_steps / soup_cost, 2) if soup_cost else None,
        }
    report["solo_estimates"] = solo_costs
    report["max_steps_used"] = max_steps
    report["notes"] = [
        "interact costs are steps including turning (move=1, turn=1) plus the interact action",
        "three-agent throughput is not computed here; the point is whether one agent alone is slow",
        "a layout is interesting when soups_per_400_steps_if_alone is well below the three-agent rate",
    ]
    return report


def make_layout_text(width: int, height: int, placements: dict[str, list[tuple[int, int]]]) -> str:
    """Generate an ASCII layout with walls on the border and EMPTY elsewhere."""
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for x in range(width):
        grid[0][x] = "W"
        grid[height - 1][x] = "W"
    for y in range(height):
        grid[y][0] = "W"
        grid[y][width - 1] = "W"
    for symbol, coords in placements.items():
        for (x, y) in coords:
            grid[y][x] = symbol
    return "\n" + "\n".join("".join(row) for row in grid) + "\n"


# Candidate three-player kitchens. Coordinates are (x, y) inside the wall border.
# Design intent: ingredient piles and the delivery point are far apart so that travel dominates,
# which is the only way a third agent adds real value (on small maps one agent self-serves).
CANDIDATES: dict[str, dict] = {
    # small: included as a reference for "one agent is enough"
    "kitchen3_small": dict(
        width=13, height=9,
        placements={
            "0": [(1, 1)], "1": [(11, 1)],
            "A": [(2, 3), (10, 3), (6, 7)],
            "P": [(4, 4), (8, 4)],
            "B": [(5, 4)], "X": [(11, 7)], "R": [(1, 7)], "L": [(2, 7)],
        },
        recipes=[[0, 0, 0], [1, 1, 1]],
    ),
    "kitchen3_mid": dict(
        width=19, height=13,
        placements={
            "0": [(1, 1)], "1": [(17, 1)],
            "A": [(3, 5), (15, 5), (9, 11)],
            "P": [(6, 6), (12, 6)],
            "B": [(7, 6), (11, 6)],
            "X": [(9, 1)], "R": [(1, 11)], "L": [(2, 11)],
        },
        recipes=[[0, 0, 0], [1, 1, 1]],
    ),
    "kitchen3_long": dict(
        width=25, height=15,
        placements={
            "0": [(1, 1)], "1": [(23, 1)],
            "A": [(3, 7), (21, 7), (12, 13)],
            "P": [(8, 8), (16, 8)],
            "B": [(9, 8), (15, 8)],
            "X": [(12, 1)], "R": [(1, 13)], "L": [(2, 13)],
        },
        recipes=[[0, 0, 0], [1, 1, 1]],
    ),
}


# Explicit ASCII candidate from external design guidance ("Three-Arm Kitchen").
THREE_ARM = """
WWWWWWWWWWWWWWWWWWW
W0 0WWWWP PWWWW  XW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWW   W
W              B BW
W   A    A    A   W
WWWWWWWWWWWWWWWWWWW
"""

CANDIDATES["three_arm"] = dict(
    ascii=THREE_ARM,
    recipes=[[0, 0, 0]],
)


# Fixed variant of the external Three-Arm design: plate piles moved into the right (delivery) arm
# and the bottom corridor opened up, so no agent start blocks the passage between arms.
THREE_ARM_V2 = """
WWWWWWWWWWWWWWWWWWW
W0 0WWWWP PWWWW  XW
W   WWWW   WWWWB BW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWW   W
W                 W
W  A     A     A  W
WWWWWWWWWWWWWWWWWWW
"""

CANDIDATES["three_arm_v2"] = dict(ascii=THREE_ARM_V2, recipes=[[0, 0, 0]])


# Bottleneck-balancing variant: ingredient piles moved to the BOTTOM of the left arm (short
# ingredient->pot leg) while the plate piles move to the bottom of the right arm and the goal stays
# at the top (long service leg), so the two roles' costs are comparable and the binding constraint
# can migrate with partner competence. P1 is kept by the long return paths, not by loading
# everything onto the ingredient arm.
THREE_ARM_BALANCED = """
WWWWWWWWWWWWWWWWWWW
W   WWWWP PWWWW  XW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W0 0WWWW   WWWWB BW
W                 W
W  A     A     A  W
WWWWWWWWWWWWWWWWWWW
"""

CANDIDATES["three_arm_balanced"] = dict(ascii=THREE_ARM_BALANCED, recipes=[[0, 0, 0]])


THREE_ARM_BALANCED2 = """
WWWWWWWWWWWWWWWWWWW
W   WWWWP PWWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWW   W
W0 0               W
W  A     A     A  W
WWWWWWWWWWWWWWWWWWX
"""
CANDIDATES["three_arm_balanced2"] = dict(ascii=THREE_ARM_BALANCED2, recipes=[[0, 0, 0]])


# Candidate B' #1: workload transfer, not cost reduction. Ingredient piles moved to the middle of
# the left arm (shorter fetch leg) and the delivery point moved to the BOTTOM-LEFT of the map, so the
# service specialist's steady-state cycle (plates -> pot -> goal -> back to plates) crosses the whole
# kitchen. Only two dog-legs, distinct wall shapes at each corner: no serpentine repetition, which
# would reintroduce observation aliasing for the feed-forward policy.
THREE_ARM_B1 = """
WWWWWWWWWWWWWWWWWWW
W   WWWWP PWWWW   W
W   WWWW   WWWWB BW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W0 0WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWW   W
W                 W
W  A     A     A  W
WWXWWWWWWWWWWWWWWWW
"""
CANDIDATES["three_arm_b1"] = dict(ascii=THREE_ARM_B1, recipes=[[0, 0, 0]])


# Candidate B' #2: keep the three-arm traffic structure of three_arm_v2 (left arm = ingredients,
# middle = pot hub, right = service) so no two roles share a corridor, but move the plate piles to
# the BOTTOM of the service arm so the steady-state service cycle is longer (target T_S ~ 72-90
# steps/soup, which is what the throughput window [0.8*mu_I/3, mu_I/3] requires).
THREE_ARM_B2 = """
WWWWWWWWWWWWWWWWWWW
W0 0WWWWP PWWWW  XW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWWB BW
W                 W
W  A     A     A  W
WWWWWWWWWWWWWWWWWWW
"""
CANDIDATES["three_arm_b2"] = dict(ascii=THREE_ARM_B2, recipes=[[0, 0, 0]])


# Candidate B' #3 (final allowed map): the service path stays entirely inside the service arm.
# Plates stay at the TOP of the right arm; the delivery point moves to the BOTTOM of the right arm,
# so the steady-state service cycle (plates -> pot -> goal -> back) is long while ingredient and
# service traffic never share a corridor.
THREE_ARM_B3 = """
WWWWWWWWWWWWWWWWWWW
W0 0WWWWP PWWWW   W
W   WWWW   WWWWB BW
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW   WWWW   W
W   WWWW R WWWW   W
W   WWWW   WWWW   W
W                 X
W  A     A     A  W
WWWWWWWWWWWWWWWWWWW
"""
CANDIDATES["three_arm_b3"] = dict(ascii=THREE_ARM_B3, recipes=[[0, 0, 0]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="kitchen3_mid", choices=sorted(CANDIDATES))
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-steps", type=int, default=320)
    args = parser.parse_args()

    spec = CANDIDATES[args.layout]
    if "ascii" in spec:
        text = spec["ascii"]
    else:
        text = make_layout_text(spec["width"], spec["height"], spec["placements"])
    layout = Layout.from_string(text, possible_recipes=spec["recipes"])
    report = analyse(layout, max_steps=args.max_steps)
    report["layout"] = args.layout
    report["num_agents"] = len(layout.agent_positions)
    print(f"layout {args.layout}: agents={report['num_agents']} walkable={report['walkable_cells']} "
          f"(per agent {report['walkable_per_agent']}) regions={report['regions']}")
    print(f"  agent regions: {report['agent_regions']}")
    for agent, est in report["solo_estimates"].items():
        print(f"  {agent}: {est}")
    out = Path(args.out) if args.out else HERE / "results" / f"layout_check_{args.layout}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[layout check] wrote {out}")


if __name__ == "__main__":
    main()
