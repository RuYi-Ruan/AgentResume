"""Scripted-route validation of a three-player OvercookedV2 layout (diagnostic only).

Purpose: before spending any training budget, check that the layout is *mechanically* solvable and
measure the real single-agent cost of one soup with a hand-planned route.

This is NOT an agent baseline: it drives one agent with a hard-coded route built from BFS paths
(moves only, because OvercookedV2 actions are absolute-direction moves that also set facing; a
blocked move still turns the agent, which is how you face an object before interacting).
The other agents stay put.

Usage:
  python validate_three_arm_route.py --layout-file layouts/three_arm.txt --max-steps 320
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.common import StaticObject  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
MOVE_VECTORS = {RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0), UP: (0, -1)}
FACING = {(1, 0): RIGHT, (0, 1): DOWN, (-1, 0): LEFT, (0, -1): UP}


def walkable(layout: Layout) -> np.ndarray:
    return layout.static_objects == StaticObject.EMPTY


def object_cells(layout: Layout) -> dict[str, list[tuple[int, int]]]:
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
            if name:
                found.setdefault(name, []).append((x, y))
    return found


def bfs_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """List of move actions from `start` to `goal` over walkable cells, or None."""
    if start == goal:
        return []
    prev: dict[tuple[int, int], tuple[tuple[int, int], int] | None] = {start: None}
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        for action, (dx, dy) in MOVE_VECTORS.items():
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[1] < mask.shape[0] and 0 <= nxt[0] < mask.shape[1]):
                continue
            if not mask[nxt[1], nxt[0]] or nxt in prev:
                continue
            prev[nxt] = (node, action)
            if nxt == goal:
                actions = []
                cur = nxt
                while prev[cur] is not None:
                    parent, act = prev[cur]
                    actions.append(act)
                    cur = parent
                return list(reversed(actions))
            queue.append(nxt)
    return None


def route_actions(mask: np.ndarray, start: tuple[int, int], target: tuple[int, int]):
    """Moves + bump + interact that take an agent from `start` to interacting with `target`.

    OvercookedV2 actions are absolute-direction moves that also set facing; a move into the object
    is blocked but still turns the agent towards it, so the last move before interacting is the
    direction pointing at the object.
    """
    tx, ty = target
    best = None
    for action, (dx, dy) in MOVE_VECTORS.items():
        ax, ay = tx - dx, ty - dy  # cell from which the agent faces the target
        if not (0 <= ay < mask.shape[0] and 0 <= ax < mask.shape[1]) or not mask[ay, ax]:
            continue
        moves = bfs_path(mask, start, (ax, ay))
        if moves is None:
            continue
        candidate = moves + [action, INTERACT]
        if best is None or len(candidate) < len(best):
            best = candidate
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-file", required=True)
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--agent", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    text = Path(args.layout_file).read_text(encoding="utf-8")
    layout = Layout.from_string(text, possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        agent_view_size=3,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        random_agent_positions=False,
        max_steps=args.max_steps,
    )
    obs, state = env.reset(jax.random.PRNGKey(0))
    mask = walkable(layout)
    objects = object_cells(layout)
    start = (int(state.agents.pos.x[args.agent]), int(state.agents.pos.y[args.agent]))
    print(f"layout: {len(layout.agent_positions)} agents, walkable {int(mask.sum())}")
    print(f"objects: { {k: v for k, v in objects.items()} }")
    print(f"agent {args.agent} starts at {start}, recipe {state.recipe}")

    ingredient = sorted(n for n in objects if n.startswith("ingredient"))[0]
    pile = objects[ingredient][0]
    pot = objects["pot"][0]
    plate = objects["plate_pile"][0]
    goal = objects["goal"][0]

    plan: list[tuple[str, list[int]]] = []
    here = start
    for trip in range(3):
        actions = route_actions(mask, here, pile)
        if actions is None:
            raise SystemExit("no route to ingredient pile")
        plan.append((f"trip{trip + 1}: to {ingredient}", actions))
        here = _advance(mask, here, actions)
        actions = route_actions(mask, here, pot)
        if actions is None:
            raise SystemExit("no route to pot")
        plan.append((f"trip{trip + 1}: to pot", actions))
        here = _advance(mask, here, actions)
    plan.append(("wait for cooking", [STAY] * 21))
    actions = route_actions(mask, here, plate)
    plan.append(("to plate pile", actions))
    here = _advance(mask, here, actions)
    actions = route_actions(mask, here, pot)
    plan.append(("to pot (pick dish)", actions))
    here = _advance(mask, here, actions)
    actions = route_actions(mask, here, goal)
    plan.append(("to goal", actions))

    total = sum(len(a) for _, a in plan)
    print(f"planned route: {len(plan)} legs, {total} actions (single agent, others stay put)")

    # drive the environment step by step
    per_leg = []
    step = 0
    delivered = None
    for name, actions in plan:
        for action in actions:
            if step >= args.max_steps:
                break
            acts = {a: np.int32(STAY) for a in env.agents}
            acts[env.agents[args.agent]] = np.int32(action)
            obs, state, reward, done, info = env.step(
                jax.random.PRNGKey(step + 1), state, acts
            )
            step += 1
            if bool(np.asarray(state.new_correct_delivery)):
                delivered = step
                break
        per_leg.append({"leg": name, "steps_used": step, "delivered_at": delivered})
        if delivered or step >= args.max_steps:
            break

    result = {
        "layout_file": args.layout_file,
        "agent": args.agent,
        "walkable_cells": int(mask.sum()),
        "planned_actions": total,
        "steps_executed": step,
        "correct_delivery_step": delivered,
        "max_steps": args.max_steps,
        "per_leg": per_leg,
    }
    print(json.dumps(result, indent=2))
    out = Path(args.out) if args.out else HERE / "results" / "route_validation_three_arm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"[route validation] wrote {out}")


def _advance(mask: np.ndarray, start: tuple[int, int], actions: list[int]) -> tuple[int, int]:
    """Replay moves (ignoring bump/interact) to know where the agent ends up."""
    x, y = start
    for action in actions:
        if action == INTERACT:
            continue
        dx, dy = MOVE_VECTORS[action]
        nx, ny = x + dx, y + dy
        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1] and mask[ny, nx]:
            x, y = nx, ny
    return (x, y)


if __name__ == "__main__":
    main()
