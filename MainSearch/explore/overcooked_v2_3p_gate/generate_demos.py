"""Generate scripted three-agent demonstrations on the Three-Arm Kitchen (BC bootstrap data).

Why: from-scratch PPO never triggers a single shaped reward on this map (3M steps, 0.0000 shaped),
and V2's reward table has no "pick up ingredient" term, so the first reward event needs a rare
conjunction. Behaviour cloning from scripted experts is the cheapest way to bootstrap.

Design (follows the external guidance):
  * Roles: two ingredient runners (one per pot) + one server (plate -> dish -> delivery). All six
    role permutations across the agent slots are used, so the cloned policy cannot learn
    "agent i is always the runner" - that would erase the structure the Oracle Gate probes.
  * The controller replans every step with teammates as dynamic obstacles and tracks each agent's
    facing explicitly (V2 moves are absolute-direction turn-and-step), so agents do not deadlock in
    the one-wide corridors.
  * Perturbations: random start delays, occasional detours and pauses, so the clone learns
    recovery rather than one fixed cooperation script.

Usage:
  python generate_demos.py --episodes 300 --max-steps 240 --out demos/three_arm_demos.npz
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.common import DynamicObject  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from validate_three_arm_route import object_cells, walkable  # noqa: E402

RIGHT, DOWN, LEFT, UP, STAY, INTERACT = 0, 1, 2, 3, 4, 5
MOVE_VECTORS = {RIGHT: (1, 0), DOWN: (0, 1), LEFT: (-1, 0), UP: (0, -1)}
# six distinct role assignments: which agent fetches for pot 0, for pot 1, and which one serves
ROLE_PERMUTATIONS = list(itertools.permutations(["runnerA", "runnerB", "server"]))


def bfs(mask: np.ndarray, start, goal, blocked=frozenset()):
    """(first_action, distance) of a shortest path, or None. `blocked` holds teammate cells."""
    if start == goal:
        return (None, 0)
    prev = {start: (None, None)}
    queue = collections.deque([start])
    while queue:
        node = queue.popleft()
        for action, (dx, dy) in MOVE_VECTORS.items():
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[1] < mask.shape[0] and 0 <= nxt[0] < mask.shape[1]):
                continue
            if not mask[nxt[1], nxt[0]] or nxt in prev:
                continue
            if nxt in blocked and nxt != goal:
                continue
            prev[nxt] = (node, action)
            if nxt == goal:
                actions = []
                cur = nxt
                while prev[cur][0] is not None:
                    parent, act = prev[cur]
                    actions.append(act)
                    cur = parent
                actions.reverse()
                return (actions[0], len(actions))
            queue.append(nxt)
    return None


def approach_and_interact(mask, start, facing, target, blocked):
    """Action to take this step in order to end up interacting with `target`.

    V2 movement is absolute-direction: pressing the direction of an object only turns (the step is
    blocked by the object), and pressing it again while already facing performs the interaction.
    """
    tx, ty = target
    for direction, (dx, dy) in MOVE_VECTORS.items():
        if (tx - dx, ty - dy) == start:
            return direction if facing != direction else INTERACT
    best = None
    for direction, (dx, dy) in MOVE_VECTORS.items():
        cell = (tx - dx, ty - dy)
        if not (0 <= cell[1] < mask.shape[0] and 0 <= cell[0] < mask.shape[1]):
            continue
        if not mask[cell[1], cell[0]] or cell in blocked:
            continue
        path = bfs(mask, start, cell, blocked)
        if path is None:
            continue
        first, distance = path
        if best is None or distance < best[0]:
            best = (distance, first)
    return best[1] if best is not None else STAY


def pot_cell(state, pot):
    cell = state.grid[pot[1], pot[0]]
    return int(cell[1]), int(cell[2])  # ingredient bitmask, cook timer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--team", default=None,
                        help="explicit roles per slot, e.g. 'server,fetcher,server' (overrides "
                             "--role); used to probe whether role complementation pays off")
    parser.add_argument("--pause-per-agent", default=None,
                        help="comma-separated pause probabilities per slot, e.g. '0.0,0.3,0.0'")
    parser.add_argument("--role", default="mixed",
                        choices=["mixed", "fetcher", "server"],
                        help="mixed = two runners + one server (default); fetcher/server = one "
                             "role only, so the clone is a role-specialised teammate")
    parser.add_argument("--pause-prob", type=float, default=0.05,
                        help="per-step chance of a short pause (handicap knob: a 'weak' archetype)")
    parser.add_argument("--pause-steps", type=int, default=2)
    parser.add_argument("--out", default="demos/three_arm_demos.npz")
    args = parser.parse_args()

    layout_path = Path(args.layout_file)
    if not layout_path.is_absolute():
        layout_path = HERE / layout_path
    layout = Layout.from_string(layout_path.read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    mask = walkable(layout)
    objects = object_cells(layout)
    piles = [objects[name][0] for name in sorted(objects) if name.startswith("ingredient")]
    pots = objects["pot"]
    plates = objects["plate_pile"]

    obs_list, act_list, flag_list, ep_list, mask_list = [], [], [], [], []
    stats = {"episodes": 0, "deliveries": 0, "shaped_events": 0, "steps": 0, "deadlocks": 0}
    episode_outcomes = []

    for episode in range(args.episodes):
        rng = np.random.default_rng(episode)
        # the scripted team is always the working mixed team (two runners + one server); --role
        # only selects WHICH agents' transitions are recorded, so a role-specialised clone is
        # trained on its own role's behaviour inside a functioning team
        if args.team:
            # explicit role assignment: entry per agent slot ('fetcher', 'server', 'runnerA', ...)
            roles = tuple(args.team.split(","))
        else:
            roles = ROLE_PERMUTATIONS[episode % len(ROLE_PERMUTATIONS)]
        per_agent_pause = ([float(x) for x in args.pause_per_agent.split(",")]
                           if args.pause_per_agent else None)
        if args.role == "mixed":
            record = (True, True, True)
        elif args.role == "fetcher":
            record = tuple(role in ("runnerA", "runnerB") for role in roles)
        else:
            record = tuple(role == "server" for role in roles)
        # handicap: the recorded (target) agents randomly stall, modelling a weaker teammate

        assigned = {role: agent for agent, role in enumerate(roles)}
        pot_of_runner = {"runnerA": pots[0], "runnerB": pots[1 % len(pots)]}
        obs, state = env.reset(jax.random.PRNGKey(10_000 + episode))
        facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
        delay = [int(rng.integers(0, 6)) for _ in range(3)]
        pause = [0, 0, 0]
        episode_shaping = 0
        episode_deliveries = 0
        step = 0
        while step < args.max_steps:
            occupied = {tuple(int(v) for v in (state.agents.pos.x[i], state.agents.pos.y[i])): i
                        for i in range(3)}
            actions = []
            for agent in range(3):
                here = tuple(int(v) for v in (state.agents.pos.x[agent], state.agents.pos.y[agent]))
                blocked = frozenset(cell for cell, other in occupied.items() if other != agent)
                if delay[agent] > 0:
                    delay[agent] -= 1
                    actions.append(STAY)
                    continue
                if pause[agent] > 0:
                    pause[agent] -= 1
                    actions.append(STAY)
                    continue
                role = roles[agent]
                inventory = int(state.agents.inventory[agent])
                if record[agent] and pause[agent] == 0 and rng.random() < (
                        per_agent_pause[agent] if per_agent_pause else args.pause_prob):
                    pause[agent] = int(rng.integers(1, max(2, args.pause_steps + 1)))
                    actions.append(STAY)
                    continue
                if rng.random() < 0.02:
                    action = int(rng.integers(0, 4))
                elif role in ("runnerA", "runnerB", "fetcher"):
                    # a pure fetcher places ingredients in a pot but never touches plates/dishes
                    pot = (pot_of_runner[role] if role in pot_of_runner
                           else pots[agent % len(pots)])
                    target = (min(piles, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                              if inventory == 0 else pot)
                    action = approach_and_interact(mask, here, facing[agent], target, blocked)
                else:
                    if inventory == 0:
                        target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                        action = approach_and_interact(mask, here, facing[agent], target, blocked)
                    else:
                        holding_plate = inventory == DynamicObject.PLATE
                        holding_dish = bool(inventory & DynamicObject.COOKED)
                        if holding_dish:
                            action = approach_and_interact(mask, here, facing[agent],
                                                           objects["goal"][0], blocked)
                        elif holding_plate:
                            cooked = [p for p in pots if pot_cell(state, p)[0] & DynamicObject.COOKED]
                            if cooked:
                                target = min(cooked, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                            else:
                                target = min(pots, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                            action = approach_and_interact(mask, here, facing[agent], target, blocked)
                        else:
                            target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                            action = approach_and_interact(mask, here, facing[agent], target, blocked)
                if action in MOVE_VECTORS:
                    facing[agent] = action
                if action == STAY:
                    # no route (teammates can seal a one-wide corridor): take any legal sidestep
                    legal = [d for d, (dx, dy) in MOVE_VECTORS.items()
                             if mask[here[1] + dy, here[0] + dx]
                             and (here[0] + dx, here[1] + dy) not in blocked]
                    if legal:
                        action = legal[int(rng.integers(len(legal)))]
                        facing[agent] = action
                actions.append(action)

            view = np.stack([np.asarray(obs[env.agents[i]]) for i in range(3)])
            acts = {env.agents[i]: np.int32(actions[i]) for i in range(3)}
            obs, state, reward, done, info = env.step(jax.random.PRNGKey(step + 1), state, acts)
            obs_list.append(view.astype(np.int8))  # obs at time t paired with the action at time t
            act_list.append(np.asarray(actions, dtype=np.int8))
            shaped = float(np.asarray(info["shaped_reward"][env.agents[0]]))
            delivered = bool(np.asarray(state.new_correct_delivery))
            episode_shaping += int(shaped > 0)
            episode_deliveries += int(delivered)
            stats["shaped_events"] += int(shaped > 0)
            stats["deliveries"] += int(delivered)
            moved = [a in MOVE_VECTORS for a in actions]
            flag_list.append(np.array([int(a == INTERACT) for a in actions]
                                      + [int(shaped > 0), int(delivered)], dtype=np.int8))
            mask_list.append(np.array([int(r) for r in record], dtype=np.int8))
            ep_list.append(episode)
            step += 1
            if bool(np.asarray(done["__all__"])):
                break
        stats["steps"] += step
        stats["episodes"] += 1
        episode_outcomes.append({"episode": episode, "steps": step, "shaping": episode_shaping,
                                 "deliveries": episode_deliveries, "roles": list(roles)})

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        obs=np.stack(obs_list).astype(np.int8),      # (T, 3, 7, 7, 35)
        action=np.stack(act_list),                   # (T, 3)
        flags=np.stack(flag_list),                   # (T, 3 + 2)
        mask=np.stack(mask_list),                    # (T, 3) valid-for-BC entries
        episode=np.asarray(ep_list, dtype=np.int16),
        layout=json.dumps({"file": str(layout_path), "recipes": args.recipes}),
    )
    stats["episodes_with_delivery"] = sum(1 for e in episode_outcomes if e["deliveries"] > 0)
    stats["episodes_with_shaping"] = sum(1 for e in episode_outcomes if e["shaping"] > 0)
    stats["total_transitions"] = int(np.stack(act_list).size)
    stats["recorded_transitions"] = int(np.stack(mask_list).sum())
    stats["role"] = args.role
    stats["pause_prob"] = args.pause_prob
    stats["mean_deliveries_per_episode"] = round(
        sum(e["deliveries"] for e in episode_outcomes) / max(1, len(episode_outcomes)), 3)
    print(json.dumps(stats, indent=2))
    (out.with_suffix(".json")).write_text(
        json.dumps({"stats": stats, "episodes": episode_outcomes}, indent=2) + "\n", encoding="utf-8")
    print(f"[demos] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
