"""Effective-throughput measurement for the bottleneck-balance gate (guidance Gate B0-2).

Geometry distances are only a proxy; what matters is the *throughput crossing*:

    mu_I_weak / 3  <  mu_S  <  mu_I_strong / 3

with mu_I the aggregate ingredient-placement rate and mu_S the delivery rate of a service
specialist. Scripted role controllers (no learning) run a fixed team on a map and the per-step event
rates are reported.

Usage:
  python measure_throughput.py --layout-file layouts/three_arm_b1.txt --episodes 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overcooked_v2_ff_gate"))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.common import DynamicObject  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from generate_demos import (  # noqa: E402
    MOVE_VECTORS,
    approach_and_interact,
    object_cells,
    pot_cell,
    walkable,
)


def ingredient_count(mask: int) -> int:
    """Ingredients in a pot bitmask: two bits per ingredient, PLATE=1, COOKED=2."""
    count, m = 0, int(mask) >> 2
    while m:
        count += m & 0x3
        m >>= 2
    return count


def step_team(env, state, roles, facing, block_aware=True):
    """One scripted step with teammates as dynamic obstacles and a deterministic sidestep.

    The demo generator already does this; the measurement harness must match it, otherwise narrow
    corridors change the measured quantity (mask, blocked) -> stall instead of throughput.
    """
    occupied = {tuple(int(v) for v in (state.agents.pos.x[i], state.agents.pos.y[i])): i
                for i in range(3)}
    acts = []
    for i in range(3):
        here = (int(state.agents.pos.x[i]), int(state.agents.pos.y[i]))
        blocked = frozenset(c for c, o in occupied.items() if o != i) if block_aware else frozenset()
        acts.append(roles[i](state, i, facing[i], blocked, here))
    return acts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-file", default="layouts/three_arm_b1.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=50000,
                        help="reset seed; run the script several times with different seeds and "
                             "average the JSONs (a single service cycle is noisy)")
    parser.add_argument("--repeats", type=int, default=6,
                        help="solo/server_cycle: independent episodes averaged (one cycle is noisy)")
    parser.add_argument("--mode", choices=["team", "server_cycle", "solo"], default="team",
                        help="team = realised rates in a full scripted team; server_cycle = the "
                             "service specialist's own cycle capacity once a pot is ready (a "
                             "bottleneck-free measure, which is what the balance gate needs)")
    parser.add_argument("--out", default=None)
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
    piles = [objects[n][0] for n in sorted(objects) if n.startswith("ingredient")]
    pots = objects["pot"]
    plates = objects["plate_pile"]
    pot_cells = [(x, y) for x, y in pots]

    def action_for(state, agent, role, facing, blocked):
        here = (int(state.agents.pos.x[agent]), int(state.agents.pos.y[agent]))
        inventory = int(state.agents.inventory[agent])
        if role in ("runnerA", "runnerB"):
            pot = pots[0] if role == "runnerA" else pots[1 % len(pots)]
            target = (min(piles, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                      if inventory == 0 else pot)
        else:
            if inventory == 0:
                target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            elif int(inventory) & int(DynamicObject.COOKED):
                target = objects["goal"][0]
            elif int(inventory) == int(DynamicObject.PLATE):
                cooked = [p for p in pots if pot_cell(state, p)[0] & int(DynamicObject.COOKED)]
                target = (min(cooked, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                          if cooked else min(pots, key=lambda p: abs(p[0] - here[0])
                                             + abs(p[1] - here[1])))
            else:
                target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
        return approach_and_interact(mask, here, facing, target, blocked)

    if args.mode == "solo":
        results = []
        # P1 ground truth: one scripted agent does everything (fetch x3 -> plate -> dish -> deliver),
        # the other two stay put. The static checker is only a pre-filter; this is the real number.
        obs, state = env.reset(jax.random.PRNGKey(args.seed))
        facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
        first_soup = None
        deliveries = 0
        step = 0
        while step < args.max_steps:
            here = (int(state.agents.pos.x[0]), int(state.agents.pos.y[0]))
            inv = int(state.agents.inventory[0])
            wall_cells = [(x, y) for y in range(mask.shape[0]) for x in range(mask.shape[1])
                          if not mask[y, x]]
            if inv == 0:
                needs = [p for p in pots if ingredient_count(pot_cell(state, p)[0]) < 3]
                target = min(piles, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1])) \
                    if needs else min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            elif (inv & int(DynamicObject.COOKED)) == 0 and inv != int(DynamicObject.PLATE):
                # holding an ingredient: place it if a pot still needs one, otherwise drop it at a
                # wall (Overcooked allows dropping) - without this the single agent stalls forever
                needs = [p for p in pots if ingredient_count(pot_cell(state, p)[0]) < 3]
                target = (min(needs, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                          if needs else min(wall_cells,
                                           key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1])))
            elif inv & int(DynamicObject.COOKED):
                target = objects["goal"][0]
            elif inv == int(DynamicObject.PLATE):
                cooked = [p for p in pots if pot_cell(state, p)[0] & int(DynamicObject.COOKED)]
                full = [p for p in pots if ingredient_count(pot_cell(state, p)[0]) >= 3]
                target = (cooked[0] if cooked else (full[0] if full
                          else min(pots, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))))
            else:
                target = min(pots, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            a = approach_and_interact(mask, here, facing[0], target, frozenset())
            if a in MOVE_VECTORS:
                facing[0] = a
            obs, state, reward, done, info = env.step(
                jax.random.PRNGKey(step + 1), state,
                {env.agents[0]: jnp.int32(a), env.agents[1]: jnp.int32(4),
                 env.agents[2]: jnp.int32(4)})
            if bool(np.asarray(state.new_correct_delivery)):
                deliveries += 1
                if first_soup is None:
                    first_soup = step + 1
                    break
            step += 1
            if bool(np.asarray(done["__all__"])):
                break
        results.append(first_soup)
        report = {"layout": str(layout_path), "mode": "solo", "seed": args.seed,
                  "first_soup_steps": results,
                  "first_soup_step_mean": float(np.mean([r for r in results if r])) if any(results)
                  else None,
                  "soups_per_320_if_alone": (320.0 / float(np.mean([r for r in results if r]))
                                             if any(results) else 0.0)}
        print(json.dumps(report, indent=2))
        out = Path(args.out) if args.out else (HERE / "results" /
                                               f"solo_{layout_path.stem}.json")
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[throughput] wrote {out}")
        return

    if args.mode == "server_cycle":
        # feed the pot with scripted runners until something is cooked, then measure the server's
        # own cycle: plate -> dish -> goal -> back to plates
        obs, state = env.reset(jax.random.PRNGKey(args.seed))
        facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
        ready_step = None
        step = 0
        while step < args.max_steps and ready_step is None:
            occupied = {tuple(int(v) for v in (state.agents.pos.x[i], state.agents.pos.y[i])): i
                        for i in range(3)}
            acts = []
            for i in range(2):
                here = (int(state.agents.pos.x[i]), int(state.agents.pos.y[i]))
                blocked = frozenset(c for c, o in occupied.items() if o != i)
                acts.append(action_for(state, i, ("runnerA", "runnerB")[i], facing[i], blocked))
            acts.append(4)
            for i in range(2):
                if acts[i] in MOVE_VECTORS:
                    facing[i] = acts[i]
            obs, state, reward, done, info = env.step(
                jax.random.PRNGKey(step + 1), state,
                {env.agents[i]: jnp.int32(acts[i]) for i in range(3)})
            if any(pot_cell(state, p)[0] & int(DynamicObject.COOKED) for p in pots):
                ready_step = step
            step += 1
        start_step = step
        deliveries = 0
        first_delivery_step = None
        returned_step = None
        while step < args.max_steps:
            occupied = {tuple(int(v) for v in (state.agents.pos.x[i], state.agents.pos.y[i])): i
                        for i in range(3)}
            # runners stay put: a single service cycle must not be contaminated by upstream
            # production or by the twenty-step cook timer
            here = (int(state.agents.pos.x[2]), int(state.agents.pos.y[2]))
            blocked = frozenset(c for c, o in occupied.items() if o != 2)
            a_server = action_for(state, 2, "server", facing[2], blocked)
            if a_server == 4:
                legal = [d for d, (dx, dy) in MOVE_VECTORS.items()
                         if mask[here[1] + dy, here[0] + dx]
                         and (here[0] + dx, here[1] + dy) not in blocked]
                if legal:
                    a_server = legal[0]
            if a_server in MOVE_VECTORS:
                facing[2] = a_server
            obs, state, reward, done, info = env.step(
                jax.random.PRNGKey(step + 1), state,
                {env.agents[0]: jnp.int32(4), env.agents[1]: jnp.int32(4),
                 env.agents[2]: jnp.int32(a_server)})
            if bool(np.asarray(state.new_correct_delivery)):
                deliveries += 1
                if first_delivery_step is None:
                    first_delivery_step = step + 1 - start_step
                if returned_step is None and int(state.agents.inventory[2]) == 0:
                    returned_step = step + 1 - start_step
                    break
            step += 1
            if bool(np.asarray(done["__all__"])):
                break
        span = max(step - start_step, 1)
        report = {"layout": str(layout_path), "mode": "server_cycle",
                  "pot_ready_step": start_step,
                  "steps_until_first_delivery": first_delivery_step,
                  "steps_until_delivery_and_return": returned_step,
                  "deliveries": deliveries,
                  "T_S_steps_per_soup": first_delivery_step,
                  "mu_S_cap_per_step": (1.0 / first_delivery_step
                                        if first_delivery_step else 0.0)}
        print(json.dumps(report, indent=2))
        out = Path(args.out) if args.out else (HERE / "results" /
                                               f"servercycle_{layout_path.stem}.json")
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[throughput] wrote {out}")
        return

    roles = ("runnerA", "runnerB", "server")
    eps_placements, eps_deliveries, eps_steps, eps_shaping = [], [], [], []
    for episode in range(args.episodes):
        obs, state = env.reset(jax.random.PRNGKey(args.seed + episode))
        facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
        placements = deliveries = shaping = steps = 0
        for step in range(args.max_steps):
            # teammates are dynamic obstacles, and a stalled agent sidesteps: without this the
            # scripted runners deadlock in the three-wide arm and the pipeline never completes
            occupied_now = {tuple(int(v) for v in (state.agents.pos.x[i], state.agents.pos.y[i])): i
                            for i in range(3)}
            actions = []
            for i in range(3):
                here = (int(state.agents.pos.x[i]), int(state.agents.pos.y[i]))
                blocked = frozenset(c for c, o in occupied_now.items() if o != i)
                a = action_for(state, i, roles[i], facing[i], blocked)
                if a == 4:
                    legal = [d for d, (dx, dy) in MOVE_VECTORS.items()
                             if mask[here[1] + dy, here[0] + dx]
                             and (here[0] + dx, here[1] + dy) not in blocked]
                    if legal:
                        a = legal[int(np.random.default_rng(step * 7 + i).integers(len(legal)))]
                actions.append(a)
            for i in range(3):
                if actions[i] in MOVE_VECTORS:
                    facing[i] = actions[i]
            pot_before = [pot_cell(state, p) for p in pots]
            obs, state, reward, done, info = env.step(
                jax.random.PRNGKey(step + 1), state,
                {env.agents[i]: jnp.int32(actions[i]) for i in range(3)})
            for before, after in zip(pot_before, [pot_cell(state, p) for p in pots]):
                # a placement is a +1 ingredient increment; raw->cooking / cooking->cooked / dish
                # taken all change the bitmask too and must NOT be counted
                if ingredient_count(after[0]) == ingredient_count(before[0]) + 1:
                    placements += 1
            deliveries += int(bool(np.asarray(state.new_correct_delivery)))
            shaping += int(float(np.asarray(info["shaped_reward"][env.agents[0]])) > 0)
            steps += 1
            if bool(np.asarray(done["__all__"])):
                break
        eps_placements.append(placements / max(steps, 1))
        eps_deliveries.append(deliveries / max(steps, 1))
        eps_shaping.append(shaping)
        eps_steps.append(steps)

    mu_i = float(np.mean(eps_placements))
    mu_s = float(np.mean(eps_deliveries))
    report = {
        "layout": str(layout_path), "episodes": args.episodes, "max_steps": args.max_steps,
        "team": {"runners": 2, "servers": 1},
        "mu_I_aggregate_placements_per_step": mu_i,
        "mu_I_per_runner": mu_i / 2,
        "mu_S_deliveries_per_step": mu_s,
        "window_for_mu_S": [mu_i * 0.8 / 3, mu_i / 3],
        "mu_S_in_window": bool(mu_i * 0.8 / 3 < mu_s < mu_i / 3),
        "mean_steps": float(np.mean(eps_steps)),
        "mean_shaping_events": float(np.mean(eps_shaping)),
    }
    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else (HERE / "results" /
                                           f"throughput_{layout_path.stem}.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[throughput] wrote {out}")


if __name__ == "__main__":
    main()
