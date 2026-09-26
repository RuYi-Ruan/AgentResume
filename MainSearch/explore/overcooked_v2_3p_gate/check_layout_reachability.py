"""Diagnostic only: execute a fixed low-level route to one correct delivery.

This scripted route is not an agent baseline and is never used for training.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl

from jaxmarl.environments.overcooked_v2.layouts import Layout, grounded_coord_ring
from jaxmarl.environments.overcooked_v2.common import StaticObject


OUT = Path(__file__).resolve().parent / "results" / "layout_reachability.json"
STAY, INTERACT = 4, 5
RIGHT, DOWN, LEFT, UP = 0, 1, 2, 3


def route(ingredient):
    if ingredient == 0:
        to_pile, to_pot = [UP, LEFT], [RIGHT, DOWN]
        back_to_pile = [LEFT, LEFT]
        to_plate = [RIGHT, RIGHT, RIGHT]
        plate_to_pot = [LEFT, DOWN]
        pot_to_goal = [RIGHT, DOWN, RIGHT]
    else:
        to_pile, to_pot = [DOWN, LEFT], [RIGHT, UP]
        back_to_pile = [LEFT, LEFT]
        to_plate = [RIGHT, RIGHT, RIGHT]
        plate_to_pot = [LEFT, UP]
        pot_to_goal = [RIGHT, UP, RIGHT]
    actions = []
    actions += to_pile + [INTERACT]
    actions += to_pot + [INTERACT]
    for _ in range(2):
        actions += back_to_pile + [INTERACT]
        actions += to_pot + [INTERACT]
    actions += to_plate + [INTERACT]
    actions += plate_to_pot + [STAY] * 20 + [INTERACT]
    actions += pot_to_goal + [INTERACT]
    return actions


def main():
    layout = Layout.from_string(
        grounded_coord_ring.replace("W       W", "W   A   W", 1),
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    static = layout.static_objects
    access = []
    for agent_id, start in enumerate(layout.agent_positions):
        seen = {start}
        pending = [start]
        adjacent_objects = set()
        while pending:
            x, y = pending.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < layout.width and 0 <= ny < layout.height):
                    continue
                item = int(static[ny, nx])
                if item == StaticObject.EMPTY and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    pending.append((nx, ny))
                elif item != StaticObject.EMPTY:
                    adjacent_objects.add(item)
        row = {
            "agent": agent_id,
            "start": list(start),
            "reachable_floor_cells": len(seen),
            "adjacent_static_object_codes": sorted(adjacent_objects),
        }
        access.append(row)
        print(f"agent={agent_id} start={start} reachable_floor_cells={len(seen)}", flush=True)
    results = []
    for ingredient in (0, 1):
        env = jaxmarl.make(
            "overcooked_v2",
            layout=layout,
            max_steps=200,
            agent_view_size=2,
            negative_rewards=True,
            sample_recipe_on_delivery=True,
            random_agent_positions=False,
        )
        step_fn = jax.jit(env.step_env)
        key = jax.random.PRNGKey(ingredient)
        _, state = env.reset(key)
        # The route is conditioned on the recipe only for this reachability check.
        # Pin the intended recipe instead of relying on a lucky random reset.
        recipe_encoding = 3 * (4 << (2 * ingredient))
        state = state.replace(recipe=jnp.array(recipe_encoding, dtype=jnp.int32))
        steps = []
        deliveries = 0
        total_reward = 0.0
        for t, action in enumerate(route(ingredient), start=1):
            key, step_key = jax.random.split(key)
            _, state, rewards, _, _ = step_fn(
                step_key,
                state,
                {"agent_0": STAY, "agent_1": STAY, "agent_2": action},
            )
            reward = float(rewards["agent_0"])
            total_reward += reward
            deliveries += int(state.new_correct_delivery)
            if action == INTERACT or reward:
                steps.append(
                    {
                        "step": t,
                        "agent_2_position": [int(state.agents.pos.x[2]), int(state.agents.pos.y[2])],
                        "inventory": int(state.agents.inventory[2]),
                        "pot": [int(x) for x in state.grid[4, 4]],
                        "reward": reward,
                        "correct_delivery": bool(state.new_correct_delivery),
                    }
                )
        results.append(
            {
                "recipe": [ingredient] * 3,
                "route_steps": len(route(ingredient)),
                "correct_deliveries": deliveries,
                "team_reward": total_reward,
                "interactions": steps,
            }
        )
        print(
            f"recipe={ingredient} steps={len(route(ingredient))} "
            f"deliveries={deliveries} reward={total_reward}", flush=True
        )
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"access": access, "routes": results}, indent=2) + "\n", encoding="utf-8")
    if any(row["correct_deliveries"] < 1 for row in results):
        raise AssertionError(f"One or more recipes did not yield a correct soup; see {OUT}")


if __name__ == "__main__":
    main()
