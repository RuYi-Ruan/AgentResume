"""Three-agent OvercookedV2 API and wall-clock smoke test; no learning claims."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxmarl
from jaxmarl.environments.overcooked_v2.layouts import Layout, grounded_coord_ring


HERE = Path(__file__).resolve().parent
EPISODE_STEPS = 200
SEEDS = (0, 1, 2)


def main() -> None:
    layout_text = grounded_coord_ring.replace("W       W", "W   A   W", 1)
    if layout_text == grounded_coord_ring:
        raise RuntimeError("Expected corridor row was absent; layout not modified")
    layout = Layout.from_string(
        layout_text, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    )
    if len(layout.agent_positions) != 3:
        raise RuntimeError(f"Expected 3 agents, found {len(layout.agent_positions)}")

    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=EPISODE_STEPS,
        agent_view_size=2,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        random_agent_positions=True,
    )
    if env.num_agents != 3:
        raise RuntimeError(f"Expected 3 env agents, found {env.num_agents}")
    agents = tuple(env.agents)
    action_sizes = {agent: env.action_space(agent).n for agent in agents}

    # JIT one transition, as the eventual JAX training loop would do.
    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)
    rows = []
    started = time.perf_counter()
    for seed in SEEDS:
        key = jax.random.PRNGKey(seed)
        key, reset_key = jax.random.split(key)
        episode_start = time.perf_counter()
        obs, state = reset_fn(reset_key)
        jax.block_until_ready(obs[agents[0]])
        if set(obs) != set(agents):
            raise RuntimeError(f"Seed {seed}: unexpected observation keys {list(obs)}")
        obs_shapes = {agent: list(obs[agent].shape) for agent in agents}
        reward_sum = 0.0
        correct_deliveries = 0
        terminal = False
        for t in range(EPISODE_STEPS):
            key, action_key, step_key = jax.random.split(key, 3)
            agent_keys = jax.random.split(action_key, len(agents))
            actions = {
                agent: jax.random.randint(agent_keys[i], (), 0, action_sizes[agent])
                for i, agent in enumerate(agents)
            }
            obs, state, rewards, dones, _ = step_fn(step_key, state, actions)
            jax.block_until_ready(state.step)
            if set(rewards) != set(agents) or set(dones) != set(agents) | {"__all__"}:
                raise RuntimeError(f"Seed {seed}: reward/done keys do not match agents")
            reward_sum += float(rewards[agents[0]])
            correct_deliveries += int(state.new_correct_delivery)
            terminal = bool(dones["__all__"])
            if terminal:
                break
        rows.append(
            {
                "seed": seed,
                "steps": t + 1,
                "terminal": terminal,
                "team_reward": reward_sum,
                "correct_deliveries": correct_deliveries,
                "observation_shapes": obs_shapes,
                "wall_seconds": round(time.perf_counter() - episode_start, 2),
            }
        )
        print(f"seed={seed} steps={t + 1} terminal={terminal} correct_deliveries={correct_deliveries} reward={reward_sum:.2f} elapsed={rows[-1]['wall_seconds']:.2f}s", flush=True)

    if not all(row["steps"] == EPISODE_STEPS and row["terminal"] for row in rows):
        raise RuntimeError("Not all episodes ended at the configured horizon")
    result = {
        "kind": "three_player_overcooked_v2_api_smoke",
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "layout": "grounded_coord_ring_plus_third_agent_at_row_1_col_4",
        "official_benchmark_layout": False,
        "num_agents": env.num_agents,
        "action_sizes": action_sizes,
        "episodes": rows,
        "total_wall_seconds": round(time.perf_counter() - started, 2),
    }
    output = HERE / "results" / "smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
