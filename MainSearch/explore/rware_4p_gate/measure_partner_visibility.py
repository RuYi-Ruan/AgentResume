"""How observable are partners in RWARE as a function of sensor range?

The ETM premise needs partner behaviour to be observable. With the default `tiny-4ag`
(sensor_range=1) a partner sits inside the observer's 3x3 window on only ~9% of steps, which is
the main reason the closed-loop gain came out near noise. This measures the same statistic for a
few sensor ranges using random-action rollouts, so we can pick a configuration where teammate
models have something to see - before spending compute on training.

Usage:
  python measure_partner_visibility.py --sensor-ranges 1 2 3 --episodes 16 --steps 500
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
sys.path.insert(0, str(HERE))

from train_rware_ippo import TINY_4AG, make_env, preprocess  # noqa: E402


def measure(sensor_range: int, episodes: int, steps: int) -> dict:
    env = make_env(500, sensor_range)
    num_agents = TINY_4AG["num_agents"]
    obs_dim = int(env.observation_spec.agents_view.shape[-1])

    # per-pair visibility fraction over live steps
    def per_pair(key):
        state, timestep = env.reset(key)

        def _step(carry, t):
            state, timestep, live = carry
            position = jnp.stack(
                [state.agents.position.x, state.agents.position.y], axis=-1
            )
            dx = jnp.abs(position[:, 0][:, None] - position[:, 0][None, :])
            dy = jnp.abs(position[:, 1][:, None] - position[:, 1][None, :])
            visible = (dx <= sensor_range) & (dy <= sensor_range)
            visible = visible & (1 - jnp.eye(num_agents, dtype=bool))
            action = jax.random.randint(jax.random.fold_in(key, t), (num_agents,), 0, 5)
            new_state, new_timestep = env.step(state, action)
            live_now = live
            live = live & ~timestep.last()
            return (new_state, new_timestep, live), (
                visible.astype(jnp.float32) * live_now,
                live_now.astype(jnp.float32),
            )

        init = (state, timestep, jnp.array(True))
        (_, _, _), (visible, live) = jax.lax.scan(_step, init, jnp.arange(steps))
        pairs = visible.sum(axis=0)
        live_steps = live.sum()
        return pairs / jnp.maximum(live_steps, 1.0)

    keys = jax.random.split(jax.random.PRNGKey(0), episodes)
    per_pair_out = jax.jit(jax.vmap(per_pair))(keys)
    fractions = np.asarray(per_pair_out).mean(axis=0)
    masked = fractions[~np.eye(num_agents, dtype=bool)]
    return {
        "sensor_range": sensor_range,
        "view_window": 2 * sensor_range + 1,
        "obs_features": obs_dim,
        "episodes": episodes,
        "steps_per_episode": steps,
        "mean_pair_visibility_fraction": float(masked.mean()),
        "min_pair_visibility_fraction": float(masked.min()),
        "max_pair_visibility_fraction": float(masked.max()),
        "pair_matrix": fractions.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-ranges", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    argv = parser.parse_args()

    results = [measure(r, argv.episodes, argv.steps) for r in argv.sensor_ranges]
    for record in results:
        print(
            f"sensor_range={record['sensor_range']} (view {record['view_window']}x{record['view_window']}, "
            f"obs {record['obs_features']}): partner visible on "
            f"{100 * record['mean_pair_visibility_fraction']:.1f}% of live steps "
            f"(min {100 * record['min_pair_visibility_fraction']:.1f}%, "
            f"max {100 * record['max_pair_visibility_fraction']:.1f}%)",
            flush=True,
        )
    out_path = HERE / "results" / "partner_visibility.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[visibility] wrote {out_path}")


if __name__ == "__main__":
    main()
