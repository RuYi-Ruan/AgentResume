"""Frozen-episode evaluation of the independent-policy RWARE checkpoints.

`train_rware_iippo.py` dumps per-agent action histograms but not the deliveries metric, so this
script loads its stacked per-agent parameters and runs the same frozen protocol as the
shared-parameter gate (32 fixed initial states, greedy and sampled actions, 500-step cap).

Usage:
  python eval_checkpoints_iippo.py --run-dir results/rware_tiny4ag_iippo_indep_seed0_8M
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from train_rware_ippo import ActorCritic, make_env, masked_logits, preprocess  # noqa: E402

TINY_4AG_NUM_AGENTS = 4
ACTION_DIM = 5
FEATURES = 66 + TINY_4AG_NUM_AGENTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--time-limit", type=int, default=500)
    argv = parser.parse_args()

    run_dir = Path(argv.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir

    env = make_env(argv.time_limit)
    network = ActorCritic(action_dim=ACTION_DIM)
    keys = jax.random.split(jax.random.PRNGKey(argv.eval_seed), argv.episodes)
    horizon = int(argv.time_limit)

    def _make(sample: bool):
        @jax.jit
        def evaluate(params, keys):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, delivered, length = carry
                    features, action_mask = preprocess(timestep.observation, TINY_4AG_NUM_AGENTS)
                    logits, _ = jax.vmap(lambda p, o: network.apply(p, o))(params, features)
                    logits = masked_logits(logits, action_mask)
                    if sample:
                        action = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(jax.random.split(jax.random.fold_in(key, t), TINY_4AG_NUM_AGENTS), logits)
                    else:
                        action = jnp.argmax(logits, axis=-1)
                    state, timestep = env.step(state, action)
                    delivered = delivered + jnp.where(live, timestep.reward, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~timestep.last()
                    return (state, timestep, live, delivered, length), None

                init = (state, timestep, jnp.array(True), 0.0, jnp.int32(0))
                (_, _, _, delivered, length), _ = jax.lax.scan(
                    _step, init, jnp.arange(horizon)
                )
                return delivered, length

            return jax.vmap(_episode)(keys)

        return evaluate

    greedy, sampled = _make(sample=False), _make(sample=True)

    points = []
    for path in sorted(run_dir.glob("checkpoint_*.pkl")):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        params = jax.tree.map(jnp.asarray, payload["params"])
        g_del, g_len = jax.device_get(greedy(params, keys))
        s_del, s_len = jax.device_get(sampled(params, keys))
        point = {
            "checkpoint": path.name,
            "cumulative_env_steps": payload["meta"].get("cumulative_env_steps"),
            "greedy_delivered_mean": float(np.asarray(g_del).mean()),
            "greedy_delivered_max": float(np.asarray(g_del).max()),
            "greedy_episode_length_mean": float(np.asarray(g_len).mean()),
            "sampled_delivered_mean": float(np.asarray(s_del).mean()),
            "sampled_delivered_max": float(np.asarray(s_del).max()),
            "sampled_episode_length_mean": float(np.asarray(s_len).mean()),
        }
        points.append(point)
        print(
            f"{point['cumulative_env_steps']:>10,} steps | greedy {point['greedy_delivered_mean']:.2f} "
            f"| sampled {point['sampled_delivered_mean']:.2f} (max {point['sampled_delivered_max']:.0f}) "
            f"| len {point['sampled_episode_length_mean']:.0f}",
            flush=True,
        )

    out_path = run_dir / "checkpoint_evals.json"
    out_path.write_text(json.dumps({"points": points}, indent=2) + "\n", encoding="utf-8")
    print(f"[eval_checkpoints_iippo] wrote {out_path}")


if __name__ == "__main__":
    main()
