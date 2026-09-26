"""Collect RWARE trajectories from independent-policy checkpoints for the ETM information gate.

For each checkpoint (a training stage) this rolls out episodes with the *sampled* policies and
stores, per step: each agent's local observation features (observer input), each agent's action
(prediction label for the other agents), plus position/direction/carrying (used only to group
"partner visible to observer" analysis), the shared reward, and episode-termination flags.

Nothing global is stored for the models: the observation features are exactly what an agent sees
at execution time (partial view + own agent-ID one-hot + action mask).

Usage:
  python collect_trajectories.py --run-dir results/<run> --episodes 32 --out data/<name>.npz
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

NUM_AGENTS = 4
ACTION_DIM = 5
TIME_LIMIT = 500


def collect_stage(params, env, keys, network):
    @jax.jit
    def rollout(params, key):
        state, timestep = env.reset(key)

        def _step(carry, t):
            state, timestep, live = carry
            features, action_mask = preprocess(timestep.observation, NUM_AGENTS)
            logits, _ = jax.vmap(lambda p, o: network.apply(p, o))(params, features)
            logits = masked_logits(logits, action_mask)
            act_keys = jax.random.split(jax.random.fold_in(key, t), NUM_AGENTS)
            action = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(
                act_keys, logits
            )
            new_state, new_timestep = env.step(state, action)
            record = {
                "features": features,
                "action": action.astype(jnp.int8),
                "mask": action_mask,
                "position": jnp.stack(
                    [state.agents.position.x, state.agents.position.y], axis=-1
                ),
                "direction": state.agents.direction.astype(jnp.int8),
                "carrying": state.agents.is_carrying.astype(jnp.int8),
                "reward": new_timestep.reward,
                "last": new_timestep.last(),
                "live": live,
            }
            live = live & ~new_timestep.last()
            return (new_state, new_timestep, live), record

        init = (state, timestep, jnp.array(True))
        (_, _, _), records = jax.lax.scan(_step, init, jnp.arange(TIME_LIMIT))
        return records

    return jax.vmap(rollout, in_axes=(None, 0))(params, keys)


def stage_keys(seed: int, stage: int, episodes: int):
    return jax.random.split(jax.random.PRNGKey(seed + 7919 * stage), episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint-stride", type=int, default=10, help="use every Nth checkpoint")
    argv = parser.parse_args()

    run_dir = Path(argv.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir
    out_path = Path(argv.out)
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(TIME_LIMIT)
    network = ActorCritic(action_dim=ACTION_DIM)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pkl"))[:: argv.checkpoint_stride]

    stages = []
    arrays: dict[str, list] = {}
    for stage, path in enumerate(checkpoints):
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        params = jax.tree.map(jnp.asarray, payload["params"])
        records = jax.device_get(collect_stage(params, env, stage_keys(argv.seed, stage, argv.episodes), network))
        for key, value in records.items():
            arrays.setdefault(key, []).append(np.asarray(value))
        stages.append(
            {
                "checkpoint": path.name,
                "cumulative_env_steps": payload["meta"].get("cumulative_env_steps"),
                "episodes": argv.episodes,
            }
        )
        delivered = float(np.asarray(records["reward"]).sum(axis=1).mean())
        print(f"[{stage + 1}/{len(checkpoints)}] {path.name} collected, mean deliveries/episode={delivered:.2f}", flush=True)

    np.savez_compressed(
        out_path,
        stages=json.dumps(stages),
        **{key: np.stack(values, axis=0) for key, values in arrays.items()},
    )
    print(f"[collect] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) with {len(stages)} stages")


if __name__ == "__main__":
    main()
