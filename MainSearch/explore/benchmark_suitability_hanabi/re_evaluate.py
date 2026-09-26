"""Re-evaluate saved Hanabi parameters without retraining or auto-reset bias."""

import argparse
import json
from pathlib import Path

from flax.serialization import from_bytes
import jax
import jax.numpy as jnp
import numpy as np

import run_shared_gate as gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    n_games = metadata["eval_games"]
    env = gate.baseline.jaxmarl.make("hanabi", num_agents=3)
    net = gate.baseline.ActorCritic(env.action_space(env.agents[0]).n, config=config)
    _, init_key = jax.random.split(jax.random.PRNGKey(metadata["seed"]))
    init_x = (
        jnp.zeros((1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape)),
        jnp.zeros((1, config["NUM_ENVS"])),
        jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
    )
    initial_params = net.init(init_key, init_x)
    final_params = from_bytes(
        initial_params, (args.run_dir / "final_params.msgpack").read_bytes()
    )
    for label, params in (("initial", initial_params), ("final", final_params)):
        for mode, sample in (("greedy", False), ("sampled", True)):
            rows = gate.evaluate(params, config, n_games, sample)
            (args.run_dir / f"eval_{label}_{mode}_corrected.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
            scores = np.array([row["score"] for row in rows])
            print(
                f"{label} {mode}: {len(rows)} games, mean={scores.mean():.3f}, "
                f"sd={scores.std():.3f}, nonzero={np.count_nonzero(scores)}"
            )


if __name__ == "__main__":
    main()
