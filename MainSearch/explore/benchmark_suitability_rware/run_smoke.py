"""Ten paired-seed random/idle 4-agent RWARE games; no learning claims."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deps"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import rware  # noqa: F401,E402  # Registers environments with Gymnasium.


def main():
    env = gym.make("rware-tiny-4ag-v2")
    if env.unwrapped.n_agents != 4:
        raise RuntimeError(f"Expected 4 robots; got {env.unwrapped.n_agents}")
    action_sizes = [int(space.n) for space in env.action_space]
    rows = []
    started = time.perf_counter()
    try:
        for seed in range(10):
            for policy in ("idle", "random"):
                observations, _ = env.reset(seed=seed)
                if len(observations) != 4:
                    raise RuntimeError("Reset did not return four observations")
                obs_shapes = [[int(dim) for dim in np.asarray(obs).shape] for obs in observations]
                rng = np.random.default_rng(100000 + seed)
                deliveries = 0
                elapsed_start = time.perf_counter()
                for step in range(1, 501):
                    if policy == "idle":
                        actions = [0] * 4
                    else:
                        actions = [int(rng.integers(0, size)) for size in action_sizes]
                    observations, rewards, terminated, truncated, _ = env.step(actions)
                    deliveries += int(sum(rewards))
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "seed": seed,
                        "policy": policy,
                        "steps": step,
                        "deliveries": deliveries,
                        "observation_shapes": obs_shapes,
                        "wall_seconds": round(time.perf_counter() - elapsed_start, 2),
                    }
                )
                print(
                    f"seed={seed} policy={policy} steps={step} deliveries={deliveries}",
                    flush=True,
                )
    finally:
        env.close()
    result = {
        "environment": "rware-tiny-4ag-v2",
        "package": "rware==2.0.0",
        "games_per_policy": 10,
        "horizon": 500,
        "num_agents": 4,
        "action_sizes": action_sizes,
        "episodes": rows,
        "total_wall_seconds": round(time.perf_counter() - started, 2),
        "mean_deliveries": {
            policy: float(np.mean([row["deliveries"] for row in rows if row["policy"] == policy]))
            for policy in ("idle", "random")
        },
    }
    result_path = HERE / "results" / "smoke.json"
    result_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
