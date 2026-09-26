"""No-learning execution/sensitivity check for five-agent SMAClite 2s3z."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import smaclite  # noqa: F401


def play(seed, mode, horizon):
    env = gym.make("smaclite/2s3z-v0")
    rng = np.random.default_rng(seed + 100_000)
    _, _ = env.reset(seed=seed)
    total = 0.0
    steps = 0
    info = {}
    try:
        for _ in range(horizon):
            available = env.unwrapped.get_avail_actions()
            if mode == "random":
                actions = [int(rng.choice(np.flatnonzero(mask))) for mask in available]
            else:
                actions = [int(np.flatnonzero(mask)[0]) for mask in available]
            _, reward, done, truncated, info = env.step(actions)
            total += float(reward)
            steps += 1
            if done or truncated:
                break
        return {
            "reward": total,
            "steps": steps,
            "done": bool(done),
            "truncated": bool(truncated),
            "info": {key: value for key, value in info.items() if isinstance(value, (bool, int, float, str))},
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "smoke_results.json")
    args = parser.parse_args()
    rows = []
    for seed in range(args.episodes):
        row = {"seed": seed, "random": play(seed, "random", args.horizon), "stop": play(seed, "stop", args.horizon)}
        rows.append(row)
        print(f"seed {seed}: random {row['random']['reward']:.2f}, stop {row['stop']['reward']:.2f}", flush=True)
    result = {"status": "structural_no_learning", "map": "2s3z", "agents": 5, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "episodes_per_condition": args.episodes,
        "random_mean_reward": float(np.mean([row["random"]["reward"] for row in rows])),
        "stop_mean_reward": float(np.mean([row["stop"]["reward"] for row in rows])),
    }, indent=2))


if __name__ == "__main__":
    main()
