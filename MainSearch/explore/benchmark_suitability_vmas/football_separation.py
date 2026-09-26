"""Structural gate for VMAS Football: programmed blue team versus idle blue team."""

import argparse
import json
from pathlib import Path

import torch
import vmas


def run(seed, worlds, horizon, blue_ai):
    env = vmas.make_env(
        scenario="football",
        num_envs=worlds,
        device="cpu",
        max_steps=horizon,
        seed=seed,
        n_blue_agents=3,
        n_red_agents=3,
        ai_blue_agents=blue_ai,
        ai_red_agents=True,
        observe_teammates=True,
    )
    env.reset(seed=seed)
    blue_scores = torch.zeros(worlds, dtype=torch.bool)
    red_scores = torch.zeros(worlds, dtype=torch.bool)
    active = torch.ones(worlds, dtype=torch.bool)
    for _ in range(horizon):
        actions = [] if blue_ai else [torch.zeros(worlds, 2) for _ in env.agents]
        _, _, done, _ = env.step(actions)
        sparse = env.scenario._sparse_reward_blue
        blue_scores |= active & (sparse > 0)
        red_scores |= active & (sparse < 0)
        active &= ~done
    distance = torch.linalg.norm(
        env.scenario.ball.state.pos - env.scenario.right_goal_pos, dim=-1
    )
    return {
        "blue_goals": int(blue_scores.sum()),
        "red_goals": int(red_scores.sum()),
        "mean_final_ball_to_blue_goal": float(distance.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "football_separation.json")
    args = parser.parse_args()
    rows = []
    for seed in range(args.seeds):
        row = {
            "seed": seed,
            "worlds": args.worlds,
            "blue_ai": run(seed, args.worlds, args.horizon, True),
            "blue_idle": run(seed, args.worlds, args.horizon, False),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {"type": "structural_diagnostic_not_learning", "horizon": args.horizon, "rows": rows}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
