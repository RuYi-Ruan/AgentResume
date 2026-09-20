"""M16 gate 2b: continue Alice-pre with PPO on the ambiguous layout."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch

from ocres.generic_agents import LayoutAwareTrainableAgent, ScriptedMacroAgent
from ocres.grid import STAY, World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.ppo import ValueNetwork, MacroTrainingEnv, clone_state_dict, collect_rollouts, evaluate_policy, ppo_update
from ocres.trainable import load_policy_checkpoint, save_policy_checkpoint, seed_everything, select_device


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m16_v2_alice_post_results.json")


class M16TrainingEnv(MacroTrainingEnv):
    def reset(self, seed):
        self.world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=self.horizon)
        rng = random.Random(5000 + int(seed))
        first, second = rng.sample(sorted(self.world.grid.passable), 2)
        state = self.world.env.state
        state.players[0].update_pos_and_or(first, (1, 0))
        state.players[1].update_pos_and_or(second, (1, 0))
        state.timestep = 0
        self.controller = LayoutAwareTrainableAgent(
            self.world.grid,
            0,
            self.actor,
            self.spec,
            device=self.device,
            horizon=self.horizon,
            max_goal_ticks=self.max_goal_ticks,
            role_hint="cook",
        )
        self.partner = ScriptedMacroAgent(self.world.grid, 1, "serve", True, self.horizon)
        self.deliveries = 0
        return self.controller.observe(self.world.env.state, self.deliveries)


def env_factory(actor, spec, device, horizon):
    return M16TrainingEnv(actor, spec, device, horizon=horizon, max_goal_ticks=50)


def paired_bootstrap(pre_rows, post_rows, seed, samples=20000):
    pre = {row["seed"]: row["deliveries"] for row in pre_rows}
    post = {row["seed"]: row["deliveries"] for row in post_rows}
    seeds = sorted(set(pre) & set(post))
    differences = np.asarray([post[value] - pre[value] for value in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(samples)])
    return {
        "mean_difference": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "seeds_improved": int(np.sum(differences > 0)),
        "seed_count": len(seeds),
        "differences": [int(value) for value in differences],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(9501, 9517)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(9601, 9617)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=5.0)
    parser.add_argument("--entropy-coef", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    device = select_device(args.device)
    actor, spec, pre_meta = load_policy_checkpoint(args.pre_checkpoint, map_location=device)
    actor = actor.to(device)
    critic = ValueNetwork(spec.dim, actor.hidden_dim).to(device)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate)

    pre_validation = evaluate_policy(
        actor, spec, device, args.validation_seeds, args.horizon, env_factory=env_factory
    )
    pre_test = evaluate_policy(actor, spec, device, args.test_seeds, args.horizon, env_factory=env_factory)
    best_mean = pre_validation["delivery_mean"]
    best_update = 0
    best_state = clone_state_dict(actor)
    curve = [{"update": 0, "validation": pre_validation}]
    print(f"pre validation={best_mean:.3f} range={pre_validation['delivery_min']}-{pre_validation['delivery_max']}", flush=True)

    for update in range(1, args.updates + 1):
        train_seeds = [rng.randrange(10001, 1000000) for _ in range(args.episodes_per_update)]
        records, episodes = collect_rollouts(
            actor,
            critic,
            spec,
            device,
            train_seeds,
            args.horizon,
            args.temperature,
            gamma=0.995,
            gae_lambda=0.95,
            env_factory=env_factory,
        )
        losses = ppo_update(
            actor,
            critic,
            optimizer,
            records,
            device,
            temperature=args.temperature,
            entropy_coef=args.entropy_coef,
        )
        item = {
            "update": update,
            "train_delivery_mean": float(np.mean([row["deliveries"] for row in episodes])),
            "train_macro_steps": len(records),
            **losses,
        }
        if update % args.eval_every == 0 or update == args.updates:
            validation = evaluate_policy(
                actor, spec, device, args.validation_seeds, args.horizon, env_factory=env_factory
            )
            item["validation"] = validation
            if validation["delivery_mean"] > best_mean:
                best_mean = validation["delivery_mean"]
                best_update = update
                best_state = clone_state_dict(actor)
            print(
                f"update={update:03d} train={item['train_delivery_mean']:.2f} "
                f"validation={validation['delivery_mean']:.2f} "
                f"range={validation['delivery_min']}-{validation['delivery_max']} "
                f"best={best_mean:.2f}@{best_update}",
                flush=True,
            )
        curve.append(item)

    actor.load_state_dict(best_state)
    post_validation = evaluate_policy(
        actor, spec, device, args.validation_seeds, args.horizon, env_factory=env_factory
    )
    post_test = evaluate_policy(actor, spec, device, args.test_seeds, args.horizon, env_factory=env_factory)
    paired = paired_bootstrap(pre_test["rows"], post_test["rows"], args.seed)
    result = {
        "milestone": "M16-V2 Alice capability growth on ambiguous_kitchen_v2",
        "selection_rule": "highest mean deliveries on fixed validation seeds; test and Bob metrics unused",
        "device": str(device),
        "seed": args.seed,
        "horizon": args.horizon,
        "temperature": args.temperature,
        "entropy_coef": args.entropy_coef,
        "validation_seeds": args.validation_seeds,
        "test_seeds": args.test_seeds,
        "pre_checkpoint": str(args.pre_checkpoint),
        "pre_checkpoint_meta": pre_meta,
        "pre_validation": pre_validation,
        "pre_test": pre_test,
        "best_update": best_update,
        "post_validation": post_validation,
        "post_test": post_test,
        "paired_test": paired,
        "curve": curve,
    }
    save_policy_checkpoint(args.post_checkpoint, actor, spec, extra=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pre_test": pre_test, "post_test": post_test, "paired": paired}, ensure_ascii=False, indent=2))
    print("checkpoint:", args.post_checkpoint)


if __name__ == "__main__":
    main()
