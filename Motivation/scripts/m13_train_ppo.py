"""M13: continue the BC Alice with PPO and select checkpoints by task score."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch
import numpy as np

sys.path.insert(0, "src")
from ocres.ppo import ValueNetwork, clone_state_dict, collect_rollouts, evaluate_policy, ppo_update
from ocres.trainable import load_policy_checkpoint, save_policy_checkpoint, seed_everything, select_device


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m11_bc_l0.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m13_ppo_post.pt")
OUTPUT = pathlib.Path("data/m13_ppo_results.json")


def paired_bootstrap_ci(pre_rows, post_rows, seed, samples=20000):
    pre = {row["seed"]: row["deliveries"] for row in pre_rows}
    post = {row["seed"]: row["deliveries"] for row in post_rows}
    common = sorted(set(pre) & set(post))
    differences = np.asarray([post[key] - pre[key] for key in common], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(samples)])
    return {
        "method": "paired episode bootstrap",
        "samples": samples,
        "mean_difference": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "all_seeds_improved": bool(np.all(differences > 0)),
        "differences": [int(value) for value in differences],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(range(2001, 2017)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(3001, 3017)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260910)
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
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.learning_rate
    )

    pre_validation = evaluate_policy(actor, spec, device, args.validation_seeds, args.horizon)
    pre_test = evaluate_policy(actor, spec, device, args.test_seeds, args.horizon)
    best_mean = pre_validation["delivery_mean"]
    best_update = 0
    best_state = clone_state_dict(actor)
    curve = [{"update": 0, "validation": pre_validation}]
    print(
        f"pre validation: mean={best_mean:.3f} "
        f"range={pre_validation['delivery_min']}-{pre_validation['delivery_max']}"
    )

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
        )
        losses = ppo_update(
            actor,
            critic,
            optimizer,
            records,
            device,
            temperature=args.temperature,
        )
        item = {
            "update": update,
            "train_delivery_mean": sum(row["deliveries"] for row in episodes) / len(episodes),
            "train_macro_steps": len(records),
            **losses,
        }
        if update % args.eval_every == 0 or update == args.updates:
            validation = evaluate_policy(actor, spec, device, args.validation_seeds, args.horizon)
            item["validation"] = validation
            if validation["delivery_mean"] > best_mean:
                best_mean = validation["delivery_mean"]
                best_update = update
                best_state = clone_state_dict(actor)
            print(
                f"update={update:03d} train={item['train_delivery_mean']:.2f} "
                f"validation={validation['delivery_mean']:.2f} "
                f"range={validation['delivery_min']}-{validation['delivery_max']} "
                f"entropy={losses['entropy']:.3f} best={best_mean:.2f}@{best_update}"
            )
        curve.append(item)

    actor.load_state_dict(best_state)
    post_validation = evaluate_policy(actor, spec, device, args.validation_seeds, args.horizon)
    post_test = evaluate_policy(actor, spec, device, args.test_seeds, args.horizon)
    paired_test = paired_bootstrap_ci(pre_test["rows"], post_test["rows"], args.seed)
    result = {
        "milestone": "M13 PPO checkpoint training",
        "selection_rule": "highest mean deliveries on fixed validation seeds; held-out test and Bob metrics unused",
        "device": str(device),
        "seed": args.seed,
        "horizon": args.horizon,
        "train_seed_policy": "fresh deterministic start seeds sampled from [10001, 1000000)",
        "validation_seeds": args.validation_seeds,
        "test_seeds": args.test_seeds,
        "pre_checkpoint": str(args.pre_checkpoint),
        "pre_checkpoint_meta": {"source_levels": pre_meta.get("source_levels"), "test": pre_meta.get("test")},
        "pre_validation": pre_validation,
        "pre_test": pre_test,
        "best_update": best_update,
        "post_validation": post_validation,
        "post_test": post_test,
        "post_minus_pre_test_deliveries": post_test["delivery_mean"] - pre_test["delivery_mean"],
        "paired_test": paired_test,
        "curve": curve,
    }
    save_policy_checkpoint(args.post_checkpoint, actor, spec, extra=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("post-pre test deliveries:", result["post_minus_pre_test_deliveries"])
    print("checkpoint:", args.post_checkpoint)
    print("results:", args.output)


if __name__ == "__main__":
    main()
