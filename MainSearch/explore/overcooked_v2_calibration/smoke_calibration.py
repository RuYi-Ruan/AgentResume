"""Shortest validation of the calibration harness (not a learning experiment).

Checks, in order:
  1. the zero-update init path returns a runner state with update_step = 0;
  2. two jitted resume segments advance update_step and env_step by the expected amounts;
  3. frozen evaluation is deterministic for fixed parameters;
  4. a checkpoint round-trip (save -> load -> plant) reproduces the same evaluation output;
  5. the environment auto-resets at max_steps (the 400-step training episode boundary is real).

Results are written to results/smoke.json. Nothing here says anything about learnability.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from run_calibration import (  # noqa: E402
    build_eval_fn,
    eval_keys_for,
    init_runner_state,
    make_train_fn,
    plant_checkpoint,
    parse_args,
    save_checkpoint,
)


def params_equal(left, right) -> bool:
    left_leaves = jax.tree_util.tree_flatten(left)[0]
    right_leaves = jax.tree_util.tree_flatten(right)[0]
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves)
    )


def auto_reset_probe(layout: str, max_steps: int = 401) -> dict:
    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        agent_view_size=2,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        random_agent_positions=True,
    )

    def _step(state, t):
        action = {a: jnp.array(0) for a in env.agents}
        obs, state, reward, done, info = env.step(jax.random.fold_in(jax.random.PRNGKey(7), t), state, action)
        return state, (state.step, done["__all__"])

    _, state = env.reset(jax.random.PRNGKey(7))
    _, (steps, dones) = jax.lax.scan(_step, state, jnp.arange(max_steps))
    steps, dones = np.asarray(steps), np.asarray(dones)
    return {
        "steps_run": max_steps,
        "env_max_steps": env.max_steps,
        "step_counter_at_boundary": int(steps[env.max_steps - 1]),
        "step_counter_after_boundary": int(steps[env.max_steps]),
        "global_done_at_boundary": bool(dones[env.max_steps - 1]),
        "global_done_after_boundary": bool(dones[env.max_steps]),
    }


def main() -> None:
    args = parse_args(
        [
            "--num-envs", "2",
            "--num-steps", "8",
            "--num-minibatches", "1",
            "--update-epochs", "1",
            "--gru-dim", "8",
            "--fc-dim", "8",
            "--updates", "4",
            "--segment-updates", "2",
            "--eval-episodes", "2",
            "--eval-seed", "5",
        ]
    )
    checks: dict = {}
    started = time.perf_counter()

    runner_state, _ = init_runner_state(args, jax.random.PRNGKey(args.seed))
    checks["init_update_step"] = int(runner_state[4])
    checks["init_params_are_arrays"] = bool(
        hasattr(runner_state[0], "params") and isinstance(runner_state[0].params, dict)
    )

    train, config = make_train_fn(args, run_updates=args.segment_updates)
    train_jit = jax.jit(train)
    steps_per_update = args.num_envs * args.num_steps
    eval_keys = eval_keys_for(args)
    evaluate, fingerprint = build_eval_fn(config, eval_keys)
    checks["eval_fingerprint"] = fingerprint(eval_keys)

    with wandb.init(mode="disabled"):
        segment_records = []
        for segment in range(2):
            result = train_jit(jax.random.PRNGKey(args.seed), runner_state)
            jax.block_until_ready(result["metrics"]["env_step"])
            runner_state = result["runner_state"]
            segment_records.append(
                {
                    "segment": segment + 1,
                    "update_step": int(runner_state[4]),
                    "env_step": int(np.asarray(result["metrics"]["env_step"])[-1]),
                    "expected_env_step": (segment + 1) * args.segment_updates * steps_per_update,
                    "original_reward_mean": float(
                        np.asarray(result["metrics"]["original_reward"]).mean()
                    ),
                }
            )
        checks["segments"] = segment_records

        params = runner_state[0].params
        first_eval = np.asarray(jax.device_get(evaluate(params, eval_keys)))
        second_eval = np.asarray(jax.device_get(evaluate(params, eval_keys)))
        checks["eval_deterministic"] = bool(np.array_equal(first_eval, second_eval))

        checkpoint_path = HERE / "results" / "smoke_checkpoint.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            checkpoint_path,
            runner_state,
            {"env_steps": segment_records[-1]["env_step"], "segment": 2, "run_name": "smoke"},
        )
        with checkpoint_path.open("rb") as handle:
            checkpoint = pickle.load(handle)
        fresh_state, _ = init_runner_state(args, jax.random.PRNGKey(args.seed))
        restored_params = plant_checkpoint(fresh_state, checkpoint)[0].params
        checks["checkpoint_step"] = int(checkpoint["step"])
        checks["checkpoint_update_step"] = int(checkpoint["update_step"])
        checks["params_identical_after_roundtrip"] = params_equal(params, restored_params)
        restored_eval = np.asarray(jax.device_get(evaluate(restored_params, eval_keys)))
        checks["eval_identical_after_roundtrip"] = bool(np.array_equal(first_eval, restored_eval))
        checks["eval_output"] = first_eval.tolist()

    checks["auto_reset"] = auto_reset_probe(args.layout)
    checks["wall_seconds"] = round(time.perf_counter() - started, 2)
    checks["smoke_config"] = {
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "segment_updates": args.segment_updates,
        "updates": args.updates,
        "layout": args.layout,
    }

    out_path = HERE / "results" / "smoke.json"
    out_path.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(f"[smoke] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
