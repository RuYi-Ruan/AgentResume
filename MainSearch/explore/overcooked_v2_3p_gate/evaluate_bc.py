"""Closed-loop evaluation of a behaviour-cloned (or PPO) checkpoint on the Three-Arm Kitchen.

The BC gate from the external guidance is closed-loop, not offline accuracy:
    P(episode produces any shaped reward) >= 75%   and   P(>=1 correct delivery) >= 25%
must hold before PPO fine-tuning is worth attempting. If offline accuracy is high but these are
low, the fix is DAgger; if offline accuracy itself is low, the fix is the network.

Episodes run as one jitted, vmapped rollout (a per-step Python loop takes ~40 minutes for 64
episodes, which would make DAgger rounds impossible).

Usage:
  python evaluate_bc.py --checkpoint bc/three_arm_bc_seed0.pkl --episodes 64
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
sys.path.insert(0, str(HERE.parent / "overcooked_v2_ff_gate"))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import MLPActorCritic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    layout_path = Path(args.layout_file)
    if not layout_path.is_absolute():
        layout_path = HERE / layout_path
    layout = Layout.from_string(layout_path.read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = HERE / ckpt_path
    with ckpt_path.open("rb") as handle:
        payload = pickle.load(handle)
    params = {"params": jax.tree.map(jnp.asarray, payload["params"])}
    network = MLPActorCritic(action_dim=6, hidden=args.hidden)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)

    def make_evaluator(sample: bool):
        @jax.jit
        def evaluate(params, keys):
            def _episode(key):
                obs, env_state = env.reset(key)

                def _step(carry, t):
                    (obs, env_state, live, delivered, shaped_events, shaped_total, length,
                     action_counts) = carry
                    flat = jnp.stack([obs[a] for a in env.agents]).reshape(len(env.agents), -1)
                    logits, _ = network.apply(params, flat.astype(jnp.float32))
                    if sample:
                        actions = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(
                            jax.random.split(jax.random.fold_in(key, t), len(env.agents)), logits)
                    else:
                        actions = jnp.argmax(logits, axis=-1)
                    env_act = {a: actions[i] for i, a in enumerate(env.agents)}
                    actions = actions.astype(jnp.int32)
                    action_counts = action_counts.at[
                        jnp.clip(actions, 0, 5)].add(jnp.where(live, 1, 0))
                    obs, env_state, reward, done, info = env.step(key, env_state, env_act)
                    step_shaped = info["shaped_reward"][env.agents[0]]
                    delivered = delivered + jnp.where(
                        live & env_state.new_correct_delivery, 1.0, 0.0)
                    shaped_events = shaped_events + jnp.where(live & (step_shaped > 0), 1.0, 0.0)
                    shaped_total = shaped_total + jnp.where(live, step_shaped, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~done["__all__"]
                    return (obs, env_state, live, delivered, shaped_events, shaped_total, length,
                            action_counts), None

                init = (obs, env_state, jnp.array(True), 0.0, 0.0, 0.0, jnp.int32(0),
                        jnp.zeros(6))
                (_, _, _, delivered, shaped_events, shaped_total, length, action_counts), _ = (
                    jax.lax.scan(_step, init, jnp.arange(args.max_steps)))
                return jnp.concatenate([jnp.stack([delivered, shaped_events, shaped_total,
                                                   length.astype(jnp.float32)]),
                                        action_counts / jnp.maximum(action_counts.sum(), 1)])

            return jax.vmap(_episode)(keys)

        return evaluate

    results = {}
    for mode in ("greedy", "sampled"):
        raw = np.asarray(jax.device_get(make_evaluator(mode == "sampled")(params, keys)))
        deliveries, shaped_events, shaped_total, lengths = raw[:, :4].T
        action_mix = raw[:, 4:].mean(axis=0)
        results[mode] = {
            "episodes": args.episodes,
            "fraction_episodes_with_shaping": float(np.mean(shaped_events > 0)),
            "fraction_episodes_with_delivery": float(np.mean(deliveries > 0)),
            "mean_deliveries_per_episode": float(deliveries.mean()),
            "max_deliveries_in_an_episode": float(deliveries.max()),
            "mean_shaped_events_per_episode": float(shaped_events.mean()),
            "mean_shaped_reward_per_episode": float(shaped_total.mean()),
            "mean_episode_length": float(lengths.mean()),
            "action_mix": {name: float(v) for name, v in
                           zip(["right", "down", "left", "up", "stay", "interact"], action_mix)},
        }
        print(f"[bc-eval] {mode}: {json.dumps(results[mode])}")

    # PPO rolls out with sampled actions, so the sampling mode is the gate; greedy is reported
    # separately because behaviour cloning often collapses the deterministic mean policy.
    passed = (results["sampled"]["fraction_episodes_with_shaping"] >= 0.75
              and results["sampled"]["fraction_episodes_with_delivery"] >= 0.25)
    print(f"[bc-eval] closed-loop gate on sampled rollouts (75% shaping / 25% delivery): "
          f"{'PASS' if passed else 'FAIL'}")
    report = {"checkpoint": str(ckpt_path), **results, "closed_loop_gate_passed": bool(passed)}
    out = Path(args.out) if args.out else ckpt_path.with_suffix(".eval.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[bc-eval] wrote {out}")


if __name__ == "__main__":
    main()
