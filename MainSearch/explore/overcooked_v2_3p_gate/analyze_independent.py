"""Analyse a trained independent-actor checkpoint: did the three identities diverge, and into what?

Reports per agent: action mix, shaped-reward share, and the pairwise TV distance between the three
actors' action distributions on the states actually visited (fixed identities should separate into
complementary roles, e.g. fetchers vs a server).

Usage:
  python analyze_independent.py --checkpoint results/.../checkpoint_004915200.pkl --episodes 32
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
from train_overcooked_ff import MLPActorCriticPriv, compact_state, critic_spec  # noqa: E402

ACTION_NAMES = ["right", "down", "left", "up", "stay", "interact"]


class Ctx:
    privileged_critic = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    layout = Layout.from_string((HERE / args.layout_file).read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    crit = critic_spec(Ctx(), env)
    crit["context"] = crit
    num_agents = len(env.agents)

    path = Path(args.checkpoint)
    if not path.is_absolute():
        path = HERE / path
    params = pickle.load(path.open("rb"))["params"]
    params = [{"params": jax.tree.map(jnp.asarray, entry["params"])} for entry in params]
    critic_dim = int(np.asarray(params[0]["params"]["Dense_3"]["kernel"]).shape[0])
    network = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)

    @jax.jit
    def run_episode(key):
        obs, state = env.reset(key)

        def _step(carry, t):
            obs, state, live, soups, actions, shaped, tv_acc = carry
            flat = jnp.stack([obs[a] for a in env.agents], axis=0).reshape(num_agents, -1)
            cin = compact_state(jax.tree.map(lambda x: x[None], state),
                                crit["context"], t).reshape(num_agents, -1)
            logits = jnp.stack([network.apply(params[i], flat[i][None], cin[i][None])[0][0]
                                for i in range(num_agents)])
            probs = jax.nn.softmax(logits, axis=-1)
            acts = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(
                jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
            # pairwise TV between the three actors on this state
            pairwise = jnp.stack([
                0.5 * jnp.sum(jnp.abs(probs[0] - probs[1])),
                0.5 * jnp.sum(jnp.abs(probs[0] - probs[2])),
                0.5 * jnp.sum(jnp.abs(probs[1] - probs[2])),
            ])
            obs, state, reward, done, info = env.step(
                key, state, {a: acts[i] for i, a in enumerate(env.agents)})
            soups = soups + jnp.where(live & state.new_correct_delivery, 1.0, 0.0)
            onehot = jax.nn.one_hot(acts, 6)
            actions = actions + onehot * jnp.where(live, 1.0, 0.0)
            shaped = shaped + jnp.stack([info["shaped_reward"][a] for a in env.agents]) * jnp.where(
                live, 1.0, 0.0)
            tv_acc = tv_acc + pairwise * jnp.where(live, 1.0, 0.0)
            live = live & ~done["__all__"]
            return (obs, state, live, soups, actions, shaped, tv_acc), None

        init = (obs, state, jnp.array(True), 0.0, jnp.zeros((num_agents, 6)), jnp.zeros(num_agents),
                jnp.zeros(3))
        soups, action_totals, shaped_totals, tv_totals = jax.lax.scan(
            _step, init, jnp.arange(args.max_steps))[0][3:]
        return jnp.concatenate([soups[None], action_totals.reshape(-1),
                                shaped_totals.reshape(-1), tv_totals.reshape(-1)])

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    raw = np.asarray(jax.vmap(run_episode)(keys))
    soups = raw[:, 0]
    action_counts = raw[:, 1:1 + num_agents * 6].reshape(args.episodes, num_agents, 6).sum(axis=0)
    shaped_sum = raw[:, 1 + num_agents * 6:1 + num_agents * 6 + num_agents]
    tv = raw[:, -3:] / float(args.max_steps)   # TV accumulated over steps -> per-step mean

    report = {
        "checkpoint": str(path),
        "episodes": args.episodes,
        "team_soups_mean": float(soups.mean()),
        "team_soups_max": float(soups.max()),
        "per_agent_action_mix": {},
        "per_agent_shaped_reward_mean": [float(v) for v in shaped_sum.mean(axis=0)],
        "pairwise_actor_tv_on_visited_states": {
            "actor0_vs_actor1": float(tv[:, 0].mean()),
            "actor0_vs_actor2": float(tv[:, 1].mean()),
            "actor1_vs_actor2": float(tv[:, 2].mean()),
        },
    }
    for i in range(num_agents):
        total = max(1.0, action_counts[i].sum())
        report["per_agent_action_mix"][f"agent{i}"] = {
            name: float(count / total) for name, count in zip(ACTION_NAMES, action_counts[i])}
    print(json.dumps(report, indent=2))
    out = Path(args.out) if args.out else path.parent / "divergence.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[analyse] wrote {out}")


if __name__ == "__main__":
    main()
