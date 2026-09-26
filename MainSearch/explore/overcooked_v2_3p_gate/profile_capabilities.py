"""Capability profile of a trained independent-actor checkpoint: per-identity [c_I, c_P, c_D].

Follows the external guidance: capability must be a *behavioural* measurement, not the training stage.
Per episode we attribute role events to the agent that performed them and normalise by episode length:

  c_I  ingredient capability  - successful pot placements per step
  c_P  pot capability         - dish pickups (a finished pot handled) per step
  c_D  service capability     - correct deliveries per step (the +20 bonus goes only to the
                                delivering agent, so this attribution is exact)

Rates are then min-max normalised across the measured checkpoints so every c lives in [0, 1] and the
resulting vector can be fed to the ego as z.

Usage:
  python profile_capabilities.py --checkpoints results/.../checkpoint_0000491520.pkl \
      results/.../checkpoint_0002457600.pkl results/.../checkpoint_0004915200.pkl
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
from jaxmarl.environments.overcooked_v2.common import DynamicObject  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import MLPActorCriticPriv, compact_state, critic_spec  # noqa: E402


class Ctx:
    privileged_critic = True


def load_agents(path: Path, network_dim: int | None = None):
    payload = pickle.load(path.open("rb"))
    agents = payload["params"]
    params = [{"params": jax.tree.map(jnp.asarray, a["params"])} for a in agents]
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="results/capability_profiles.json")
    args = parser.parse_args()

    layout = Layout.from_string((HERE / args.layout_file).read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    crit = critic_spec(Ctx(), env)
    crit["context"] = crit
    num_agents = len(env.agents)
    pot_cells = crit["pot_cells"]

    def make_rollout(params, critic_dim):
        network = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)

        @jax.jit
        def run(key):
            obs, state = env.reset(key)

            def _step(carry, t):
                obs, state, live, soups, placements, dishes, deliveries, steps = carry
                flat = jnp.stack([obs[a] for a in env.agents], axis=0).reshape(num_agents, -1)
                cin = compact_state(jax.tree.map(lambda x: x[None], state),
                                    crit["context"], t).reshape(num_agents, -1)
                logits = jnp.stack(
                    [network.apply(params[i], flat[i][None], cin[i][None])[0][0]
                     for i in range(num_agents)])
                acts = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(
                    jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
                inv_before = state.agents.inventory
                pot_before = jnp.stack([state.grid[c[1], c[0], 1] for c in pot_cells])
                obs, state, reward, done, info = env.step(
                    key, state, {a: acts[i] for i, a in enumerate(env.agents)})
                live_f = jnp.where(live, 1.0, 0.0)
                # ingredient placement: some pot's ingredient count increased and the agent's hands
                # went from holding an ingredient to empty
                pot_after = jnp.stack([state.grid[c[1], c[0], 1] for c in pot_cells])
                placed = jnp.any(pot_after != pot_before)
                handed = (inv_before != 0) & (state.agents.inventory == 0)
                placements = placements + jnp.where(placed & handed, 1.0, 0.0) * live_f
                # dish pickup: the agent now holds a cooked dish that it did not hold before
                dish_now = (state.agents.inventory & int(DynamicObject.COOKED)) != 0
                dish_was = (inv_before & int(DynamicObject.COOKED)) != 0
                dishes = dishes + jnp.where(dish_now & ~dish_was, 1.0, 0.0) * live_f
                # correct delivery: only the delivering agent receives the +20 bonus
                # a correct delivery is attributed to the agent that was holding the dish and no
                # longer is (the env's +20 is reduced by the per-step negative-reward penalty, so
                # thresholding the reward would miss deliveries)
                delivered_now = jnp.asarray(state.new_correct_delivery) & live
                deliveries = deliveries + jnp.where(
                    delivered_now & dish_was & ~dish_now, 1.0, 0.0)
                soups = soups + jnp.where(live & state.new_correct_delivery, 1.0, 0.0)
                steps = steps + live_f
                live = live & ~done["__all__"]
                return (obs, state, live, soups, placements, dishes, deliveries, steps), None

            init = (obs, state, jnp.array(True), 0.0, jnp.zeros(num_agents), jnp.zeros(num_agents),
                    jnp.zeros(num_agents), jnp.zeros(num_agents))
            out = jax.lax.scan(_step, init, jnp.arange(args.max_steps))[0][3:]
            soups, placements, dishes, deliveries, steps = out
            return jnp.concatenate([soups[None], placements, dishes, deliveries, steps])

        return run

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    raw_profiles = []
    for spec in args.checkpoints:
        path = Path(spec)
        if not path.is_absolute():
            path = HERE / path
        params = load_agents(path)
        critic_dim = int(np.asarray(params[0]["params"]["Dense_3"]["kernel"]).shape[0])
        raw = np.asarray(jax.vmap(make_rollout(params, critic_dim))(keys))
        soups = raw[:, 0]
        events = raw[:, 1:1 + 3 * num_agents].reshape(args.episodes, 3, num_agents).sum(axis=0)
        steps = raw[:, 1 + 3 * num_agents:].sum(axis=0)
        rates = events / np.maximum(steps, 1.0)[None, :]
        raw_profiles.append({"checkpoint": str(path), "team_soups_mean": float(soups.mean()),
                             "team_soups_max": float(soups.max()),
                             "rates": rates.tolist(), "steps": steps.tolist()})

    totals = np.array([p["rates"] for p in raw_profiles])           # (ckpts, 3 roles, agents)
    lo = totals.min(axis=(0, 2))                                    # (3,)
    hi = totals.max(axis=(0, 2))                                    # (3,)
    span = np.maximum(hi - lo, 1e-9)
    norm_all = (totals - lo[None, :, None]) / span[None, :, None]   # (ckpts, 3, agents)
    report = {"episodes": args.episodes, "seed": args.seed,
              "roles": ["c_ingredient", "c_pot", "c_service"],
              "note": "rates are per-step event counts; c_* are min-max normalised across the "
                      "measured checkpoints so that every entry lies in [0, 1]",
              "profiles": []}
    for idx, profile in enumerate(raw_profiles):
        norm = norm_all[idx]
        report["profiles"].append({
            "checkpoint": profile["checkpoint"],
            "team_soups_mean": profile["team_soups_mean"],
            "team_soups_max": profile["team_soups_max"],
            "raw_rates": profile["rates"],
            "capability_vectors": {f"agent{i}": [float(norm[r, i]) for r in range(3)]
                                   for i in range(num_agents)},
        })
    print(json.dumps(report, indent=2))
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[profile] wrote {out}")


if __name__ == "__main__":
    main()
