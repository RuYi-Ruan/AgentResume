"""Bottleneck phase diagram: sweep partner B's ingredient capability x partner C's service capability.

From identical initial states, the ego is forced into each candidate role (scripted fetcher or
scripted server) while partner B (agent 1) is an ingredient-capable checkpoint and partner C
(agent 2) is a service-capable checkpoint.  D(B, C) = R_service - R_ingredient.  A sign flip of D
inside the naturally learned capability range means a bottleneck migration exists and that capability
information has decision value; if D stays negative everywhere, the map's bottleneck is fixed by its
topology (long transport chain) rather than by partner competence.

(Latent-capability test: same state, same visible role, different latent competence.

Two cheap measurements requested before spending any Oracle-gate budget:

A. Counterfactual role flip. From *identical* initial states (same reset keys), the ego is forced
   to play a fixed scripted role (fetcher or server) while partner B is either the mid-competence or
   the late-competence checkpoint of the same identity (both are ingredient specialists that look
   alike). If the best ego role flips between B=mid and B=late, teammate capability has decision
   value that the current frame does not reveal.

B. Capability separability. Records the ego observation sequence for each episode so that a probe
   can test whether the *current frame* predicts partner competence and whether *history* does.

Usage:
  python latent_capability_test.py --partner-mid <ckpt>:0 --partner-late <ckpt>:0 --episodes 24
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
from train_ocv2_oracle_gate import MLPActorOnly, actor_subset  # noqa: E402
from generate_demos import (  # noqa: E402
    INTERACT,
    MOVE_VECTORS,
    STAY,
    approach_and_interact,
    object_cells,
    pot_cell,
    walkable,
)

ROLE_MASK = {"fetcher": ("runnerA", "runnerB"), "server": ("server",)}


class Ctx:
    privileged_critic = True


def load_partner(path: Path, agent: int):
    payload = pickle.load(path.open("rb"))
    agents = payload["params"]
    inner = agents[agent]["params"] if isinstance(agents, list) else agents["params"]
    return actor_subset(jax.tree.map(jnp.asarray, inner))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b-checkpoints", nargs="+", required=True,
                        help="ingredient-capable partner checkpoints (agent slot of each 'path:idx')")
    parser.add_argument("--c-checkpoints", nargs="+", required=True,
                        help="service-capable partner checkpoints")
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--throttle-k", type=int, default=0,
                        help="diagnostic only: force the B partner to STAY every K-th step "
                             "(K=2 halves its role throughput, K=3 gives ~1.5x slowdown)")
    parser.add_argument("--out", default="results/latent_capability.json")
    args = parser.parse_args()

    layout = Layout.from_string((HERE / args.layout_file).read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    crit = critic_spec(Ctx(), env)
    crit["context"] = crit
    mask = walkable(layout)
    objects = object_cells(layout)
    piles = [objects[n][0] for n in sorted(objects) if n.startswith("ingredient")]
    pots = objects["pot"]
    plates = objects["plate_pile"]

    def load_spec(spec):
        path_str, idx_str = spec.rsplit(":", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = HERE / path
        return load_partner(path, int(idx_str))

    b_variants = {spec: load_spec(spec) for spec in args.b_checkpoints}
    c_variants = {spec: load_spec(spec) for spec in args.c_checkpoints}
    partner_net = MLPActorOnly(action_dim=6)

    def scripted_action(state, agent, role, facing, blocked=frozenset()):
        here = (int(state.agents.pos.x[agent]), int(state.agents.pos.y[agent]))
        inventory = int(state.agents.inventory[agent])
        if role == "fetcher":
            pot = pots[agent % len(pots)]
            target = (min(piles, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                      if inventory == 0 else pot)
            return approach_and_interact(mask, here, facing, target, blocked)
        if inventory == 0:
            target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            return approach_and_interact(mask, here, facing, target, blocked)
        holding_plate = inventory == int(DynamicObject.PLATE)
        holding_dish = bool(inventory & int(DynamicObject.COOKED))
        if holding_dish:
            return approach_and_interact(mask, here, facing, objects["goal"][0], blocked)
        if holding_plate:
            cooked = [p for p in pots if pot_cell(state, p)[0] & int(DynamicObject.COOKED)]
            target = (min(cooked, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                      if cooked else min(pots, key=lambda p: abs(p[0] - here[0])
                                         + abs(p[1] - here[1])))
            return approach_and_interact(mask, here, facing, target, blocked)
        target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
        return approach_and_interact(mask, here, facing, target, frozenset())

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    results = {}
    frames = {}
    for b_spec, b_params in b_variants.items():
      for c_spec, c_params in c_variants.items():
        for role in ("fetcher", "server"):
            soups, lengths, obs_seq = [], [], []
            for episode, key in enumerate(keys):
                obs, state = env.reset(key)
                facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
                delivered = 0
                step = 0
                while step < args.max_steps:
                    # ego (agent 0): scripted role; agent 1: learned partner; agent 2: scripted server
                    occupied = {tuple(int(v) for v in (state.agents.pos.x[i],
                                                       state.agents.pos.y[i])): i
                                for i in range(3)}
                    ego_here = (int(state.agents.pos.x[0]), int(state.agents.pos.y[0]))
                    ego_blocked = frozenset(c for c, o in occupied.items() if o != 0)
                    a_ego = scripted_action(state, 0, role, facing[0], ego_blocked)
                    if a_ego == 4:
                        legal = [d for d, (dx, dy) in MOVE_VECTORS.items()
                                 if mask[ego_here[1] + dy, ego_here[0] + dx]
                                 and (ego_here[0] + dx, ego_here[1] + dy) not in ego_blocked]
                        if legal:
                            a_ego = legal[step % len(legal)]
                    p1 = partner_net.apply(
                        b_params, jnp.asarray(np.asarray(obs[env.agents[1]]).reshape(1, -1)))[0]
                    a_partner = int(jax.random.categorical(
                        jax.random.fold_in(key, step * 3 + 1), p1))
                    if args.throttle_k > 0 and step % args.throttle_k == 0:
                        a_partner = 4  # STAY: pure throughput handicap, role behaviour unchanged
                    p2 = partner_net.apply(
                        c_params, jnp.asarray(np.asarray(obs[env.agents[2]]).reshape(1, -1)))[0]
                    a_third = int(jax.random.categorical(
                        jax.random.fold_in(key, step * 3 + 2), p2))
                    if a_ego in MOVE_VECTORS:
                        facing[0] = a_ego
                    obs, state, reward, done, info = env.step(
                        key, state, {env.agents[0]: jnp.int32(a_ego),
                                     env.agents[1]: jnp.int32(a_partner),
                                     env.agents[2]: jnp.int32(a_third)})
                    delivered += int(bool(np.asarray(state.new_correct_delivery)))
                    obs_seq.append(np.asarray(obs[env.agents[0]]).reshape(-1))
                    step += 1
                    if bool(np.asarray(done["__all__"])):
                        break
                soups.append(delivered)
                lengths.append(step)
            key_name = (f"{Path(b_spec).name.split('.')[0]}|{Path(c_spec).name.split('.')[0]}"
                        f"|k{args.throttle_k}|{role}")
            results[key_name] = {
                "role": role, "b": b_spec, "c": c_spec,
                "team_soups_mean": float(np.mean(soups)),
                "team_soups_max": int(np.max(soups)),
                "episodes_with_delivery": int(np.sum(np.asarray(soups) > 0)),
                "episodes": args.episodes,
            }
    def r(b_spec, c_spec, role):
        return results[f"{Path(b_spec).name.split('.')[0]}|"
                       f"{Path(c_spec).name.split('.')[0]}|k{args.throttle_k}|{role}"]["team_soups_mean"]

    grid = []
    for b_spec in b_variants:
        for c_spec in c_variants:
            grid.append({"b": b_spec, "c": c_spec,
                         "r_ingredient": r(b_spec, c_spec, "fetcher"),
                         "r_service": r(b_spec, c_spec, "server"),
                         "D_service_minus_ingredient": r(b_spec, c_spec, "server")
                         - r(b_spec, c_spec, "fetcher")})
    report = {"throttle_k": args.throttle_k,
              "b_checkpoints": list(b_variants), "c_checkpoints": list(c_variants),
              "results": results, "grid": grid,
              "flip_cells": [g for g in grid if g["D_service_minus_ingredient"] > 0]}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for key, value in results.items():
        print(f"  {key:14s} soups {value['team_soups_mean']:.2f} "
              f"(max {value['team_soups_max']}, {value['episodes_with_delivery']}/"
              f"{value['episodes']} eps)")
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[latent] wrote {out}")


if __name__ == "__main__":
    main()
