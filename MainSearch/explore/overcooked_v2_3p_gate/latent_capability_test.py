"""Latent-capability test: same state, same visible role, different latent competence.

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
    parser.add_argument("--partner-mid", required=True, help="checkpoint:agent_index")
    parser.add_argument("--partner-late", required=True, help="checkpoint:agent_index")
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
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

    partners = {}
    for tag, spec in (("mid", args.partner_mid), ("late", args.partner_late)):
        path_str, idx_str = spec.rsplit(":", 1)
        path = Path(path_str)
        if not path.is_absolute():
            path = HERE / path
        partners[tag] = load_partner(path, int(idx_str))
    partner_net = MLPActorOnly(action_dim=6)

    def scripted_action(state, agent, role, facing):
        here = (int(state.agents.pos.x[agent]), int(state.agents.pos.y[agent]))
        inventory = int(state.agents.inventory[agent])
        if role == "fetcher":
            pot = pots[agent % len(pots)]
            target = (min(piles, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                      if inventory == 0 else pot)
            return approach_and_interact(mask, here, facing, target, frozenset())
        if inventory == 0:
            target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
            return approach_and_interact(mask, here, facing, target, frozenset())
        holding_plate = inventory == int(DynamicObject.PLATE)
        holding_dish = bool(inventory & int(DynamicObject.COOKED))
        if holding_dish:
            return approach_and_interact(mask, here, facing, objects["goal"][0], frozenset())
        if holding_plate:
            cooked = [p for p in pots if pot_cell(state, p)[0] & int(DynamicObject.COOKED)]
            target = (min(cooked, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
                      if cooked else min(pots, key=lambda p: abs(p[0] - here[0])
                                         + abs(p[1] - here[1])))
            return approach_and_interact(mask, here, facing, target, frozenset())
        target = min(plates, key=lambda p: abs(p[0] - here[0]) + abs(p[1] - here[1]))
        return approach_and_interact(mask, here, facing, target, frozenset())

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    results = {}
    frames = {"mid": [], "late": []}
    for tag in ("mid", "late"):
        params = partners[tag]
        for role in ("fetcher", "server"):
            soups, lengths, obs_seq = [], [], []
            for episode, key in enumerate(keys):
                obs, state = env.reset(key)
                facing = [int(np.asarray(state.agents.dir[i])) for i in range(3)]
                delivered = 0
                step = 0
                while step < args.max_steps:
                    # ego (agent 0): scripted role; agent 1: learned partner; agent 2: scripted server
                    a_ego = scripted_action(state, 0, role, facing[0])
                    p_obs = jnp.asarray(np.asarray(obs[env.agents[1]]).reshape(1, -1))
                    logits = partner_net.apply(params, p_obs)[0]
                    a_partner = int(jax.random.categorical(
                        jax.random.fold_in(key, step), logits))
                    a_third = scripted_action(state, 2, "server", facing[2])
                    if a_ego in MOVE_VECTORS:
                        facing[0] = a_ego
                    if a_third in MOVE_VECTORS:
                        facing[2] = a_third
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
                frames[tag].append(np.asarray(obs_seq, dtype=np.int8))
            results[f"{tag}_{role}"] = {
                "role": role, "partner": tag,
                "team_soups_mean": float(np.mean(soups)),
                "team_soups_max": int(np.max(soups)),
                "episodes_with_delivery": int(np.sum(np.asarray(soups) > 0)),
                "episodes": args.episodes,
            }
    report = {"partner_mid": args.partner_mid, "partner_late": args.partner_late,
              "results": results,
              "flip_gap_mid": results["mid_fetcher"]["team_soups_mean"]
              - results["mid_server"]["team_soups_mean"],
              "flip_gap_late": results["late_fetcher"]["team_soups_mean"]
              - results["late_server"]["team_soups_mean"]}
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for key, value in results.items():
        print(f"  {key:14s} soups {value['team_soups_mean']:.2f} "
              f"(max {value['team_soups_max']}, {value['episodes_with_delivery']}/"
              f"{value['episodes']} eps)")
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(HERE / "results" / "latent_frames.npz",
                        mid=np.array([f for f in frames["mid"]], dtype=object),
                        late=np.array([f for f in frames["late"]], dtype=object),
                        allow_pickle=True)
    print(f"[latent] wrote {out}")


if __name__ == "__main__":
    main()
