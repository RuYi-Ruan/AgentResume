"""Paired evaluation of the three-arm capability Oracle Gate (archetype vocabulary version).

Partners are frozen *role archetypes* (e.g. strong/weak fetcher, strong/weak server). For every
partner pairing the two arms are evaluated on the SAME fixed initial states, giving per-episode
paired differences with a standard error, plus matched-state policy-shift metrics (TV / JS /
common-random-numbers switch rate) between the two egos.

Usage:
  python evaluate_oracle_gate.py --arm-dir results/ocv2_archgate_no_oracle_seed0_1M \
      --other-dir results/ocv2_archgate_oracle_seed0_1M \
      --partner-checkpoints bc/arch_fetcher_strong.pkl bc/arch_fetcher_weak.pkl \
                             bc/arch_server_strong.pkl bc/arch_server_weak.pkl \
      --pairs "0,3;1,2" --episodes 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overcooked_v2_ff_gate"))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import MLPActorCriticPriv, critic_spec  # noqa: E402
from train_ocv2_oracle_gate import (  # noqa: E402
    MLPActorOnly,
    actor_subset,
    load_plain_params,
    per_env_state,
)


class Ctx:
    privileged_critic = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument("--other-dir", required=True)
    parser.add_argument("--partner-checkpoints", nargs="+", required=True)
    parser.add_argument("--pairs", default="0,3;1,2")
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out", default="results/oracle_gate_eval.json")
    args = parser.parse_args()

    layout = Layout.from_string((HERE / args.layout_file).read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    crit = critic_spec(Ctx(), env)
    crit["context"] = crit
    num_agents = len(env.agents)
    vocab = len(args.partner_checkpoints)
    pairs = [[int(i) for i in spec.split(",")] for spec in args.pairs.split(";")]
    obs_flat = int(np.prod(env.observation_space().shape))
    z_dim = 2 * vocab

    level_paths = [Path(p) if Path(p).is_absolute() else HERE / p
                   for p in args.partner_checkpoints]
    level_params = jax.tree.map(lambda *xs: jnp.stack(xs),
                                *[actor_subset(load_plain_params(p)["params"])
                                  for p in level_paths])
    partner_net = MLPActorOnly(action_dim=6)

    arms = {}
    for name, run_dir in (("no_oracle", args.arm_dir), ("oracle", args.other_dir)):
        path = Path(run_dir)
        if not path.is_absolute():
            path = HERE / path
        ckpt = sorted(path.glob("checkpoint_*.pkl"))[-1]
        params = load_plain_params(ckpt)
        arms[name] = {"params": {"params": jax.tree.map(jnp.asarray, params["params"])},
                      "checkpoint": str(ckpt)}
        print(f"[eval] {name}: {ckpt.name}")

    critic_dim = crit["dim"] + z_dim
    networks = {name: MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim) for name in arms}

    def ego_input(arm, views, arch):
        flat = views[0].reshape(-1)
        if arm != "oracle":
            return flat
        z = jnp.concatenate([jax.nn.one_hot(arch[0], vocab), jax.nn.one_hot(arch[1], vocab)])
        return jnp.concatenate([flat, z])

    def make_evaluator(arm: str, sample: bool):
        network = networks[arm]

        @jax.jit
        def evaluate(params, keys, pair):
            def _episode(key):
                obs, state = env.reset(key)

                def _step(carry, t):
                    obs, state, live, delivered, length = carry
                    views = jnp.stack([obs[a] for a in env.agents], axis=0)
                    batched = jax.tree.map(lambda x: x[None], state)
                    gstate = jnp.concatenate([per_env_state(batched, crit, t)[0],
                                              jax.nn.one_hot(pair[0], vocab),
                                              jax.nn.one_hot(pair[1], vocab)])
                    logits, _ = network.apply(params, ego_input(arm, views, pair)[None],
                                              gstate[None])
                    logits = logits[0]
                    a_ego = (jax.random.categorical(jax.random.fold_in(key, t), logits)
                             if sample else jnp.argmax(logits))
                    def partner_logits_for(slot: int, arch_index):
                        return jax.vmap(lambda o: partner_net.apply(
                            jax.tree.map(lambda x: x[arch_index], level_params), o))(
                            views[slot].reshape(-1)[None])[0]
                    lp1 = partner_logits_for(1, pair[0])
                    lp2 = partner_logits_for(2, pair[1])
                    stacked = jnp.stack([lp1, lp2], axis=0)
                    if sample:
                        a_partners = jax.vmap(lambda k, lg: jax.random.categorical(k, lg))(
                            jax.random.split(jax.random.fold_in(key, t + 77), 2), stacked)
                    else:
                        a_partners = jnp.argmax(stacked, axis=-1)
                    actions = jnp.concatenate([a_ego[None], a_partners])
                    obs, state, reward, done, info = env.step(
                        key, state, {a: actions[i] for i, a in enumerate(env.agents)})
                    delivered = delivered + jnp.where(live & state.new_correct_delivery, 1.0, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~done["__all__"]
                    return (obs, state, live, delivered, length), None

                init = (obs, state, jnp.array(True), 0.0, jnp.int32(0))
                (_, _, _, delivered, length), _ = jax.lax.scan(
                    _step, init, jnp.arange(args.max_steps))
                return delivered, length.astype(jnp.float32)

            return jax.vmap(_episode)(keys)

        return evaluate

    def make_shift_fn(pair):
        network_a, network_b = networks["no_oracle"], networks["oracle"]

        @jax.jit
        def shift(params_a, params_b, keys):
            def _episode(key):
                obs, state = env.reset(key)

                def _step(carry, t):
                    obs, state, live, tv, js, switch, steps = carry
                    views = jnp.stack([obs[a] for a in env.agents], axis=0)
                    batched = jax.tree.map(lambda x: x[None], state)
                    gstate = jnp.concatenate([per_env_state(batched, crit, t)[0],
                                              jax.nn.one_hot(pair[0], vocab),
                                              jax.nn.one_hot(pair[1], vocab)])
                    logits_a, _ = network_a.apply(
                        params_a, ego_input("no_oracle", views, pair)[None], gstate[None])
                    logits_b, _ = network_b.apply(
                        params_b, ego_input("oracle", views, pair)[None], gstate[None])
                    p_a = jax.nn.softmax(logits_a[0])
                    p_b = jax.nn.softmax(logits_b[0])
                    step_tv = 0.5 * jnp.sum(jnp.abs(p_a - p_b))
                    mix = 0.5 * (p_a + p_b)
                    step_js = 0.5 * jnp.sum(p_a * (jnp.log(p_a + 1e-9) - jnp.log(mix + 1e-9))) \
                        + 0.5 * jnp.sum(p_b * (jnp.log(p_b + 1e-9) - jnp.log(mix + 1e-9)))
                    g = jax.random.gumbel(jax.random.fold_in(key, t), p_a.shape)
                    pick_a = jnp.argmax(jnp.log(p_a + 1e-9) + g)
                    pick_b = jnp.argmax(jnp.log(p_b + 1e-9) + g)
                    tv = tv + jnp.where(live, step_tv, 0.0)
                    js = js + jnp.where(live, step_js, 0.0)
                    switch = switch + jnp.where(live & (pick_a != pick_b), 1.0, 0.0)
                    steps = steps + jnp.where(live, 1.0, 0.0)
                    p1 = jax.vmap(lambda o: partner_net.apply(
                        jax.tree.map(lambda x: x[pair[0]], level_params), o))(
                        views[1].reshape(-1)[None])[0]
                    p2 = jax.vmap(lambda o: partner_net.apply(
                        jax.tree.map(lambda x: x[pair[1]], level_params), o))(
                        views[2].reshape(-1)[None])[0]
                    a_partners = jnp.argmax(jnp.stack([p1, p2], axis=0), axis=-1)
                    a_ego = jnp.argmax(logits_a[0])
                    obs, state, _, done, _ = env.step(
                        key, state,
                        {a: jnp.concatenate([a_ego[None], a_partners])[i]
                         for i, a in enumerate(env.agents)})
                    live = live & ~done["__all__"]
                    return (obs, state, live, tv, js, switch, steps), None

                init = (obs, state, jnp.array(True), 0.0, 0.0, 0.0, 0.0)
                return jnp.stack(jax.lax.scan(_step, init, jnp.arange(args.max_steps))[0][3:])

            return jax.vmap(_episode)(keys)

        return shift

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    report = {"checkpoints": {k: v["checkpoint"] for k, v in arms.items()},
              "partner_vocabulary": [p.name for p in level_paths],
              "pairs": pairs, "episodes": args.episodes, "arms": {k: args.arm_dir
                                                                 if k == "no_oracle"
                                                                 else args.other_dir
                                                                 for k in arms}}
    for mode in ("greedy", "sampled"):
        evaluators = {name: make_evaluator(name, mode == "sampled") for name in arms}
        per_pair, pooled_a, pooled_b = {}, [], []
        for index, pair in enumerate(pairs):
            out = {}
            for name in arms:
                delivered, _ = jax.device_get(
                    evaluators[name](arms[name]["params"], keys, jnp.asarray(pair)))
                out[name] = np.asarray(delivered)
            diff = out["oracle"] - out["no_oracle"]
            se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
            per_pair[f"pair{index}_{pair[0]}{pair[1]}"] = {
                "no_oracle_mean": float(out["no_oracle"].mean()),
                "oracle_mean": float(out["oracle"].mean()),
                "paired_difference": float(diff.mean()),
                "paired_se": se,
                "t_statistic": float(diff.mean() / (se + 1e-12)),
            }
            pooled_a.append(out["no_oracle"])
            pooled_b.append(out["oracle"])
        diff = np.concatenate(pooled_b) - np.concatenate(pooled_a)
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        per_pair["overall"] = {
            "no_oracle_mean": float(np.concatenate(pooled_a).mean()),
            "oracle_mean": float(np.concatenate(pooled_b).mean()),
            "paired_difference": float(diff.mean()),
            "paired_se": se,
            "t_statistic": float(diff.mean() / (se + 1e-12)),
        }
        report[f"paired_{mode}"] = per_pair
        print(f"\n[{mode}] paired team soups per episode")
        for key, values in per_pair.items():
            print(f"  {key:14s} no_oracle {values['no_oracle_mean']:.2f} | "
                  f"oracle {values['oracle_mean']:.2f} | diff "
                  f"{values['paired_difference']:+.2f} +/- {values['paired_se']:.2f} "
                  f"(t={values['t_statistic']:.2f})")

    shift_rows = [np.asarray(jax.device_get(
        make_shift_fn(pair)(arms["no_oracle"]["params"], arms["oracle"]["params"], keys)
    )).reshape(args.episodes, 4) for pair in pairs]
    shifted = np.concatenate(shift_rows)
    report["policy_shift"] = {
        "tv_mean": float(shifted[:, 0].sum() / shifted[:, 3].sum()),
        "js_mean_nats": float(shifted[:, 1].sum() / shifted[:, 3].sum()),
        "crn_action_switch_rate": float(shifted[:, 2].sum() / shifted[:, 3].sum()),
        "matched_steps": int(shifted[:, 3].sum()),
    }
    print("\n[policy shift on matched states]")
    print(json.dumps(report["policy_shift"], indent=2))

    overall = report["paired_sampled"]["overall"]
    report["gate_passed"] = bool(overall["paired_difference"] >= 0.5
                                 and overall["t_statistic"] > 1.0)
    report["gate_strong"] = bool(overall["paired_difference"] >= 1.0)
    print(f"\n[eval] gate: delta {overall['paired_difference']:+.2f} "
          f"({'PASS' if report['gate_passed'] else 'FAIL'}"
          f"{', STRONG' if report['gate_strong'] else ''})")
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
