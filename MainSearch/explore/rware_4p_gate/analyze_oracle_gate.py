"""Decision analysis of the RWARE capability Oracle Gate.

Two readings, both requested by the external guidance:

1. Paired delivery comparison. Both arms are evaluated on the *same* fixed initial states and the
   *same* frozen partner competence level, so the difference is a per-episode paired difference
   with a proper standard error (not a difference of two independent means).

2. Action disagreement rate. On identical situations (same episode, same level, same lock-stepped
   environment state) how often do the two arms' argmax ego actions differ? This is the direct
   evidence for "does capability knowledge change the coordination decision", independent of how
   many points it is worth in this particular layout.

Screening only: the frozen partner checkpoints are used to make competence a real independent
variable; this is not an ETM result.

Usage:
  python analyze_oracle_gate.py --eval-episodes 32 --eval-seed 0
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
sys.path.insert(0, str(HERE))

from train_rware_ippo import (  # noqa: E402
    ActorCritic,
    make_env,
    masked_logits,
    preprocess,
)
from train_rware_oracle_gate import (  # noqa: E402
    LEVEL_CHECKPOINTS,
    NUM_LEVELS,
    build_eval_fn,
    load_level_params,
)

ARMS = ["no_oracle", "oracle"]
LEVEL_NAMES = ["low", "mid", "high"]


def load_final_params(run_dir: Path, key: str):
    checkpoints = sorted(run_dir.glob("checkpoint_*.pkl"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints in {run_dir}")
    with checkpoints[-1].open("rb") as handle:
        payload = pickle.load(handle)
    return jax.tree.map(jnp.asarray, payload[key]), checkpoints[-1]


def paired_difference(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    diff = values_b - values_a
    n = diff.size
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    mean = float(diff.mean())
    return {
        "mean_difference": mean,
        "paired_se": se,
        "t_statistic": mean / se if se and se > 0 else float("nan"),
        "n_pairs": int(n),
        "a_mean": float(values_a.mean()),
        "b_mean": float(values_b.mean()),
    }


def make_disagreement_fn(config: dict, env, level_params, time_limit: int,
                         ego_features_a: int, ego_features_b: int):
    """Lock-stepped rollout: both arms see identical states; count argmax action disagreement."""
    num_agents = config["num_agents"]
    net = ActorCritic(action_dim=config["action_dim"])

    def ego_input(features, level, with_oracle: bool):
        if not with_oracle:
            return features[0]
        return jnp.concatenate([features[0], jax.nn.one_hot(level, NUM_LEVELS)], axis=-1)

    # network input widths are compile-time constants here (no_oracle first, oracle second)
    uses_oracle_a = ego_features_a != config["num_features"]
    uses_oracle_b = ego_features_b != config["num_features"]

    def _make(reference: int):
        @jax.jit
        def run(params_a, params_b, keys, level):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, diff, total, kl = carry
                    features, action_mask = preprocess(timestep.observation, num_agents)
                    logits_a, _ = net.apply(
                        params_a, ego_input(features, level, uses_oracle_a))
                    logits_b, _ = net.apply(
                        params_b, ego_input(features, level, uses_oracle_b))
                    logits_a = masked_logits(logits_a, action_mask[0])
                    logits_b = masked_logits(logits_b, action_mask[0])
                    action_a = jnp.argmax(logits_a)
                    action_b = jnp.argmax(logits_b)
                    diff = diff + jnp.where(live & (action_a != action_b), 1.0, 0.0)
                    total = total + jnp.where(live, 1.0, 0.0)
                    p_a = jax.nn.softmax(logits_a)
                    p_b = jax.nn.softmax(logits_b)
                    step_kl = jnp.sum(p_a * (jnp.log(p_a + 1e-9) - jnp.log(p_b + 1e-9)))
                    kl = kl + jnp.where(live, step_kl, 0.0)

                    chosen = jax.tree.map(lambda x: x[level], level_params)
                    partner_params = jax.tree.map(lambda x: x[1:], chosen)
                    logits_p, _ = jax.vmap(lambda p, o: net.apply(p, o))(
                        partner_params, features[1:])
                    partner_actions = jnp.argmax(masked_logits(logits_p, action_mask[1:]), axis=-1)
                    ego_action = jnp.where(reference == 0, action_a, action_b)
                    action = jnp.concatenate([ego_action[None], partner_actions])
                    state, timestep = env.step(state, action)
                    live = live & ~timestep.last()
                    return (state, timestep, live, diff, total, kl), None

                init = (state, timestep, jnp.array(True), 0.0, 0.0, 0.0)
                diff, total, kl = jax.lax.scan(_step, init, jnp.arange(time_limit))[0][3:]
                return jnp.stack([diff, total, kl])

            return jax.vmap(_episode)(keys)

        return run

    return _make


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--out", default="results/oracle_gate_analysis.json")
    args = parser.parse_args()

    configs, params, checkpoints = {}, {}, {}
    for arm in ARMS:
        run_dir = HERE / "results" / f"rware_oracle_{arm}_seed0_8M"
        metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        configs[arm] = metadata["config"]
        params[arm], checkpoints[arm] = load_final_params(run_dir, "ego_params")
        print(f"[gate] {arm}: {checkpoints[arm].name} ego_features={configs[arm]['ego_features']}")

    partner_run = Path(json.loads(
        (HERE / "results" / "rware_oracle_no_oracle_seed0_8M" / "run.json").read_text())["partner_run"])
    level_params = load_level_params(partner_run)
    config = configs["no_oracle"]
    env = make_env(time_limit=config["time_limit"], sensor_range=config["sensor_range"])
    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_episodes)

    report: dict = {
        "kind": "rware_oracle_gate_analysis",
        "not_an_etm_result": True,
        "partner_run": str(partner_run),
        "partner_levels": LEVEL_CHECKPOINTS,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "checkpoints": {arm: checkpoints[arm].name for arm in ARMS},
    }

    # 1. paired deliveries per level and per mode
    per_level: dict = {}
    for mode in ("greedy", "sampled"):
        evaluators, _ = build_eval_fn(configs["no_oracle"], env, level_params,
                                      args.eval_episodes, args.eval_seed)
        evaluators_oracle, _ = build_eval_fn(configs["oracle"], env, level_params,
                                             args.eval_episodes, args.eval_seed)
        per_level[mode] = {}
        pooled_a, pooled_b = [], []
        for level in range(NUM_LEVELS):
            # the evaluators return (delivered, episode_length); keep deliveries only
            raw_a, _ = jax.device_get(evaluators[mode](params["no_oracle"], keys,
                                                       jnp.asarray(level)))
            raw_b, _ = jax.device_get(evaluators_oracle[mode](params["oracle"], keys,
                                                             jnp.asarray(level)))
            raw_a = np.asarray(raw_a)
            raw_b = np.asarray(raw_b)
            per_level[mode][LEVEL_NAMES[level]] = paired_difference(raw_a, raw_b)
            pooled_a.append(raw_a)
            pooled_b.append(raw_b)
        pooled = paired_difference(np.concatenate(pooled_a), np.concatenate(pooled_b))
        per_level[mode]["overall"] = pooled
    report["paired_deliveries"] = per_level

    # 2. action disagreement on identical states
    disagreement_fn = make_disagreement_fn(
        config, env, level_params, config["time_limit"],
        configs["no_oracle"]["ego_features"], configs["oracle"]["ego_features"])
    disagreement = {}
    for reference in ARMS:
        run = disagreement_fn(ARMS.index(reference))
        per_level_dis = {}
        pooled = []
        for level in range(NUM_LEVELS):
            raw = np.asarray(jax.device_get(run(
                params["no_oracle"], params["oracle"], keys, jnp.asarray(level))))
            diff, total, kl = raw.T
            per_level_dis[LEVEL_NAMES[level]] = {
                "action_disagreement_rate": float(diff.sum() / max(total.sum(), 1)),
                "episodes_with_any_disagreement": float(np.mean(diff > 0)),
                "mean_action_kl": float(kl.mean()),
                "steps": int(total.sum()),
            }
            pooled.append(raw)
        pooled_raw = np.concatenate(pooled)
        per_level_dis["overall"] = {
            "action_disagreement_rate": float(pooled_raw[:, 0].sum() / max(pooled_raw[:, 1].sum(), 1)),
            "episodes_with_any_disagreement": float(np.mean(pooled_raw[:, 0] > 0)),
            "mean_action_kl": float(pooled_raw[:, 2].mean()),
            "steps": int(pooled_raw[:, 1].sum()),
        }
        disagreement[reference] = per_level_dis
    report["action_disagreement"] = disagreement

    print("\n=== paired deliveries (mean per episode, 32 fixed states per level) ===")
    for mode in ("greedy", "sampled"):
        print(f"  [{mode}]")
        for key, stats in per_level[mode].items():
            print(f"    {key:8s} no_oracle {stats['a_mean']:.2f} | oracle {stats['b_mean']:.2f} "
                  f"| diff {stats['mean_difference']:+.2f} +/- {stats['paired_se']:.2f} "
                  f"(t={stats['t_statistic']:.2f})")
    print("\n=== action disagreement on identical states ===")
    for reference, stats in disagreement.items():
        print(f"  [trajectory driven by {reference}]")
        for key, values in stats.items():
            print(f"    {key:8s} disagreement {values['action_disagreement_rate'] * 100:.1f}% "
                  f"| episodes with any disagreement {values['episodes_with_any_disagreement'] * 100:.0f}% "
                  f"| mean KL {values['mean_action_kl']:.4f}")

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[gate] wrote {out}")


if __name__ == "__main__":
    main()
