"""Partner profile table for the Oracle information-value gate (`ORACLE_GATE_CONTRACT.md` S1/S1.1).

What this produces
------------------
`results/oracle_gate/profiles.json`: the three-dimensional profile of every one of the **12
candidate partner policies** = identities `{1,2,3,4}` x stages `{u50, u600, u1250}` read from
`results/smax_2s3z_indep_1250x64x64/`, plus the **frozen** normalisation (`mean`, `std` over
exactly those 12 candidates) and the derived per-candidate `z`.

Measurement convention (must be quoted with any number taken from here)
----------------------------------------------------------------------
* Observer = `ally_0` (identity 0).  Partners = identities 1..4.  Only the 12 candidate policies
  above are measured; nothing else.
* Reference team: **every identity other than the tested one uses its own u1250 parameters**
  (identity 0 included).  The tested identity is replaced by the candidate stage parameters:
      config(identity=i, stage=s) = u1250 params for j != i, u1250 checkpoint's identity-i params
      replaced by stage-s params of identity i.
  Rationale: the u1250 checkpoint is the converged reference team of this run, i.e. the same
  frozen-reference convention as `evaluate_reference_team.py`, so the profile difference between
  two stages of one identity is attributable to that identity's parameters alone.
  Consequence to state explicitly: for a fixed stage `s`, all four identity candidates of that
  stage share the *same* four teammates, and the four **u1250 candidates are the identical
  configuration** (all-u1250 team) - their profiles are therefore read off the same 20 battles
  (they differ only in which identity is scored).  This is a property of the frozen contract, not
  an extra sample.
* Measurement battles: fixed seed set A = [2234, 2235, 2236, 2237, 2238] x 4 episodes per seed
  = 20 episode executions, replayed with identical reset keys for all 12 candidates.
  A is disjoint from the training seed stream and from the test set C = [3234, 3235, 3236, 3237]
  (asserted at runtime).  The 12 x 20 = 240 executions are 240 *executions of a fixed battle set*,
  NOT 240 independent samples.
* Metric definitions (contract S1), aggregated over the 20 battles as a ratio of pooled totals
  (the same "quantity / total episode length" reading the contract uses):
    - `damage_per_len` = (sum over battles of the damage this identity dealt) / (sum of lengths)
    - `alive_frac`     = (sum over battles of this identity's alive steps) / (sum of lengths)
    - `focus_share`    = (sum of valid attacks that had >= 1 other ally on the same target at the
                          same step) / (sum of valid attacks); 0.0 when there is no valid attack.
  "Valid attack" = this identity fired at an enemy that lost health in that environment step,
  i.e. exactly the attacks that `Trainer._evaluate`'s even-split attribution credits damage for.
  Damage/target decoding and even-split attribution are the ones of
  `train_smax_2s3z_independent.Trainer._evaluate` (L631-651) and `test_preflight.check_g`.
* Self-check (printed, and mirrored in `profiles_selfcheck.json`): per battle, the damage
  attributed to all five allies must account for the enemies' own health ledger
  (`unit_health` deltas) up to the wall deaths that have no shooter, exactly as `check_g` rule (g).

Interpretation constraints (contract S6, repeated on purpose)
------------------------------------------------------------
A separated / valid interface with no measurable benefit only supports "no benefit was found in
this configuration", never "partner information has no value".  No causal attribution.  The
20 executions are not independent samples.  `sd(seed)` is neither a confidence interval nor a
criterion.

Usage (absolute interpreter path required on this machine)::

  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      measure_partner_profiles.py --tiny
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      measure_partner_profiles.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import train_smax_2s3z_independent as train  # noqa: E402
from evaluate_reference_team import (  # noqa: E402
    build_mixed_params,
    discover_checkpoints,
    load_checkpoint_params,
    load_config,
    resolve_reference,
    resolve_run_dir,
    select_stages,
)

DEFAULT_RUN_DIR = "results/smax_2s3z_indep_1250x64x64"
DEFAULT_STAGES = "50,600,1250"
DEFAULT_SEEDS = "2234,2235,2236,2237,2238"
DEFAULT_EPISODES = 4
TEST_SEEDS_C = [3234, 3235, 3236, 3237]  # contract S3, only for the disjointness assertion

METRIC_NAMES = ["damage_per_len", "alive_frac", "focus_share"]
PARTNER_IDENTITIES = [1, 2, 3, 4]
Z_LAYOUT = "identity order [1,2,3,4], 3 dims each, concatenated -> 12 dims"

DISCLAIMER = (
    "PROFILE MEASUREMENT ONLY.  The 12 x 20 episode executions are 20 fixed battles replayed per "
    "candidate, NOT independent samples; sd over seed groups is neither a confidence interval nor "
    "a criterion.  No causal attribution.  If the profiles turn out separable but the gate shows "
    "no benefit, the only admissible statement is 'no benefit was found in this configuration', "
    "never 'partner information has no value'."
)


# --------------------------------------------------------------------------------------------
# Measurement (mirrors Trainer._evaluate's attribution, adds the focus counters)
# --------------------------------------------------------------------------------------------
def make_measure_fn(trainer: train.Trainer, num_episodes: int, seeds: Sequence[int]):
    """Jitted `(params, keys) -> (team, per_agent)` over `len(seeds) x num_episodes` battles.

    `team`      : (S, E, 4) -> [won, episode_return, episode_length, enemy_health_loss]
    `per_agent` : (S, E, A, 6) -> [dealt, taken, kills, alive_steps, valid_attacks, focused]
    """
    config = trainer.config
    env, base_env, network = trainer.env, trainer.base_env, trainer.network
    num_agents = config["num_agents"]
    num_allies, num_enemies = config["num_allies"], config["num_enemies"]
    num_movement = config["num_movement_actions"]
    num_units = num_allies + num_enemies
    unit_index = jnp.arange(num_units)
    agents = env.agents
    horizon = config["horizon"]
    hidden_dim = config["gru_hidden_dim"]

    def _evaluate(params, keys):
        def _episode(key):
            obs, env_state = env.reset(key)
            hidden = tuple(
                train.ScannedRNN.initialize_carry(1, hidden_dim) for _ in range(num_agents)
            )

            def _step(carry, t):
                (env_state, obs, hidden, live, dealt, alive_steps, valid, focused, taken, kills,
                 raw_loss, ep_return, length, won) = carry
                smax = env_state.env_state.state
                alive_before = smax.unit_alive
                health_before = smax.unit_health
                avail = base_env.get_avail_actions(env_state.env_state)
                obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
                avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)

                actions, new_hidden = [], []
                for i in range(num_agents):
                    h_i, pi_i, _ = network.apply(
                        params[i], hidden[i],
                        (obs_stack[i][None, None], jnp.zeros((1, 1), dtype=bool),
                         avail_stack[i][None]),
                    )
                    logits = jnp.where(avail_stack[i][None] > 0, pi_i.logits, -1e9)
                    actions.append(jnp.argmax(logits, axis=-1)[0, 0])
                    new_hidden.append(h_i)
                env_act = {a: actions[i] for i, a in enumerate(agents)}
                new_obs, new_env_state, reward, done_new, info = env.step(
                    jax.random.fold_in(key, t), env_state, env_act
                )

                # --- per-identity attribution of this environment step (Trainer._evaluate) -------
                smax_new = new_env_state.env_state.state
                health_after = smax_new.unit_health
                alive_after = smax_new.unit_alive
                prev_attack = smax_new.prev_attack_actions
                health_loss = jnp.clip(health_before - health_after, 0.0)
                fired = prev_attack >= num_movement
                target = jnp.where(
                    unit_index < num_allies,
                    prev_attack,
                    num_allies - 1 - (prev_attack - num_movement),
                )
                shoots = (
                    (target[:, None] == unit_index[None, :])
                    & fired[:, None]
                    & alive_before[:, None]
                    & (unit_index[:, None] != unit_index[None, :])
                )
                shooters = shoots.sum(0)
                share = shoots / jnp.maximum(shooters, 1)[None, :]
                dealt_step = (share * health_loss[None, :]).sum(1)
                taken_step = health_loss * (shooters > 0)
                died = (alive_before & ~alive_after).astype(jnp.float32)
                kills_step = share @ died

                # --- focus: this identity's attacks that damaged an enemy another ally also hit --
                # An enemy target can only be shot by allies, so `shooters[j] >= 2` on an enemy
                # column means "this ally + at least one other ally aimed at the same enemy".
                ally_hits_enemy = (
                    shoots[:num_allies][:, num_allies:] & (health_loss[num_allies:] > 0)[None, :]
                )
                contested = (shooters[num_allies:] >= 2)[None, :]
                valid_step = ally_hits_enemy.sum(1)
                focused_step = (ally_hits_enemy & contested).sum(1)

                live_f = live.astype(jnp.float32)
                dealt = dealt + dealt_step[:num_allies] * live_f
                taken = taken + taken_step[:num_allies] * live_f
                kills = kills + kills_step[:num_allies] * live_f
                alive_steps = alive_steps + alive_before[:num_allies] * live_f
                valid = valid + valid_step * live_f
                focused = focused + focused_step * live_f
                # Enemy-side ledger for the self-check: all health the enemies actually lost.
                raw_loss = raw_loss + jnp.sum(health_loss[num_allies:]) * live_f

                done_all = done_new["__all__"]
                ep_return = ep_return + jnp.where(live, reward[agents[0]], 0.0)
                length = length + jnp.where(live, 1, 0)
                won = jnp.where(
                    live & done_all,
                    jnp.asarray(info["returned_won_episode"][0], dtype=jnp.float32),
                    won,
                )
                new_carry = (
                    new_env_state, new_obs, tuple(new_hidden), live & ~done_all,
                    dealt, alive_steps, valid, focused, taken, kills, raw_loss,
                    ep_return, length, won,
                )
                return new_carry, None

            ally_zeros = jnp.zeros((num_allies,), dtype=jnp.float32)
            init = (
                env_state, obs, hidden, jnp.asarray(True),
                ally_zeros, ally_zeros, ally_zeros, ally_zeros, ally_zeros, ally_zeros,
                jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.int32), jnp.zeros((), dtype=jnp.float32),
            )
            final = jax.lax.scan(_step, init, jnp.arange(horizon))[0]
            (_, _, _, _, dealt, alive_steps, valid, focused, taken, kills, raw_loss, ep_return,
             length, won) = final
            team = jnp.stack([won, ep_return, length.astype(jnp.float32), raw_loss])
            per_agent = jnp.stack([dealt, taken, kills, alive_steps, valid, focused], axis=1)
            return team, per_agent

        return jax.vmap(lambda k: _episode(k))(keys)

    impl = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
    eval_keys = jnp.stack(
        [jax.random.split(jax.random.PRNGKey(int(s)), num_episodes) for s in seeds]
    )
    return impl, eval_keys


# --------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------
def candidate_metrics(team: np.ndarray, per_agent: np.ndarray, identity: int,
                      seeds: Sequence[int]) -> dict:
    """Pooled S1 metrics for one identity over the fixed battle set + per-seed provenance."""
    lengths = team[:, :, 2]
    dealt = per_agent[:, :, identity, 0]
    alive = per_agent[:, :, identity, 3]
    valid = per_agent[:, :, identity, 4]
    focused = per_agent[:, :, identity, 5]

    total_length = float(lengths.sum())
    total_valid = float(valid.sum())
    pooled = [
        float(dealt.sum() / total_length) if total_length else 0.0,
        float(alive.sum() / total_length) if total_length else 0.0,
        float(focused.sum() / total_valid) if total_valid > 0 else 0.0,
    ]
    per_seed = []
    for s, seed in enumerate(seeds):
        seed_length = float(lengths[s].sum())
        seed_valid = float(valid[s].sum())
        per_seed.append({
            "seed": int(seed),
            "episodes": int(lengths.shape[1]),
            "episode_lengths": [int(round(float(x))) for x in lengths[s]],
            "damage_dealt": float(dealt[s].sum()),
            "alive_steps": float(alive[s].sum()),
            "valid_attacks": float(seed_valid),
            "focused_attacks": float(focused[s].sum()),
            "damage_per_len": float(dealt[s].sum() / seed_length) if seed_length else 0.0,
            "alive_frac": float(alive[s].sum() / seed_length) if seed_length else 0.0,
            "focus_share": float(focused[s].sum() / seed_valid) if seed_valid > 0 else 0.0,
        })
    return {
        "metrics": pooled,
        "pooled_totals": {
            "total_length": total_length,
            "damage_dealt": float(dealt.sum()),
            "alive_steps": float(alive.sum()),
            "valid_attacks": total_valid,
            "focused_attacks": float(focused.sum()),
        },
        "per_seed": per_seed,
    }


def normalisation(rows: Sequence[Sequence[float]]) -> tuple[list[float], list[float]]:
    """Contract S1: mean/std over exactly the 12 candidates' profiles; std + 1e-8."""
    array = np.asarray(rows, dtype=np.float64)
    mean = array.mean(axis=0)
    std = array.std(axis=0) + 1e-8
    return [float(x) for x in mean], [float(x) for x in std]


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Oracle-gate partner profile table (contract S1).")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR,
                        help=f"training run with the partner checkpoints (default: {DEFAULT_RUN_DIR})")
    parser.add_argument("--reference-checkpoint", default=None,
                        help="frozen reference checkpoint (update count / filename / path); "
                             "default = the largest update count in the run (u1250)")
    parser.add_argument("--stages", default=DEFAULT_STAGES,
                        help=f"comma-separated stage update counts (default: {DEFAULT_STAGES})")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS,
                        help=f"fixed measurement seeds, set A (default: {DEFAULT_SEEDS})")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help=f"episodes per seed (default: {DEFAULT_EPISODES})")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: results/oracle_gate/profiles.json)")
    parser.add_argument("--selfcheck-output", default=None,
                        help="self-check JSON path (default: results/oracle_gate/profiles_selfcheck.json)")
    parser.add_argument("--tiny", action="store_true",
                        help="minimal run-through: 1 seed x 1 episode, same table format")
    args = parser.parse_args(argv)
    if args.tiny:
        args.seeds = DEFAULT_SEEDS.split(",")[0]
        args.episodes = 1
    try:
        args.seed_list = [int(x) for x in str(args.seeds).replace(" ", "").split(",") if x]
    except ValueError:
        parser.error("--seeds must be comma-separated integers")
    if not args.seed_list:
        parser.error("--seeds must contain at least one seed")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoints = discover_checkpoints(run_dir)
    reference_update, reference_path = resolve_reference(args.reference_checkpoint, run_dir,
                                                         checkpoints)
    stages = select_stages(args.stages, checkpoints)
    stage_updates = [int(u) for u, _ in stages]

    # --- contract S3: measurement battles must not overlap training or the test set ------------
    train_seed = int(load_config(run_dir, checkpoints[-1])["seed"])
    overlap_train = sorted(set(args.seed_list) & {train_seed})
    overlap_test = sorted(set(args.seed_list) & set(TEST_SEEDS_C))
    if overlap_train or overlap_test:
        raise SystemExit(
            f"measurement seeds overlap the training seed ({overlap_train}) or test set C "
            f"({overlap_test}); contract S3 forbids this"
        )

    config = load_config(run_dir, checkpoints[-1])
    trainer = train.Trainer(config)
    measure, eval_keys = make_measure_fn(trainer, args.episodes, args.seed_list)

    print(f"[partner profiles] {DISCLAIMER}", flush=True)
    print(f"  run dir            : {run_dir}", flush=True)
    print(f"  reference team     : {reference_path.name} (update_count={reference_update}) for "
          f"every identity except the tested one", flush=True)
    print(f"  stages             : " + ", ".join(
        f"u{u} ({path.name})" for u, path in stages), flush=True)
    print(f"  measurement battles: seeds {args.seed_list} x {args.episodes} episodes = "
          f"{len(args.seed_list) * args.episodes} fixed battles, replayed for all "
          f"{len(PARTNER_IDENTITIES) * len(stages)} candidates "
          f"({len(PARTNER_IDENTITIES) * len(stages) * len(args.seed_list) * args.episodes} "
          f"episode EXECUTIONS, not independent samples)", flush=True)

    reference_params = load_checkpoint_params(reference_path)
    stage_params = {u: load_checkpoint_params(path) for u, path in stages}

    raw: dict[str, dict[str, list[float]]] = {}
    detail: dict[str, dict[str, dict]] = {}
    ledger: list[dict] = []
    per_candidate_seconds: list[dict] = []

    for identity in PARTNER_IDENTITIES:
        raw[str(identity)] = {}
        detail[str(identity)] = {}
        for update_count, path in stages:
            mixed = build_mixed_params(reference_params, stage_params[update_count], identity)
            t0 = time.perf_counter()
            team, per_agent = measure(mixed, eval_keys)
            team_np = np.asarray(jax.device_get(team), dtype=np.float64)
            per_agent_np = np.asarray(jax.device_get(per_agent), dtype=np.float64)
            elapsed = time.perf_counter() - t0

            block = candidate_metrics(team_np, per_agent_np, identity, args.seed_list)
            raw[str(identity)][str(update_count)] = block["metrics"]
            detail[str(identity)][str(update_count)] = {
                "checkpoint": path.name,
                "seconds": round(elapsed, 3),
                **block,
            }
            per_candidate_seconds.append({"identity": identity, "update_count": update_count,
                                          "seconds": round(elapsed, 3)})
            m = block["metrics"]
            print(f"  [identity {identity} u{update_count:>4}] damage_per_len={m[0]:.6f} "
                  f"alive_frac={m[1]:.6f} focus_share={m[2]:.6f} "
                  f"(valid={block['pooled_totals']['valid_attacks']:.0f}, "
                  f"focused={block['pooled_totals']['focused_attacks']:.0f}, "
                  f"len={block['pooled_totals']['total_length']:.0f}, {elapsed:.1f}s)", flush=True)

            # --- self-check: attributed damage vs the environment's own enemy health ledger ----
            attributed = float(per_agent_np[:, :, :, 0].sum())
            ledger_loss = float(team_np[:, :, 3].sum())
            gap = ledger_loss - attributed
            ledger.append({
                "identity": identity,
                "update_count": update_count,
                "enemy_health_loss": ledger_loss,
                "attributed_to_allies": attributed,
                "gap": gap,
                "relative_gap": (gap / ledger_loss) if ledger_loss > 0 else 0.0,
                "ok": gap >= -1e-3 and (ledger_loss <= 0 or gap / ledger_loss <= 0.02),
            })

    rows = [raw[str(i)][str(u)] for i in PARTNER_IDENTITIES for u in stage_updates]
    mean, std = normalisation(rows)
    z = {
        str(i): {str(u): [float((raw[str(i)][str(u)][d] - mean[d]) / std[d]) for d in range(3)]
                 for u in stage_updates}
        for i in PARTNER_IDENTITIES
    }

    total_battles = len(args.seed_list) * args.episodes
    note = (
        f"measurement battles = fixed seed set A = {args.seed_list} x {args.episodes} episodes "
        f"= {total_battles} battles (identical reset keys for all 12 candidates; A is disjoint "
        f"from the training seed {train_seed} and from test set C {TEST_SEEDS_C}). "
        "reference team = every identity other than the tested one keeps its u1250 parameters "
        f"(identity 0 included, from {reference_path.name}); the tested identity is replaced by "
        "its stage parameters. Consequence: the four u1250 candidates are the identical team "
        "configuration and therefore read off the same "
        f"{total_battles} battles (they differ only in which identity is scored). "
        "Each metric is a ratio of pooled totals over those battles: damage_per_len = total "
        "damage this identity dealt / total episode length; alive_frac = total alive steps / "
        "total episode length; focus_share = valid attacks whose target was also attacked by at "
        "least one other ally in the same step / all valid attacks (0 when there is none). "
        "A valid attack is an attack that damaged an enemy, using the same attack-index decoding "
        "and even-split attribution as Trainer._evaluate (L631-651) and test_preflight rule (g); "
        f"{len(PARTNER_IDENTITIES) * len(stages)} x {total_battles} = "
        f"{len(PARTNER_IDENTITIES) * len(stages) * total_battles} episode EXECUTIONS, not "
        "independent samples. No causal attribution; sd(seed) is neither a confidence interval "
        "nor a criterion."
    )

    payload = {
        "kind": "partner_profile_table",
        "source_run": DEFAULT_RUN_DIR,
        "stages": stage_updates,
        "partner_identities": PARTNER_IDENTITIES,
        "metric_names": METRIC_NAMES,
        "measurement": {
            "seeds": args.seed_list,
            "episodes_per_seed": args.episodes,
            "reference_team": (
                f"other identities fixed at their u1250 params ({reference_path.name}); tested "
                "identity replaced by its stage params"
            ),
            "note": note,
        },
        "raw": raw,
        "normalization": {"mean": mean, "std": std},
        "z": z,
        "z_layout": Z_LAYOUT,
    }

    out_path = Path(args.output) if args.output else (HERE / "results" / "oracle_gate" /
                                                      "profiles.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    selfcheck_path = (Path(args.selfcheck_output) if args.selfcheck_output
                      else out_path.with_name("profiles_selfcheck.json"))
    selfcheck_path.write_text(json.dumps({
        "kind": "partner_profile_selfcheck",
        "profiles": str(out_path),
        "attribution_ledger_rule": (
            "enemy health loss (all causes) vs damage attributed to the five allies with the "
            "even-split rule; gap = wall deaths / deaths without a shooter, must be small and "
            "non-negative (test_preflight rule (g))"
        ),
        "attribution_ledger": ledger,
        "attribution_ledger_all_ok": all(rec["ok"] for rec in ledger),
        "per_candidate": detail,
        "per_candidate_seconds": per_candidate_seconds,
    }, indent=2, allow_nan=False), encoding="utf-8")

    wall = time.perf_counter() - started
    print("\n  raw profiles (pooled over the fixed battles):", flush=True)
    header = f"  {'identity':>8} {'stage':>10} {'damage_per_len':>16} {'alive_frac':>12} {'focus_share':>12}"
    print(header, flush=True)
    for i in PARTNER_IDENTITIES:
        for u in stage_updates:
            v = raw[str(i)][str(u)]
            print(f"  {i:>8} {'u' + str(u):>10} {v[0]:>16.6f} {v[1]:>12.6f} {v[2]:>12.6f}",
                  flush=True)

    print("\n  normalisation (frozen, over the 12 candidates):", flush=True)
    for name, value in zip(METRIC_NAMES, mean):
        print(f"    mean {name:<16} = {value:.6f}", flush=True)
    for name, value in zip(METRIC_NAMES, std):
        print(f"    std  {name:<16} = {value:.6f}", flush=True)
    print("\n  z = (profile - mean)/std:", flush=True)
    for i in PARTNER_IDENTITIES:
        for u in stage_updates:
            v = z[str(i)][str(u)]
            print(f"    identity {i} u{u:<5} [{v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}]", flush=True)

    print("\n  self-check: damage attribution vs the environment's enemy health ledger:", flush=True)
    for rec in ledger:
        print(f"    identity {rec['identity']} u{rec['update_count']:<5} "
              f"enemy_health_loss={rec['enemy_health_loss']:.2f} "
              f"attributed_to_allies={rec['attributed_to_allies']:.2f} "
              f"gap={rec['gap']:.2f} ({100 * rec['relative_gap']:.2f}%) ok={rec['ok']}",
              flush=True)
    print(f"    all {len(ledger)} ledger checks ok = "
          f"{all(rec['ok'] for rec in ledger)}", flush=True)

    labels = [(i, u) for i in PARTNER_IDENTITIES for u in stage_updates]
    z_rows = [z[str(i)][str(u)] for i, u in labels]
    print("\n  similarity between candidates (frozen z space, max|dz| over the 3 dims):", flush=True)
    for idx, label in enumerate(labels):
        others = [(float(np.max(np.abs(np.asarray(z_rows[idx]) - np.asarray(z_rows[k])))),
                   labels[k]) for k in range(len(labels)) if k != idx]
        distance, nearest = min(others)
        print(f"    identity {label[0]} u{label[1]:<5} nearest = identity {nearest[0]} "
              f"u{nearest[1]}  max|dz|={distance:.4f}", flush=True)
    close = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            distance = float(np.max(np.abs(np.asarray(z_rows[a]) - np.asarray(z_rows[b]))))
            if distance < 0.25:
                close.append((distance, labels[a], labels[b]))
    print("    candidate pairs closer than 0.25 in z space (hard to tell apart):", flush=True)
    for distance, la, lb in sorted(close):
        print(f"      identity {la[0]} u{la[1]} vs identity {lb[0]} u{lb[1]}: "
              f"max|dz|={distance:.4f}", flush=True)
    if not close:
        print("      (none)", flush=True)

    print(f"\n  wrote {out_path}", flush=True)
    print(f"  wrote {selfcheck_path}", flush=True)
    print(f"  per-candidate seconds: " + ", ".join(
        f"i{r['identity']}u{r['update_count']}={r['seconds']:.1f}" for r in per_candidate_seconds),
        flush=True)
    print(f"  wall {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
