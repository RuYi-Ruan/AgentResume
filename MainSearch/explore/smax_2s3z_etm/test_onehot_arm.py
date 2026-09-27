"""Measured short checks for the `onehot` arm (ticket 2026-09-27; `--tiny`, seconds).

`--arm onehot` feeds the observer the battle's partner-stage combination as a per-identity
stage-index one-hot (identity order `[1,2,3,4]`, 3 dims each, stage order `[u50,u600,u1250]`).  It
is an **oracle-level stage label** - which of the 12 candidate parameter sets each partner plays -
not a measured behaviour profile (that is the `profile` arm) and not a constant (that is
`placeholder`).  Its purpose is the primary comparison `onehot - placeholder`.

Checks, each measured on the real code (this file only checks; the trainer/evaluator own the
behaviour):

  a. one-hot blocks    - all 81 combinations: every 3-dim identity block holds exactly one `1`, its
                         position is that partner's stage index, and the whole 12-dim vector holds
                         exactly four 1s (324 blocks checked);
  b. cross-module      - the trainer's enumeration and z mapping, the evaluator's enumeration
                         (including the `--combos-subset` picks the gate run uses) and the
                         evaluator's swap path agree on all 81 combinations, against a
                         `itertools.product` recomputation; three combinations are printed with
                         their 12-dim z;
  c. other arms intact - `placeholder` is still a constant zero even when handed the measured table,
                         and `profile`'s z is still exactly the normalized `profiles.json` table,
                         whose sha256 still matches the one the existing gate report recorded;
  d. parity            - the three arms have the same input dimension (139), the same parameter
                         count and bitwise identical initial parameters for the same `--seed`;
  e. end-to-end        - `train_observer_gate.py --tiny --arm onehot` runs and writes a `run.json`
                         naming the arm, the oracle-level one-hot semantics, the 139-dim input and
                         no input profile table (its own temporary run directory).

Run:
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      test_onehot_arm.py
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evaluate_observer_gate as E  # noqa: E402
import train_observer_gate as G  # noqa: E402

EVAL_REPORT = HERE / "results" / "oracle_gate" / "observer_gate_eval_full.json"


class Reporter:
    def __init__(self) -> None:
        self.results = []
        print("=" * 100)
        print("SMAX 2s3z oracle-gate onehot-arm checks (`--tiny`)")
        print("=" * 100)

    def check(self, name: str, ok: bool, lines: list[str], seconds: float) -> None:
        self.results.append((name, bool(ok)))
        print(f"\n[check {name}] {'PASS' if ok else 'FAIL'}  ({seconds:.1f}s)")
        for line in lines:
            print(f"    {line}")

    def finish(self) -> int:
        print("\n" + "=" * 100)
        for name, ok in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  check {name}")
        failed = [name for name, ok in self.results if not ok]
        print(f"summary: {len(self.results) - len(failed)}/{len(self.results)} checks passed")
        print("=" * 100)
        return 1 if failed else 0


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def independent_combos() -> list[tuple[int, ...]]:
    """The 81 partner-stage combinations, recomputed from scratch (not via the module)."""
    return list(itertools.product(range(len(G.STAGE_UPDATES)), repeat=len(G.PARTNER_IDENTITIES)))


def independent_subset(combos: list[tuple[int, ...]], k: int) -> list[tuple[int, ...]]:
    """`G.enumerate_combos(k)`'s documented subset rule, recomputed here."""
    picked = np.unique(np.round(np.linspace(0, len(combos) - 1, k)).astype(int))
    return [combos[int(i)] for i in picked]


def stage_vector(slot: int, ep_index: int) -> list[int]:
    """The stage indices the four identities play in `(slot, ep_index)`, from the streams alone."""
    return [int(G.partner_stage_index(np.int32(slot), np.int32(ep_index), np.int32(identity)))
            for identity in G.PARTNER_IDENTITIES]


def z_via_device(trainer, combos) -> np.ndarray:
    """`trainer.z_vector` for every combination, in one batched call."""
    return np.asarray(jax.device_get(
        jax.vmap(trainer.z_vector)(jnp.asarray(np.asarray(combos, np.int32)))
    ))


# --------------------------------------------------------------------------------------------
# a. the one-hot blocks themselves
# --------------------------------------------------------------------------------------------
def check_a(rep: Reporter) -> None:
    started = time.perf_counter()
    combos = independent_combos()
    table = G.onehot_z_table()
    bad_shape, bad_block, bad_position, bad_total, bad_table = [], [], [], [], []
    blocks = 0
    for combo in combos:
        z = np.asarray(G.onehot_z_vector(combo), dtype=np.float64)
        if z.shape != (G.PROFILE_DIM,):
            bad_shape.append((combo, z.shape))
            continue
        if not np.array_equal(z, np.asarray(G.z_vector_from_table(table, combo), np.float64)):
            bad_table.append(combo)
        if z.sum() != float(len(G.PARTNER_IDENTITIES)):
            bad_total.append((combo, float(z.sum())))
        for j in range(len(G.PARTNER_IDENTITIES)):
            block = z[3 * j:3 * j + 3]
            blocks += 1
            if not (block.sum() == 1.0 and np.all((block == 0.0) | (block == 1.0))):
                bad_block.append((combo, j, block.tolist()))
            elif int(np.argmax(block)) != int(combo[j]):
                bad_position.append((combo, j, block.tolist(), int(combo[j])))
    ok = not (bad_shape or bad_block or bad_position or bad_total or bad_table)
    lines = [
        f"{len(combos)} combinations x {len(G.PARTNER_IDENTITIES)} identities = {blocks} 3-dim "
        f"blocks checked; stage order {list(G.STAGE_LABELS)}, identity order "
        f"{list(G.PARTNER_IDENTITIES)}, PROFILE_DIM={G.PROFILE_DIM}",
        f"blocks with exactly one 1 and no other value: {blocks - len(bad_block)}/{blocks} "
        f"(bad: {bad_block[:3]})",
        f"hot index equals the partner's stage index: {blocks - len(bad_position)}/{blocks} "
        f"(bad: {bad_position[:3]})",
        f"whole-vector 1-count == 4: {len(combos) - len(bad_total)}/{len(combos)} "
        f"(bad: {bad_total[:3]}); z is the z_table row path: {len(combos) - len(bad_table)}/"
        f"{len(combos)} (bad: {bad_table[:3]})",
        f"examples: " + "; ".join(
            f"{G.combo_label(c)} -> {np.asarray(G.onehot_z_vector(c), dtype=int).tolist()}"
            for c in (combos[0], (1, 2, 0, 1), combos[-1])
        ),
    ]
    rep.check("a one-hot blocks over all 81 combinations", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# b. the trainer, the evaluator and an independent recomputation agree
# --------------------------------------------------------------------------------------------
def check_b(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    combos = independent_combos()
    trainer = ctx["trainers"]["onehot"]

    same_enum = list(G.enumerate_combos(None)) == combos
    subset_checks = {}
    for k in (1, 2, 3, 5, 9):
        subset_checks[k] = list(G.enumerate_combos(k)) == independent_subset(combos, k)
    n_distinct = len(set(G.enumerate_combos(None)))

    expected = np.stack([np.asarray(G.onehot_z_vector(c), np.float32) for c in combos])
    trainer_z = z_via_device(trainer, combos)
    # The evaluator's swap path: `swap_analysis` reads `trainer.z_table` through this helper.
    eval_path = np.stack([np.asarray(G.z_vector_from_table(np.asarray(trainer.z_table), c),
                                     np.float32) for c in combos])
    # The training path: `z_batch` builds z from the partner streams for the running episode.
    batch_pairs = [(0, 0), (1, 1), (3, 7), (2, 13), (0, 40)]
    slots = jnp.asarray(np.asarray([p[0] for p in batch_pairs], np.int32))
    eps = jnp.asarray(np.asarray([p[1] for p in batch_pairs], np.int32))
    batch_z = np.asarray(jax.device_get(trainer.z_batch(slots, eps)))
    batch_expected = np.stack([np.asarray(G.onehot_z_vector(stage_vector(s, e)), np.float32)
                               for s, e in batch_pairs])
    batch_stages = {f"slot {s}, episode {e}": stage_vector(s, e) for s, e in batch_pairs[:3]}

    # The evaluator's §5 swap on the one-hot channel: the inspected partner really plays its LATE
    # stage while its own 3 dims are replaced by the EARLY stage's one-hot.
    matched_combo = [int(s) for s in E.default_matched_combo(combos, G.PARTNER_IDENTITIES[0])]
    identity, param_stage, profile_stage = E.resolve_swap_target(
        matched_combo, G.PARTNER_IDENTITIES[0], 0
    )
    j = G.PARTNER_IDENTITIES.index(int(identity))
    stale_combo = list(matched_combo)
    stale_combo[j] = int(profile_stage)
    table_np = np.asarray(trainer.z_table)
    z_correct = np.asarray(G.z_vector_from_table(table_np, matched_combo), np.float64)
    z_stale = np.asarray(G.z_vector_from_table(table_np, stale_combo), np.float64)
    correct_hot = int(np.argmax(z_correct[3 * j:3 * j + 3]))
    stale_hot = int(np.argmax(z_stale[3 * j:3 * j + 3]))
    dz = float(np.max(np.abs(z_stale - z_correct)))
    identical_rest = bool(np.array_equal(np.delete(z_stale, [3 * j, 3 * j + 1, 3 * j + 2]),
                                         np.delete(z_correct, [3 * j, 3 * j + 1, 3 * j + 2])))
    swap_ok = (int(param_stage) > int(profile_stage) and correct_hot == int(param_stage)
               and stale_hot == int(profile_stage) and abs(dz - 1.0) < 1e-9 and identical_rest)

    ok = (same_enum and all(subset_checks.values()) and n_distinct == len(combos)
          and np.array_equal(trainer_z, expected) and np.array_equal(eval_path, expected)
          and np.array_equal(batch_z, batch_expected) and swap_ok)
    lines = [
        f"independent itertools.product order == G.enumerate_combos(None): {same_enum} "
        f"({len(combos)} combinations, {n_distinct} distinct); enumeration used by the evaluator: "
        f"same function",
        f"--combos-subset picks identical for k in {sorted(subset_checks)} "
        f"({all(subset_checks.values())})",
        f"trainer.z_vector (batched, device) == one-hot for all {len(combos)} combinations: "
        f"{np.array_equal(trainer_z, expected)} (max |diff| "
        f"{float(np.max(np.abs(trainer_z - expected))):.1e})",
        f"evaluator swap path (z_vector_from_table on the trainer's z_table) == one-hot: "
        f"{np.array_equal(eval_path, expected)}",
        f"training path z_batch == one-hot of the streams' stage indices for {batch_pairs}: "
        f"{np.array_equal(batch_z, batch_expected)}",
        "stage indices from the streams: "
        + "; ".join(f"{k} -> {v} -> {G.combo_label(v)}" for k, v in batch_stages.items()),
        f"§5 stale swap on the one-hot channel: {G.combo_label(matched_combo)} -> identity "
        f"{int(identity)} really plays {G.STAGE_LABELS[param_stage]} while its dims "
        f"[{3 * j}, {3 * j + 3}) become that identity's {G.STAGE_LABELS[profile_stage]} one-hot "
        f"(hot {correct_hot} -> {stale_hot}, max |dz|={dz:.1f}, other identities untouched: "
        f"{identical_rest}): {swap_ok}",
        "examples (combo -> 12-dim z): " + "; ".join(
            f"{G.combo_label(c)} -> {np.asarray(G.onehot_z_vector(c), dtype=int).tolist()}"
            for c in (combos[0], (1, 2, 0, 1), combos[-1])
        ),
    ]
    rep.check("b trainer/evaluator agree on combination order and z (all 81)",
              ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# c. the other two arms are untouched
# --------------------------------------------------------------------------------------------
def check_c(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    combos = independent_combos()
    table = ctx["table"]
    mean = np.asarray(table["normalization"]["mean"], dtype=np.float64)
    std = np.asarray(table["normalization"]["std"], dtype=np.float64)
    expected_profile = np.zeros((len(G.PARTNER_IDENTITIES), len(G.STAGE_UPDATES), 3), np.float32)
    for j, identity in enumerate(G.PARTNER_IDENTITIES):
        for s, stage in enumerate(G.STAGE_UPDATES):
            raw = np.asarray(table["raw"][str(identity)][str(stage)], dtype=np.float64)
            expected_profile[j, s] = ((raw - mean) / std).astype(np.float32)

    placeholder = ctx["trainers"]["placeholder"]
    profile = ctx["trainers"]["profile"]
    ph_zero_table = bool(np.all(np.asarray(placeholder.z_table) == 0.0))
    ph_z = z_via_device(placeholder, combos[:3])
    ph_zero_z = bool(np.all(ph_z == 0.0))
    ph_batch = np.asarray(jax.device_get(
        placeholder.z_batch(jnp.asarray(np.asarray([0, 1, 2], np.int32)),
                            jnp.asarray(np.asarray([0, 5, 11], np.int32)))
    ))
    ph_zero_batch = bool(np.all(ph_batch == 0.0))

    profile_table_ok = np.array_equal(np.asarray(profile.z_table, np.float32), expected_profile)
    profile_z = z_via_device(profile, combos[:3])
    profile_z_ok = np.array_equal(
        profile_z,
        np.stack([np.asarray(G.z_vector_from_table(expected_profile, c), np.float32)
                  for c in combos[:3]]),
    )
    sha_now = G.sha256_file(ctx["table_source"])
    sha_recorded = None
    if EVAL_REPORT.is_file():
        payload = json.loads(EVAL_REPORT.read_text(encoding="utf-8"))
        sha_recorded = payload.get("profile_table", {}).get("sha256")
    sha_unchanged = sha_recorded is None or sha_recorded == sha_now

    onehot_z = z_via_device(ctx["trainers"]["onehot"], combos)
    expected_onehot = np.stack([np.asarray(G.onehot_z_vector(c), np.float32) for c in combos])
    # The onehot trainer was handed the measured table and must have replaced it with the one-hot
    # table; its z must be the one-hot, not the profile's z.
    onehot_handed_table = np.array_equal(np.asarray(ctx["z_table"], np.float32), expected_profile)
    onehot_dropped_table = np.array_equal(
        np.asarray(ctx["trainers"]["onehot"].z_table, np.float32),
        np.asarray(G.onehot_z_table(), np.float32),
    )
    onehot_not_profile = not np.array_equal(onehot_z, z_via_device(profile, combos))
    ok = (ph_zero_table and ph_zero_z and ph_zero_batch and profile_table_ok and profile_z_ok
          and sha_unchanged and onehot_handed_table and onehot_dropped_table and onehot_not_profile
          and np.array_equal(onehot_z, expected_onehot))
    lines = [
        f"placeholder z_table all-zero: {ph_zero_table}; z_vector all-zero for "
        f"{len(combos[:3])} combinations: {ph_zero_z}; z_batch (training path) all-zero: "
        f"{ph_zero_batch} - the measured table was handed to the placeholder trainer and was "
        f"ignored (by definition)",
        f"profile z_table == independently normalized profiles.json (raw-mean)/std: "
        f"{profile_table_ok} (shape {np.asarray(profile.z_table).shape})",
        f"profile z_vector == the table's z for {len(combos[:3])} combinations: {profile_z_ok} "
        f"({G.combo_label(combos[0])} -> {profile_z[0].round(4).tolist()})",
        f"profiles.json sha256 {sha_now[:16]}... unchanged vs the recorded gate report "
        f"({sha_recorded[:16] + '...' if sha_recorded else 'no report to compare'}): {sha_unchanged}",
        f"onehot: the measured table really was handed in ({onehot_handed_table}), its z_table is "
        f"the one-hot table instead ({onehot_dropped_table}), and its z is the one-hot for all "
        f"{len(combos)} combinations ({np.array_equal(onehot_z, expected_onehot)}) while differing "
        f"from the profile arm's z ({onehot_not_profile})",
    ]
    rep.check("c placeholder stays zero and profile still equals profiles.json", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# d. the three arms are parameter-count and initialisation twins
# --------------------------------------------------------------------------------------------
def check_d(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    trainers, runners = ctx["trainers"], ctx["runners"]
    dims = {arm: int(trainers[arm].config["observer_input_dim"]) for arm in G.ARMS}
    profile_dims = {arm: int(trainers[arm].config["profile_dim"]) for arm in G.ARMS}
    counts = {arm: int(sum(x.size for x in jax.tree.leaves(runners[arm].params[0])))
              for arm in G.ARMS}
    same_init = {arm: G.tree_bitwise_equal(runners["onehot"].params[0], runners[arm].params[0])
                 for arm in ("placeholder", "profile")}
    expected_dim = int(ctx["config"]["obs_dim"]) + G.PROFILE_DIM
    ok = (all(d == expected_dim for d in dims.values()) and expected_dim == 139
          and all(p == G.PROFILE_DIM for p in profile_dims.values())
          and len(set(counts.values())) == 1 and all(same_init.values()))
    lines = [
        f"--tiny observer_input_dim per arm: {dims} (obs {ctx['config']['obs_dim']} + z "
        f"{G.PROFILE_DIM} = {expected_dim}); profile_dim per arm: {profile_dims}",
        f"observer parameter count per arm: {counts} (all equal: {len(set(counts.values())) == 1})",
        f"initial parameters for --seed {ctx['seed']} bitwise identical to the onehot arm: "
        f"{same_init}",
    ]
    rep.check("d same input dim, parameter count and initial parameters", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# e. an end-to-end tiny onehot run
# --------------------------------------------------------------------------------------------
def check_e(rep: Reporter) -> None:
    started = time.perf_counter()
    # `--save-dir` must live under a `results/` directory, so the throwaway run goes into a
    # temporary tree that satisfies that guard and is deleted again on success.
    tmp_root = Path(tempfile.mkdtemp(prefix="onehot_arm_check_"))
    out_dir = tmp_root / "results" / "oracle_gate_onehot_check"
    cmd = [sys.executable, str(HERE / "train_observer_gate.py"), "--tiny", "--arm", "onehot",
           "--save-dir", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    run_json = out_dir / "run.json"
    payload = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else {}
    checkpoint = out_dir / "checkpoint_00000004.bin"
    tail = (proc.stdout or "").strip().splitlines()
    ok = (proc.returncode == 0 and payload.get("arm") == "onehot"
          and payload.get("z_semantics") == G.Z_SEMANTICS_BY_ARM["onehot"]
          and "onehot" in str(payload.get("z_channel"))
          and payload.get("profile_table") is None
          and int(payload.get("config", {}).get("observer_input_dim", -1)) == 139
          and checkpoint.is_file()
          and "onehot" in str(payload.get("profile_table_note")))
    lines = [
        f"command: {' '.join(cmd)}",
        f"exit code {proc.returncode}; run.json arm={payload.get('arm')!r}, "
        f"z_channel={payload.get('z_channel')!r}, observer_input_dim="
        f"{payload.get('config', {}).get('observer_input_dim')}, profile_table="
        f"{payload.get('profile_table')!r}, profile_table_note={payload.get('profile_table_note')!r}",
        f"checkpoint written: {checkpoint.name} exists={checkpoint.is_file()}; "
        f"z_semantics is the oracle-level one-hot text: "
        f"{payload.get('z_semantics') == G.Z_SEMANTICS_BY_ARM['onehot']}",
        f"stdout tail: {tail[-2:]}",
    ]
    if not ok:
        lines.append(f"full stdout: {tail}")
        lines.append(f"stderr tail: {(proc.stderr or '').strip().splitlines()[-6:]}")
        lines.append(f"run directory KEPT for inspection: {out_dir}")
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
        lines.append(f"temporary run directory removed again ({tmp_root})")
    rep.check("e tiny --arm onehot runs end to end", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
def main() -> int:
    args = G.parse_args(["--tiny", "--arm", "profile"])
    config = G.base_train.build_config(args)
    partner_dir = Path(args.partner_run_dir)
    partner_dir = partner_dir if partner_dir.is_absolute() else (HERE / partner_dir)
    partner_stack = G.load_partner_stack(partner_dir, args.partner_stages)
    table, table_source, is_fixture = G.load_profile_table(args.profiles)
    z_table = G.z_table_from_profile_table(table)

    # One trainer and one initial state per arm: the trainer derives its own channel from the arm
    # name, so the placeholder/onehot arms are handed the measured table on purpose to prove they
    # never read it.
    trainers = {arm: G.ObserverGateTrainer(config, partner_stack, z_table, arm=arm) for arm in G.ARMS}
    runners = {arm: trainers[arm].init_runner(jax.random.PRNGKey(args.seed)) for arm in G.ARMS}
    ctx = {"config": config, "trainers": trainers, "runners": runners, "seed": args.seed,
           "partner_stack": partner_stack, "table": table, "table_source": table_source,
           "is_fixture": is_fixture, "z_table": z_table}

    rep = Reporter()
    print(f"tiny config: num_envs={config['num_envs']} rollout_length={config['rollout_length']} "
          f"updates={config['total_updates']} segment_updates={config['segment_updates']}; "
          f"observer_input_dim={trainers['onehot'].config['observer_input_dim']}; "
          f"partners={list(G.PARTNER_IDENTITIES)} partner_stages={list(args.partner_stages)}; "
          f"profile table {table_source} (fixture={is_fixture}); eval arms {list(E.EVAL_ARMS)}, "
          f"primary pair {E.PRIMARY_PAIR}")
    check_a(rep)
    check_b(rep, ctx)
    check_c(rep, ctx)
    check_d(rep, ctx)
    check_e(rep)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
