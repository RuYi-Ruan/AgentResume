"""Diagnostic evaluation with a FROZEN reference team (fixed-reference identity profiling).

What this measures
------------------
The training-time evaluation in `train_smax_2s3z_independent.py` puts all five current
checkpoints' parameters on the floor at once, so a change in identity `i`'s metrics can come
from its own progress *or* from its four teammates getting stronger.  This script closes that
gap with a **frozen reference team**:

    identity i      <- the parameters of a chosen stage checkpoint (early / middle / late)
    identity j != i <- the parameters of ONE frozen reference checkpoint (default: the final,
                       i.e. largest-update_count, checkpoint of the same run)

Every (identity `i`, stage) configuration therefore differs from the other configurations of the
same identity only in identity `i`'s parameters, so the measured change of `i`'s own profile is
attributable to `i` alone.

This is a DIAGNOSTIC evaluation.  It isolates an identity's own progress for inspection; it is
NOT the P2 / continuous-learning main experiment and does not replace it (the reference team is
frozen, so the joint dynamics under simultaneous learning are precisely what is *not* tested).

Statistics (read this before quoting any number)
------------------------------------------------
The evaluation uses a FIXED set of `S` seeds x `E` episodes = `S*E` battles.  The *same* battles
are replayed at every stage (the reset keys are identical at every stage), so a run with `C`
configurations performs `C * S * E` **episode executions**, NOT `C * S * E` independent samples.
Quoting the execution count as a sample size would badly overstate the evidence.

  * `metrics` / `per_seed` / `across_seed_std` (a.k.a. "sd(seed)") describe how the per-seed
    group means differ between the `S` seed groups.  This is a spread across seed groups only:
    it is NOT a confidence interval, and it is NOT a P2 pass/fail criterion.
  * The P2-style quantity computed here is the **same-battle stage difference**:
    `late - early` paired on the SAME seed AND the SAME episode index (the same battle, replayed
    with only identity `i` swapped).  `paired_diff.per_pair` holds every pair; its cross-seed
    consistency is reported as the mean (and std) of the per-seed paired means.  A stage effect
    that is consistent in sign across every seed is much stronger evidence than a small
    difference that moves around between seed groups.

Output: `results/<run>/reference_team_eval.json` (written) plus a human-readable table (stdout).

Implementation note (parameters)
--------------------------------
The five identities' parameters are read straight from the checkpoint's flax-serialized `params`
field (`load_checkpoint_params`).  This is bitwise-identical to `train.load_checkpoint` (same tree
structure, same leaves, same dtype, same evaluator output - verified), but it avoids
`Trainer.init_runner`, whose parameter-initialisation compile
(`nn.initializers.orthogonal` -> `jnp.diag`) intermittently kills the interpreter on this
Windows/JAX build (Windows fatal exception `0x80000001` / `0xC0000005` inside
`jax._src.interpreters.mlir`).  Evaluation only reads parameters, so the rest of the runner state
is not needed.

Usage (absolute interpreter path required on this machine)::

  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      evaluate_reference_team.py --run-dir results/preflight_cli_a \
      --eval-seeds 1234,1235 --eval-episodes 2
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

import train_smax_2s3z_independent as train

HERE = Path(__file__).resolve().parent
CHECKPOINT_RE = re.compile(r"^checkpoint_(\d+)\.bin$")

# Identity metrics reported per stage (the five required ones + alive_fraction as context).
METRIC_KEYS = (
    "damage_dealt",
    "damage_taken",
    "kills",
    "alive_steps",
    "alive_fraction",
    "nearest_enemy_dist",
)
TEAM_KEYS = ("won", "return", "length")

DISCLAIMER = (
    "DIAGNOSTIC EVALUATION ONLY.  This fixes four teammates to a frozen reference checkpoint while "
    "swapping a single identity across stages, so it isolates that identity's own profile change. "
    "It is NOT the P2 / continuous-learning measurement and does not replace the main experiment: "
    "simultaneous learning, teammate co-adaptation and non-stationarity are exactly what a frozen "
    "reference team removes."
)

STATISTICS_NOTE = (
    "The same S seeds x E episodes (the same S*E battles, identical reset keys) are replayed at "
    "every stage, so a C-configuration run performs C*S*E EPISODE EXECUTIONS, not C*S*E "
    "independent samples; do not quote the execution count as a sample size.  'across_seed_std' "
    "(a.k.a. sd(seed)) only describes the spread between the S seed-group means: it is NOT a "
    "confidence interval and NOT a P2 pass criterion.  The P2-style quantity is the same-battle "
    "stage difference (late - early paired on the same seed AND the same episode index) in "
    "paired_diff, together with its cross-seed consistency (per-seed paired means and their "
    "spread)."
)


# --------------------------------------------------------------------------------------------
# Run / checkpoint discovery
# --------------------------------------------------------------------------------------------
def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else (HERE / path)
    path = path.resolve()
    if not path.is_dir():
        raise SystemExit(f"--run-dir is not a directory: {path}")
    return path


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """All `checkpoint_XXXXXXXX.bin` in `run_dir`, sorted by the filename's update count."""
    found = []
    for entry in run_dir.iterdir():
        match = CHECKPOINT_RE.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    found.sort(key=lambda item: item[0])
    if not found:
        raise SystemExit(f"no checkpoint_*.bin found in {run_dir}")
    return found


def resolve_reference(value: str | None, run_dir: Path,
                      checkpoints: list[tuple[int, Path]]) -> tuple[int, Path]:
    """`--reference-checkpoint` as an update count, a filename, or a path; default = latest."""
    if value is None:
        return checkpoints[-1]
    if re.fullmatch(r"\d+", str(value)):
        wanted = int(value)
        for update_count, path in checkpoints:
            if update_count == wanted:
                return update_count, path
        raise SystemExit(
            f"--reference-checkpoint {wanted}: no such update count in {run_dir} "
            f"(available: {[u for u, _ in checkpoints]})"
        )
    path = Path(value)
    path = path if path.is_absolute() else (run_dir / path)
    if not path.is_file():
        raise SystemExit(f"--reference-checkpoint not found: {path}")
    match = CHECKPOINT_RE.match(path.name)
    return (int(match.group(1)) if match else -1), path.resolve()


def select_stages(spec: str | None, checkpoints: list[tuple[int, Path]]) -> list[tuple[int, Path]]:
    """Stages to sweep: explicit update counts ("0,600,1250") or earliest/middle/latest."""
    by_update = dict(checkpoints)
    if spec:
        try:
            wanted = [int(x) for x in str(spec).replace(" ", "").split(",") if x]
        except ValueError:
            raise SystemExit("--stages must be comma-separated update counts, e.g. 0,600,1250")
        if not wanted:
            raise SystemExit("--stages was given but empty")
        if len(set(wanted)) != len(wanted):
            raise SystemExit(f"--stages has duplicates: {wanted}")
        missing = [u for u in wanted if u not in by_update]
        if missing:
            raise SystemExit(
                f"--stages {missing}: no checkpoint with that update count in this run "
                f"(available: {sorted(by_update)})"
            )
        return [(u, by_update[u]) for u in wanted]
    if len(checkpoints) == 1:
        indices = [0]
    elif len(checkpoints) == 2:
        indices = [0, 1]
    else:
        indices = sorted({0, len(checkpoints) // 2, len(checkpoints) - 1})
    return [checkpoints[k] for k in indices]


def label_stages(num_stages: int) -> list[str]:
    if num_stages == 1:
        return ["only"]
    if num_stages == 2:
        return ["early", "late"]
    if num_stages == 3:
        return ["early", "middle", "late"]
    return ["early"] + [f"stage{k}" for k in range(1, num_stages - 1)] + ["late"]


# --------------------------------------------------------------------------------------------
# Configuration and parameters
# --------------------------------------------------------------------------------------------
def load_config(run_dir: Path, checkpoint: Path) -> dict:
    """Reuse the exact training config: run.json if present, else the checkpoint's own metadata."""
    candidates = []
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            candidates.append(json.loads(run_json.read_text(encoding="utf-8")).get("config"))
        except (json.JSONDecodeError, OSError):
            pass
    try:
        candidates.append(train.read_checkpoint_meta(checkpoint).get("config"))
    except Exception:  # noqa: BLE001 - metadata is optional; fall through to the CLI defaults
        pass
    for config in candidates:
        if isinstance(config, dict) and {"map_name", "gru_hidden_dim", "fc_dim_size"} <= set(config):
            return config
    return train.build_config(train.parse_args([]))


def tree_max_abs_diff(a, b) -> float:
    """Leaf-wise max |a - b| over two pytrees with the same structure."""
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(leaves_a) != len(leaves_b):
        raise ValueError("pytree structure mismatch")
    if not leaves_a:
        return 0.0
    return max(
        float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
        for x, y in zip(leaves_a, leaves_b)
    )


def build_mixed_params(reference_params, stage_params, identity: int):
    """Reference teammates everywhere; identity `identity` takes its stage parameters."""
    return tuple(
        stage_params[j] if j == identity else reference_params[j]
        for j in range(len(reference_params))
    )


def load_checkpoint_params(path: Path):
    """The five identities' parameter pytrees, read straight from the checkpoint's `params` field.

    This is bitwise-identical to `train.load_checkpoint` (verified: same tree structure, same
    leaves, same dtype, same evaluator output), but it skips `Trainer.init_runner` - whose
    parameter-initialisation compile (`nn.initializers.orthogonal` -> `jnp.diag`) intermittently
    kills the interpreter on this Windows/JAX build (Windows fatal exception 0x80000001 /
    0xC0000005 inside `jax._src.interpreters.mlir`).  Evaluation only ever reads parameters, so
    the surrounding runner state (optimizer, hidden state, env state) is not needed here.
    """
    payload = serialization.msgpack_restore(path.read_bytes())["params"]
    return tuple(
        jax.tree.map(jnp.asarray, payload[key]) for key in sorted(payload, key=int)
    )


def params_only(params) -> Any:
    """Stand-in for `RunnerState`: `Trainer.evaluate` reads only `runner.params` (its own docstring:
    'Fixed-seed evaluation of `runner.params` only')."""
    return SimpleNamespace(params=params)


def jsonable(value):
    """JSON-safe scalar: non-finite floats (e.g. an unreachable-nearest-enemy NaN) -> null."""
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


# --------------------------------------------------------------------------------------------
# Metric extraction
# --------------------------------------------------------------------------------------------
def episode_metric_arrays(team, per_agent, identity: int) -> dict:
    """Per-(seed, episode) identity metrics from the trainer's raw evaluator output.

    ``team``      : (S, E, 3) -> [won, episode_return, episode_length]
    ``per_agent`` : (S, E, A, 6) -> [dealt, taken, kills, alive_steps, dist_sum, dist_cnt]
    Damage/kills attribution and the nearest-enemy pooling are the trainer's own definitions
    (`Trainer._evaluate`); only the aggregation granularity differs (per episode, not per seed).
    """
    lengths = team[:, :, 2]
    dealt = per_agent[:, :, identity, 0]
    taken = per_agent[:, :, identity, 1]
    kills = per_agent[:, :, identity, 2]
    alive = per_agent[:, :, identity, 3]
    dist_sum = per_agent[:, :, identity, 4]
    dist_cnt = per_agent[:, :, identity, 5]
    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(dist_cnt > 0, dist_sum / np.maximum(dist_cnt, 1e-12), np.nan)
    return {
        "damage_dealt": dealt,
        "damage_taken": taken,
        "kills": kills,
        "alive_steps": alive,
        "alive_fraction": alive / np.maximum(lengths, 1e-12),
        "nearest_enemy_dist": dist,
    }


def identity_stage_block(result: dict, raw: tuple, identity: int, seeds: list[int]) -> dict:
    """Identity `identity`'s own metrics for one evaluated (stage, identity) configuration.

    `result` is `Trainer.evaluate`'s structured output (stage-level metrics + per-seed means, the
    definitions the training loop itself uses); `raw` is the raw `_evaluate_impl` output used for
    the per-episode values that make same-battle pairing possible.
    """
    per_seed_raw = result["per_seed"]
    metrics = result["identities"][identity]
    per_seed = [
        {
            "seed": ps["seed"],
            **{key: jsonable(ps["identities"][identity][key]) for key in METRIC_KEYS},
        }
        for ps in per_seed_raw
    ]
    across_seed_std = {
        key: jsonable(np.std([row[key] for row in per_seed if row[key] is not None]))
        for key in METRIC_KEYS
    }

    team, per_agent = (np.asarray(jax.device_get(x), dtype=np.float64) for x in raw)
    arrays = episode_metric_arrays(team, per_agent, identity)
    num_seeds, num_episodes = team.shape[0], team.shape[1]
    per_episode = [
        {
            "seed": seeds[s],
            "episode": e,
            **{key: jsonable(arrays[key][s, e]) for key in METRIC_KEYS},
        }
        for s in range(num_seeds)
        for e in range(num_episodes)
    ]
    team_per_episode = [
        {
            "seed": seeds[s],
            "episode": e,
            "won": int(round(float(team[s, e, 0]))),
            "return": float(team[s, e, 1]),
            "length": float(team[s, e, 2]),
        }
        for s in range(num_seeds)
        for e in range(num_episodes)
    ]
    team_per_seed = [
        {
            "seed": ps["seed"],
            "episodes": ps["episodes"],
            "wins": ps["wins"],
            "win_rate": jsonable(ps["win_rate"]),
            "mean_return": jsonable(ps["mean_return"]),
            "mean_length": jsonable(ps["mean_length"]),
        }
        for ps in per_seed_raw
    ]
    return {
        "metrics": {key: jsonable(metrics[key]) for key in METRIC_KEYS},
        "across_seed_std": across_seed_std,
        "per_seed": per_seed,
        "per_episode": per_episode,
        "team": {
            "episodes": int(result["episodes"]),
            "episode_executions": int(result["episodes"]),
            "wins": int(result["wins"]),
            "win_rate": jsonable(result["win_rate"]),
            "mean_return": jsonable(result["mean_return"]),
            "mean_length": jsonable(result["mean_length"]),
        },
        "team_per_seed": team_per_seed,
        "team_per_episode": team_per_episode,
    }


def _floats(values):
    """Rows of a metric as float64; JSON `null` (a non-finite value) becomes NaN, i.e. unpaired."""
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _paired(values_from, values_to):
    """Elementwise (to - from) over finite pairs; returns (diffs, mean, std, pairs_used)."""
    a, b = _floats(values_from), _floats(values_to)
    mask = np.isfinite(a) & np.isfinite(b)
    diffs = (b - a)[mask]
    count = int(diffs.size)
    if count == 0:
        return [], None, None, 0
    return [float(x) for x in diffs], float(diffs.mean()), float(diffs.std()), count


def paired_difference(from_block: dict, to_block: dict) -> dict:
    """Same-battle (same seed AND same episode index) `to - from`; never a pooled re-sampling."""
    from_pairs = {(row["seed"], row["episode"]): row for row in from_block["per_episode"]}
    to_rows = to_block["per_episode"]
    order = [(row["seed"], row["episode"]) for row in to_rows]
    if set(from_pairs) != set(order):
        raise ValueError("paired stages must replay the same battles (same seeds and episodes)")
    seeds = sorted({seed for seed, _ in order})

    def _summarise(values_from, values_to):
        diffs, mean, std, count = _paired(values_from, values_to)
        per_seed_mean = []
        for seed in seeds:
            selected = [k for k, pair in enumerate(order) if pair[0] == seed]
            _, seed_mean, _, _ = _paired([values_from[k] for k in selected],
                                         [values_to[k] for k in selected])
            per_seed_mean.append({"seed": seed, "mean": jsonable(seed_mean)})
        finite = [row["mean"] for row in per_seed_mean if row["mean"] is not None]
        positive = sum(1 for m in finite if m > 0)
        return {
            "per_pair": diffs,
            "pairs_used": count,
            "mean": jsonable(mean),
            "std_within_pairs": jsonable(std),
            "per_seed_mean": per_seed_mean,
            "across_seed_mean": jsonable(np.mean(finite) if finite else None),
            "across_seed_std": jsonable(np.std(finite) if finite else None),
            # cross-seed consistency: how many seed-group means agree with the overall sign
            "seeds_positive": positive,
            "seeds_negative": sum(1 for m in finite if m < 0),
            "seeds_used": len(finite),
        }

    identity_metrics = {
        key: _summarise([from_pairs[p][key] for p in order], [row[key] for row in to_rows])
        for key in METRIC_KEYS
    }
    team_from = {(row["seed"], row["episode"]): row for row in from_block["team_per_episode"]}
    team_metrics = {
        key: _summarise([team_from[p][key] for p in order],
                        [row[key] for row in to_block["team_per_episode"]])
        for key in TEAM_KEYS
    }
    return {"metrics": identity_metrics, "team": team_metrics}


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------
def format_stage_table(stage_labels: list[str], entries: dict, agent_names: list[str]) -> str:
    header = (f"  {'identity':<16}{'stage':<8}{'dealt':>10}{'taken':>10}{'kills':>8}"
              f"{'alive_steps':>12}{'near_dist':>11}")
    rows = [header]
    for i, name in enumerate(agent_names):
        for label in stage_labels:
            metrics = entries[(label, i)]["metrics"]
            rows.append(
                f"  {name:<16}{label:<8}"
                f"{_fmt(metrics['damage_dealt']):>10}{_fmt(metrics['damage_taken']):>10}"
                f"{_fmt(metrics['kills']):>8}{_fmt(metrics['alive_steps']):>12}"
                f"{_fmt(metrics['nearest_enemy_dist']):>11}"
            )
    return "\n".join(rows)


def _fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def format_paired_table(pairs: dict, agent_names: list[str], from_label: str,
                        to_label: str) -> str:
    metrics_order = ("damage_dealt", "damage_taken", "kills", "alive_steps",
                     "nearest_enemy_dist")
    header = (f"  {'identity':<16}{'d_dealt':>10}{'d_taken':>10}{'d_kills':>9}"
              f"{'d_alive':>10}{'d_dist':>10}")
    rows = [header]
    for i, name in enumerate(agent_names):
        metrics = pairs[i]["metrics"]
        rows.append(
            f"  {name:<16}"
            + "".join(f"{_fmt(metrics[key]['mean']):>10}" for key in metrics_order)
        )
    rows.append(f"  same-battle {to_label} - {from_label}, mean over all pairs "
                f"(pairs = seed x episode index)")
    rows.append("  cross-seed consistency (seed-group means: positives / negatives; zeros are "
                "counted in neither):")
    for i, name in enumerate(agent_names):
        metrics = pairs[i]["metrics"]
        cells = []
        for key in metrics_order:
            rec = metrics[key]
            if not rec["seeds_used"]:
                cells.append("-")
            else:
                cells.append(f"{rec['seeds_positive']}+/{rec['seeds_negative']}-")
        rows.append(f"  {name:<16}" + "".join(f"{cell:>10}" for cell in cells))
    return "\n".join(rows)


# --------------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-reference-team diagnostic evaluation (not the P2 main experiment)."
    )
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="training run directory (contains checkpoint_*.bin)")
    parser.add_argument("--run-dir", dest="run_dir_opt", default=None,
                        help="same as the positional argument")
    parser.add_argument("--reference-checkpoint", default=None,
                        help="frozen teammate checkpoint: an update count, a filename, or a path; "
                             "default = the checkpoint with the largest update_count")
    parser.add_argument("--stages", default=None,
                        help='comma-separated update counts, e.g. "0,600,1250"; default = the '
                             "earliest, middle and latest checkpoint of the run")
    parser.add_argument("--eval-seeds", default="1234,1235,1236,1237,1238",
                        help="comma-separated FIXED evaluation seeds; identical at every stage so "
                             "stage-to-stage differences are paired on the same battles")
    parser.add_argument("--eval-episodes", type=int, default=4,
                        help="episodes per fixed seed per configuration")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: <run-dir>/reference_team_eval.json)")
    args = parser.parse_args(argv)
    args.run_dir = args.run_dir_opt or args.run_dir
    if not args.run_dir:
        parser.error("a run directory is required (positional or --run-dir)")
    try:
        args.eval_seeds = [int(x) for x in str(args.eval_seeds).replace(" ", "").split(",") if x]
    except ValueError:
        parser.error("--eval-seeds must be comma-separated integers")
    if not args.eval_seeds:
        parser.error("--eval-seeds must contain at least one seed")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_dir = resolve_run_dir(args.run_dir)
    checkpoints = discover_checkpoints(run_dir)
    reference_update, reference_path = resolve_reference(args.reference_checkpoint, run_dir,
                                                         checkpoints)
    stages = select_stages(args.stages, checkpoints)
    labels = label_stages(len(stages))
    started = time.perf_counter()

    config = load_config(run_dir, checkpoints[-1])
    trainer = train.Trainer(config)
    trainer.make_evaluator(args.eval_episodes, args.eval_seeds)

    battles = len(args.eval_seeds) * args.eval_episodes
    configurations = len(stages) * trainer.num_agents
    print(f"[reference-team eval] {DISCLAIMER}", flush=True)
    print(f"  run dir            : {run_dir}", flush=True)
    print(f"  reference teammates: {reference_path.name} (update_count={reference_update})",
          flush=True)
    print(f"  stages             : " + ", ".join(
        f"{label}=update {u} ({path.name})" for label, (u, path) in zip(labels, stages)),
        flush=True)
    print(f"  evaluation         : {len(args.eval_seeds)} fixed seeds {args.eval_seeds} x "
          f"{args.eval_episodes} episodes = {battles} battles, replayed identically at every "
          f"stage; {configurations} configurations -> {configurations * battles} episode "
          f"executions (NOT {configurations * battles} independent samples)", flush=True)

    reference_params = load_checkpoint_params(reference_path)
    stage_params = [(label, update_count, path, load_checkpoint_params(path))
                    for label, (update_count, path) in zip(labels, stages)]

    num_agents = trainer.num_agents
    agent_names = [f"{trainer.env.agents[i]}/{config['unit_type_names'][i]}"
                   for i in range(num_agents)]

    entries: dict[tuple[str, int], dict] = {}
    isolation: dict[tuple[str, int], dict] = {}
    config_team: list[dict] = []
    config_seconds: list[dict] = []
    for label, update_count, path, params in stage_params:
        for i in range(num_agents):
            mixed_params = build_mixed_params(reference_params, params, i)
            # --- isolation proof: swapping identity i must leave the other four bitwise equal ---
            diffs = [tree_max_abs_diff(mixed_params[j], reference_params[j])
                     for j in range(num_agents)]
            isolation[(label, i)] = {
                "stage": label,
                "stage_update_count": update_count,
                "stage_checkpoint": path.name,
                "target_identity": i,
                "leaf_max_abs_diff_vs_reference": diffs,
                "target_vs_reference_max_abs_diff": diffs[i],
                "other_identities_max_abs_diff": (float(max(
                    d for j, d in enumerate(diffs) if j != i)) if num_agents > 1 else 0.0),
                "only_target_identity_changed": all(
                    d == 0.0 for j, d in enumerate(diffs) if j != i),
            }
            t0 = time.perf_counter()
            # `evaluate` gives the trainer's own stage-level definitions + per-seed means;
            # the raw call below gives per-episode values for same-battle pairing.  Both run the
            # same jitted function, so the second call is just one more forward pass.
            result = trainer.evaluate(params_only(mixed_params))
            raw = trainer._evaluate_impl(mixed_params, trainer.eval_keys)  # noqa: SLF001
            elapsed = time.perf_counter() - t0
            block = identity_stage_block(result, raw, i, trainer.eval_seeds)
            entries[(label, i)] = block
            config_team.append({
                "identity": i,
                "agent": agent_names[i],
                "stage": label,
                "update_count": update_count,
                **block["team"],
            })
            config_seconds.append({"identity": i, "stage": label, "seconds": round(elapsed, 3)})
            print(f"  [{label} u={update_count:>6} {agent_names[i]:<14}] "
                  f"team did {block['team']['wins']}/{block['team']['episodes']} "
                  f"(ret {_fmt(block['team']['mean_return'], 3)}); own dealt "
                  f"{_fmt(block['metrics']['damage_dealt'])} "
                  f"kills {_fmt(block['metrics']['kills'])} "
                  f"({elapsed:.1f}s)", flush=True)

    isolation_pass = all(rec["only_target_identity_changed"] for rec in isolation.values())
    pairs = {i: paired_difference(entries[(labels[0], i)], entries[(labels[-1], i)])
             for i in range(num_agents)} if len(stages) >= 2 else {}

    identities_out = []
    for i in range(num_agents):
        identities_out.append({
            "index": i,
            "agent": trainer.env.agents[i],
            "unit_type": config["unit_type_names"][i],
            "stages": {
                label: {
                    **entries[(label, i)],
                    "update_count": update_count,
                    "checkpoint": path.name,
                    "isolation": isolation[(label, i)],
                }
                for label, update_count, path, _ in stage_params
            },
            "paired_diff": ({
                "from_stage": labels[0],
                "to_stage": labels[-1],
                "from_update_count": stages[0][0],
                "to_update_count": stages[-1][0],
                "pairing": "same seed AND same episode index (the same battle replayed)",
                **pairs[i],
            } if pairs else None),
        })

    payload = {
        "kind": "smax_2s3z_reference_team_eval",
        "diagnostic_only": True,
        "disclaimer": DISCLAIMER,
        "statistics_note": STATISTICS_NOTE,
        "statistics": {
            "eval_seeds": args.eval_seeds,
            "eval_episodes_per_seed": args.eval_episodes,
            "battles_per_configuration": battles,
            "configurations": configurations,
            "episode_executions": configurations * battles,
            "independent_samples": None,
            "sd_seed_meaning": "spread across the seed-group means only; not a CI, not a P2 "
                               "criterion",
            "p2_style_quantity": "same-battle early->late paired difference (same seed and same "
                                 "episode index) plus its cross-seed consistency",
        },
        "run_dir": str(run_dir),
        "reference_checkpoint": reference_path.name,
        "reference_update_count": reference_update,
        "params_source": (
            "checkpoint 'params' field via flax.serialization.msgpack_restore; bitwise-equal to "
            "Trainer.load_checkpoint (verified), without Trainer.init_runner, whose parameter-init "
            "compile intermittently crashes this Windows/JAX build"),
        "eval_seeds": args.eval_seeds,
        "eval_episodes": args.eval_episodes,
        "stages": [
            {"label": label, "update_count": update_count, "checkpoint": path.name}
            for label, update_count, path, _ in stage_params
        ],
        "stage_labels": labels,
        "params_isolation_check": {
            "all_passed": isolation_pass,
            "per_identity": [
                {
                    "identity": i,
                    "agent": agent_names[i],
                    "per_stage": {label: isolation[(label, i)]
                                  for label, _, _, _ in stage_params},
                }
                for i in range(num_agents)
            ],
        },
        "identities": identities_out,
        "team_context": config_team,
        "config_seconds": config_seconds,
        "wall_seconds": round(time.perf_counter() - started, 3),
    }

    out_path = Path(args.output) if args.output else (run_dir / "reference_team_eval.json")
    out_path = out_path if out_path.is_absolute() else (HERE / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("", flush=True)
    print("  identity profiles per stage (identity i swapped, other 4 frozen):", flush=True)
    print(format_stage_table(labels, entries, agent_names), flush=True)
    print("", flush=True)
    if pairs:
        print(f"  same-battle paired difference {labels[-1]}(u={stages[-1][0]}) - "
              f"{labels[0]}(u={stages[0][0]}), mean over pairs:", flush=True)
        print(format_paired_table(pairs, agent_names, labels[0], labels[-1]), flush=True)
    else:
        print("  only one stage available: no early->late paired difference", flush=True)
    print("", flush=True)
    print("  parameter isolation (leaf-wise max |mixed - reference| per identity):", flush=True)
    for i in range(num_agents):
        cells = " ".join(
            f"{label}={isolation[(label, i)]['leaf_max_abs_diff_vs_reference'][i]:.3e}"
            for label, _, _, _ in stage_params
        )
        worst_other = max(isolation[(label, i)]["other_identities_max_abs_diff"]
                          for label, _, _, _ in stage_params)
        print(f"    {agent_names[i]:<14} target {cells}; other 4 identities max="
              f"{worst_other:.3e}", flush=True)
    print(f"  isolation check: {'PASS' if isolation_pass else 'FAIL'} "
          f"(every non-target identity leaf max|diff| == 0 in all configurations)", flush=True)
    print("", flush=True)
    print("  team context per configuration (frozen reference teammates; NOT a conclusion):",
          flush=True)
    for row in config_team:
        print(f"    {row['agent']:<14} {row['stage']:<7} "
              f"win_rate={_fmt(row['win_rate'])} mean_return={_fmt(row['mean_return'])} "
              f"mean_length={_fmt(row['mean_length'], 1)}", flush=True)
    print("", flush=True)
    print(f"  statistics: {STATISTICS_NOTE}", flush=True)
    print(f"  {DISCLAIMER}", flush=True)
    print(f"  wrote {out_path}  (wall {payload['wall_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
