"""Oracle-gate trainer: a single learning observer (`ally_0`) against four frozen partners.

Frozen interface: `MainSearch/explore/ORACLE_GATE_CONTRACT.md` (v1).  Everything in this file that
touches the partner sequence, the random streams, the profile vector or the evaluation protocol
follows that document; nothing here redefines those conventions.

What this is
------------
SMAX `2s3z`: five allies (agent 0/1 = Stalker, agent 2/3/4 = Zealot) against the scripted enemy
policy.  **Only `ally_0` learns.**  Agents 1..4 are *frozen* policies that keep their own network
and their own GRU hidden state; they never receive a gradient.

`z` (contract §1) is read **only** from `results/oracle_gate/profiles.json` - and only by the
`profile` arm; the `placeholder` and `onehot` channels are defined in this file and read no table.
These scripts never re-derive damage or focus attribution: that measurement belongs to
`measure_partner_profiles.py`, which follows the environment's own hit rule (`_world_step` /
`update_agent_health`: range, both alive, `i != target`, weapon cooldown `<= 0`) at sub-step
granularity.  The `profile` arm's z therefore means: **"this partner's behaviour profile as
measured under the fixed reference team (the other identities at their u1250 parameters) on the
fixed measurement situations A"** - it is *not* a team-independent statement about a partner's true
ability, and it is a 3-number summary of `[damage_per_len, alive_frac, focus_share]`, nothing more.
`profiles.json` is authoritative; any legacy copy of the old (v1) table is ignored.

Three arms, identical in every way except the observer's second input channel.  The channel is
12-dimensional in all three; only what is put into it differs:

* `--arm profile`      observer input = `obs(ally_0)` ++ `z` (12 dims, contract §1) - the
                       three-metric behaviour profile **measured** by `measure_partner_profiles.py`
                       and read from `results/oracle_gate/profiles.json`;
* `--arm placeholder`  observer input = `obs(ally_0)` ++ `0` (the same 12 dims, all zero) - the null
                       channel, a constant at training *and* at evaluation;
* `--arm onehot`       observer input = `obs(ally_0)` ++ `onehot` (the same 12 dims) - the battle's
                       partner-stage combination as a per-identity **stage-index one-hot**:
                       identity order `[1,2,3,4]`, 3 dims each, stage order `[u50,u600,u1250]`, so
                       dims `[3j, 3j+3)` of identity `j` are `e_k` where `k` is the stage that
                       identity is drawn to play in this battle.

`onehot` is an **oracle-level encoding of the real stage label** - which of the 12 candidate
parameter sets each partner is drawn to play.  It is *not* a measured behaviour profile (that is
the `profile` arm), it is *not* the null channel (that is `placeholder`), and it is emphatically
*not* a claim about the partner's team-independent ability: it carries the stage index and nothing
else.  Its purpose is to separate "the observer receives partner information at all" from "the
observer receives a constant": any `onehot - placeholder` gap is attributable to the information
the channel carries, with the same architecture, budget, training sequence and random streams.

All three arms therefore have **exactly the same input dimension and the same number of parameters**
(`config["observer_input_dim"] = obs_dim + 12`), the same network recipe (the frozen
`train_smax_2s3z_independent.IdentityActorCritic` GRU actor-critic), the same PPO/GAE maths, the
same official segmented LR annealing and the same single `optax.apply_updates` (the inherited
`Trainer._update_one`, which never uses `TrainState.apply_gradients(grads=tx.update(...))`).
The observer starts from the *same* initial parameters in every arm (the init key is derived from
`--seed` only), so the arms are paired from step 0.

Partners (contract §0, §2)
--------------------------
The 12 candidate partner policies are `stage x identity` from the existing run
`results/smax_2s3z_indep_1250x64x64`: stages `{u50, u600, u1250}` = updates `{50, 600, 1250}` for
each identity `{1,2,3,4}`.  A partner may only ever use *its own identity's* checkpoint parameters,
and a partner never switches stage inside an episode.

Random streams (contract §2) - the arms must see the same partner world
--------------------------------------------------------------------------
Every stream below is a pure function of `(slot, episode_index, in-episode step)` (plus the
identity where relevant) and of nothing else - in particular **not** of the action-sampling rng
and **not** of the roll-out index:

* `partner_stage_index(slot, ep, identity)`       - which of `{u50,u600,u1250}` that partner plays;
* `env_reset_key(slot, ep)`                       - the reset key of that episode;
* `env_step_key(slot, ep, t)`                     - the environment's in-episode step key;
* `partner_action_key(slot, ep, identity, t)`     - the partner-action stream key (§2; recorded in
                                                    the diagnostics, see the errata below).

`t` is the **persistent in-episode step** `ep_length`, kept per slot in `GateRunnerState` and
*never* reset by a roll-out boundary - not the roll-out's own `step` counter, which restarts at 0
in every roll-out (errata 1 below).

The observer's own actions are sampled from a separate rng chain (`runner.rng`, split once per
roll-out step), exactly like the frozen trainer.  Because the partner stage and the environment's
step keys do not read that chain, and because the reset state of episode `(slot, ep)` is *injected*
from `env_reset_key(slot, ep)` (see `_rollout`), two arms whose episode lengths differ still play
exactly the same battle at the same `(slot, episode_index)`, at every in-episode position.

Errata applied on top of the contract text (review ticket 2026-09-27; the four items below override
the earlier implementation, the contract's contract-level conventions are untouched)
--------------------------------------------------------------------------------------------
1. In-episode step index.  `env_step_key` / `partner_action_key` are indexed by the persistent
   in-episode step (`ep_length`), not by the roll-out's `step`.  The roll-out counter restarts at 0
   every roll-out, so with it two arms with different episode lengths would hand different keys to
   the same in-episode position and would re-use keys across roll-outs.  With `ep_length`,
   `(slot, episode_index, in-episode step)` always maps to the same key, independent of the
   episode length and of where the roll-out boundaries fall.
2. Partner actions are **argmax** (`partner_action_argmax`), in training *and* in evaluation - the
   same rule `measure_partner_profiles.py` uses for the profile measurement; the observer still
   samples.  The profiles are **not** re-measured.  The contract §2 partner-action *stream* is kept
   (and recorded in the diagnostics as `diag_partner_action_key`) because it is still the stream
   definition the two arms must share; it no longer feeds a sampling step.
3. `--resume` never rewrites the run's `run.json`.  It writes `resumes/run_resume-NNN.json` next to
   it, recording `resumed_from_run_json`, `resumed_from_checkpoint` and the update count it
   continued from.  A crashed run's `run.json` and segment history are the record of the failure
   being investigated - never overwrite or delete them.
4. Recovery.  Resume an existing run with `--resume <checkpoint>` **and the original config**, not
   with a fresh `--save-dir`.  KNOWN LOCAL RISK: on this Windows box JAX parameter initialisation
   occasionally aborts the process during compilation - no Python traceback, exit code
   `0xC0000001` / `0xC0000005`.  That is a recorded run risk of this machine, so the recipe is
   "resume from the last good checkpoint and keep the failed run's logs", not "just start again".

`--tiny` runs the same code end-to-end in seconds/minutes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as flax_core
from flax import serialization

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import distrax  # noqa: E402
import train_smax_2s3z_independent as base_train  # noqa: E402

# --------------------------------------------------------------------------------------------
# Frozen constants (contract §0-§3)
# --------------------------------------------------------------------------------------------
PARTNER_IDENTITIES = (1, 2, 3, 4)          # partner agent indices; observer = ally_0
STAGE_UPDATES = (50, 600, 1250)            # candidate partner stages in the existing run
STAGE_LABELS = ("u50", "u600", "u1250")
METRIC_NAMES = ("damage_per_len", "alive_frac", "focus_share")
PROFILE_DIM = 3 * len(PARTNER_IDENTITIES)  # 12
NUM_COMBOS = len(STAGE_UPDATES) ** len(PARTNER_IDENTITIES)  # 81

# The three arms (ticket 2026-09-27 added `onehot` to the contract's `profile` / `placeholder`).
# They differ in the observer's second input channel only; the dimension is 12 in all three.
ARMS = ("profile", "placeholder", "onehot")

# What each arm's channel is made of, and what it means.  These strings are recorded in `run.json`
# and in the evaluation payload, so the semantics cannot drift between code, metadata and report.
Z_CHANNEL_BY_ARM = {
    "profile": "obs ++ z(12, per-battle partner combination)",
    "placeholder": "obs ++ zeros(12)",
    "onehot": "obs ++ onehot(12, per-battle partner combination)",
}
Z_SEMANTICS_BY_ARM = {
    "profile": (
        "z is the partner's 3-metric behaviour profile as measured under the FIXED reference "
        "team (every other identity kept at its u1250 parameters) on the fixed measurement "
        "situations A, normalized by the frozen 12-candidate mean/std (contract §1 v2: the "
        "metrics come from the environment's real hit events at sub-step granularity, and they "
        "are read from results/oracle_gate/profiles.json, never re-derived here).  It is a "
        "profile under that reference team, NOT a team-independent statement of the partner's "
        "true ability."
    ),
    "placeholder": (
        "the placeholder arm feeds a constant zero vector in the same 12 dims, at training and at "
        "evaluation alike: it is the null channel (no partner information whatsoever), not a "
        "profile and not a label."
    ),
    "onehot": (
        "z is this battle's partner-stage combination as a per-identity stage-index one-hot "
        "(identity order [1,2,3,4], 3 dims each; stage order [u50,u600,u1250] -> index [0,1,2]): "
        "dims [3j, 3j+3) of identity j are e_k, k being the stage identity j is drawn to play in "
        "this battle.  It is an ORACLE-level encoding of the real stage label - which of the 12 "
        "candidate parameter sets each partner uses - so it is NOT a measured behaviour profile "
        "(that is the profile arm), NOT the null channel (that is placeholder), and NOT a "
        "team-independent statement of the partner's true ability: it carries the stage index and "
        "nothing else."
    ),
}

# Disjoint seed sets (contract §3).  A = profile measurement, B = training flow (`--seed`),
# C = gate evaluation.  Never mix them.
MEASURE_SEEDS_A = (2234, 2235, 2236, 2237, 2238)
TEST_SEEDS_C = (3234, 3235, 3236, 3237)

# Stream seeds (contract §2).  Changing any of them changes only its own stream.
PARTNER_SEED = 20260927       # partner stage choice
RESET_SEED = 20260928         # episode reset key
ENV_STEP_SEED = 20260929      # environment step key
PARTNER_ACTION_SEED = 20260930  # partner action sampling
EVAL_RESET_SEED = 20260931    # gate evaluation: battle reset key
EVAL_STEP_SEED = 20260932     # gate evaluation: environment step key
EVAL_ACTION_SEED = 20260933   # gate evaluation: partner action sampling

PROFILES_PATH = HERE / "results" / "oracle_gate" / "profiles.json"
PROFILES_FIXTURE_PATH = HERE / "results" / "oracle_gate" / "fixtures" / "profiles_fixture.json"
DEFAULT_PARTNER_RUN_DIR = HERE / "results" / "smax_2s3z_indep_1250x64x64"

DIAGNOSTIC_KEYS = ("diag_ep_index", "diag_t_in_episode", "diag_stage", "diag_obs_ally_0",
                   "diag_rollout_step", "diag_env_step_key", "diag_partner_action_key",
                   "diag_partner_action")

_U32 = jnp.uint32
_KNUTH = _U32(0x9E3779B1)
_SHIFT = _U32(15)


# --------------------------------------------------------------------------------------------
# Random streams (contract §2)
# --------------------------------------------------------------------------------------------
def _hash_scalar(base_seed: int, *values) -> jnp.ndarray:
    """`hash(base_seed, *values)` as a uint32 hash; broadcasts over array-valued inputs."""
    hashed = jnp.asarray(base_seed, dtype=_U32)
    for value in values:
        hashed = (hashed ^ jnp.asarray(value, dtype=_U32)) * _KNUTH
        hashed = hashed ^ (hashed >> _SHIFT)
    return hashed


def stream_key(base_seed: int, *values) -> jnp.ndarray:
    """PRNG key(s) `PRNGKey(hash(base_seed, *values))`; shape `values_shape + (2,)`.

    `jax.random.PRNGKey` accepts scalar seeds only, so the hash is vmapped over its elements; the
    result is a pure, deterministic function of the arguments (no global state, jit-safe).
    """
    hashed = _hash_scalar(base_seed, *values)
    flat = jnp.reshape(jnp.asarray(hashed, dtype=_U32), (-1,))
    keys = jax.vmap(jax.random.PRNGKey)(flat)
    return jnp.reshape(keys, tuple(hashed.shape) + (2,))


def partner_stage_index(slot, ep_index, identity, seed: int = PARTNER_SEED) -> jnp.ndarray:
    """Index into `STAGE_UPDATES` for one partner identity, from `(slot, ep_index)` only."""
    keys = stream_key(seed, slot, ep_index, identity)
    flat = jnp.reshape(keys, (-1, 2))
    index = jax.vmap(lambda key: jax.random.randint(key, (), 0, len(STAGE_UPDATES)))(flat)
    return jnp.reshape(index, tuple(keys.shape[:-1]))


def env_reset_key(slot, ep_index, seed: int = RESET_SEED) -> jnp.ndarray:
    """Reset key of episode `ep_index` in environment slot `slot`."""
    return stream_key(seed, slot, ep_index)


def env_step_key(slot, ep_index, step, seed: int = ENV_STEP_SEED) -> jnp.ndarray:
    """Environment step key: in-episode stochasticity only (resets are injected explicitly)."""
    return stream_key(seed, slot, ep_index, step)


def partner_action_key(slot, ep_index, identity, step, seed: int = PARTNER_ACTION_SEED):
    """Partner-action stream key `(slot, ep, identity, in-episode step)`; contract §2.

    The partners act by `partner_action_argmax` (errata 2), so this key no longer seeds a sampling
    step; it is still the contract's stream definition and is recorded in the diagnostics
    (`diag_partner_action_key`) so the two arms can be checked to share it.
    """
    return stream_key(seed, slot, ep_index, identity, step)


def partner_action_argmax(logits) -> jnp.ndarray:
    """The partners' action: the argmax of their policy logits (errata 2).

    `measure_partner_profiles.py` measures the profile with argmax actors, so training and
    evaluation must use the same rule; only the observer samples (from its own rng chain).  Used by
    the training roll-out, by the gate battles and by the matched-state rollouts.
    """
    return jnp.argmax(logits, axis=-1)


def categorical_sample(keys, logits) -> jnp.ndarray:
    """Sample one categorical action per lane, each lane with its own key.

    `distrax`/`jax.random.categorical` accept a *single* key, so per-lane keys are vmapped; the
    lane axis is `keys.shape[:-1]` and `logits` must hold exactly that many rows.

    Since the partner rule is argmax (errata 2) the trainers no longer sample partner actions; this
    helper stays as the pre-errata rule that the preflight uses as its negative control
    ("sampling would not be invariant to the partner-action stream").
    """
    flat_keys = jnp.reshape(keys, (-1, 2))
    flat_logits = jnp.reshape(logits, (flat_keys.shape[0], logits.shape[-1]))
    samples = jax.vmap(
        lambda key, logit: distrax.Categorical(logits=logit).sample(seed=key)
    )(flat_keys, flat_logits)
    return jnp.reshape(samples, tuple(keys.shape[:-1]))


def observer_action_key(action_rng) -> jnp.ndarray:
    """The observer's action-sampling key, derived from the action stream only."""
    return jax.random.fold_in(action_rng, 0)


def enumerate_combos(subset: int | None = None) -> list[tuple[int, ...]]:
    """All 81 partner-stage combinations in lexicographic order (identity order [1,2,3,4]).

    `subset` keeps `subset` combinations spread evenly over the full list (a *subset*, never to be
    reported as the contract's full 81-combination protocol).
    """
    combos = list(itertools.product(range(len(STAGE_UPDATES)), repeat=len(PARTNER_IDENTITIES)))
    if subset is None or subset >= len(combos):
        return combos
    if subset < 1:
        raise ValueError("combo subset must be >= 1")
    picked = np.unique(np.round(np.linspace(0, len(combos) - 1, subset)).astype(int))
    return [combos[int(k)] for k in picked]


def combo_label(combo: Sequence[int]) -> str:
    return "[" + ",".join(STAGE_LABELS[int(s)] for s in combo) + "]"


# --------------------------------------------------------------------------------------------
# Profile table (contract §1.1)
# --------------------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalization_from_raw(raw: Dict[str, Dict[str, Sequence[float]]]) -> tuple[list, list]:
    """Calibration mean/std over **all 12 candidates** (contract §1), recomputed from `raw`."""
    profiles = [
        np.asarray(raw[str(ident)][str(stage)], dtype=np.float64)
        for ident in PARTNER_IDENTITIES
        for stage in STAGE_UPDATES
    ]
    stacked = np.stack(profiles)                      # (12, 3)
    return list(stacked.mean(0)), list(stacked.std(0) + 1e-8)


def validate_profile_table(table: Any, source: Path) -> None:
    """Fail loudly on any deviation from contract §1.1 (schema, sets, frozen normalization)."""
    where = f"{source}"
    if not isinstance(table, dict):
        raise SystemExit(f"profile table {where}: not a JSON object")
    if table.get("kind") != "partner_profile_table":
        raise SystemExit(f"profile table {where}: kind != partner_profile_table")
    if [int(s) for s in table.get("stages", [])] != list(STAGE_UPDATES):
        raise SystemExit(f"profile table {where}: stages != {list(STAGE_UPDATES)}")
    if [int(i) for i in table.get("partner_identities", [])] != list(PARTNER_IDENTITIES):
        raise SystemExit(f"profile table {where}: partner_identities != {list(PARTNER_IDENTITIES)}")
    if list(table.get("metric_names", [])) != list(METRIC_NAMES):
        raise SystemExit(f"profile table {where}: metric_names != {list(METRIC_NAMES)}")
    raw = table.get("raw")
    if not isinstance(raw, dict):
        raise SystemExit(f"profile table {where}: missing 'raw'")
    for ident in PARTNER_IDENTITIES:
        if str(ident) not in raw:
            raise SystemExit(f"profile table {where}: raw missing identity {ident}")
        for stage in STAGE_UPDATES:
            if str(stage) not in raw[str(ident)]:
                raise SystemExit(f"profile table {where}: raw missing identity {ident}/{stage}")
            entry = np.asarray(raw[str(ident)][str(stage)], dtype=np.float64)
            if entry.shape != (len(METRIC_NAMES),) or not np.all(np.isfinite(entry)):
                raise SystemExit(
                    f"profile table {where}: raw[{ident}][{stage}] is not {len(METRIC_NAMES)} "
                    f"finite numbers"
                )
    norm = table.get("normalization")
    if not isinstance(norm, dict) or "mean" not in norm or "std" not in norm:
        raise SystemExit(f"profile table {where}: missing normalization.mean/std")
    mean, std = normalization_from_raw(raw)
    stored_mean = np.asarray(norm["mean"], dtype=np.float64)
    stored_std = np.asarray(norm["std"], dtype=np.float64)
    if stored_mean.shape != (len(METRIC_NAMES),) or stored_std.shape != (len(METRIC_NAMES),):
        raise SystemExit(f"profile table {where}: normalization vectors must be 3-dimensional")
    if not np.allclose(stored_mean, mean, atol=1e-6, rtol=0.0):
        raise SystemExit(f"profile table {where}: stored mean != mean of the 12 candidates")
    if not np.allclose(stored_std, std, atol=1e-6, rtol=0.0):
        raise SystemExit(f"profile table {where}: stored std != std of the 12 candidates")
    measured = set(int(s) for s in table.get("measurement", {}).get("seeds", []))
    clash = measured & set(TEST_SEEDS_C)
    if clash:
        raise SystemExit(f"profile table {where}: measurement seeds overlap the test set C: {clash}")
    if "z_layout" not in table:
        raise SystemExit(f"profile table {where}: missing z_layout")


def load_profile_table(path: str | Path | None = None) -> tuple[dict, Path, bool]:
    """Load the frozen profile table; fall back to the local fixture when the real one is absent.

    Returns `(table, source_path, is_fixture)`.  The fixture has the contract's schema and is only
    a stand-in - the report must name the path it used.
    """
    if path is not None:
        source = Path(path)
        source = source if source.is_absolute() else (HERE / source)
        if not source.is_file():
            raise SystemExit(f"--profiles {source} does not exist")
        is_fixture = "fixture" in source.name
    elif PROFILES_PATH.is_file():
        source, is_fixture = PROFILES_PATH, False
    elif PROFILES_FIXTURE_PATH.is_file():
        source, is_fixture = PROFILES_FIXTURE_PATH, True
    else:
        raise SystemExit(
            f"no profile table: neither {PROFILES_PATH} nor the fixture "
            f"{PROFILES_FIXTURE_PATH} exists"
        )
    table = json.loads(source.read_text(encoding="utf-8"))
    validate_profile_table(table, source)
    return table, source, is_fixture


def z_table_from_profile_table(table: dict) -> np.ndarray:
    """(4, 3, 3) normalized profiles, indexed by `[partner_index, stage_index, metric]`."""
    mean = np.asarray(table["normalization"]["mean"], dtype=np.float64)
    std = np.asarray(table["normalization"]["std"], dtype=np.float64)
    out = np.zeros((len(PARTNER_IDENTITIES), len(STAGE_UPDATES), len(METRIC_NAMES)), np.float32)
    for j, ident in enumerate(PARTNER_IDENTITIES):
        for s, stage in enumerate(STAGE_UPDATES):
            raw = np.asarray(table["raw"][str(ident)][str(stage)], dtype=np.float64)
            out[j, s] = ((raw - mean) / std).astype(np.float32)
    return out


def z_vector_from_table(z_table: np.ndarray, combo: Sequence[int]) -> np.ndarray:
    """12-dim `z` for a partner-stage combination, in identity order `[1,2,3,4]`."""
    z_table = np.asarray(z_table)
    return np.concatenate([z_table[j, int(combo[j])] for j in range(len(PARTNER_IDENTITIES))])


# --------------------------------------------------------------------------------------------
# One-hot channel (the third arm; ticket 2026-09-27)
# --------------------------------------------------------------------------------------------
def onehot_z_table() -> np.ndarray:
    """The one-hot arm's `(4, 3, 3)` z table: row `[j, s]` is stage `s`'s one-hot of length 3.

    The table is indexed exactly like the profile table (`[partner_index, stage_index, metric]`)
    and is *not* read from `profiles.json` - the one-hot arm never touches the measured profile.
    Because the concatenation order is fixed (identity `[1,2,3,4]`, 3 dims each), the trainer's own
    table machinery turns this into the contract's 12-dim `z` for free:
    `z_vector_from_table(onehot_z_table(), combo)` - see `onehot_z_vector`.  Stage order is
    `[u50, u600, u1250]`, so the hot index is the stage index the partner is drawn to play.
    """
    table = np.zeros((len(PARTNER_IDENTITIES), len(STAGE_UPDATES), len(METRIC_NAMES)), np.float32)
    for stage in range(len(STAGE_UPDATES)):
        table[:, stage, stage] = 1.0
    return table


def onehot_z_vector(combo: Sequence[int]) -> np.ndarray:
    """12-dim one-hot `z` for a partner-stage combination (the one-hot arm's channel)."""
    return z_vector_from_table(onehot_z_table(), combo).astype(np.float32)


# --------------------------------------------------------------------------------------------
# Partner parameters (frozen; identity x stage)
# --------------------------------------------------------------------------------------------
def load_checkpoint_params(path: Path):
    """The five identities' parameter pytrees from a checkpoint's `params` field.

    Same technique as `evaluate_reference_team.load_checkpoint_params`: reads the msgpack payload
    directly, so it never runs the parameter-initialisation compile that intermittently kills this
    Windows/JAX build.
    """
    payload = serialization.msgpack_restore(Path(path).read_bytes())["params"]
    return tuple(jax.tree.map(jnp.asarray, payload[key]) for key in sorted(payload, key=int))


def stage_checkpoint_paths(run_dir: Path, stages: Sequence[int] = STAGE_UPDATES) -> list[Path]:
    return [Path(run_dir) / f"checkpoint_{int(stage):08d}.bin" for stage in stages]


def load_partner_stack(run_dir: Path, stages: Sequence[int] = STAGE_UPDATES):
    """`stack[j][s]` = identity `PARTNER_IDENTITIES[j]`'s parameters at stage `stages[s]`."""
    paths = stage_checkpoint_paths(run_dir, stages)
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"partner checkpoint missing: {path}")
    per_stage = [load_checkpoint_params(path) for path in paths]
    return tuple(
        tuple(per_stage[s][ident] for s in range(len(stages))) for ident in PARTNER_IDENTITIES
    )


def tree_bitwise_equal(a, b) -> bool:
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(leaves_a) != len(leaves_b):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(leaves_a, leaves_b))


def tree_max_abs_diff(a, b) -> float:
    leaves_a, leaves_b = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(leaves_a) != len(leaves_b):
        raise ValueError("pytree structure mismatch")
    if not leaves_a:
        return 0.0
    return max(
        float(np.max(np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))))
        for x, y in zip(leaves_a, leaves_b)
    )


# --------------------------------------------------------------------------------------------
# Training state
# --------------------------------------------------------------------------------------------
class GateRunnerState(NamedTuple):
    """Everything a resumed observer run needs; every field is checkpointed."""

    params: tuple            # (observer,) - ONE entry; the partners are frozen constants
    opt_states: tuple        # (observer,)
    hidden: tuple            # (observer,) GRU hidden, (num_envs, gru_dim)
    partner_hidden: tuple    # 4 GRU hidden states, (num_envs, gru_dim) each
    env_state: Any           # SMAXLogEnvState
    obs: Any                 # {agent: (num_envs, obs_dim)}
    last_done: jnp.ndarray   # (num_envs,) bool
    ep_index: jnp.ndarray    # (num_envs,) int32, completed episodes per slot
    rng: jnp.ndarray         # action-sampling stream only
    update_count: jnp.ndarray
    episode_returns: jnp.ndarray
    episode_lengths: jnp.ndarray


def _select_env(mask: jnp.ndarray, a, b):
    """Per-environment select of two batched leaves; `mask` is `(num_envs,)`."""
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.ndim == 0:
        return jnp.where(jnp.any(mask), a, b)
    return jnp.where(mask.reshape((mask.shape[0],) + (1,) * (a.ndim - 1)), a, b)


class ObserverGateTrainer(base_train.Trainer):
    """The frozen trainer's recipe, restricted to one learning observer with four frozen partners."""

    def __init__(self, config: dict, partner_stack, z_table, arm: str = "profile",
                 env_overrides: Dict[str, Any] | None = None):
        config = dict(config)
        if env_overrides:
            config["env_kwargs"] = {**config["env_kwargs"], **env_overrides}
        super().__init__(config)
        # `build_config` derived the horizon from a probe environment; honour the real one.
        self.config["horizon"] = int(self.base_env.max_steps)
        # A battle runs to its real termination (`done["__all__"]`), never to a fixed step count:
        # the local SMAX environment flags `done` *before* incrementing its own step counter
        # (`smax_env.is_terminal` compares the pre-increment `state.step` with `max_steps`), so a
        # battle that ends on the time limit needs `max_steps + 1` calls.  `eval_step_limit` is
        # only a loop guard and must stay at least that high; a battle still alive when it runs out
        # is TRUNCATED and `eval_battles` reports it as a failure instead of a result (errata 4).
        self.eval_step_limit = self.config["horizon"] + 1
        self.config["observer_input_dim"] = self.config["obs_dim"] + PROFILE_DIM
        self.config["profile_dim"] = PROFILE_DIM
        self.config["partner_identities"] = list(PARTNER_IDENTITIES)
        self.config["partner_stage_updates"] = list(STAGE_UPDATES)
        self.config["arm"] = arm
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
        self.arm = arm
        self.partner_stack = partner_stack
        # The channel of the two arms that have a *fixed* channel is derived from the arm name, so
        # no caller can hand an arm the wrong input: `placeholder` is a constant zero by definition
        # and `onehot` is the combination's stage label, never the measured profile.  Only
        # `profile` uses the table the caller passes in (`profiles.json`, contract §1).
        if arm == "placeholder":
            self.z_table = jnp.asarray(np.zeros((len(PARTNER_IDENTITIES), len(STAGE_UPDATES),
                                                 len(METRIC_NAMES)), np.float32))
        elif arm == "onehot":
            self.z_table = jnp.asarray(onehot_z_table())
        else:
            self.z_table = jnp.asarray(np.asarray(z_table, np.float32))
        # Diagnostics (contract §7 checks) are opt-in: they are what the preflight inspects.
        self.collect_diagnostics = False
        self.last_diagnostics: Dict[str, jnp.ndarray] = {}
        self._battle_group_impl = jax.jit(self._battle_group)

    # -- z ------------------------------------------------------------------------------------
    def z_vector(self, stage_indices) -> jnp.ndarray:
        """12-dim z for per-identity stage indices (a `(4,)` combo or traced scalars)."""
        return jnp.concatenate(
            [jnp.take(self.z_table[j], stage_indices[j], axis=0)
             for j in range(len(PARTNER_IDENTITIES))], axis=-1
        )

    def z_batch(self, slots: jnp.ndarray, ep_index: jnp.ndarray) -> jnp.ndarray:
        """`(num_envs, 12)` z for the episode currently running in each environment slot."""
        stage_idx = jnp.stack([
            partner_stage_index(slots, ep_index, jnp.int32(ident)) for ident in PARTNER_IDENTITIES
        ])
        return jnp.concatenate(
            [jnp.take(self.z_table[j], stage_idx[j], axis=0)
             for j in range(len(PARTNER_IDENTITIES))], axis=-1
        )

    # -- initialisation ----------------------------------------------------------------------
    def init_runner(self, rng) -> GateRunnerState:
        config = self.config
        num_envs = config["num_envs"]
        gru = config["gru_hidden_dim"]
        rng, init_rng, action_rng = jax.random.split(jnp.asarray(rng), 3)
        init_hstate = base_train.ScannedRNN.initialize_carry(num_envs, gru)
        init_x = (
            jnp.zeros((1, num_envs, config["observer_input_dim"])),
            jnp.zeros((1, num_envs)),
            jnp.zeros((1, num_envs, config["action_dim"])),
        )
        params = (flax_core.unfreeze(self.network.init(init_rng, init_hstate, init_x)),)
        opt_states = (self.tx.init(params[0]),)
        slots = jnp.arange(num_envs, dtype=jnp.int32)
        obs, env_state = jax.vmap(self.env.reset)(env_reset_key(slots, jnp.zeros_like(slots)))
        return GateRunnerState(
            params=params,
            opt_states=opt_states,
            hidden=(base_train.ScannedRNN.initialize_carry(num_envs, gru),),
            partner_hidden=tuple(
                base_train.ScannedRNN.initialize_carry(num_envs, gru)
                for _ in PARTNER_IDENTITIES
            ),
            env_state=env_state,
            obs=obs,
            last_done=jnp.zeros((num_envs,), dtype=bool),
            ep_index=jnp.zeros((num_envs,), dtype=jnp.int32),
            rng=action_rng,
            update_count=jnp.zeros((), dtype=jnp.int32),
            episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
            episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    # -- roll-out ----------------------------------------------------------------------------
    def _rollout(self, runner: GateRunnerState):
        config, env, base_env, network = self.config, self.env, self.base_env, self.network
        num_envs, horizon = config["num_envs"], config["rollout_length"]
        action_dim = config["action_dim"]
        gru = config["gru_hidden_dim"]
        agents = env.agents
        slots = jnp.arange(num_envs, dtype=jnp.int32)
        collect = bool(self.collect_diagnostics)

        def _env_step(carry, step):
            (env_state, obs, last_done, hidden, partner_hidden, ep_index, rng, ep_return,
             ep_length) = carry
            # The action stream is a chain of its own (contract §2): nothing below reads it.
            rng, action_rng = jax.random.split(rng)
            obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
            avail = jax.vmap(base_env.get_avail_actions)(env_state.env_state)
            avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)

            stage_idx = jnp.stack([
                partner_stage_index(slots, ep_index, jnp.int32(ident))
                for ident in PARTNER_IDENTITIES
            ])  # (4, num_envs)
            z = jnp.concatenate(
                [jnp.take(self.z_table[j], stage_idx[j], axis=0)
                 for j in range(len(PARTNER_IDENTITIES))], axis=-1
            )  # (num_envs, 12); all-zero in the placeholder arm, the stage one-hot in the onehot arm

            # In-episode step index (errata 1): `ep_length` counts the steps already played in the
            # episode running in each slot, so it is the position inside the battle - it does not
            # restart at a roll-out boundary and does not depend on the arm's episode length.
            step_index = ep_length
            env_keys = env_step_key(slots, ep_index, step_index)
            partner_keys = (
                jnp.stack([partner_action_key(slots, ep_index, jnp.int32(ident), step_index)
                           for ident in PARTNER_IDENTITIES]) if collect else None
            )

            # --- observer ---------------------------------------------------------------------
            x_observer = jnp.concatenate([obs_stack[0], z], axis=-1)
            h_observer, pi_observer, value = network.apply(
                runner.params[0], hidden[0],
                (x_observer[None], last_done[None], avail_stack[0]),
            )
            action_observer = pi_observer.sample(seed=observer_action_key(action_rng)).squeeze(0)
            log_prob = pi_observer.log_prob(action_observer).squeeze(0)

            # --- partners: frozen, own identity, own stage, own hidden state, own rng stream ---
            partner_actions, new_partner_hidden = [], []
            for j, ident in enumerate(PARTNER_IDENTITIES):
                stage_logits, stage_hidden = [], []
                for s in range(len(STAGE_UPDATES)):
                    h_s, pi_s, _ = network.apply(
                        self.partner_stack[j][s], partner_hidden[j],
                        (obs_stack[ident][None], last_done[None], avail_stack[ident]),
                    )
                    stage_logits.append(pi_s.logits)
                    stage_hidden.append(h_s)
                logits, new_hidden = stage_logits[-1], stage_hidden[-1]
                for s in range(len(STAGE_UPDATES) - 2, -1, -1):
                    mask = stage_idx[j] == s
                    # logits carry a leading time axis of 1 (one step per call), the hidden state
                    # does not: the per-slot stage mask broadcasts over the matching axis.
                    logits = jnp.where(mask[None, :, None], stage_logits[s], logits)
                    new_hidden = jnp.where(mask[:, None], stage_hidden[s], new_hidden)
                pi_partner_logits = logits.squeeze(0)
                # Errata 2: the partners are argmax actors (the profile-measurement rule).
                partner_actions.append(partner_action_argmax(pi_partner_logits))
                new_partner_hidden.append(new_hidden)

            action_stack = jnp.stack([action_observer] + partner_actions)
            env_act = {agent: action_stack[i] for i, agent in enumerate(agents)}
            new_obs, new_env_state, reward, done_new, info = jax.vmap(env.step)(
                env_keys, env_state, env_act
            )
            done_all = jnp.asarray(done_new["__all__"], dtype=bool)

            # Deterministic reset injection (contract §2): episode `ep_index+1` of slot `slot`
            # always starts from `env.reset(env_reset_key(slot, ep_index+1))`, whatever the length
            # of the previous episode was in this arm.
            next_ep_index = ep_index + done_all.astype(jnp.int32)
            reset_obs, reset_state = jax.vmap(base_env.reset)(env_reset_key(slots, next_ep_index))
            new_obs = jax.tree.map(lambda r, s: _select_env(done_all, r, s), reset_obs, new_obs)
            new_env_state = new_env_state.replace(
                env_state=jax.tree.map(
                    lambda r, s: _select_env(done_all, r, s), reset_state,
                    new_env_state.env_state,
                )
            )

            reward_stack = jnp.stack([reward[a] for a in agents]).astype(jnp.float32)
            won = jnp.asarray(info["returned_won_episode"][..., 0], dtype=jnp.float32)
            t_in_episode = ep_length          # step index inside the episode that just ran (0-based)
            ep_return = ep_return + reward_stack[0]
            ep_length = ep_length + 1
            broadcast = lambda x: jnp.broadcast_to(x[None], (1, num_envs))  # noqa: E731

            transition = {
                "obs": x_observer[None],
                "avail": avail_stack[0][None],
                "done": last_done[None],
                "action": action_observer[None],
                "log_prob": log_prob[None],
                "value": value.squeeze(0)[None],
                "reward": reward_stack[0][None],
                "global_done": broadcast(done_all.astype(jnp.float32)),
                "finished": broadcast(done_all.astype(jnp.float32)),
                "finished_return": broadcast(jnp.where(done_all, ep_return, 0.0)),
                "finished_length": broadcast(jnp.where(done_all, ep_length, 0).astype(jnp.float32)),
                "finished_won": broadcast(jnp.where(done_all, won, 0.0)),
            }
            if collect:
                transition["diag_ep_index"] = ep_index
                transition["diag_t_in_episode"] = t_in_episode
                transition["diag_stage"] = stage_idx
                transition["diag_obs_ally_0"] = obs_stack[0]
                # The streams the two arms must share, and the actions actually taken, so the
                # preflight can check the alignment without re-deriving it from the environment.
                transition["diag_rollout_step"] = jnp.broadcast_to(step, (num_envs,))
                transition["diag_env_step_key"] = env_keys
                transition["diag_partner_action_key"] = partner_keys
                transition["diag_partner_action"] = jnp.stack(partner_actions)
            new_carry = (
                new_env_state, new_obs, done_all, (h_observer,), tuple(new_partner_hidden),
                next_ep_index, rng,
                jnp.where(done_all, 0.0, ep_return),
                jnp.where(done_all, 0, ep_length),
            )
            return new_carry, transition

        carry = (runner.env_state, runner.obs, runner.last_done, runner.hidden,
                 runner.partner_hidden, runner.ep_index, runner.rng,
                 runner.episode_returns, runner.episode_lengths)
        carry, transitions = jax.lax.scan(_env_step, carry, jnp.arange(horizon))
        (env_state, obs, last_done, hidden, partner_hidden, ep_index, rng, ep_return,
         ep_length) = carry

        # Bootstrap value: same convention as the frozen trainer (availability ignored).
        obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
        z_last = self.z_batch(slots, ep_index)
        x_last = jnp.concatenate([obs_stack[0], z_last], axis=-1)
        ones = jnp.ones((num_envs, action_dim), dtype=jnp.float32)
        last_value = network.apply(
            runner.params[0], hidden[0], (x_last[None], last_done[None], ones)
        )[2].squeeze()
        new_runner = runner._replace(
            hidden=hidden, partner_hidden=partner_hidden, env_state=env_state, obs=obs,
            last_done=last_done, ep_index=ep_index, rng=rng, episode_returns=ep_return,
            episode_lengths=ep_length,
        )
        return new_runner, transitions, last_value[None]

    # -- orchestration -----------------------------------------------------------------------
    def rollout_batch(self, runner: GateRunnerState):
        """One roll-out + the observer's GAE; diagnostics are parked on the instance."""
        hstate_init = runner.hidden
        new_runner, transitions, last_values = self.rollout(runner)
        self.last_diagnostics = {key: transitions.pop(key) for key in DIAGNOSTIC_KEYS
                                 if key in transitions}
        advantages, returns = self.gae(transitions, last_values)
        return new_runner, (jax.tree.map(lambda x: x[:, 0], transitions),), advantages, returns, \
            hstate_init

    def apply_identity_updates(self, runner, trans_by_identity, advantages, returns, hstate_init,
                               active=None):
        """Only identity 0 (the observer) is ever updated; the partners are not in `runner`.

        `Trainer.train_updates` forwards `active=None`, which the parent would expand to all five
        identities - here it means "the observer", since `runner.params` holds exactly one entry.
        """
        active = (0,) if active is None else tuple(active)
        return super().apply_identity_updates(
            runner, trans_by_identity, advantages, returns, hstate_init, active=active
        )

    def _evaluate(self, params, keys):  # pragma: no cover - deliberately not used here
        raise NotImplementedError(
            "use ObserverGateTrainer.eval_battles: the gate protocol pairs battles by "
            "(partner-stage combination, test situation) instead of a single shared policy"
        )

    # -- gate evaluation (§4 protocol; also used for the per-segment monitor) ----------------
    def eval_battles(self, params, combos: Sequence[Sequence[int]], seeds: Sequence[int],
                     z_mode: str = "table"):
        """Per-(combination, seed) battles, each run to its real termination.

        Returns `won`, `return`, `length` and `truncated` arrays shaped `(len(combos), len(seeds))`.
        All arms are evaluated on the *same* combos and the *same* seeds (contract §4);
        `z_mode="zero"` is the placeholder arm's input (its channel is a constant zero, never the
        battle's z), while `z_mode="table"` is what the profile and onehot arms use: it reads the
        arm's own `z_table`, which for the onehot arm is the stage one-hot table and for the
        profile arm the measured profile table.  `length` is the number of steps actually played
        and `truncated` marks battles that were still running when the loop guard ran out
        (`max_steps` is **not** the loop bound - errata 4); any truncated battle raises instead of
        being reported as a normal result.
        """
        if z_mode not in ("table", "zero"):
            raise SystemExit(f"unknown z_mode {z_mode!r}")
        seeds_arr = jnp.asarray(np.asarray(seeds, dtype=np.int32))
        shape = (len(combos), len(seeds))
        out = {k: np.zeros(shape, dtype=np.float64) for k in ("won", "return", "length")}
        truncated = np.zeros(shape, dtype=bool)
        for ci, combo in enumerate(combos):
            partner_params = tuple(self.partner_stack[j][int(combo[j])]
                                   for j in range(len(PARTNER_IDENTITIES)))
            z = (jnp.zeros((PROFILE_DIM,), jnp.float32) if z_mode == "zero"
                 else self.z_vector(jnp.asarray(np.asarray(combo, np.int32))))
            team = np.asarray(jax.device_get(
                self._battle_group_impl(params, partner_params, z, seeds_arr)
            ), dtype=np.float64)
            out["won"][ci] = team[:, 0]
            out["return"][ci] = team[:, 1]
            out["length"][ci] = team[:, 2]
            truncated[ci] = team[:, 3] > 0.0
        out["truncated"] = truncated
        if truncated.any():
            seeds_np = np.asarray(seeds)
            where = [(combo_label(combos[ci]), int(seeds_np[si]))
                     for ci, si in zip(*np.nonzero(truncated))]
            raise SystemExit(
                f"TRUNCATED evaluation: {int(truncated.sum())} battle(s) were still running when "
                f"the loop guard (eval_step_limit={self.eval_step_limit}) ran out: {where}. The "
                f"battle(s) never reached done['__all__'], so they are not results - keep "
                f"eval_step_limit >= max_steps + 1 "
                f"(max_steps={self.config['horizon']}, the environment flags done one step late)."
            )
        return out

    def _battle_group(self, params, partner_params, z, seeds):
        return jax.vmap(lambda seed: self._battle_episode(params, partner_params, z, seed))(seeds)

    def _battle_episode(self, params, partner_params, z, seed):
        """One deterministic battle: observer argmax, frozen argmax partners, fixed streams (§4).

        Runs from `env.reset(stream_key(EVAL_RESET_SEED, seed))` until `done["__all__"]`, up to the
        `eval_step_limit` guard; the returned 4th entry flags a battle that was still alive when the
        guard ran out (see `eval_battles`, errata 4).
        """
        config, env, base_env, network = self.config, self.env, self.base_env, self.network
        agents = env.agents
        gru, action_dim = config["gru_hidden_dim"], config["action_dim"]
        limit = int(self.eval_step_limit)

        obs, env_state = env.reset(stream_key(EVAL_RESET_SEED, seed))
        hidden = base_train.ScannedRNN.initialize_carry(1, gru)
        partner_hidden = tuple(base_train.ScannedRNN.initialize_carry(1, gru)
                               for _ in PARTNER_IDENTITIES)
        zero_done = jnp.zeros((1, 1), dtype=bool)

        def _step(carry):
            (step, obs, env_state, hidden, partner_hidden, live, ep_return, ep_length, won) = carry
            obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
            avail = base_env.get_avail_actions(env_state.env_state)
            avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)

            x_observer = jnp.concatenate([obs_stack[0], z], axis=-1)[None, None]
            h_observer, pi_observer, _ = network.apply(
                params, hidden, (x_observer, zero_done, avail_stack[0][None])
            )
            logits = jnp.where(avail_stack[0][None] > 0, pi_observer.logits, -1e9)
            actions = [jnp.argmax(logits, axis=-1)[0, 0]]

            new_partner_hidden = []
            for j, ident in enumerate(PARTNER_IDENTITIES):
                h_partner, pi_partner, _ = network.apply(
                    partner_params[j], partner_hidden[j],
                    (obs_stack[ident][None, None], zero_done, avail_stack[ident][None]),
                )
                # Errata 2: argmax, the same rule the profile measurement used.
                actions.append(partner_action_argmax(pi_partner.logits)[0, 0])
                new_partner_hidden.append(h_partner)

            env_act = {agent: actions[i] for i, agent in enumerate(agents)}
            new_obs, new_env_state, reward, done_new, info = env.step(
                stream_key(EVAL_STEP_SEED, seed, step), env_state, env_act
            )
            done_all = jnp.asarray(done_new["__all__"], dtype=bool)
            ep_return = ep_return + jnp.where(live, reward[agents[0]], 0.0)
            ep_length = ep_length + jnp.where(live, 1, 0)
            won = jnp.where(
                live & done_all,
                jnp.asarray(info["returned_won_episode"][0], dtype=jnp.float32),
                won,
            )
            return (step + 1, new_obs, new_env_state, h_observer, tuple(new_partner_hidden),
                    live & ~done_all, ep_return, ep_length, won)

        def _running(carry) -> jnp.ndarray:
            step, _, _, _, _, live, _, _, _ = carry
            return live & (step < limit)

        init = (jnp.zeros((), jnp.int32), obs, env_state, hidden, partner_hidden,
                jnp.asarray(True), jnp.zeros((), jnp.float32), jnp.zeros((), jnp.int32),
                jnp.zeros((), jnp.float32))
        final = jax.lax.while_loop(_running, _step, init)
        (_, _, _, _, _, live, ep_return, ep_length, won) = final
        # `live` is still True only for a battle the loop guard cut short: it never terminated.
        return jnp.stack([won, ep_return, ep_length.astype(jnp.float32), live.astype(jnp.float32)])

# --------------------------------------------------------------------------------------------
# Checkpointing (same pattern as the frozen trainer; GateRunnerState has extra fields)
# --------------------------------------------------------------------------------------------
def save_checkpoint(path: Path, runner: GateRunnerState, meta: dict) -> Path:
    payload = {name: getattr(runner, name) for name in GateRunnerState._fields}
    payload["_meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(payload))
    return path


def read_checkpoint_meta(path: Path) -> dict:
    return serialization.msgpack_restore(Path(path).read_bytes())["_meta"]


def load_checkpoint(path: Path, template: GateRunnerState) -> GateRunnerState:
    payload = serialization.from_bytes(
        {name: getattr(template, name) for name in GateRunnerState._fields} | {"_meta": {}},
        Path(path).read_bytes(),
    )
    restored = GateRunnerState(**{name: payload[name] for name in GateRunnerState._fields})
    return jax.tree.map(jnp.asarray, restored)


def load_observer_params(path: Path):
    """The observer's parameter pytree (the single entry of `params` in a gate checkpoint)."""
    payload = serialization.msgpack_restore(Path(path).read_bytes())["params"]
    return jax.tree.map(jnp.asarray, payload["0"])


def run_dir_for(save_dir: str | None, run_name: str) -> Path:
    path = Path(save_dir) if save_dir else (HERE / "results" / "oracle_gate" / run_name)
    path = path if path.is_absolute() else (HERE / path)
    if "results" not in path.parts:
        raise SystemExit(f"--save-dir must live under a 'results/' directory (got {path})")
    return path


def resume_metadata_path(out_dir: Path) -> Path:
    """Where a resumed run records its own metadata: `resumes/run_resume-NNN.json`.

    A resumed run MUST NOT rewrite the `run.json` of the run it continues: that file records how the
    original run was launched and its segment history is the evidence of the failure under
    investigation.  Each resume gets its own numbered file in `resumes/` instead, and records which
    `run.json` and which checkpoint it continued from (errata 3).
    """
    folder = Path(out_dir) / "resumes"
    folder.mkdir(parents=True, exist_ok=True)
    index = 1
    while (folder / f"run_resume-{index:03d}.json").exists():
        index += 1
    return folder / f"run_resume-{index:03d}.json"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
# Flags mirrored from `train_smax_2s3z_independent.py` (same names, same meaning).  The frozen
# trainer owns their defaults and the `--tiny` mapping; `parse_args` only overrides what the CLI
# actually passes, so the budget recipe stays identical to the existing run.
SHARED_FLAGS = (
    "map", "num_envs", "rollout_length", "updates", "segment_updates", "ppo_epochs",
    "num_minibatches", "lr", "anneal_lr", "clip_eps", "ent_coef", "vf_coef", "gamma",
    "gae_lambda", "max_grad_norm", "fc_dim_size", "gru_hidden_dim", "eval_episodes", "seed",
)


def _add_shared_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--map", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--segment-updates", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--vf-coef", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--fc-dim-size", type=int, default=None)
    parser.add_argument("--gru-hidden-dim", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seeds", default=None,
                        help="comma-separated FIXED monitor seeds, identical at every segment; "
                             "must be disjoint from the measurement set A and the test set C")
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-observer oracle-gate trainer (profile vs placeholder vs onehot arm)."
    )
    _add_shared_flags(parser)
    parser.add_argument("--arm", choices=ARMS, required=True,
                        help="profile: obs++z(12, measured profile table); placeholder: "
                             "obs++zeros(12); onehot: obs++the battle combination's 12-dim "
                             "per-identity stage-index one-hot (oracle-level stage label)")
    parser.add_argument("--profiles", default=None,
                        help="profile table (default: results/oracle_gate/profiles.json, falling "
                             "back to results/oracle_gate/fixtures/profiles_fixture.json)")
    parser.add_argument("--partner-run-dir", default=str(DEFAULT_PARTNER_RUN_DIR),
                        help="run directory holding the frozen partner checkpoints")
    parser.add_argument("--partner-stages", default=",".join(str(s) for s in STAGE_UPDATES),
                        help="partner stages (update counts); the contract fixes 50,600,1250")
    parser.add_argument("--segment-eval-combos", type=int, default=None,
                        help="combinations used for the per-segment monitor (default 9; tiny 2). "
                             "This is a SUBSET of the 81-combination gate protocol")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every-segments", type=int, default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--tiny", action="store_true",
                        help="small smoke scale (2 envs, 8 rollout steps, 4 updates)")
    args = parser.parse_args(argv)

    base = base_train.parse_args(["--tiny"] if args.tiny else [])
    for name in SHARED_FLAGS:
        value = getattr(args, name)
        if value is not None:
            setattr(base, name, value)
    if args.eval_seed is not None:
        base.eval_seeds = [int(args.eval_seed)]
    elif args.eval_seeds is not None:
        try:
            base.eval_seeds = [int(x) for x in str(args.eval_seeds).replace(" ", "").split(",") if x]
        except ValueError:
            parser.error("--eval-seeds must be comma-separated integers")
        if not base.eval_seeds:
            parser.error("--eval-seeds must contain at least one seed")

    merged = argparse.Namespace(**vars(base))
    merged.arm = args.arm
    merged.profiles = args.profiles
    merged.partner_run_dir = args.partner_run_dir
    merged.partner_stages = tuple(
        int(x) for x in str(args.partner_stages).replace(" ", "").split(",") if x
    )
    merged.segment_eval_combos = (args.segment_eval_combos if args.segment_eval_combos is not None
                                  else (2 if args.tiny else 9))
    merged.run_name = args.run_name
    merged.save_dir = args.save_dir
    merged.save_every_segments = args.save_every_segments
    merged.resume = args.resume
    merged.tiny = args.tiny

    if merged.updates % merged.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    if merged.num_envs % merged.num_minibatches:
        parser.error("--num-envs must be divisible by --num-minibatches")
    if merged.save_every_segments < 1:
        parser.error("--save-every-segments must be >= 1")
    if merged.segment_eval_combos < 1:
        parser.error("--segment-eval-combos must be >= 1")
    if merged.partner_stages != tuple(STAGE_UPDATES):
        parser.error(f"the contract fixes the partner stage set to {list(STAGE_UPDATES)}")
    leaked = set(int(s) for s in merged.eval_seeds) & (set(MEASURE_SEEDS_A) | set(TEST_SEEDS_C))
    if leaked:
        parser.error(
            f"monitor seed set E must be disjoint from the measurement set A {list(MEASURE_SEEDS_A)} "
            f"and the test set C {list(TEST_SEEDS_C)}; got {sorted(leaked)}"
        )
    return merged


def monitoring_combos(subset: int) -> list[tuple[int, ...]]:
    """Combinations used by the per-segment monitor (a documented *subset* of the 81)."""
    return enumerate_combos(subset)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or f"observer_gate_{args.arm}"
    out_dir = run_dir_for(args.save_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = base_train.build_config(args)
    partner_dir = Path(args.partner_run_dir)
    partner_dir = partner_dir if partner_dir.is_absolute() else (HERE / partner_dir)
    partner_stack = load_partner_stack(partner_dir, args.partner_stages)

    table_note = "not loaded (placeholder arm: the z channel is a constant zero)"
    table_meta = None
    z_table = np.zeros((len(PARTNER_IDENTITIES), len(STAGE_UPDATES), len(METRIC_NAMES)), np.float32)
    if args.arm == "profile" or args.profiles is not None:
        # Only the profile arm *uses* the table; the other two arms may still validate one when
        # `--profiles` is given (handy for the tri-arm evaluation), but their channel never reads it.
        table, source, is_fixture = load_profile_table(args.profiles)
        table_note = f"{source}{' (FIXTURE - replace with the measured table)' if is_fixture else ''}"
        table_meta = {"path": str(source), "sha256": sha256_file(source), "is_fixture": is_fixture}
        if args.arm == "profile":
            z_table = z_table_from_profile_table(table)
        else:
            table_meta["used_for_input"] = False
        print(f"[profile table] {table_note}", flush=True)
    if args.arm == "onehot":
        # No profile table at all: the channel is the battle combination's stage one-hot (see the
        # module docstring) - an oracle-level stage label, not a measured profile.  The trainer
        # derives the same table from the arm name; computing it here keeps `main` self-contained.
        z_table = onehot_z_table()
        table_note = ("not used (onehot arm: z is the per-identity stage-index one-hot of the "
                      "battle's combination - an oracle-level stage label, not a measured profile)")

    trainer = ObserverGateTrainer(config, partner_stack, z_table, arm=args.arm)
    # From here on the trainer's config is canonical: it carries the derived input dimension, the
    # real horizon and the frozen partner/stream identity recorded in run.json and checkpoints.
    config = trainer.config
    runner = trainer.init_runner(jax.random.PRNGKey(args.seed))
    resumed_from = None
    resumed_from_run_json = None
    resumed_from_update = None
    if args.resume:
        ckpt_path = Path(args.resume)
        ckpt_path = ckpt_path if ckpt_path.is_absolute() else (HERE / ckpt_path)
        meta = read_checkpoint_meta(ckpt_path)
        stored = meta["config"]
        for key in ("map_name", "num_envs", "rollout_length", "obs_dim", "action_dim",
                    "gru_hidden_dim", "fc_dim_size", "num_agents", "seed", "env_kwargs",
                    "observer_input_dim"):
            if stored.get(key) != config.get(key) and stored.get(key) is not None:
                raise SystemExit(f"--resume config mismatch on {key}: {stored.get(key)} != "
                                 f"{config.get(key)}")
        if meta.get("arm") and meta["arm"] != args.arm:
            raise SystemExit(f"--resume arm mismatch: checkpoint is {meta['arm']}, not {args.arm}")
        runner = load_checkpoint(ckpt_path, runner)
        resumed_from = str(ckpt_path)
        resumed_from_update = int(runner.update_count)
        run_json = out_dir / "run.json"
        resumed_from_run_json = str(run_json) if run_json.is_file() else None
        print(f"[resume] loaded {ckpt_path} (update_count={resumed_from_update}, "
              f"segment={meta.get('segment')}); continuing the run documented by "
              f"{resumed_from_run_json}", flush=True)

    # A fresh run documents itself in `run.json`; a resumed run must not touch that file, so it
    # writes its own metadata under `resumes/` and records what it continued from (errata 3).
    metadata_path = resume_metadata_path(out_dir) if args.resume else (out_dir / "run.json")
    metadata = {
        "kind": "smax_2s3z_oracle_gate_observer",
        "contract": "MainSearch/explore/ORACLE_GATE_CONTRACT.md (v1)",
        "arm": args.arm,
        "argv": sys.argv[1:],
        "config": config,
        "z_channel": Z_CHANNEL_BY_ARM[args.arm],
        "z_semantics": Z_SEMANTICS_BY_ARM[args.arm],
        "profile_table": table_meta,
        "profile_table_note": table_note,
        "partner_run_dir": str(partner_dir),
        "partner_stages": list(args.partner_stages),
        "partner_identities": list(PARTNER_IDENTITIES),
        "observer_agent": "ally_0",
        "stream_seeds": {
            "partner_seed": PARTNER_SEED, "reset_seed": RESET_SEED,
            "env_step_seed": ENV_STEP_SEED, "partner_action_seed": PARTNER_ACTION_SEED,
            "source": ("contract §2 + errata 1: streams are functions of (slot, episode_index, "
                       "in-episode step) only - the in-episode step is the persistent ep_length, "
                       "never the roll-out step"),
        },
        "monitor": {
            "seeds": list(args.eval_seeds),
            "combos": len(monitoring_combos(args.segment_eval_combos)),
            "note": ("per-segment monitor only: a SUBSET of the 81 combinations, on monitor seed "
                     "set E (disjoint from the measurement set A and the test set C).  It is NOT "
                     "the §4 gate evaluation"),
        },
        "params": {"observer": int(sum(x.size for x in jax.tree.leaves(runner.params[0]))),
                   "partner_per_identity": int(sum(
                       x.size for x in jax.tree.leaves(partner_stack[0][0])))},
        "partner_action_rule": (
            "argmax (errata 2): the partners act by argmax in training AND in evaluation, the same "
            "rule measure_partner_profiles.py used; only the observer samples.  The contract §2 "
            "partner-action stream is still computed and recorded in the diagnostics."
        ),
        "in_episode_step_index": (
            "errata 1: env_step_key / partner_action_key are indexed by the persistent in-episode "
            "step ep_length, never by the roll-out step"
        ),
        "resumed_from": resumed_from,
        "resumed_from_run_json": resumed_from_run_json,
        "resumed_from_update_count": resumed_from_update,
        "resume_policy": (
            "errata 3: a resumed run writes its own metadata file under resumes/ and never "
            "rewrites run.json; failed runs keep their run.json and their logs"
        ),
        "write_metadata_path": str(metadata_path),
        "save_dir": str(out_dir),
        "segments": [],
        "wall_seconds_total": None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"[oracle gate] arm={args.arm} observer=ally_0 partners={list(PARTNER_IDENTITIES)} "
          f"frozen; input_dim={config['observer_input_dim']} "
          f"(obs {config['obs_dim']} + z {PROFILE_DIM})", flush=True)
    print(f"  z semantics: {Z_SEMANTICS_BY_ARM[args.arm]}", flush=True)
    print(f"  z channel  : {Z_CHANNEL_BY_ARM[args.arm]}"
          f"{' (the per-episode combination z from the contract §1 table)' if args.arm == 'profile' else ''}"
          f"{' (constant by definition)' if args.arm == 'placeholder' else ''}"
          f"{' (derived from the combination, no table read)' if args.arm == 'onehot' else ''}",
          flush=True)
    print("  partners   : argmax (errata 2) - the same rule as the profile measurement; only the "
          "observer samples", flush=True)
    print(f"  metadata   : {metadata_path}"
          f"{'  (resume: run.json is left untouched)' if args.resume else '  (fresh run)'}",
          flush=True)
    print(f"  partner stack: {partner_dir} stages {list(args.partner_stages)}", flush=True)
    print(f"  budget: updates={args.updates} segment_updates={args.segment_updates} "
          f"num_envs={config['num_envs']} rollout={config['rollout_length']} "
          f"-> {args.updates * config['num_envs'] * config['rollout_length']} env steps", flush=True)

    monitor_combos = monitoring_combos(args.segment_eval_combos)
    metadata["monitor"]["combos_list"] = [combo_label(c) for c in monitor_combos]
    monitor_subset = len(monitor_combos) < NUM_COMBOS
    if monitor_subset:
        print(f"  monitor: {len(monitor_combos)}/{NUM_COMBOS} combinations "
              f"{[combo_label(c) for c in monitor_combos]} on seeds {list(args.eval_seeds)} "
              f"(SUBSET of the §4 protocol)", flush=True)

    updates_done = int(runner.update_count)
    if resumed_from is None:
        initial = save_checkpoint(
            out_dir / "checkpoint_00000000.bin", runner,
            {"update_count": 0, "segment": 0, "config": config, "argv": sys.argv[1:],
             "kind": "initial_state", "arm": args.arm},
        )
        metadata["initial_checkpoint"] = initial.name
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[initial state] saved {initial.name} (update_count=0, untrained observer; the "
              f"frozen partners come from {partner_dir.name})", flush=True)
    if updates_done % args.segment_updates:
        raise SystemExit(f"resumed update_count {updates_done} is not a multiple of "
                         f"--segment-updates {args.segment_updates}")
    num_segments = args.updates // args.segment_updates
    started = time.perf_counter()

    for segment in range(updates_done // args.segment_updates, num_segments):
        lr_at_start = trainer.current_lr(int(runner.update_count))
        segment_started = time.perf_counter()
        runner, records = trainer.train_updates(runner, args.segment_updates)
        jax.block_until_ready(runner.update_count)
        train_seconds = time.perf_counter() - segment_started
        metrics = {k: np.asarray(jax.device_get(v)) for k, v in records.items()}

        # The monitor uses the arm's own channel: `zero` is the placeholder arm (whose table is a
        # constant zero anyway); the profile and onehot arms both go through their table.
        z_mode = "zero" if args.arm == "placeholder" else "table"
        battle = trainer.eval_battles(runner.params[0], monitor_combos, args.eval_seeds,
                                      z_mode=z_mode)
        primary = float(battle["return"].mean())      # primary metric: mean team return
        secondary = float(battle["won"].sum())        # auxiliary: wins
        battles = battle["return"].size
        cumulative = int(runner.update_count) * config["num_envs"] * config["rollout_length"]
        env_steps = args.segment_updates * config["num_envs"] * config["rollout_length"]
        lr_now = trainer.current_lr(int(runner.update_count))
        record = {
            "segment": segment + 1,
            "updates_done": int(runner.update_count),
            "cumulative_env_steps": cumulative,
            "train_seconds": round(train_seconds, 2),
            "env_steps_per_second": round(env_steps / train_seconds, 1),
            "lr_at_segment_start": lr_at_start,
            "lr": lr_now,
            "train_finished_episodes": float(metrics["train_finished_episodes"].sum()),
            "train_won_sum": float(metrics["train_won_sum"].sum()),
            "train_return_sum": float(metrics["train_return_sum"].sum()),
            "loss_total": [float(v) for v in metrics["total_loss"].mean(0)],
            "entropy": [float(v) for v in metrics["entropy"].mean(0)],
            "approx_kl": [float(v) for v in metrics["approx_kl"].mean(0)],
            "clip_frac": [float(v) for v in metrics["clip_frac"].mean(0)],
            "monitor": {
                "seeds": list(args.eval_seeds),
                "combos": [combo_label(c) for c in monitor_combos],
                "is_subset_of_81": monitor_subset,
                "battles": battles,
                "primary_mean_team_return": primary,
                "secondary_wins": secondary,
                "secondary_win_rate": secondary / battles,
                "per_combo_mean_return": [float(x) for x in battle["return"].mean(1)],
                "per_combo_wins": [int(x) for x in battle["won"].sum(1)],
            },
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        if (segment + 1) % args.save_every_segments == 0 or (segment + 1) == num_segments:
            ckpt = save_checkpoint(
                out_dir / f"checkpoint_{int(runner.update_count):08d}.bin", runner,
                {"update_count": int(runner.update_count), "segment": segment + 1,
                 "config": config, "argv": sys.argv[1:], "arm": args.arm,
                 "profile_table": table_meta},
            )
            record["checkpoint"] = ckpt.name
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        finished = record["train_finished_episodes"]
        train_win = (record["train_won_sum"] / finished) if finished else float("nan")
        train_return = (record["train_return_sum"] / finished) if finished else float("nan")
        print(
            f"[segment {segment + 1}/{num_segments}] updates={int(runner.update_count)}/"
            f"{args.updates} env_steps={cumulative} train={train_seconds:.1f}s "
            f"({record['env_steps_per_second']:.0f} steps/s) lr={lr_now:.3e}"
            f"{'  ckpt=' + record['checkpoint'] if 'checkpoint' in record else ''}", flush=True,
        )
        print(f"  train  : finished_episodes={finished:.0f} won={record['train_won_sum']:.0f} "
              f"win_rate={train_win:.3f} mean_return={train_return:.2f} "
              f"loss={[round(v, 3) for v in record['loss_total']]} "
              f"entropy={[round(v, 3) for v in record['entropy']]}", flush=True)
        print(f"  monitor: [PRIMARY] mean team return={primary:.3f} over {battles} battles "
              f"({len(monitor_combos)} combos x {len(args.eval_seeds)} seeds)"
              f"{' SUBSET' if monitor_subset else ''} | [aux] wins={secondary:.0f}"
              f"/{battles} ({secondary / battles:.3f})", flush=True)
        print("  monitor per combo: " + "  ".join(
            f"{combo_label(c)}:{int(w)}/{len(args.eval_seeds)}(ret {r:.2f})"
            for c, w, r in zip(monitor_combos, battle["won"].sum(1), battle["return"].mean(1))
        ), flush=True)

    print(
        f"[oracle gate {args.arm}] finished {int(runner.update_count)} updates "
        f"({int(runner.update_count) * config['num_envs'] * config['rollout_length']} env steps) "
        f"in {metadata['wall_seconds_total']}s; wrote {metadata_path}"
        f"{' (run.json untouched)' if args.resume else ''}", flush=True,
    )
    print("  note: the per-segment monitor is not the §4 gate comparison; that is "
          "evaluate_observer_gate.py on a fixed test set, and even 'no gain' there is not "
          "evidence that partner information is worthless.", flush=True)


if __name__ == "__main__":
    main()
