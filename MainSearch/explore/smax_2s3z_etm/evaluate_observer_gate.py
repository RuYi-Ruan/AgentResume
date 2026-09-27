"""Oracle-gate evaluation: the §4 paired battle protocol and the §5 profile-swap check.

Frozen interface: `MainSearch/explore/ORACLE_GATE_CONTRACT.md` (v1).  Run with the repository's
absolute interpreter path, e.g.

  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      evaluate_observer_gate.py --tiny --combos-subset 2

§4 (gate evaluation)
--------------------
All `3^4 = 81` partner-stage combinations x the fixed test seed set `C = [3234,3235,3236,3237]`
= **324 battles per arm**.  All three arms are evaluated on **exactly the same 324
(combination, situation) pairs**:

* primary metric  = mean team return (mean over the 324 battles);
* secondary metric = number of won battles (auxiliary only, never the decision).

The pairing unit is `(partner-stage combination, test situation)`: an effect is the per-pair
difference of two arms, and its uncertainty is a **bootstrap over those 324 pairs**.  That
bootstrap describes *this fixed set of situations*; it is **not** a sample from SMAX as a whole,
and the battles are **not** a re-verification of the training seed (contract §4, §6).

Three arms, all playing the same partners in the same battles; the only difference is the
observer's second input channel (12 dims in each arm, contracts §1 + ticket 2026-09-27):

* `onehot`      - the battle's partner-stage combination as a per-identity stage-index one-hot
                  (identity order `[1,2,3,4]`, 3 dims each, stage order `[u50,u600,u1250]`);
* `placeholder` - a constant zero vector;
* `profile`     - the battle combination's `z` read from `results/oracle_gate/profiles.json`.

**The primary comparison is `onehot - placeholder`** - "does the observer gain anything from being
told which partner policies it faces at all?".  `onehot` is an **oracle-level stage label**: which
of the 12 candidate parameter sets the partner is drawn to play, and nothing else.  It is *not* a
measured behaviour profile (that is the `profile` arm) and *not* the null channel (that is
`placeholder`), and it is *not* a claim about the partner's team-independent ability.  The
remaining pairs (`onehot - profile`, `profile - placeholder`) are reported alongside it: the first
says whether the extra content of the measured profile beats the bare stage label, the second is
the contract's original comparison.

`profile`'s `z` itself is read from `results/oracle_gate/profiles.json` and means: **the partner's
three-metric behaviour profile as measured under the fixed reference team (every other identity
kept at its u1250 parameters) on the fixed measurement situations A** (contract §1 v2: real hit
events, weapon cooldown, sub-step granularity - measured by `measure_partner_profiles.py`, never
re-derived here), normalized by the frozen 12-candidate mean/std.  It is a profile *under that
reference team*, not a team-independent statement about the partner's ability.

§5 (z-swap check, matched states)
---------------------------------
On one batch of matched states (same battles, same trajectory positions) an arm's observer is run
twice: once with the correct `z` and once with a **stale same-identity z** - the partner is
inspected in a combination where it really plays its **late** parameters (`u1250`) and the z dims
handed to the observer for that same identity are replaced by the *same identity's* **early**
(`u50`) description.  For the `profile` arm that stale description is its measured `u50` profile;
for the `onehot` arm it is the `u50` one-hot (the one-hot dim that is `1` moves from the late stage
to the early stage).  Both forward passes are given **the same GRU hidden states** - the hstates
are injected explicitly and compared leaf-by-leaf - so any difference is attributable to the z
channel alone.  The check is run for the `onehot` and `profile` arms; it is *skipped* for
`placeholder`, whose channel is a constant, so no swap could change any input.

Reported: action-distribution TV and JS, argmax switching rate.

**Interpretation constraint (contract §5, §6)**: a changed action distribution only shows that the
input *affects* the decision.  It says nothing about whether the change is *beneficial*.  Nothing
here is a causal claim, and the battle counts are not independent samples.

Errata (review ticket 2026-09-27; overrides the earlier implementation)
----------------------------------------------------------------------
1. Stale, not future.  The swap default used to start from the lexicographically first combination
   (`[u50,u50,u50,u50]`) and replace that partner's `u50` profile by its `u1250` one - i.e. it gave
   the observer a profile of the *future* while the partner played its early parameters.  The
   default now comes from `default_matched_combo`, a combination in which the inspected partner
   really uses its **latest** stage, and `resolve_swap_target` guarantees
   `param_stage > profile_stage` (stale); the automatic flip changes *which* identity is inspected,
   never the direction.
2. Partner actions are **argmax** in the evaluation too (`G.partner_action_argmax`), the rule the
   profile measurement used; the observer plays argmax in a battle, while in the matched-state
   rollouts the recorded position is what matters.
3. Every battle runs to its **real termination** (`done["__all__"]`), not to a fixed step count:
   the local SMAX environment flags `done` before incrementing its own step counter, so a battle
   that ends on the time limit needs `max_steps + 1` calls.  `eval_step_limit` is a loop guard and
   a battle still running when it runs out is reported as a **truncated failure**, never as a
   normal result.
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

import train_smax_2s3z_independent as base_train  # noqa: E402
import train_observer_gate as G  # noqa: E402

DEFAULT_PROFILE_RUN = HERE / "results" / "oracle_gate" / "observer_gate_profile"
DEFAULT_PLACEHOLDER_RUN = HERE / "results" / "oracle_gate" / "observer_gate_placeholder"
DEFAULT_ONEHOT_RUN = HERE / "results" / "oracle_gate" / "observer_gate_onehot"
DEFAULT_OUT = HERE / "results" / "oracle_gate" / "observer_gate_eval.json"

# The three arms of the trainer, and the pairs compared on the shared (combination, situation)
# grid.  The FIRST pair is this ticket's primary comparison - "is partner information worth
# anything at all?" - measured against the null channel; the other pairs are reported alongside it.
EVAL_ARMS = ("onehot", "placeholder", "profile")
COMPARISON_PAIRS = (
    ("onehot", "placeholder"),
    ("onehot", "profile"),
    ("profile", "placeholder"),
)
PRIMARY_PAIR = COMPARISON_PAIRS[0]
# §5 runs on the arms whose channel carries information (the placeholder channel is a constant, so
# a stale swap there would compare an input with itself).
SWAP_ARMS = ("onehot", "profile")

DISCLAIMER_FIXED_SET = (
    "The bootstrap describes THIS FIXED SET of situations (the enumerated (partner-stage "
    "combination, test situation) pairs).  It is NOT a random sample from SMAX as a whole and "
    "says nothing about other maps, partner sets, budgets or seeds."
)
DISCLAIMER_NOT_RECHECK = (
    "These battles are NOT a re-verification of the training seed: they are a separate fixed test "
    "set (seed set C).  Replaying them adds no independent sample size to the training run."
)
DISCLAIMER_USAGE = (
    "Input affects decisions != the change is beneficial: the z-swap check shows only that the z "
    "channel reaches the policy, never that acting on it helps."
)
DISCLAIMER_NULL = (
    "Even if the profile table is separable and the interface works, a null result supports only "
    "'no gain found in this configuration', never 'partner information has no value'."
)
# The three channels' meanings live in the trainer (single source of truth): see
# `train_observer_gate.Z_SEMANTICS_BY_ARM` and `Z_CHANNEL_BY_ARM`.
Z_SEMANTICS_BY_ARM = G.Z_SEMANTICS_BY_ARM
Z_CHANNEL_BY_ARM = G.Z_CHANNEL_BY_ARM

DISCLAIMERS = (DISCLAIMER_FIXED_SET, DISCLAIMER_NOT_RECHECK, DISCLAIMER_USAGE, DISCLAIMER_NULL)


# --------------------------------------------------------------------------------------------
# Run / checkpoint discovery and configuration
# --------------------------------------------------------------------------------------------
def resolve_dir(value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else (HERE / path)
    return path.resolve()


def discover_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for entry in sorted(run_dir.iterdir()):
        if entry.is_file() and entry.name.startswith("checkpoint_") and entry.suffix == ".bin":
            try:
                found.append((int(entry.stem.split("_")[1]), entry.resolve()))
            except (IndexError, ValueError):
                continue
    if not found:
        raise SystemExit(f"no checkpoints in {run_dir}")
    return sorted(found)


def resolve_checkpoint(run_dir: Path, value: str | None) -> tuple[int, Path]:
    """`value` = update count, filename, or path; default = latest checkpoint in `run_dir`."""
    checkpoints = discover_checkpoints(run_dir)
    if value is None:
        return checkpoints[-1]
    text = str(value)
    if text.isdigit():
        for update, path in checkpoints:
            if update == int(text):
                return update, path
        raise SystemExit(f"{run_dir}: no checkpoint at update {text}")
    path = Path(text)
    path = path if path.is_absolute() else (HERE / path)
    if not path.is_file():
        path = run_dir / text
    if not path.is_file():
        raise SystemExit(f"cannot resolve checkpoint {value!r} in {run_dir}")
    match = [c for c in checkpoints if c[1] == path.resolve()]
    return (match[0] if match else (-1, path.resolve()))


def load_run_config(run_dir: Path, checkpoint: Path, fallback_args=None) -> dict:
    """The run's own config (`run.json`, else the checkpoint metadata), else the CLI defaults."""
    candidates = []
    run_json = run_dir / "run.json"
    if run_json.is_file():
        try:
            candidates.append(json.loads(run_json.read_text(encoding="utf-8")).get("config"))
        except (json.JSONDecodeError, OSError):
            pass
    try:
        candidates.append(G.read_checkpoint_meta(checkpoint).get("config"))
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    for config in candidates:
        if isinstance(config, dict) and {"map_name", "gru_hidden_dim", "fc_dim_size",
                                         "observer_input_dim"} <= set(config):
            return config
    if fallback_args is None:
        raise SystemExit(f"no usable config in {run_dir}; pass a checkpoint with metadata")
    return base_train.build_config(fallback_args)


CONFIG_KEYS = ("map_name", "obs_dim", "action_dim", "num_agents", "gru_hidden_dim",
               "fc_dim_size", "observer_input_dim", "profile_dim", "horizon", "env_kwargs")


def check_arm_compatibility(config: dict, other: dict, other_name: str) -> None:
    """Any two arms being compared must differ in the input channel only - everything else must
    match."""
    for key in CONFIG_KEYS:
        if config.get(key) != other.get(key):
            raise SystemExit(f"arm mismatch on {key}: {config.get(key)} != "
                             f"{other.get(key)} ({other_name})")


# --------------------------------------------------------------------------------------------
# §4: pairing and bootstrap
# --------------------------------------------------------------------------------------------
def paired_differences(profile: np.ndarray, placeholder: np.ndarray) -> np.ndarray:
    """Elementwise `profile - placeholder` over the *same* (combination, situation) grid."""
    profile = np.asarray(profile, dtype=np.float64)
    placeholder = np.asarray(placeholder, dtype=np.float64)
    if profile.shape != placeholder.shape:
        raise SystemExit(f"paired comparison needs the same grid, got {profile.shape} vs "
                         f"{placeholder.shape}")
    return profile - placeholder


def bootstrap_mean(values: np.ndarray, samples: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the mean of `values` (resampling the pairs with replacement)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(int(samples), values.size))
    means = values[draws].mean(axis=1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd_of_pairs": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "bootstrap_mean": float(means.mean()),
        "bootstrap_sd": float(means.std(ddof=1)),
        "bootstrap_ci95_low": float(np.percentile(means, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(means, 97.5)),
        "meaning": ("spread of the mean over resamples of THIS fixed pair set; not a sample of "
                    "SMAX as a whole"),
    }


# --------------------------------------------------------------------------------------------
# §5: matched states and the profile swap
# --------------------------------------------------------------------------------------------
def collect_matched_states(trainer, observer_params, combo: Sequence[int], seeds, steps: int = 8):
    """Record `(observer input, GRU hidden state, available actions)` at matched positions.

    Short rollouts of the *correct-z* observer (the arm's own channel) on `combo` x `seeds`; every
    recorded position keeps the hidden state the observer actually had when it saw that input, so
    the later swap can inject identical hidden states into both forward passes.  The observer plays
    argmax and the partners play argmax too (errata 2, as in training and in the gate battles).
    """
    partners = tuple(trainer.partner_stack[j][int(combo[j])]
                     for j in range(len(G.PARTNER_IDENTITIES)))
    z = trainer.z_vector(jnp.asarray(np.asarray(combo, np.int32)))
    seeds = jnp.asarray(np.asarray(seeds, dtype=np.int32))
    env, base_env, network, config = trainer.env, trainer.base_env, trainer.network, trainer.config
    agents = env.agents
    gru = config["gru_hidden_dim"]
    zero_done = jnp.zeros((1, 1), dtype=bool)

    def _episode(observer_params, partner_params, z_vector, seed):
        obs, env_state = env.reset(G.stream_key(G.EVAL_RESET_SEED, seed))
        hidden = base_train.ScannedRNN.initialize_carry(1, gru)
        partner_hidden = tuple(base_train.ScannedRNN.initialize_carry(1, gru)
                               for _ in G.PARTNER_IDENTITIES)

        def _step(carry, step):
            obs, env_state, hidden, partner_hidden = carry
            obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
            avail = base_env.get_avail_actions(env_state.env_state)
            avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)
            x_observer = jnp.concatenate([obs_stack[0], z_vector], axis=-1)   # (obs_dim + 12,)

            h_observer, pi_observer, _ = network.apply(
                observer_params, hidden, (x_observer[None, None], zero_done,
                                          avail_stack[0][None])
            )
            logits = jnp.where(avail_stack[0][None] > 0, pi_observer.logits, -1e9)
            actions = [jnp.argmax(logits, axis=-1)[0, 0]]
            new_partner_hidden = []
            for j, ident in enumerate(G.PARTNER_IDENTITIES):
                h_partner, pi_partner, _ = network.apply(
                    partner_params[j], partner_hidden[j],
                    (obs_stack[ident][None, None], zero_done, avail_stack[ident][None]),
                )
                # Errata 2: the partners act by argmax, exactly as in training and in the gate
                # battles (and as in the profile measurement).
                actions.append(G.partner_action_argmax(pi_partner.logits)[0, 0])
                new_partner_hidden.append(h_partner)
            env_act = {agent: actions[i] for i, agent in enumerate(agents)}
            new_obs, new_env_state, _, _, _ = env.step(
                G.stream_key(G.EVAL_STEP_SEED, seed, step), env_state, env_act
            )
            # recorded BEFORE the step: the input the observer saw and the hidden state it had.
            record = (x_observer, hidden[0], avail_stack[0])
            return (new_obs, new_env_state, h_observer, tuple(new_partner_hidden)), record

        _, records = jax.lax.scan(
            _step, (obs, env_state, hidden, partner_hidden), jnp.arange(steps)
        )
        return records

    def _run(observer_params, partner_params, z_vector, seed_batch):
        return jax.vmap(
            lambda seed: _episode(observer_params, partner_params, z_vector, seed)
        )(seed_batch)

    obs, hstate, avail = jax.tree.map(
        np.asarray, jax.jit(_run)(observer_params, partners, z, seeds)
    )   # each (S, steps, ...)
    obs = np.transpose(obs, (1, 0, 2)).reshape(-1, obs.shape[-1])
    hstate = np.transpose(hstate, (1, 0, 2)).reshape(-1, hstate.shape[-1])
    avail = np.transpose(avail, (1, 0, 2)).reshape(-1, avail.shape[-1])
    step_index = np.repeat(np.arange(steps), len(np.asarray(seeds)))
    seed_index = np.tile(np.asarray(seeds), steps)
    return {
        "obs": obs,
        "hstate": hstate,
        "avail": avail,
        "z": np.asarray(z),
        "combo": [int(s) for s in combo],
        "seeds": [int(s) for s in np.asarray(seeds)],
        "steps": int(steps),
        "step_index": step_index,
        "seed_index": seed_index,
        "n_states": int(obs.shape[0]),
    }


def _logits_for(trainer, params, hstate, obs, avail):
    """One forward pass of the observer over a batch of matched states (no GRU resets)."""
    dones = jnp.zeros((1, obs.shape[0]), dtype=bool)
    _, pi, _ = trainer.network.apply(params, hstate, (obs[None], dones, avail))
    return pi.logits[0]


def default_matched_combo(combos: Sequence[Sequence[int]],
                          swap_identity: int = G.PARTNER_IDENTITIES[0]) -> Sequence[int]:
    """The combination the §5 swap uses by default: the inspected partner plays its **latest** stage.

    The check is "the partner really plays its late parameters while the observer is told its early
    profile", so the default must start from a combination in which that partner is at the latest
    stage.  (The lexicographic first combination is all-`u50`: swapping there can only produce a
    *future* profile - the erratum 1 mistake.)  Ties prefer the combination that is later overall.
    """
    j = list(G.PARTNER_IDENTITIES).index(int(swap_identity))
    return max(combos, key=lambda c: (int(c[j]), sum(int(s) for s in c), tuple(int(s) for s in c)))


def resolve_swap_target(combo: Sequence[int], swap_identity: int = G.PARTNER_IDENTITIES[0],
                        stale_stage: int = 0) -> tuple[int, int, int]:
    """`(identity, param_stage, profile_stage)` of a **stale** swap: `param_stage > profile_stage`.

    `param_stage` is the stage the inspected partner really plays in this combination; the observer
    is handed that same identity's `stale_stage` (early) profile instead.  The preferred identity is
    `swap_identity`; when it does not play a stage later than `stale_stage` the automatic flip picks
    an identity that does (it changes *which* partner is inspected, never the direction).  A
    combination in which no identity plays a later stage cannot host a stale swap at all and is an
    error instead of silently becoming a future-profile comparison.
    """
    identities = list(G.PARTNER_IDENTITIES)
    stages = [int(s) for s in combo]
    preferred = identities.index(int(swap_identity)) if int(swap_identity) in identities else 0
    order = [preferred] + [j for j in range(len(identities)) if j != preferred]
    for j in order:
        if stages[j] > int(stale_stage):
            return identities[j], stages[j], int(stale_stage)
    latest = max(order, key=lambda k: stages[k])       # ties keep the preferred identity
    earlier = [s for s in range(len(G.STAGE_UPDATES)) if s < stages[latest]]
    if not earlier:
        raise SystemExit(
            f"combination {G.combo_label(combo)} cannot host a stale z swap: no identity "
            f"plays a stage later than {G.STAGE_LABELS[int(stale_stage)]}, and a swap in the other "
            f"direction would compare the observer against a description of the FUTURE (errata 1)."
        )
    return identities[latest], stages[latest], int(min(earlier))


def swap_analysis(trainer, observer_params, matched: dict, combo: Sequence[int],
                  swap_identity: int = G.PARTNER_IDENTITIES[0],
                  stale_stage: int = 0) -> dict:
    """Compare the observer's action distribution under the correct vs a stale same-identity z.

    The inspected partner (see `resolve_swap_target`) plays its late parameters while the observer's
    z dims for that identity come either from the stage it really plays (`param_stage`) or from the
    same identity's early `profile_stage` - a *stale* description.  `trainer` supplies the arm's own
    z table: for the `profile` arm the stale dims are that identity's measured `u50` profile, for
    the `onehot` arm they are its `u50` one-hot (the hot index moves from the late stage to the
    early one).  The two forward passes receive **identical GRU hidden states** (injected explicitly
    and compared leaf-by-leaf).  The observation part of the input is identical too; only the 3 z
    dims of that partner differ.  A third pass with a perturbed hidden state is the negative control
    that proves the hidden state is actually consumed by the forward pass.
    """
    obs_dim = int(trainer.config["obs_dim"])
    table = np.asarray(trainer.z_table, dtype=np.float64)
    n = int(matched["obs"].shape[0])
    combo = [int(s) for s in combo]
    identity, param_stage, profile_stage = resolve_swap_target(combo, swap_identity, stale_stage)
    j = G.PARTNER_IDENTITIES.index(int(identity))
    stale = int(profile_stage)
    stale_combo = list(combo)
    stale_combo[j] = stale

    z_correct = G.z_vector_from_table(table, combo)
    z_stale = G.z_vector_from_table(table, stale_combo)
    z_recorded = np.asarray(matched["obs"][:, obs_dim:])
    z_matches_recording = bool(np.allclose(z_recorded, np.tile(z_correct, (n, 1)), atol=1e-6))

    obs_shared = np.asarray(matched["obs"][:, :obs_dim])
    obs_correct = np.concatenate([obs_shared, np.tile(z_correct, (n, 1))], axis=1)
    obs_stale = np.concatenate([obs_shared, np.tile(z_stale, (n, 1))], axis=1)
    inputs_differ_only_in_z = bool(np.array_equal(obs_correct[:, :obs_dim],
                                                  obs_stale[:, :obs_dim]))

    h_a = jnp.asarray(matched["hstate"])
    h_b = jnp.array(matched["hstate"])          # a separate array with the same values
    h_perturbed = 2.0 * h_a + jnp.float32(0.1)  # negative control: the forward must consume hstate
    hstates_equal_before = G.tree_bitwise_equal(h_a, h_b)

    avail = jnp.asarray(matched["avail"])
    logits_correct = _logits_for(trainer, observer_params, h_a, jnp.asarray(obs_correct), avail)
    logits_stale = _logits_for(trainer, observer_params, h_b, jnp.asarray(obs_stale), avail)
    logits_perturbed = _logits_for(trainer, observer_params, h_perturbed,
                                   jnp.asarray(obs_correct), avail)
    hstates_equal_after = G.tree_bitwise_equal(h_a, h_b)

    log_p = jax.nn.log_softmax(logits_correct, axis=-1)
    log_q = jax.nn.log_softmax(logits_stale, axis=-1)
    log_m = jnp.logaddexp(log_p, log_q) - jnp.log(2.0)
    tv = 0.5 * jnp.abs(jnp.exp(log_p) - jnp.exp(log_q)).sum(-1)
    js = (0.5 * (jnp.exp(log_p) * (log_p - log_m)).sum(-1)
          + 0.5 * (jnp.exp(log_q) * (log_q - log_m)).sum(-1))
    switch = (jnp.argmax(logits_correct, -1) != jnp.argmax(logits_stale, -1)).astype(jnp.float32)
    control = jnp.max(jnp.abs(logits_perturbed - logits_correct))

    tv = np.asarray(jax.device_get(tv), dtype=np.float64)
    js = np.asarray(jax.device_get(js), dtype=np.float64)
    switch = np.asarray(jax.device_get(switch), dtype=np.float64)
    return {
        "matched_states": n,
        "matched_seeds": matched["seeds"],
        "battle_steps_recorded": matched["steps"],
        "swap": {
            "partner_identity": int(identity),
            "requested_identity": int(swap_identity),
            "identity_flipped": int(identity) != int(swap_identity),
            "direction": "stale",
            "param_stage": G.STAGE_LABELS[param_stage],
            "param_stage_index": int(param_stage),
            "profile_stage": G.STAGE_LABELS[stale],
            "profile_stage_index": int(stale),
            "param_stage_later_than_profile_stage": bool(int(param_stage) > int(stale)),
            "correct_stage": G.STAGE_LABELS[param_stage],
            "stale_stage": G.STAGE_LABELS[stale],
            "z_dims_replaced": [3 * j, 3 * j + 3],
            "z_max_abs_change": float(np.max(np.abs(z_stale - z_correct))),
            "note": ("the partner really plays its param_stage parameters and the observer is given "
                     "the SAME identity's early profile_stage instead; same identity, same "
                     "trajectory positions, same GRU hidden state, only those 3 z dims differ"),
        },
        "hstate_check": {
            "same_hstate_injected": bool(hstates_equal_before),
            "same_hstate_after_both_forwards": bool(hstates_equal_after),
            "negative_control_max_logit_shift_from_perturbed_hstate":
                float(np.asarray(jax.device_get(control))),
        },
        "inputs_differ_only_in_z": inputs_differ_only_in_z,
        "recorded_z_matches_correct_z": z_matches_recording,
        "action_distribution": {
            "tv_mean": float(tv.mean()),
            "tv_max": float(tv.max()),
            "tv_min": float(tv.min()),
            "states_with_zero_tv": int(np.sum(tv <= 0.0)),
            "js_mean_nats": float(js.mean()),
            "js_max_nats": float(js.max()),
            "argmax_switch_rate": float(switch.mean()),
            "argmax_switch_count": int(round(switch.sum())),
        },
        "interpretation": DISCLAIMER_USAGE,
    }


# --------------------------------------------------------------------------------------------
# CLI / main
# --------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile-run", default=str(DEFAULT_PROFILE_RUN))
    parser.add_argument("--placeholder-run", default=str(DEFAULT_PLACEHOLDER_RUN))
    parser.add_argument("--onehot-run", default=str(DEFAULT_ONEHOT_RUN),
                        help="the onehot arm's run directory (the primary comparison's first arm)")
    parser.add_argument("--profile-checkpoint", default=None,
                        help="update count, filename or path (default: latest)")
    parser.add_argument("--placeholder-checkpoint", default=None)
    parser.add_argument("--onehot-checkpoint", default=None)
    parser.add_argument("--profiles", default=None, help="profile table (default: the real table, "
                                                         "else the fixture)")
    parser.add_argument("--partner-run-dir", default=str(G.DEFAULT_PARTNER_RUN_DIR))
    parser.add_argument("--partner-stages", default=",".join(str(s) for s in G.STAGE_UPDATES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in G.TEST_SEEDS_C),
                        help="test seed set C (contract §3/§4)")
    parser.add_argument("--combos-subset", type=int, default=None,
                        help="use only N of the 81 combinations (a SUBSET; the contract's protocol "
                             "is all 81 x 4 = 324 battles per arm)")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--swap-identity", type=int, default=G.PARTNER_IDENTITIES[0],
                        help="preferred partner identity to inspect; the automatic flip moves to "
                             "another identity when this one does not play a late stage (the "
                             "direction - late parameters, early profile - never flips)")
    parser.add_argument("--swap-stale-stage", type=int, default=0,
                        help="stage index substituted as the STALE (early) profile (0 = u50); the "
                             "inspected partner's real, later stage stays the premise")
    parser.add_argument("--matched-seeds", default=None,
                        help="seed set for the matched-state rollouts (default: --seeds)")
    parser.add_argument("--matched-steps", type=int, default=8)
    parser.add_argument("--matched-combos", type=int, default=1)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--tiny", action="store_true",
                        help="short demonstration: 2 combinations instead of 81 (clearly reported "
                             "as a subset); the full 324-battle protocol is the default")
    args = parser.parse_args(argv)
    if args.tiny and args.combos_subset is None:
        args.combos_subset = 2
    try:
        args.seeds = [int(x) for x in str(args.seeds).replace(" ", "").split(",") if x]
        if args.matched_seeds is None:
            args.matched_seeds = list(args.seeds)
        else:
            args.matched_seeds = [int(x) for x in
                                 str(args.matched_seeds).replace(" ", "").split(",") if x]
    except ValueError:
        parser.error("--seeds / --matched-seeds must be comma-separated integers")
    if not args.seeds:
        parser.error("--seeds must contain at least one seed")
    if args.combos_subset is not None and args.combos_subset < 1:
        parser.error("--combos-subset must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    started = time.perf_counter()

    run_dirs = {"onehot": resolve_dir(args.onehot_run),
                "placeholder": resolve_dir(args.placeholder_run),
                "profile": resolve_dir(args.profile_run)}
    checkpoints = {
        "onehot": resolve_checkpoint(run_dirs["onehot"], args.onehot_checkpoint),
        "placeholder": resolve_checkpoint(run_dirs["placeholder"], args.placeholder_checkpoint),
        "profile": resolve_checkpoint(run_dirs["profile"], args.profile_checkpoint),
    }
    updates = {arm: checkpoints[arm][0] for arm in EVAL_ARMS}
    params = {arm: G.load_observer_params(checkpoints[arm][1]) for arm in EVAL_ARMS}

    # The three arms must differ in the input channel only; any arm may be the config's source, so
    # take the primary arm's config and check the other two against it.
    config = load_run_config(run_dirs[PRIMARY_PAIR[0]], checkpoints[PRIMARY_PAIR[0]][1])
    for arm in EVAL_ARMS:
        if arm != PRIMARY_PAIR[0]:
            check_arm_compatibility(config, load_run_config(run_dirs[arm], checkpoints[arm][1]),
                                    str(run_dirs[arm]))
    partner_dir = resolve_dir(args.partner_run_dir)
    stages = tuple(int(x) for x in str(args.partner_stages).replace(" ", "").split(",") if x)
    if stages != tuple(G.STAGE_UPDATES):
        raise SystemExit(f"the contract fixes the partner stage set to {list(G.STAGE_UPDATES)}")
    partner_stack = G.load_partner_stack(partner_dir, stages)
    table, table_source, is_fixture = G.load_profile_table(args.profiles)
    z_table = G.z_table_from_profile_table(table)

    # One trainer per arm: the trainer owns the arm's own z channel (its `z_table` is the one-hot
    # table for `onehot`, a constant zero for `placeholder` and the measured table for `profile`),
    # so no arm can be evaluated through another arm's channel.
    trainers = {
        "onehot": G.ObserverGateTrainer(config, partner_stack, G.onehot_z_table(), arm="onehot"),
        "placeholder": G.ObserverGateTrainer(config, partner_stack, None, arm="placeholder"),
        "profile": G.ObserverGateTrainer(config, partner_stack, z_table, arm="profile"),
    }
    combos = G.enumerate_combos(args.combos_subset)
    is_subset = len(combos) < G.NUM_COMBOS
    battles_per_arm = len(combos) * len(args.seeds)
    primary_label = f"{PRIMARY_PAIR[0]} - {PRIMARY_PAIR[1]}"

    print("=" * 100, flush=True)
    print("[oracle gate evaluation] §4 tri-arm paired protocol + §5 z-swap check", flush=True)
    for arm in EVAL_ARMS:
        print(f"  {arm + ' arm':<17}: {checkpoints[arm][1]} (update {updates[arm]})   "
              f"z = {Z_CHANNEL_BY_ARM[arm]}", flush=True)
    print(f"  profile table    : {table_source}"
          f"{' (FIXTURE - replace with the measured table)' if is_fixture else ''} "
          f"(used by the profile arm only)", flush=True)
    print(f"  partner stack    : {partner_dir} stages {list(stages)}", flush=True)
    print(f"  protocol         : {len(combos)}/{G.NUM_COMBOS} partner-stage combinations"
          f"{' (SUBSET)' if is_subset else ' (all)'} x {len(args.seeds)} test situations "
          f"{args.seeds} = {battles_per_arm} battles per arm; all {len(EVAL_ARMS)} arms use the SAME "
          f"{battles_per_arm} (combination, situation) pairs", flush=True)
    if is_subset:
        print("  NOTE: this is a SUBSET of the contract §4 protocol (81 combinations x 4 "
              "situations = 324 battles per arm); the numbers below are not the gate result.",
              flush=True)
    print(f"  [PRIMARY] comparison: {primary_label} on the {battles_per_arm} shared pairs "
          f"(bootstrap {args.bootstrap_samples} resamples, seed {args.bootstrap_seed}); "
          f"the other pairs are reported alongside it", flush=True)
    print(f"  battle termination: every battle runs to done['__all__'] (env max_steps="
          f"{trainers['onehot'].config['horizon']}, loop guard eval_step_limit="
          f"{trainers['onehot'].eval_step_limit}); a battle that never terminates is reported as a "
          f"truncated FAILURE", flush=True)
    print(f"  partner rule     : argmax (errata 2, the profile-measurement rule); the observer "
          f"plays argmax in a battle", flush=True)

    results = {}
    for arm in EVAL_ARMS:
        z_mode = "zero" if arm == "placeholder" else "table"
        t0 = time.perf_counter()
        out = trainers[arm].eval_battles(params[arm], combos, args.seeds, z_mode=z_mode)
        results[arm] = out
        lengths, counts = np.unique(out["length"], return_counts=True)
        histogram = ", ".join(f"{int(x)}x{int(c)}" for x, c in zip(lengths, counts))
        print(f"  [{arm:<11}] [PRIMARY] mean team return={out['return'].mean():.4f} | "
              f"[aux] wins={int(out['won'].sum())}/{battles_per_arm} "
              f"({out['won'].mean():.3f}) | mean length={out['length'].mean():.2f} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        print(f"              terminated={int((~out['truncated']).sum())}/{battles_per_arm} "
              f"(truncated={int(out['truncated'].sum())}); length histogram "
              f"(length x battles): {histogram}", flush=True)

    # --- §4: the paired differences, primary first -------------------------------------------
    comparisons = {}
    for arm_a, arm_b in COMPARISON_PAIRS:
        name = f"{arm_a} - {arm_b}"
        paired = {key: paired_differences(results[arm_a][key], results[arm_b][key])
                  for key in ("return", "won", "length")}
        stats = {key: bootstrap_mean(paired[key], args.bootstrap_samples, args.bootstrap_seed)
                 for key in ("return", "won", "length")}
        comparisons[(arm_a, arm_b)] = {
            "name": name, "arm_a": arm_a, "arm_b": arm_b,
            "unit": "(combination index, situation index)",
            "n_pairs": int(paired["return"].size),
            "return": stats["return"], "wins": stats["won"], "length": stats["length"],
            "per_pair_return_diff": [float(x) for x in paired["return"].ravel()],
            "per_combo_mean_return_diff": [float(x) for x in paired["return"].mean(1)],
            "primary": (arm_a, arm_b) == PRIMARY_PAIR,
        }

    primary = comparisons[PRIMARY_PAIR]
    print("", flush=True)
    print(f"  §4 PRIMARY paired difference ({primary_label}), pairing unit = (combination, "
          f"situation):", flush=True)
    ret = primary["return"]
    print(f"    PRIMARY  mean team return : {ret['mean']:+.4f} "
          f"(pairs n={ret['n']}, sd of pairs={ret['sd_of_pairs']:.4f})", flush=True)
    print(f"             bootstrap over the {ret['n']} pairs "
          f"({ret['bootstrap_samples']} resamples, seed {ret['bootstrap_seed']}): mean "
          f"{ret['bootstrap_mean']:+.4f}, sd {ret['bootstrap_sd']:.4f}, 95% "
          f"[{ret['bootstrap_ci95_low']:+.4f}, {ret['bootstrap_ci95_high']:+.4f}]", flush=True)
    wins = primary["wins"]
    length = primary["length"]
    print(f"    [aux]    wins             : {wins['mean']:+.4f} "
          f"(bootstrap sd {wins['bootstrap_sd']:.4f}, 95% "
          f"[{wins['bootstrap_ci95_low']:+.4f}, {wins['bootstrap_ci95_high']:+.4f}]) - "
          f"auxiliary only", flush=True)
    print(f"    [aux]    episode length   : {length['mean']:+.4f} "
          f"(bootstrap sd {length['bootstrap_sd']:.4f})", flush=True)
    print("", flush=True)
    print("  §4 other pairs (same pairs, same statistic, same bootstrap):", flush=True)
    for arm_a, arm_b in COMPARISON_PAIRS[1:]:
        other = comparisons[(arm_a, arm_b)]
        stats = other["return"]
        print(f"    {other['name']:<20} mean team return {stats['mean']:+.4f} "
              f"(bootstrap sd {stats['bootstrap_sd']:.4f}, 95% "
              f"[{stats['bootstrap_ci95_low']:+.4f}, {stats['bootstrap_ci95_high']:+.4f}]); "
              f"wins {other['wins']['mean']:+.4f} [aux]", flush=True)
    print("", flush=True)
    print(f"  per-combination mean team return ({' | '.join(EVAL_ARMS)} | {primary_label}):",
          flush=True)
    for ci, combo in enumerate(combos):
        per_arm = " | ".join(f"{results[arm]['return'][ci].mean():+8.3f}" for arm in EVAL_ARMS)
        print(f"    {G.combo_label(combo):<26} {per_arm} | "
              f"{primary['per_combo_mean_return_diff'][ci]:+8.3f}", flush=True)

    # --- §5 ------------------------------------------------------------------------------------
    # One matched-state batch per arm, so each arm's swap uses that arm's own channel: for the
    # onehot arm the stale description is the same identity's EARLY-stage one-hot.
    matched_combo = default_matched_combo(combos, args.swap_identity)
    swaps = {}
    for arm in SWAP_ARMS:
        trainer_arm = trainers[arm]
        matched = collect_matched_states(trainer_arm, params[arm], matched_combo,
                                        args.matched_seeds, args.matched_steps)
        swaps[arm] = swap_analysis(trainer_arm, params[arm], matched, matched_combo,
                                   args.swap_identity, args.swap_stale_stage)
        swap = swaps[arm]
        print("", flush=True)
        print(f"  §5 z-swap [{arm}] on {swap['matched_states']} matched states "
              f"({len(args.matched_seeds)} seeds x {args.matched_steps} steps of "
              f"{G.combo_label(matched_combo)})", flush=True)
        print(f"    swap: partner {swap['swap']['partner_identity']}"
              f"{' (identity flipped from ' + str(swap['swap']['requested_identity']) + ')' if swap['swap']['identity_flipped'] else ''}"
              f" really plays {swap['swap']['param_stage']} while the observer is given its "
              f"{swap['swap']['profile_stage']} z -> direction "
              f"{swap['swap']['direction']} "
              f"(param stage {swap['swap']['param_stage_index']} > z stage "
              f"{swap['swap']['profile_stage_index']}); z dims "
              f"{swap['swap']['z_dims_replaced']}, max |dz|={swap['swap']['z_max_abs_change']:.4f}",
              flush=True)
        print(f"    same hstate injected: {swap['hstate_check']['same_hstate_injected']}; "
              f"still equal after both forwards: "
              f"{swap['hstate_check']['same_hstate_after_both_forwards']}; inputs differ only in z: "
              f"{swap['inputs_differ_only_in_z']}; negative control (perturbed hstate) max logit "
              f"shift: {swap['hstate_check']['negative_control_max_logit_shift_from_perturbed_hstate']:.4f}",
              flush=True)
        print(f"    action distribution: TV mean {swap['action_distribution']['tv_mean']:.4f} "
              f"(max {swap['action_distribution']['tv_max']:.4f}), JS mean "
              f"{swap['action_distribution']['js_mean_nats']:.4f} nats, argmax switch rate "
              f"{swap['action_distribution']['argmax_switch_rate']:.3f}", flush=True)
    print("", flush=True)
    print(f"  §5 z-swap [placeholder] skipped: its channel is a constant zero vector, so a stale "
          f"swap would compare an input with itself", flush=True)

    payload = {
        "kind": "smax_2s3z_oracle_gate_eval",
        "contract": "MainSearch/explore/ORACLE_GATE_CONTRACT.md (v1)",
        "ticket": "2026-09-27 onehot arm (third arm; primary comparison onehot - placeholder)",
        "disclaimers": list(DISCLAIMERS),
        "statistics_note": DISCLAIMER_FIXED_SET,
        "primary_metric": "mean team return",
        "secondary_metric": "wins (auxiliary only)",
        "primary_comparison": primary_label,
        "pairing_unit": "partner-stage combination AND test situation (same battle replayed)",
        "protocol": {
            "combinations": len(combos),
            "combinations_available": G.NUM_COMBOS,
            "is_subset": is_subset,
            "combo_labels": [G.combo_label(c) for c in combos],
            "seeds": list(args.seeds),
            "seed_set": "C (contract §3) - the fixed gate test set",
            "arms": list(EVAL_ARMS),
            "battles_per_arm": battles_per_arm,
            "battles_total": len(EVAL_ARMS) * battles_per_arm,
            "z_channel": {arm: Z_CHANNEL_BY_ARM[arm] for arm in EVAL_ARMS},
            "matched": "all arms use the same (combination, situation) pairs",
        },
        "arms": {
            arm: {"run_dir": str(run_dirs[arm]), "checkpoint": checkpoints[arm][1].name,
                  "update_count": updates[arm], "z_channel": Z_CHANNEL_BY_ARM[arm]}
            for arm in EVAL_ARMS
        },
        "profile_table": {"path": str(table_source), "sha256": G.sha256_file(table_source),
                          "is_fixture": is_fixture, "used_by_arms": ["profile"]},
        "z_semantics": {arm: Z_SEMANTICS_BY_ARM[arm] for arm in EVAL_ARMS},
        "battle_termination": {
            "rule": ("every battle runs to its real termination (done['__all__']); max_steps is not "
                     "the loop bound (errata 3 - the environment flags done one step late)"),
            "env_max_steps": int(trainers["onehot"].config["horizon"]),
            "eval_step_limit": int(trainers["onehot"].eval_step_limit),
            "truncated_battles": {arm: int(results[arm]["truncated"].sum()) for arm in EVAL_ARMS},
            "length_histogram": {
                arm: {int(k): int(v) for k, v in zip(*np.unique(results[arm]["length"],
                                                               return_counts=True))}
                for arm in EVAL_ARMS
            },
            "note": ("a battle still running when the guard runs out raises instead of being "
                     "reported as a normal result"),
        },
        "partner_action_rule": ("argmax in training, evaluation and matched-state rollouts (errata "
                                "2, the profile-measurement rule); the observer samples in "
                                "training and plays argmax in a battle"),
        "partner_run_dir": str(partner_dir),
        "partner_stages": list(stages),
        "results": {
            arm: {
                "mean_team_return": float(results[arm]["return"].mean()),
                "wins": int(results[arm]["won"].sum()),
                "win_rate": float(results[arm]["won"].mean()),
                "mean_length": float(results[arm]["length"].mean()),
                "per_combo_mean_return": [float(x) for x in results[arm]["return"].mean(1)],
                "per_combo_wins": [int(x) for x in results[arm]["won"].sum(1)],
                "per_battle_return": [float(x) for x in results[arm]["return"].ravel()],
                "per_battle_won": [int(x) for x in results[arm]["won"].ravel()],
                "per_battle_length": [float(x) for x in results[arm]["length"].ravel()],
                "per_battle_terminated": [bool(x) for x in
                                          (~results[arm]["truncated"]).ravel()],
            }
            for arm in EVAL_ARMS
        },
        # `paired` mirrors the PRIMARY comparison for the era when the only pair was
        # profile - placeholder; `comparisons` carries all pairs of this tri-arm run.
        "paired": comparisons[PRIMARY_PAIR],
        "comparisons": {f"{a} - {b}": comparisons[(a, b)] for a, b in COMPARISON_PAIRS},
        "z_swap": {
            "arms": dict(swaps),
            "matched_combo": G.combo_label(matched_combo),
            "skipped": {"placeholder": ("its channel is a constant zero vector, so a stale swap "
                                        "would compare an input with itself")},
            "note": ("for each arm the same identity's EARLY-stage description replaces the one it "
                     "really plays, with identical GRU hidden states injected into both forwards"),
        },
        # Legacy top-level keys of the profile arm's §5 check (kept for the earlier reports).
        "profile_swap": swaps["profile"],
        "profile_swap_combo": G.combo_label(matched_combo),
        "interpretation": list(DISCLAIMERS),
        "wall_seconds": round(time.perf_counter() - started, 3),
    }
    out_path = Path(args.out)
    out_path = out_path if out_path.is_absolute() else (HERE / out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("", flush=True)
    for note in DISCLAIMERS:
        print(f"  note: {note}", flush=True)
    print(f"  wrote {out_path}  (wall {payload['wall_seconds']:.1f}s)", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
