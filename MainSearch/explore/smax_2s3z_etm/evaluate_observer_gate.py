"""Oracle-gate evaluation: the §4 paired battle protocol and the §5 profile-swap check.

Frozen interface: `MainSearch/explore/ORACLE_GATE_CONTRACT.md` (v1).  Run with the repository's
absolute interpreter path, e.g.

  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      evaluate_observer_gate.py --tiny --combos-subset 2

§4 (gate evaluation)
--------------------
All `3^4 = 81` partner-stage combinations x the fixed test seed set `C = [3234,3235,3236,3237]`
= **324 battles per arm**.  Both arms are evaluated on **exactly the same 324
(combination, situation) pairs**:

* primary metric  = mean team return (mean over the 324 battles);
* secondary metric = number of won battles (auxiliary only, never the decision).

The pairing unit is `(partner-stage combination, test situation)`: the effect is the per-pair
difference `profile - placeholder`, and the uncertainty is a **bootstrap over those 324 pairs**.
That bootstrap describes *this fixed set of situations*; it is **not** a sample from SMAX as a
whole, and the battles are **not** a re-verification of the training seed (contract §4, §6).

Both arms play the same partners in the same battles; the only difference is the observer's second
input channel: the `profile` arm receives the battle's `z` (12 dims), the `placeholder` arm
receives a constant zero vector (its channel is a fixed constant by definition, at training *and*
at evaluation).

`z` itself is read from `results/oracle_gate/profiles.json` and means: **the partner's three-metric
behaviour profile as measured under the fixed reference team (every other identity kept at its
u1250 parameters) on the fixed measurement situations A** (contract §1 v2: real hit events, weapon
cooldown, sub-step granularity - measured by `measure_partner_profiles.py`, never re-derived here),
normalized by the frozen 12-candidate mean/std.  It is a profile *under that reference team*, not a
team-independent statement about the partner's ability.

§5 (profile-swap check, matched states)
---------------------------------------
On one batch of matched states (same battles, same trajectory positions) the `profile` arm's
observer is run twice: once with the correct `z` and once with a **stale same-identity profile** -
the partner is inspected in a combination where it really plays its **late** parameters (`u1250`)
and the z dims handed to the observer for that same identity are replaced by its **early** `u50`
profile.  Both forward passes are given **the same GRU hidden states** - the hstates are injected
explicitly and compared leaf-by-leaf - so any difference is attributable to the z channel alone.

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
DEFAULT_OUT = HERE / "results" / "oracle_gate" / "observer_gate_eval.json"

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
    "Input affects decisions != the change is beneficial: the profile-swap check shows only that "
    "the z channel reaches the policy, never that acting on it helps."
)
DISCLAIMER_NULL = (
    "Even if the profile table is separable and the interface works, a null result supports only "
    "'no gain found in this configuration', never 'partner information has no value'."
)
Z_SEMANTICS = (
    "z = the partner's 3-metric behaviour profile measured under the FIXED reference team (every "
    "other identity at its u1250 parameters) on measurement situations A, normalized by the frozen "
    "12-candidate mean/std (contract §1 v2).  It is a profile under that reference team, not a "
    "team-independent capability, and it is read from profiles.json - never re-derived here."
)

DISCLAIMERS = (DISCLAIMER_FIXED_SET, DISCLAIMER_NOT_RECHECK, DISCLAIMER_USAGE, DISCLAIMER_NULL,
               Z_SEMANTICS)


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
    """The two arms must differ in the input channel only - everything else must match."""
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

    Short rollouts of the *correct-profile* observer on `combo` x `seeds`; every recorded position
    keeps the hidden state the observer actually had when it saw that input, so the later swap can
    inject identical hidden states into both forward passes.  The observer plays argmax and the
    partners play argmax too (errata 2, as in training and in the gate battles).
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
            f"combination {G.combo_label(combo)} cannot host a stale profile swap: no identity "
            f"plays a stage later than {G.STAGE_LABELS[int(stale_stage)]}, and a swap in the other "
            f"direction would compare the observer against a profile of the FUTURE (errata 1)."
        )
    return identities[latest], stages[latest], int(min(earlier))


def swap_analysis(trainer, observer_params, matched: dict, combo: Sequence[int],
                  swap_identity: int = G.PARTNER_IDENTITIES[0],
                  stale_stage: int = 0) -> dict:
    """Compare the observer's action distribution under the correct vs a stale same-identity z.

    The inspected partner (see `resolve_swap_target`) plays its late parameters while the observer's
    z dims for that identity come either from the stage it really plays (`param_stage`) or from the
    same identity's early `profile_stage` - a *stale* description.  The two forward passes receive
    **identical GRU hidden states** (injected explicitly and compared leaf-by-leaf).  The
    observation part of the input is identical too; only the 3 z dims of that partner differ.  A
    third pass with a perturbed hidden state is the negative control that proves the hidden state is
    actually consumed by the forward pass.
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
    parser.add_argument("--profile-checkpoint", default=None,
                        help="update count, filename or path (default: latest)")
    parser.add_argument("--placeholder-checkpoint", default=None)
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

    profile_dir = resolve_dir(args.profile_run)
    placeholder_dir = resolve_dir(args.placeholder_run)
    profile_update, profile_ckpt = resolve_checkpoint(profile_dir, args.profile_checkpoint)
    placeholder_update, placeholder_ckpt = resolve_checkpoint(placeholder_dir,
                                                              args.placeholder_checkpoint)
    profile_params = G.load_observer_params(profile_ckpt)
    placeholder_params = G.load_observer_params(placeholder_ckpt)

    config = load_run_config(profile_dir, profile_ckpt)
    check_arm_compatibility(config, load_run_config(placeholder_dir, placeholder_ckpt),
                            str(placeholder_dir))
    partner_dir = resolve_dir(args.partner_run_dir)
    stages = tuple(int(x) for x in str(args.partner_stages).replace(" ", "").split(",") if x)
    if stages != tuple(G.STAGE_UPDATES):
        raise SystemExit(f"the contract fixes the partner stage set to {list(G.STAGE_UPDATES)}")
    partner_stack = G.load_partner_stack(partner_dir, stages)
    table, table_source, is_fixture = G.load_profile_table(args.profiles)
    z_table = G.z_table_from_profile_table(table)

    trainer = G.ObserverGateTrainer(config, partner_stack, z_table, arm="profile")
    combos = G.enumerate_combos(args.combos_subset)
    is_subset = len(combos) < G.NUM_COMBOS
    battles_per_arm = len(combos) * len(args.seeds)

    print("=" * 100, flush=True)
    print("[oracle gate evaluation] §4 paired protocol + §5 profile-swap check", flush=True)
    print(f"  profile arm      : {profile_ckpt} (update {profile_update})", flush=True)
    print(f"  placeholder arm  : {placeholder_ckpt} (update {placeholder_update})", flush=True)
    print(f"  profile table    : {table_source}"
          f"{' (FIXTURE - replace with the measured table)' if is_fixture else ''}", flush=True)
    print(f"  partner stack    : {partner_dir} stages {list(stages)}", flush=True)
    print(f"  protocol         : {len(combos)}/{G.NUM_COMBOS} partner-stage combinations"
          f"{' (SUBSET)' if is_subset else ' (all)'} x {len(args.seeds)} test situations "
          f"{args.seeds} = {battles_per_arm} battles per arm; both arms use the SAME "
          f"{battles_per_arm} (combination, situation) pairs", flush=True)
    if is_subset:
        print("  NOTE: this is a SUBSET of the contract §4 protocol (81 combinations x 4 "
              "situations = 324 battles per arm); the numbers below are not the gate result.",
              flush=True)
    print(f"  z channel        : profile = the battle's z; placeholder = constant zeros "
          f"({G.PROFILE_DIM} dims in both arms)", flush=True)
    print(f"  z semantics      : {Z_SEMANTICS}", flush=True)
    print(f"  battle termination: every battle runs to done['__all__'] (env max_steps="
          f"{trainer.config['horizon']}, loop guard eval_step_limit={trainer.eval_step_limit}); a "
          f"battle that never terminates is reported as a truncated FAILURE", flush=True)
    print(f"  partner rule     : argmax (errata 2, the profile-measurement rule); the observer "
          f"plays argmax in a battle", flush=True)

    results = {}
    for arm, params, z_mode in (("profile", profile_params, "table"),
                                ("placeholder", placeholder_params, "zero")):
        t0 = time.perf_counter()
        out = trainer.eval_battles(params, combos, args.seeds, z_mode=z_mode)
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

    paired = {}
    for key in ("return", "won", "length"):
        paired[key] = paired_differences(results["profile"][key], results["placeholder"][key])
    return_stats = bootstrap_mean(paired["return"], args.bootstrap_samples, args.bootstrap_seed)
    win_stats = bootstrap_mean(paired["won"], args.bootstrap_samples, args.bootstrap_seed)
    length_stats = bootstrap_mean(paired["length"], args.bootstrap_samples, args.bootstrap_seed)

    print("", flush=True)
    print("  §4 paired difference (profile - placeholder), pairing unit = "
          "(combination, situation):", flush=True)
    print(f"    PRIMARY  mean team return : {return_stats['mean']:+.4f} "
          f"(pairs n={return_stats['n']}, sd of pairs={return_stats['sd_of_pairs']:.4f})", flush=True)
    print(f"             bootstrap over the {return_stats['n']} pairs "
          f"({return_stats['bootstrap_samples']} resamples, seed "
          f"{return_stats['bootstrap_seed']}): mean {return_stats['bootstrap_mean']:+.4f}, "
          f"sd {return_stats['bootstrap_sd']:.4f}, "
          f"95% [{return_stats['bootstrap_ci95_low']:+.4f}, "
          f"{return_stats['bootstrap_ci95_high']:+.4f}]", flush=True)
    print(f"    [aux]    wins             : {win_stats['mean']:+.4f} "
          f"(bootstrap sd {win_stats['bootstrap_sd']:.4f}, 95% "
          f"[{win_stats['bootstrap_ci95_low']:+.4f}, {win_stats['bootstrap_ci95_high']:+.4f}]) - "
          f"auxiliary only", flush=True)
    print(f"    [aux]    episode length   : {length_stats['mean']:+.4f} "
          f"(bootstrap sd {length_stats['bootstrap_sd']:.4f})", flush=True)
    print("", flush=True)
    print("  per-combination mean team return (profile | placeholder | diff):", flush=True)
    for ci, combo in enumerate(combos):
        print(f"    {G.combo_label(combo):<26} "
              f"{results['profile']['return'][ci].mean():+8.3f} | "
              f"{results['placeholder']['return'][ci].mean():+8.3f} | "
              f"{paired['return'][ci].mean():+8.3f}", flush=True)

    # --- §5 ------------------------------------------------------------------------------------
    matched_combo = default_matched_combo(combos, args.swap_identity)
    matched = collect_matched_states(trainer, profile_params, matched_combo, args.matched_seeds,
                                     args.matched_steps)
    swap = swap_analysis(trainer, profile_params, matched, matched_combo, args.swap_identity,
                         args.swap_stale_stage)
    print("", flush=True)
    print(f"  §5 profile swap on {swap['matched_states']} matched states "
          f"({len(args.matched_seeds)} seeds x {args.matched_steps} steps of "
          f"{G.combo_label(matched_combo)})", flush=True)
    print(f"    swap: partner {swap['swap']['partner_identity']}"
          f"{' (identity flipped from ' + str(swap['swap']['requested_identity']) + ')' if swap['swap']['identity_flipped'] else ''}"
          f" really plays {swap['swap']['param_stage']} while the observer is given its "
          f"{swap['swap']['profile_stage']} profile -> direction "
          f"{swap['swap']['direction']} "
          f"(param stage {swap['swap']['param_stage_index']} > profile stage "
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

    payload = {
        "kind": "smax_2s3z_oracle_gate_eval",
        "contract": "MainSearch/explore/ORACLE_GATE_CONTRACT.md (v1)",
        "disclaimers": list(DISCLAIMERS),
        "statistics_note": DISCLAIMER_FIXED_SET,
        "primary_metric": "mean team return",
        "secondary_metric": "wins (auxiliary only)",
        "pairing_unit": "partner-stage combination AND test situation (same battle replayed)",
        "protocol": {
            "combinations": len(combos),
            "combinations_available": G.NUM_COMBOS,
            "is_subset": is_subset,
            "combo_labels": [G.combo_label(c) for c in combos],
            "seeds": list(args.seeds),
            "seed_set": "C (contract §3) - the fixed gate test set",
            "battles_per_arm": battles_per_arm,
            "battles_total": 2 * battles_per_arm,
            "z_channel": {"profile": "battle z (12 dims)", "placeholder": "constant zeros (12 dims)"},
            "matched": "both arms use the same (combination, situation) pairs",
        },
        "arms": {
            "profile": {"run_dir": str(profile_dir), "checkpoint": profile_ckpt.name,
                        "update_count": profile_update},
            "placeholder": {"run_dir": str(placeholder_dir),
                            "checkpoint": placeholder_ckpt.name,
                            "update_count": placeholder_update},
        },
        "profile_table": {"path": str(table_source), "sha256": G.sha256_file(table_source),
                          "is_fixture": is_fixture},
        "z_semantics": Z_SEMANTICS,
        "battle_termination": {
            "rule": ("every battle runs to its real termination (done['__all__']); max_steps is not "
                     "the loop bound (errata 3 - the environment flags done one step late)"),
            "env_max_steps": int(trainer.config["horizon"]),
            "eval_step_limit": int(trainer.eval_step_limit),
            "truncated_battles": {
                arm: int(results[arm]["truncated"].sum()) for arm in ("profile", "placeholder")
            },
            "length_histogram": {
                arm: {int(k): int(v) for k, v in zip(*np.unique(results[arm]["length"],
                                                               return_counts=True))}
                for arm in ("profile", "placeholder")
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
            for arm in ("profile", "placeholder")
        },
        "paired": {
            "unit": "(combination index, situation index)",
            "n_pairs": int(paired["return"].size),
            "return": return_stats,
            "wins": win_stats,
            "length": length_stats,
            "per_pair_return_diff": [float(x) for x in paired["return"].ravel()],
        },
        "profile_swap": swap,
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
