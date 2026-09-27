"""Contract §7 preflight for the oracle-gate scripts (`--tiny`, minutes).

Checks (contract `MainSearch/explore/ORACLE_GATE_CONTRACT.md` §7), each measured on the real code:

  1. frozen partners  - every partner's parameters stay leaf-wise unchanged through a training
                        step, and each partner slot holds *its own identity's* checkpoints
                        (12 exact stage/identity matches), with a behavioural counterfactual
                        (swapping two identities' parameters changes the roll-out);
  2. partner sequence - two arms whose episodes have *different lengths* still play the same
                        partner combination and the same reset state at the same
                        `(slot, episode_index)`, recomputed independently from the streams;
  3. stream separation- the action-sampling rng never changes the partner/reset streams (and the
                        partner seeds never change the action stream);
  4. profile vector   - z is 12-dimensional in identity order `[1,2,3,4]`, the placeholder arm's
                        channel is exactly zero, the normalization is the frozen mean/std of the
                        12 candidates and reproducible, and both arms have identical input
                        dimension and parameter count;
  5. profile swap     - the swap forward passes inject leaf-wise equal GRU hidden states, and the
                        swapped z does change the action distribution (TV > 0); the swapped
                        dimension is the STALE one (late parameters, early profile);
  6. paired protocol  - the primary statistic is a per-`(combination, situation)` paired
                        difference over 81 x 4 = 324 pairs, with the bootstrap resampling those
                        pairs (wins stay auxiliary).

Checks 7-10 cover the four errata of the review ticket (2026-09-27):

  7. in-episode step index - `env_step_key` / `partner_action_key` are indexed by the persistent
                        in-episode step, so two arms with different roll-out *and* episode lengths
                        draw identical keys for the same `(slot, episode, in-episode step)` - even
                        for positions that fall into a later roll-out - and the pre-errata
                        roll-out-step indexing is shown to break (reverse control);
  8. argmax partners  - the partners act by argmax on the training path (recomputed independently
                        from the reset state) and on the evaluation path (replacing the argmax
                        helper by the pre-errata sampling rule moves the battle);
  9. stale swap       - the §5 swap is late-parameters + same-identity early profile; the default
                        combination is no longer the all-u50 one, all 81 combinations resolve to a
                        stale swap and the all-u50 request is refused instead of flipped to a
                        FUTURE profile;
 10. real termination - a battle runs to `done["__all__"]` (the local environment flags `done` one
                        step late, so the time limit needs `max_steps + 1` steps), and a battle cut
                        short by the loop guard is reported as a truncated FAILURE (reverse
                        control: the guard lowered to `max_steps`).

Run:
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      test_oracle_preflight.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import distrax  # noqa: E402
import evaluate_observer_gate as E  # noqa: E402
import train_observer_gate as G  # noqa: E402


class Reporter:
    def __init__(self) -> None:
        self.results = []
        print("=" * 100)
        print("SMAX 2s3z oracle-gate preflight (contract §7) - tiny configuration")
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
def episode_records(diagnostics: dict) -> dict:
    """`{(slot, episode_index): (stage vector, ally_0 observation, step the episode started at)}`.

    The diagnostics carry the per-step episode index, the step within the episode, the partner
    stage per identity and ally_0's raw observation, so the first step of episode `ep` exposes the
    state that episode was reset to (and the z the observer was given).
    """
    ep_index = np.asarray(diagnostics["diag_ep_index"])            # (T, E)
    t_in_episode = np.asarray(diagnostics["diag_t_in_episode"])    # (T, E)
    stage = np.asarray(diagnostics["diag_stage"])                  # (T, 4, E)
    obs = np.asarray(diagnostics["diag_obs_ally_0"])               # (T, E, obs_dim)
    records: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for step in range(ep_index.shape[0]):
        for slot in range(ep_index.shape[1]):
            if int(t_in_episode[step, slot]) != 0:
                continue
            key = (slot, int(ep_index[step, slot]))
            records.setdefault(key, (stage[step, :, slot].copy(), obs[step, slot].copy(), step))
    return records


def observed_episode_lengths(records: dict, slots: int, horizon: int) -> dict:
    """Episode lengths seen inside one roll-out (the last one per slot is truncated by the horizon).

    Derived from the step offsets at which each `(slot, episode)` started, so it measures the
    episode lengths actually played rather than the configured limit.
    """
    out = {}
    for slot in range(slots):
        starts = sorted((step, ep) for (s, ep), (_, _, step) in records.items() if s == slot)
        for index, (step, ep) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else horizon
            out[(slot, ep)] = end - step
    return out


def rollout_bundle(trainer, runner):
    """One roll-out; returns `(diagnostics, transitions)` (`collect_diagnostics = True` needed)."""
    transitions = trainer.rollout(runner)[1]
    diagnostics = {key: transitions[key] for key in G.DIAGNOSTIC_KEYS}
    return diagnostics, transitions


def rollout_diagnostics(trainer, runner) -> dict:
    """One roll-out; returns the diagnostics it produced (`collect_diagnostics = True` needed)."""
    return rollout_bundle(trainer, runner)[0]


def expected_reset_obs(trainer, slot: int, ep_index: int) -> np.ndarray:
    """ally_0's observation of `env.reset(env_reset_key(slot, ep_index))`, computed independently."""
    obs, _ = trainer.base_env.reset(G.env_reset_key(jnp.int32(slot), jnp.int32(ep_index)))
    return np.asarray(obs["ally_0"])


# --------------------------------------------------------------------------------------------
# check 1 - frozen partners, each identity on its own parameters
# --------------------------------------------------------------------------------------------
def check_1(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    trainer, runner = ctx["trainer"], ctx["runner"]
    stacks, partner_dir, stages = ctx["partner_stack"], ctx["partner_dir"], ctx["stages"]
    lines, ok = [], True

    ckpt_params = [G.load_checkpoint_params(p) for p in G.stage_checkpoint_paths(partner_dir, stages)]
    own_diff = {}
    for j, identity in enumerate(G.PARTNER_IDENTITIES):
        for s, stage in enumerate(stages):
            own_diff[(identity, stage)] = G.tree_max_abs_diff(stacks[j][s], ckpt_params[s][identity])
    worst_own = max(own_diff.values())
    lines.append(f"identity/stage -> own checkpoint slice: 12 comparisons, max leaf |diff| = "
                 f"{worst_own:.3e} (must be 0)")
    ok &= worst_own == 0.0

    cross = []
    for j, identity in enumerate(G.PARTNER_IDENTITIES):
        for s, stage in enumerate(stages):
            for other in G.PARTNER_IDENTITIES:
                if other != identity:
                    cross.append(G.tree_max_abs_diff(stacks[j][s], ckpt_params[s][other]))
    lines.append(f"cross-identity comparisons: {len(cross)}, min leaf |diff| = {min(cross):.3e} "
                 f"(> 0 means the identities really are different networks)")
    ok &= min(cross) > 0.0

    lines.append(f"observer runner params: {len(runner.params)} entry "
                 f"({sum(x.size for x in jax.tree.leaves(runner.params[0]))} parameters); the "
                 f"partners are not part of the training state at all")
    ok &= len(runner.params) == 1

    before = jax.tree.map(lambda x: np.array(x), stacks)
    observer_before = jax.tree.map(lambda x: np.array(x), runner.params[0])
    trained, _ = trainer.train_updates(runner, 1)
    frozen = G.tree_bitwise_equal(before, trainer.partner_stack)
    still_own = max(
        G.tree_max_abs_diff(trainer.partner_stack[j][s], ckpt_params[s][identity])
        for j, identity in enumerate(G.PARTNER_IDENTITIES) for s in range(len(stages))
    )
    observer_moved = G.tree_max_abs_diff(trained.params[0], observer_before)
    lines.append(f"after one PPO update: partner stack bitwise unchanged = {frozen}; still equal "
                 f"to its own checkpoint slices (max |diff| {still_own:.3e}); observer moved "
                 f"(leaf max |diff| {observer_moved:.3e})")
    ok &= frozen and still_own == 0.0 and observer_moved > 0.0

    # behavioural counterfactual: give identities 2 and 3 each other's parameters
    swapped = (stacks[0], stacks[2], stacks[1], stacks[3])
    swapped_trainer = G.ObserverGateTrainer(trainer.config, swapped, ctx["z_table"],
                                            arm="profile")
    swapped_trainer.collect_diagnostics = True
    swapped_diag = rollout_diagnostics(swapped_trainer, runner)
    baseline_diag = rollout_diagnostics(trainer, runner)
    shift = float(np.max(np.abs(np.asarray(swapped_diag["diag_obs_ally_0"])
                                - np.asarray(baseline_diag["diag_obs_ally_0"]))))
    lines.append(f"counterfactual (identities 2 and 3 exchange their parameters): max |obs "
                 f"difference| over the roll-out = {shift:.3e} (> 0 means the roll-out really "
                 f"consumes each partner's own parameters)")
    ok &= shift > 0.0

    rep.check("1 frozen partners / own identity", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 2 - the same (slot, episode) drives the same partner combination and reset in both arms
# --------------------------------------------------------------------------------------------
def check_2(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    short_a = G.ObserverGateTrainer(ctx["config"], ctx["partner_stack"], ctx["z_table"],
                                    arm="profile", env_overrides={"max_steps": 5})
    short_b = G.ObserverGateTrainer(ctx["config"], ctx["partner_stack"], ctx["z_table"],
                                    arm="placeholder", env_overrides={"max_steps": 3})
    short_a.collect_diagnostics = short_b.collect_diagnostics = True
    runner_a = short_a.init_runner(jax.random.PRNGKey(7))
    runner_b = short_b.init_runner(jax.random.PRNGKey(7))
    diag_a = rollout_diagnostics(short_a, runner_a)
    diag_b = rollout_diagnostics(short_b, runner_b)
    records_a, records_b = episode_records(diag_a), episode_records(diag_b)
    slots = int(ctx["config"]["num_envs"])
    horizon = int(ctx["config"]["rollout_length"])
    lengths_a = observed_episode_lengths(records_a, slots, horizon)
    lengths_b = observed_episode_lengths(records_b, slots, horizon)
    starts_a = {k: v[2] for k, v in records_a.items()}
    starts_b = {k: v[2] for k, v in records_b.items()}

    lines.append(f"arm A (profile, max_steps={short_a.base_env.max_steps}): episode start steps "
                 f"{starts_a}, observed lengths {lengths_a}")
    lines.append(f"arm B (placeholder, max_steps={short_b.base_env.max_steps}): episode start "
                 f"steps {starts_b}, observed lengths {lengths_b}")
    lengths_differ = lengths_a != lengths_b
    lines.append(f"the two arms really do have different episode lengths: {lengths_differ} "
                 f"(same tiny roll-out: num_envs={slots}, rollout_length={horizon}); the same "
                 f"(slot, episode) therefore happens at different step offsets, and the "
                 f"trajectories live in different positions of the roll-out")
    ok &= lengths_differ

    common = sorted(set(records_a) & set(records_b))
    lines.append(f"common (slot, episode) pairs: {common}")
    ok &= len(common) > 0

    stage_ok, obs_ok, replay_ok = True, True, True
    for key in common:
        slot, ep = key
        for j, identity in enumerate(G.PARTNER_IDENTITIES):
            expected = int(G.partner_stage_index(jnp.int32(slot), jnp.int32(ep), jnp.int32(identity)))
            stage_ok &= int(records_a[key][0][j]) == expected == int(records_b[key][0][j])
        obs_ok &= np.array_equal(records_a[key][1], records_b[key][1])
        expected_obs = expected_reset_obs(short_a, slot, ep)
        replay_ok &= np.array_equal(records_a[key][1], expected_obs)
        replay_ok &= np.array_equal(records_b[key][1], expected_obs)

    lines.append(f"partner-stage vectors equal across arms and equal to the pure stream function: "
                 f"{stage_ok}")
    obs_diff = (max(float(np.max(np.abs(records_a[k][1] - records_b[k][1]))) for k in common)
                if common else float("nan"))
    lines.append(f"episode-start observation bitwise identical across the two arms: {obs_ok} "
                 f"(max |diff| = {obs_diff:.3e})")
    lines.append(f"episode-start observation equals an independent "
                 f"env.reset(env_reset_key(slot, ep)): {replay_ok}")
    ok &= stage_ok and obs_ok and replay_ok
    rep.check("2 same (slot, episode) across arms", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 3 - the action stream is separate from the partner / reset streams
# --------------------------------------------------------------------------------------------
def check_3(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    trainer, runner = ctx["trainer"], ctx["runner"]
    slots = jnp.arange(4, dtype=jnp.int32)
    ep = jnp.full((4,), 3, jnp.int32)
    step = jnp.int32(5)

    def partner_domain(seed_shift: int):
        return (
            jnp.stack([G.partner_stage_index(slots, ep, jnp.int32(i),
                                             seed=G.PARTNER_SEED + seed_shift)
                       for i in G.PARTNER_IDENTITIES]),
            G.env_reset_key(slots, ep, seed=G.RESET_SEED + seed_shift),
            G.env_step_key(slots, ep, step, seed=G.ENV_STEP_SEED + seed_shift),
            G.partner_action_key(slots, ep, jnp.int32(1), step,
                                 seed=G.PARTNER_ACTION_SEED + seed_shift),
        )

    use, other = partner_domain(0), partner_domain(1)
    differs = [bool(np.any(np.asarray(a) != np.asarray(b))) for a, b in zip(use, other)]
    lines.append(f"partner-domain streams respond to their own seeds (stage, reset, env-step, "
                 f"partner-action): {differs}")
    ok &= all(differs)

    action_rng_a = jax.random.PRNGKey(101)
    action_rng_b = jax.random.PRNGKey(202)
    logits = jnp.zeros((32, ctx["config"]["action_dim"]))
    pi = distrax.Categorical(logits=logits)
    sample_a = np.asarray(pi.sample(seed=G.observer_action_key(action_rng_a)))
    sample_again = np.asarray(pi.sample(seed=G.observer_action_key(action_rng_a)))
    sample_b = np.asarray(pi.sample(seed=G.observer_action_key(action_rng_b)))
    differing = int(np.sum(sample_a != sample_b))
    lines.append(f"action stream (32 lanes, same call as the observer's own sampling): "
                 f"reproducible for a fixed action rng = {np.array_equal(sample_a, sample_again)}; "
                 f"a different action rng changes {differing}/32 actions; the partner seeds above "
                 f"do not enter the action stream at all")
    ok &= np.array_equal(sample_a, sample_again) and differing > 0

    # roll-out level: changing ONLY the action rng leaves the partner stages and every reset state
    # of an (slot, episode) untouched, while the trajectories themselves do move.
    diag_a, transitions_a = rollout_bundle(trainer, runner)
    diag_b, transitions_b = rollout_bundle(trainer, runner._replace(rng=jax.random.PRNGKey(999)))
    records_a, records_b = episode_records(diag_a), episode_records(diag_b)
    common = sorted(set(records_a) & set(records_b))
    stage_equal = all(np.array_equal(records_a[k][0], records_b[k][0]) for k in common)
    obs_equal = all(np.array_equal(records_a[k][1], records_b[k][1]) for k in common)
    action_diff = int(np.sum(np.asarray(transitions_a["action"]) != np.asarray(transitions_b["action"])))
    obs_shift = float(np.max(np.abs(np.asarray(diag_a["diag_obs_ally_0"])
                                    - np.asarray(diag_b["diag_obs_ally_0"]))))
    lines.append(f"two roll-outs differing only in the action rng: common (slot, episode) "
                 f"{common}; partner stages identical = {stage_equal}; reset observations "
                 f"identical = {obs_equal}; the roll-outs themselves diverge "
                 f"({action_diff} observer actions differ, max |obs diff| = {obs_shift:.3e})")
    ok &= stage_equal and obs_equal and len(common) > 0 and action_diff > 0
    rep.check("3 action stream vs partner/reset streams", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 4 - z layout, placeholder zeros, frozen normalization, identical arm shapes
# --------------------------------------------------------------------------------------------
def check_4(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    table, source = ctx["table"], ctx["table_source"]
    z_table = np.asarray(ctx["z_table"], dtype=np.float64)

    lines.append(f"profile table: {source} (fixture={ctx['is_fixture']}), metrics "
                 f"{list(table['metric_names'])}, stages {list(table['stages'])}")
    lines.append(f"z layout: {G.PROFILE_DIM} dims = 4 identities x 3 metrics, identity order "
                 f"{list(G.PARTNER_IDENTITIES)} ({table['z_layout']})")
    ok &= G.PROFILE_DIM == 12 and z_table.shape == (4, 3, 3)

    mean = np.asarray(table["normalization"]["mean"], dtype=np.float64)
    std = np.asarray(table["normalization"]["std"], dtype=np.float64)
    recomputed_mean, recomputed_std = G.normalization_from_raw(table["raw"])
    mean_ok = float(np.max(np.abs(mean - np.asarray(recomputed_mean)))) == 0.0
    std_ok = float(np.max(np.abs(std - np.asarray(recomputed_std)))) == 0.0
    lines.append(f"normalization recomputed from the 12 candidates equals the stored table: "
                 f"mean {mean_ok}, std {std_ok}; sha256 {G.sha256_file(source)[:16]}...")
    ok &= mean_ok and std_ok

    max_layout_error = 0.0
    for ident in G.PARTNER_IDENTITIES:
        for stage in G.STAGE_UPDATES:
            rebuilt = (np.asarray(table["raw"][str(ident)][str(stage)], dtype=np.float64) - mean) / std
            combo = [G.STAGE_UPDATES.index(stage) if i == ident else 0 for i in G.PARTNER_IDENTITIES]
            z = np.asarray(G.z_vector_from_table(z_table, combo))
            block = slice(3 * G.PARTNER_IDENTITIES.index(ident),
                          3 * G.PARTNER_IDENTITIES.index(ident) + 3)
            max_layout_error = max(max_layout_error,
                                   float(np.max(np.abs(z[block] - rebuilt.astype(np.float32)))))
    lines.append(f"identity-ordered blocks equal (raw - mean)/std of that identity/stage: max "
                 f"|diff| = {max_layout_error:.3e}")
    ok &= max_layout_error == 0.0

    table2, source2, _ = G.load_profile_table(source)
    frozen = np.array_equal(G.z_table_from_profile_table(table), G.z_table_from_profile_table(table2))
    lines.append(f"re-loading the table gives bitwise identical z ({frozen}); source re-read from "
                 f"{source2}")
    ok &= frozen

    trainer, runner = ctx["trainer"], ctx["runner"]
    transitions = trainer.rollout(runner)[1]
    stage = np.asarray({key: transitions[key] for key in G.DIAGNOSTIC_KEYS}["diag_stage"])  # (T,4,E)
    obs_dim = int(trainer.config["obs_dim"])
    observer_obs = np.asarray(transitions["obs"][:, 0])           # (T, E, obs_dim + 12)
    steps, num_envs = stage.shape[0], stage.shape[2]
    expected = np.zeros((steps, num_envs, G.PROFILE_DIM), dtype=observer_obs.dtype)
    for t in range(steps):
        for e in range(num_envs):
            expected[t, e] = np.concatenate([ctx["z_table"][j, int(stage[t, j, e])]
                                             for j in range(len(G.PARTNER_IDENTITIES))])
    z_error = float(np.max(np.abs(observer_obs[:, :, obs_dim:] - expected)))
    lines.append(f"profile arm: the observer's last {G.PROFILE_DIM} input columns equal the table's "
                 f"z for the episode's combination, at every step: max |diff| = {z_error:.3e}")
    ok &= z_error == 0.0

    placeholder = G.ObserverGateTrainer(ctx["config"], ctx["partner_stack"], ctx["z_table"],
                                        arm="placeholder")
    placeholder.collect_diagnostics = True
    ph_runner = placeholder.init_runner(jax.random.PRNGKey(7))
    ph_obs = np.asarray(placeholder.rollout(ph_runner)[1]["obs"][:, 0])
    zero_error = float(np.max(np.abs(ph_obs[:, :, obs_dim:])))
    lines.append(f"placeholder arm: the observer's z channel is exactly zero at every step (max "
                 f"|value| = {zero_error:.3e}) while the input width is {ph_obs.shape[-1]} in both "
                 f"arms")
    ok &= zero_error == 0.0 and ph_obs.shape[-1] == observer_obs.shape[-1]

    profile_init = G.ObserverGateTrainer(ctx["config"], ctx["partner_stack"], ctx["z_table"],
                                         arm="profile").init_runner(jax.random.PRNGKey(0))
    placeholder_init = placeholder.init_runner(jax.random.PRNGKey(0))
    same_count = sum(x.size for x in jax.tree.leaves(profile_init.params[0])) == sum(
        x.size for x in jax.tree.leaves(placeholder_init.params[0]))
    same_shapes = G.tree_bitwise_equal(profile_init.params[0], placeholder_init.params[0])
    lines.append(f"both arms start from the same initial observer parameters (same --seed): "
                 f"{same_shapes}; parameter count equal: {same_count} "
                 f"({sum(x.size for x in jax.tree.leaves(profile_init.params[0]))} parameters)")
    ok &= same_count and same_shapes
    rep.check("4 z layout / placeholder zeros / frozen normalization", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 5 - profile swap with identical hidden states
# --------------------------------------------------------------------------------------------
def check_5(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    trainer, runner = ctx["trainer"], ctx["runner"]
    # Errata 1: the swap needs a combination in which the inspected partner really plays a LATE
    # stage, so its profile can then be replaced by that identity's EARLY one (stale, not future).
    combo = list(E.default_matched_combo(G.enumerate_combos(None), G.PARTNER_IDENTITIES[0]))
    matched = E.collect_matched_states(trainer, runner.params[0], combo, G.TEST_SEEDS_C, steps=4)
    swap = E.swap_analysis(trainer, runner.params[0], matched, combo,
                           swap_identity=G.PARTNER_IDENTITIES[0], stale_stage=0)
    swapped = swap["swap"]
    lines = [
        f"matched states: {swap['matched_states']} (seeds {swap['matched_seeds']} x "
        f"{swap['battle_steps_recorded']} steps of {G.combo_label(combo)})",
        f"swap: partner {swapped['partner_identity']} really plays {swapped['param_stage']} while "
        f"the observer is given its {swapped['profile_stage']} profile -> direction "
        f"{swapped['direction']} (param stage {swapped['param_stage_index']} > profile stage "
        f"{swapped['profile_stage_index']}: {swapped['param_stage_later_than_profile_stage']}), "
        f"z dims {swapped['z_dims_replaced']} (max |dz| {swapped['z_max_abs_change']:.4f})",
        f"hstate injected equal = {swap['hstate_check']['same_hstate_injected']}; still leaf-wise "
        f"equal after both forward passes = "
        f"{swap['hstate_check']['same_hstate_after_both_forwards']}; inputs differ only in z = "
        f"{swap['inputs_differ_only_in_z']}",
        f"recorded z equals the combination's z: {swap['recorded_z_matches_correct_z']}; negative "
        f"control (perturbed hstate) max logit shift = "
        f"{swap['hstate_check']['negative_control_max_logit_shift_from_perturbed_hstate']:.4f}",
        f"action distribution: TV mean {swap['action_distribution']['tv_mean']:.6f} "
        f"(max {swap['action_distribution']['tv_max']:.6f}), JS mean "
        f"{swap['action_distribution']['js_mean_nats']:.6f} nats, argmax switch rate "
        f"{swap['action_distribution']['argmax_switch_rate']:.3f}",
        "interpretation: a changed action distribution only shows the input affects the decision; "
        "it does NOT show the change is beneficial",
    ]
    ok = (swap["hstate_check"]["same_hstate_injected"]
          and swap["hstate_check"]["same_hstate_after_both_forwards"]
          and swap["inputs_differ_only_in_z"]
          and swap["recorded_z_matches_correct_z"]
          and swapped["param_stage_later_than_profile_stage"]
          and swap["hstate_check"]["negative_control_max_logit_shift_from_perturbed_hstate"] > 0.0
          and swap["action_distribution"]["tv_max"] > 0.0)
    rep.check("5 profile swap / identical hstate", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 6 - paired primary metric over 81 x 4 = 324 pairs
# --------------------------------------------------------------------------------------------
def check_6(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    full_combos = G.enumerate_combos(None)
    seeds = list(G.TEST_SEEDS_C)
    n_pairs = len(full_combos) * len(seeds)
    lines.append(f"protocol enumeration: {len(full_combos)} partner-stage combinations x "
                 f"{len(seeds)} test situations {seeds} = {n_pairs} pairs; both arms use the same "
                 f"pairs")
    ok &= len(full_combos) == 81 and n_pairs == 324

    arm_a = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    arm_b = np.array([[10.0, 12.0, 14.0, 16.0], [18.0, 20.0, 22.0, 24.0]])
    paired = E.paired_differences(arm_a, arm_b)          # first arm - second arm, same index
    manual = arm_a - arm_b
    mispaired = arm_a - np.roll(arm_b, 1, axis=1)
    lines.append(f"paired difference is elementwise `first - second` on the SAME (combination, "
                 f"situation) index: {np.array_equal(paired, manual)}; a misaligned pairing "
                 f"(second arm's situations rotated) gives a different result: "
                 f"{not np.array_equal(paired, mispaired)}")
    ok &= np.array_equal(paired, manual) and not np.array_equal(paired, mispaired)

    stats_one = E.bootstrap_mean(paired, samples=500, seed=0)
    stats_two = E.bootstrap_mean(paired, samples=500, seed=0)
    covers = stats_one["bootstrap_ci95_low"] <= stats_one["mean"] <= stats_one["bootstrap_ci95_high"]
    lines.append(f"bootstrap: {stats_one['n']} pairs, mean {stats_one['mean']:+.4f}, "
                 f"{stats_one['bootstrap_samples']} resamples -> sd "
                 f"{stats_one['bootstrap_sd']:.4f}, 95% "
                 f"[{stats_one['bootstrap_ci95_low']:+.4f}, {stats_one['bootstrap_ci95_high']:+.4f}] "
                 f"brackets the mean: {covers}; deterministic for a fixed bootstrap seed: "
                 f"{stats_one == stats_two}")
    ok &= stats_one == stats_two and stats_one["n"] == 8 and covers

    profile_battles = ctx["trainer"].eval_battles(ctx["runner"].params[0], full_combos[:2], seeds,
                                                  z_mode="table")
    placeholder_battles = ctx["trainer"].eval_battles(
        ctx["placeholder_runner"].params[0], full_combos[:2], seeds, z_mode="zero"
    )
    real_paired = E.paired_differences(profile_battles["return"], placeholder_battles["return"])
    manual_real = np.array([[profile_battles["return"][c, s] - placeholder_battles["return"][c, s]
                             for s in range(len(seeds))] for c in range(2)])
    lines.append(f"tiny (2 combinations x {len(seeds)} situations) real evaluation: paired returns "
                 f"shape {real_paired.shape} = {len(seeds) * 2} pairs, identical to a manual "
                 f"per-index subtraction: {np.allclose(real_paired, manual_real)}; "
                 f"profile mean return {profile_battles['return'].mean():.4f}, placeholder "
                 f"{placeholder_battles['return'].mean():.4f}, paired mean "
                 f"{real_paired.mean():+.4f}")
    lines.append(f"primary metric = mean team return; wins stay auxiliary "
                 f"(profile {int(profile_battles['won'].sum())} vs placeholder "
                 f"{int(placeholder_battles['won'].sum())} of "
                 f"{profile_battles['won'].size} battles)")
    ok &= real_paired.shape == (2, len(seeds)) and np.allclose(real_paired, manual_real)
    rep.check("6 paired primary metric / bootstrap", ok, lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 7 - errata 1: the in-episode step index aligns across roll-outs and across episode lengths
# --------------------------------------------------------------------------------------------
def key_records(diagnostics: dict) -> dict:
    """`{(slot, ep, in-episode step): {stream key, partner keys/actions, roll-out step}}`.

    One entry per recorded position, keyed by the *battle identity* of the position, so two arms
    that reach it at different roll-out steps can be compared position by position.
    """
    ep = np.asarray(diagnostics["diag_ep_index"])                  # (T, E)
    t = np.asarray(diagnostics["diag_t_in_episode"])               # (T, E)
    env_key = np.asarray(diagnostics["diag_env_step_key"])         # (T, E, 2)
    partner_key = np.asarray(diagnostics["diag_partner_action_key"])   # (T, 4, E, 2)
    partner_action = np.asarray(diagnostics["diag_partner_action"])    # (T, 4, E)
    rollout_step = np.asarray(diagnostics["diag_rollout_step"])    # (T, E)
    records = {}
    for s in range(ep.shape[0]):
        for slot in range(ep.shape[1]):
            key = (slot, int(ep[s, slot]), int(t[s, slot]))
            records.setdefault(key, {
                "env_key": env_key[s, slot].copy(),
                "partner_key": partner_key[s, :, slot].copy(),
                "partner_action": partner_action[s, :, slot].copy(),
                "rollout_step": int(rollout_step[s, slot]),
            })
    return records


def check_7(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    config = ctx["config"]
    stack, z_table = ctx["partner_stack"], ctx["z_table"]
    # Two arms that differ in BOTH the roll-out length and the episode length: arm A plays a
    # 6-step episode and is rolled out 4 steps at a time (so one episode spans two roll-outs),
    # arm B plays the ordinary 100-step episode and is rolled out 24 steps at a time.
    arm_a = G.ObserverGateTrainer({**config, "rollout_length": 4}, stack, z_table, arm="profile",
                                  env_overrides={"max_steps": 5})
    arm_b = G.ObserverGateTrainer({**config, "rollout_length": 3 * int(config["rollout_length"])},
                                  stack, z_table, arm="profile")
    arm_a.collect_diagnostics = arm_b.collect_diagnostics = True
    runner_a = arm_a.init_runner(jax.random.PRNGKey(11))
    runner_b = arm_b.init_runner(jax.random.PRNGKey(11))

    records_a = {}
    rollouts = []
    for _ in range(3):
        runner_a, transitions, _ = arm_a.rollout(runner_a)
        rollouts.append(key_records({key: transitions[key] for key in G.DIAGNOSTIC_KEYS}))
        records_a.update(rollouts[-1])
    transitions_b = arm_b.rollout(runner_b)[1]  # call the roll-out once, then extract
    records_b = key_records({key: transitions_b[key] for key in G.DIAGNOSTIC_KEYS})

    lines.append(f"arm A: episode length {arm_a.config['horizon'] + 1}, roll-out length "
                 f"{arm_a.config['rollout_length']} -> three roll-outs cover "
                 f"{[f'{len(r)} positions' for r in rollouts]}; arm B: episode length "
                 f"{arm_b.config['horizon'] + 1}, roll-out length "
                 f"{arm_b.config['rollout_length']}")
    lines.append(f"arm A positions: {sorted(records_a)}")
    lines.append(f"arm B positions: {sorted(records_b)}")
    common = sorted(set(records_a) & set(records_b))
    lines.append(f"positions reached by both arms: {common}")
    ok &= len(common) > 0

    lines.append(f"positions of arm A reached in a LATER roll-out than the first (in-episode step "
                 f">= roll-out length {arm_a.config['rollout_length']}): "
                 f"{sorted(k for k in records_a if k[2] >= arm_a.config['rollout_length'])}")

    env_ok, partner_ok, arm_ok = True, True, True
    for key in common:
        slot, ep, t = key
        expected_env = G.env_step_key(jnp.int32(slot), jnp.int32(ep), jnp.int32(t))
        env_ok &= bool(np.array_equal(records_a[key]["env_key"],
                                      np.asarray(jax.device_get(expected_env))))
        for j, ident in enumerate(G.PARTNER_IDENTITIES):
            expected = G.partner_action_key(jnp.int32(slot), jnp.int32(ep), jnp.int32(ident),
                                            jnp.int32(t))
            partner_ok &= bool(np.array_equal(records_a[key]["partner_key"][j],
                                              np.asarray(jax.device_get(expected))))
            partner_ok &= bool(np.array_equal(records_b[key]["partner_key"][j],
                                              np.asarray(jax.device_get(expected))))
        arm_ok &= bool(np.array_equal(records_a[key]["env_key"], records_b[key]["env_key"]))
    lines.append(f"env_step_key(slot, ep, in-episode step) reproduced independently by "
                 f"arm A and arm B: {arm_ok}; equals env_step_key(slot, ep, t) exactly (A): "
                 f"{env_ok}; partner_action_key(slot, ep, identity, t) exactly (A and B): "
                 f"{partner_ok}")
    ok &= env_ok and partner_ok and arm_ok

    # reverse control: the pre-errata convention indexed the streams by the ROLL-OUT step.
    stale_positions, example = [], None
    for key, record in sorted(records_a.items()):
        slot, ep, t = key
        if record["rollout_step"] == t:
            continue
        old = np.asarray(jax.device_get(G.env_step_key(jnp.int32(slot), jnp.int32(ep),
                                                       jnp.int32(record["rollout_step"]))))
        if not np.array_equal(old, record["env_key"]):
            stale_positions.append((key, record["rollout_step"]))
            if example is None:
                duplicate = records_a.get((slot, ep, record["rollout_step"]))
                example = (key, record["rollout_step"],
                           duplicate is not None and np.array_equal(duplicate["env_key"], old))
    lines.append(f"reverse control (old convention, roll-out step {sorted(set(r['rollout_step'] for r in records_a.values()))} "
                 f"instead of the in-episode step): {len(stale_positions)} of {len(records_a)} "
                 f"positions get a different env_step_key, e.g. {example[0] if example else None} "
                 f"would use the key of in-episode step {example[1] if example else None}"
                 f"{' - the same key as an earlier position of the same episode' if example and example[2] else ''}")
    ok &= len(stale_positions) > 0 and bool(example and example[2])

    in_episode = sorted(k[2] for k in records_a)
    lines.append(f"in-episode steps observed in arm A: {in_episode[0]}..{in_episode[-1]} "
                 f"(the episode continues across the roll-out boundary at step "
                 f"{arm_a.config['rollout_length']})")
    rep.check("7 in-episode step index / cross-roll-out alignment", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 8 - errata 2: the partners act by argmax on both paths
# --------------------------------------------------------------------------------------------
def check_8(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    trainer, runner = ctx["trainer"], ctx["runner"]

    probe = np.array([[2.0, 1.0, 0.5], [0.25, 0.25, -1.0], [1.0, 3.0, 2.0]], dtype=np.float32)
    action = np.asarray(G.partner_action_argmax(jnp.asarray(probe)))
    lines.append(f"partner_action_argmax on 3x3 logits (row 1 is a tie) = {action.tolist()}, "
                 f"argmax = {probe.argmax(-1).tolist()}, equal: "
                 f"{np.array_equal(action, probe.argmax(-1))}")
    ok &= np.array_equal(action, probe.argmax(-1))

    # training path: recompute the partners' first action of episode (slot 0, ep 0) independently
    # from the reset state and compare with the action the roll-out actually fed the environment.
    diag = rollout_diagnostics(trainer, runner)
    slot, ep = 0, 0
    obs0, env_state0 = trainer.env.reset(G.env_reset_key(jnp.int32(slot), jnp.int32(ep)))
    avail0 = trainer.base_env.get_avail_actions(env_state0.env_state)
    agents = trainer.env.agents
    obs_stack = jnp.stack([obs0[a] for a in agents]).astype(jnp.float32)
    avail_stack = jnp.stack([avail0[a] for a in agents]).astype(jnp.float32)
    zero_done = jnp.zeros((1, 1), dtype=bool)
    h0 = G.base_train.ScannedRNN.initialize_carry(1, trainer.config["gru_hidden_dim"])
    t0 = int(np.asarray(diag["diag_t_in_episode"])[0, slot])
    expected, recorded, sampled = [], [], []
    for j, ident in enumerate(G.PARTNER_IDENTITIES):
        stage = int(np.asarray(diag["diag_stage"])[0, j, slot])
        _, pi, _ = trainer.network.apply(
            trainer.partner_stack[j][stage], h0,
            (obs_stack[ident][None, None], zero_done, avail_stack[ident][None]),
        )
        expected.append(int(np.asarray(
            jax.device_get(G.partner_action_argmax(pi.logits)))[0, 0]))
        recorded.append(int(np.asarray(diag["diag_partner_action"])[0, j, slot]))
        key = G.partner_action_key(jnp.int32(slot), jnp.int32(ep), jnp.int32(ident), jnp.int32(t0))
        sampled.append(int(np.asarray(jax.device_get(
            G.categorical_sample(key, pi.logits[0, 0])))))
    lines.append(f"training roll-out, first step of (slot {slot}, episode {ep}, in-episode step "
                 f"{t0}): independently recomputed argmax {expected} == recorded partner actions "
                 f"{recorded}: {expected == recorded}; the pre-errata sampling rule would have "
                 f"played {sampled} ({sum(a != b for a, b in zip(sampled, expected))}/"
                 f"{len(expected)} identities differ)")
    ok &= expected == recorded

    # evaluation path: the battle result follows the argmax helper; replacing it by the pre-errata
    # sampling rule changes the trajectory, so the battle really does use that helper.
    short, short_runner = ctx["short"], ctx["short_runner"]
    partner_params = tuple(trainer.partner_stack[j][0] for j in range(len(G.PARTNER_IDENTITIES)))
    z = trainer.z_vector(jnp.asarray(np.zeros(len(G.PARTNER_IDENTITIES), np.int32)))
    seed = jnp.int32(ctx["seeds"][0])
    battle = np.asarray(jax.device_get(
        short._battle_episode(short_runner.params[0], partner_params, z, seed)))

    def sampled_rule(logits):
        keys = jnp.broadcast_to(G.stream_key(G.EVAL_ACTION_SEED, 0, 0),
                                tuple(logits.shape[:-1]) + (2,))
        return G.categorical_sample(keys, logits)

    original = G.partner_action_argmax
    G.partner_action_argmax = sampled_rule
    try:
        sampled_battle = np.asarray(jax.device_get(
            short._battle_episode(short_runner.params[0], partner_params, z, seed)))
    finally:
        G.partner_action_argmax = original
    lines.append(f"gate battle (seed {int(ctx['seeds'][0])}, {G.combo_label([0,0,0,0])}): "
                 f"length/return/won/terminated = {battle.tolist()}; with the pre-errata sampling "
                 f"rule swapped back in it becomes {sampled_battle.tolist()} - changed: "
                 f"{not np.array_equal(battle, sampled_battle)}")
    ok &= not np.array_equal(battle, sampled_battle)

    # the matched-state path of the §5 check uses the same argmax rule (the observer's recorded
    # inputs do not depend on the partner stream any more).
    matched_a = E.collect_matched_states(trainer, runner.params[0], [0, 0, 0, 0],
                                         ctx["seeds"][:2], steps=3)
    saved = G.EVAL_ACTION_SEED
    G.EVAL_ACTION_SEED = saved + 4242
    try:
        matched_b = E.collect_matched_states(trainer, runner.params[0], [0, 0, 0, 0],
                                             ctx["seeds"][:2], steps=3)
    finally:
        G.EVAL_ACTION_SEED = saved
    same = (np.array_equal(matched_a["obs"], matched_b["obs"])
            and np.array_equal(matched_a["hstate"], matched_b["hstate"]))
    lines.append(f"matched-state rollouts are invariant to the partner-action seed ("
                 f"EVAL_ACTION_SEED {saved} vs {saved + 4242}): {same} - the partner stream is not "
                 f"sampled from any more")
    ok &= same
    rep.check("8 partners act by argmax (training and evaluation)", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 9 - errata 3: the profile swap is STALE (late parameters, early profile)
# --------------------------------------------------------------------------------------------
def check_9(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    trainer, runner = ctx["trainer"], ctx["runner"]
    combos = G.enumerate_combos(None)
    identity = G.PARTNER_IDENTITIES[0]
    combo = list(E.default_matched_combo(combos, identity))
    lines.append(f"default matched-state combination: {G.combo_label(combo)}; the lexicographic "
                 f"first combination ({G.combo_label(combos[0])}) is all-u50 and can only produce a "
                 f"FUTURE profile swap, so it is no longer the default")

    matched = E.collect_matched_states(trainer, runner.params[0], combo, G.TEST_SEEDS_C, steps=2)
    swap = E.swap_analysis(trainer, runner.params[0], matched, combo, identity, 0)
    swapped = swap["swap"]
    j = G.PARTNER_IDENTITIES.index(swapped["partner_identity"])
    lines.append(f"actual swap: identity {swapped['partner_identity']} "
                 f"(requested {swapped['requested_identity']}, flipped "
                 f"{swapped['identity_flipped']}) | param stage {swapped['param_stage']} "
                 f"(index {swapped['param_stage_index']}) | profile stage "
                 f"{swapped['profile_stage']} (index {swapped['profile_stage_index']}) | direction "
                 f"{swapped['direction']} | param later than profile: "
                 f"{swapped['param_stage_later_than_profile_stage']} | z dims "
                 f"{swapped['z_dims_replaced']}")
    lines.append(f"the inspected partner really plays that stage in the combination: "
                 f"combo[{j}]={int(combo[j])} == param stage {swapped['param_stage_index']}")
    ok &= (swapped["direction"] == "stale"
           and swapped["param_stage_index"] > swapped["profile_stage_index"]
           and swapped["param_stage_later_than_profile_stage"]
           and int(combo[j]) == swapped["param_stage_index"])

    # every combination: the resolved swap is stale, and the only combination that cannot host one
    # is all-u50 (no identity plays a stage later than u50) - refused instead of silently flipped.
    future, refused, flipped = [], [], 0
    for combo_i in combos:
        for ident in G.PARTNER_IDENTITIES:
            try:
                picked, param, profile = E.resolve_swap_target(combo_i, ident, 0)
            except SystemExit:
                refused.append((G.combo_label(combo_i), ident))
                continue
            k = G.PARTNER_IDENTITIES.index(picked)
            if not param > profile or int(combo_i[k]) != param:
                future.append((G.combo_label(combo_i), ident, picked, param, profile))
            flipped += int(picked != ident)
    all_u50 = [(label, ident) for label, ident in refused if label == G.combo_label([0, 0, 0, 0])]
    lines.append(f"all {len(combos)} combinations x {len(G.PARTNER_IDENTITIES)} identities: future "
                 f"swaps {len(future)}; refused as impossible-to-be-stale {sorted(set(refused))} "
                 f"(only the all-u50 combination, {len(all_u50)} requests); automatic identity "
                 f"flips {flipped}")
    ok &= not future and sorted(set(refused)) == sorted(set(all_u50)) and flipped > 0

    refused_message = None
    try:
        E.resolve_swap_target(combos[0], identity, 0)
    except SystemExit as exc:
        refused_message = str(exc)
    lines.append(f"reverse control (the erratum's default, all-u50 parameters): "
                 f"resolve_swap_target refuses it instead of returning u1250 as the 'stale' "
                 f"profile: {refused_message is not None}")
    if refused_message:
        lines.append(f"  raised: {refused_message}")
    ok &= refused_message is not None and "FUTURE" in (refused_message or "")
    rep.check("9 swap direction is stale (late params, early profile)", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# check 10 - errata 4: battles run to real termination, truncation is a failure
# --------------------------------------------------------------------------------------------
def check_10(rep: Reporter, ctx: dict) -> None:
    started = time.perf_counter()
    lines, ok = [], True
    short, short_runner = ctx["short"], ctx["short_runner"]
    max_steps = int(short.config["horizon"])
    combos = G.enumerate_combos(2)
    seeds = list(G.TEST_SEEDS_C)
    out = short.eval_battles(short_runner.params[0], combos, seeds, z_mode="table")
    sizes, counts = np.unique(out["length"], return_counts=True)
    histogram = {int(k): int(v) for k, v in zip(sizes, counts)}
    lines.append(f"short env: max_steps={max_steps}, loop guard eval_step_limit="
                 f"{short.eval_step_limit} (= max_steps+1); measured battle length distribution "
                 f"(length: battles) {histogram} over {out['length'].size} battles")
    lines.append(f"battles that reached done['__all__']: "
                 f"{int((~out['truncated']).sum())}/{out['length'].size}; longest battle "
                 f"{float(out['length'].max()):.0f} steps - the pre-errata bound (max_steps="
                 f"{max_steps}) would have cut that last step off and returned it as a finished "
                 f"battle")
    ok &= bool((~out["truncated"]).all()) and float(out["length"].max()) == float(max_steps + 1)

    full = ctx["trainer"]
    real = ctx["trainer"].eval_battles(ctx["runner"].params[0], G.enumerate_combos(1),
                                       [G.TEST_SEEDS_C[0]], z_mode="table")
    lines.append(f"contract configuration: env max_steps={full.config['horizon']} -> "
                 f"eval_step_limit={full.eval_step_limit} (= max_steps+1); a real battle of "
                 f"{G.combo_label(G.enumerate_combos(1)[0])} on situation {G.TEST_SEEDS_C[0]} "
                 f"ended after {real['length'].ravel().tolist()} steps, terminated "
                 f"{bool((~real['truncated']).all())} (shorter than the time limit only if a team "
                 f"was wiped out)")
    ok &= full.eval_step_limit == int(full.config["horizon"]) + 1
    ok &= bool((~real["truncated"]).all())

    saved = short.eval_step_limit
    short.eval_step_limit = max_steps
    message = None
    try:
        short.eval_battles(short_runner.params[0], combos[:1], seeds[:1], z_mode="table")
    except SystemExit as exc:
        message = str(exc)
    finally:
        short.eval_step_limit = saved
    lines.append(f"reverse control: with the guard lowered to max_steps={max_steps} the same "
                 f"battle does not terminate inside the loop and is reported as a TRUNCATED "
                 f"failure instead of a result: {message is not None}")
    if message:
        lines.append(f"  raised: {message}")
    ok &= message is not None and "TRUNCATED" in (message or "")
    rep.check("10 battles run to real termination / truncation detected", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
def main() -> int:
    args = G.parse_args(["--tiny", "--arm", "profile"])
    config = G.base_train.build_config(args)
    partner_dir = Path(args.partner_run_dir)
    partner_dir = partner_dir if partner_dir.is_absolute() else (HERE / partner_dir)
    partner_stack = G.load_partner_stack(partner_dir, args.partner_stages)
    table, table_source, is_fixture = G.load_profile_table(args.profiles)
    z_table = G.z_table_from_profile_table(table)

    trainer = G.ObserverGateTrainer(config, partner_stack, z_table, arm="profile")
    trainer.collect_diagnostics = True
    runner = trainer.init_runner(jax.random.PRNGKey(7))
    placeholder = G.ObserverGateTrainer(config, partner_stack, z_table, arm="placeholder")
    placeholder.collect_diagnostics = True
    placeholder_runner = placeholder.init_runner(jax.random.PRNGKey(7))
    # A short environment (max_steps=5 -> 6-step episodes) keeps the termination checks cheap; the
    # local environment flags `done` before it increments its own step counter, so 5 is exactly the
    # bound that used to cut the last step off.
    short = G.ObserverGateTrainer(config, partner_stack, z_table, arm="profile",
                                  env_overrides={"max_steps": 5})
    short.collect_diagnostics = True
    short_runner = short.init_runner(jax.random.PRNGKey(7))

    ctx = {
        "config": config, "trainer": trainer, "runner": runner, "placeholder": placeholder,
        "placeholder_runner": placeholder_runner, "partner_stack": partner_stack,
        "partner_dir": partner_dir, "stages": args.partner_stages, "table": table,
        "table_source": table_source, "is_fixture": is_fixture, "z_table": z_table,
        "short": short, "short_runner": short_runner, "seeds": list(G.TEST_SEEDS_C),
    }
    rep = Reporter()
    print(f"tiny config: num_envs={config['num_envs']} rollout_length={config['rollout_length']} "
          f"updates={config['total_updates']} segment_updates={config['segment_updates']} "
          f"observer_input_dim={trainer.config['observer_input_dim']} "
          f"partners={list(G.PARTNER_IDENTITIES)} partner_stages={list(args.partner_stages)}")
    check_1(rep, ctx)
    check_2(rep, ctx)
    check_3(rep, ctx)
    check_4(rep, ctx)
    check_5(rep, ctx)
    check_6(rep, ctx)
    check_7(rep, ctx)
    check_8(rep, ctx)
    check_9(rep, ctx)
    check_10(rep, ctx)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
