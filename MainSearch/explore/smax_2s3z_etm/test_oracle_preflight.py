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
                        swapped z does change the action distribution (TV > 0);
  6. paired protocol  - the primary statistic is a per-`(combination, situation)` paired
                        difference over 81 x 4 = 324 pairs, with the bootstrap resampling those
                        pairs (wins stay auxiliary).

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
    combo = [0, 0, 0, 0]
    matched = E.collect_matched_states(trainer, runner.params[0], combo, G.TEST_SEEDS_C, steps=4)
    swap = E.swap_analysis(trainer, runner.params[0], matched, combo,
                           swap_identity=G.PARTNER_IDENTITIES[0], stale_stage=0)
    lines = [
        f"matched states: {swap['matched_states']} (seeds {swap['matched_seeds']} x "
        f"{swap['battle_steps_recorded']} steps of {G.combo_label(combo)})",
        f"swap: partner {swap['swap']['partner_identity']} "
        f"{swap['swap']['correct_stage']} -> stale {swap['swap']['stale_stage']}, z dims "
        f"{swap['swap']['z_dims_replaced']} (max |dz| {swap['swap']['z_max_abs_change']:.4f})",
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

    ctx = {
        "config": config, "trainer": trainer, "runner": runner, "placeholder": placeholder,
        "placeholder_runner": placeholder_runner, "partner_stack": partner_stack,
        "partner_dir": partner_dir, "stages": args.partner_stages, "table": table,
        "table_source": table_source, "is_fixture": is_fixture, "z_table": z_table,
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
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
