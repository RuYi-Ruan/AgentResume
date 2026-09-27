"""Preflight checks for `train_smax_2s3z_independent.py`.

Run with the absolute interpreter path:
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe test_preflight.py

Every check runs the real trainer code (the same jitted roll-out / update / evaluation used by the
CLI) and prints the measured numbers.  Checks:

  (a) independent update      - updating identity i only leaves every other identity's params and
                                optimizer state bitwise unchanged; the target identity does move;
                                a full update moves all five identities to pairwise different
                                parameter trees.
  (b) learning-rate timeline  - the LR optax actually uses at optimizer step n versus the official
                                analytic form lr0*(1 - floor(n / (minibatches * epochs)) / updates);
                                also checks that the LR is constant inside a PPO update and drops
                                by exactly lr0/updates at update boundaries.
  (c) checkpoint resume       - "2 segments + save + load + 2 segments" versus "4 segments straight
                                through": params diff, optimizer-state (Adam m/v + count) diff and
                                update-counter consistency.
  (d) evaluation isolation    - evaluation changes neither params nor optimizer state nor the
                                carried RNG, and it does read the current parameters.
  (e) GRU timing              - per-identity hidden states after 2 rollout steps differ from the
                                initial state and from each other; the hidden state handed to the
                                loss is exactly the roll-out-start hidden state (recomputed
                                independently), and using a wrong hidden state changes the loss.
  (f) CLI cross-process       - optional: compares the checkpoint of a CLI run that resumed from
                                update 2 with the checkpoint of an uninterrupted CLI run.
  (g) metric attribution      - numpy re-implementation of the per-identity damage/kill attribution
                                rule against the environment's own health bookkeeping.
  (h) in-roll-out reset       - roll-out longer than the episode limit: the hidden state is reset
                                mid-sequence and the loss still replays from the pre-roll-out state.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import train_smax_2s3z_independent as T  # noqa: E402

# Directories of the two CLI runs used by check (f) (informational unless present).
CLI_RUN_DIR = HERE / "results" / "preflight_cli_a"
CLI_RESUMED_DIR = HERE / "results" / "preflight_cli_b"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def leaf_max_diff(a, b) -> float:
    """Max absolute difference over all leaves (0.0 for identical trees)."""
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb), "different leaf counts"
    if not la:
        return 0.0
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(la, lb))


def tree_equal(a, b) -> bool:
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


def int_leaves(state):
    return [int(np.asarray(x)) for x in jax.tree.leaves(state)
            if jnp.issubdtype(jnp.asarray(x).dtype, jnp.integer)]


def float_leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)
            if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating)]


def opt_state_diff(a, b) -> float:
    """Max abs difference over the floating-point leaves of an optimizer state (Adam m/v etc.)."""
    return leaf_max_diff(float_leaves(a), float_leaves(b))


def opt_component_diff(a, b, needle: str) -> tuple[float, int]:
    """Max abs difference over the optimizer-state leaves whose path contains `needle`.

    Used to report Adam's first moment (`mu`) and second moment (`nu`) separately.
    """
    paths_a = jax.tree_util.tree_flatten_with_path(a)[0]
    paths_b = jax.tree_util.tree_flatten_with_path(b)[0]
    assert len(paths_a) == len(paths_b)
    worst, count = 0.0, 0
    for (path_a, x), (path_b, y) in zip(paths_a, paths_b):
        key_a, key_b = jax.tree_util.keystr(path_a), jax.tree_util.keystr(path_b)
        assert key_a == key_b, "optimizer states have different structures"
        if needle in key_a:
            worst = max(worst, float(np.max(np.abs(np.asarray(x) - np.asarray(y)))))
            count += 1
    return worst, count


class Reporter:
    def __init__(self, config: dict, argv: list[str]):
        self.results = []
        print("=" * 100)
        print("SMAX 2s3z independent-identity trainer - preflight")
        print(f"argv: {' '.join(argv)}")
        print(f"config: map={config['map_name']} num_envs={config['num_envs']} "
              f"rollout_length={config['rollout_length']} updates={config['total_updates']} "
              f"segment_updates={config['segment_updates']} ppo_epochs={config['ppo_epochs']} "
              f"num_minibatches={config['num_minibatches']} lr={config['lr']} "
              f"anneal_lr={config['anneal_lr']} eval_episodes={config['eval_episodes']} "
              f"eval_seeds={config['eval_seeds']} seed={config['seed']} "
              f"obs_dim={config['obs_dim']} action_dim={config['action_dim']} "
              f"num_agents={config['num_agents']} gru={config['gru_hidden_dim']} "
              f"total_optimizer_steps={config['total_optimizer_steps']} "
              f"steps_per_update={config['steps_per_update']}")
        print("=" * 100)

    def check(self, name: str, ok: bool, lines: list[str], seconds: float) -> None:
        self.results.append((name, ok))
        print(f"\n[{name}] {'PASS' if ok else 'FAIL'}  ({seconds:.1f}s)")
        for line in lines:
            print(f"    {line}")

    def finish(self) -> int:
        print("\n" + "=" * 100)
        for name, ok in self.results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        failed = [name for name, ok in self.results if not ok]
        print(f"summary: {len(self.results) - len(failed)}/{len(self.results)} checks passed")
        print("=" * 100)
        return 1 if failed else 0


# --------------------------------------------------------------------------------------------
# (a) independent update
# --------------------------------------------------------------------------------------------
def check_a(trainer: T.Trainer, rep: Reporter) -> None:
    started = time.perf_counter()
    num_agents = trainer.num_agents
    runner = trainer.init_runner(jax.random.PRNGKey(11))
    rolled, trans, advantages, returns, hstate_init = trainer.rollout_batch(runner)

    target = 0
    single, losses = trainer.apply_identity_updates(
        rolled, trans, advantages, returns, hstate_init, active=(target,)
    )

    lines = []
    others_ok = True
    for i in range(num_agents):
        if i == target:
            continue
        d_params = leaf_max_diff(single.params[i], rolled.params[i])
        d_opt = opt_state_diff(single.opt_states[i], rolled.opt_states[i])
        same = tree_equal(single.params[i], rolled.params[i]) and tree_equal(
            single.opt_states[i], rolled.opt_states[i]
        )
        others_ok &= same
        lines.append(f"identity {i}: params max|diff|={d_params:.3e} "
                     f"opt_state max|diff|={d_opt:.3e} bitwise_identical={same}")
    target_delta = leaf_max_diff(single.params[0], rolled.params[0])
    target_moved = target_delta > 0.0
    lines.append(f"identity {target} (updated): params max|diff|={target_delta:.3e} "
                 f"loss={float(losses[target]['total_loss']):.4f}")
    untouched = (tree_equal(single.hidden, rolled.hidden)
                 and tree_equal(single.env_state, rolled.env_state)
                 and tree_equal(single.obs, rolled.obs))
    lines.append(f"hidden/env/obs untouched by the update call: {untouched}")

    full, _ = trainer.apply_identity_updates(rolled, trans, advantages, returns, hstate_init)
    changes = [leaf_max_diff(full.params[i], rolled.params[i]) for i in range(num_agents)]
    pairwise = min(leaf_max_diff(full.params[i], full.params[j])
                   for i in range(num_agents) for j in range(i + 1, num_agents))
    lines.append("full update param change per identity: "
                 + ", ".join(f"{c:.3e}" for c in changes))
    lines.append(f"min pairwise max|diff| between identities after a full update: {pairwise:.3e}")
    ok = (others_ok and target_moved and all(c > 0 for c in changes) and pairwise > 0.0
          and untouched)
    rep.check("(a) independent update (only the target identity moves)", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (b) learning-rate timeline
# --------------------------------------------------------------------------------------------
def check_b(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    started = time.perf_counter()
    lr0 = config["lr"]
    updates = config["total_updates"]
    per_update = config["steps_per_update"]
    total_steps = config["total_optimizer_steps"]
    schedule = trainer.lr_schedule
    tx = trainer.tx

    # A one-scalar parameter tree makes the optimizer's effective step size readable: with a
    # constant gradient of 1 and max_grad_norm = 0.25 the clipped gradient is exactly 0.25, so
    # adam's step magnitude is lr * 0.25 / (0.25 + eps) and lr can be recovered exactly.
    params = {"w": jnp.zeros((), dtype=jnp.float32)}
    opt_state = tx.init(params)
    grad = {"w": jnp.ones((), dtype=jnp.float32)}
    clipped_grad = min(1.0, config["max_grad_norm"])
    eps = config["adam_eps"]

    rows, rel_errors, mag_errors = [], [], []
    constant_ok = True
    drop_ok = True
    counts_ok = True
    previous_lr = None
    for n in range(total_steps):
        counts = int_leaves(opt_state)
        count = max(counts)
        counts_ok &= (count == n)
        lr_used = float(schedule(count))
        analytic = lr0 * (1.0 - (n // per_update) / updates)
        rel = abs(lr_used - analytic) / max(abs(analytic), 1e-30)
        rel_errors.append(rel)
        updates_, opt_state = tx.update(grad, opt_state, params)
        params = jax.tree.map(lambda p, u: p + u, params, updates_)
        magnitude = float(np.abs(np.asarray(jax.tree.leaves(updates_)[0])))
        lr_observed = magnitude * (clipped_grad + eps) / clipped_grad
        mag_errors.append(abs(lr_observed - analytic) / max(abs(analytic), 1e-30))
        if previous_lr is not None:
            if n % per_update == 0:
                drop_ok &= np.isclose(lr_used, previous_lr - lr0 / updates, rtol=1e-5, atol=1e-15)
            else:
                constant_ok &= (lr_used == previous_lr)
        previous_lr = lr_used
        if n < per_update + 1 or n == total_steps - 1:
            rows.append(f"opt_step n={n:2d} (update {n // per_update}) "
                        f"schedule(count={count})={lr_used:.9e} "
                        f"analytic={analytic:.9e} rel_err={rel:.2e} "
                        f"observed_step_magnitude={magnitude:.6e} -> lr_observed={lr_observed:.9e} "
                        f"rel_err={mag_errors[-1]:.2e}")

    max_rel = max(rel_errors)
    max_mag_rel = max(mag_errors)
    ok = (max_rel < 1e-6) and counts_ok and constant_ok and drop_ok
    lines = [
        f"N = updates*epochs*minibatches = {updates}*{config['ppo_epochs']}*"
        f"{config['num_minibatches']} = {total_steps} optimizer steps; "
        f"steps per PPO update = {per_update}",
        f"analytic form: lr(n) = lr0*(1 - floor(n/{per_update})/{updates})",
        *rows,
        f"max relative error schedule-vs-analytic over all {total_steps} optimizer steps: "
        f"{max_rel:.3e}",
        f"max relative error observed-effective-LR-vs-analytic (float32 + Adam eps): "
        f"{max_mag_rel:.3e}",
        f"optax counter equals n at every step: {counts_ok}",
        f"LR constant inside each PPO update: {constant_ok}; "
        f"drops by exactly lr0/updates={lr0 / updates:.6e} at update boundaries: {drop_ok}",
    ]
    rep.check("(b) learning-rate timeline (official piecewise-constant formula)", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (c) checkpoint / resume
# --------------------------------------------------------------------------------------------
def check_c(trainer: T.Trainer, config: dict, rep: Reporter, out_dir: Path) -> None:
    started = time.perf_counter()
    segments = 2
    updates_per_segment = config["segment_updates"]

    continuous = trainer.init_runner(jax.random.PRNGKey(config["seed"]))
    continuous, _ = trainer.train_updates(continuous, updates_per_segment)
    midpoint = continuous
    continuous, _ = trainer.train_updates(continuous, updates_per_segment)

    interrupted = trainer.init_runner(jax.random.PRNGKey(config["seed"]))
    interrupted, _ = trainer.train_updates(interrupted, updates_per_segment)
    prefix_ok = (leaf_max_diff(interrupted.params, midpoint.params) == 0.0
                 and opt_state_diff(interrupted.opt_states, midpoint.opt_states) == 0.0)

    ckpt = T.save_checkpoint(
        out_dir / "preflight_resume.bin", interrupted,
        {"update_count": int(interrupted.update_count), "segment": 1, "config": config},
    )
    meta = T.read_checkpoint_meta(ckpt)
    template = trainer.init_runner(jax.random.PRNGKey(config["seed"]))
    restored = T.load_checkpoint(ckpt, template)
    restore_ok = (leaf_max_diff(restored.params, interrupted.params) == 0.0
                  and opt_state_diff(restored.opt_states, interrupted.opt_states) == 0.0
                  and int(restored.update_count) == int(interrupted.update_count))
    resumed, _ = trainer.train_updates(restored, updates_per_segment)

    params_diff = leaf_max_diff(resumed.params, continuous.params)
    opt_diff = opt_state_diff(resumed.opt_states, continuous.opt_states)
    mu_diff, mu_leaves = opt_component_diff(resumed.opt_states, continuous.opt_states, "mu")
    nu_diff, nu_leaves = opt_component_diff(resumed.opt_states, continuous.opt_states, "nu")
    counts_a = sorted({c for st in resumed.opt_states for c in int_leaves(st)})
    counts_b = sorted({c for st in continuous.opt_states for c in int_leaves(st)})
    count_ok = (int(resumed.update_count) == int(continuous.update_count)
                and counts_a == counts_b)
    hidden_diff = leaf_max_diff(resumed.hidden, continuous.hidden)
    env_ok = tree_equal(resumed.env_state, continuous.env_state)
    rng_ok = tree_equal(resumed.rng, continuous.rng)
    prefix_updates = updates_per_segment * segments

    lines = [
        f"{segments} segments x {updates_per_segment} updates; continuous ran "
        f"{int(continuous.update_count)} updates, interrupted+resumed ran "
        f"{int(interrupted.update_count)} + {updates_per_segment} = "
        f"{int(resumed.update_count)} updates",
        f"segment-1 prefix of the interrupted run equals the continuous run: {prefix_ok}",
        f"checkpoint file {ckpt.name} ({ckpt.stat().st_size} bytes); stored meta "
        f"update_count={meta['update_count']} segment={meta['segment']}; "
        f"restored state equals the saved state: {restore_ok}",
        f"params   max|leaf diff| (continuous vs save+resume): {params_diff:.3e}",
        f"opt state (Adam m/v) max|leaf diff|:                {opt_diff:.3e} "
        f"[Adam mu: {mu_diff:.3e} over {mu_leaves} leaves; "
        f"Adam nu: {nu_diff:.3e} over {nu_leaves} leaves]",
        f"optimizer step counts: resumed={counts_a} continuous={counts_b} -> equal: {count_ok}",
        f"update_count: resumed={int(resumed.update_count)} "
        f"continuous={int(continuous.update_count)} (expected {prefix_updates})",
        f"hidden state max|leaf diff|: {hidden_diff:.3e}; env_state identical: {env_ok}; "
        f"carried rng identical: {rng_ok}",
    ]
    ok = (params_diff < 1e-5 and opt_diff < 1e-5 and count_ok and prefix_ok and restore_ok
          and env_ok and rng_ok and int(continuous.update_count) == prefix_updates
          and mu_leaves > 0 and nu_leaves > 0)
    rep.check("(c) checkpoint save -> load -> continue == uninterrupted run", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (d) evaluation does not update anything
# --------------------------------------------------------------------------------------------
def check_d(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    started = time.perf_counter()
    runner = trainer.init_runner(jax.random.PRNGKey(23))
    initial = trainer.evaluate(runner)
    runner, _ = trainer.train_updates(runner, config["segment_updates"])

    params_before = jax.tree.map(lambda x: x, runner.params)
    opt_before = jax.tree.map(lambda x: x, runner.opt_states)
    hidden_before = jax.tree.map(lambda x: x, runner.hidden)
    rng_before = runner.rng
    count_before = int(runner.update_count)

    evaluated = trainer.evaluate(runner)
    evaluated_again = trainer.evaluate(runner)

    params_ok = tree_equal(runner.params, params_before)
    opt_ok = tree_equal(runner.opt_states, opt_before)
    hidden_ok = tree_equal(runner.hidden, hidden_before)
    rng_ok = bool(np.array_equal(np.asarray(runner.rng), np.asarray(rng_before)))
    count_ok = int(runner.update_count) == count_before
    deterministic = (evaluated["wins"] == evaluated_again["wins"]
                     and abs(evaluated["mean_return"] - evaluated_again["mean_return"]) == 0.0
                     and abs(evaluated["identities"][0]["damage_dealt"]
                             - evaluated_again["identities"][0]["damage_dealt"]) == 0.0)

    def metric_vector(m: dict) -> np.ndarray:
        vector = [m["win_rate"], m["mean_return"], m["mean_length"]]
        for identity in m["identities"][:3]:
            vector += [identity["damage_dealt"], identity["damage_taken"],
                       identity["alive_steps"], identity["nearest_enemy_dist"]]
        return np.asarray(vector, dtype=np.float64)

    reads_params = not np.allclose(metric_vector(initial), metric_vector(evaluated),
                                   rtol=0.0, atol=0.0)
    opt_diff_after = leaf_max_diff(float_leaves(runner.opt_states), float_leaves(opt_before))
    counts_after = sorted({c for st in runner.opt_states for c in int_leaves(st)})

    lines = [
        f"evaluation ({evaluated['episodes']} fixed episodes, seeds {config['eval_seeds']}): "
        f"won {evaluated['wins']}/{evaluated['episodes']}, "
        f"mean_return={evaluated['mean_return']:.4f}, "
        f"mean_length={evaluated['mean_length']:.2f}",
        f"params unchanged by evaluation: {params_ok} "
        f"(max|diff|={leaf_max_diff(runner.params, params_before):.3e})",
        f"opt_state (Adam m/v + count) unchanged: {opt_ok} "
        f"(max|diff|={opt_diff_after:.3e}, counts={counts_after})",
        f"hidden unchanged: {hidden_ok}; carried rng unchanged: {rng_ok}; "
        f"update_count unchanged: {count_ok} (={count_before})",
        f"repeat evaluation bitwise identical: {deterministic}",
        f"evaluation reads the current parameters (after {config['segment_updates']} updates: "
        f"won {evaluated['wins']}/{evaluated['episodes']} "
        f"mean_return={evaluated['mean_return']:.4f} vs at init won {initial['wins']}/"
        f"{initial['episodes']} mean_return={initial['mean_return']:.4f}): {reads_params}",
    ]
    ok = params_ok and opt_ok and hidden_ok and rng_ok and count_ok and deterministic and reads_params
    rep.check("(d) evaluation is isolated from training state", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (e) GRU timing / hidden-state alignment
# --------------------------------------------------------------------------------------------
def check_e(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    started = time.perf_counter()
    num_agents = trainer.num_agents

    # --- (e1) hidden states after 2 rollout steps: changed, and all identities different -------
    args2 = T.parse_args(["--tiny", "--rollout-length", "2", "--updates", "2",
                          "--segment-updates", "2", "--num-envs", "2", "--eval-episodes", "2"])
    config2 = T.build_config(args2)
    trainer2 = T.Trainer(config2)
    runner2 = trainer2.init_runner(jax.random.PRNGKey(7))
    h_start = runner2.hidden
    after_1, transitions_1, _ = trainer2.rollout(runner2)
    h_1 = after_1.hidden
    after_2, _, _ = trainer2.rollout(after_1)
    h_2 = after_2.hidden

    moved = [leaf_max_diff(h_1[i], h_start[i]) for i in range(num_agents)]
    pairwise_1 = min(leaf_max_diff(h_1[i], h_1[j])
                     for i in range(num_agents) for j in range(i + 1, num_agents))
    carried = [leaf_max_diff(h_2[i], h_1[i]) for i in range(num_agents)]
    reset_flags_ok = tree_equal(transitions_1["done"][0], runner2.last_done)
    initial_hidden_identical = all(tree_equal(h_start[0], h_start[i]) for i in range(num_agents))

    # --- (e2) the hidden state handed to the loss is the roll-out-start hidden state ----------
    # Advance one roll-out first so the carried hidden states are non-zero (a fresh runner starts
    # from exact zeros, which would make the "wrong hidden state" comparison vacuous).
    runner = trainer.init_runner(jax.random.PRNGKey(29))
    advanced, _, _ = trainer.rollout(runner)
    rolled, trans, advantages, returns, hstate_init = trainer.rollout_batch(advanced)
    target = 2
    single, losses = trainer.apply_identity_updates(
        rolled, trans, advantages, returns, hstate_init, active=(target,)
    )
    expected_sub_rng = jax.random.split(rolled.rng)[1]

    def run_update(hstate):
        return trainer.update_one(
            rolled.params[target], rolled.opt_states[target], trans[target], advantages[target],
            returns[target], hstate, expected_sub_rng,
        )

    manual = run_update(hstate_init[target])
    decoy_end = run_update(rolled.hidden[target])
    decoy_zero = run_update(jnp.zeros_like(hstate_init[target]))

    aligned = leaf_max_diff(manual[0], single.params[target]) == 0.0
    hstate_is_start = leaf_max_diff(hstate_init[target], advanced.hidden[target]) == 0.0
    decoy_differs = leaf_max_diff(hstate_init[target], rolled.hidden[target]) > 0.0
    loss_start = float(manual[3]["total_loss"])
    loss_end = float(decoy_end[3]["total_loss"])
    loss_zero = float(decoy_zero[3]["total_loss"])
    end_is_wrong = leaf_max_diff(decoy_end[0], single.params[target]) > 0.0
    zero_is_wrong = leaf_max_diff(decoy_zero[0], single.params[target]) > 0.0

    lines = [
        f"(e1) 2-step roll-out, {num_agents} identities: initial hidden states identical to each "
        f"other (all zeros): {initial_hidden_identical}",
        "(e1) per-identity hidden change over 2 steps (max|leaf diff| vs initial): "
        + ", ".join(f"{d:.4e}" for d in moved),
        f"(e1) min pairwise max|leaf diff| between identities after 2 steps: {pairwise_1:.4e} "
        f"(all distinct: {pairwise_1 > 0 and all(d > 0 for d in moved)})",
        "(e1) hidden carried into the next roll-out (max|leaf diff| step 2 vs step 4): "
        + ", ".join(f"{d:.4e}" for d in carried),
        f"(e1) roll-out reset flags at t=0 equal the runner's carried last_done: {reset_flags_ok}",
        f"(e2) the hstate the trainer hands to the loss is the pre-roll-out hidden state: "
        f"{hstate_is_start} (max|leaf diff| vs runner.hidden={leaf_max_diff(hstate_init[target], advanced.hidden[target]):.3e}); "
        f"it differs from the post-roll-out hidden state: {decoy_differs} "
        f"(max|leaf diff|={leaf_max_diff(hstate_init[target], rolled.hidden[target]):.3e})",
        f"(e2) update with the roll-out-start hidden state reproduces the trainer's update "
        f"bitwise: {aligned} (max|leaf diff|={leaf_max_diff(manual[0], single.params[target]):.3e})",
        f"(e2) loss with the correct hidden state: {loss_start:.6f}; with the post-roll-out "
        f"(wrong) hidden state: {loss_end:.6f} (differs: {loss_end != loss_start}, and it moves "
        f"the parameters differently: {end_is_wrong}); with a zeroed hidden state: {loss_zero:.6f} "
        f"(differs: {loss_zero != loss_start}, moves parameters: {zero_is_wrong})",
    ]
    ok = (all(d > 0 for d in moved) and pairwise_1 > 0.0 and all(d > 0 for d in carried)
          and reset_flags_ok and aligned and hstate_is_start and decoy_differs
          and loss_end != loss_start and end_is_wrong and loss_zero != loss_start and zero_is_wrong
          and initial_hidden_identical)
    rep.check("(e) GRU time axis and hidden-state alignment", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (f) CLI cross-process resume check (uses the artifacts of the two CLI smoke runs)
# --------------------------------------------------------------------------------------------
def check_f(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    started = time.perf_counter()
    straight = CLI_RUN_DIR / "checkpoint_00000004.bin"
    resumed = CLI_RESUMED_DIR / "checkpoint_00000004.bin"
    if not straight.exists() or not resumed.exists():
        rep.check("(f) CLI resume cross-check (informational)", True,
                  [f"skipped: run the two CLI smoke runs first "
                   f"({straight.relative_to(HERE)} / {resumed.relative_to(HERE)} missing)"],
                  time.perf_counter() - started)
        return
    template = trainer.init_runner(jax.random.PRNGKey(config["seed"]))
    a = T.load_checkpoint(straight, template)
    b = T.load_checkpoint(resumed, template)
    params_diff = leaf_max_diff(a.params, b.params)
    opt_diff = opt_state_diff(a.opt_states, b.opt_states)
    lines = [
        f"CLI run A (uninterrupted, --tiny, 2 segments) checkpoint: {straight.name}",
        f"CLI run B (--resume from update 2) checkpoint: {resumed.name}",
        f"update_count A={int(a.update_count)} B={int(b.update_count)}",
        f"params max|leaf diff| A vs B: {params_diff:.3e}",
        f"opt state max|leaf diff| A vs B: {opt_diff:.3e}",
    ]
    rep.check("(f) CLI resume cross-check (cross-process)", params_diff < 1e-5 and opt_diff < 1e-5,
              lines, time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (g) per-identity damage attribution against the environment's own bookkeeping
# --------------------------------------------------------------------------------------------
def check_g(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    """Re-implement the attribution rule of `Trainer._evaluate` in numpy on the raw environment
    state and check that the damage credited to the allies accounts for the damage the environment
    actually applied to the enemies (`state.unit_health` deltas)."""
    started = time.perf_counter()
    base_env, env = trainer.base_env, trainer.env
    num_allies, num_enemies = config["num_allies"], config["num_enemies"]
    num_movement = config["num_movement_actions"]
    units = np.arange(num_allies + num_enemies)

    def episode(key, offset):
        obs, state = env.reset(key)
        dealt = np.zeros(num_allies)
        taken = np.zeros(num_allies)
        raw_enemy_loss = 0.0
        reward_sum = 0.0
        steps = 0
        won = 0
        for t in range(config["horizon"]):
            health_before = np.asarray(state.env_state.state.unit_health)
            alive_before = np.asarray(state.env_state.state.unit_alive)
            avail = base_env.get_avail_actions(state.env_state)
            key, sub = jax.random.split(key)
            actions = {}
            for i, agent in enumerate(env.agents):
                choices = np.flatnonzero(np.asarray(avail[agent]).astype(bool))
                rng = np.random.default_rng(offset * 977 + t * 13 + i)
                actions[agent] = jnp.asarray(rng.choice(choices))
            obs, state, reward, done, info = env.step(sub, state, actions)
            health_after = np.asarray(state.env_state.state.unit_health)
            health_loss = np.clip(health_before - health_after, 0.0, None)
            prev_attack = np.asarray(state.env_state.state.prev_attack_actions)
            target = np.where(units < num_allies, prev_attack,
                              num_allies - 1 - (prev_attack - num_movement))
            fired = prev_attack >= num_movement
            shoots = ((target[:, None] == units[None, :]) & fired[:, None] & alive_before[:, None]
                      & (units[:, None] != units[None, :]))
            shooters = shoots.sum(0)
            share = shoots / np.maximum(shooters, 1)[None, :]
            dealt += (share * health_loss[None, :]).sum(1)[:num_allies]
            taken += (health_loss * (shooters > 0))[:num_allies]
            raw_enemy_loss += float(health_loss[num_allies:].sum())
            reward_sum += float(reward[env.agents[0]])
            steps += 1
            if bool(done["__all__"]):
                won = int(np.asarray(info["returned_won_episode"])[0])
                break
        return dealt, taken, raw_enemy_loss, reward_sum, steps, won

    keys = jax.random.split(jax.random.PRNGKey(4321), 3)
    lines, ok = [], True
    for offset, key in enumerate(keys):
        dealt, taken, raw_loss, reward_sum, steps, won = episode(key, offset)
        gap = raw_loss - dealt.sum()
        relative_gap = (gap / raw_loss) if raw_loss > 0 else 0.0
        ok &= (gap >= -1e-3) and (relative_gap <= 0.02) and (raw_loss <= 0 or dealt.sum() > 0)
        ok &= bool(np.all(dealt >= 0.0) and np.all(taken >= 0.0))
        lines.append(
            f"episode {offset}: steps={steps} won={won} | enemy raw health loss (all causes)="
            f"{raw_loss:.2f}, attributed damage dealt by the allies={dealt.sum():.2f} "
            f"(gap={gap:.2f}, {100 * relative_gap:.2f}% - wall deaths have no shooter) | "
            f"per-ally dealt={np.round(dealt, 2).tolist()} taken={np.round(taken, 2).tolist()} | "
            f"team reward sum={reward_sum:.3f}"
        )
    lines.append("rule checked: the same attack-index mapping and even-split attribution as "
                 "Trainer._evaluate (lines 631-651)")
    rep.check("(g) per-identity damage attribution matches the environment", ok, lines,
              time.perf_counter() - started)


# --------------------------------------------------------------------------------------------
# (h) GRU timing across an episode boundary inside a roll-out
# --------------------------------------------------------------------------------------------
def check_h(trainer: T.Trainer, config: dict, rep: Reporter) -> None:
    """`rollout_length=110 > max_steps=100` guarantees that every environment crosses an episode
    boundary inside the roll-out, i.e. the RNN has to reset the hidden state mid-sequence.  The
    PPO loss must replay exactly the same resets from the *pre-roll-out* hidden state."""
    started = time.perf_counter()
    args = T.parse_args(["--tiny", "--rollout-length", "110", "--num-envs", "2",
                         "--num-minibatches", "2", "--eval-episodes", "2",
                         "--updates", "2", "--segment-updates", "2"])
    config_long = T.build_config(args)
    long_trainer = T.Trainer(config_long)
    runner = long_trainer.init_runner(jax.random.PRNGKey(31))
    # one short roll-out first so the carried hidden state is non-zero (a mid-episode start)
    advanced, _, _ = long_trainer.rollout(runner)
    rolled, trans, advantages, returns, hstate_init = long_trainer.rollout_batch(advanced)

    target = 2
    done_flags = np.asarray(trans[target]["done"])
    global_done = np.asarray(trans[target]["global_done"])
    resets = int(done_flags.sum())
    hstate_nonzero = leaf_max_diff(hstate_init[target], jnp.zeros_like(hstate_init[target])) > 0.0
    hstate_moved = leaf_max_diff(hstate_init[target], rolled.hidden[target]) > 0.0

    single, losses = long_trainer.apply_identity_updates(
        rolled, trans, advantages, returns, hstate_init, active=(target,)
    )
    sub_rng = jax.random.split(rolled.rng)[1]
    manual = long_trainer.update_one(
        rolled.params[target], rolled.opt_states[target], trans[target], advantages[target],
        returns[target], hstate_init[target], sub_rng,
    )
    aligned = leaf_max_diff(manual[0], single.params[target]) == 0.0
    decoy = long_trainer.update_one(
        rolled.params[target], rolled.opt_states[target], trans[target], advantages[target],
        returns[target], rolled.hidden[target], sub_rng,
    )
    decoy_differs = leaf_max_diff(decoy[0], single.params[target]) > 0.0

    lines = [
        f"roll-out length 110 > max_steps 100 -> at least one episode boundary per environment; "
        f"identity {target}: per-unit reset flags seen inside the roll-out = {resets} "
        f"(global done flags = {int(global_done.sum())})",
        f"roll-out-start hidden state is non-zero (mid-episode start): {hstate_nonzero}; it moved "
        f"over the roll-out (max|leaf diff| vs post-roll-out={leaf_max_diff(hstate_init[target], rolled.hidden[target]):.3e}): {hstate_moved}",
        f"loss replay from the pre-roll-out hidden state reproduces the trainer's update bitwise "
        f"across that reset: {aligned} (max|leaf diff|={leaf_max_diff(manual[0], single.params[target]):.3e})",
        f"using the post-roll-out hidden state instead moves the parameters differently: "
        f"{decoy_differs} (max|leaf diff|={leaf_max_diff(decoy[0], single.params[target]):.3e})",
    ]
    ok = resets > 0 and int(global_done.sum()) > 0 and hstate_nonzero and hstate_moved and aligned \
        and decoy_differs
    rep.check("(h) GRU hidden state across an in-roll-out episode boundary", ok, lines,
              time.perf_counter() - started)


def main() -> int:
    argv = ["--tiny"] + sys.argv[1:]
    args = T.parse_args(argv)
    config = T.build_config(args)
    trainer = T.Trainer(config)
    trainer.make_evaluator(args.eval_episodes, args.eval_seeds)
    config["eval_episodes"] = args.eval_episodes
    config["eval_seeds"] = list(args.eval_seeds)
    config["eval_seed"] = args.eval_seeds[0]

    rep = Reporter(config, argv)
    out_dir = HERE / "results" / "preflight"
    out_dir.mkdir(parents=True, exist_ok=True)
    check_a(trainer, rep)
    check_b(trainer, config, rep)
    check_c(trainer, config, rep, out_dir)
    check_d(trainer, config, rep)
    check_e(trainer, config, rep)
    check_f(trainer, config, rep)
    check_g(trainer, config, rep)
    check_h(trainer, config, rep)
    return rep.finish()


if __name__ == "__main__":
    sys.exit(main())
