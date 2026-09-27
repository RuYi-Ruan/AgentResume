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
    - `damage_per_len` = (sum over battles of the damage this identity's **real hits** dealt) /
                         (sum of lengths)
    - `alive_frac`     = (sum over battles of this identity's alive steps) / (sum of lengths)
    - `focus_share`    = (sum of this identity's real hits whose target at least one *other* ally
                          also hit **in the same sub-step**) / (sum of its real hits);
                          0.0 when there is no real hit.
  The episode length counts non-terminal environment steps (the terminal transition is excluded
  from every accumulator, as in v1), and alive steps are counted at the start of an environment
  step.

Attribution (v2 - this is what changed relative to v1)
-----------------------------------------------------
A "real hit" is what the environment itself applies in
`SMAX._world_step` (L657) -> `update_agent_health` (L684, gate at L697-708): the unit
commanded an attack (`attack_action >= num_movement_actions`), the target is inside
`unit_type_attack_ranges[unit_type_i]`, attacker and target are both alive, `i != target`, **and
`unit_weapon_cooldowns[i] <= 0.0`**.  The test is evaluated once per **sub-step** - 8 sub-steps
per environment step (`world_steps_per_env_step`), every sub-step replaying the same decoded joint
action (`step_env_no_decode`, L354-392) with its own sub-step key
(`step_key, world_step_key = jax.random.split(step_key)`, L367).

This script does **not** re-implement the environment: `replicate_env_step` replays
`MultiAgentEnv.step` -> `HeuristicEnemySMAX.step_env` -> `SMAX.step_env_no_decode` with the
environment's own `_world_step` / `_kill_agents_touching_walls` / `_update_dead_agents` /
`_push_units_away` and the environment's own key schedule, and reports the hits it applied.
Two checks run inside the measurement, for every candidate and every one of the 20 battles, and
any mismatch raises instead of writing a table:
  1. **sub-step ledger** (measured: 10 712 - 12 712 checks per candidate, 142 888 in total, all
     exact): the sub-step is the environment's own, so the check compares the pre-sub-step health
     with the health `_world_step` itself returned, i.e. **before**
     `_kill_agents_touching_walls` / `_update_dead_agents` / `_push_units_away` run (health lost to
     walls is therefore never mistaken for an attribution error):
     `clamp(pre_health - sum(raw hit damage), 0) == _world_step.unit_health`, element-wise for all
     10 units and all 8 sub-steps;
  2. **replica == environment** (measured: 1 319 - 1 570 checks per candidate, 17 628 in total):
     the replayed post-state equals the state the real `SMAXLogWrapper.step` produced (health,
     positions, weapon cooldowns, alive) at every non-terminal environment step.
Observations, rewards, terminal flags, episode lengths and wins all come from the real wrapper
step, so the measured trajectories are the same battles as v1 measured; apart from the attribution
rule itself, the only deliberate difference to v1 is the raw-vs-clipped damage accounting spelled
out below.

Damage accounting (stated explicitly, since it defines the ledger check above):
* **No sharing by remaining health.**  Each attacker is credited with the raw damage its own hit
  carried, `unit_type_attacks[unit_type_attacker]`.  A hit that overkills its target is still
  credited in full; damage is never redistributed or clipped down to the health that was left.
* The *applied* (clipped) quantity is therefore ≤ the raw one; the pooled totals of both, and the
  overkill between them, are printed per candidate and stored in `profiles_selfcheck.json`
  (`damage_dealt` = raw, `damage_dealt_applied` = clipped).  `damage_per_len` uses the raw one.
  v1 used the clipped quantity (it read enemy health *deltas*), so part of any damage difference
  between v1 and v2 is this accounting change, not the attribution change.
* `damage_taken` is the raw incoming hit damage (the same convention, symmetric to
  `damage_dealt`); it is provenance, not one of the three contract metrics.

Why v1 was wrong: it credited every ally that *commanded* an attack at an enemy that lost health
anywhere in the whole environment step and split that loss evenly among them, so a unit that was
still on cooldown (or out of range, or shooting a corpse) received a share of a teammate's damage
and was counted as joining the focus (test A in `--selfcheck` exhibits exactly that on a
constructed state).  The *total* enemy-health ledger still balanced - which is why the v1
self-check (the `check_g` rule (g) style check) passed - but the damage was attributed to the
wrong units.  Whether a given profile moves up or down is *not* presupposed anywhere in this
script or its output: both the numerator and the denominator of `focus_share` change, and the
sign is read off the measurement (`--legacy` prints the observed old/new table).

Interpretation constraints (contract S6, repeated on purpose)
------------------------------------------------------------
A separated / valid interface with no measurable benefit only supports "no benefit was found in
this configuration", never "partner information has no value".  No causal attribution.  The
20 executions are not independent samples.  `sd(seed)` is neither a confidence interval nor a
criterion.

Self-check (`--selfcheck`, prints and writes `results/oracle_gate/attribution_selfcheck.json`)
----------------------------------------------------------------------------------------------
* **Test A** (the bug's typical situation): a state is constructed in which two allies are ordered
  onto the same enemy and one of them is still on cooldown.  Asserted: the gated ally is not
  credited with a hit, is not counted as focusing, the ready ally's damage equals the health the
  environment removed, the sub-step ledger balances, and the v1 even-split rule is shown crediting
  *both* allies and calling it focus.  Negative control (`cooldown_gate=False`): the cooldown
  gate is removed from the recomputation, which must make the ledger check fail.
* **Test B** (sub-step ledger on a real rollout): for the first `--selfcheck-steps` environment
  steps of a real battle, every sub-step's health delta must equal the sum of the hit events, and
  the replica must reproduce the wrapper's state.  Negative control: the same rollout with the
  cooldown gate disabled must fail the ledger check.

Usage (absolute interpreter path required on this machine)::

  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      measure_partner_profiles.py --tiny
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      measure_partner_profiles.py --selfcheck
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
# Faithful hit events: the environment's own sub-step attack rule
# --------------------------------------------------------------------------------------------
# `SMAX._world_step` -> `update_agent_health` (L666-731) gates every attack on
#     norm(pos_i - pos[target]) < unit_type_attack_ranges[unit_type_i]
#     & unit_alive[i] & unit_alive[target] & (i != target) & (unit_weapon_cooldowns[i] <= 0.0)
# once per *sub-step* (8 per environment step), then aggregates the per-unit damage with
# `scatter_add` and floors the health at 0.  v1 of this script skipped all of that: it credited
# every ally that commanded an attack at an enemy that lost health anywhere in the environment
# step and split the loss evenly among them, so units that could not fire (cooldown, range, dead
# target) received a share of a teammate's damage and were counted as joining the focus.  The
# functions below recompute the environment's own test on the pre-sub-step state and cross-check
# it against the health `_world_step` itself applied - a mismatch means the recomputation is
# wrong and is raised, never averaged away.
LEDGER_TOL = 1e-3

PER_AGENT_CHANNELS = [
    "damage_dealt",          # raw damage carried by this identity's real hits (contract S1)
    "damage_taken",          # raw damage carried by the hits this identity received
    "kills",                 # enemy deaths credited to it (share of that sub-step's hitters)
    "alive_steps",           # contract S1 `alive_frac`
    "hit_events",            # contract S1 `focus_share` denominator
    "focused_hits",          # contract S1 `focus_share` numerator
    "damage_dealt_applied",  # the same hits counted as the health the environment removed
]
CH_DAMAGE_DEALT = 0
CH_DAMAGE_TAKEN = 1
CH_KILLS = 2
CH_ALIVE_STEPS = 3
CH_HIT_EVENTS = 4
CH_FOCUSED_HITS = 5
CH_DAMAGE_DEALT_APPLIED = 6

# Per-battle diagnostics of the measurement: summed over battles, except `*_max_dev` (max).
DIAG_NAMES = [
    "substep_checks", "substep_mismatches", "substep_max_dev",
    "envstep_checks", "envstep_mismatches", "envstep_max_dev",
    "commanded_attacks", "blocked_by_cooldown", "blocked_by_range_alive_self",
    "ally_hit_events", "raw_damage_total", "applied_damage_total",
    "applied_damage_to_allies", "applied_damage_to_enemies", "enemy_health_loss",
]
DIAG_INDEX = {name: i for i, name in enumerate(DIAG_NAMES)}


def attack_events(smax, state, actions, world_step_key, *, cooldown_gate: bool = True):
    """Recompute the environment's own hit test for one sub-step, and verify it.

    `state` is the pre-sub-step `smax_env.State`, `actions` the decoded
    `(movement_actions, attack_actions)` pair the environment is about to apply and
    `world_step_key` the sub-step key.  Returns `(events, post_state)`, where `post_state` is the
    environment's own `_world_step` output and `events` carries this sub-step's `hit` / `target` /
    `damage` / `focused` / gate counters plus the ledger deviation `dev` (module docstring).
    `cooldown_gate=False` removes the cooldown gate from the *recomputation* only: a deliberately
    wrong attribution whose ledger check has to fail (the self-check's negative control).
    """
    num_allies = smax.num_allies
    num_movement = smax.num_movement_actions
    unit_index = jnp.arange(smax.num_allies + smax.num_enemies)
    attack_action = actions[1]

    # target decoding exactly as `update_agent_health` (L684-695): the two teams number their
    # opponents in opposite orders, and a movement action targets the unit itself
    target = jnp.where(
        unit_index < num_allies,
        attack_action + num_allies - num_movement,
        num_allies - 1 - (attack_action - num_movement),
    )
    target = jnp.where(attack_action < num_movement, unit_index, target)

    commanded = attack_action >= num_movement
    in_range = (
        jnp.linalg.norm(state.unit_positions - state.unit_positions[target], axis=-1)
        < smax.unit_type_attack_ranges[state.unit_types]
    )
    pass_other = in_range & state.unit_alive & state.unit_alive[target] & (unit_index != target)
    cooldown_ready = state.unit_weapon_cooldowns <= 0.0
    gate = cooldown_ready if cooldown_gate else jnp.ones_like(cooldown_ready)
    hit = commanded & pass_other & gate
    damage = jnp.where(hit, smax.unit_type_attacks[state.unit_types], 0.0)

    # the environment's own transition for this sub-step: the ledger reference
    post = smax._world_step(key=world_step_key, state=state, actions=actions)
    applied = jnp.maximum(state.unit_health - post.unit_health, 0.0)  # only hits remove health
    dealt_by_unit = jnp.zeros_like(state.unit_health).at[target].add(damage)
    expected_health = jnp.maximum(state.unit_health - dealt_by_unit, 0.0)
    dev = jnp.max(jnp.abs(expected_health - post.unit_health))

    hits_on_target = jnp.zeros_like(state.unit_health).at[target].add(hit.astype(jnp.float32))
    ally_hits_on_target = jnp.zeros_like(state.unit_health).at[target].add(
        jnp.where(hit & (unit_index < num_allies), 1.0, 0.0)
    )
    events = {
        "hit": hit,
        "focused": hit & (unit_index < num_allies) & (ally_hits_on_target[target] >= 2.0),
        "target": target,
        "damage": damage,
        "raw_taken": dealt_by_unit,     # raw damage carried by the hits landing on this unit
        "applied_taken": applied,       # health `_world_step` actually removed from this unit
        "applied_dealt": jnp.where(
            hit, applied[target] / jnp.maximum(hits_on_target[target], 1.0), 0.0
        ),
        "hits_on_target": hits_on_target,
        "commanded": commanded,
        "blocked_cooldown": commanded & pass_other & ~cooldown_ready,
        "blocked_other": commanded & ~pass_other,
        "pre_alive": state.unit_alive,
        "pre_health": state.unit_health,
        "post_health": post.unit_health,
        "expected_health": expected_health,
        "dev": dev,
    }
    return events, post


def logged_world_steps(smax, key, state, actions, *, cooldown_gate: bool = True):
    """Replay `SMAX.step_env_no_decode` (L354-392) with the environment's own transition
    functions, returning `(final_state, per_sub)` with the real hit events per sub-step.

    `per_sub[name]` keeps the sub-step axis first.  The key schedule is the environment's own
    (`step_key, world_step_key = jax.random.split(step_key)` per sub-step).
    """
    def _one(carry, _):
        state, step_key = carry
        step_key, world_step_key = jax.random.split(step_key)
        events, post = attack_events(smax, state, actions, world_step_key,
                                     cooldown_gate=cooldown_gate)
        state = smax._kill_agents_touching_walls(post)
        state = smax._update_dead_agents(state)
        state = smax._push_units_away(state)
        died = events["pre_alive"] & ~state.unit_alive
        events["killed"] = jnp.where(
            events["hit"],
            died[events["target"]] / jnp.maximum(events["hits_on_target"][events["target"]], 1.0),
            0.0,
        )
        state = state.replace(prev_movement_actions=actions[0], prev_attack_actions=actions[1])
        return (state, step_key), events

    (state, _), per_sub = jax.lax.scan(
        _one, (state, key), None, length=smax.world_steps_per_env_step
    )
    return state, per_sub


def step_stats(per_sub, tol: float = LEDGER_TOL):
    """Pool the sub-step events of one environment step into per-unit sums + ledger scalars."""
    return {
        "damage": per_sub["damage"].sum(axis=0),
        "raw_taken": per_sub["raw_taken"].sum(axis=0),
        "applied_taken": per_sub["applied_taken"].sum(axis=0),
        "applied_dealt": per_sub["applied_dealt"].sum(axis=0),
        "killed": per_sub["killed"].sum(axis=0),
        "hit": per_sub["hit"].sum(axis=0).astype(jnp.float32),
        "focused": per_sub["focused"].sum(axis=0).astype(jnp.float32),
        "commanded": per_sub["commanded"].sum().astype(jnp.float32),
        "blocked_cooldown": per_sub["blocked_cooldown"].sum().astype(jnp.float32),
        "blocked_other": per_sub["blocked_other"].sum().astype(jnp.float32),
        "checks": jnp.asarray(per_sub["dev"].shape[0], dtype=jnp.float32),
        "mismatches": (per_sub["dev"] > tol).sum().astype(jnp.float32),
        "max_dev": per_sub["dev"].max(),
    }


def replicate_env_step(base_env, key, log_state, action_dict, *, cooldown_gate: bool = True):
    """Replay the state transition of `SMAXLogWrapper.step` with the environment's own functions,
    additionally returning the real per-sub-step hit events.

    Key schedule of `MultiAgentEnv.step` -> `HeuristicEnemySMAX.step_env` ->
    `SMAX.step_env_no_decode`: one split for the wrapper's (here unused) auto-reset key, one split
    separating the enemy heuristic from the world step, then one split per sub-step inside the
    scan.  The returned state is the post-state *without* the wrapper's auto-reset, so at a
    terminal environment step it differs from the wrapper's (which resets) by construction.
    """
    smax = base_env._env
    enemy_state = log_state.env_state                     # HeuristicEnemySMAX.State
    jaxmarl_state = enemy_state.state
    enemy_obs = smax.get_obs_unit_list(jaxmarl_state)
    enemy_obs = jnp.array([enemy_obs[a] for a in base_env.enemy_agents])
    world_key, _reset_key = jax.random.split(key)                  # MultiAgentEnv.step
    step_key, action_key = jax.random.split(world_key)             # HeuristicEnemySMAX.step_env
    enemy_actions, _policy_state = base_env.get_enemy_actions(
        action_key, enemy_state.enemy_policy_state, enemy_obs
    )
    enemy_actions = jnp.array([enemy_actions[a] for a in base_env.enemy_agents])
    ally_actions = jnp.array([action_dict[a] for a in base_env.agents])
    enemy_movement, enemy_attack = smax._decode_discrete_actions(enemy_actions)
    ally_movement, ally_attack = smax._decode_discrete_actions(ally_actions)
    decoded = (
        jnp.concatenate([ally_movement, enemy_movement], axis=0),
        jnp.concatenate([ally_attack, enemy_attack], axis=0),
    )
    state, per_sub = logged_world_steps(smax, step_key, jaxmarl_state, decoded,
                                        cooldown_gate=cooldown_gate)
    state = state.replace(
        done=smax.is_terminal(state),
        prev_movement_actions=decoded[0],
        prev_attack_actions=decoded[1],
        step=state.step + 1,
    )
    return state, per_sub


# --------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------
def make_measure_fn(trainer: train.Trainer, num_episodes: int, seeds: Sequence[int],
                    *, cooldown_gate: bool = True):
    """Jitted `(params, keys) -> (team, per_agent, diagnostics)` over `len(seeds) x
    num_episodes` battles.

    `team`        : (S, E, 4)    -> [won, episode_return, episode_length, enemy_health_loss]
    `per_agent`   : (S, E, A, 7) -> `PER_AGENT_CHANNELS`
    `diagnostics` : (S, E, D)    -> `DIAG_NAMES`, per battle

    Each environment step is executed twice on purpose: once through the real
    `SMAXLogWrapper.step` (authoritative observations / rewards / terminal flags, so the battles
    are the ones v1 measured) and once through `replicate_env_step`, which replays the same
    transition with the environment's own sub-step chain and exposes the real hit events.  The
    diagnostics prove the two agree, per sub-step (health ledger) and per environment step (state).
    """
    config = trainer.config
    env, base_env, network = trainer.env, trainer.base_env, trainer.network
    num_agents = config["num_agents"]
    num_allies, num_enemies = config["num_allies"], config["num_enemies"]
    num_units = num_allies + num_enemies
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
                env_state, obs, hidden, live, acc, diag, ep_return, length, won = carry
                smax = env_state.env_state.state
                alive_before = smax.unit_alive
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
                step_key = jax.random.fold_in(key, t)

                # --- the environment's own step: authoritative obs / reward / terminal flag -----
                new_obs, new_env_state, reward, done_new, info = env.step(step_key, env_state,
                                                                         env_act)
                # --- the same transition, replayed through the environment's own sub-step chain -
                replica_state, per_sub = replicate_env_step(base_env, step_key, env_state, env_act,
                                                            cooldown_gate=cooldown_gate)
                stats = step_stats(per_sub)

                real_post = new_env_state.env_state.state
                done_all = done_new["__all__"]
                live_f = live.astype(jnp.float32)

                # check (2): the replica must equal the state the real wrapper produced; the
                # wrapper auto-resets on its terminal step, where the two cannot agree
                envstep_checked = ~done_all
                envstep_dev = jnp.max(jnp.stack([
                    jnp.max(jnp.abs(real_post.unit_health - replica_state.unit_health)),
                    jnp.max(jnp.abs(real_post.unit_positions - replica_state.unit_positions)),
                    jnp.max(jnp.abs(real_post.unit_weapon_cooldowns
                                    - replica_state.unit_weapon_cooldowns)),
                    jnp.max(jnp.abs(real_post.unit_alive.astype(jnp.float32)
                                    - replica_state.unit_alive.astype(jnp.float32))),
                ]))

                # check (1): sub-step health ledger, accumulated in `stats` above
                acc = acc + jnp.stack([
                    jnp.where(live, row, 0.0) for row in (
                        stats["damage"], stats["raw_taken"], stats["killed"],
                        alive_before.astype(jnp.float32), stats["hit"], stats["focused"],
                        stats["applied_dealt"],
                    )
                ])
                live_diag = jnp.where(live, 1.0, 0.0)
                diag = diag.at[DIAG_INDEX["substep_checks"]].add(stats["checks"] * live_diag)
                diag = diag.at[DIAG_INDEX["substep_mismatches"]].add(
                    stats["mismatches"] * live_diag)
                diag = diag.at[DIAG_INDEX["substep_max_dev"]].set(jnp.maximum(
                    diag[DIAG_INDEX["substep_max_dev"]], jnp.where(live, stats["max_dev"], 0.0)))
                diag = diag.at[DIAG_INDEX["envstep_checks"]].add(
                    jnp.where(envstep_checked, live_f, 0.0))
                diag = diag.at[DIAG_INDEX["envstep_mismatches"]].add(
                    jnp.where(envstep_checked & (envstep_dev > LEDGER_TOL), live_f, 0.0))
                diag = diag.at[DIAG_INDEX["envstep_max_dev"]].set(jnp.maximum(
                    diag[DIAG_INDEX["envstep_max_dev"]],
                    jnp.where(envstep_checked, envstep_dev, 0.0)))
                diag = diag.at[DIAG_INDEX["commanded_attacks"]].add(
                    stats["commanded"] * live_diag)
                diag = diag.at[DIAG_INDEX["blocked_by_cooldown"]].add(
                    stats["blocked_cooldown"] * live_diag)
                diag = diag.at[DIAG_INDEX["blocked_by_range_alive_self"]].add(
                    stats["blocked_other"] * live_diag)
                diag = diag.at[DIAG_INDEX["ally_hit_events"]].add(
                    stats["hit"][:num_allies].sum() * live_diag)
                diag = diag.at[DIAG_INDEX["raw_damage_total"]].add(
                    stats["damage"].sum() * live_diag)
                diag = diag.at[DIAG_INDEX["applied_damage_total"]].add(
                    stats["applied_dealt"].sum() * live_diag)
                diag = diag.at[DIAG_INDEX["applied_damage_to_allies"]].add(
                    stats["applied_taken"][:num_allies].sum() * live_diag)
                diag = diag.at[DIAG_INDEX["applied_damage_to_enemies"]].add(
                    stats["applied_taken"][num_allies:].sum() * live_diag)
                # enemy-side ledger over the whole environment step, v1's `raw_loss`: clipped and
                # excluding the terminal step (the wrapper auto-resets there)
                diag = diag.at[DIAG_INDEX["enemy_health_loss"]].add(
                    jnp.sum(jnp.clip(smax.unit_health - real_post.unit_health, 0.0)[num_allies:])
                    * live_diag)

                ep_return = ep_return + jnp.where(live, reward[agents[0]], 0.0)
                length = length + jnp.where(live, 1, 0)
                won = jnp.where(
                    live & done_all,
                    jnp.asarray(info["returned_won_episode"][0], dtype=jnp.float32),
                    won,
                )
                new_carry = (new_env_state, new_obs, tuple(new_hidden), live & ~done_all, acc,
                             diag, ep_return, length, won)
                return new_carry, None

            init = (
                env_state, obs, hidden, jnp.asarray(True),
                jnp.zeros((len(PER_AGENT_CHANNELS), num_units), dtype=jnp.float32),
                jnp.zeros((len(DIAG_NAMES),), dtype=jnp.float32),
                jnp.zeros((), dtype=jnp.float32), jnp.zeros((), dtype=jnp.int32),
                jnp.zeros((), dtype=jnp.float32),
            )
            final = jax.lax.scan(_step, init, jnp.arange(horizon))[0]
            (_, _, _, _, acc, diag, ep_return, length, won) = final
            team = jnp.stack([won, ep_return, length.astype(jnp.float32),
                              diag[DIAG_INDEX["enemy_health_loss"]]])
            per_agent = jnp.swapaxes(acc[:, :num_allies], 0, 1)
            return team, per_agent, diag

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
    dealt = per_agent[:, :, identity, CH_DAMAGE_DEALT]
    taken = per_agent[:, :, identity, CH_DAMAGE_TAKEN]
    kills = per_agent[:, :, identity, CH_KILLS]
    alive = per_agent[:, :, identity, CH_ALIVE_STEPS]
    hits = per_agent[:, :, identity, CH_HIT_EVENTS]
    focused = per_agent[:, :, identity, CH_FOCUSED_HITS]
    applied = per_agent[:, :, identity, CH_DAMAGE_DEALT_APPLIED]

    total_length = float(lengths.sum())
    total_hits = float(hits.sum())
    pooled = [
        float(dealt.sum() / total_length) if total_length else 0.0,
        float(alive.sum() / total_length) if total_length else 0.0,
        float(focused.sum() / total_hits) if total_hits > 0 else 0.0,
    ]
    per_seed = []
    for s, seed in enumerate(seeds):
        seed_length = float(lengths[s].sum())
        seed_hits = float(hits[s].sum())
        per_seed.append({
            "seed": int(seed),
            "episodes": int(lengths.shape[1]),
            "episode_lengths": [int(round(float(x))) for x in lengths[s]],
            "damage_dealt": float(dealt[s].sum()),
            "damage_dealt_applied": float(applied[s].sum()),
            "damage_taken": float(taken[s].sum()),
            "kills": float(kills[s].sum()),
            "alive_steps": float(alive[s].sum()),
            "hit_events": seed_hits,
            "focused_hits": float(focused[s].sum()),
            "damage_per_len": float(dealt[s].sum() / seed_length) if seed_length else 0.0,
            "alive_frac": float(alive[s].sum() / seed_length) if seed_length else 0.0,
            "focus_share": float(focused[s].sum() / seed_hits) if seed_hits > 0 else 0.0,
        })
    return {
        "metrics": pooled,
        "pooled_totals": {
            "total_length": total_length,
            "damage_dealt": float(dealt.sum()),
            "damage_dealt_applied": float(applied.sum()),
            "damage_taken": float(taken.sum()),
            "kills": float(kills.sum()),
            "alive_steps": float(alive.sum()),
            "hit_events": total_hits,
            "focused_hits": float(focused.sum()),
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
# Self-check: the attribution rule, on a controlled state and on a real rollout
# --------------------------------------------------------------------------------------------
def diag_report(diag: np.ndarray) -> dict:
    """Reduce one candidate's per-battle diagnostics ((S, E, D)) to plain numbers."""
    return {
        name: float(diag[..., i].max() if name.endswith("max_dev") else diag[..., i].sum())
        for i, name in enumerate(DIAG_NAMES)
    }


def legacy_split_attribution(state, attack_actions, health_loss, num_allies, num_movement):
    """v1's rule, kept so that the self-check can show what it does on a controlled state.

    Every unit that *commanded* an attack (`attack_action >= num_movement_actions`) at a target
    that lost health anywhere in the environment step got an even share of that loss, and the same
    command counted as a "valid attack" for `focus_share` whenever the target was an enemy that
    lost health.  There is no range / alive / self / cooldown gate and no sub-step resolution.
    """
    num_units = int(attack_actions.shape[0])
    unit_index = jnp.arange(num_units)
    fired = attack_actions >= num_movement
    target = jnp.where(unit_index < num_allies, attack_actions,
                       num_allies - 1 - (attack_actions - num_movement))
    shoots = (
        (target[:, None] == unit_index[None, :]) & fired[:, None] & state.unit_alive[:, None]
        & (unit_index[:, None] != unit_index[None, :])
    )
    shooters = shoots.sum(0)
    share = shoots / jnp.maximum(shooters, 1)[None, :]
    dealt = (share * health_loss[None, :]).sum(1)
    ally_hits_enemy = shoots[:num_allies][:, num_allies:] & (health_loss[num_allies:] > 0)[None, :]
    contested = (shooters[num_allies:] >= 2)[None, :]
    return {
        "damage_dealt": dealt[:num_allies],
        "valid_attacks": ally_hits_enemy.sum(1).astype(jnp.float32),
        "focused_attacks": (ally_hits_enemy & contested).sum(1).astype(jnp.float32),
    }


def selfcheck_test_a(trainer: train.Trainer) -> dict:
    """Test A - the bug's typical situation, without any policy in the loop.

    Two stalkers are placed in range of the same enemy and both are ordered to attack it; the
    first is off cooldown, the second is still on cooldown (`unit_weapon_cooldowns > 0`), which is
    invisible to v1's rule.  The pre-state and the decoded actions go straight into the
    environment's own `_world_step` path (`attack_events`).
    """
    smax = trainer.base_env._env
    num_allies = smax.num_allies
    num_movement = smax.num_movement_actions
    num_units = smax.num_agents
    enemy_0 = num_allies                                    # first enemy unit
    ready, gated = 0, 1                                     # two stalkers (identities 1, 2)
    key = jax.random.PRNGKey(20260927)
    _obs, reset_state = smax.reset(key)

    positions = (reset_state.unit_positions
                 .at[ready].set(jnp.array([10.0, 10.0]))
                 .at[gated].set(jnp.array([10.0, 12.0]))
                 .at[enemy_0].set(jnp.array([11.0, 10.0])))
    cooldowns = jnp.full((num_units,), 100.0).at[ready].set(0.0).at[gated].set(0.4)
    state = reset_state.replace(
        unit_positions=positions,
        unit_health=smax.unit_type_health[reset_state.unit_types],
        unit_alive=jnp.ones((num_units,), dtype=jnp.bool_),
        unit_weapon_cooldowns=cooldowns,
    )

    # both stalkers command "attack the first enemy unit"; every other unit stops
    attack_action = num_movement + enemy_0 - num_allies               # allies number enemies +0
    discrete = jnp.full((num_units,), num_movement - 1, dtype=jnp.int32)
    discrete = discrete.at[ready].set(attack_action).at[gated].set(attack_action)
    movement, attack = smax._decode_discrete_actions(discrete)
    decoded = (movement, attack)

    events, post = attack_events(smax, state, decoded, key, cooldown_gate=True)
    ungated_events, ungated_post = attack_events(smax, state, decoded, key, cooldown_gate=False)
    legacy = legacy_split_attribution(state, attack, events["applied_taken"], num_allies,
                                      num_movement)

    damage_ready = float(events["damage"][ready])
    damage_gated = float(events["damage"][gated])
    health_loss = float(events["pre_health"][enemy_0] - post.unit_health[enemy_0])
    legacy_dealt = np.asarray(legacy["damage_dealt"])
    legacy_valid = float(np.asarray(legacy["valid_attacks"]).sum())
    legacy_focused = float(np.asarray(legacy["focused_attacks"]).sum())

    checks = {
        "gated_ally_not_credited": damage_gated == 0.0 and not bool(events["hit"][gated]),
        "ready_ally_credited": damage_ready == float(smax.unit_type_attacks[state.unit_types[ready]])
                               and bool(events["hit"][ready]),
        "damage_matches_env_health_loss": damage_ready == health_loss,
        "gated_ally_not_focusing": float(events["focused"].sum()) == 0.0,
        "cooldown_gate_counted": float(events["blocked_cooldown"].sum()) == 1.0,
        "substep_ledger_balances": float(events["dev"]) <= LEDGER_TOL,
        # negative control: without the cooldown gate the recomputation invents a second hit and
        # the ledger must reject it
        "ungated_control_invents_hit": float(ungated_events["hit"].sum()) == 2.0,
        "ungated_control_focus_inflated": float(ungated_events["focused"].sum()) == 2.0,
        "ungated_control_ledger_fails": float(ungated_events["dev"]) > LEDGER_TOL,
        # what v1 did on the same state
        "legacy_splits_damage": bool(np.allclose(legacy_dealt[:2], [health_loss / 2.0] * 2)),
        "legacy_counts_both_as_focus": legacy_valid == 2.0 and legacy_focused == 2.0,
    }
    lines = [
        f"constructed state: ally_0 (stalker) at (10,10) cooldown=0.0, ally_1 (stalker) at "
        f"(10,12) cooldown=0.4, enemy unit {enemy_0} (health {float(state.unit_health[enemy_0]):.0f}"
        f", type {int(state.unit_types[enemy_0])}) at (11,10); both ordered to attack it, all "
        f"other units stop",
        f"environment gate (cooldown_gate=True) : ally_0 -> unit_{enemy_0} "
        f"dmg={damage_ready:.1f}, ally_1 -> (blocked) dmg="
        f"{damage_gated:.1f}; focused={float(events['focused'].sum()):.0f}; "
        f"blocked_by_cooldown={float(events['blocked_cooldown'].sum()):.0f}; "
        f"env health loss on unit_{enemy_0}={health_loss:.1f}; ledger max|dev|="
        f"{float(events['dev']):.3e} -> {'OK' if checks['substep_ledger_balances'] else 'FAIL'}",
        f"negative control (cooldown_gate=False): hits = ally_0 + ally_1 -> unit_{enemy_0} "
        f"dmg={float(ungated_events['damage'][ready]):.1f} each; focused="
        f"{float(ungated_events['focused'].sum()):.0f}; ledger max|dev|="
        f"{float(ungated_events['dev']):.3e} -> the claim 'both hit' is rejected because the "
        f"environment removed only {health_loss:.1f}",
        f"v1 even-split rule on the same state  : damage_dealt="
        f"{np.round(legacy_dealt[:num_allies], 2).tolist()}, valid_attacks="
        f"{np.asarray(legacy['valid_attacks'])[:num_allies].astype(int).tolist()}, "
        f"focused_attacks={np.asarray(legacy['focused_attacks'])[:num_allies].astype(int).tolist()}"
        f" (focus_share={legacy_focused / legacy_valid if legacy_valid else 0.0:.2f})",
        f"v2 hit-event rule on the same state   : damage_dealt="
        f"{np.asarray(events['damage'])[:num_allies].tolist()}, hit_events="
        f"{np.asarray(events['hit'])[:num_allies].astype(int).tolist()}, focused_hits="
        f"{np.asarray(events['focused'])[:num_allies].astype(int).tolist()} (focus_share=0.00)",
    ]
    return {"test": "A", "ok": all(checks.values()), "checks": checks, "lines": lines}


def selfcheck_test_b(trainer: train.Trainer, params, *, env_steps: int = 12, seed: int = 2234,
                     cooldown_gate: bool = True, max_print: int = 8) -> dict:
    """Test B - the sub-step health ledger on a real rollout.

    The real `SMAXLogWrapper.step` drives the battle (so this is literally the first measured
    battle of seed 2234), the logged replica replays the same transition from the same pre-state,
    and every sub-step must satisfy `maximum(pre_health - real hit damage, 0) == post_health`
    while the replica reproduces the wrapper's state.
    """
    config = trainer.config
    env, base_env, network = trainer.env, trainer.base_env, trainer.network
    agents = env.agents
    num_agents = config["num_agents"]
    num_movement = config["num_movement_actions"]
    key = jax.random.split(jax.random.PRNGKey(int(seed)), config.get("eval_episodes", 4))[0]

    obs, env_state = env.reset(key)
    hidden = tuple(train.ScannedRNN.initialize_carry(1, config["gru_hidden_dim"])
                   for _ in range(num_agents))

    substep_checks = substep_mismatches = 0
    substep_max_dev = 0.0
    envstep_checks = envstep_mismatches = 0
    envstep_max_dev = 0.0
    lines: list[str] = []
    first_mismatch: list[str] = []
    ts = 0
    for t in range(int(env_steps)):
        ts = t
        avail = base_env.get_avail_actions(env_state.env_state)
        obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
        avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)
        actions, new_hidden = [], []
        for i in range(num_agents):
            h_i, pi_i, _ = network.apply(
                params[i], hidden[i],
                (obs_stack[i][None, None], jnp.zeros((1, 1), dtype=bool), avail_stack[i][None]),
            )
            logits = jnp.where(avail_stack[i][None] > 0, pi_i.logits, -1e9)
            actions.append(jnp.argmax(logits, axis=-1)[0, 0])
            new_hidden.append(h_i)
        env_act = {a: actions[i] for i, a in enumerate(agents)}
        step_key = jax.random.fold_in(key, t)
        new_obs, new_env_state, _reward, done_new, _info = env.step(step_key, env_state, env_act)
        replica, per_sub = replicate_env_step(base_env, step_key, env_state, env_act,
                                              cooldown_gate=cooldown_gate)
        pre = np.asarray(jax.device_get(per_sub["pre_health"]))
        post = np.asarray(jax.device_get(per_sub["post_health"]))
        dmg = np.asarray(jax.device_get(per_sub["damage"]))
        target = np.asarray(jax.device_get(per_sub["target"]))
        hit = np.asarray(jax.device_get(per_sub["hit"]))
        devs = np.asarray(jax.device_get(per_sub["dev"]))
        substep_checks += int(devs.size)
        substep_mismatches += int((devs > LEDGER_TOL).sum())
        substep_max_dev = max(substep_max_dev, float(devs.max()))

        done_all = bool(np.asarray(done_new["__all__"]))
        if not done_all:
            real = new_env_state.env_state.state
            dev_state = max(
                float(np.max(np.abs(np.asarray(real.unit_health)
                                    - np.asarray(replica.unit_health)))),
                float(np.max(np.abs(np.asarray(real.unit_positions)
                                    - np.asarray(replica.unit_positions)))),
                float(np.max(np.abs(np.asarray(real.unit_weapon_cooldowns)
                                    - np.asarray(replica.unit_weapon_cooldowns)))),
            )
            envstep_checks += 1
            envstep_mismatches += int(dev_state > LEDGER_TOL)
            envstep_max_dev = max(envstep_max_dev, dev_state)

        for s in range(devs.shape[0]):
            events = ", ".join(
                f"unit_{i}->unit_{int(target[s, i])} dmg={dmg[s, i]:.1f} "
                f"hp {pre[s, int(target[s, i])]:.1f}->{post[s, int(target[s, i])]:.1f}"
                for i in np.flatnonzero(hit[s])
            )
            line = (f"t={t:3d} sub={s} dev={devs[s]:.3e} "
                    + (events if events else "(no hit)"))
            if len(lines) < max_print:
                lines.append(line)
            if devs[s] > LEDGER_TOL and not first_mismatch:
                first_mismatch.append(line)

        obs, env_state, hidden = new_obs, new_env_state, tuple(new_hidden)
        if done_all:
            break

    ok = substep_mismatches == 0 and envstep_mismatches == 0 and substep_checks > 0
    lines.append(
        f"rolled {ts + 1} environment steps x 8 sub-steps: sub-step ledger checks="
        f"{substep_checks} mismatches={substep_mismatches} max|dev|={substep_max_dev:.3e} | "
        f"replica vs real wrapper (health/positions/cooldowns): checks={envstep_checks} "
        f"mismatches={envstep_mismatches} max|dev|={envstep_max_dev:.3e}"
    )
    if cooldown_gate and not ok:
        lines.append("LEDGER CHECK FAILED - the hit recomputation does not match the environment")
    if first_mismatch:
        lines.append("first failing sub-step: " + first_mismatch[0])
    return {
        "test": "B",
        "cooldown_gate": bool(cooldown_gate),
        "env_steps": ts + 1,
        "substep_checks": substep_checks,
        "substep_mismatches": substep_mismatches,
        "substep_max_dev": substep_max_dev,
        "envstep_checks": envstep_checks,
        "envstep_mismatches": envstep_mismatches,
        "envstep_max_dev": envstep_max_dev,
        "ok": ok,
        "lines": lines,
    }


def run_selfcheck(args) -> int:
    """Run test A and test B (both modes) and write `attribution_selfcheck.json`."""
    started = time.perf_counter()
    run_dir = resolve_run_dir(args.run_dir)
    checkpoints = discover_checkpoints(run_dir)
    reference_update, reference_path = resolve_reference(args.reference_checkpoint, run_dir,
                                                         checkpoints)
    config = load_config(run_dir, checkpoints[-1])
    trainer = train.Trainer(config)
    reference_params = load_checkpoint_params(reference_path)

    print("[attribution self-check] environment hit rule: range / both alive / i != target / "
          "weapon cooldown <= 0, evaluated per sub-step", flush=True)
    print(f"  run dir : {run_dir}", flush=True)
    print(f"  params  : {reference_path.name} (update_count={reference_update}) for all five "
          f"identities of the rollout in test B", flush=True)
    print(f"  tolerance on the per-sub-step health ledger: {LEDGER_TOL}", flush=True)

    results = []
    t0 = time.perf_counter()
    test_a = selfcheck_test_a(trainer)
    a_seconds = time.perf_counter() - t0
    print(f"\n=== Test A ({a_seconds:.1f}s) ===", flush=True)
    for line in test_a["lines"]:
        print("  " + line, flush=True)
    for name, value in test_a["checks"].items():
        print(f"    {name:<34} {value}", flush=True)
    results.append(test_a)

    for gating in (True, False):
        t0 = time.perf_counter()
        test_b = selfcheck_test_b(trainer, reference_params, env_steps=args.selfcheck_steps,
                                  cooldown_gate=gating)
        seconds = time.perf_counter() - t0
        title = "Test B, environment gate (cooldown_gate=True)" if gating else \
            "Test B, negative control (cooldown_gate=False)"
        print(f"\n=== {title} ({seconds:.1f}s) ===", flush=True)
        for line in test_b["lines"]:
            print("  " + line, flush=True)
        test_b["seconds"] = round(seconds, 3)
        results.append(test_b)

    ok = test_a["ok"] and results[1]["ok"] and results[2]["substep_mismatches"] > 0
    verdict = (
        "PASS: with the environment's own gate the sub-step ledger balances exactly and the "
        "replica reproduces the wrapper state; without the cooldown gate the same rollout FAILS "
        "the ledger check (" + str(results[2]["substep_mismatches"]) + " of "
        + str(results[2]["substep_checks"]) + " sub-steps reject the invented hits)"
    )
    print(f"\n  verdict: {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"  {verdict}", flush=True)

    out_path = Path(args.selfcheck_test_output) if args.selfcheck_test_output else (
        HERE / "results" / "oracle_gate" / "attribution_selfcheck.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "kind": "attribution_selfcheck",
        "rule": ("real hit = commanded attack & target in unit_type_attack_ranges[type] & attacker "
                 "alive & target alive & i != target & unit_weapon_cooldowns[i] <= 0.0, evaluated "
                 "per sub-step of SMAX.step_env_no_decode (8 sub-steps per environment step)"),
        "ledger_tolerance": LEDGER_TOL,
        "ok": ok,
        "verdict": verdict,
        "test_a": {k: v for k, v in test_a.items() if k != "lines"},
        "test_b_gated": results[1],
        "test_b_ungated": results[2],
        "seconds": round(time.perf_counter() - started, 3),
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n  wrote {out_path}", flush=True)
    print(f"  wall {time.perf_counter() - started:.1f}s", flush=True)
    return 0 if ok else 1


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
    parser.add_argument("--selfcheck", action="store_true",
                        help="run the attribution self-check (test A: constructed cooldown case, "
                             "test B: sub-step ledger on a real rollout, both with the negative "
                             "control) and exit; writes results/oracle_gate/"
                             "attribution_selfcheck.json")
    parser.add_argument("--selfcheck-steps", type=int, default=12,
                        help="environment steps of the real rollout in self-check test B "
                             "(default: 12 = 96 sub-steps)")
    parser.add_argument("--selfcheck-test-output", default=None,
                        help="output JSON path of `--selfcheck` (default: results/oracle_gate/"
                             "attribution_selfcheck.json)")
    parser.add_argument("--no-cooldown-gate", action="store_true",
                        help="NEGATIVE CONTROL ONLY: drop the weapon-cooldown gate from the hit "
                             "recomputation; the sub-step ledger check must then fail")
    parser.add_argument("--legacy", default=None,
                        help="legacy v1 profile table to compare against after the measurement "
                             "(default: the sibling profiles_v1_legacy.json if it exists)")
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
    if args.selfcheck:
        raise SystemExit(run_selfcheck(args))
    measure, eval_keys = make_measure_fn(trainer, args.episodes, args.seed_list,
                                         cooldown_gate=not args.no_cooldown_gate)

    print(f"[partner profiles] {DISCLAIMER}", flush=True)
    print(f"  run dir            : {run_dir}", flush=True)
    print(f"  reference team     : {reference_path.name} (update_count={reference_update}) for "
          f"every identity except the tested one", flush=True)
    print(f"  stages             : " + ", ".join(
        f"u{u} ({path.name})" for u, path in stages), flush=True)
    print(f"  measurement battles: seeds {args.seed_list} x {args.episodes} episodes = "
          f"{len(args.seed_list) * args.episodes} fixed battles, replayed for all "
          f"{len(PARTNER_IDENTITIES) * len(stages)} candidates "
          f"{len(PARTNER_IDENTITIES) * len(stages) * len(args.seed_list) * args.episodes} "
          f"episode EXECUTIONS, not independent samples)", flush=True)
    print("  attribution        : real hits = commanded attacks the environment applied "
          "(target in range, both alive, i != target, weapon cooldown <= 0), evaluated per "
          "sub-step; each attacker is credited with its own raw hit damage "
          "(unit_type_attacks[type]), never a share of the target's remaining health, and "
          "overkill still counts to the attacker.  The per-sub-step ledger is "
          "clamp(pre_health - sum(raw hit damage), 0) == _world_step.unit_health, compared "
          "before the wall-death handling" + (" [COOLDOWN GATE DISABLED - negative control]"
                                              if args.no_cooldown_gate else ""), flush=True)

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
            team, per_agent, diag = measure(mixed, eval_keys)
            team_np = np.asarray(jax.device_get(team), dtype=np.float64)
            per_agent_np = np.asarray(jax.device_get(per_agent), dtype=np.float64)
            diag_np = np.asarray(jax.device_get(diag), dtype=np.float64)
            elapsed = time.perf_counter() - t0

            block = candidate_metrics(team_np, per_agent_np, identity, args.seed_list)
            attribution = diag_report(diag_np)
            raw[str(identity)][str(update_count)] = block["metrics"]
            detail[str(identity)][str(update_count)] = {
                "checkpoint": path.name,
                "seconds": round(elapsed, 3),
                **block,
                "attribution_selfcheck": attribution,
            }
            per_candidate_seconds.append({"identity": identity, "update_count": update_count,
                                          "seconds": round(elapsed, 3)})
            m, totals = block["metrics"], block["pooled_totals"]
            print(f"  [identity {identity} u{update_count:>4}] damage_per_len={m[0]:.6f} "
                  f"alive_frac={m[1]:.6f} focus_share={m[2]:.6f} "
                  f"(hit_events={totals['hit_events']:.0f}, "
                  f"focused_hits={totals['focused_hits']:.0f}, "
                  f"len={totals['total_length']:.0f}, {elapsed:.1f}s)", flush=True)

            # --- the two attribution checks of the measurement (module docstring) --------------
            record = {"identity": identity, "update_count": update_count, **attribution}
            record["overkill"] = (attribution["raw_damage_total"]
                                  - attribution["applied_damage_total"])
            record["ok"] = (attribution["substep_mismatches"] == 0
                            and attribution["envstep_mismatches"] == 0)
            ledger.append(record)
            print(f"      attribution: sub-step ledger checks="
                  f"{attribution['substep_checks']:.0f} mismatches="
                  f"{attribution['substep_mismatches']:.0f} max|dev|="
                  f"{attribution['substep_max_dev']:.3e} | replica vs env: checks="
                  f"{attribution['envstep_checks']:.0f} mismatches="
                  f"{attribution['envstep_mismatches']:.0f} max|dev|="
                  f"{attribution['envstep_max_dev']:.3e} | commanded attacks="
                  f"{attribution['commanded_attacks']:.0f}, ally hits="
                  f"{attribution['ally_hit_events']:.0f}, blocked by cooldown="
                  f"{attribution['blocked_by_cooldown']:.0f}, blocked by range/alive/self="
                  f"{attribution['blocked_by_range_alive_self']:.0f}", flush=True)
            if not record["ok"]:
                raise SystemExit(
                    f"ATTRIBUTION SELF-CHECK FAILED for identity {identity} u{update_count}: the "
                    "recomputed hit events do not reproduce the environment's own health ledger "
                    f"(sub-step mismatches={attribution['substep_mismatches']:.0f}/"
                    f"{attribution['substep_checks']:.0f}, max|dev|="
                    f"{attribution['substep_max_dev']:.3e}; environment-step state mismatches="
                    f"{attribution['envstep_mismatches']:.0f}, max|dev|="
                    f"{attribution['envstep_max_dev']:.3e}). Refusing to write a profile table."
                )

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
        "damage this identity's real hits dealt / total episode length; alive_frac = total alive "
        "steps / total episode length; focus_share = real hits whose target at least one other "
        "ally also hit in the same sub-step / all real hits (0 when there is none). A real hit is "
        "a commanded attack the environment actually applied: target inside "
        "unit_type_attack_ranges[attacker type], both attacker and target alive, attacker != "
        "target, attacker weapon cooldown <= 0, evaluated per sub-step (8 sub-steps per "
        "environment step) of SMAX.step_env_no_decode. Attribution therefore follows the "
        "environment's own rule (this is the v2 fix; v1 credited every commanded attack on an "
        "enemy that lost health anywhere in the environment step, split evenly, and ignored the "
        "gates). Observations, rewards, terminal flags and episode lengths come from the real "
        "SMAXLogWrapper.step, so the battles are the ones v1 measured; each sub-step's health "
        "ledger is verified inside the measurement (profiles_selfcheck.json) and the replica is "
        "checked against the wrapper state at every non-terminal environment step. "
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
        "attribution_rule": (
            "a real hit is a commanded attack the environment actually applied: target inside "
            "unit_type_attack_ranges[attacker type], attacker and target alive, attacker != "
            "target, unit_weapon_cooldowns[attacker] <= 0.0, evaluated once per sub-step of "
            "SMAX.step_env_no_decode (8 sub-steps per environment step). v1 credited every "
            "commanded attack on an enemy that lost health anywhere in the environment step, "
            "split the loss evenly and ignored all gates."
        ),
        "overkill_convention": (
            "each attacker is credited with the raw damage its own hit carried "
            "(unit_type_attacks[attacker type]); damage is never shared out of the target's "
            "remaining health and overkill still counts to the attacker. damage_dealt = raw, "
            "damage_dealt_applied = the health actually removed (clipped at 0); v1 used the "
            "clipped quantity. damage_taken uses the same raw convention as damage_dealt."
        ),
        "ledger_check_placement": (
            "the sub-step ledger compares the pre-sub-step health against the health "
            "_world_step itself returned, i.e. before _kill_agents_touching_walls / "
            "_update_dead_agents / _push_units_away, so wall deaths can never be mistaken for an "
            "attribution error"
        ),
        "attribution_checks": [
            "per sub-step: clamp(pre_health - sum(raw hit damage), 0) == _world_step.unit_health, "
            "element-wise over all 10 units, against the output of the environment's own "
            "_world_step *before* the wall-death handling "
            "(substep_checks / substep_mismatches / substep_max_dev)",
            "per environment step: the replayed post-state equals the state the real "
            "SMAXLogWrapper.step produced - health, positions, weapon cooldowns, alive "
            "(envstep_checks / envstep_mismatches / envstep_max_dev)",
        ],
        "attribution_ledger": ledger,
        "attribution_ledger_all_ok": all(rec["ok"] for rec in ledger),
        "per_agent_channels": PER_AGENT_CHANNELS,
        "diagnostic_names": DIAG_NAMES,
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

    print("\n  attribution self-check (per candidate, over the 20 fixed battles):", flush=True)
    for rec in ledger:
        print(f"    identity {rec['identity']} u{rec['update_count']:<5} "
              f"sub-step ledger checks={rec['substep_checks']:.0f} "
              f"mismatches={rec['substep_mismatches']:.0f} "
              f"max|dev|={rec['substep_max_dev']:.3e} | replica vs real env: checks="
              f"{rec['envstep_checks']:.0f} mismatches={rec['envstep_mismatches']:.0f} "
              f"max|dev|={rec['envstep_max_dev']:.3e} ok={rec['ok']}", flush=True)
    print(f"    all {len(ledger)} attribution checks ok = "
          f"{all(rec['ok'] for rec in ledger)}", flush=True)
    print("    gate counters over the 20 battles (all five units, summed over sub-steps):",
          flush=True)
    for rec in ledger:
        print(f"      identity {rec['identity']} u{rec['update_count']:<5} commanded="
              f"{rec['commanded_attacks']:.0f} hits={rec['ally_hit_events']:.0f} "
              f"blocked_by_cooldown={rec['blocked_by_cooldown']:.0f} "
              f"blocked_by_range_alive_self={rec['blocked_by_range_alive_self']:.0f} "
              f"raw_damage={rec['raw_damage_total']:.0f} applied={rec['applied_damage_total']:.0f} "
              f"overkill={rec['overkill']:.0f} enemy_health_loss="
              f"{rec['enemy_health_loss']:.0f}", flush=True)

    legacy_path = Path(args.legacy) if args.legacy else out_path.with_name(
        "profiles_v1_legacy.json")
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        print(f"\n  old (v1 attribution, {legacy_path.name}) vs new "
              f"(sub-step hit events) profiles:", flush=True)
        print(f"    {'identity':>8} {'stage':>6} {'metric':>16} {'v1':>12} {'v2':>12} "
              f"{'delta':>12} {'rel':>9}", flush=True)
        biggest: list[tuple[float, str]] = []
        for i in PARTNER_IDENTITIES:
            for u in stage_updates:
                old_row = legacy["raw"][str(i)][str(u)]
                new_row = raw[str(i)][str(u)]
                for d, name in enumerate(METRIC_NAMES):
                    old_value, new_value = float(old_row[d]), float(new_row[d])
                    delta = new_value - old_value
                    relative = delta / old_value if old_value else float("nan")
                    print(f"    {i:>8} {'u' + str(u):>6} {name:>16} {old_value:>12.6f} "
                          f"{new_value:>12.6f} {delta:>+12.6f} {relative:>+8.1%}", flush=True)
                    biggest.append((abs(relative) if old_value else 0.0,
                                    f"identity {i} u{u} {name}"))
        print("    largest relative changes:", flush=True)
        for magnitude, label in sorted(biggest, reverse=True)[:6]:
            print(f"      {label:<32} |rel|={magnitude:.1%}", flush=True)

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
