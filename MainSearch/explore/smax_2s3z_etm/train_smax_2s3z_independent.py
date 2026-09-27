"""SMAX `2s3z` with five fixed-identity, independently optimised GRU policies.

What this is
------------
The official reference (`jaxmarl_ref/baselines/IPPO/ippo_rnn_smax.py`) trains *one* GRU policy
whose weights are shared by all five allies; each unit only gets its own hidden-state slot.  This
trainer keeps the official GRU-IPPO recipe (same network, same PPO/GAE maths, same environment and
same win criterion) but gives every identity its *own* complete parameter set - GRU, embedding,
actor head and local critic head - and its *own* optimizer state.  Nothing is shared between the
five identities.

Design invariants (each one is checked by `test_preflight.py`)
-------------------------------------------------------------
1.  Independent update.  `RunnerState.params` / `RunnerState.opt_states` are tuples of *separate*
    pytrees, one per identity.  `Trainer.apply_identity_updates` runs one PPO update per identity
    and writes back only that identity's entry, so updating identity `i` cannot touch identity
    `j != i` (not even bitwise).
2.  Optimizer application.  Inside `Trainer._update_one` the update is applied as
    ``upd, opt_state = tx.update(grads, opt_state, params)`` followed by
    ``params = optax.apply_updates(params, upd)``.  `TrainState.apply_gradients(grads=tx.update(...))`
    would run Adam a *second* time and keep two divergent optimizer states - the bug that broke the
    earlier feed-forward trainer.
3.  Learning-rate annealing uses the *official* formula (verified against
    `baselines/IPPO/ippo_rnn_smax.py:137-142`), evaluated per optimizer step::

        frac = 1 - (count // (num_minibatches * ppo_epochs)) / total_updates
        lr(count) = lr0 * frac

    `count` is optax's optimizer-step counter, so the LR is constant inside one PPO update and
    drops by one step at each update boundary; over a whole run it decays to 0 after
    `total_updates * ppo_epochs * num_minibatches` optimizer steps.  Note this is *not*
    algebraically identical to `optax.linear_schedule(transition_steps=...*)`: the linear schedule
    lowers the LR on every optimizer step while the official one lowers it once per PPO update.
    The time scale (when the LR reaches 0) is the same; the shape is not.  We follow the official
    shape because the whole point of this trainer is a minimal, comparable transformation of the
    official recipe.  `total_optimizer_steps` is written into the run metadata.
4.  GRU time axis.  Each identity's hidden state is carried through the whole rollout scan *and*
    across updates.  The minibatch re-runs the sequence over the full time axis with the
    roll-out-start hidden state of the very same environments (`hstate_init`, permuted with the
    same permutation as the transitions), so the hidden state used by the loss is exactly the one
    the roll-out started from - never reset, never taken from another identity.
5.  One identity = one PPO batch.  Identity `i` only ever sees its own observations, actions,
    values and its own per-identity GAE; its PPO batch is `num_envs * rollout_length` samples
    (envs are the minibatch axis, the time axis stays intact for the GRU).
6.  Checkpointing.  `save_checkpoint` / `load_checkpoint` (flax.serialization) persist the *full*
    training state - every identity's params + optimizer state + hidden state, the environment
    state, the observation, the carried RNG and the update counter - so `--resume` continues the
    run exactly where it stopped.  The official entry point has no saving logic at all, so this
    trainer owns it.
7.  Evaluation is isolated.  `Trainer.evaluate` only reads `runner.params`: it never receives, and
    therefore can never mutate, the optimizer states, the hidden states, the carried RNG or the
    update counter.  Evaluation episodes use a fixed evaluation seed that is disjoint from training.

Fixed evaluation
----------------
Every `--segment-updates` PPO updates the current policies are evaluated on `--eval-episodes`
episodes whose reset keys are generated once from `--eval-seed` (so the same battles are replayed
at every segment).  Evaluation is deterministic (argmax over the valid actions).  Wins are counted
with the environment's own flag `info["returned_won_episode"]` provided by `SMAXLogWrapper`, never
with a reward threshold, and both the win count and the episode count are printed.

Per-identity evaluation metrics: damage dealt, damage taken, kills, alive steps and the mean
distance to the nearest enemy.  Damage/kills are attributed to the shooters recorded in
`state.prev_attack_actions` (the decoded attack action of the last world frame of the environment
step) and split evenly when several units target the same unit; health lost to walls/self-damage
has no shooter and is therefore attributed to nobody.

Usage (absolute interpreter path required on this machine)
----------------------------------------------------------
  D:/omp/MainSearch/explore/benchmark_suitability_smax/.venv/Scripts/python.exe \
      train_smax_2s3z_independent.py --tiny
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import core as flax_core
from flax import linen as nn
from flax import serialization
from flax.linen.initializers import constant, orthogonal

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
if str(JAXMARL_REF) not in sys.path:
    sys.path.insert(0, str(JAXMARL_REF))

import distrax  # noqa: E402
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario  # noqa: E402
from jaxmarl.wrappers.baselines import SMAXLogWrapper  # noqa: E402

# Same environment configuration as the official 2s3z config (config/ippo_rnn_smax.yaml).
ENV_KWARGS = {"see_enemy_actions": True, "walls_cause_death": True, "attack_mode": "closest"}
ADAM_EPS = 1e-5


# --------------------------------------------------------------------------------------------
# Network (identical to the official ActorCriticRNN, but instantiated once per identity)
# --------------------------------------------------------------------------------------------
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False}
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis], self.initialize_carry(*rnn_state.shape), rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Dummy key: the default GRU state init is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class IdentityActorCritic(nn.Module):
    """One ally's actor-critic: shared-topology GRU policy, one instance per fixed identity."""

    action_dim: int
    fc_dim: int = 128
    gru_dim: int = 128

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        actor_mean = nn.Dense(
            self.gru_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        action_logits = actor_mean - (1.0 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(
            self.fc_dim, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return hidden, pi, jnp.squeeze(critic, axis=-1)


# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
def build_config(args) -> dict:
    base_env = HeuristicEnemySMAX(scenario=map_name_to_scenario(args.map), **ENV_KWARGS)
    num_allies = base_env.num_allies
    env = SMAXLogWrapper(base_env)
    config = {
        "map_name": args.map,
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "total_updates": args.updates,
        "segment_updates": args.segment_updates,
        "ppo_epochs": args.ppo_epochs,
        "num_minibatches": args.num_minibatches,
        "lr": args.lr,
        "anneal_lr": bool(args.anneal_lr),
        "clip_eps": args.clip_eps,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "max_grad_norm": args.max_grad_norm,
        "adam_eps": ADAM_EPS,
        "fc_dim_size": args.fc_dim_size,
        "gru_hidden_dim": args.gru_hidden_dim,
        "seed": args.seed,
        "env_kwargs": dict(ENV_KWARGS),
        # derived
        "num_agents": num_allies,
        "num_allies": num_allies,
        "num_enemies": base_env.num_enemies,
        "num_movement_actions": base_env.num_movement_actions,
        "action_dim": int(base_env.action_space(base_env.agents[0]).n),
        "obs_dim": int(base_env.observation_space(base_env.agents[0]).shape[0]),
        "horizon": int(base_env.max_steps),
        "num_actors": args.num_envs * num_allies,
        "minibatch_envs": args.num_envs // args.num_minibatches,
        # optax schedules count OPTIMIZER steps: ppo_epochs * num_minibatches per PPO update.
        "steps_per_update": args.ppo_epochs * args.num_minibatches,
        "total_optimizer_steps": args.updates * args.ppo_epochs * args.num_minibatches,
        "lr_schedule": "official_piecewise_constant" if args.anneal_lr else "constant",
        "unit_types": [int(t) for t in np.asarray(base_env.scenario)[:num_allies]],
        "unit_type_names": [base_env.unit_type_names[int(t)]
                            for t in np.asarray(base_env.scenario)[:num_allies]],
    }
    if config["action_dim"] != base_env.num_ally_actions:
        raise SystemExit("action_dim does not match env.num_ally_actions")
    if config["num_envs"] % config["num_minibatches"]:
        raise SystemExit("--num-envs must be divisible by --num-minibatches")
    return config


def make_lr_schedule(config):
    """Official IPPO-RNN-SMAX annealing, evaluated per optimizer step.

    The official trainer (`ippo_rnn_smax.py:137-142`) divides by the number of optimizer steps per
    PPO update, so the rate is constant inside one update and drops once per update.  optax calls
    this schedule once per `tx.update`, hence `count` is an optimizer-step counter.
    """
    lr0 = config["lr"]
    if not config["anneal_lr"]:
        return lr0
    steps_per_update = config["num_minibatches"] * config["ppo_epochs"]
    total_updates = config["total_updates"]

    def official_schedule(count):
        frac = 1.0 - (count // steps_per_update) / total_updates
        return lr0 * frac

    return official_schedule


def make_optimizer(config):
    lr = make_lr_schedule(config)
    return optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.adam(learning_rate=lr, eps=config["adam_eps"]),
    )


# --------------------------------------------------------------------------------------------
# Training state
# --------------------------------------------------------------------------------------------
class RunnerState(NamedTuple):
    """Everything a resumed run needs; every field is checkpointed."""

    params: tuple          # num_agents separate parameter pytrees
    opt_states: tuple      # num_agents separate optimizer states
    hidden: tuple          # num_agents separate GRU hidden states, each (num_envs, gru_dim)
    env_state: Any         # SMAXLogEnvState
    obs: Any               # {agent: (num_envs, obs_dim)}
    last_done: jnp.ndarray  # (num_agents, num_envs) bool, RNN reset flag
    rng: jnp.ndarray
    update_count: jnp.ndarray  # scalar int32, number of finished PPO updates
    episode_returns: jnp.ndarray  # (num_envs,)
    episode_lengths: jnp.ndarray  # (num_envs,) int32


class Trainer:
    """Owns the environment, the five identity networks and the jitted training/eval steps."""

    def __init__(self, config: dict):
        self.config = config
        self.base_env = HeuristicEnemySMAX(
            scenario=map_name_to_scenario(config["map_name"]), **config["env_kwargs"]
        )
        self.env = SMAXLogWrapper(self.base_env)
        self.num_agents = config["num_agents"]
        self.network = IdentityActorCritic(
            action_dim=config["action_dim"], fc_dim=config["fc_dim_size"],
            gru_dim=config["gru_hidden_dim"],
        )
        self.tx = make_optimizer(config)
        self.lr_schedule = make_lr_schedule(config)
        self.rollout = jax.jit(self._rollout)
        self.gae = jax.jit(self._gae)
        self.update_one = jax.jit(self._update_one)
        self._evaluate_impl = jax.jit(self._evaluate)  # built lazily by make_evaluator

    # -- initialisation ----------------------------------------------------------------------
    def init_runner(self, rng) -> RunnerState:
        config = self.config
        num_envs = config["num_envs"]
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        init_hstate = ScannedRNN.initialize_carry(num_envs, config["gru_hidden_dim"])
        init_x = (
            jnp.zeros((1, num_envs, config["obs_dim"])),
            jnp.zeros((1, num_envs)),
            jnp.zeros((1, num_envs, config["action_dim"])),
        )
        params = tuple(
            flax_core.unfreeze(self.network.init(k, init_hstate, init_x))
            for k in jax.random.split(init_rng, self.num_agents)
        )
        opt_states = tuple(self.tx.init(p) for p in params)
        obsv, env_state = jax.vmap(self.env.reset)(jax.random.split(reset_rng, num_envs))
        hidden = tuple(
            ScannedRNN.initialize_carry(num_envs, config["gru_hidden_dim"])
            for _ in range(self.num_agents)
        )
        return RunnerState(
            params=params,
            opt_states=opt_states,
            hidden=hidden,
            env_state=env_state,
            obs=obsv,
            last_done=jnp.zeros((self.num_agents, num_envs), dtype=bool),
            rng=rng,
            update_count=jnp.zeros((), dtype=jnp.int32),
            episode_returns=jnp.zeros((num_envs,), dtype=jnp.float32),
            episode_lengths=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    # -- roll-out ----------------------------------------------------------------------------
    def _rollout(self, runner: RunnerState):
        config, env, base_env, network = self.config, self.env, self.base_env, self.network
        num_envs, horizon = config["num_envs"], config["rollout_length"]
        num_agents = self.num_agents
        agents = env.agents

        def _env_step(carry, _):
            env_state, obs, last_done, hidden, rng, ep_return, ep_length = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
            avail = jax.vmap(base_env.get_avail_actions)(env_state.env_state)
            avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)

            actions, log_probs, values, new_hidden = [], [], [], []
            for i in range(num_agents):
                # Identity i uses its own parameters AND its own hidden slot.
                h_i, pi_i, v_i = network.apply(
                    runner.params[i], hidden[i],
                    (obs_stack[i][None], last_done[i][None], avail_stack[i]),
                )
                a_i = pi_i.sample(seed=jax.random.fold_in(act_rng, i)).squeeze(0)
                actions.append(a_i)
                log_probs.append(pi_i.log_prob(a_i).squeeze(0))
                values.append(v_i.squeeze())
                new_hidden.append(h_i)

            action_stack = jnp.stack(actions)
            env_act = {a: action_stack[i] for i, a in enumerate(agents)}
            new_obs, new_env_state, reward, done_new, info = jax.vmap(env.step)(
                jax.random.split(step_rng, num_envs), env_state, env_act
            )
            reward_stack = jnp.stack([reward[a] for a in agents]).astype(jnp.float32)
            done_stack = jnp.stack([done_new[a] for a in agents]).astype(bool)
            global_done = done_new["__all__"]
            # SMAXLogWrapper returns `returned_won_episode` with a per-agent axis (identical for
            # all allies): under vmap the batch axis comes first, so pick ally 0 of each env.
            won = jnp.asarray(info["returned_won_episode"][..., 0], dtype=jnp.float32)
            ep_return = ep_return + reward_stack[0]
            ep_length = ep_length + 1

            transition = {
                "obs": obs_stack,
                "avail": avail_stack,
                "done": last_done,
                "action": action_stack,
                "log_prob": jnp.stack(log_probs),
                "value": jnp.stack(values),
                "reward": reward_stack,
                # The episode statistics below are properties of the *battle*, identical for every
                # ally (SMAX gives all allies the same team reward); they are broadcast over the
                # identity axis so every entry of the roll-out batch has the same leading shape
                # (time, identity, env, ...).
                "global_done": jnp.broadcast_to(global_done[None], reward_stack.shape),
                "finished": jnp.broadcast_to(
                    global_done.astype(jnp.float32)[None], reward_stack.shape
                ),
                "finished_return": jnp.broadcast_to(
                    jnp.where(global_done, ep_return, 0.0)[None], reward_stack.shape
                ),
                "finished_length": jnp.broadcast_to(
                    jnp.where(global_done, ep_length, 0).astype(jnp.float32)[None],
                    reward_stack.shape,
                ),
                "finished_won": jnp.broadcast_to(
                    jnp.where(global_done, won, 0.0)[None], reward_stack.shape
                ),
            }
            new_carry = (
                new_env_state, new_obs, done_stack, tuple(new_hidden), rng,
                jnp.where(global_done, 0.0, ep_return),
                jnp.where(global_done, 0, ep_length),
            )
            return new_carry, transition

        carry = (runner.env_state, runner.obs, runner.last_done, runner.hidden, runner.rng,
                 runner.episode_returns, runner.episode_lengths)
        carry, transitions = jax.lax.scan(_env_step, carry, None, horizon)
        env_state, obs, last_done, hidden, rng, ep_return, ep_length = carry

        # Bootstrap value with the post-rollout observation; like the official trainer the
        # bootstrap pass ignores action availability (it does not touch the critic head).
        obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
        ones = jnp.ones((num_envs, config["action_dim"]), dtype=jnp.float32)
        last_values = jnp.stack([
            network.apply(runner.params[i], hidden[i],
                          (obs_stack[i][None], last_done[i][None], ones))[2].squeeze()
            for i in range(num_agents)
        ])
        new_runner = runner._replace(
            hidden=hidden, env_state=env_state, obs=obs, last_done=last_done, rng=rng,
            episode_returns=ep_return, episode_lengths=ep_length,
        )
        return new_runner, transitions, last_values

    # -- GAE (per identity) ------------------------------------------------------------------
    def _gae(self, transitions, last_values):
        config = self.config
        gamma, lam = config["gamma"], config["gae_lambda"]

        def _one(trans, last_value):
            def _scan(carry, data):
                gae, next_value = carry
                delta = (
                    data["reward"] + gamma * next_value * (1 - data["global_done"])
                    - data["value"]
                )
                gae = delta + gamma * lam * (1 - data["global_done"]) * gae
                return (gae, data["value"]), gae

            _, advantages = jax.lax.scan(
                _scan, (jnp.zeros_like(last_value), last_value), trans, reverse=True
            )
            return advantages, advantages + trans["value"]

        # transitions carry (time, identity, env, ...); last_values are (identity, env).
        advantages, returns = jax.vmap(_one, in_axes=(1, 0))(transitions, last_values)
        return advantages, returns

    # -- PPO loss for a single identity ------------------------------------------------------
    def identity_loss(self, params, hstate, trans, advantages, returns):
        """PPO loss of one identity. `hstate` is the roll-out-start hidden state of this identity."""
        config = self.config
        clip_eps = config["clip_eps"]
        _, pi, value = self.network.apply(
            params, hstate, (trans["obs"], trans["done"], trans["avail"])
        )
        log_prob = pi.log_prob(trans["action"])

        value_pred_clipped = trans["value"] + (value - trans["value"]).clip(
            -clip_eps, clip_eps
        )
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - returns), jnp.square(value_pred_clipped - returns)
        ).mean()

        log_ratio = log_prob - trans["log_prob"]
        ratio = jnp.exp(log_ratio)
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_loss = -jnp.minimum(
            ratio * adv, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        ).mean()
        entropy = pi.entropy().mean()
        total_loss = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_frac = jnp.mean(jnp.abs(ratio - 1) > clip_eps)
        return total_loss, (value_loss, actor_loss, entropy, approx_kl, clip_frac, ratio.mean())

    # -- one PPO update for a single identity -------------------------------------------------
    def _update_one(self, params, opt_state, trans, advantages, returns, hstate, rng):
        config = self.config
        tx = self.tx
        num_envs = config["num_envs"]
        num_minibatches = config["num_minibatches"]
        mb_envs = config["minibatch_envs"]

        def _epoch(carry, _):
            params, opt_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            # The minibatch axis is the environment axis; the time axis stays intact so the GRU
            # is replayed over the full sequence.  The roll-out-start hidden state is permuted
            # with exactly the same permutation, which keeps every trajectory paired with its
            # own hidden state.
            perm = jax.random.permutation(perm_rng, num_envs).reshape(num_minibatches, mb_envs)
            # (time, envs, ...) -> (time, minibatches, envs-per-minibatch, ...) -> minibatch axis
            # first, which is the axis the minibatch scan runs over.
            mb_trans = jax.tree.map(
                lambda x: jnp.swapaxes(jnp.take(x, perm, axis=1), 0, 1), trans
            )
            mb_adv = jnp.swapaxes(jnp.take(advantages, perm, axis=1), 0, 1)
            mb_ret = jnp.swapaxes(jnp.take(returns, perm, axis=1), 0, 1)
            mb_hstate = jnp.take(hstate, perm, axis=0)

            def _minibatch(carry, data):
                params, opt_state = carry
                trans_mb, adv_mb, ret_mb, hstate_mb = data
                (loss, aux), grads = jax.value_and_grad(self.identity_loss, has_aux=True)(
                    params, hstate_mb, trans_mb, adv_mb, ret_mb
                )
                # Exactly one optimizer application: tx.update already returns the parameter
                # update.  TrainState.apply_gradients(grads=tx.update(...)) would apply it a
                # second time and desynchronise the Adam state.
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), {
                    "total_loss": loss,
                    "value_loss": aux[0],
                    "actor_loss": aux[1],
                    "entropy": aux[2],
                    "approx_kl": aux[3],
                    "clip_frac": aux[4],
                    "ratio": aux[5],
                }

            (params, opt_state), losses = jax.lax.scan(
                _minibatch, (params, opt_state), (mb_trans, mb_adv, mb_ret, mb_hstate)
            )
            return (params, opt_state, rng), losses

        (params, opt_state, rng), losses = jax.lax.scan(
            _epoch, (params, opt_state, rng), None, config["ppo_epochs"]
        )
        return params, opt_state, rng, jax.tree.map(lambda x: x.mean(), losses)

    # -- orchestration -----------------------------------------------------------------------
    def rollout_batch(self, runner: RunnerState):
        """One roll-out + per-identity GAE.

        Returns the post-roll-out runner, the per-identity transitions/advantages/returns and the
        `hstate_init` tuple - the hidden states the roll-out *started* from.  The PPO loss replays
        the sequence from its first step, so it must be given exactly those hidden states (the
        official trainer does the same: it passes the pre-roll-out `init_hstate` to the update).
        """
        hstate_init = runner.hidden
        new_runner, transitions, last_values = self.rollout(runner)
        advantages, returns = self.gae(transitions, last_values)
        trans_by_identity = tuple(
            jax.tree.map(lambda x: x[:, i], transitions) for i in range(self.num_agents)
        )
        return new_runner, trans_by_identity, advantages, returns, hstate_init

    def apply_identity_updates(self, runner, trans_by_identity, advantages, returns, hstate_init,
                               active=None):
        """One PPO update for every identity in `active` (default: all).  Others are untouched.

        `hstate_init[i]` must be the hidden state identity `i` started the roll-out with; it is the
        carry the loss replays the trajectory from.
        """
        active = tuple(range(self.num_agents)) if active is None else tuple(active)
        params = list(runner.params)
        opt_states = list(runner.opt_states)
        rng = runner.rng
        losses: Dict[int, Any] = {}
        for i in active:
            rng, sub_rng = jax.random.split(rng)
            params[i], opt_states[i], _, losses[i] = self.update_one(
                params[i], opt_states[i], trans_by_identity[i], advantages[i], returns[i],
                hstate_init[i], sub_rng,
            )
        new_runner = runner._replace(params=tuple(params), opt_states=tuple(opt_states), rng=rng)
        return new_runner, losses

    def train_updates(self, runner: RunnerState, num_updates: int, active=None):
        """`num_updates` PPO updates; returns the new runner and one record per update."""
        records = []
        for _ in range(num_updates):
            runner, trans_by_identity, advantages, returns, hstate_init = self.rollout_batch(runner)
            runner, losses = self.apply_identity_updates(
                runner, trans_by_identity, advantages, returns, hstate_init, active=active
            )
            identity_order = sorted(losses)
            records.append({
                "total_loss": [losses[i]["total_loss"] for i in identity_order],
                "value_loss": [losses[i]["value_loss"] for i in identity_order],
                "entropy": [losses[i]["entropy"] for i in identity_order],
                "approx_kl": [losses[i]["approx_kl"] for i in identity_order],
                "clip_frac": [losses[i]["clip_frac"] for i in identity_order],
                "train_finished_episodes": jnp.sum(trans_by_identity[0]["finished"]),
                "train_won_sum": jnp.sum(trans_by_identity[0]["finished_won"]),
                "train_return_sum": jnp.sum(trans_by_identity[0]["finished_return"]),
            })
            runner = runner._replace(update_count=runner.update_count + 1)
        if not records:
            return runner, {}
        return runner, jax.tree.map(lambda *xs: jnp.stack(xs), *records)

    # -- evaluation --------------------------------------------------------------------------
    def make_evaluator(self, num_episodes: int, seed: int):
        """Build the jitted evaluator over fixed evaluation keys (independent of training)."""
        self._evaluate_impl = jax.jit(self._evaluate)
        self.eval_keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
        self.eval_episodes = num_episodes
        self.eval_seed = seed
        return self._evaluate_impl

    def _evaluate(self, params, keys):
        """Deterministic episodes; only `params` is an input, so `runner` cannot be mutated."""
        config, env, base_env, network = self.config, self.env, self.base_env, self.network
        num_agents, num_allies, num_enemies = (
            config["num_agents"], config["num_allies"], config["num_enemies"]
        )
        num_movement = config["num_movement_actions"]
        num_units = num_allies + num_enemies  # allies + enemies live in one SMAX state vector
        unit_index = jnp.arange(num_units)
        agents = env.agents
        horizon = config["horizon"]
        zeros_agents = jnp.zeros((num_agents,), dtype=jnp.float32)
        zero_i = jnp.zeros((), dtype=jnp.int32)

        def _episode(key):
            obs, env_state = env.reset(key)
            hidden = tuple(
                ScannedRNN.initialize_carry(1, config["gru_hidden_dim"])
                for _ in range(num_agents)
            )

            def _step(carry, t):
                (env_state, obs, hidden, live, dealt, taken, kills, alive_steps, dist_sum,
                 dist_cnt, ep_return, length, won) = carry
                smax = env_state.env_state.state
                alive_before = smax.unit_alive
                health_before = smax.unit_health
                positions = smax.unit_positions
                avail = base_env.get_avail_actions(env_state.env_state)
                obs_stack = jnp.stack([obs[a] for a in agents]).astype(jnp.float32)
                avail_stack = jnp.stack([avail[a] for a in agents]).astype(jnp.float32)

                actions, new_hidden = [], []
                for i in range(num_agents):
                    # Identity i, its own parameters and its own hidden state.
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

                # --- per-identity attribution of this environment step -------------------------
                smax_new = new_env_state.env_state.state
                health_after = smax_new.unit_health
                alive_after = smax_new.unit_alive
                prev_attack = smax_new.prev_attack_actions
                health_loss = jnp.clip(health_before - health_after, 0.0)
                fired = prev_attack >= num_movement
                # allies: decoded attack action == global index of the targeted enemy;
                # enemies: the index runs backwards over the allied team.
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

                ally_pos, enemy_pos = positions[:num_allies], positions[num_allies:]
                distances = jnp.linalg.norm(
                    ally_pos[:, None, :] - enemy_pos[None, :, :], axis=-1
                )
                visible = alive_before[:num_allies][:, None] & alive_before[num_allies:][None, :]
                nearest = jnp.where(visible, distances, jnp.inf).min(axis=1)
                has_measure = jnp.isfinite(nearest) & live

                live_f = live.astype(jnp.float32)
                dealt = dealt + dealt_step[:num_allies] * live_f
                taken = taken + taken_step[:num_allies] * live_f
                kills = kills + kills_step[:num_allies] * live_f
                alive_steps = alive_steps + alive_before[:num_allies] * live_f
                dist_sum = dist_sum + jnp.where(has_measure, nearest, 0.0)
                dist_cnt = dist_cnt + has_measure.astype(jnp.float32)

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
                    dealt, taken, kills, alive_steps, dist_sum, dist_cnt, ep_return, length, won,
                )
                return new_carry, None

            init = (
                env_state, obs, hidden, jnp.asarray(True),
                zeros_agents, zeros_agents, zeros_agents, zeros_agents, zeros_agents,
                zeros_agents, jnp.zeros((), dtype=jnp.float32), zero_i,
                jnp.zeros((), dtype=jnp.float32),
            )
            final = jax.lax.scan(_step, init, jnp.arange(horizon))[0]
            (_, _, _, _, dealt, taken, kills, alive_steps, dist_sum, dist_cnt, ep_return,
             length, won) = final
            team = jnp.stack([won, ep_return, length.astype(jnp.float32)])
            per_agent = jnp.stack(
                [dealt, taken, kills, alive_steps, dist_sum, dist_cnt], axis=1
            )
            return team, per_agent

        return jax.vmap(lambda k: _episode(k))(keys)

    def evaluate(self, runner: RunnerState) -> dict:
        """Fixed-seed evaluation of `runner.params` only; the runner itself is never touched."""
        team, per_agent = self._evaluate_impl(runner.params, self.eval_keys)
        team = np.asarray(jax.device_get(team), dtype=np.float64)
        per_agent = np.asarray(jax.device_get(per_agent), dtype=np.float64)
        num_eps = team.shape[0]
        wins = int(round(team[:, 0].sum()))
        mean_length = float(team[:, 2].mean())
        identities = []
        for i in range(self.num_agents):
            metrics = per_agent[:, i, :]
            dist_sum, dist_cnt = float(metrics[:, 4].sum()), float(metrics[:, 5].sum())
            identities.append({
                "index": i,
                "agent": self.env.agents[i],
                "unit_type": self.config["unit_type_names"][i],
                "damage_dealt": float(metrics[:, 0].mean()),
                "damage_taken": float(metrics[:, 1].mean()),
                "kills": float(metrics[:, 2].mean()),
                "alive_steps": float(metrics[:, 3].mean()),
                "alive_fraction": float(metrics[:, 3].mean() / mean_length) if mean_length else 0.0,
                "nearest_enemy_dist": (dist_sum / dist_cnt) if dist_cnt > 0 else float("nan"),
            })
        return {
            "episodes": num_eps,
            "wins": wins,
            "win_rate": wins / num_eps,
            "mean_return": float(team[:, 1].mean()),
            "mean_length": mean_length,
            # per-episode detail: makes the win count and the length spread auditable
            "episode_wons": [int(round(w)) for w in team[:, 0]],
            "episode_returns": [float(r) for r in team[:, 1]],
            "episode_lengths": [int(round(l)) for l in team[:, 2]],
            "identities": identities,
        }

    def current_lr(self, update_count: int) -> float:
        """LR that the *next* optimizer step will use, at the current update counter."""
        step = update_count * self.config["ppo_epochs"] * self.config["num_minibatches"]
        if callable(self.lr_schedule):
            return float(self.lr_schedule(min(step, self.config["total_optimizer_steps"])))
        return float(self.lr_schedule)


# --------------------------------------------------------------------------------------------
# Checkpointing (the official entry point has no saving logic; this trainer owns it)
# --------------------------------------------------------------------------------------------
def save_checkpoint(path: Path, runner: RunnerState, meta: dict) -> Path:
    payload = {name: getattr(runner, name) for name in RunnerState._fields}
    payload["_meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(payload))
    return path


def read_checkpoint_meta(path: Path) -> dict:
    return serialization.msgpack_restore(path.read_bytes())["_meta"]


def load_checkpoint(path: Path, template: RunnerState) -> RunnerState:
    data = path.read_bytes()
    payload = serialization.from_bytes(
        {name: getattr(template, name) for name in RunnerState._fields} | {"_meta": {}}, data
    )
    restored = RunnerState(**{name: payload[name] for name in RunnerState._fields})
    return jax.tree.map(jnp.asarray, restored)


def checkpoint_dir_for(save_dir: str | None, run_name: str) -> Path:
    path = Path(save_dir) if save_dir else HERE / "results" / run_name
    path = path if path.is_absolute() else (HERE / path)
    if "results" not in path.parts:
        raise SystemExit(f"--save-dir must live under a 'results/' directory (got {path})")
    return path


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="2s3z")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--segment-updates", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--num-minibatches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
                        help="official SMAX IPPO annealing: lr0*(1 - (opt_step // (minibatches*"
                             "epochs)) / updates), per optimizer step (official config sets "
                             "ANNEAL_LR: True)")
    parser.add_argument("--clip-eps", type=float, default=0.05)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.25)
    parser.add_argument("--fc-dim-size", type=int, default=128)
    parser.add_argument("--gru-hidden-dim", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--save-every-segments", type=int, default=1)
    parser.add_argument("--resume", default=None,
                        help="checkpoint file written by this trainer; training continues from it")
    parser.add_argument("--tiny", action="store_true",
                        help="small smoke-run scale (2 envs, 8 rollout steps, 4 updates)")
    args = parser.parse_args(argv)
    tiny = {
        "num_envs": 2, "rollout_length": 8, "updates": 4, "segment_updates": 2,
        "ppo_epochs": 2, "num_minibatches": 2, "eval_episodes": 4,
    }
    defaults = {
        "num_envs": 64, "rollout_length": 64, "updates": 2000, "segment_updates": 25,
        "ppo_epochs": 4, "num_minibatches": 4, "eval_episodes": 16,
    }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, tiny[name] if args.tiny else value)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    if args.save_every_segments < 1:
        parser.error("--save-every-segments must be >= 1")
    return args


def format_identity_table(identities: Sequence[dict]) -> str:
    header = (f"  {'identity':<16}{'dealt':>9}{'taken':>9}{'kills':>8}"
              f"{'alive_steps':>13}{'nearest_enemy_dist':>20}")
    rows = [header]
    for m in identities:
        rows.append(
            f"  {m['agent'] + '/' + m['unit_type']:<16}"
            f"{m['damage_dealt']:>9.3f}{m['damage_taken']:>9.3f}{m['kills']:>8.3f}"
            f"{m['alive_steps']:>13.1f}{m['nearest_enemy_dist']:>20.3f}"
        )
    return "\n".join(rows)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("smax_2s3z_indep_%Y%m%d_%H%M%S")
    out_dir = checkpoint_dir_for(args.save_dir, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    trainer = Trainer(config)
    trainer.make_evaluator(args.eval_episodes, args.eval_seed)

    runner = trainer.init_runner(jax.random.PRNGKey(args.seed))
    resumed_from = None
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_absolute():
            ckpt_path = HERE / ckpt_path
        meta = read_checkpoint_meta(ckpt_path)
        stored = meta["config"]
        for key in ("map_name", "num_envs", "rollout_length", "obs_dim", "action_dim",
                    "gru_hidden_dim", "fc_dim_size", "num_agents", "seed", "env_kwargs"):
            if stored.get(key) != config[key]:
                raise SystemExit(f"--resume config mismatch on {key}: {stored.get(key)} != "
                                 f"{config[key]}")
        runner = load_checkpoint(ckpt_path, runner)
        resumed_from = str(ckpt_path)
        print(f"[resume] loaded {ckpt_path} (update_count="
              f"{int(runner.update_count)}, segment={meta.get('segment')})", flush=True)

    metadata = {
        "kind": "smax_2s3z_independent_identities",
        "map": args.map,
        "argv": sys.argv[1:],
        "config": config,
        "lr_schedule": config["lr_schedule"],
        "eval_seed": args.eval_seed,
        "eval_episodes": args.eval_episodes,
        "params_per_identity": int(sum(x.size for x in jax.tree.leaves(runner.params[0]))),
        "resumed_from": resumed_from,
        "save_dir": str(out_dir),
        "segments": [],
        "wall_seconds_total": None,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    updates_done = int(runner.update_count)
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
        # per-identity arrays are (identity, update); episode statistics are per update.
        metrics = {k: np.asarray(jax.device_get(v)) for k, v in records.items()}

        eval_metrics = trainer.evaluate(runner)
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
            "loss_total_by_identity": [float(v) for v in metrics["total_loss"].mean(1)],
            "entropy_by_identity": [float(v) for v in metrics["entropy"].mean(1)],
            "approx_kl_by_identity": [float(v) for v in metrics["approx_kl"].mean(1)],
            "clip_frac_by_identity": [float(v) for v in metrics["clip_frac"].mean(1)],
            "eval": eval_metrics,
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        if (segment + 1) % args.save_every_segments == 0 or (segment + 1) == num_segments:
            ckpt = save_checkpoint(
                out_dir / f"checkpoint_{int(runner.update_count):08d}.bin",
                runner,
                {"update_count": int(runner.update_count), "segment": segment + 1,
                 "config": config, "argv": sys.argv[1:]},
            )
            record["checkpoint"] = ckpt.name
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        finished = record["train_finished_episodes"]
        train_win = (record["train_won_sum"] / finished) if finished else float("nan")
        train_return = (record["train_return_sum"] / finished) if finished else float("nan")
        print(
            f"[segment {segment + 1}/{num_segments}] updates={int(runner.update_count)}/"
            f"{args.updates} env_steps={cumulative} train={train_seconds:.1f}s "
            f"({record['env_steps_per_second']:.0f} steps/s) lr={lr_now:.3e}"
            f"{'  ckpt=' + record['checkpoint'] if 'checkpoint' in record else ''}",
            flush=True,
        )
        print(
            f"  train : finished_episodes={finished:.0f} won={record['train_won_sum']:.0f} "
            f"win_rate={train_win:.3f} mean_return={train_return:.2f} "
            f"loss={[round(v, 3) for v in record['loss_total_by_identity']]}",
            flush=True,
        )
        print(
            f"  eval  : {eval_metrics['episodes']} fixed episodes (seed {args.eval_seed}) -> "
            f"won {eval_metrics['wins']}/{eval_metrics['episodes']} "
            f"({eval_metrics['win_rate']:.3f}), mean_return={eval_metrics['mean_return']:.2f}, "
            f"mean_length={eval_metrics['mean_length']:.1f} "
            f"lengths={eval_metrics['episode_lengths']} wons={eval_metrics['episode_wons']}",
            flush=True,
        )
        print(format_identity_table(eval_metrics["identities"]), flush=True)

    print(
        f"[smax 2s3z independent] finished {int(runner.update_count)} updates "
        f"({int(runner.update_count) * config['num_envs'] * config['rollout_length']} env steps) "
        f"in {metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
