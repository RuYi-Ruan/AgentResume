"""OvercookedV2 learnability gate with a small feed-forward policy (no CNN, no RNN).

Motivation (measured on this machine, `probe_policy_cost.py`): the official CNN+GRU pipeline
runs at ~205-1,000 env steps/s here, while the environment itself does 210,000 steps/s and a
small MLP over the flattened ego view reaches ~5,700 steps/s in a minimal PPO loop. The paper
warns that a plain feed-forward architecture learns poorly in OvercookedV2, so this gate asks the
precise question: does a cheap FF policy learn the task at all within an affordable budget?

Scope: learning gate only (shared parameters over agents, like the V2 calibration). Independent
per-agent policies, three-player maps and the ETM gates come next, and only if this passes.

Frozen evaluation: 32 fixed reset keys, deterministic (argmax) and sampled actions, 400-step
horizon, reporting correct deliveries, wrong deliveries, shaped return, episode length.

Usage (absolute interpreter path required on this machine):
  python train_overcooked_ff.py --layout grounded_coord_simple --num-envs 64 \
      --rollout-length 64 --updates 2441 --segment-updates 50 --eval-episodes 32
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))

import jaxmarl  # noqa: E402

DELIVERY_REWARD = 20.0


from jaxmarl.environments.overcooked_v2.common import (  # noqa: E402
    DynamicObject,
    StaticObject,
)


class MLPActorCritic(nn.Module):
    action_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor = nn.relu(actor)
        actor = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)

        critic = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic = nn.relu(critic)
        critic = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


class MLPActorCriticPriv(nn.Module):
    """Asymmetric actor-critic: actor sees the local view, critic sees a privileged global state.

    Reason: the 7x7x30 ego view cannot reveal pot contents, cook timers or teammates' hands, so
    V(local view) is nearly unlearnable (measured explained variance ~0) and the advantages become
    noise. The actor is unchanged, so the compared policy is unchanged; only the training-time
    critic gets information it will never act on. Critic layer names (Dense_3..5) and actor layer
    names (Dense_0..2) match MLPActorCritic, so behaviour-cloned actor weights load directly.
    """

    action_dim: int
    critic_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x, critic_x=None):
        actor = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor = nn.relu(actor)
        actor = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)

        if critic_x is None:
            # actor-only forward (evaluation, imitation regulariser): skip the critic entirely
            return logits, jnp.zeros(x.shape[:-1])
        critic_in = critic_x
        critic = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(
            critic_in)
        critic = nn.relu(critic)
        critic = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(
            critic)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


def resolve_layout(args):
    """Layout is either a registry name or ASCII text read from --layout-file."""
    if not args.layout_file:
        return args.layout
    from jaxmarl.environments.overcooked_v2.layouts import Layout

    text = Path(args.layout_file).read_text(encoding="utf-8")
    recipes = json.loads(args.recipes) if args.recipes else [[0, 0, 0], [1, 1, 1]]
    return Layout.from_string(text, possible_recipes=recipes)


def make_env(args):
    view = None if args.full_observability else args.agent_view_size
    return jaxmarl.make(
        "overcooked_v2",
        layout=resolve_layout(args),
        agent_view_size=view,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        random_agent_positions=args.random_agent_positions,
        max_steps=args.max_steps,
    )


def critic_spec(args, env):
    """Compact privileged state (CTDE critic input), far below the 7410-dim full observation.

    Contents: per-agent pose/heading/inventory flags, per-pot contents and cook state, the active
    recipe, and episode progress. Static walls and object positions carry no per-step information.
    """
    layout = env.layout
    pot_cells = [(int(x), int(y)) for y, x in np.argwhere(
        np.asarray(layout.static_objects) == int(StaticObject.POT))]
    num_ingredients = int(layout.num_ingredients)
    recipes = [int(r) for r in np.asarray(env.possible_recipes).reshape(-1)]
    grid_height, grid_width = layout.static_objects.shape
    dim = (len(env.agents) * 7 + len(pot_cells) * (4 + num_ingredients)
           + len(recipes) + 1)
    return {"dim": dim, "pot_cells": pot_cells, "num_ingredients": num_ingredients,
            "recipes": recipes, "num_agents": len(env.agents),
            "width": grid_width, "height": grid_height, "max_steps": int(env.max_steps),
            "enabled": args.privileged_critic}


def compact_state(env_state, ctx, t):
    """Batched compact state: (num_envs * num_agents, dim)."""
    batch = env_state.grid.shape[0]
    agents = env_state.agents
    inv = agents.inventory.astype(jnp.int32)
    inv_flags = jnp.stack([
        (inv == 0).astype(jnp.float32),
        (inv == int(DynamicObject.PLATE)).astype(jnp.float32),
        ((inv & int(DynamicObject.COOKED)) != 0).astype(jnp.float32),
        (((inv != 0) & (inv != int(DynamicObject.PLATE))
          & ((inv & int(DynamicObject.COOKED)) == 0)).astype(jnp.float32)),
    ], axis=-1)
    agent_feats = jnp.concatenate([
        agents.pos.x[:, :, None].astype(jnp.float32) / ctx["width"],
        agents.pos.y[:, :, None].astype(jnp.float32) / ctx["height"],
        agents.dir[:, :, None].astype(jnp.float32) / 4.0,
        inv_flags,
    ], axis=-1).reshape(batch, -1)

    pot_feats = []
    for (px, py) in ctx["pot_cells"]:
        cell = env_state.grid[:, py, px]
        items = cell[:, 1].astype(jnp.int32)
        timer = cell[:, 2].astype(jnp.float32)
        count = jax.vmap(DynamicObject.ingredient_count)(items).astype(jnp.float32)
        per_pot = [count / 3.0, timer / 20.0,
                   ((items & int(DynamicObject.COOKED)) != 0).astype(jnp.float32),
                   (items == 0).astype(jnp.float32)]
        for idx in range(ctx["num_ingredients"]):
            per_pot.append((((items >> (2 + 2 * idx)) & 0x3) > 0).astype(jnp.float32))
        pot_feats.append(jnp.stack(per_pot, axis=-1))
    pot_block = (jnp.concatenate(pot_feats, axis=-1) if pot_feats
                 else jnp.zeros((batch, 0), jnp.float32))

    recipe = env_state.recipe.astype(jnp.int32)
    recipe_feats = jnp.stack(
        [(recipe == int(r)).astype(jnp.float32) for r in ctx["recipes"]], axis=-1)
    progress = jnp.full((batch, 1), 1.0, jnp.float32) * (t / max(1, ctx["max_steps"]))
    per_env = jnp.concatenate([agent_feats, pot_block, recipe_feats, progress], axis=-1)
    return jnp.repeat(per_env, ctx["num_agents"], axis=0)


def build_config(args, env, crit) -> dict:
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": env.num_agents,
        "action_dim": env.action_space(env.agents[0]).n,
        "obs_shape": tuple(env.observation_space().shape),
        "privileged_critic": args.privileged_critic,
        "critic_dim": crit["dim"],
        "value_normalization": args.value_normalization,
        "team_reward": args.team_reward,
        "shaped_coeff": args.shaped_coeff,
        "shaped_anneal_updates": args.shaped_anneal_updates,
        "obs_flat": int(np.prod(env.observation_space().shape)),
        "lr": args.lr,
        "lr_actor": args.lr_actor if args.lr_actor is not None else args.lr,
        "lr_critic": args.lr_critic if args.lr_critic is not None else args.lr,
        "ppo_epochs": args.ppo_epochs,
        "num_minibatches": args.num_minibatches,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": args.clip_eps,
        "ent_coef": args.ent_coef if args.ent_coef is not None else 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": args.seed,
    }


def make_train(config: dict, run_updates: int, env, init_params=None, bc_pool=None,
               warmup_updates=0, ref_kl=None, crit=None):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_actors = num_envs * num_agents
    obs_flat = config["obs_flat"]
    if config["privileged_critic"]:
        network = MLPActorCriticPriv(action_dim=config["action_dim"],
                                     critic_dim=config["critic_dim"])
    else:
        network = MLPActorCritic(action_dim=config["action_dim"])

    def obs_flat_of(obs):
        return jnp.stack([obs[a] for a in env.agents]).reshape(
            num_actors, *config["obs_shape"]
        ).reshape(num_actors, -1).astype(jnp.float32)

    actor_mask = None
    if warmup_updates > 0:
        def _branch_mask(tree):
            names = list(tree.keys())
            half = len(names) // 2  # actor layers are declared first in MLPActorCritic
            actor_names = set(names[:half])
            return {name: {leaf: jnp.asarray(name in actor_names) for leaf in sub}
                    for name, sub in tree.items()}

        if config["privileged_critic"]:
            mask_params = network.init(jax.random.PRNGKey(0),
                                       jnp.zeros((1, config["obs_flat"])),
                                       jnp.zeros((1, config["critic_dim"])))["params"]
        else:
            mask_params = network.init(jax.random.PRNGKey(0),
                                       jnp.zeros((1, config["obs_flat"])))["params"]
        actor_mask = {"params": _branch_mask(mask_params)}

    def init_runner_state(rng):
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        if config["privileged_critic"]:
            params = network.init(init_rng, jnp.zeros((num_actors, obs_flat)),
                                  jnp.zeros((num_actors, config["critic_dim"])))
        else:
            params = network.init(init_rng, jnp.zeros((num_actors, obs_flat)))
        if init_params is not None:
            # demonstration warm start (BC): copy layer by layer, keeping any layer whose shape
            # differs (a privileged critic's first layer takes the global state instead)
            merged = dict(params["params"])
            for name, subtree in init_params.items():
                fresh = merged.get(name)
                if fresh is None:
                    continue
                if all(np.shape(fresh[leaf]) == np.shape(value)
                       for leaf, value in subtree.items()):
                    merged[name] = jax.tree.map(jnp.asarray, subtree)
            params = {"params": merged}
        layer_names = list(params["params"].keys())
        half = len(layer_names) // 2  # actor layers are declared first in MLPActorCritic
        actor_layers = set(layer_names[:half])
        labels = {"params": {
            name: {leaf: ("actor" if name in actor_layers else "critic") for leaf in sub}
            for name, sub in params["params"].items()}}
        tx = optax.multi_transform(
            {
                "actor": optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                                     optax.adam(config["lr_actor"], eps=1e-5)),
                "critic": optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                                      optax.adam(config["lr_critic"], eps=1e-5)),
            },
            labels,
        )
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        return (
            train_state,
            env_state,
            obsv,
            jnp.zeros((num_actors,), dtype=bool),
            rng,
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
            jnp.zeros(num_envs),
            jnp.zeros(3),  # running reward statistics: count, mean, var
        )

    def bc_lambda_value(update_step):
        if bc_pool is None:
            return jnp.zeros(())
        return bc_pool["floor"] + (bc_pool["lambda0"] - bc_pool["floor"]) * jnp.clip(
            1.0 - update_step / bc_pool["decay"], 0.0, 1.0)

    def rollout(runner_state, shaped_coeff):
        def _step(carry, t):
            (train_state, env_state, obs, done, rng, ep_return, ep_length, ep_correct,
             stats) = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            flat = obs_flat_of(obs)
            gstate = (compact_state(env_state, crit["context"], t)
                      if config["privileged_critic"] else None)
            logits, value = (network.apply(train_state.params, flat, gstate)
                             if gstate is not None
                             else network.apply(train_state.params, flat))
            action = jax.random.categorical(act_rng, logits)
            log_prob = jax.nn.log_softmax(logits, axis=-1)
            log_prob = jnp.take_along_axis(log_prob, action[:, None], axis=-1)[:, 0]

            env_act = {
                a: action.reshape(num_agents, num_envs)[i] for i, a in enumerate(env.agents)
            }
            new_obs, new_env_state, reward, done_new, info = jax.vmap(env.step)(
                jax.random.split(step_rng, num_envs), env_state, env_act
            )
            last = done_new["__all__"]
            raw = jnp.stack([reward[a] for a in env.agents]).reshape(num_agents, num_envs)
            # V2 pays the +20 delivery bonus only to the agent that delivers; with a shared policy
            # the research-relevant objective is the team's delivery count, so optionally optimise
            # the mean reward over agents (identical treatment for every experimental arm).
            raw_per_env = raw.mean(axis=0) if config["team_reward"] else raw[0]
            shaped = jnp.stack([info["shaped_reward"][a] for a in env.agents]).reshape(
                num_agents, num_envs
            )[0]
            # training reward = raw delivery signal + shaped reward (official-baseline style term;
            # the coefficient is kept fixed here instead of annealing to zero)
            raw_per_env = raw_per_env + shaped_coeff * shaped
            correct = jnp.asarray(new_env_state.new_correct_delivery, dtype=jnp.float32)

            ep_return = ep_return + raw_per_env
            ep_length = ep_length + 1
            ep_correct = ep_correct + correct

            transition = {
                "obs": flat,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": jnp.repeat(raw_per_env, num_agents),
                "done": jnp.repeat(last, num_agents),
                "critic_obs": (gstate if gstate is not None
                               else jnp.zeros((num_actors, 1))),
                "shaped": jnp.repeat(shaped, num_agents),
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            finished_correct = jnp.where(last, ep_correct, 0)
            new_carry = (
                train_state,
                new_env_state,
                new_obs,
                jnp.stack([done_new[a] for a in env.agents]).reshape(-1),
                rng,
                jnp.where(last, 0.0, ep_return),
                jnp.where(last, 0, ep_length),
                jnp.where(last, 0, ep_correct),
                stats,
            )
            return new_carry, (
                transition,
                finished_return,
                finished_length,
                finished_correct,
            )

        carry, (transitions, finished_returns, finished_lengths, finished_correct) = (
            jax.lax.scan(_step, runner_state,
                         jnp.arange(config["rollout_length"]))
        )
        (train_state, env_state, obs, done, rng, ep_return, ep_length, ep_correct,
         stats) = carry
        flat = obs_flat_of(obs)
        if config["privileged_critic"]:
            gstate = compact_state(env_state, crit["context"], config["rollout_length"])
            _, last_value = network.apply(train_state.params, flat, gstate)
        else:
            _, last_value = network.apply(train_state.params, flat)
        new_runner_state = (
            train_state, env_state, obs, done, rng, ep_return, ep_length, ep_correct, stats
        )
        return new_runner_state, transitions, last_value, (
            finished_returns,
            finished_lengths,
            finished_correct,
        )

    def compute_gae(transitions, last_value):
        def _scan(carry, data):
            gae, next_value = carry
            value = data["value"]
            delta = data["reward"] + config["gamma"] * next_value * (1 - data["done"]) - value
            gae = delta + config["gamma"] * config["gae_lambda"] * (1 - data["done"]) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _scan, (jnp.zeros_like(last_value), last_value), transitions, reverse=True
        )
        return advantages, advantages + transitions["value"]

    def update(train_state, transitions, advantages, returns, rng, update_index):
        batch_size = num_actors * config["rollout_length"]
        flat = lambda x: x.reshape(batch_size, *x.shape[2:])
        obs_f = flat(transitions["obs"])
        critic_f = flat(transitions["critic_obs"]) if config["privileged_critic"] else None
        action_f = flat(transitions["action"])
        logprob_f = flat(transitions["log_prob"])
        value_f = flat(transitions["value"])
        adv_f = advantages.reshape(batch_size)
        ret_f = returns.reshape(batch_size)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        ref_beta = 0.0
        if ref_kl is not None:
            ref_beta = ref_kl["floor"] + (ref_kl["beta0"] - ref_kl["floor"]) * jnp.clip(
                1.0 - update_index / ref_kl["decay"], 0.0, 1.0)
        bc_lambda = 0.0
        demo_obs = demo_act = None
        if bc_pool is not None:
            frac = jnp.clip(1.0 - update_index / bc_pool["decay"], 0.0, 1.0)
            bc_lambda = bc_pool["floor"] + (bc_pool["lambda0"] - bc_pool["floor"]) * frac
            rng, demo_rng = jax.random.split(rng)
            n_side = bc_pool["batch"] // 2
            idx_u = jax.random.randint(demo_rng, (n_side,), 0, bc_pool["obs_uniform"].shape[0])
            idx_c = jax.random.randint(demo_rng, (n_side,), 0, bc_pool["obs_critical"].shape[0])
            demo_obs = jnp.concatenate([bc_pool["obs_uniform"][idx_u],
                                        bc_pool["obs_critical"][idx_c]])
            demo_act = jnp.concatenate([bc_pool["act_uniform"][idx_u],
                                        bc_pool["act_critical"][idx_c]])

        def _loss(params, mb):
            mb_obs, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value, mb_critic = mb
            logits, new_value = (network.apply(params, mb_obs, mb_critic)
                                 if config["privileged_critic"]
                                 else network.apply(params, mb_obs))
            new_log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1), mb_action[:, None], axis=-1
            )[:, 0]
            ratio = jnp.exp(new_log_prob - mb_log_prob)
            actor_loss = -jnp.minimum(
                ratio * mb_adv,
                jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * mb_adv,
            ).mean()
            value_clipped = mb_value + (new_value - mb_value).clip(
                -config["clip_eps"], config["clip_eps"]
            )
            value_loss = 0.5 * jnp.maximum(
                jnp.square(new_value - mb_ret), jnp.square(value_clipped - mb_ret)
            ).mean()
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(logits, axis=-1), axis=-1))
            approx_kl = jnp.mean(mb_log_prob - new_log_prob)
            clip_fraction = jnp.mean(
                (jnp.abs(ratio - 1.0) > config["clip_eps"]).astype(jnp.float32))
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
            ref_kl_value = jnp.zeros(())
            if ref_kl is not None and demo_obs is not None:
                # KL(pi_BC || pi_theta) on the *states PPO actually visits*, not demo states
                logits_ref, _ = ref_kl["module"].apply({"params": ref_kl["params"]}, mb_obs)
                p_ref = jax.nn.softmax(logits_ref, axis=-1)
                log_p_new = jax.nn.log_softmax(logits, axis=-1)
                ref_kl_value = jnp.mean(jnp.sum(
                    p_ref * (jnp.log(p_ref + 1e-9) - log_p_new), axis=-1))
                total = total + ref_beta * ref_kl_value
            bc_loss = jnp.zeros(())
            if demo_obs is not None:
                logits_demo, _ = network.apply(params, demo_obs)
                logp = jax.nn.log_softmax(logits_demo, axis=-1)
                bc_loss = -jnp.mean(jnp.take_along_axis(logp, demo_act[:, None], axis=-1)[:, 0])
                total = total + bc_lambda * bc_loss
            return total, (actor_loss, value_loss, entropy, bc_loss, ref_kl_value,
                           approx_kl, clip_fraction)

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            idxs = jax.random.permutation(perm_rng, batch_size).reshape(
                config["num_minibatches"], minibatch
            )

            def _minibatch(carry, idx):
                train_state, _ = carry
                mb = (
                    obs_f[idx],
                    action_f[idx],
                    logprob_f[idx],
                    adv_f[idx],
                    ret_f[idx],
                    value_f[idx],
                    critic_f[idx] if critic_f is not None else obs_f[idx],
                )
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                if actor_mask is not None:
                    grads = jax.tree.map(
                        lambda g, m: jnp.where(
                            (update_index < warmup_updates) & m, jnp.zeros_like(g), g),
                        grads, actor_mask,
                    )
                return (train_state.apply_gradients(grads=grads), None), {
                    "total": loss,
                    "actor": aux[0],
                    "value": aux[1],
                    "entropy": aux[2],
                    "bc_loss": aux[3],
                    "ref_kl": aux[4],
                    "approx_kl": aux[5],
                    "clip_fraction": aux[6],
                }

            (train_state, _), losses = jax.lax.scan(_minibatch, (train_state, None), idxs)
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            _epoch, (train_state, rng), None, config["ppo_epochs"]
        )
        residual = ret_f - value_f
        explained_var = 1.0 - residual.var() / (ret_f.var() + 1e-8)
        losses["explained_var"] = jnp.broadcast_to(explained_var, losses["total"].shape)
        losses["return_std"] = jnp.broadcast_to(ret_f.std(), losses["total"].shape)
        return train_state, rng, losses

    def train(rng, runner_state=None, start_step=0):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, step):
            update_step = start_step + step
            if config["shaped_anneal_updates"] > 0:
                coeff = config["shaped_coeff"] * jnp.clip(
                    1.0 - update_step / config["shaped_anneal_updates"], 0.0, 1.0)
            else:
                coeff = jnp.asarray(config["shaped_coeff"], jnp.float32)
            runner_state, transitions, last_value, (f_ret, f_len, f_correct) = rollout(
                runner_state, coeff
            )
            if config["value_normalization"]:
                count, mean, var = runner_state[8]
                rew = transitions["reward"]
                n = rew.size
                batch_mean = rew.mean()
                batch_var = jnp.maximum(rew.var(), 0.0)
                delta = batch_mean - mean
                total = count + n
                new_mean = mean + delta * n / jnp.maximum(total, 1.0)
                new_var = jnp.maximum(
                    (var * count + batch_var * n + delta ** 2 * count * n
                     / jnp.maximum(total, 1.0)) / jnp.maximum(total, 1.0), 1e-6)
                rew_norm = (rew - new_mean) / jnp.sqrt(new_var)
                transitions = {**transitions, "reward": rew_norm}
                runner_state = (*runner_state[:8],
                                jnp.stack([total, new_mean, new_var]))
            advantages, returns = compute_gae(transitions, last_value)
            train_state = runner_state[0]
            train_state, rng, losses = update(
                train_state, transitions, advantages, returns, runner_state[4], update_step
            )
            metric = {
                "loss_total": losses["total"].mean(),
                "entropy": losses["entropy"].mean(),
                "bc_loss": losses["bc_loss"].mean(),
                "bc_lambda": bc_lambda_value(update_step),
                "ref_kl": losses["ref_kl"].mean(),
                "explained_var": losses["explained_var"].mean(),
                "return_std": losses["return_std"].mean(),
                "approx_kl": losses["approx_kl"].mean(),
                "clip_fraction": losses["clip_fraction"].mean(),
                "value_loss": losses["value"].mean(),
                "shaped_reward": transitions["shaped"].mean(),
                "raw_reward": transitions["reward"].mean(),
                "num_finished_episodes": jnp.sum(f_len > 0),
                "finished_return_sum": jnp.sum(f_ret),
                "finished_correct_sum": jnp.sum(f_correct),
                "adv_std": transitions["reward"].std() * 0 + advantages.std(),
                "adv_return_corr": jnp.corrcoef(advantages.reshape(-1),
                                                (advantages + transitions["value"]).reshape(-1))[0, 1],
                "reward_mean": (runner_state[8][1] if config["value_normalization"]
                                else transitions["reward"].mean()),
                "reward_std": (jnp.sqrt(runner_state[8][2]) if config["value_normalization"]
                               else transitions["reward"].std()),
                "env_step": (update_step + 1) * num_envs * config["rollout_length"],
            }
            new_runner_state = (
                train_state,
                runner_state[1],
                runner_state[2],
                runner_state[3],
                rng,
                runner_state[5],
                runner_state[6],
                runner_state[7],
                runner_state[8],
            )
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, network


def build_eval_fn(config: dict, env, num_episodes: int, seed: int, horizon: int):
    if config["privileged_critic"]:
        network = MLPActorCriticPriv(action_dim=config["action_dim"],
                                     critic_dim=config["critic_dim"])
    else:
        network = MLPActorCritic(action_dim=config["action_dim"])
    obs_shape = config["obs_shape"]
    num_agents = config["num_agents"]
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)

    def _make(sample: bool):
        @jax.jit
        def evaluate(params, keys):
            def _episode(key):
                obs, env_state = env.reset(key)

                def _step(carry, t):
                    obs, env_state, live, delivered, wrong, shaped, length = carry
                    flat = jnp.stack([obs[a] for a in env.agents]).reshape(
                        num_agents, *obs_shape
                    ).reshape(num_agents, -1).astype(jnp.float32)
                    logits, _ = network.apply(params, flat)
                    if sample:
                        action = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
                    else:
                        action = jnp.argmax(logits, axis=-1)
                    env_act = {a: action[i] for i, a in enumerate(env.agents)}
                    obs, env_state, reward, done, info = env.step(key, env_state, env_act)
                    step_raw = reward[env.agents[0]]
                    step_shaped = info["shaped_reward"][env.agents[0]]
                    delivered = delivered + jnp.where(
                        live & env_state.new_correct_delivery, 1.0, 0.0
                    )
                    wrong = wrong + jnp.where(
                        live & (step_raw <= -DELIVERY_REWARD), 1.0, 0.0
                    )
                    shaped = shaped + jnp.where(live, step_shaped, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~done["__all__"]
                    return (obs, env_state, live, delivered, wrong, shaped, length), None

                init = (obs, env_state, jnp.array(True), 0.0, 0.0, 0.0, jnp.int32(0))
                (_, _, _, delivered, wrong, shaped, length), _ = jax.lax.scan(
                    _step, init, jnp.arange(horizon)
                )
                return jnp.stack([delivered, wrong, shaped, length.astype(jnp.float32)])

            return jax.vmap(_episode)(keys)

        return evaluate

    def fingerprint(keys):
        def _first(key):
            obs, _ = env.reset(key)
            return jnp.stack([obs[a] for a in env.agents]).reshape(
                num_agents, *obs_shape
            ).astype(jnp.float32)

        views = jax.jit(jax.vmap(_first))(keys)
        return {"num_episodes": int(views.shape[0]), "obs_sum": float(np.asarray(views).sum())}

    return {"greedy": _make(sample=False), "sampled": _make(sample=True)}, fingerprint, keys


def summarize(raw: np.ndarray) -> dict:
    return {
        "correct_deliveries_mean": float(raw[:, 0].mean()),
        "correct_deliveries_max": float(raw[:, 0].max()),
        "wrong_deliveries_mean": float(raw[:, 1].mean()),
        "shaped_return_mean": float(raw[:, 2].mean()),
        "episode_length_mean": float(raw[:, 3].mean()),
        "episodes_with_delivery": int((raw[:, 0] > 0).sum()),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="grounded_coord_simple")
    parser.add_argument("--agent-view-size", type=int, default=2)
    parser.add_argument("--layout-file", default=None,
                        help="ASCII layout file; overrides --layout")
    parser.add_argument("--recipes", default=None,
                        help="JSON list of recipes, e.g. '[[0,0,0],[1,1,1]]'")
    parser.add_argument("--full-observability", action="store_true",
                        help="pass agent_view_size=None (whole grid visible)")
    parser.add_argument("--random-agent-positions", action=argparse.BooleanOptionalAction, default=True,
                        help="use --no-random-agent-positions for fixed starts")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2441)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--lr-actor", type=float, default=None,
                        help="separate actor learning rate (default: --lr)")
    parser.add_argument("--lr-critic", type=float, default=None,
                        help="separate critic learning rate (default: --lr)")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-normalization", action=argparse.BooleanOptionalAction,
                        default=True, help="normalise rewards by running statistics")
    parser.add_argument("--shaped-coeff", type=float, default=1.0,
                        help="training reward = raw + coeff * shaped (official baseline uses 1)")
    parser.add_argument("--shaped-anneal-updates", type=int, default=0,
                        help="0 keeps the shaping coefficient fixed (recommended here)")
    parser.add_argument("--team-reward", action=argparse.BooleanOptionalAction, default=False,
                        help="optimise the mean over agents' rewards (team objective)")
    parser.add_argument("--privileged-critic", action="store_true",
                        help="critic sees a compact global state (asymmetric actor-critic)")
    parser.add_argument("--ref-kl-beta", type=float, default=0.0,
                        help="weight of KL(pi_BC || pi_theta) on rollout states")
    parser.add_argument("--ref-kl-decay-updates", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bc-regularizer", default=None,
                        help="demonstration npz used for the decaying imitation regularizer")
    parser.add_argument("--bc-lambda0", type=float, default=0.5)
    parser.add_argument("--bc-lambda-floor", type=float, default=0.0,
                        help="permanent demonstration-regulariser floor")
    parser.add_argument("--ref-kl-floor", type=float, default=0.0,
                        help="permanent reference-KL floor")
    parser.add_argument("--critic-warmup-updates", type=int, default=0,
                        help="freeze the actor branch for the first N updates (critic-only)")
    parser.add_argument("--ent-coef", type=float, default=None)
    parser.add_argument("--bc-decay-updates", type=int, default=300)
    parser.add_argument("--bc-pool", type=int, default=30000,
                        help="max demo transitions per pool (uniform / critical)")
    parser.add_argument("--bc-batch", type=int, default=512)
    parser.add_argument("--bc-init", default=None,
                        help="pickle with BC actor weights ({'params': ...}) used to warm start")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("ocv2_ff_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args)
    crit = critic_spec(args, env)
    crit["context"] = crit
    config = build_config(args, env, crit)
    print(f"[ocv2 ff gate] privileged critic: {config['privileged_critic']} "
          f"(critic_dim={config['critic_dim']})")
    if config["num_agents"] * config["num_envs"] % config["num_minibatches"]:
        raise SystemExit("num_actors must be divisible by num_minibatches")

    init_params = None
    bc_full_params = None
    if args.bc_init:
        bc_path = Path(args.bc_init)
        if not bc_path.is_absolute():
            bc_path = HERE / bc_path
        with bc_path.open("rb") as handle:
            bc_payload = pickle.load(handle)
        bc_full_params = jax.tree.map(np.asarray, bc_payload["params"])
        init_params = jax.tree.map(np.asarray, bc_payload["params"])
        if args.privileged_critic:
            # keep the cloned actor layers only; the privileged critic is freshly initialised
            init_params = {k: v for k, v in init_params.items()
                           if k in ("Dense_0", "Dense_1", "Dense_2")}
        print(f"[ocv2 ff gate] BC warm start from {bc_path} "
              f"(meta={bc_payload.get('meta')})")
    bc_pool = None
    if args.bc_regularizer:
        demo_path = Path(args.bc_regularizer)
        if not demo_path.is_absolute():
            demo_path = HERE / demo_path
        data = np.load(demo_path)
        obs_raw = data["obs"].reshape(-1, int(np.prod(data["obs"].shape[2:]))).astype(np.float32)
        act_raw = data["action"].reshape(-1).astype(np.int32)
        flags = data["flags"]
        n_agents = data["obs"].shape[1]
        interact = flags[:, :n_agents].astype(bool)
        critical = np.zeros_like(interact)
        for offset in range(-3, 4):
            shifted = np.roll(interact, offset, axis=0)
            if offset > 0:
                shifted[:offset] = False
            elif offset < 0:
                shifted[offset:] = False
            critical |= shifted
        event = ((flags[:, n_agents] | flags[:, n_agents + 1]) != 0)
        critical |= np.repeat(event[:, None], n_agents, axis=1)
        critical = critical.reshape(-1)
        rng_pool = np.random.default_rng(args.seed)
        uniform_idx = rng_pool.permutation(len(obs_raw))[: args.bc_pool]
        critical_idx = rng_pool.permutation(np.flatnonzero(critical))[: args.bc_pool]
        bc_pool = {
            "obs_uniform": jnp.asarray(obs_raw[uniform_idx]),
            "act_uniform": jnp.asarray(act_raw[uniform_idx]),
            "obs_critical": jnp.asarray(obs_raw[critical_idx]),
            "act_critical": jnp.asarray(act_raw[critical_idx]),
            "lambda0": args.bc_lambda0,
            "floor": args.bc_lambda_floor,
            "decay": float(max(1, args.bc_decay_updates)),
            "batch": args.bc_batch,
        }
        print(f"[ocv2 ff gate] imitation regularizer: lambda0={args.bc_lambda0} "
              f"decay={args.bc_decay_updates} updates, pools "
              f"{len(uniform_idx)}/{len(critical_idx)} from {demo_path}")
    ref_kl = None
    if args.ref_kl_beta > 0 and bc_full_params is not None:
        ref_kl = {"params": bc_full_params,
                  "module": MLPActorCritic(action_dim=config["action_dim"]),
                  "floor": args.ref_kl_floor,
                  "beta0": args.ref_kl_beta,
                  "decay": float(max(1, args.ref_kl_decay_updates))}
        print(f"[ocv2 ff gate] reference-policy KL: beta0={args.ref_kl_beta} "
              f"decay={args.ref_kl_decay_updates} updates")
    train, _ = make_train(config, args.segment_updates, env, init_params=init_params,
                          bc_pool=bc_pool, warmup_updates=args.critic_warmup_updates,
                          ref_kl=ref_kl, crit=crit)
    train_jit = jax.jit(train)
    evaluators, fingerprint, eval_keys = build_eval_fn(
        config, env, args.eval_episodes, args.eval_seed, horizon=env.max_steps
    )

    metadata = {
        "kind": "overcooked_v2_ff_learnability_gate",
        "layout_file": args.layout_file,
        "recipes": args.recipes,
        "shared_parameters": True,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": {k: v for k, v in config.items() if k != "obs_shape"},
        "obs_shape": list(config["obs_shape"]),
        "bc_init": args.bc_init,
        "bc_regularizer": {"path": args.bc_regularizer, "lambda0": args.bc_lambda0,
                           "decay_updates": args.bc_decay_updates},
        "bc_pool": args.bc_pool,
        "critic_warmup_updates": args.critic_warmup_updates,
        "ref_kl": {"beta0": args.ref_kl_beta, "decay_updates": args.ref_kl_decay_updates},
        "clip_eps_used": config["clip_eps"],
        "ent_coef_used": config["ent_coef"],
        "privileged_critic": config["privileged_critic"],
        "critic_dim": config["critic_dim"],
        "env_max_steps": env.max_steps,
        "layout": args.layout,
        "eval_fingerprint": fingerprint(eval_keys),
        "segments": [],
        "wall_seconds_total": None,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rng = jax.random.PRNGKey(args.seed)
    runner_state = None
    num_segments = args.updates // args.segment_updates
    started = time.perf_counter()
    for segment in range(num_segments):
        segment_started = time.perf_counter()
        result = (train_jit(rng, None, 0) if runner_state is None
                  else train_jit(rng, runner_state, segment * args.segment_updates))
        jax.block_until_ready(result["metrics"]["env_step"])
        train_seconds = time.perf_counter() - segment_started
        runner_state = result["runner_state"]
        metrics = {k: np.asarray(v) for k, v in jax.device_get(result["metrics"]).items()}

        params = runner_state[0].params
        greedy_raw = np.asarray(jax.device_get(evaluators["greedy"](params, eval_keys)))
        sampled_raw = np.asarray(jax.device_get(evaluators["sampled"](params, eval_keys)))

        cumulative = (segment + 1) * args.segment_updates * args.num_envs * args.rollout_length
        finished = float(metrics["num_finished_episodes"].sum())
        record = {
            "segment": segment + 1,
            "cumulative_env_steps": cumulative,
            "train_seconds": round(train_seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * args.num_envs * args.rollout_length / train_seconds, 1
            ),
            "train_finished_episodes": finished,
            "train_correct_per_episode": (
                float(metrics["finished_correct_sum"].sum() / finished) if finished else 0.0
            ),
            "train_shaped_reward": float(metrics["shaped_reward"].mean()),
            "train_raw_reward": float(metrics["raw_reward"].mean()),
            "entropy": float(metrics["entropy"].mean()),
            "explained_var": float(metrics["explained_var"].mean()),
            "adv_std": float(metrics["adv_std"].mean()),
            "adv_return_corr": float(metrics["adv_return_corr"].mean()),
            "reward_mean": float(metrics["reward_mean"].mean()),
            "reward_std": float(metrics["reward_std"].mean()),
            "return_std": float(metrics["return_std"].mean()),
            "value_loss": float(metrics["value_loss"].mean()),
            "approx_kl": float(metrics["approx_kl"].mean()),
            "clip_fraction": float(metrics["clip_fraction"].mean()),
            "eval_greedy": summarize(greedy_raw),
            "eval_sampled": summarize(sampled_raw),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        payload = {
            "params": jax.tree.map(np.asarray, params),
            "opt_state": jax.tree.map(np.asarray, runner_state[0].opt_state),
            "meta": {"cumulative_env_steps": cumulative, "segment": segment + 1},
        }
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[segment {segment + 1}/{num_segments}] steps={cumulative} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"train_correct/ep={record['train_correct_per_episode']:.2f} "
            f"shaped={record['train_shaped_reward']:.4f} | "
            f"eval greedy {record['eval_greedy']['correct_deliveries_mean']:.2f} "
            f"sampled {record['eval_sampled']['correct_deliveries_mean']:.2f} "
            f"(max {record['eval_sampled']['correct_deliveries_max']:.0f})",
            flush=True,
        )

    print(
        f"[ocv2 ff gate] finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
