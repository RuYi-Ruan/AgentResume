"""ETM interface diagnostics on RWARE (follow-up to the closed-loop gate).

The closed-loop gate (see RESULTS.md) found no reproducible benefit from feeding predicted
partner actions into the policy observation: online - none was +0.53 at sensor_range=1 and
-0.84 at sensor_range=2, while the online teammate model's predictions did improve. This script
tests whether that null is an *interface* artefact, by training the same budget with three
different interfaces:

  * `online_visible` : predictions are fed only for partners that are inside the observer's view
                       at that step (others zeroed), i.e. no guesswork about unseen partners.
  * `online_onehot`  : the argmax partner action is fed as a one-hot instead of the full 5-way
                       distribution (sharper, lower-dimensional input).
  * `aux_loss`       : predictions are NOT fed to the policy; instead the policy network shares a
                       trunk with a partner-prediction head trained by an auxiliary cross-entropy
                       term (masked to visible partners).

Everything else stays identical to the closed-loop gate: four independent policies, teammate
models trained only on visible partner steps (no label leak), 8.19M env steps, 32 fixed-initial-
state frozen evaluation with greedy and sampled actions.

Usage (absolute interpreter path required on this machine):
  python train_rware_etm_interface.py --arm aux_loss --sensor-range 2 --updates 1000
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
sys.path.insert(0, str(HERE))

from train_rware_ippo import (  # noqa: E402
    TINY_4AG,
    ActorCritic,
    entropy_of,
    fingerprint_of,
    log_prob_of,
    make_env,
    masked_logits,
    preprocess,
)
from train_rware_etm_closed_loop import TeammateModel  # noqa: E402

ARMS = ["online_visible", "online_onehot", "aux_loss"]


class ActorCriticAux(nn.Module):
    """Shared trunk with actor, critic and partner-prediction heads (auxiliary-loss arm)."""

    action_dim: int
    num_partners: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        trunk = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        trunk = nn.relu(trunk)

        actor = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(trunk)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)

        critic = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(trunk)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)

        partner = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(trunk)
        partner = nn.relu(partner)
        partner_logits = nn.Dense(
            self.num_partners * self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
        )(partner)
        return logits, jnp.squeeze(value, axis=-1), partner_logits


def build_config(args: argparse.Namespace, num_features: int) -> dict:
    num_agents = TINY_4AG["num_agents"]
    use_predictions = args.arm in ("online_visible", "online_onehot")
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": num_agents,
        "num_partners": num_agents - 1,
        "action_dim": 5,
        "num_features": num_features,
        "policy_features": num_features
        + ((num_agents - 1) * 5 if use_predictions else 0),
        "lr": 2.5e-4,
        "etm_lr": 1e-3,
        "aux_coef": args.aux_coef,
        "ppo_epochs": 4,
        "num_minibatches": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "time_limit": args.time_limit,
        "sensor_range": args.sensor_range,
        "arm": args.arm,
        "seed": args.seed,
    }


def make_train(config: dict, run_updates: int, env):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_partners = config["num_partners"]
    action_dim = config["action_dim"]
    num_features = config["num_features"]
    use_predictions = config["arm"] in ("online_visible", "online_onehot")
    use_aux = config["arm"] == "aux_loss"

    policy_net = ActorCritic(action_dim=action_dim) if not use_aux else ActorCriticAux(
        action_dim=action_dim, num_partners=num_partners
    )
    etm_net = TeammateModel(action_dim=action_dim)

    def init_runner_state(rng):
        rng, policy_rng, etm_rng, reset_rng = jax.random.split(rng, 4)
        policy_params = jax.vmap(
            lambda k: policy_net.init(k, jnp.zeros((1, config["policy_features"])))
        )(jax.random.split(policy_rng, num_agents))
        tx_policy = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        )
        policy_states = jax.vmap(
            lambda p: TrainState.create(apply_fn=policy_net.apply, params=p, tx=tx_policy)
        )(policy_params)

        etm_states = []
        etm_keys = jax.random.split(etm_rng, num_agents)
        tx_etm = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config["etm_lr"]))
        for i in range(num_agents):
            partner_keys = jax.random.split(etm_keys[i], num_partners)
            partner_states = []
            for j in range(num_partners):
                params = etm_net.init(partner_keys[j], jnp.zeros((1, num_features)))
                partner_states.append(
                    TrainState.create(apply_fn=etm_net.apply, params=params, tx=tx_etm)
                )
            etm_states.append(tuple(partner_states))
        etm_states = tuple(etm_states)

        state, timestep = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        return (
            policy_states,
            etm_states,
            state,
            timestep,
            rng,
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        )

    def stacked_etm_params(etm_states):
        per_agent = []
        for agent_states in etm_states:
            per_agent.append(
                jax.tree.map(lambda *xs: jnp.stack(xs), *[st.params for st in agent_states])
            )
        return jax.tree.map(lambda *xs: jnp.stack(xs), *per_agent)

    def partner_logits_of(etm_states, features):
        params = stacked_etm_params(etm_states)
        features_ae = jnp.swapaxes(features, 0, 1)

        def per_agent(agent_params, agent_features):
            return jax.vmap(etm_net.apply, in_axes=(0, None))(agent_params, agent_features)

        preds = jax.vmap(per_agent, in_axes=(0, 0))(params, features_ae)  # (A, P, E, 5)
        return jnp.transpose(preds, (2, 0, 1, 3))  # (E, A, P, 5)

    def policy(policy_states, etm_states, features, action_mask, rng, visible_mask):
        if use_predictions:
            logits_pred = partner_logits_of(etm_states, features)
            probs = jax.nn.softmax(logits_pred, axis=-1)
            if config["arm"] == "online_onehot":
                probs = jax.nn.one_hot(jnp.argmax(probs, axis=-1), action_dim)
            if config["arm"] == "online_visible":
                probs = probs * visible_mask[:, :, :, None]
            inputs = jnp.concatenate(
                [features, jax.lax.stop_gradient(probs).reshape(features.shape[0], num_agents, -1)],
                axis=-1,
            )
        else:
            inputs = features
        inputs_ae = jnp.swapaxes(inputs, 0, 1)
        out = jax.vmap(lambda p, o: policy_net.apply(p, o), in_axes=(0, 0))(
            policy_states.params, inputs_ae
        )
        if use_aux:
            logits_ae, value_ae, partner_logits_ae = out
            partner_logits = jnp.swapaxes(partner_logits_ae, 0, 1)  # (E, A, P*5)
        else:
            logits_ae, value_ae = out
            partner_logits = None
        logits = jnp.swapaxes(logits_ae, 0, 1)
        value = jnp.swapaxes(value_ae, 0, 1)
        logits = masked_logits(logits, action_mask)
        act_keys = jax.random.split(rng, num_envs)
        action = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(act_keys, logits)
        log_prob = log_prob_of(logits, action)
        return action, log_prob, value, inputs, partner_logits

    def visibility_of(position):
        """Works for position shapes (..., num_agents, 2) -> (..., num_agents, num_agents)."""
        x = position[..., 0]
        y = position[..., 1]
        dx = jnp.abs(jnp.expand_dims(x, -1) - jnp.expand_dims(x, -2))
        dy = jnp.abs(jnp.expand_dims(y, -1) - jnp.expand_dims(y, -2))
        visible = (dx <= config["sensor_range"]) & (dy <= config["sensor_range"])
        return visible & ~jnp.eye(num_agents, dtype=bool)

    def partner_index_visibility(visible):
        """(E, A, A) -> (E, A, P): visibility of each observer's partners in ascending order."""
        indices = [p for p in range(num_agents)]
        per_agent = []
        for observer in range(num_agents):
            partners = [p for p in indices if p != observer]
            per_agent.append(jnp.stack([visible[:, observer, p] for p in partners], axis=-1))
        return jnp.stack(per_agent, axis=1)

    def rollout(runner_state):
        def _step(carry, t):
            policy_states, etm_states, state, timestep, rng, ep_return, ep_length = carry
            rng, act_rng = jax.random.split(rng)
            features, action_mask = preprocess(timestep.observation, num_agents)
            position = jnp.stack(
                [state.agents.position.x, state.agents.position.y], axis=-1
            ).astype(jnp.int16)
            visible = visibility_of(position)
            visible_partners = partner_index_visibility(visible)

            action, log_prob, value, inputs, partner_logits = policy(
                policy_states, etm_states, features, action_mask, act_rng, visible_partners
            )
            new_state, new_timestep = jax.vmap(env.step)(state, action)
            last = new_timestep.last()
            reward = new_timestep.reward
            ep_return = ep_return + reward
            ep_length = ep_length + 1
            transition = {
                "obs": inputs,
                "mask": action_mask,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": jnp.broadcast_to(reward[:, None], (num_envs, num_agents)),
                "done": jnp.broadcast_to(last[:, None], (num_envs, num_agents)),
                "features": features,
                "position": position,
                "visible": visible_partners,
                "partner_logits": partner_logits,
                "live": jnp.broadcast_to(~last[:, None], (num_envs, num_agents)),
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            new_carry = (
                policy_states,
                etm_states,
                new_state,
                new_timestep,
                rng,
                jnp.where(last, 0.0, ep_return),
                jnp.where(last, 0, ep_length),
            )
            return new_carry, (transition, finished_return, finished_length)

        carry, (transitions, finished_returns, finished_lengths) = jax.lax.scan(
            _step, runner_state, None, config["rollout_length"]
        )
        return carry, (transitions, finished_returns, finished_lengths)

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

    def per_agent(x):
        return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))).reshape(
            num_agents, -1, *x.shape[3:]
        )

    def update_policy(policy_state, batch, rng):
        obs, mask, action, log_prob, value, adv, ret = (
            batch["obs"],
            batch["mask"],
            batch["action"],
            batch["log_prob"],
            batch["value"],
            batch["adv"],
            batch["ret"],
        )
        aux_labels = batch["aux_labels"]  # (N, P) partner actions (aux arm only)
        aux_mask = batch["aux_mask"]  # (N, P) visibility of each partner
        batch_size = obs.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_mask, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value, mb_aux_labels, mb_aux_mask = mb
            out = policy_net.apply(params, mb_obs)
            if use_aux:
                logits, new_value, partner_logits = out
            else:
                logits, new_value = out
            logits = masked_logits(logits, mb_mask)
            new_log_prob = log_prob_of(logits, mb_action)
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
            entropy = entropy_of(logits).mean()
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
            aux_loss = jnp.zeros(())
            if use_aux:
                per_partner = partner_logits.reshape(
                    partner_logits.shape[0], num_partners, action_dim
                )
                ce = optax.softmax_cross_entropy_with_integer_labels(
                    per_partner, mb_aux_labels
                )
                mask_f = mb_aux_mask.astype(jnp.float32)
                aux_loss = (ce * mask_f).sum() / jnp.maximum(mask_f.sum(), 1.0)
                total = total + config["aux_coef"] * aux_loss
            return total, (actor_loss, value_loss, entropy, aux_loss)

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            idxs = jax.random.permutation(perm_rng, batch_size).reshape(
                config["num_minibatches"], minibatch
            )

            def _minibatch(carry, idx):
                train_state, _ = carry
                mb = (
                    obs[idx],
                    mask[idx],
                    action[idx],
                    log_prob[idx],
                    adv[idx],
                    ret[idx],
                    value[idx],
                    aux_labels[idx],
                    aux_mask[idx],
                )
                (loss, aux_out), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                return (train_state.apply_gradients(grads=grads), None), {
                    "total": loss,
                    "actor": aux_out[0],
                    "value": aux_out[1],
                    "entropy": aux_out[2],
                    "aux": aux_out[3],
                }

            (train_state, _), losses = jax.lax.scan(_minibatch, (train_state, None), idxs)
            return (train_state, rng), losses

        (policy_state, rng), losses = jax.lax.scan(
            _epoch, (policy_state, rng), None, config["ppo_epochs"]
        )
        return policy_state, rng, losses

    def train(rng, runner_state=None):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, update_step):
            carry, (transitions, finished_returns, finished_lengths) = rollout(runner_state)
            policy_states, etm_states, state, timestep, rng, ep_return, ep_length = carry

            features_next, mask_next = preprocess(timestep.observation, num_agents)
            position_next = jnp.stack(
                [state.agents.position.x, state.agents.position.y], axis=-1
            ).astype(jnp.int16)
            rng, boot_rng = jax.random.split(rng)
            _, _, last_value, _, _ = policy(
                policy_states,
                etm_states,
                features_next,
                mask_next,
                boot_rng,
                partner_index_visibility(visibility_of(position_next)),
            )
            advantages, returns = compute_gae(transitions, last_value)

            batches = {
                "obs": per_agent(transitions["obs"]),
                "mask": per_agent(transitions["mask"]),
                "action": per_agent(transitions["action"]),
                "log_prob": per_agent(transitions["log_prob"]),
                "value": per_agent(transitions["value"]),
                "adv": per_agent(advantages),
                "ret": per_agent(returns),
            }
            action_a = per_agent(transitions["action"])  # (A, N)
            visible_a = jnp.transpose(transitions["visible"], (2, 0, 1, 3)).reshape(
                num_agents, -1, num_partners
            )  # (A, N, P)
            aux_labels = []
            aux_masks = []
            for observer in range(num_agents):
                partners = [p for p in range(num_agents) if p != observer]
                aux_labels.append(jnp.stack([action_a[p] for p in partners], axis=-1))
                aux_masks.append(visible_a[observer])
            batches["aux_labels"] = jnp.stack(aux_labels, axis=0)  # (A, N, P)
            batches["aux_mask"] = jnp.stack(aux_masks, axis=0)
            agent_rngs = jax.random.split(rng, num_agents)
            policy_states, _, policy_losses = jax.vmap(update_policy)(
                policy_states, batches, agent_rngs
            )

            etm_loss = jnp.zeros(())
            if use_aux:
                etm_loss = policy_losses["aux"].mean()
            if use_predictions:
                features_a = per_agent(transitions["features"])
                action_a = per_agent(transitions["action"])
                visible_a = jnp.transpose(transitions["visible"], (2, 0, 1, 3)).reshape(
                    num_agents, -1, num_partners
                )
                new_etm_states = []
                losses = []
                for observer in range(num_agents):
                    partners = [p for p in range(num_agents) if p != observer]
                    agent_states = list(etm_states[observer])
                    for idx, partner in enumerate(partners):
                        labels = action_a[partner]
                        mask_p = visible_a[observer][:, idx].astype(jnp.float32)
                        state_i = agent_states[idx]

                        def loss_fn(params, labels=labels, mask_p=mask_p, state=state_i):
                            logits = etm_net.apply(params, features_a[observer])
                            ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
                            return (ce * mask_p).sum() / jnp.maximum(mask_p.sum(), 1.0)

                        loss, grads = jax.value_and_grad(loss_fn)(state_i.params)
                        agent_states[idx] = state_i.apply_gradients(grads=grads)
                        losses.append(loss)
                    new_etm_states.append(tuple(agent_states))
                etm_states = tuple(new_etm_states)
                etm_loss = jnp.mean(jnp.stack(losses))

            metric = {
                "loss_total": policy_losses["total"].mean(),
                "entropy": policy_losses["entropy"].mean(),
                "etm_loss": etm_loss,
                "num_finished_episodes": jnp.sum(finished_lengths > 0),
                "finished_return_sum": jnp.sum(finished_returns),
                "finished_length_sum": jnp.sum(finished_lengths),
                "env_step": (update_step + 1) * num_envs * config["rollout_length"],
            }
            new_runner_state = (
                policy_states,
                etm_states,
                state,
                timestep,
                rng,
                ep_return,
                ep_length,
            )
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, policy_net, etm_net


def build_eval_fn(config: dict, env, num_episodes: int, seed: int):
    num_agents = config["num_agents"]
    num_partners = config["num_partners"]
    use_predictions = config["arm"] in ("online_visible", "online_onehot")
    use_aux = config["arm"] == "aux_loss"
    policy_net = ActorCritic(action_dim=config["action_dim"]) if not use_aux else ActorCriticAux(
        action_dim=config["action_dim"], num_partners=num_partners
    )
    etm_net = TeammateModel(action_dim=config["action_dim"])
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)

    def _predict(etm_params, features):
        def per_agent(agent_params, agent_features):
            return jax.vmap(etm_net.apply, in_axes=(0, None))(agent_params, agent_features)

        return jax.vmap(per_agent, in_axes=(0, 0))(etm_params, features)

    def _make(sample: bool):
        @jax.jit
        def evaluate(policy_params, etm_params, keys):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, delivered, length = carry
                    features, action_mask = preprocess(timestep.observation, num_agents)
                    if use_predictions:
                        position = jnp.stack(
                            [state.agents.position.x, state.agents.position.y], axis=-1
                        )
                        dx = jnp.abs(position[:, 0][:, None] - position[:, 0][None, :])
                        dy = jnp.abs(position[:, 1][:, None] - position[:, 1][None, :])
                        visible = (dx <= config["sensor_range"]) & (dy <= config["sensor_range"])
                        visible = visible & (1 - jnp.eye(num_agents, dtype=bool))
                        logits_pred = _predict(etm_params, features)  # (A, P, 5)
                        probs = jax.nn.softmax(logits_pred, axis=-1)
                        if config["arm"] == "online_onehot":
                            probs = jax.nn.one_hot(jnp.argmax(probs, axis=-1), config["action_dim"])
                        if config["arm"] == "online_visible":
                            per_agent_mask = []
                            for observer in range(num_agents):
                                partners = [p for p in range(num_agents) if p != observer]
                                per_agent_mask.append(
                                    jnp.stack([visible[observer, p] for p in partners], axis=-1)
                                )
                            mask_p = jnp.stack(per_agent_mask, axis=0)
                            probs = probs * mask_p[:, :, None]
                        inputs = jnp.concatenate(
                            [features, probs.reshape(num_agents, -1)], axis=-1
                        )
                    else:
                        inputs = features
                    out = jax.vmap(lambda p, o: policy_net.apply(p, o))(policy_params, inputs)
                    logits = out[0]
                    logits = masked_logits(logits, action_mask)
                    if sample:
                        action = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
                    else:
                        action = jnp.argmax(logits, axis=-1)
                    state, timestep = env.step(state, action)
                    delivered = delivered + jnp.where(live, timestep.reward, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~timestep.last()
                    return (state, timestep, live, delivered, length), None

                init = (state, timestep, jnp.array(True), 0.0, jnp.int32(0))
                (_, _, _, delivered, length), _ = jax.lax.scan(
                    _step, init, jnp.arange(config["time_limit"])
                )
                return delivered, length

            return jax.vmap(_episode)(keys)

        return evaluate

    return {"greedy": _make(sample=False), "sampled": _make(sample=True)}, keys


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=int, default=500)
    parser.add_argument("--sensor-range", type=int, default=2)
    parser.add_argument("--aux-coef", type=float, default=0.5)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime(f"rware_iface_{args.arm}_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.time_limit, args.sensor_range)
    num_features = int(env.observation_spec.agents_view.shape[-1]) + TINY_4AG["num_agents"]
    config = build_config(args, num_features)
    train, policy_net, etm_net = make_train(config, args.segment_updates, env)
    train_jit = jax.jit(train)
    evaluators, eval_keys = build_eval_fn(config, env, args.eval_episodes, args.eval_seed)

    metadata = {
        "kind": "rware_etm_interface_diagnostic",
        "arm": args.arm,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": config,
        "eval_fingerprint": fingerprint_of(env, config["num_agents"])(eval_keys),
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
        result = train_jit(rng) if runner_state is None else train_jit(rng, runner_state)
        jax.block_until_ready(result["metrics"]["env_step"])
        train_seconds = time.perf_counter() - segment_started
        runner_state = result["runner_state"]
        metrics = {k: np.asarray(v) for k, v in jax.device_get(result["metrics"]).items()}

        policy_params = runner_state[0].params
        if config["arm"] == "aux_loss":
            etm_params = jnp.zeros((config["num_agents"],))
        else:
            per_agent_params = []
            for agent_states in runner_state[1]:
                per_agent_params.append(
                    jax.tree.map(lambda *xs: jnp.stack(xs), *[st.params for st in agent_states])
                )
            etm_params = jax.tree.map(lambda *xs: jnp.stack(xs), *per_agent_params)
        g_del, g_len = jax.device_get(evaluators["greedy"](policy_params, etm_params, eval_keys))
        s_del, s_len = jax.device_get(evaluators["sampled"](policy_params, etm_params, eval_keys))

        cumulative = (segment + 1) * args.segment_updates * args.num_envs * args.rollout_length
        finished = float(metrics["num_finished_episodes"].sum())
        record = {
            "segment": segment + 1,
            "cumulative_env_steps": cumulative,
            "train_seconds": round(train_seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * args.num_envs * args.rollout_length / train_seconds, 1
            ),
            "train_episodes_finished": finished,
            "train_delivered_per_episode": (
                float(metrics["finished_return_sum"].sum() / finished) if finished else 0.0
            ),
            "etm_loss": float(np.mean(metrics["etm_loss"])),
            "entropy": float(np.mean(metrics["entropy"])),
            "eval_greedy_delivered_mean": float(np.asarray(g_del).mean()),
            "eval_greedy_delivered_max": float(np.asarray(g_del).max()),
            "eval_sampled_delivered_mean": float(np.asarray(s_del).mean()),
            "eval_sampled_delivered_max": float(np.asarray(s_del).max()),
            "eval_sampled_episode_length_mean": float(np.asarray(s_len).mean()),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        payload = {
            "policy_params": jax.tree.map(np.asarray, policy_params),
            "etm_params": jax.tree.map(np.asarray, etm_params),
            "meta": {"cumulative_env_steps": cumulative, "arm": args.arm},
        }
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump(payload, handle)
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{args.arm} {segment + 1}/{num_segments}] steps={cumulative} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"train_deliv/ep={record['train_delivered_per_episode']:.2f} etm={record['etm_loss']:.3f} | "
            f"eval greedy {record['eval_greedy_delivered_mean']:.2f} "
            f"sampled {record['eval_sampled_delivered_mean']:.2f} "
            f"(max {record['eval_sampled_delivered_max']:.0f})",
            flush=True,
        )

    print(
        f"[rware iface] arm={args.arm} finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
