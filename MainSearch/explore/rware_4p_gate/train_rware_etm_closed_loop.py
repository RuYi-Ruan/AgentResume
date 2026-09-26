"""ETM closed-loop gate on RWARE: does feeding an observer's predictions of its partners into
the policy observation improve cooperation?

Design (three arms, identical budget, identical frozen evaluation):
  * `none`        : policy sees only its own observation. No teammate model.
  * `frozen_etm`  : each agent trains one teammate model per partner for the first
                    `--etm-warmup-updates` updates (prediction loss on steps where the partner is
                    inside the observer's field of view), then freezes it; its predicted partner
                    action distributions are concatenated to the agent's policy input.
  * `online_etm`  : same, but the teammate models keep updating for the whole run.

Key invariants kept from the earlier gates: four fixed-identity independent policies (own
parameters/optimizer/clipping/sampling key each), predictions enter the policy input under
`stop_gradient` so the prediction objective cannot be distorted by the policy, teammate models
are trained only on steps where the partner is actually visible to the observer (no label leak),
and evaluation uses 32 fixed initial states with both greedy and sampled actions.

Usage (absolute interpreter path required on this machine):
  python train_rware_etm_closed_loop.py --arm online_etm --updates 1500 --segment-updates 50
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
from train_rware_iippo import behaviour_probe  # noqa: E402



class TeammateModel(nn.Module):
    """Predicts one partner's next action distribution from the observer's own view."""

    action_dim: int = 5
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)


def build_config(args: argparse.Namespace, num_features: int) -> dict:
    num_agents = TINY_4AG["num_agents"]
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": num_agents,
        "num_partners": num_agents - 1,
        "action_dim": 5,
        "num_features": num_features,
        "lr": 2.5e-4,
        "etm_lr": 1e-3,
        "ppo_epochs": 4,
        "num_minibatches": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "time_limit": 500,
        "seed": args.seed,
        "arm": args.arm,
        "etm_warmup_updates": args.etm_warmup_updates,
        "sensor_range": args.sensor_range,
    }


def make_train(config: dict, run_updates: int, env, train_etm: bool = True):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_partners = config["num_partners"]
    action_dim = config["action_dim"]
    num_features = config["num_features"]
    use_etm = config["arm"] != "none"
    policy_features = num_features + (num_partners * action_dim if use_etm else 0)

    policy_net = ActorCritic(action_dim=action_dim)
    etm_net = TeammateModel(action_dim=action_dim)

    def init_runner_state(rng):
        rng, policy_rng, etm_rng, reset_rng = jax.random.split(rng, 4)
        policy_params = jax.vmap(
            lambda k: policy_net.init(k, jnp.zeros((1, policy_features)))
        )(jax.random.split(policy_rng, num_agents))
        tx_policy = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        )
        policy_states = jax.vmap(
            lambda p: TrainState.create(apply_fn=policy_net.apply, params=p, tx=tx_policy)
        )(policy_params)

        etm_states_list = []
        etm_keys = jax.random.split(etm_rng, num_agents)
        tx_etm = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(config["etm_lr"]))
        for i in range(num_agents):
            agent_key = etm_keys[i]
            partner_keys = jax.random.split(agent_key, num_partners)
            partner_states = []
            for j in range(num_partners):
                params = etm_net.init(partner_keys[j], jnp.zeros((1, num_features)))
                partner_states.append(
                    TrainState.create(apply_fn=etm_net.apply, params=params, tx=tx_etm)
                )
            etm_states_list.append(tuple(partner_states))
        etm_states = tuple(etm_states_list)

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
        """tuple(agent) of tuple(partner) TrainStates -> (A, P, ...) params for vmap."""
        per_agent = []
        for agent_states in etm_states:
            per_agent.append(
                jax.tree.map(lambda *xs: jnp.stack(xs), *[st.params for st in agent_states])
            )
        return jax.tree.map(lambda *xs: jnp.stack(xs), *per_agent)

    def partner_predictions(etm_states, features):
        """features (E, A, F) -> predicted action logits (E, A, P, action_dim)."""
        params = stacked_etm_params(etm_states)  # (A, P, ...)
        features_ae = jnp.swapaxes(features, 0, 1)  # (A, E, F)

        def per_agent(agent_params, agent_features):
            return jax.vmap(etm_net.apply, in_axes=(0, None))(agent_params, agent_features)

        preds = jax.vmap(per_agent, in_axes=(0, 0))(params, features_ae)  # (A, P, E, 5)
        return jnp.transpose(preds, (2, 0, 1, 3))

    def policy_input(features, probs):
        if not use_etm:
            return features
        return jnp.concatenate(
            [features, probs.reshape(features.shape[0], features.shape[1], -1)], axis=-1
        )

    def policy(policy_states, etm_states, features, action_mask, rng):
        probs = None
        if use_etm:
            logits_pred = partner_predictions(etm_states, features)
            probs = jax.lax.stop_gradient(jax.nn.softmax(logits_pred, axis=-1))
        inputs = policy_input(features, probs) if use_etm else features
        inputs_ae = jnp.swapaxes(inputs, 0, 1)  # (A, E, F')
        logits_ae, value_ae = jax.vmap(
            lambda p, o: policy_net.apply(p, o), in_axes=(0, 0)
        )(policy_states.params, inputs_ae)
        logits = jnp.swapaxes(logits_ae, 0, 1)  # (E, A, action_dim)
        value = jnp.swapaxes(value_ae, 0, 1)  # (E, A)
        logits = masked_logits(logits, action_mask)
        act_keys = jax.random.split(rng, num_envs)
        action = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(act_keys, logits)
        log_prob = log_prob_of(logits, action)
        return action, log_prob, value, inputs

    def rollout(runner_state):
        def _step(carry, t):
            policy_states, etm_states, state, timestep, rng, ep_return, ep_length = carry
            rng, act_rng = jax.random.split(rng)
            features, action_mask = preprocess(timestep.observation, num_agents)
            position = jnp.stack(
                [state.agents.position.x, state.agents.position.y], axis=-1
            ).astype(jnp.int16)
            action, log_prob, value, inputs = policy(
                policy_states, etm_states, features, action_mask, act_rng
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
                "partner_action": action,
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
        """(steps, E, A, ...) -> (A, steps*E, ...)."""
        return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))).reshape(
            num_agents, -1, *x.shape[3:]
        )

    def update_policy(policy_state, batch, rng):
        obs, mask, action, log_prob, value, adv, ret = (
            batch["obs"], batch["mask"], batch["action"], batch["log_prob"],
            batch["value"], batch["adv"], batch["ret"],
        )
        batch_size = obs.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_mask, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value = mb
            logits, new_value = policy_net.apply(params, mb_obs)
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
            return (
                actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy,
                (actor_loss, value_loss, entropy),
            )

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            idxs = jax.random.permutation(perm_rng, batch_size).reshape(
                config["num_minibatches"], minibatch
            )

            def _minibatch(carry, idx):
                train_state, _ = carry
                mb = (obs[idx], mask[idx], action[idx], log_prob[idx], adv[idx], ret[idx], value[idx])
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                return (train_state.apply_gradients(grads=grads), None), {
                    "total": loss, "entropy": aux[2]
                }

            (train_state, _), losses = jax.lax.scan(_minibatch, (train_state, None), idxs)
            return (train_state, rng), losses

        (policy_state, rng), losses = jax.lax.scan(
            _epoch, (policy_state, rng), None, config["ppo_epochs"]
        )
        return policy_state, rng, losses

    def update_etm(partner_states, features, all_actions, visible, partner_ids):
        """Cross-entropy on visible partner steps, one teammate model per partner.

        partner_states: tuple of per-partner TrainStates (scalar step each, so Adam is happy);
        features: (N, F) observer views; all_actions: (A, N); visible: (N, A).
        """
        new_states = []
        losses = []
        for idx, partner in enumerate(partner_ids):
            labels_p = all_actions[partner]
            mask_p = visible[:, partner].astype(jnp.float32)
            state = partner_states[idx]

            def loss_fn(params, labels_p=labels_p, mask_p=mask_p):
                logits = etm_net.apply(params, features)
                ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels_p)
                return (ce * mask_p).sum() / jnp.maximum(mask_p.sum(), 1.0)

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            new_states.append(state.apply_gradients(grads=grads))
            losses.append(loss)
        return tuple(new_states), jnp.mean(jnp.stack(losses))

    def train(rng, runner_state=None):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, update_step):
            carry, (transitions, finished_returns, finished_lengths) = rollout(runner_state)
            policy_states, etm_states, state, timestep, rng, ep_return, ep_length = carry

            features_next, mask_next = preprocess(timestep.observation, num_agents)
            rng, boot_rng = jax.random.split(rng)
            _, _, last_value, _ = policy(
                policy_states, etm_states, features_next, mask_next, boot_rng,
            )
            advantages, returns = compute_gae(transitions, last_value)

            obs_a = per_agent(transitions["obs"])
            mask_a = per_agent(transitions["mask"])
            action_a = per_agent(transitions["action"])
            logprob_a = per_agent(transitions["log_prob"])
            value_a = per_agent(transitions["value"])
            adv_a = per_agent(advantages)
            ret_a = per_agent(returns)
            features_a = per_agent(transitions["features"])

            # visibility of partner p from observer o at the recorded step
            position = transitions["position"]  # (steps, E, A, 2)
            dx = jnp.abs(position[:, :, :, 0][:, :, :, None] - position[:, :, None, :, 0])
            dy = jnp.abs(position[:, :, :, 1][:, :, :, None] - position[:, :, None, :, 1])
            visible = (dx <= config["sensor_range"]) & (dy <= config["sensor_range"])
            visible = visible & (1 - jnp.eye(num_agents, dtype=bool))[None, None]
            visible_a = per_agent(visible)  # (A, steps*E, A)

            agent_batches = {
                "obs": obs_a,
                "mask": mask_a,
                "action": action_a,
                "log_prob": logprob_a,
                "value": value_a,
                "adv": adv_a,
                "ret": ret_a,
            }
            agent_rngs = jax.random.split(rng, num_agents)
            policy_states, _, policy_losses = jax.vmap(update_policy)(
                policy_states, agent_batches, agent_rngs
            )

            etm_loss = jnp.zeros(())
            if use_etm and train_etm:
                new_etm_states = []
                etm_losses = []
                for observer in range(num_agents):
                    partner_ids = [p for p in range(num_agents) if p != observer]
                    updated, loss = update_etm(
                        etm_states[observer],
                        features_a[observer],
                        action_a,
                        visible_a[observer],
                        partner_ids,
                    )
                    new_etm_states.append(updated)
                    etm_losses.append(loss)
                etm_states = tuple(new_etm_states)
                etm_loss = jnp.mean(jnp.stack(etm_losses))

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
                policy_states, etm_states, state, timestep, rng, ep_return, ep_length
            )
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, policy_net, etm_net


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["none", "frozen_etm", "online_etm"], required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--etm-warmup-updates", type=int, default=300)
    parser.add_argument("--time-limit", type=int, default=500)
    parser.add_argument("--sensor-range", type=int, default=1)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")

    run_name = args.run_name or time.strftime(f"rware_etm_{args.arm}_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.time_limit, args.sensor_range)
    num_features = int(env.observation_spec.agents_view.shape[-1]) + TINY_4AG["num_agents"]
    config = build_config(args, num_features)
    train_with_etm_fn, policy_net, etm_net = make_train(
        config, args.segment_updates, env, train_etm=True
    )
    train_frozen_fn, _, _ = make_train(config, args.segment_updates, env, train_etm=False)
    train_with_etm = jax.jit(train_with_etm_fn)
    train_frozen = jax.jit(train_frozen_fn)
    warmup_segments = args.etm_warmup_updates // args.segment_updates

    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_episodes)

    def _make_eval(sample: bool):
        @jax.jit
        def evaluate(policy_params, etm_params, keys):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, delivered, length = carry
                    features, action_mask = preprocess(timestep.observation, config["num_agents"])
                    if config["arm"] == "none":
                        inputs = features
                    else:
                        pred_logits = jax.vmap(
                            lambda p, o: jax.vmap(lambda pp: etm_net.apply(pp, o))(p)
                        )(etm_params, features)
                        probs = jax.nn.softmax(pred_logits, axis=-1)
                        inputs = jnp.concatenate(
                            [features, probs.reshape(features.shape[0], -1)], axis=-1
                        )
                    logits, _ = jax.vmap(lambda p, o: policy_net.apply(p, o))(policy_params, inputs)
                    logits = masked_logits(logits, action_mask)
                    if sample:
                        action = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(
                            jax.random.split(jax.random.fold_in(key, t), config["num_agents"]),
                            logits,
                        )
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

    greedy, sampled = _make_eval(sample=False), _make_eval(sample=True)

    metadata = {
        "kind": "rware_etm_closed_loop_gate",
        "arm": args.arm,
        "config": config,
        "scenario": TINY_4AG,
        "eval_fingerprint": fingerprint_of(env, config["num_agents"])(keys),
        "segments": [],
        "wall_seconds_total": None,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rng = jax.random.PRNGKey(args.seed)
    runner_state = None
    num_segments = args.updates // args.segment_updates
    started = time.perf_counter()
    for segment in range(num_segments):
        use_etm_now = args.arm == "online_etm" or (
            args.arm == "frozen_etm" and segment < warmup_segments
        )
        train_step = train_with_etm if use_etm_now else train_frozen
        segment_started = time.perf_counter()
        result = train_step(rng) if runner_state is None else train_step(rng, runner_state)
        jax.block_until_ready(result["metrics"]["env_step"])
        train_seconds = time.perf_counter() - segment_started
        runner_state = result["runner_state"]
        metrics = {k: np.asarray(v) for k, v in jax.device_get(result["metrics"]).items()}

        policy_params = runner_state[0].params
        agent_etm_params = []
        for agent_states in runner_state[1]:
            agent_etm_params.append(
                jax.tree.map(lambda *xs: jnp.stack(xs), *[st.params for st in agent_states])
            )
        etm_params = (
            jax.tree.map(lambda *xs: jnp.stack(xs), *agent_etm_params)
            if args.arm != "none"
            else jnp.zeros((config["num_agents"],))
        )
        g_del, g_len = jax.device_get(greedy(policy_params, etm_params, keys))
        s_del, s_len = jax.device_get(sampled(policy_params, etm_params, keys))

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
            f"[{args.arm} segment {segment + 1}/{num_segments}] steps={cumulative} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"train_deliv/ep={record['train_delivered_per_episode']:.2f} "
            f"etm_loss={record['etm_loss']:.3f} | eval greedy={record['eval_greedy_delivered_mean']:.2f} "
            f"sampled={record['eval_sampled_delivered_mean']:.2f} (max {record['eval_sampled_delivered_max']:.0f})",
            flush=True,
        )

    print(
        f"[rware etm closed loop] arm={args.arm} finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
