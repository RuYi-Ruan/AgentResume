"""RWARE 4-agent gate with four INDEPENDENT policies (one parameter set per robot).

Why: the ETM main experiment requires at least three fixed-identity agents that each learn
on their own and each hold models of the other partners. Shared parameters (as in
`train_rware_ippo.py`) do not provide that. This variant keeps the scenario, observation
encoding, hyperparameters and frozen evaluation identical to the shared-parameter gate, so the
two are directly comparable; the only change is that every agent owns its actor/critic
parameters, its optimizer state, its gradient clipping and its sampling key.

Behaviour-change evidence: `--dump-behaviour <dir>` writes, per segment, the per-agent action
distribution observed on a fixed evaluation set, so policy change over training (and the
difference between agents) can be quantified afterwards.

Usage (absolute interpreter path required on this machine):
  python train_rware_iippo.py --updates 1000 --segment-updates 50 --num-envs 64 \
      --dump-behaviour results/<run>/behaviour
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


def build_config(args: argparse.Namespace) -> dict:
    num_agents = TINY_4AG["num_agents"]
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": num_agents,
        "action_dim": 5,
        "num_features": 66 + num_agents,
        "lr": 2.5e-4,
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
    }


def make_train(config: dict, run_updates: int, env):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_actors = num_envs * num_agents
    network = ActorCritic(action_dim=config["action_dim"])

    def init_runner_state(rng):
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        keys = jax.random.split(init_rng, num_agents)
        agent_params = jax.vmap(
            lambda k: network.init(k, jnp.zeros((1, config["num_features"])))
        )(keys)
        tx = optax.chain(
            optax.clip_by_global_norm(config["max_grad_norm"]),
            optax.adam(config["lr"], eps=1e-5),
        )

        def make_state(params):
            return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        train_states = jax.vmap(make_state)(agent_params)
        state, timestep = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        return (
            train_states,
            state,
            timestep,
            rng,
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        )

    def apply_all(params, features):
        """features: (num_agents, num_features) -> logits (A, action_dim), value (A,)."""
        return jax.vmap(lambda p, o: network.apply(p, o))(params, features)

    def policy(train_states, timestep, rng):
        features, action_mask = preprocess(timestep.observation, num_agents)
        logits, value = jax.vmap(apply_all, in_axes=(None, 0))(
            train_states.params, features
        )
        logits = masked_logits(logits, action_mask)
        act_rngs = jax.random.split(rng, num_envs)
        action = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(act_rngs, logits)
        log_prob = log_prob_of(logits, action)
        return action, log_prob, value, features, action_mask

    def rollout(runner_state):
        def _step(carry, _):
            train_states, state, timestep, rng, ep_return, ep_length = carry
            rng, act_rng = jax.random.split(rng)
            action, log_prob, value, features, action_mask = policy(
                train_states, timestep, act_rng
            )
            new_state, new_timestep = jax.vmap(env.step)(state, action)
            last = new_timestep.last()
            reward = new_timestep.reward
            ep_return = ep_return + reward
            ep_length = ep_length + 1
            transition = {
                "obs": features,
                "mask": action_mask,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": jnp.broadcast_to(reward[:, None], (num_envs, num_agents)),
                "done": jnp.broadcast_to(last[:, None], (num_envs, num_agents)),
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            new_carry = (
                train_states,
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
        train_states, state, timestep, rng, ep_return, ep_length = carry
        _, _, last_value, _, _ = policy(train_states, timestep, jax.random.PRNGKey(0))
        return (train_states, state, timestep, rng, ep_return, ep_length), (
            transitions,
            last_value,
            finished_returns,
            finished_lengths,
        )

    def compute_gae(transitions, last_value):
        def _scan(carry, data):
            gae, next_value = carry
            value = data["value"]
            delta = (
                data["reward"] + config["gamma"] * next_value * (1 - data["done"]) - value
            )
            gae = delta + config["gamma"] * config["gae_lambda"] * (1 - data["done"]) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _scan,
            (jnp.zeros_like(last_value), last_value),
            transitions,
            reverse=True,
        )
        return advantages, advantages + transitions["value"]

    def per_agent_batches(transitions, advantages, returns):
        """(steps, E, A, ...) -> dict of (A, steps*E, ...) plus (A,) rng seeds."""
        def to_agent(x):
            return jnp.transpose(x, (2, 0, 1) + tuple(range(3, x.ndim))).reshape(
                num_agents, -1, *x.shape[3:]
            )

        return {
            "obs": to_agent(transitions["obs"]),
            "mask": to_agent(transitions["mask"]),
            "action": to_agent(transitions["action"]),
            "log_prob": to_agent(transitions["log_prob"]),
            "value": to_agent(transitions["value"]),
            "adv": to_agent(advantages),
            "ret": to_agent(returns),
        }

    def update_agent(train_state, batch, rng):
        obs = batch["obs"]
        mask = batch["mask"]
        action = batch["action"]
        log_prob = batch["log_prob"]
        value = batch["value"]
        adv = batch["adv"]
        ret = batch["ret"]
        batch_size = obs.shape[0]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_mask, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value = mb
            logits, new_value = network.apply(params, mb_obs)
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
            return total, (actor_loss, value_loss, entropy)

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)
            idxs = permutation.reshape(config["num_minibatches"], minibatch)

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
                )
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, None), {
                    "total": loss,
                    "actor": aux[0],
                    "value": aux[1],
                    "entropy": aux[2],
                }

            (train_state, _), losses = jax.lax.scan(
                _minibatch, (train_state, None), idxs
            )
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            _epoch, (train_state, rng), None, config["ppo_epochs"]
        )
        return train_state, rng, losses

    def update_all(train_states, batches, rng):
        agent_rngs = jax.random.split(rng, num_agents)
        train_states, _, losses = jax.vmap(update_agent)(
            train_states, batches, agent_rngs
        )
        return train_states, losses

    def train(rng, runner_state=None):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, update_step):
            runner_state, (transitions, last_value, finished_returns, finished_lengths) = (
                rollout(runner_state)
            )
            advantages, returns = compute_gae(transitions, last_value)
            train_states, state, timestep, rng, ep_return, ep_length = runner_state
            batches = per_agent_batches(transitions, advantages, returns)
            train_states, losses = update_all(train_states, batches, rng)
            rng, _ = jax.random.split(rng)
            metric = {
                "loss_total": losses["total"].mean(),
                "entropy": losses["entropy"].mean(),
                "num_finished_episodes": jnp.sum(finished_lengths > 0),
                "finished_return_sum": jnp.sum(finished_returns),
                "finished_length_sum": jnp.sum(finished_lengths),
                "env_step": (update_step + 1) * config["num_envs"] * config["rollout_length"],
            }
            new_runner_state = (
                train_states,
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

    return train, network


def behaviour_probe(config: dict, env, network, num_episodes: int, seed: int):
    """Per-agent action histograms on a fixed evaluation set (agent-wise independence/tracking)."""
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    num_agents = config["num_agents"]

    @jax.jit
    def probe(params, keys):
        def _episode(key):
            state, timestep = env.reset(key)

            def _step(carry, t):
                state, timestep, live, hist, probs = carry
                features, action_mask = preprocess(timestep.observation, num_agents)
                logits, _ = jax.vmap(lambda p, o: network.apply(p, o))(params, features)
                logits = masked_logits(logits, action_mask)
                action = jnp.argmax(logits, axis=-1)
                one_hot = jax.nn.one_hot(action, config["action_dim"])
                hist = hist + jnp.where(live, one_hot, 0.0)
                probs = probs + jnp.where(
                    live, jax.nn.softmax(logits, axis=-1), 0.0
                )
                state, timestep = env.step(state, action)
                live = live & ~timestep.last()
                return (state, timestep, live, hist, probs), None

            init = (
                state,
                timestep,
                jnp.array(True),
                jnp.zeros((num_agents, config["action_dim"])),
                jnp.zeros((num_agents, config["action_dim"])),
            )
            (_, _, _, hist, probs), _ = jax.lax.scan(
                _step, init, jnp.arange(config["time_limit"])
            )
            return hist, probs

        return jax.vmap(_episode)(keys)

    return probe, keys


def save_checkpoint(path: Path, runner_state, meta: dict) -> None:
    payload = {
        "params": jax.tree.map(np.asarray, runner_state[0].params),
        "opt_state": jax.tree.map(np.asarray, runner_state[0].opt_state),
        "step": int(runner_state[0].step[0]),
        "meta": meta,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dump-behaviour", action="store_true", help="write per-segment action histograms")
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("rware_iippo_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    env = make_env(config["time_limit"])
    train, network = make_train(config, args.segment_updates, env)
    train_jit = jax.jit(train)
    probe, probe_keys = behaviour_probe(
        config, env, network, args.eval_episodes, args.eval_seed
    )

    metadata = {
        "kind": "rware_4p_independent_ippo_learning_gate",
        "independent_policies": True,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": config,
        "scenario": TINY_4AG,
        "eval_fingerprint": fingerprint_of(env, config["num_agents"])(probe_keys),
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

        histograms = None
        if args.dump_behaviour:
            raw_hist, raw_probs = jax.device_get(probe(runner_state[0].params, probe_keys))
            histograms = np.asarray(raw_hist)
            probabilities = np.asarray(raw_probs)

        cumulative_env_steps = (
            (segment + 1) * args.segment_updates * args.num_envs * args.rollout_length
        )
        finished = float(metrics["num_finished_episodes"].sum())
        record = {
            "segment": segment + 1,
            "cumulative_env_steps": cumulative_env_steps,
            "train_seconds": round(train_seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * args.num_envs * args.rollout_length / train_seconds, 1
            ),
            "train_episodes_finished": finished,
            "train_delivered_per_episode": (
                float(metrics["finished_return_sum"].sum() / finished) if finished else 0.0
            ),
            "train_episode_length_mean": (
                float(metrics["finished_length_sum"].sum() / finished) if finished else 0.0
            ),
            "loss_total": float(metrics["loss_total"].mean()),
            "entropy": float(np.asarray(metrics["entropy"]).mean()),
        }
        if histograms is not None:
            totals = histograms.sum(axis=0)  # (agents, action_dim)
            record["action_histograms"] = totals.tolist()
            record["action_distributions"] = (totals / totals.sum(axis=-1, keepdims=True)).tolist()
            mean_probs = probabilities.mean(axis=0)  # (agents, action_dim)
            record["sampled_action_probabilities"] = (mean_probs / mean_probs.sum(axis=-1, keepdims=True)).tolist()

        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        save_checkpoint(
            out_dir / f"checkpoint_{cumulative_env_steps:010d}.pkl",
            runner_state,
            {"cumulative_env_steps": cumulative_env_steps, "segment": segment + 1},
        )
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[segment {segment + 1}/{num_segments}] env_steps={cumulative_env_steps} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"delivered/ep={record['train_delivered_per_episode']:.2f} "
            f"entropy={record['entropy']:.3f}",
            flush=True,
        )

    print(
        f"[rware independent gate] finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
