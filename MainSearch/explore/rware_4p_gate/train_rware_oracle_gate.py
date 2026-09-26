"""Oracle Gate on RWARE: does *true* teammate-capability information change the ego's decisions?

Rationale (see ../EXPERIMENT_SUMMARY_2026-09-26.md): the RWARE closed-loop experiments showed that
explicit teammate models produced no reproducible benefit, but they never tested the causal
premise directly. If even an oracle - the ego being *told* the true capability level of its
partners - cannot improve team returns, then no learned teammate model can help in this task, and
the layout is unsuitable for ETM.

Design:
  * Environment: RWARE `tiny-4ag` (4 agents, auto-reset at time_limit). The sensor range must
    match the frozen partner checkpoints (the 20.48M-step run used sensor_range=1).
  * Agent 0 is the ego and the only agent being trained.
  * Agents 1..3 are FROZEN partners taken from a checkpoint of the 20.48M-step independent-policy
    run, at three competence levels (low / mid / high). At every episode boundary one level is
    sampled per environment; all three partners that episode come from that level (agent indices
    1..3 of the checkpoint, so the partners differ from each other).
    NOTE: frozen checkpoints are used for *screening only* (structural diagnostic). The main
    experiment requires partners that keep learning online from a warm start.
  * Arms:
      - `no_oracle` : ego sees only its own observation.
      - `oracle`    : ego additionally sees a one-hot of the partners' true competence level.

Evaluation: the same 32 fixed initial states played once per competence level (96 episodes per
checkpoint), deterministic (argmax) and sampled actions; per-level and overall delivered shelves.
Pass criterion for the layout: `oracle - no_oracle >= 1.0` delivered shelves per episode.
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
    log_prob_of,
    make_env,
    masked_logits,
    preprocess,
)

ARMS = ["no_oracle", "oracle"]
NUM_LEVELS = 3
LEVEL_CHECKPOINTS = [  # low, mid, high competence (cumulative env steps of the partner run)
    "checkpoint_0000409600.pkl",
    "checkpoint_0008601600.pkl",
    "checkpoint_0016793600.pkl",
]


def load_level_params(run_dir: Path):
    """(levels, agents, ...) frozen partner parameters from the training run."""
    per_level = []
    for name in LEVEL_CHECKPOINTS:
        with (run_dir / name).open("rb") as handle:
            params = pickle.load(handle)["params"]
        per_level.append(jax.tree.map(jnp.asarray, params))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *per_level)


def build_config(args: argparse.Namespace, num_features: int) -> dict:
    num_agents = TINY_4AG["num_agents"]
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": num_agents,
        "action_dim": 5,
        "num_features": num_features,
        "ego_features": num_features + (NUM_LEVELS if args.arm == "oracle" else 0),
        "lr": 2.5e-4,
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
        "seed": args.seed,
    }


def make_train(config: dict, run_updates: int, env, level_params):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    action_dim = config["action_dim"]
    num_features = config["num_features"]
    net = ActorCritic(action_dim=action_dim)

    def ego_input(features, levels):
        """Ego features, optionally with the oracle one-hot of the partners' competence level."""
        if config["ego_features"] == num_features:
            return features[:, 0]
        return jnp.concatenate([features[:, 0], jax.nn.one_hot(levels, NUM_LEVELS)], axis=-1)

    def init_runner_state(rng):
        rng, init_rng, reset_rng, level_rng = jax.random.split(rng, 4)
        params = net.init(init_rng, jnp.zeros((1, config["ego_features"])))
        train_state = TrainState.create(
            apply_fn=net.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            ),
        )
        state, timestep = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        levels = jax.random.randint(level_rng, (num_envs,), 0, NUM_LEVELS)
        return (
            train_state,
            state,
            timestep,
            levels,
            rng,
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        )

    def rollout(runner_state):
        def _step(carry, _):
            train_state, state, timestep, levels, rng, ep_return, ep_length = carry
            rng, ego_rng, partner_rng, level_rng = jax.random.split(rng, 4)
            features, action_mask = preprocess(timestep.observation, num_agents)
            obs_ego = ego_input(features, levels)  # snapshot before the level is resampled

            logits_ego, value = net.apply(
                train_state.params, obs_ego
            )
            logits_ego = masked_logits(logits_ego, action_mask[:, 0])
            action_ego = jax.random.categorical(ego_rng, logits_ego)
            log_prob_ego = log_prob_of(logits_ego, action_ego)

            def _one_env_partner(params_by_level, env_features, env_mask, level, key):
                chosen = jax.tree.map(lambda x: x[level], params_by_level)
                partner_params = jax.tree.map(lambda x: x[1:], chosen)
                logits_p, _ = jax.vmap(lambda p, o: net.apply(p, o))(
                    partner_params, env_features[1:]
                )
                logits_p = masked_logits(logits_p, env_mask[1:])
                keys = jax.random.split(key, num_agents - 1)
                return jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(
                    keys, logits_p
                )

            partner_actions = jax.vmap(
                _one_env_partner, in_axes=(None, 0, 0, 0, 0)
            )(level_params, features, action_mask, levels, jax.random.split(partner_rng, num_envs))

            actions = jnp.concatenate([action_ego[:, None], partner_actions], axis=1)
            new_state, new_timestep = jax.vmap(env.step)(state, actions)
            last = new_timestep.last()
            reward = new_timestep.reward
            ep_return = ep_return + reward
            ep_length = ep_length + 1
            levels = jnp.where(
                last, jax.random.randint(level_rng, (num_envs,), 0, NUM_LEVELS), levels
            )

            transition = {
                "obs": obs_ego,
                "mask": action_mask[:, 0],
                "action": action_ego,
                "log_prob": log_prob_ego,
                "value": value,
                "reward": reward,
                "done": last,
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            new_carry = (
                train_state,
                new_state,
                new_timestep,
                levels,
                rng,
                jnp.where(last, 0.0, ep_return),
                jnp.where(last, 0, ep_length),
            )
            return new_carry, (transition, finished_return, finished_length)

        carry, (transitions, f_ret, f_len) = jax.lax.scan(
            _step, runner_state, None, config["rollout_length"]
        )
        train_state, state, timestep, levels, rng, ep_return, ep_length = carry
        features, _ = preprocess(timestep.observation, num_agents)
        _, last_value = net.apply(train_state.params, ego_input(features, levels))
        new_runner_state = (train_state, state, timestep, levels, rng, ep_return, ep_length)
        return new_runner_state, transitions, last_value, (f_ret, f_len)

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

    def update(train_state, transitions, advantages, returns, rng):
        batch_size = num_envs * config["rollout_length"]
        flat = lambda x: x.reshape(batch_size, *x.shape[2:])
        obs_f = flat(transitions["obs"])
        mask_f = flat(transitions["mask"])
        action_f = flat(transitions["action"])
        logprob_f = flat(transitions["log_prob"])
        value_f = flat(transitions["value"])
        adv_f = advantages.reshape(batch_size)
        ret_f = returns.reshape(batch_size)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_mask, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value = mb
            logits, new_value = net.apply(params, mb_obs)
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
                mb = (
                    obs_f[idx], mask_f[idx], action_f[idx], logprob_f[idx],
                    adv_f[idx], ret_f[idx], value_f[idx],
                )
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                return (train_state.apply_gradients(grads=grads), None), {
                    "total": loss,
                    "entropy": aux[2],
                }

            (train_state, _), losses = jax.lax.scan(_minibatch, (train_state, None), idxs)
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            _epoch, (train_state, rng), None, config["ppo_epochs"]
        )
        return train_state, rng, losses

    def train(rng, runner_state=None):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, update_step):
            runner_state, transitions, last_value, (f_ret, f_len) = rollout(runner_state)
            advantages, returns = compute_gae(transitions, last_value)
            train_state, rng, losses = update(
                runner_state[0], transitions, advantages, returns, runner_state[4]
            )
            metric = {
                "loss_total": losses["total"].mean(),
                "entropy": losses["entropy"].mean(),
                "num_finished_episodes": jnp.sum(f_len > 0),
                "finished_return_sum": jnp.sum(f_ret),
                "finished_length_sum": jnp.sum(f_len),
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
            )
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, net


def build_eval_fn(config: dict, env, level_params, num_episodes: int, seed: int):
    num_agents = config["num_agents"]
    num_features = config["num_features"]
    net = ActorCritic(action_dim=config["action_dim"])
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)

    def _make(sample: bool):
        @jax.jit
        def evaluate(ego_params, keys, level):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, delivered, length = carry
                    features, action_mask = preprocess(timestep.observation, num_agents)
                    obs_ego = features[0]
                    if config["ego_features"] != num_features:
                        obs_ego = jnp.concatenate(
                            [obs_ego, jax.nn.one_hot(level, NUM_LEVELS)], axis=-1
                        )
                    logits_ego, _ = net.apply(ego_params, obs_ego)
                    logits_ego = masked_logits(logits_ego, action_mask[0])
                    if sample:
                        action_ego = jax.random.categorical(
                            jax.random.fold_in(key, t), logits_ego
                        )
                    else:
                        action_ego = jnp.argmax(logits_ego)

                    chosen = jax.tree.map(lambda x: x[level], level_params)
                    partner_params = jax.tree.map(lambda x: x[1:], chosen)
                    logits_p, _ = jax.vmap(lambda p, o: net.apply(p, o))(
                        partner_params, features[1:]
                    )
                    logits_p = masked_logits(logits_p, action_mask[1:])
                    if sample:
                        partner_actions = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(
                            jax.random.split(jax.random.fold_in(key, t + 1000), num_agents - 1),
                            logits_p,
                        )
                    else:
                        partner_actions = jnp.argmax(logits_p, axis=-1)
                    action = jnp.concatenate([action_ego[None], partner_actions])
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
    parser.add_argument("--sensor-range", type=int, default=1,
                        help="must match the partner checkpoints (the 20M run used 1)")
    parser.add_argument("--partner-run", default="results/rware_tiny4ag_iippo_indep_seed0_20M")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime(f"rware_oracle_{args.arm}_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.time_limit, args.sensor_range)
    num_features = int(env.observation_spec.agents_view.shape[-1]) + TINY_4AG["num_agents"]
    config = build_config(args, num_features)
    partner_run = Path(args.partner_run)
    if not partner_run.is_absolute():
        partner_run = HERE / partner_run
    level_params = load_level_params(partner_run)

    train, _ = make_train(config, args.segment_updates, env, level_params)
    train_jit = jax.jit(train)
    evaluators, eval_keys = build_eval_fn(
        config, env, level_params, args.eval_episodes, args.eval_seed
    )

    metadata = {
        "kind": "rware_oracle_capability_gate",
        "arm": args.arm,
        "frozen_partner_checkpoints_used_for_screening_only": True,
        "partner_run": str(partner_run),
        "partner_levels": LEVEL_CHECKPOINTS,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": config,
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

        ego_params = runner_state[0].params
        per_level = {}
        for level in range(NUM_LEVELS):
            g_del, _ = jax.device_get(
                evaluators["greedy"](ego_params, eval_keys, jnp.asarray(level))
            )
            s_del, s_len = jax.device_get(
                evaluators["sampled"](ego_params, eval_keys, jnp.asarray(level))
            )
            per_level[f"level{level}"] = {
                "greedy_delivered_mean": float(np.asarray(g_del).mean()),
                "sampled_delivered_mean": float(np.asarray(s_del).mean()),
                "sampled_delivered_max": float(np.asarray(s_del).max()),
                "sampled_episode_length_mean": float(np.asarray(s_len).mean()),
            }
        overall = {
            "sampled_delivered_mean": float(
                np.mean([v["sampled_delivered_mean"] for v in per_level.values()])
            ),
            "greedy_delivered_mean": float(
                np.mean([v["greedy_delivered_mean"] for v in per_level.values()])
            ),
        }

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
            "entropy": float(np.mean(metrics["entropy"])),
            "eval_overall": overall,
            "eval_per_level": per_level,
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "ego_params": jax.tree.map(np.asarray, ego_params),
                    "meta": {"cumulative_env_steps": cumulative, "arm": args.arm},
                },
                handle,
            )
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{args.arm} {segment + 1}/{num_segments}] steps={cumulative} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"train_deliv/ep={record['train_delivered_per_episode']:.2f} | "
            f"eval overall {overall['sampled_delivered_mean']:.2f} "
            f"(low {per_level['level0']['sampled_delivered_mean']:.2f} / "
            f"mid {per_level['level1']['sampled_delivered_mean']:.2f} / "
            f"high {per_level['level2']['sampled_delivered_mean']:.2f})",
            flush=True,
        )

    print(
        f"[rware oracle gate] arm={args.arm} finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
