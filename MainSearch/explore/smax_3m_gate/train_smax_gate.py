"""SMAX `3m` learning gate: three fixed-identity allies with fully observable partners.

Why this environment (see ../benchmark_screening_2026_09/RESULTS.md): it is already installed,
runs at ~9,200 env steps/s with a GRU policy on this CPU, and - unlike RWARE, where a partner is
inside the observer's view only ~9% of the time - each agent's 9x9 view contains its allies and
the enemies, so partner behaviour is both observable and consequential (focus fire, positioning).

Scope: learning gate only. Shared parameters over the three allies; the ETM stage needs three
independent policies, which comes next and only if this passes.

Frozen evaluation: 100 fixed battle seeds, deterministic (argmax over valid actions), reporting
win rate, mean return and mean battle length.

Usage (absolute interpreter path required on this machine):
  python train_smax_gate.py --map 3m --num-envs 64 --rollout-length 64 --updates 2000
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

from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario  # noqa: E402
from jaxmarl.environments.smax.smax_env import SMAX  # noqa: E402
from jaxmarl.wrappers.baselines import SMAXLogWrapper  # noqa: E402


class MLPActorCritic(nn.Module):
    action_dim: int
    hidden: int = 128

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


def make_env(args):
    scenario = map_name_to_scenario(args.map)
    env = HeuristicEnemySMAX(
        scenario=scenario,
        see_enemy_actions=True,
        walls_cause_death=True,
        attack_mode="closest",
    )
    return env


def build_config(args, env) -> dict:
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": env.num_agents,
        "action_dim": env.action_space(env.agents[0]).n,
        "obs_dim": int(env.observation_space(env.agents[0]).shape[0]),
        "lr": args.lr,
        "ppo_epochs": args.ppo_epochs,
        "num_minibatches": args.num_minibatches,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": args.clip_eps,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
    }


def make_train(config: dict, run_updates: int, env: SMAXLogWrapper, base_env: SMAX):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_actors = num_envs * num_agents
    obs_dim = config["obs_dim"]
    network = MLPActorCritic(action_dim=config["action_dim"])

    def obs_batch_of(obs):
        return jnp.stack([obs[a] for a in env.agents]).reshape(num_actors, obs_dim).astype(
            jnp.float32
        )

    def avail_batch_of(env_state):
        avail = jax.vmap(base_env.get_avail_actions)(env_state.env_state)
        return jnp.stack([avail[a] for a in env.agents]).reshape(num_actors, -1).astype(bool)

    def masked(logits, avail):
        return jnp.where(avail, logits, -1e9)

    def init_runner_state(rng):
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        params = network.init(init_rng, jnp.zeros((num_actors, obs_dim)))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            ),
        )
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
        )

    def rollout(runner_state):
        def _step(carry, _):
            train_state, env_state, obs, done, rng, ep_return, ep_length, ep_won = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            flat = obs_batch_of(obs)
            avail = avail_batch_of(env_state)
            logits, value = network.apply(train_state.params, flat)
            logits = masked(logits, avail)
            action = jax.random.categorical(act_rng, logits)
            log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1), action[:, None], axis=-1
            )[:, 0]

            env_act = {
                a: action.reshape(num_agents, num_envs)[i] for i, a in enumerate(env.agents)
            }
            new_obs, new_env_state, reward, done_new, info = jax.vmap(env.step)(
                jax.random.split(step_rng, num_envs), env_state, env_act
            )
            last = done_new["__all__"]
            raw = jnp.stack([reward[a] for a in env.agents]).reshape(num_agents, num_envs)
            raw_per_env = raw[0]
            ep_return = ep_return + raw_per_env
            ep_length = ep_length + 1
            # JaxMARL convention (SMAXLogWrapper): a win is a final-step reward >= 1.0
            won = jnp.asarray(info["returned_won_episode"][:, 0], dtype=jnp.float32)

            transition = {
                "obs": flat,
                "avail": avail,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": jnp.repeat(raw_per_env, num_agents),
                "done": jnp.repeat(last, num_agents),
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            finished_won = jnp.where(last, won, 0.0)
            new_carry = (
                train_state,
                new_env_state,
                new_obs,
                jnp.stack([done_new[a] for a in env.agents]).reshape(-1),
                rng,
                jnp.where(last, 0.0, ep_return),
                jnp.where(last, 0, ep_length),
                jnp.where(last, 0.0, ep_won),
            )
            return new_carry, (transition, finished_return, finished_length, finished_won)

        carry, (transitions, f_ret, f_len, f_won) = jax.lax.scan(
            _step, runner_state, None, config["rollout_length"]
        )
        train_state, env_state, obs, done, rng, ep_return, ep_length, ep_won = carry
        flat = obs_batch_of(obs)
        avail = avail_batch_of(env_state)
        _, last_value = network.apply(train_state.params, flat)
        new_runner_state = (train_state, env_state, obs, done, rng, ep_return, ep_length, ep_won)
        return new_runner_state, transitions, last_value, (f_ret, f_len, f_won)

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
        batch_size = num_actors * config["rollout_length"]
        flat = lambda x: x.reshape(batch_size, *x.shape[2:])
        obs_f = flat(transitions["obs"])
        avail_f = flat(transitions["avail"])
        action_f = flat(transitions["action"])
        logprob_f = flat(transitions["log_prob"])
        value_f = flat(transitions["value"])
        adv_f = advantages.reshape(batch_size)
        ret_f = returns.reshape(batch_size)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_avail, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value = mb
            logits, new_value = network.apply(params, mb_obs)
            logits = masked(logits, mb_avail)
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
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
            return total, (actor_loss, value_loss, entropy)

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
                    avail_f[idx],
                    action_f[idx],
                    logprob_f[idx],
                    adv_f[idx],
                    ret_f[idx],
                    value_f[idx],
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
            runner_state, transitions, last_value, (f_ret, f_len, f_won) = rollout(runner_state)
            advantages, returns = compute_gae(transitions, last_value)
            train_state, rng, losses = update(
                runner_state[0], transitions, advantages, returns, runner_state[4]
            )
            metric = {
                "loss_total": losses["total"].mean(),
                "entropy": losses["entropy"].mean(),
                "num_finished_episodes": jnp.sum(f_len > 0),
                "finished_return_sum": jnp.sum(f_ret),
                "finished_won_sum": jnp.sum(f_won),
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
            )
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, network


def build_eval_fn(config: dict, env: SMAXLogWrapper, base_env: SMAX, num_episodes: int, seed: int):
    network = MLPActorCritic(action_dim=config["action_dim"])
    obs_dim = config["obs_dim"]
    num_agents = config["num_agents"]
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    horizon = base_env.max_steps

    def _make(sample: bool):
        @jax.jit
        def evaluate(params, keys):
            def _episode(key):
                obs, env_state = env.reset(key)

                def _step(carry, t):
                    env_state, obs, live, ep_return, length, won = carry
                    avail = base_env.get_avail_actions(env_state.env_state)
                    avail = jnp.stack([avail[a] for a in env.agents]).astype(bool)
                    flat = jnp.stack([obs[a] for a in env.agents]).astype(jnp.float32)
                    logits, _ = network.apply(params, flat)
                    logits = jnp.where(avail, logits, -1e9)
                    if sample:
                        action = jax.vmap(
                            lambda k, logit: jax.random.categorical(k, logit)
                        )(jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
                    else:
                        action = jnp.argmax(logits, axis=-1)
                    env_act = {a: action[i] for i, a in enumerate(env.agents)}
                    obs, env_state, reward, done, info = env.step(
                        jax.random.fold_in(key, t), env_state, env_act
                    )
                    step_reward = reward[env.agents[0]]
                    ep_return = ep_return + jnp.where(live, step_reward, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    won = jnp.where(live & (step_reward >= 1.0), 1.0, won)
                    live = live & ~done["__all__"]
                    return (env_state, obs, live, ep_return, length, won), None

                init = (env_state, obs, jnp.array(True), 0.0, jnp.int32(0), 0.0)
                (_, _, _, ep_return, length, won), _ = jax.lax.scan(
                    _step, init, jnp.arange(horizon)
                )
                return jnp.stack([ep_return, length.astype(jnp.float32), won])

            return jax.vmap(_episode)(keys)

        return evaluate

    def fingerprint(keys):
        def _first(key):
            obs, _ = env.reset(key)
            return jnp.stack([obs[a] for a in env.agents]).astype(jnp.float32)

        views = jax.jit(jax.vmap(_first))(keys)
        return {"num_episodes": int(views.shape[0]), "obs_sum": float(np.asarray(views).sum())}

    return {"greedy": _make(sample=False), "sampled": _make(sample=True)}, fingerprint, keys


def summarize(raw: np.ndarray) -> dict:
    return {
        "mean_return": float(raw[:, 0].mean()),
        "win_rate": float(raw[:, 2].mean()),
        "mean_length": float(raw[:, 1].mean()),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="3m")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=64)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-3)
    parser.add_argument("--clip-eps", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=0.25)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("smax_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = make_env(args)
    config = build_config(args, base_env)
    env = SMAXLogWrapper(base_env)
    if config["num_agents"] * config["num_envs"] % config["num_minibatches"]:
        raise SystemExit("num_actors must be divisible by num_minibatches")

    train, _ = make_train(config, args.segment_updates, env, base_env)
    train_jit = jax.jit(train)
    evaluators, fingerprint, eval_keys = build_eval_fn(
        config, env, base_env, args.eval_episodes, args.eval_seed
    )

    metadata = {
        "kind": "smax_3m_learnability_gate",
        "shared_parameters": True,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": config,
        "map": args.map,
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
        result = train_jit(rng) if runner_state is None else train_jit(rng, runner_state)
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
            "train_win_rate": (
                float(metrics["finished_won_sum"].sum() / finished) if finished else 0.0
            ),
            "train_return_mean": (
                float(metrics["finished_return_sum"].sum() / finished) if finished else 0.0
            ),
            "entropy": float(metrics["entropy"].mean()),
            "eval_greedy": summarize(greedy_raw),
            "eval_sampled": summarize(sampled_raw),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        payload = {
            "params": jax.tree.map(np.asarray, params),
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
            f"train_win={record['train_win_rate']:.3f} ret={record['train_return_mean']:.2f} | "
            f"eval greedy win {record['eval_greedy']['win_rate']:.3f} "
            f"return {record['eval_greedy']['mean_return']:.2f} | "
            f"sampled win {record['eval_sampled']['win_rate']:.3f}",
            flush=True,
        )

    print(
        f"[smax gate] finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
