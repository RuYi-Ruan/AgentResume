"""Fixed-identity independent actors on the Three-Arm Kitchen (the P2 setup).

Why: the research needs partners whose competence *evolves through experience* while keeping fixed
identities. A shared policy cannot express that (all three agents are one function), and it also
failed to improve beyond its behaviour-cloning initialisation.

Design (follows the external guidance):
  * Three separate actors pi_1, pi_2, pi_3, each with its own optimizer state and gradient stream;
    all warm-started from the same behaviour-cloned generalist, then free to diverge.
  * Per-agent centralised critics V_i(s) that see the compact privileged state (35 dims) rather than
    the local 7x7 view (the local-view critic measured explained variance ~0).
  * Training reward = per-agent (raw + shaped) with a fixed shaping coefficient; evaluation metric
    stays raw team soups.
  * Stabilisers: value normalisation, critic-only warm-up, decaying demo imitation + reference-KL
    anchors with a small permanent floor.

Implementation note: every jitted function takes the parameters as *arguments*; nothing that changes
per update is closed over, so nothing recompiles mid-training.

Usage:
  python train_ocv2_independent.py --updates 1100 --segment-updates 20 --run-name ocv2_indep_seed0_9M
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

HERE = Path(__file__).resolve().parent
FF_GATE = HERE.parent / "overcooked_v2_ff_gate"
sys.path.insert(0, str(FF_GATE))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import (  # noqa: E402
    MLPActorCritic,
    MLPActorCriticPriv,
    compact_state,
    critic_spec,
)
from train_ocv2_oracle_gate import load_plain_params  # noqa: E402

# The behaviour-cloned actor always occupies these layers in MLPActorCriticPriv; never infer the
# split from dictionary order.
ACTOR_LAYERS = {"Dense_0", "Dense_1", "Dense_2"}


class Ctx:
    privileged_critic = True


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1100)
    parser.add_argument("--segment-updates", type=int, default=20)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=1e-5,
                        help="the 1470-dim observation (L1 ~30) makes the policy extremely "
                             "sensitive: 2.5e-4 moved the logits by ~5 nats per update and turned "
                             "the policy deterministic within 20 updates (measured)")
    parser.add_argument("--target-kl", type=float, default=0.01,
                        help="per-update trust region: once mean KL(current||rollout) exceeds this, "
                             "the remaining minibatch updates are scaled to zero (soft early stop)")
    parser.add_argument("--lr-critic", type=float, default=3e-5,
                        help="per-agent critics see only 1/3 of the reward stream (sparse delivery "
                             "bonus), so they need a smaller step than a shared critic (measured: "
                             "2.5e-4 diverges, 1e-5/3e-5 stable)")
    parser.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--critic-warmup-updates", type=int, default=100)
    parser.add_argument("--freeze-critic", action="store_true",
                        help="diagnostic: never update the critics (isolates rollout/GAE from the "
                             "critic update path)")
    parser.add_argument("--shaped-coeff", type=float, default=1.0)
    parser.add_argument("--value-normalization", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--bc-init", default="bc/three_arm_bc_seed0.pkl")
    parser.add_argument("--bc-demos", default="demos/three_arm_demos.npz")
    parser.add_argument("--bc-lambda0", type=float, default=1.0)
    parser.add_argument("--bc-lambda-floor", type=float, default=0.1)
    parser.add_argument("--bc-decay-updates", type=int, default=400)
    parser.add_argument("--ref-kl-beta", type=float, default=1.0)
    parser.add_argument("--ref-kl-floor", type=float, default=0.1)
    parser.add_argument("--ref-kl-decay-updates", type=int, default=400)
    parser.add_argument("--bc-pool", type=int, default=10000)
    parser.add_argument("--bc-batch", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    layout = Layout.from_string((HERE / args.layout_file).read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    crit = critic_spec(Ctx(), env)
    crit["context"] = crit
    num_agents = len(env.agents)
    num_envs = args.num_envs
    num_actors = num_envs * num_agents
    obs_flat = int(np.prod(env.observation_space().shape))
    critic_dim = crit["dim"]

    bc_payload = load_plain_params(resolve(args.bc_init))
    # keep the nested {kernel, bias} sub-dicts: they are copied into the fresh parameter tree
    bc_actor = {k: v for k, v in bc_payload["params"].items()
                if k in ("Dense_0", "Dense_1", "Dense_2")}
    ref_module = MLPActorCritic(action_dim=6)
    ref_params = bc_payload  # plain module weights, used for the reference-policy KL

    demo_file = np.load(resolve(args.bc_demos))
    # keep the demonstration set in its stored int8 form and convert only the pool subset: the
    # full float32 copy would be ~1.3 GB and competes with concurrent training runs
    demo_obs_all = demo_file["obs"].reshape(-1, int(np.prod(demo_file["obs"].shape[2:])))
    demo_act_all = demo_file["action"].reshape(-1).astype(np.int32)
    pool_rng = np.random.default_rng(args.seed)
    pool_idx = pool_rng.permutation(len(demo_obs_all))[: args.bc_pool]
    demo_obs = jnp.asarray(demo_obs_all[pool_idx].astype(np.float32))
    demo_act = jnp.asarray(demo_act_all[pool_idx])
    del demo_obs_all, demo_act_all

    network = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)
    agent_module = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)

    def init_params(rng):
        fresh = agent_module.init(rng, jnp.zeros((num_actors, obs_flat)),
                                  jnp.zeros((num_actors, critic_dim)))["params"]
        merged = dict(fresh)
        for name, subtree in bc_actor.items():
            if name in merged and all(np.shape(merged[name][leaf]) == np.shape(value)
                                      for leaf, value in subtree.items()):
                merged[name] = subtree
        return {"params": merged}

    def make_optimizer(params):
        assert ACTOR_LAYERS <= set(params["params"].keys()), "actor layers missing"
        labels = {"params": {n: {leaf: ("actor" if n in ACTOR_LAYERS else "critic")
                                 for leaf in sub} for n, sub in params["params"].items()}}
        return optax.multi_transform(
            {"actor": optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                                  optax.adam(args.lr_actor, eps=1e-5)),
             "critic": optax.chain(optax.clip_by_global_norm(args.max_grad_norm),
                                   optax.adam(args.lr_critic, eps=1e-5))},
            labels)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)
    init_rngs = jax.random.split(init_rng, num_agents)
    params = [init_params(k) for k in init_rngs]
    optimizers = [make_optimizer(p) for p in params]
    opt_states = [opt.init(p) for opt, p in zip(optimizers, params)]

    actor_mask = {"params": {n: {leaf: jnp.asarray(n in ACTOR_LAYERS) for leaf in sub}
                             for n, sub in params[0]["params"].items()}}

    def obs_flat_of(obs):
        return jnp.stack([obs[a] for a in env.agents], axis=1).reshape(num_envs, num_agents, -1)

    def init_runner(rng):
        rng, reset_rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        return (obsv, env_state, rng, jnp.zeros(3))

    @jax.jit
    def rollout(params, runner, shaped_coeff):
        def _step(carry, t):
            obsv, env_state, rng, stats = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            flat = obs_flat_of(obsv)                                # (envs, agents, obs)
            cin = compact_state(env_state, crit["context"], t).reshape(num_envs, num_agents, -1)
            acts = []
            logps = []
            values = []
            for i in range(num_agents):
                logits, value = network.apply(params[i], flat[:, i], cin[:, i])
                a = jax.random.categorical(jax.random.fold_in(act_rng, i), logits)
                lp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1),
                                         a[:, None], axis=-1)[:, 0]
                acts.append(a)
                logps.append(lp)
                values.append(value)
            action = jnp.stack(acts, axis=1)
            env_act = {a: action[:, i] for i, a in enumerate(env.agents)}
            new_obsv, new_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(step_rng, num_envs), env_state, env_act)
            last = done["__all__"]
            rew = jnp.stack([reward[a] for a in env.agents], axis=1)
            shaped = jnp.stack([info["shaped_reward"][a] for a in env.agents], axis=1)
            transitions = {
                "obs": flat,
                "critic_obs": cin,
                "action": action,
                "log_prob": jnp.stack(logps, axis=1),
                "value": jnp.stack(values, axis=1),
                "reward": rew + shaped_coeff * shaped,
                "shaped": shaped,
                "done": jnp.repeat(last[:, None], num_agents, axis=1),
            }
            return (new_obsv, new_state, rng, stats), transitions

        runner, transitions = jax.lax.scan(_step, runner, jnp.arange(args.rollout_length))
        obsv, env_state, rng, stats = runner
        flat = obs_flat_of(obsv)
        cin = compact_state(env_state, crit["context"], args.rollout_length).reshape(
            num_envs, num_agents, -1)
        last_value = jnp.stack(
            [network.apply(params[i], flat[:, i], cin[:, i])[1] for i in range(num_agents)],
            axis=1)
        return (obsv, env_state, rng, stats), transitions, last_value

    @jax.jit
    def compute_gae(transitions, last_value):
        def _scan(carry, data):
            gae, next_value = carry
            delta = (data["reward"] + args.gamma * next_value * (1 - data["done"])
                     - data["value"])
            gae = delta + args.gamma * args.gae_lambda * (1 - data["done"]) * gae
            return (gae, data["value"]), gae

        _, advantages = jax.lax.scan(
            _scan, (jnp.zeros_like(last_value), last_value), transitions, reverse=True)
        return advantages, advantages + transitions["value"]

    def agent_loss(p, mb):
        mb_obs, mb_cin, mb_action, mb_logp, mb_adv, mb_ret, mb_val = mb
        logits, value = network.apply(p, mb_obs, mb_cin)
        log_all = jax.nn.log_softmax(logits, axis=-1)
        new_logp = jnp.take_along_axis(log_all, mb_action[:, None], axis=-1)[:, 0]
        ratio = jnp.exp(new_logp - mb_logp)
        actor_loss = -jnp.minimum(
            ratio * mb_adv,
            jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_adv).mean()
        v_clip = mb_val + (value - mb_val).clip(-args.clip_eps, args.clip_eps)

        def _robust(err):
            if args.value_loss == "huber":
                abs_err = jnp.abs(err)
                return jnp.where(abs_err <= args.huber_delta, 0.5 * err ** 2,
                                 args.huber_delta * (abs_err - 0.5 * args.huber_delta))
            return 0.5 * err ** 2

        value_loss = jnp.maximum(_robust(value - mb_ret), _robust(v_clip - mb_ret)).mean()
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.mean(jnp.sum(probs * log_all, axis=-1))
        return actor_loss, value_loss, entropy

    def new_logp_of(params, mb):
        logits, _ = network.apply(params, mb[0], mb[1])
        log_all = jax.nn.log_softmax(logits, axis=-1)
        return jnp.take_along_axis(log_all, mb[2][:, None], axis=-1)[:, 0]

    def make_update(rng_key):
        @jax.jit
        def update(params, opt_state, transitions, advantages, returns, rng, idx):
            batch = num_envs * args.rollout_length
            obs = transitions["obs"].reshape(batch, num_agents, -1)[:, rng_key]
            cin = transitions["critic_obs"].reshape(batch, num_agents, -1)[:, rng_key]
            act = transitions["action"].reshape(batch, num_agents)[:, rng_key]
            logp = transitions["log_prob"].reshape(batch, num_agents)[:, rng_key]
            val = transitions["value"].reshape(batch, num_agents)[:, rng_key]
            adv = advantages.reshape(batch, num_agents)[:, rng_key]
            ret = returns.reshape(batch, num_agents)[:, rng_key]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            bc_lambda = args.bc_lambda_floor + (args.bc_lambda0 - args.bc_lambda_floor) * jnp.clip(
                1.0 - idx / args.bc_decay_updates, 0.0, 1.0)
            ref_beta = args.ref_kl_floor + (args.ref_kl_beta - args.ref_kl_floor) * jnp.clip(
                1.0 - idx / args.ref_kl_decay_updates, 0.0, 1.0)
            rng, demo_rng, perm_rng = jax.random.split(rng, 3)
            didx = jax.random.randint(demo_rng, (args.bc_batch,), 0, demo_obs.shape[0])
            d_obs = demo_obs[didx]
            d_act = demo_act[didx]

            def loss_fn(p, mb):
                actor_loss, value_loss, entropy = agent_loss(p, mb)
                total = actor_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                ref_logits, _ = ref_module.apply(ref_params, mb[0])
                p_ref = jax.nn.softmax(ref_logits, axis=-1)
                log_all = jax.nn.log_softmax(network.apply(p, mb[0], mb[1])[0], axis=-1)
                kl = jnp.mean(jnp.sum(p_ref * (jnp.log(p_ref + 1e-9) - log_all), axis=-1))
                d_logits, _ = network.apply(p, d_obs, jnp.zeros((d_obs.shape[0], critic_dim)))
                d_log = jax.nn.log_softmax(d_logits, axis=-1)
                bc_loss = -jnp.mean(jnp.take_along_axis(d_log, d_act[:, None], axis=-1)[:, 0])
                total = total + ref_beta * kl + bc_lambda * bc_loss
                return total, (actor_loss, value_loss, entropy, bc_loss, kl)

            minibatch = batch // args.num_minibatches

            def _epoch(carry, _):
                p_cur, opt_state, rng = carry
                rng, sub = jax.random.split(rng)
                idxs = jax.random.permutation(sub, batch).reshape(
                    args.num_minibatches, minibatch)

                def _minibatch(carry, ix):
                    p_cur, opt_state = carry
                    mb = (obs[ix], cin[ix], act[ix], logp[ix], adv[ix], ret[ix], val[ix])
                    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p_cur, mb)
                    grads = jax.tree.map(
                        lambda g, m: jnp.where(
                            ((idx < args.critic_warmup_updates) | args.freeze_critic) & m,
                            jnp.zeros_like(g), g), grads, actor_mask)
                    if args.freeze_critic:
                        grads = jax.tree.map(jnp.zeros_like, grads)
                    # KL of the current policy against the rollout policy (trust region)
                    approx_kl = jnp.mean(mb[3] - new_logp_of(p_cur, mb))

                    def _apply(_):
                        # ONE optimizer application: tx.update returns the parameter update, and it
                        # must be applied with optax.apply_updates. Passing it to
                        # TrainState.apply_gradients would run Adam a second time (and keep two
                        # divergent optimizer states) - the bug that caused the collapse.
                        upd, new_state = tx.update(grads, opt_state, p_cur)
                        return optax.apply_updates(p_cur, upd), new_state

                    def _skip(_):
                        return p_cur, opt_state

                    p_new, opt_state = jax.lax.cond(
                        approx_kl <= args.target_kl, _apply, _skip, operand=None)
                    return (p_new, opt_state), {
                        "loss": loss, "actor_loss": aux[0], "value_loss": aux[1],
                        "entropy": aux[2], "bc_loss": aux[3], "ref_kl": aux[4],
                        "approx_kl": approx_kl}

                (p_cur, opt_state), losses = jax.lax.scan(
                    _minibatch, (p_cur, opt_state), idxs)
                return (p_cur, opt_state, rng), losses

            tx = optimizers[rng_key]
            (params_out, opt_state, rng), losses = jax.lax.scan(
                _epoch, (params, opt_state, rng), None, args.ppo_epochs)
            return params_out, opt_state, rng, losses

        return update

    updates = [make_update(i) for i in range(num_agents)]

    @jax.jit
    def evaluate(params, keys):
        def _episode(key):
            obs, state = env.reset(key)

            def _step(carry, t):
                obs, state, live, delivered, steps = carry
                flat = jnp.stack([obs[a] for a in env.agents], axis=0).reshape(num_agents, -1)
                cin = compact_state(jax.tree.map(lambda x: x[None], state),
                                    crit["context"], t).reshape(num_agents, -1)
                acts = []
                for i in range(num_agents):
                    logits, _ = network.apply(params[i], flat[i][None], cin[i][None])
                    acts.append(jax.random.categorical(
                        jax.random.fold_in(key, t * num_agents + i), logits[0]))
                action = jnp.stack(acts)
                obs, state, reward, done, info = env.step(
                    key, state, {a: action[i] for i, a in enumerate(env.agents)})
                delivered = delivered + jnp.where(live & state.new_correct_delivery, 1.0, 0.0)
                steps = steps + jnp.where(live, 1, 0)
                live = live & ~done["__all__"]
                return (obs, state, live, delivered, steps), None

            init = (obs, state, jnp.array(True), 0.0, jnp.int32(0))
            (_, _, _, delivered, steps), _ = jax.lax.scan(
                _step, init, jnp.arange(args.max_steps))
            return delivered, steps.astype(jnp.float32)

        return jax.vmap(_episode)(keys)

    eval_keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_episodes)
    run_name = args.run_name or time.strftime("ocv2_indep_%Y%m%d_%H%M%S")
    out_dir = resolve(f"results/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": "ocv2_independent_actors", "not_an_etm_result": True,
                "argv": sys.argv[1:], "segments": [], "wall_seconds_total": None,
                "init_from": str(resolve(args.bc_init)),
                "config": {"num_envs": num_envs, "rollout_length": args.rollout_length,
                           "obs_flat": obs_flat, "critic_dim": critic_dim,
                           "lr_actor": args.lr_actor, "lr_critic": args.lr_critic,
                           "value_loss": args.value_loss, "huber_delta": args.huber_delta,
                           "clip_eps": args.clip_eps, "ent_coef": args.ent_coef,
                           "shaped_coeff": args.shaped_coeff, "value_normalization":
                               args.value_normalization,
                           "critic_warmup_updates": args.critic_warmup_updates,
                           "bc_lambda0": args.bc_lambda0, "bc_lambda_floor": args.bc_lambda_floor,
                           "ref_kl_beta": args.ref_kl_beta, "ref_kl_floor": args.ref_kl_floor,
                           "seed": args.seed}}
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    runner = init_runner(rng)
    started = time.perf_counter()
    num_segments = args.updates // args.segment_updates
    for segment in range(num_segments):
        t0 = time.perf_counter()
        per_agent_metrics = []
        adv_absmax = ret_absmax = 0.0
        value_absmean = reward_norm_std = reward_norm_mean = 0.0
        for step in range(args.segment_updates):
            idx = segment * args.segment_updates + step
            runner, transitions, last_value = rollout(params, runner, args.shaped_coeff)
            if args.value_normalization:
                count, mean, var = runner[3]
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
                transitions = {**transitions, "reward": (rew - new_mean) / jnp.sqrt(new_var)}
                runner = (runner[0], runner[1], runner[2], jnp.stack([total, new_mean, new_var]))
            advantages, returns = compute_gae(transitions, last_value)
            adv_absmax = float(jnp.abs(advantages).max())
            ret_absmax = float(jnp.abs(returns).max())
            value_absmean = float(jnp.abs(transitions["value"]).mean())
            reward_norm_std = float(jnp.asarray(transitions["reward"]).std())
            reward_norm_mean = float(jnp.asarray(transitions["reward"]).mean())
            for i in range(num_agents):
                rng, sub = jax.random.split(runner[2])
                runner = (runner[0], runner[1], rng, runner[3])
                params[i], opt_states[i], _, agent_losses = updates[i](
                    params[i], opt_states[i], transitions, advantages, returns, sub, idx)
                per_agent_metrics.append({k: float(np.asarray(v).mean())
                                          for k, v in agent_losses.items()})
        seconds = time.perf_counter() - t0
        cumulative = (segment + 1) * args.segment_updates * num_envs * args.rollout_length
        params_host = [jax.tree.map(np.asarray, p) for p in params]
        delivered, _ = jax.device_get(evaluate(params, eval_keys))
        delivered = np.asarray(delivered)
        record = {
            "segment": segment + 1, "cumulative_env_steps": cumulative,
            "train_seconds": round(seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * num_envs * args.rollout_length / seconds, 1),
            "eval_team_soups_mean": float(delivered.mean()),
            "eval_team_soups_max": float(delivered.max()),
            "eval_episodes_with_delivery": int((delivered > 0).sum()),
            "loss_by_agent": [m["loss"] for m in per_agent_metrics],
            "value_loss_by_agent": [m["value_loss"] for m in per_agent_metrics],
            "bc_loss_by_agent": [m["bc_loss"] for m in per_agent_metrics],
            "ref_kl_by_agent": [m["ref_kl"] for m in per_agent_metrics],
            "entropy_by_agent": [m["entropy"] for m in per_agent_metrics],
            "approx_kl_by_agent": [m["approx_kl"] for m in per_agent_metrics],
            "loss_max": max((m["loss"] for m in per_agent_metrics), default=0.0),
            "value_loss_max": max((m["value_loss"] for m in per_agent_metrics), default=0.0),
            "bc_loss_max": max((m["bc_loss"] for m in per_agent_metrics), default=0.0),
            "ref_kl_max": max((m["ref_kl"] for m in per_agent_metrics), default=0.0),
            "entropy_min": min((m["entropy"] for m in per_agent_metrics), default=0.0),
            "advantage_absmax": adv_absmax,
            "return_absmax": ret_absmax,
            "value_absmean": value_absmean,
            "reward_norm_std": reward_norm_std,
            "reward_norm_mean": reward_norm_mean,
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump({"params": params_host,
                         "meta": {"cumulative_env_steps": cumulative}}, handle)
        (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
        print(f"[segment {segment + 1}/{num_segments}] steps={cumulative} "
              f"train={seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
              f"eval team soups {record['eval_team_soups_mean']:.2f} "
              f"(max {record['eval_team_soups_max']:.0f}, "
              f"{record['eval_episodes_with_delivery']}/{args.eval_episodes} eps)", flush=True)

    print(f"[indep] finished {args.updates} updates in {metadata['wall_seconds_total']}s; "
          f"wrote {out_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
