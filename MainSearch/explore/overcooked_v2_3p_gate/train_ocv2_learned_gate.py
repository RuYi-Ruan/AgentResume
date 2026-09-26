"""Learning-partner Oracle Gate: do *learned* teammate capabilities still have decision value?

The scripted-archetype gate proved the environment can reward capability-conditioned role choice.
This gate repeats the test with partners taken from the trained independent actors themselves
(fixed identities at a chosen training stage), which is the configuration the main experiment uses.

  * partners: per pair, two (checkpoint, agent-index) entries; agents 1-2 are frozen to them
  * ego: agent 0, the only learner, warm-started from the behaviour-cloned generalist
  * z: the partners' 3D capability vectors [c_ingredient, c_pot, c_service] (measured by
    profile_capabilities.py), NOT the training stage - the stage is a proxy for time, not capability
  * arms: `no_oracle` (actor sees only its own view) vs `oracle` (actor also sees z); the critic
    sees the compact state in both arms, and z only when value aliasing is measured (controlled by
    --critic-sees-capability)
  * evaluation at the end: paired, per condition and overall, on the same fixed initial states,
    with policy-shift metrics (TV / common-random-numbers switch rate) on matched states

Usage:
  python train_ocv2_learned_gate.py --arm oracle \
      --pair "ckptLate:0,ckptLate:1" --pair "ckptLate:0,ckptLate:2" \
      --capabilities results/capability_profiles.json --updates 250 --run-name learned_gate_oracle
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
from flax.training.train_state import TrainState  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import (  # noqa: E402
    MLPActorCritic,
    MLPActorCriticPriv,
    compact_state,
    critic_spec,
)
from train_ocv2_oracle_gate import (  # noqa: E402
    MLPActorOnly,
    actor_subset,
    load_plain_params,
)

ACTOR_LAYERS = {"Dense_0", "Dense_1", "Dense_2"}
CAP_DIM = 3
ARMS = ("no_oracle", "oracle")


class Ctx:
    privileged_critic = True


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def load_independent_partner(path: Path, agent: int):
    """Actor layers of one identity from an independent-actor checkpoint (list of agent params)."""
    payload = pickle.load(path.open("rb"))
    agents = payload["params"]
    inner = agents[agent]["params"] if isinstance(agents, list) else agents["params"]
    return actor_subset(jax.tree.map(jnp.asarray, inner))


def load_capabilities(profile_file: Path):
    """{('checkpoint name', agent index): [c_ingredient, c_pot, c_service]} from the profile run."""
    data = json.loads(profile_file.read_text(encoding="utf-8"))
    table = {}
    for profile in data["profiles"]:
        name = Path(profile["checkpoint"]).name
        for agent, values in profile["capability_vectors"].items():
            table[(name, int(agent.replace("agent", "")))] = [float(v) for v in values]
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--pair", action="append", required=True,
                        help="'checkpoint:agent_index,checkpoint:agent_index'; repeat per condition")
    parser.add_argument("--capabilities", default="results/capability_profiles.json")
    parser.add_argument("--critic-sees-capability", action="store_true",
                        help="only if value aliasing is measured (EV collapses with multiple "
                             "partner conditions); identical in both arms")
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--segment-updates", type=int, default=25)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=1e-5)
    parser.add_argument("--lr-critic", type=float, default=3e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--value-loss", choices=["mse", "huber"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--critic-warmup-updates", type=int, default=100)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--shaped-coeff", type=float, default=1.0)
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
    num_envs, num_pair = args.num_envs, len(args.pair)
    obs_flat = int(np.prod(env.observation_space().shape))
    z_dim = 2 * CAP_DIM
    critic_dim = crit["dim"] + (z_dim if args.critic_sees_capability else 0)

    cap_table = load_capabilities(resolve(args.capabilities))
    pairs, cond_caps = [], []
    for spec in args.pair:
        entries = [e.split(":") for e in spec.split(",")]
        pair_params, caps = [], []
        for path_str, idx_str in entries:
            path = resolve(path_str)
            idx = int(idx_str)
            pair_params.append(load_independent_partner(path, idx))
            key = (path.name, idx)
            if key not in cap_table:
                raise SystemExit(f"capability vector missing for {key}")
            caps.append(cap_table[key])
        pairs.append(jax.tree.map(lambda *xs: jnp.stack(xs), *pair_params))
        cond_caps.append(caps)
        print(f"[gate] condition {len(pairs) - 1}: partners {[e for e in spec.split(',')]} "
              f"capabilities {caps}")

    bc_path = resolve(args.bc_init)
    bc_params = load_plain_params(bc_path)["params"]
    ego_flat = obs_flat + (z_dim if args.arm == "oracle" else 0)

    network = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)
    partner_net = MLPActorOnly(action_dim=6)

    def init_params(rng):
        fresh = network.init(rng, jnp.zeros((1, ego_flat)), jnp.zeros((1, critic_dim)))["params"]
        merged = dict(fresh)
        if args.arm == "oracle":
            kernel = jnp.asarray(bc_params["Dense_0"]["kernel"])
            merged["Dense_0"] = {"kernel": jnp.concatenate(
                [kernel, jnp.zeros((z_dim, kernel.shape[1]))], axis=0),
                "bias": jnp.asarray(bc_params["Dense_0"]["bias"])}
            merged["Dense_1"] = jax.tree.map(jnp.asarray, bc_params["Dense_1"])
            merged["Dense_2"] = jax.tree.map(jnp.asarray, bc_params["Dense_2"])
        else:
            for name, sub in bc_params.items():
                if name in merged and all(np.shape(merged[name][l]) == np.shape(v)
                                          for l, v in sub.items()):
                    merged[name] = jax.tree.map(jnp.asarray, sub)
        return {"params": merged}

    def make_optimizer(params):
        assert ACTOR_LAYERS <= set(params["params"].keys())
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
    params = init_params(init_rng)
    optimizer = make_optimizer(params)
    opt_state = optimizer.init(params)
    actor_mask = {"params": {n: {leaf: jnp.asarray(n in ACTOR_LAYERS) for leaf in sub}
                             for n, sub in params["params"].items()}}

    cap_table_all = jnp.asarray([[c for c in caps] for caps in cond_caps])   # (conditions, 2, 3)

    def cap_vectors(cond):
        return cap_table_all[cond].reshape(-1)

    def init_runner(rng):
        rng, sub = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(sub, num_envs))
        return (obsv, env_state, jnp.zeros(num_envs, dtype=jnp.int32), rng, jnp.zeros(3))

    def obs_flat_of(obsv):
        return jnp.stack([obsv[a] for a in env.agents], axis=1).reshape(num_envs, num_agents, -1)

    def ego_input(flat, z):
        return jnp.concatenate([flat, z], axis=-1) if args.arm == "oracle" else flat

    @jax.jit
    def rollout(params, runner, shaped_coeff):
        def _step(carry, t):
            obsv, env_state, cond, rng, stats = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            flat = obs_flat_of(obsv)
            cin_base = compact_state(env_state, crit["context"], t).reshape(num_envs, num_agents, -1)
            z_all = jax.vmap(cap_vectors)(cond)
            z_ego = z_all[:, None, :] if args.critic_sees_capability else None
            cin = (jnp.concatenate([cin_base, jnp.broadcast_to(
                z_ego, (num_envs, num_agents, z_dim))], axis=-1)
                if args.critic_sees_capability else cin_base)
            logits, value = network.apply(params, ego_input(flat[:, 0], z_all),
                                          cin[:, 0])
            a_ego = jax.random.categorical(act_rng, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1),
                                       a_ego[:, None], axis=-1)[:, 0]

            acts_list = [a_ego]
            for slot, agent_idx in ((0, 1), (1, 2)):
                # every condition's partner policy for this slot, then gather this env's condition
                all_logits = jax.vmap(jax.vmap(partner_net.apply, in_axes=(None, 0)),
                                      in_axes=(0, None))(pairs[slot], flat[:, agent_idx])
                picked = jnp.take_along_axis(all_logits, cond[None, :, None], axis=0)[0]
                acts_list.append(jax.random.categorical(
                    jax.random.fold_in(act_rng, agent_idx), picked))
            actions = jnp.stack(acts_list, axis=1)
            env_act = {a: actions[:, i] for i, a in enumerate(env.agents)}
            new_obsv, new_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(step_rng, num_envs), env_state, env_act)
            last = done["__all__"]
            shaped = jnp.stack([info["shaped_reward"][a] for a in env.agents], axis=1)[:, 0]
            transitions = {
                "obs": ego_input(flat[:, 0], z_all),
                "critic_obs": cin[:, 0],
                "action": a_ego,
                "log_prob": logp,
                "value": value,
                "reward": reward[env.agents[0]] + shaped_coeff * shaped,
                "done": last,
                "shaped": shaped,
            }
            return (new_obsv, new_state, cond, rng, stats), transitions

        runner, transitions = jax.lax.scan(_step, runner, jnp.arange(args.rollout_length))
        obsv, env_state, cond, rng, stats = runner
        flat = obs_flat_of(obsv)
        cin_base = compact_state(env_state, crit["context"], args.rollout_length).reshape(
            num_envs, num_agents, -1)
        z_all = jax.vmap(cap_vectors)(cond)
        cin = (jnp.concatenate([cin_base, jnp.broadcast_to(
            z_all[:, None, :], (num_envs, num_agents, z_dim))], axis=-1)
            if args.critic_sees_capability else cin_base)
        _, last_value = network.apply(params, ego_input(flat[:, 0], z_all), cin[:, 0])
        return (obsv, env_state, cond, rng, stats), transitions, last_value

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

    demo_file = np.load(resolve(args.bc_demos))
    demo_obs_all = demo_file["obs"].reshape(-1, int(np.prod(demo_file["obs"].shape[2:])))
    demo_act_all = demo_file["action"].reshape(-1).astype(np.int32)
    pool_rng = np.random.default_rng(args.seed)
    pool_idx = pool_rng.permutation(len(demo_obs_all))[: args.bc_pool]
    demo_obs = jnp.asarray(demo_obs_all[pool_idx].astype(np.float32))
    demo_act = jnp.asarray(demo_act_all[pool_idx])

    ref_module = MLPActorCritic(action_dim=6)
    ref_params = {"params": bc_params}

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
        return actor_loss, value_loss, entropy, logits

    @jax.jit
    def update(params, opt_state, transitions, advantages, returns, rng, idx):
        batch = num_envs * args.rollout_length
        obs = transitions["obs"].reshape(batch, -1)
        cin = transitions["critic_obs"].reshape(batch, -1)
        act = transitions["action"].reshape(batch)
        logp = transitions["log_prob"].reshape(batch)
        val = transitions["value"].reshape(batch)
        adv = (advantages.reshape(batch) - advantages.mean()) / (advantages.std() + 1e-8)
        ret = returns.reshape(batch)
        bc_lambda = args.bc_lambda_floor + (args.bc_lambda0 - args.bc_lambda_floor) * jnp.clip(
            1.0 - idx / args.bc_decay_updates, 0.0, 1.0)
        ref_beta = args.ref_kl_floor + (args.ref_kl_beta - args.ref_kl_floor) * jnp.clip(
            1.0 - idx / args.ref_kl_decay_updates, 0.0, 1.0)
        rng, demo_rng, perm_rng = jax.random.split(rng, 3)
        didx = jax.random.randint(demo_rng, (args.bc_batch,), 0, demo_obs.shape[0])
        d_obs, d_act = demo_obs[didx], demo_act[didx]

        def loss_fn(p, mb):
            actor_loss, value_loss, entropy, logits = agent_loss(p, mb)
            total = actor_loss + args.vf_coef * value_loss - args.ent_coef * entropy
            # reference policy = the FROZEN behaviour-cloned actor (not the current parameters)
            ref_logits, _ = ref_module.apply(ref_params, mb[0][:, :obs_flat])
            p_ref = jax.nn.softmax(ref_logits, axis=-1)
            log_all = jax.nn.log_softmax(logits, axis=-1)
            kl = jnp.mean(jnp.sum(p_ref * (jnp.log(p_ref + 1e-9) - log_all), axis=-1))
            d_in = (jnp.concatenate([d_obs, jnp.zeros((d_obs.shape[0], ego_flat - obs_flat))], -1)
                    if args.arm == "oracle" else d_obs)
            d_logits, _ = network.apply(p, d_in, jnp.zeros((d_obs.shape[0], critic_dim)))
            d_log = jax.nn.log_softmax(d_logits, axis=-1)
            bc_loss = -jnp.mean(jnp.take_along_axis(d_log, d_act[:, None], axis=-1)[:, 0])
            total = total + ref_beta * kl + bc_lambda * bc_loss
            return total, (actor_loss, value_loss, entropy, bc_loss, kl)

        def _epoch(carry, _):
            p_cur, opt_state, rng = carry
            rng, sub = jax.random.split(rng)
            idxs = jax.random.permutation(sub, batch).reshape(
                args.num_minibatches, batch // args.num_minibatches)

            def _minibatch(carry, ix):
                p_cur, opt_state = carry
                mb = (obs[ix], cin[ix], act[ix], logp[ix], adv[ix], ret[ix], val[ix])
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p_cur, mb)
                grads = jax.tree.map(
                    lambda g, m: jnp.where((idx < args.critic_warmup_updates) & m,
                                           jnp.zeros_like(g), g), grads, actor_mask)
                approx_kl = jnp.mean(mb[3] - jax.nn.log_softmax(
                    network.apply(p_cur, mb[0], mb[1])[0], axis=-1)[
                        jnp.arange(mb[3].shape[0]), mb[2]])

                def _apply(_):
                    upd, new_state = optimizer.update(grads, opt_state, p_cur)
                    return optax.apply_updates(p_cur, upd), new_state

                def _skip(_):
                    return p_cur, opt_state

                p_new, opt_state = jax.lax.cond(approx_kl <= args.target_kl, _apply, _skip,
                                                operand=None)
                return (p_new, opt_state), {
                    "loss": loss, "actor_loss": aux[0], "value_loss": aux[1],
                    "entropy": aux[2], "bc_loss": aux[3], "ref_kl": aux[4],
                    "approx_kl": approx_kl}

            (p_cur, opt_state), losses = jax.lax.scan(_minibatch, (p_cur, opt_state), idxs)
            return (p_cur, opt_state, rng), losses

        (params_out, opt_state, rng), losses = jax.lax.scan(
            _epoch, (params, opt_state, rng), None, args.ppo_epochs)
        return params_out, opt_state, rng, losses

    @jax.jit
    def evaluate(params, keys, cond):
        def _episode(key):
            obsv, env_state = env.reset(key)
            z = cap_vectors(cond)

            def _step(carry, t):
                obsv, env_state, live, soups, steps = carry
                flat = jnp.stack([obsv[a] for a in env.agents], axis=0).reshape(num_agents, -1)
                cin = compact_state(jax.tree.map(lambda x: x[None], env_state),
                                    crit["context"], t).reshape(num_agents, -1)
                if args.critic_sees_capability:
                    cin = jnp.concatenate(
                        [cin, jnp.broadcast_to(z[None, :], (num_agents, z_dim))], axis=-1)
                ego = jnp.concatenate([flat[0], z]) if args.arm == "oracle" else flat[0]
                logits, _ = network.apply(params, ego[None], cin[0][None])
                a_ego = jax.random.categorical(jax.random.fold_in(key, t), logits[0])
                acts = [a_ego]
                for slot, agent_idx in ((0, 1), (1, 2)):
                    picked = partner_net.apply(
                        jax.tree.map(lambda x: x[cond], pairs[slot]), flat[agent_idx][None])[0]
                    acts.append(jax.random.categorical(
                        jax.random.fold_in(key, t * 10 + agent_idx), picked))
                action = jnp.stack(acts)
                obsv, env_state, reward, done, info = env.step(
                    key, env_state, {a: action[i] for i, a in enumerate(env.agents)})
                soups = soups + jnp.where(live & env_state.new_correct_delivery, 1.0, 0.0)
                steps = steps + jnp.where(live, 1, 0)
                live = live & ~done["__all__"]
                return (obsv, env_state, live, soups, steps), None

            init = (obsv, env_state, jnp.array(True), 0.0, jnp.int32(0))
            (_, _, _, soups, steps), _ = jax.lax.scan(_step, init, jnp.arange(args.max_steps))
            return soups, steps.astype(jnp.float32)

        return jax.vmap(_episode)(keys)

    run_name = args.run_name or time.strftime(f"learned_{args.arm}_%Y%m%d_%H%M%S")
    out_dir = resolve(f"results/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": "ocv2_learning_partner_oracle_gate", "arm": args.arm,
                "pairs": args.pair, "capabilities": cond_caps,
                "critic_sees_capability": args.critic_sees_capability,
                "not_an_etm_result": True, "argv": sys.argv[1:], "segments": [],
                "wall_seconds_total": None}
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    runner = init_runner(rng)
    rng = runner[3]
    started = time.perf_counter()
    num_segments = args.updates // args.segment_updates
    for segment in range(num_segments):
        t0 = time.perf_counter()
        per_update = []
        for step in range(args.segment_updates):
            idx = segment * args.segment_updates + step
            runner, transitions, last_value = rollout(params, runner, args.shaped_coeff)
            rng = runner[3]
            advantages, returns = compute_gae(transitions, last_value)
            params, opt_state, rng, losses = update(
                params, opt_state, transitions, advantages, returns, rng, idx)
            runner = (runner[0], runner[1], runner[2], rng, runner[4])
            per_update.append({k: float(np.asarray(v).mean()) for k, v in losses.items()})
        seconds = time.perf_counter() - t0
        cumulative = (segment + 1) * args.segment_updates * num_envs * args.rollout_length
        cond_means = []
        eval_keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_episodes)
        for cond in range(num_pair):
            soups, _ = jax.device_get(evaluate(params, eval_keys, jnp.asarray(cond)))
            cond_means.append(float(np.asarray(soups).mean()))
        record = {
            "segment": segment + 1, "cumulative_env_steps": cumulative,
            "train_seconds": round(seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * num_envs * args.rollout_length / seconds, 1),
            "eval_team_soups_per_condition": cond_means,
            "entropy": float(np.mean([u["entropy"] for u in per_update])),
            "value_loss": float(np.mean([u["value_loss"] for u in per_update])),
            "bc_loss": float(np.mean([u["bc_loss"] for u in per_update])),
            "ref_kl": float(np.mean([u["ref_kl"] for u in per_update])),
            "approx_kl": float(np.mean([u["approx_kl"] for u in per_update])),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump({"params": jax.tree.map(np.asarray, params),
                         "meta": {"cumulative_env_steps": cumulative}}, handle)
        (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                          encoding="utf-8")
        print(f"[segment {segment + 1}/{num_segments}] steps={cumulative} "
              f"({record['env_steps_per_second']:.0f} steps/s) "
              f"eval soups {['%.2f' % c for c in cond_means]} "
              f"entropy {record['entropy']:.4f} value {record['value_loss']:.2f} "
              f"bc {record['bc_loss']:.3f} ref_kl {record['ref_kl']:.4f}", flush=True)

    print(f"[gate] finished {args.updates} updates in {metadata['wall_seconds_total']}s; "
          f"wrote {out_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
