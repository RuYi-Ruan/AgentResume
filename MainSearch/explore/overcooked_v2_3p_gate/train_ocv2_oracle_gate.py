"""Capability Oracle Gate on the Three-Arm Kitchen (screening only, not an ETM result).

Question: does telling the ego its teammates' *current competence* change the coordination decision
and the team's soup count? If even the true capability level does not help, a learned teammate model
cannot help on this task and the environment is excluded.

Design (follows the external guidance):
  * Ego = agent 0, the only learner. Agents 1-2 are FROZEN partners taken from checkpoints at three
    competence levels of a reference run (training progress = real competence ladder; the frozen
    checkpoints are a screening device only, the main experiment must use continuously learning
    partners).
  * Arms: `no_oracle` (actor sees only its own view) vs `oracle` (actor additionally sees per-partner
    capability one-hots). The z weights are ZERO-INITIALISED so pi(o, z=0) == pi_BC(o) and all arms
    start from exactly the same policy.
  * The critic always sees the compact privileged state AND the true capability one-hots, identical
    in both arms, so any gain must come from the actor's decision, not from a better trained critic.
  * Paired evaluation: 32 fixed initial states x 3 levels, both arms on the same states, per-episode
    paired differences with a standard error; plus policy-shift metrics (TV / JS / CRN switch rate).

Usage:
  python train_ocv2_oracle_gate.py --arm oracle --partner-run <run dir with checkpoints> \
      --checkpoints c1.pkl c2.pkl [c3.pkl] --updates 250 --run-name ocv2_oracle_oracle_seed0_1M
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

import flax.linen as nn  # noqa: E402
import jaxmarl  # noqa: E402
from flax.training.train_state import TrainState  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import (  # noqa: E402
    MLPActorCritic,
    MLPActorCriticPriv,
    compact_state,
    critic_spec,
)

class MLPActorOnly(nn.Module):
    """Actor-only mirror of MLPActorCritic (layers Dense_0..2 keep the same names/shapes).

    Frozen partners come from checkpoints of runs with different critic variants, so only the actor
    layers are portable; this module applies exactly those.
    """

    action_dim: int
    hidden: int = 256

    @nn.compact
    def __call__(self, x):
        h = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        h = nn.relu(h)
        h = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(h)
        h = nn.relu(h)
        return nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(h)


def actor_subset(params):
    return {"params": {k: v for k, v in params.items()
                       if k in ("Dense_0", "Dense_1", "Dense_2")}}


ARMS = ("no_oracle", "oracle")
NUM_LEVELS = 3  # kept for the competence-ladder variant; the archetype gate uses --pairs


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else HERE / p


def load_plain_params(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    inner = payload.get("params", payload.get("ego_params"))
    inner = jax.tree.map(jnp.asarray, inner)
    if isinstance(inner, dict) and set(inner.keys()) == {"params"}:
        return inner
    return {"params": inner}


def build_env(args):
    layout_path = resolve(args.layout_file)
    layout = Layout.from_string(layout_path.read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    return env, layout


class Args:  # minimal namespace for critic_spec
    privileged_critic = True


def make_train(config: dict, env, crit, level_params, run_updates: int, arms: str,
               init_params=None, bc_pool=None, warmup_updates=0, ref_kl=None):
    num_envs = config["num_envs"]
    num_agents = len(env.agents)
    num_actors = num_envs * num_agents
    obs_flat = config["obs_flat"]
    vocab = config["vocab"]        # number of partner archetypes in the vocabulary
    z_dim = 2 * vocab              # one-hot per partner: WHO my teammates are
    pairs = jnp.asarray(config["pairs_list"])
    ego_flat = obs_flat + (z_dim if arms == "oracle" else 0)
    critic_dim = config["critic_dim"] + z_dim
    network = MLPActorCriticPriv(action_dim=6, critic_dim=critic_dim)
    partner_net = MLPActorOnly(action_dim=6)

    actor_mask = None
    if warmup_updates > 0:
        mask_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, ego_flat)),
                                   jnp.zeros((1, critic_dim)))["params"]
        names = list(mask_params.keys())
        actor_names = set(names[: len(names) // 2])
        actor_mask = {"params": {n: {leaf: jnp.asarray(n in actor_names) for leaf in sub}
                                 for n, sub in mask_params.items()}}

    def init_runner_state(rng):
        rng, init_rng, reset_rng, level_rng = jax.random.split(rng, 4)
        params = network.init(init_rng, jnp.zeros((num_actors, ego_flat)),
                              jnp.zeros((num_actors, critic_dim)))
        if init_params is not None:
            merged = dict(params["params"])
            if arms == "oracle":
                # capability block is ZERO-initialised so pi(o, z=0) == pi_BC(o) exactly, and both
                # arms therefore start from the identical policy
                bc_kernel = jnp.asarray(init_params["Dense_0"]["kernel"])
                merged["Dense_0"] = {
                    "kernel": jnp.concatenate(
                        [bc_kernel, jnp.zeros((z_dim, bc_kernel.shape[1]))], axis=0),
                    "bias": jnp.asarray(init_params["Dense_0"]["bias"]),
                }
                for layer in ("Dense_1", "Dense_2"):
                    merged[layer] = jax.tree.map(jnp.asarray, init_params[layer])
            else:
                for name, subtree in init_params.items():
                    fresh = merged.get(name)
                    if fresh is None:
                        continue
                    if all(np.shape(fresh[leaf]) == np.shape(value)
                           for leaf, value in subtree.items()):
                        merged[name] = jax.tree.map(jnp.asarray, subtree)
            params = {"params": merged}
        # critic: zero the capability block of its first layer (identical handling in both arms)
        critic_kernel = params["params"]["Dense_3"]["kernel"]
        params["params"]["Dense_3"] = {
            "kernel": jnp.concatenate(
                [critic_kernel[:-z_dim], jnp.zeros((z_dim, critic_kernel.shape[1]))], axis=0),
            "bias": params["params"]["Dense_3"]["bias"],
        }
        layer_names = list(params["params"].keys())
        actor_layers = set(layer_names[: len(layer_names) // 2])
        labels = {"params": {n: {leaf: ("actor" if n in actor_layers else "critic")
                                 for leaf in sub} for n, sub in params["params"].items()}}
        tx = optax.multi_transform(
            {"actor": optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                                  optax.adam(config["lr_actor"], eps=1e-5)),
             "critic": optax.chain(optax.clip_by_global_norm(config["max_grad_norm"]),
                                   optax.adam(config["lr_critic"], eps=1e-5))},
            labels)
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        # sample one of the informative partner pairings, then randomise which partner sits in
        # which slot (the capability vector, not the slot, must drive the ego's decision)
        level_rng, pair_rng, swap_rng = jax.random.split(level_rng, 3)
        pair_idx = jax.random.randint(pair_rng, (num_envs,), 0, pairs.shape[0])
        swap = jax.random.randint(swap_rng, (num_envs,), 0, 2)
        first = jnp.where(swap == 0, pairs[pair_idx, 0], pairs[pair_idx, 1])
        second = jnp.where(swap == 0, pairs[pair_idx, 1], pairs[pair_idx, 0])
        levels = jnp.stack([first, second], axis=-1)
        return (train_state, env_state, obsv, jnp.zeros(num_envs, dtype=bool), levels, rng,
                jnp.zeros(3))

    def ego_input(view, levels, env_index):
        flat = view.reshape(-1)
        if arms != "oracle":
            return flat
        z = jnp.concatenate([jax.nn.one_hot(levels[env_index, i], vocab)
                             for i in range(levels.shape[1])])
        return jnp.concatenate([flat, z])

    def rollout(runner_state):
        def _step(carry, t):
            train_state, env_state, obs, done, levels, rng, stats = carry
            rng, act_rng = jax.random.split(rng)
            views = jnp.stack([obs[a] for a in env.agents], axis=1)  # (envs, agents, ...)
            ego_views = views[:, 0].reshape(num_envs, -1)
            z_all = jnp.concatenate(
                [jax.nn.one_hot(levels[:, i], vocab) for i in range(levels.shape[1])],
                axis=-1)
            actor_in = jnp.concatenate([ego_views, z_all], axis=-1) if arms == "oracle" else ego_views
            gstate = jnp.concatenate([per_env_state(env_state, crit, t), z_all], axis=-1)
            logits, value = network.apply(train_state.params, actor_in, gstate)
            action_ego = jax.random.categorical(act_rng, logits)
            log_prob = jnn_log_softmax_take(logits, action_ego)

            def partner_action(agent_idx):
                # one fused pass for all competence levels, then select: avoids per-env gathers
                all_logits = jax.vmap(jax.vmap(partner_net.apply, in_axes=(None, 0)),
                                     in_axes=(0, None))(
                    level_params, views[:, agent_idx].reshape(num_envs, -1))
                picked = jnp.take_along_axis(
                    all_logits, levels[:, agent_idx - 1][None, :, None], axis=0)[0]
                return jax.random.categorical(
                    jax.random.fold_in(act_rng, agent_idx), picked)

            acts = {env.agents[0]: action_ego,
                    env.agents[1]: partner_action(1),
                    env.agents[2]: partner_action(2)}
            new_obs, new_state, reward, done_new, info = jax.vmap(env.step)(
                jax.random.split(rng, num_envs), env_state, acts)
            last = done_new["__all__"]
            transition = {
                "obs": actor_in,
                "critic_obs": gstate,
                "action": action_ego,
                "log_prob": log_prob,
                "value": value,
                # training reward = raw delivery signal + shaped reward (fixed coefficient 1,
                # matching the official baseline's shaping term without annealing to zero)
                "reward": reward[env.agents[0]] + info["shaped_reward"][env.agents[0]],
                "done": last,
                "shaped": info["shaped_reward"][env.agents[0]],
            }
            new_carry = (train_state, new_state, new_obs, last, levels, rng, stats)
            return new_carry, transition

        runner_state, transitions = jax.lax.scan(
            _step, runner_state, jnp.arange(config["rollout_length"]))
        train_state, env_state, obs, done, levels, rng, stats = runner_state
        z_all = jnp.concatenate(
            [jax.nn.one_hot(levels[:, i], vocab) for i in range(levels.shape[1])], axis=-1)
        gstate = jnp.concatenate(
            [per_env_state(env_state, crit, config["rollout_length"]), z_all], axis=-1)
        ego_views = jnp.stack([obs[a] for a in env.agents], axis=1)[:, 0].reshape(num_envs, -1)
        actor_in = jnp.concatenate([ego_views, z_all], axis=-1) if arms == "oracle" else ego_views
        _, last_value = network.apply(train_state.params, actor_in, gstate)
        return (train_state, env_state, obs, done, levels, rng, stats), transitions, last_value

    def update(train_state, transitions, advantages, returns, rng, update_index):
        batch = transitions["reward"].size
        adv = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
        ret = returns.reshape(-1)
        obs_f = transitions["obs"].reshape(batch, -1)
        crit_f = transitions["critic_obs"].reshape(batch, -1)
        act_f = transitions["action"].reshape(batch)
        logp_f = transitions["log_prob"].reshape(batch)
        val_f = transitions["value"].reshape(batch)
        bc_lambda = 0.0
        demo_obs = demo_act = None
        if bc_pool is not None:
            frac = jnp.clip(1.0 - update_index / bc_pool["decay"], 0.0, 1.0)
            bc_lambda = bc_pool["floor"] + (bc_pool["lambda0"] - bc_pool["floor"]) * frac
            rng, demo_rng = jax.random.split(rng)
            n_side = bc_pool["batch"] // 2
            idx = jax.random.randint(demo_rng, (n_side,), 0, bc_pool["obs"].shape[0])
            demo_obs = bc_pool["obs"][idx]
            demo_act = bc_pool["act"][idx]
        ref_beta = 0.0
        if ref_kl is not None:
            ref_beta = ref_kl["floor"] + (ref_kl["beta0"] - ref_kl["floor"]) * jnp.clip(
                1.0 - update_index / ref_kl["decay"], 0.0, 1.0)

        def _loss(params, mb):
            mb_obs, mb_crit, mb_act, mb_logp, mb_adv, mb_ret, mb_val = mb
            logits, value = network.apply(params, mb_obs, mb_crit)
            logp = jax.nn.log_softmax(logits, axis=-1)
            new_logp = jnp.take_along_axis(logp, mb_act[:, None], axis=-1)[:, 0]
            ratio = jnp.exp(new_logp - mb_logp)
            actor_loss = -jnp.minimum(
                ratio * mb_adv,
                jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * mb_adv).mean()
            value_clipped = mb_val + (value - mb_val).clip(-config["clip_eps"], config["clip_eps"])
            value_loss = 0.5 * jnp.maximum(jnp.square(value - mb_ret),
                                           jnp.square(value_clipped - mb_ret)).mean()
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.mean(jnp.sum(probs * logp, axis=-1))
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
            if demo_obs is not None:
                # demonstrations carry no capability labels: z = 0
                demo_in = (jnp.concatenate([demo_obs, jnp.zeros((demo_obs.shape[0], z_dim))], -1)
                           if arms == "oracle" else demo_obs)
                d_logits, _ = network.apply(params, demo_in, jnp.zeros(
                    (demo_obs.shape[0], critic_dim)))
                d_logp = jax.nn.log_softmax(d_logits, axis=-1)
                bc_loss = -jnp.mean(jnp.take_along_axis(d_logp, demo_act[:, None], axis=-1)[:, 0])
                total = total + bc_lambda * bc_loss
            if ref_kl is not None:
                r_logits, _ = ref_kl["module"].apply(
                    {"params": ref_kl["params"]}, mb_obs[:, :config["obs_flat"]])
                p_ref = jax.nn.softmax(r_logits, axis=-1)
                kl = jnp.mean(jnp.sum(p_ref * (jnp.log(p_ref + 1e-9) - logp), axis=-1))
                total = total + ref_beta * kl
            return total, (actor_loss, value_loss, entropy)

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            idxs = jax.random.permutation(perm_rng, batch).reshape(
                config["num_minibatches"], batch // config["num_minibatches"])

            def _minibatch(carry, idx):
                train_state, _ = carry
                mb = (obs_f[idx], crit_f[idx], act_f[idx], logp_f[idx], adv[idx], ret[idx],
                      val_f[idx])
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb)
                if actor_mask is not None:
                    grads = jax.tree.map(
                        lambda g, m: jnp.where(
                            (update_index < warmup_updates) & m, jnp.zeros_like(g), g),
                        grads, actor_mask)
                return (train_state.apply_gradients(grads=grads), None), {
                    "total": loss, "actor": aux[0], "value": aux[1], "entropy": aux[2]}

            (train_state, _), losses = jax.lax.scan(_minibatch, (train_state, None), idxs)
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            _epoch, (train_state, rng), None, config["ppo_epochs"])
        residual = transitions["value"].reshape(-1) - returns.reshape(-1)
        ev = 1.0 - residual.var() / (returns.reshape(-1).var() + 1e-8)
        losses["explained_var"] = jnp.broadcast_to(ev, losses["total"].shape)
        return train_state, rng, losses

    def compute_gae(transitions, last_value):
        def _scan(carry, data):
            gae, next_value = carry
            delta = data["reward"] + config["gamma"] * next_value * (1 - data["done"]) - data["value"]
            gae = delta + config["gamma"] * config["gae_lambda"] * (1 - data["done"]) * gae
            return (gae, data["value"]), gae

        _, advantages = jax.lax.scan(_scan, (jnp.zeros_like(last_value), last_value),
                                     transitions, reverse=True)
        return advantages, advantages + transitions["value"]

    def train(rng, runner_state=None, start_step=0):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, step):
            idx = start_step + step
            runner_state, transitions, last_value = rollout(runner_state)
            advantages, returns = compute_gae(transitions, last_value)
            train_state, rng, losses = update(runner_state[0], transitions, advantages, returns,
                                              runner_state[5], idx)
            metric = {
                "env_step": (idx + 1) * num_envs * config["rollout_length"],
                "train_reward": transitions["reward"].mean(),
                "train_shaped": transitions["shaped"].mean(),
                "entropy": losses["entropy"].mean(),
                "explained_var": losses["explained_var"].mean(),
            }
            new_state = (train_state, runner_state[1], runner_state[2], runner_state[3],
                         runner_state[4], rng, runner_state[6])
            return new_state, metric

        runner_state, metrics = jax.lax.scan(_update_step, runner_state,
                                             jnp.arange(run_updates))
        return {"runner_state": runner_state, "metrics": metrics}

    return train, network


def per_env_state(env_state, crit, t):
    """Compact privileged state for one representative agent per env: (num_envs, dim)."""
    full = compact_state(env_state, crit["context"], t)
    num_agents = env_state.grid.shape[0]
    return full.reshape(num_agents, -1, full.shape[-1])[:, 0]


def jnn_log_softmax_take(logits, action):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, action[:, None], axis=-1)[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--partner-checkpoints", nargs="+", required=True,
                        help="one checkpoint per partner archetype in the vocabulary")
    parser.add_argument("--pairs", default="0,1",
                        help="semicolon-separated archetype index pairs, e.g. '0,3;1,2'")
    parser.add_argument("--bc-init", default=None)
    parser.add_argument("--bc-demos", default="demos/three_arm_demos.npz")
    parser.add_argument("--bc-lambda0", type=float, default=1.0)
    parser.add_argument("--bc-lambda-floor", type=float, default=0.1)
    parser.add_argument("--bc-decay-updates", type=int, default=400)
    parser.add_argument("--ref-kl-beta", type=float, default=1.0)
    parser.add_argument("--ref-kl-floor", type=float, default=0.1)
    parser.add_argument("--ref-kl-decay-updates", type=int, default=400)
    parser.add_argument("--critic-warmup-updates", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--segment-updates", type=int, default=25)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr-actor", type=float, default=2.5e-4)
    parser.add_argument("--lr-critic", type=float, default=2.5e-4)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    env, layout = build_env(args)
    crit = critic_spec(Args(), env)
    crit["context"] = crit
    pair_list = [[int(i) for i in spec.split(",")] for spec in args.pairs.split(";")]
    vocab = len(args.partner_checkpoints)
    ego_flat = int(np.prod(env.observation_space().shape)) + (2 * vocab
                                                             if args.arm == "oracle" else 0)
    config = {
        "num_envs": args.num_envs, "rollout_length": args.rollout_length,
        "obs_flat": int(np.prod(env.observation_space().shape)), "obs_shape":
            tuple(env.observation_space().shape),
        "critic_dim": crit["dim"], "gamma": 0.99, "gae_lambda": 0.95,
        "clip_eps": args.clip_eps, "ent_coef": args.ent_coef, "vf_coef": 0.5,
        "max_grad_norm": 0.5, "lr_actor": args.lr_actor, "lr_critic": args.lr_critic,
        "ppo_epochs": args.ppo_epochs, "num_minibatches": args.num_minibatches,
        "seed": args.seed, "ego_flat": ego_flat, "vocab": vocab,
        "pairs_list": pair_list,
    }

    level_paths = [resolve(p) for p in args.partner_checkpoints]
    per_level = [actor_subset(load_plain_params(p)["params"]) for p in level_paths]
    level_params = jax.tree.map(lambda *xs: jnp.stack(xs), *per_level)
    print(f"[gate] partner levels: {[p.name for p in level_paths]}")

    init_params = load_plain_params(resolve(args.bc_init))["params"] if args.bc_init else None
    demo_path = resolve(args.bc_demos)
    data = np.load(demo_path)
    obs_raw = data["obs"].reshape(-1, int(np.prod(data["obs"].shape[2:])))
    act_raw = data["action"].reshape(-1).astype(np.int32)
    rng_pool = np.random.default_rng(args.seed)
    idx = rng_pool.permutation(len(obs_raw))[:10000]
    bc_pool = {"obs": jnp.asarray(obs_raw[idx].astype(np.float32)),
               "act": jnp.asarray(act_raw[idx]),
               "lambda0": args.bc_lambda0, "floor": args.bc_lambda_floor,
               "decay": float(args.bc_decay_updates), "batch": 512}
    ref_kl = None
    if args.ref_kl_beta > 0 and init_params is not None:
        ref_kl = {"params": init_params, "module": MLPActorCritic(action_dim=6),
                  "beta0": args.ref_kl_beta, "floor": args.ref_kl_floor,
                  "decay": float(args.ref_kl_decay_updates)}

    train, network = make_train(config, env, crit, level_params, args.segment_updates, args.arm,
                                init_params=init_params, bc_pool=bc_pool,
                                warmup_updates=args.critic_warmup_updates, ref_kl=ref_kl)
    train_jit = jax.jit(train)

    run_name = args.run_name or time.strftime(f"ocv2_oracle_{args.arm}_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"kind": "ocv2_capability_oracle_gate", "arm": args.arm,
                "frozen_partner_checkpoints_screening_only": True,
                "partner_checkpoints": [str(p) for p in level_paths],
                "not_an_etm_result": True, "argv": sys.argv[1:],
                "config": {k: v for k, v in config.items() if k != "obs_shape"},
                "obs_shape": list(config["obs_shape"]), "segments": [],
                "wall_seconds_total": None}
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rng = jax.random.PRNGKey(args.seed)
    runner_state = None
    started = time.perf_counter()
    num_segments = args.updates // args.segment_updates
    for segment in range(num_segments):
        t0 = time.perf_counter()
        result = (train_jit(rng, None, 0) if runner_state is None
                  else train_jit(rng, runner_state, segment * args.segment_updates))
        jax.block_until_ready(result["metrics"]["env_step"])
        seconds = time.perf_counter() - t0
        runner_state = result["runner_state"]
        metrics = {k: np.asarray(v) for k, v in jax.device_get(result["metrics"]).items()}
        params = runner_state[0].params
        cumulative = (segment + 1) * args.segment_updates * args.num_envs * args.rollout_length
        record = {
            "segment": segment + 1, "cumulative_env_steps": cumulative,
            "train_seconds": round(seconds, 2),
            "env_steps_per_second": round(
                args.segment_updates * args.num_envs * args.rollout_length / seconds, 1),
            "train_reward": float(metrics["train_reward"].mean()),
            "train_shaped": float(metrics["train_shaped"].mean()),
            "entropy": float(metrics["entropy"].mean()),
            "explained_var": float(metrics["explained_var"].mean()),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        with (out_dir / f"checkpoint_{cumulative:010d}.pkl").open("wb") as handle:
            pickle.dump({"params": jax.tree.map(np.asarray, params),
                         "meta": {"cumulative_env_steps": cumulative}},
                        handle)
        (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"[segment {segment + 1}/{num_segments}] steps={cumulative} "
              f"train={seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
              f"reward={record['train_reward']:.3f} shaped={record['train_shaped']:.4f} "
              f"EV={record['explained_var']:.3f}", flush=True)

    print(f"[gate] finished {args.updates} updates in {metadata['wall_seconds_total']}s; "
          f"wrote {out_dir / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
