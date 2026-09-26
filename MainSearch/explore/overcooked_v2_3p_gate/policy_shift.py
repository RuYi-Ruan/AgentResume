"""Policy-shift metrics between two checkpoints on matched states (external guidance).

Why not raw argmax disagreement: a policy can be fine under sampling yet have an argmax that is a
"Stay self-loop" at the initial observation. The defensible readings are:

  * TV(o) = 0.5 * sum_a |pi_1(a|o) - pi_2(a|o)|   (Total Variation distance)
  * JS(o)                                          (Jensen-Shannon divergence, nats)
  * common-random-numbers switch rate: with ONE shared Gumbel vector g,
        a_1 = argmax_a [log pi_1(a) + g_a],  a_2 = argmax_a [log pi_2(a) + g_a]
    then P(a_1 != a_2). Independent sampling would inflate this even for identical policies.

States are drawn from a *reference rollout* (the two policies' own greedy rollouts often never leave
the start state), so the comparison is on the state distribution a competent policy visits.

Usage:
  python policy_shift.py --policy-a bc/three_arm_bc_seed0.pkl --policy-b results/.../checkpoint_*.pkl \
      --reference a --episodes 32 --out results/policy_shift.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "overcooked_v2_ff_gate"))
sys.path.insert(0, str(HERE))

import jaxmarl  # noqa: E402
from jaxmarl.environments.overcooked_v2.layouts import Layout  # noqa: E402
from train_overcooked_ff import MLPActorCritic  # noqa: E402


def load_params(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    inner = payload.get("params", payload.get("ego_params"))
    inner = jax.tree.map(jnp.asarray, inner)
    if isinstance(inner, dict) and set(inner.keys()) == {"params"}:
        return inner  # checkpoints from the PPO trainer already carry the params wrapper
    return {"params": inner}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-a", required=True)
    parser.add_argument("--policy-b", required=True)
    parser.add_argument("--reference", choices=["a", "b"], default="a",
                        help="which policy generates the reference state distribution")
    parser.add_argument("--layout-file", default="layouts/three_arm_v2.txt")
    parser.add_argument("--recipes", default="[[0,0,0]]")
    parser.add_argument("--agent-view-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-stride", type=int, default=4,
                        help="keep every Nth step of the reference rollout")
    parser.add_argument("--out", default="results/policy_shift.json")
    args = parser.parse_args()

    layout_path = Path(args.layout_file)
    if not layout_path.is_absolute():
        layout_path = HERE / layout_path
    layout = Layout.from_string(layout_path.read_text(encoding="utf-8"),
                                possible_recipes=json.loads(args.recipes))
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=args.agent_view_size,
                       negative_rewards=True, sample_recipe_on_delivery=True,
                       random_agent_positions=False, max_steps=args.max_steps)
    num_agents = len(env.agents)

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else HERE / path

    params_a = load_params(resolve(args.policy_a))
    params_b = load_params(resolve(args.policy_b))
    ref_params = params_a if args.reference == "a" else params_b
    net = MLPActorCritic(action_dim=6, hidden=args.hidden)

    def logits_of(params, obs_flat):
        logits, _ = net.apply(params, obs_flat.astype(jnp.float32))
        return logits

    @jax.jit
    def collect(params, key):
        """One sampled episode; returns flattened observations and step mask."""
        obs, env_state = env.reset(key)

        def _step(carry, t):
            obs, env_state, live = carry
            flat = jnp.stack([obs[a] for a in env.agents]).reshape(num_agents, -1)
            logits = logits_of(params, flat)
            actions = jax.vmap(lambda k, logit: jax.random.categorical(k, logit))(
                jax.random.split(jax.random.fold_in(key, t), num_agents), logits)
            obs, env_state, reward, done, info = env.step(
                key, env_state, {a: actions[i] for i, a in enumerate(env.agents)})
            live = live & ~done["__all__"]
            return (obs, env_state, live), (flat, live)

        init = (obs, env_state, jnp.array(True))
        (obs, env_state, live), (states, mask) = jax.lax.scan(
            _step, init, jnp.arange(args.max_steps))
        return states, mask

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    states, mask = jax.vmap(collect, in_axes=(None, 0))(ref_params, keys)
    states = np.asarray(states)[:, ::args.state_stride]          # (episodes, steps, agents, dim)
    mask = np.asarray(mask)[:, ::args.state_stride]
    kept = states[mask]                                          # (N, agents, dim)
    print(f"[shift] reference states: {kept.shape[0]} steps x {kept.shape[1]} agents "
          f"(from {args.episodes} episodes, reference={args.reference})")

    flat_states = jnp.asarray(kept.reshape(-1, kept.shape[-1]))
    logits_a = np.asarray(logits_of(params_a, flat_states))
    logits_b = np.asarray(logits_of(params_b, flat_states))
    p_a = np.exp(logits_a - logits_a.max(axis=-1, keepdims=True))
    p_a /= p_a.sum(axis=-1, keepdims=True)
    p_b = np.exp(logits_b - logits_b.max(axis=-1, keepdims=True))
    p_b /= p_b.sum(axis=-1, keepdims=True)

    tv = 0.5 * np.abs(p_a - p_b).sum(axis=-1)
    mix = 0.5 * (p_a + p_b)
    def _kl(p, q):
        return np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-12) / np.maximum(q, 1e-12)), 0.0),
                      axis=-1)
    js = 0.5 * _kl(p_a, mix) + 0.5 * _kl(p_b, mix)

    rng = np.random.default_rng(args.seed)
    gumbel = rng.gumbel(size=logits_a.shape).astype(np.float32)
    with np.errstate(divide="ignore"):
        g_a = np.log(np.maximum(p_a, 1e-12)) + gumbel
        g_b = np.log(np.maximum(p_b, 1e-12)) + gumbel
    switch = (g_a.argmax(axis=-1) != g_b.argmax(axis=-1))
    greedy_diff = (logits_a.argmax(axis=-1) != logits_b.argmax(axis=-1))

    # switch rate conditioned on the action the reference-sampled policy took
    from_a = np.zeros(6)
    for action in range(6):
        sel = g_a.argmax(axis=-1) == action
        if sel.any():
            from_a[action] = float(switch[sel].mean())

    report = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "reference": args.reference,
        "matched_states": int(flat_states.shape[0]),
        "tv_mean": float(tv.mean()),
        "tv_p90": float(np.percentile(tv, 90)),
        "js_mean_nats": float(js.mean()),
        "crn_action_switch_rate": float(switch.mean()),
        "greedy_action_difference_rate": float(greedy_diff.mean()),
        "switch_rate_by_a_action": from_a.tolist(),
        "episodes": args.episodes,
        "state_stride": args.state_stride,
    }
    print(json.dumps({k: v for k, v in report.items()
                      if k != "switch_rate_by_a_action"}, indent=2))
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[shift] wrote {out}")


if __name__ == "__main__":
    main()
