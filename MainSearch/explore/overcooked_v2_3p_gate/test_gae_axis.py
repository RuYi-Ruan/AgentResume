"""Synthetic GAE unit test: does the (T, E, A) implementation match a hand-written double loop?

Requested by the external guidance after the independent-actor critic kept diverging with value
normalisation switched off: verify per-agent GAE/return semantics, including the "leakage" invariant
that changing one agent's rewards must not change another agent's advantages.

Usage: python test_gae_axis.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

T, E, A = 4, 2, 3
GAMMA, LAM = 0.99, 0.95


def reference(r, v, done, last_v):
    """Plain double-loop GAE (ground truth)."""
    adv = np.zeros_like(r)
    ret = np.zeros_like(r)
    next_v = last_v.copy()
    next_a = np.zeros_like(last_v)
    for t in range(T - 1, -1, -1):
        delta = r[t] + GAMMA * (1 - done[t]) * next_v - v[t]
        next_a = delta + GAMMA * LAM * (1 - done[t]) * next_a
        adv[t] = next_a
        ret[t] = next_a + v[t]
        next_v = v[t]
    return adv, ret


def jax_impl(r, v, done, last_v):
    """Mirrors train_ocv2_independent.compute_gae (leading axis = time)."""
    transitions = {"reward": r, "value": v, "done": done}

    def _scan(carry, data):
        gae, next_value = carry
        delta = data["reward"] + GAMMA * next_value * (1 - data["done"]) - data["value"]
        gae = delta + GAMMA * LAM * (1 - data["done"]) * gae
        return (gae, data["value"]), gae

    _, advantages = jax.lax.scan(
        _scan, (jnp.zeros_like(last_v), last_v), transitions, reverse=True)
    return np.asarray(advantages), np.asarray(advantages + transitions["value"])


def main() -> None:
    rng = np.random.default_rng(0)
    # strongly agent-specific scales: a detector for agent-axis mix-ups
    scale = np.array([1.0, 10.0, 100.0]).reshape(1, 1, A)
    r = (rng.normal(size=(T, E, A)) * scale).astype(np.float32)
    v = (rng.normal(size=(T, E, A)) * scale * 0.5).astype(np.float32)
    done = np.zeros((T, E, A), dtype=np.float32)
    done[2, 0, :] = 1.0                     # env 0 terminates at t=2 for all agents
    done[3, 1, 1] = 1.0                     # env 1, agent 1 terminates at the last step only
    last_v = (rng.normal(size=(E, A)) * scale.reshape(1, A)).astype(np.float32)

    adv_ref, ret_ref = reference(r, v, done, last_v)
    adv_jax, ret_jax = jax_impl(jnp.asarray(r), jnp.asarray(v), jnp.asarray(done),
                                jnp.asarray(last_v))
    print("shape invariants:")
    print(f"  reward {r.shape} value {v.shape} done {done.shape} last_value {last_v.shape}")
    print(f"  advantage {adv_jax.shape} return {ret_jax.shape}")
    ok_adv = np.allclose(adv_ref, adv_jax, atol=1e-4)
    ok_ret = np.allclose(ret_ref, ret_jax, atol=1e-4)
    print(f"GAE matches double loop: {ok_adv} (max diff "
          f"{np.abs(adv_ref - adv_jax).max():.2e})")
    print(f"return matches double loop: {ok_ret} (max diff "
          f"{np.abs(ret_ref - ret_jax).max():.2e})")

    # leakage test: zero agent 0's rewards -> agents 1 and 2 must be untouched
    r2 = r.copy()
    r2[:, :, 0] = 0.0
    adv2, _ = jax_impl(jnp.asarray(r2), jnp.asarray(v), jnp.asarray(done), jnp.asarray(last_v))
    leak = np.abs(adv2[:, :, 1:] - adv_jax[:, :, 1:]).max()
    print(f"leakage test (zero agent 0 reward -> agents 1,2 advantages): max change {leak:.2e} "
          f"({'OK' if leak < 1e-5 else 'LEAK'})")

    # axis-permutation detector: swap agent and env axes and check the result changes
    adv_perm, _ = jax_impl(jnp.asarray(np.transpose(r, (0, 2, 1))), 
                           jnp.asarray(np.transpose(v, (0, 2, 1))),
                           jnp.asarray(np.transpose(done, (0, 2, 1))),
                           jnp.asarray(last_v.T))
    print(f"sanity: permuting env/agent axes changes the result: "
          f"{not np.allclose(adv_perm, adv_jax, atol=1e-4)}")

    print("\nPer-agent advantage table (env 0, agent 0 vs agent 2 differ by ~100x):")
    for agent in range(A):
        print(f"  agent {agent}: adv{t_} = {np.round(adv_jax[:, 0, agent], 2)}")
    del t_
    print("\nRESULT:", "PASS" if (ok_adv and ok_ret and leak < 1e-5) else "FAIL")


t_ = 0

if __name__ == "__main__":
    main()
