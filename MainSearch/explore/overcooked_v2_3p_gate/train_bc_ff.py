"""Behaviour-clone the scripted three-arm experts into the FF policy used by PPO.

Follows the external guidance:
  * Balanced batch: 50% uniform transitions + 50% critical transitions (interact steps, +-3 steps
    around them, shaping and delivery steps), because the demo data is dominated by walking and a
    plain CE loss would never learn INTERACT.
  * Report macro accuracy and INTERACT recall, not just overall accuracy.
  * Episode-level 80/10/10 split (no transition leakage between splits).
  * The saved parameters have exactly the MLPActorCritic layout of train_overcooked_ff.py, so the
    PPO trainer can start from the BC checkpoint without any conversion.

Also runs the aliasing diagnostic the guidance asked for: how often the same ego view maps to
different expert actions (H(A|O) style conflict rate), which decides FF vs recurrent.

Usage:
  python train_bc_ff.py --demos demos/three_arm_demos.npz --out bc/three_arm_bc_seed0.pkl
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
import optax

HERE = Path(__file__).resolve().parent
FF_GATE = HERE.parent / "overcooked_v2_ff_gate"
sys.path.insert(0, str(FF_GATE))

from train_overcooked_ff import MLPActorCritic  # noqa: E402

INTERACT = 5
ACTION_NAMES = ["right", "down", "left", "up", "stay", "interact"]


def build_samples(data):
    obs = data["obs"]                       # (T, 3, 7, 7, 35) int8
    action = data["action"]                 # (T, 3) int8
    flags = data["flags"]                   # (T, 3 + 2) int8
    episode = data["episode"]               # (T,)
    n_agents = obs.shape[1]
    obs_r = obs.reshape(obs.shape[0] * n_agents, -1).astype(np.float32)  # flattened view
    action_r = action.reshape(-1).astype(np.int64)
    agent_idx = np.tile(np.arange(n_agents), obs.shape[0])
    episode_r = np.repeat(episode, n_agents)

    interact = flags[:, :n_agents].astype(bool)
    event = (flags[:, n_agents] | flags[:, n_agents + 1]).astype(bool)
    critical = np.zeros_like(interact)
    for offset in range(-3, 4):
        shifted = np.roll(interact, offset, axis=0)
        if offset > 0:
            shifted[:offset] = False
        elif offset < 0:
            shifted[offset:] = False
        critical |= shifted
    critical |= np.repeat(event[:, None], n_agents, axis=1)
    critical_r = critical.reshape(-1)
    if "mask" in data:
        # role-specialised demonstration sets record only the target role's agents
        keep = data["mask"].reshape(-1).astype(bool)
        obs_r, action_r, critical_r, episode_r = (
            obs_r[keep], action_r[keep], critical_r[keep], episode_r[keep])
    return obs_r, action_r, critical_r, episode_r


def aliasing_diagnostic(obs_r, action_r):
    """Conflict rate of identical ego views carrying different expert actions."""
    keys = np.ascontiguousarray(obs_r.astype(np.int8).reshape(len(obs_r), -1))
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    action_onehot = np.zeros((len(action_r), 6), dtype=np.int32)
    action_onehot[np.arange(len(action_r)), action_r] = 1
    per_group_actions = np.zeros((len(counts), 6), dtype=np.int32)
    np.add.at(per_group_actions, inverse, action_onehot)
    probs = per_group_actions / np.maximum(counts[:, None], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0), axis=1)
    multi = per_group_actions.sum(axis=1) > 1
    repeat_weight = counts[multi].sum() / max(1, counts.sum())
    dominant = per_group_actions.max(axis=1) / np.maximum(counts, 1)
    return {
        "unique_views": int(len(counts)),
        "mean_conditional_entropy_nats": float(np.average(entropy, weights=counts)),
        "fraction_of_samples_with_conflicting_actions": float(repeat_weight),
        "mean_dominant_action_probability": float(np.average(dominant, weights=counts)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="demos/three_arm_demos.npz")
    parser.add_argument("--out", default="bc/three_arm_bc_seed0.pkl")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--critical-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overfit-check", action="store_true",
                        help="sanity check: can the net overfit 1000 transitions?")
    args = parser.parse_args()

    demo_path = Path(args.demos)
    if not demo_path.is_absolute():
        demo_path = HERE / demo_path
    data = np.load(demo_path)
    obs_r, action_r, critical_r, episode_r = build_samples(data)
    print(f"[bc] {len(obs_r)} transitions, {int(critical_r.sum())} critical "
          f"({critical_r.mean() * 100:.1f}%)")

    aliasing = aliasing_diagnostic(obs_r, action_r)
    print("[bc] aliasing diagnostic:", json.dumps(aliasing, indent=2))

    episodes = np.unique(episode_r)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(episodes)
    n_train = int(0.8 * len(episodes))
    n_val = int(0.1 * len(episodes))
    splits = {
        "train": set(episodes[:n_train].tolist()),
        "val": set(episodes[n_train:n_train + n_val].tolist()),
        "test": set(episodes[n_train + n_val:].tolist()),
    }
    masks = {name: np.isin(episode_r, list(eps)) for name, eps in splits.items()}

    module = MLPActorCritic(action_dim=6, hidden=args.hidden)
    params = module.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, obs_r.shape[1])))["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(args.lr))
    opt_state = optimizer.init(params)

    def loss_fn(params, batch_obs, batch_act):
        logits, _ = module.apply({"params": params}, batch_obs)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(log_probs[jnp.arange(batch_act.shape[0]), batch_act]), logits

    @jax.jit
    def train_step(params, opt_state, batch_obs, batch_act):
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch_obs, batch_act)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, logits

    def evaluate(params, mask):
        obs_m, act_m = obs_r[mask], action_r[mask]
        preds = np.empty(len(act_m), dtype=np.int64)
        for start in range(0, len(act_m), 4096):
            logits, _ = module.apply({"params": params}, jnp.asarray(obs_m[start:start + 4096]))
            preds[start:start + 4096] = np.asarray(jnp.argmax(logits, axis=-1))
        overall = float((preds == act_m).mean())
        recalls = []
        for cls in range(6):
            sel = act_m == cls
            if sel.sum():
                recalls.append(float((preds[sel] == cls).mean()))
        return {
            "overall_accuracy": overall,
            "macro_accuracy": float(np.mean(recalls)),
            "interact_recall": recalls[INTERACT] if len(recalls) > INTERACT else 0.0,
            "per_class_recall": {ACTION_NAMES[i]: r for i, r in enumerate(recalls)},
            "n": int(len(act_m)),
        }

    train_mask = masks["train"]
    obs_tr, act_tr = obs_r[train_mask], action_r[train_mask]
    crit_tr = critical_r[train_mask]
    obs_va, act_va = obs_r[masks["val"]], action_r[masks["val"]]
    if args.overfit_check:
        obs_tr, act_tr, crit_tr = obs_r[:1000], action_r[:1000], critical_r[:1000]
        obs_va, act_va = obs_r[:1000], action_r[:1000]

    idx_uniform_pool = np.arange(len(crit_tr))
    idx_critical_pool = np.flatnonzero(crit_tr)
    print(f"[bc] pool sizes: uniform {len(idx_uniform_pool)}, critical {len(idx_critical_pool)}")
    best = {"val_loss": float("inf"), "epoch": -1}
    history = []
    for epoch in range(args.epochs):
        order_uniform = rng.permutation(idx_uniform_pool)
        order_critical = rng.permutation(idx_critical_pool)
        n_batches = max(1, len(crit_tr) // args.batch)
        epoch_loss = 0.0
        for batch_index in range(n_batches):
            n_crit_batch = max(1, int(args.critical_fraction * args.batch))
            n_uniform_batch = args.batch - n_crit_batch
            take_u = order_uniform[(batch_index * n_uniform_batch) % len(order_uniform):
                                   (batch_index * n_uniform_batch) % len(order_uniform) + n_uniform_batch]
            take_c = order_critical[(batch_index * n_crit_batch) % len(order_critical):
                                    (batch_index * n_crit_batch) % len(order_critical) + n_crit_batch]
            idx = np.concatenate([take_u, take_c])
            if len(idx) < 2:
                continue
            params, opt_state, loss, _ = train_step(
                params, opt_state, jnp.asarray(obs_tr[idx]), jnp.asarray(act_tr[idx]))
            epoch_loss += float(loss)
        val_losses = []
        for start in range(0, len(obs_va), 4096):
            _, logits = loss_fn(params, jnp.asarray(obs_va[start:start + 4096]),
                                jnp.asarray(act_va[start:start + 4096]))
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            val_losses.append(float(-jnp.mean(
                log_probs[jnp.arange(logits.shape[0]), jnp.asarray(act_va[start:start + 4096])])))
        val_loss = float(np.mean(val_losses))
        train_metrics = evaluate(params, train_mask)
        val_metrics = evaluate(params, masks["val"])
        history.append({"epoch": epoch, "train_loss": epoch_loss / max(1, n_batches),
                        "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()
                                                 if not isinstance(v, dict)}})
        print(f"[bc] epoch {epoch:3d} train_loss={epoch_loss / max(1, n_batches):.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_metrics['overall_accuracy']:.4f} "
              f"val_macro={val_metrics['macro_accuracy']:.4f} "
              f"interact_recall={val_metrics['interact_recall']:.4f}")
        if val_loss < best["val_loss"] - 1e-4:
            best = {"val_loss": val_loss, "epoch": epoch, "params": params}
        elif epoch - best["epoch"] > args.patience:
            print(f"[bc] early stop at epoch {epoch} (best epoch {best['epoch']})")
            break

    params = best["params"]
    test_metrics = evaluate(params, masks["test"])
    train_metrics = evaluate(params, train_mask)
    report = {
        "demos": str(demo_path),
        "transitions": int(len(obs_r)),
        "critical_fraction": float(critical_r.mean()),
        "aliasing": aliasing,
        "best_epoch": int(best["epoch"]),
        "train": train_metrics,
        "test": test_metrics,
        "history": history,
        "hyperparameters": vars(args),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump({"params": jax.tree.map(np.asarray, params),
                     "meta": {"source": "bc", "demos": str(demo_path)}}, handle)
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("[bc] test:", json.dumps(test_metrics, indent=2))
    print(f"[bc] wrote {out}")


if __name__ == "__main__":
    main()
