"""ETM information gate on RWARE: does an observer's dynamic model of a partner beat stale or
frame-only alternatives at predicting that partner's next action?

Data: trajectories collected by `collect_trajectories.py` at several training stages (the
partner's policy is different, and demonstrably changing, at each stage). Features are exactly
what an observer sees at execution time; labels are the partner's actions.

Variants compared on the SAME held-out late-stage episodes and the same architecture family:

  * `current`     : observer's own view at time t (single frame), trained on late-stage data.
  * `history`     : observer's last K views (equal-capacity alternative to "memory"), trained on
                    late-stage data.
  * `frozen_etm`  : same input as `current`, but trained ONLY on the earliest stage's data
                    (stale model of the partner, as in "we learned the partner once").
  * `online_etm`  : same input as `current`, trained on all stages up to the test stage
                    (partner evidence keeps being incorporated) = the dynamic-cognition arm.

Reported: accuracy (and macro-F1) on late-stage held-out episodes, for all steps and for steps
where the partner is geometrically visible to the observer. Diagnostics only — this is an
information gate, not an ETM closed-loop result.

Usage:
  python etm_info_gate.py --data data/rware_indep_stages.npz --history 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

HERE = Path(__file__).resolve().parent
NUM_AGENTS = 4
ACTION_DIM = 5
SENSOR_RANGE = 1


class ActionPredictor(nn.Module):
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        x = nn.relu(x)
        return nn.Dense(ACTION_DIM, kernel_init=nn.initializers.orthogonal(0.01))(x)


def train_model(x_train, y_train, x_val, y_val, seed=0, steps=2000, batch=512, lr=1e-3):
    network = ActionPredictor()
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    params = network.init(init_rng, jnp.zeros((1, x_train.shape[-1])))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    def loss_fn(params, xb, yb):
        logits = network.apply(params, xb)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean()
        acc = (jnp.argmax(logits, axis=-1) == yb).mean()
        return loss, acc

    @jax.jit
    def step(state, rng):
        rng, idx_rng = jax.random.split(rng)
        idx = jax.random.randint(idx_rng, (batch,), 0, x_train.shape[0])
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, x_train[idx], y_train[idx]
        )
        return state.apply_gradients(grads=grads), loss, acc

    def evaluate(state, x, y):
        logits = network.apply(state.params, x)
        pred = jnp.argmax(logits, axis=-1)
        accuracy = float((pred == y).mean())
        f1s = []
        for cls in range(ACTION_DIM):
            tp = float(((pred == cls) & (y == cls)).sum())
            fp = float(((pred == cls) & (y != cls)).sum())
            fn = float(((pred != cls) & (y == cls)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return accuracy, float(np.mean(f1s))

    for i in range(steps):
        state, loss, acc = step(state, rng)
        rng, _ = jax.random.split(rng)
        if (i + 1) % 500 == 0:
            val_acc, val_f1 = evaluate(state, x_val, y_val)
            print(f"    step {i + 1}: train loss {float(loss):.3f} train acc {float(acc):.3f} "
                  f"val acc {val_acc:.3f} macroF1 {val_f1:.3f}", flush=True)
    return state, evaluate


def to_xy(features, labels, valid):
    return features[valid], labels[valid]


def pair_dataset(features, actions, live, positions, observer, partner, history,
                 require_visible: bool = False):
    """features: (T, E, A, F); actions: (T, E, A); live: (T, E); positions: (T, E, A, 2).

    `live` is per episode (episode termination is shared by all agents), so it is used as-is.
    With `require_visible`, only steps where the partner is inside the observer's sensor window
    are kept as valid prediction targets (otherwise ~91% of steps are unobservable, which makes
    a constant class prior the strongest predictor).
    """
    t_steps = live.shape[0]
    partner_action = actions[:, :, partner]
    valid = live & (np.arange(t_steps)[:, None] > history - 1)
    visible = (
        (np.abs(positions[:, :, observer, 0] - positions[:, :, partner, 0]) <= SENSOR_RANGE)
        & (np.abs(positions[:, :, observer, 1] - positions[:, :, partner, 1]) <= SENSOR_RANGE)
    )
    if require_visible:
        valid = valid & visible
    frames = []
    for k in range(history - 1, -1, -1):
        view = features[:, :, observer, :]
        if k == 0:
            frames.append(view)
        else:
            pad = np.zeros((k, view.shape[1], view.shape[2]), dtype=view.dtype)
            frames.append(np.concatenate([pad, view[:-k]], axis=0))
    history_features = np.concatenate(frames, axis=-1)
    current_features = features[:, :, observer, :]
    return {
        "label": partner_action,
        "valid": valid,
        "visible": visible,
        "current": current_features,
        "history": history_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--only-visible", action="store_true",
                        help="keep only steps where the partner is inside the observer view")
    parser.add_argument("--out", default=None)
    argv = parser.parse_args()

    data_path = Path(argv.data)
    if not data_path.is_absolute():
        data_path = HERE / data_path
    payload = np.load(data_path, allow_pickle=True)
    stages = json.loads(str(payload["stages"]))

    def time_first(x):
        """Collector stores (stage, episode, step, ...); models want (stage, step, episode, ...)."""
        x = np.asarray(x)
        if x.ndim < 3:
            return x
        return np.transpose(x, (0, 2, 1) + tuple(range(3, x.ndim)))

    features = time_first(payload["features"])  # (S, T, E, A, F)
    actions = time_first(payload["action"]).astype(np.int64)
    live = time_first(payload["live"]).astype(bool)
    positions = time_first(payload["position"])
    num_stages = features.shape[0]
    episodes = features.shape[2]
    train_epi = slice(0, argv.train_episodes)
    test_epi = slice(argv.train_episodes, episodes)
    print(f"data: {data_path.name} | stages={num_stages} | features={features.shape} | "
          f"train episodes {argv.train_episodes}, test episodes {episodes - argv.train_episodes}")
    print(f"stage steps: {[s['cumulative_env_steps'] for s in stages]}")

    def pack(observer, partner, stage, episode_slice, history):
        ds = pair_dataset(
            features[stage], actions[stage], live[stage], positions[stage],
            observer, partner, history, require_visible=argv.only_visible,
        )
        valid = ds["valid"][:, episode_slice]
        labels = ds["label"][:, episode_slice]
        current, y = to_xy(ds["current"][:, episode_slice], labels, valid)
        hist, _ = to_xy(ds["history"][:, episode_slice], labels, valid)
        return {
            "current": current,
            "history": hist,
            "labels": y,
            "visible": (valid & ds["visible"][:, episode_slice])[valid],
        }

    results = {}
    for observer in range(NUM_AGENTS):
        for partner in range(NUM_AGENTS):
            if observer == partner:
                continue
            key = f"observer{observer}_partner{partner}"
            late = pack(observer, partner, num_stages - 1, test_epi, argv.history)
            x_test, y_test, visible = late["current"], late["labels"], late["visible"]

            def training_data(name):
                if name == "frozen_early":
                    data = pack(observer, partner, 0, train_epi, argv.history)
                    return data["current"], data["labels"], data["current"], data["labels"]
                if name == "late_only":
                    data = pack(observer, partner, num_stages - 1, train_epi, argv.history)
                    return data["current"], data["labels"], data["current"], data["labels"]
                if name == "online_all":
                    xs, ys = [], []
                    for stage in range(num_stages):
                        data = pack(observer, partner, stage, train_epi, argv.history)
                        xs.append(data["current"])
                        ys.append(data["labels"])
                    x_all = np.concatenate(xs, axis=0)
                    y_all = np.concatenate(ys, axis=0)
                    return x_all, y_all, x_all, y_all
                if name == "history_late":
                    data = pack(observer, partner, num_stages - 1, train_epi, argv.history)
                    return data["history"], data["labels"], data["history"], data["labels"]
                raise ValueError(name)

            record = {}

            # Prior-only baselines: how much of the gain is just an action-frequency shift?
            def majority_baseline(labels):
                counts = np.bincount(labels, minlength=ACTION_DIM)
                return int(np.argmax(counts)), float(counts.max() / counts.sum())

            early_labels = pack(observer, partner, 0, train_epi, argv.history)["labels"]
            late_labels = pack(observer, partner, num_stages - 1, train_epi, argv.history)["labels"]
            for name, labels in (("stale_prior", early_labels), ("late_prior", late_labels)):
                cls, freq = majority_baseline(labels)
                acc = float((y_test == cls).mean())
                vis = float((y_test[visible] == cls).mean()) if visible.sum() > 0 else float("nan")
                record[name] = {
                    "accuracy": acc,
                    "macro_f1": float("nan"),
                    "accuracy_partner_visible": vis,
                    "majority_class": cls,
                    "majority_frequency_in_train": freq,
                    "train_samples": int(labels.shape[0]),
                    "test_samples": int(x_test.shape[0]),
                    "visible_samples": int(visible.sum()),
                }
                print(f"  [{key}] {name}: majority class {cls} (train freq {freq:.3f}) "
                      f"-> test acc {acc:.3f}", flush=True)

            for name in ["frozen_early", "late_only", "online_all", "history_late"]:
                x_tr, y_tr, x_val, y_val = training_data(name)
                x_te = late["history"] if name == "history_late" else x_test
                print(f"  [{key}] {name}: train {x_tr.shape[0]} samples "
                      f"(dim {x_tr.shape[-1]}), test {x_te.shape[0]}", flush=True)
                state, evaluate = train_model(
                    jnp.asarray(x_tr), jnp.asarray(y_tr),
                    jnp.asarray(x_val), jnp.asarray(y_val),
                    seed=argv.seed, steps=argv.train_steps,
                )
                acc, f1 = evaluate(state, jnp.asarray(x_te), jnp.asarray(y_test))
                logits = jax.device_get(
                    jax.jit(lambda p, x: state.apply_fn(p, x))(state.params, jnp.asarray(x_te))
                )
                preds = np.asarray(logits).argmax(axis=-1)
                vis_acc = (
                    float((preds[visible] == y_test[visible]).mean())
                    if visible.sum() > 0
                    else float("nan")
                )
                record[name] = {
                    "accuracy": acc,
                    "macro_f1": f1,
                    "accuracy_partner_visible": vis_acc,
                    "train_samples": int(x_tr.shape[0]),
                    "test_samples": int(x_test.shape[0]),
                    "visible_samples": int(visible.sum()),
                }
                print(f"    -> {name}: acc {acc:.3f} macroF1 {f1:.3f} acc(visible) {vis_acc:.3f}",
                      flush=True)
            results[key] = record

    out_path = Path(argv.out) if argv.out else data_path.with_name(data_path.stem + "_etm_info_gate.json")
    out_path.write_text(json.dumps({"stages": stages, "results": results}, indent=2) + "\n", encoding="utf-8")

    print("\nsummary (mean over the 12 observer/partner pairs):")
    for name in ["stale_prior", "late_prior", "frozen_early", "late_only", "online_all", "history_late"]:
        accs = [results[k][name]["accuracy"] for k in results]
        vis = [results[k][name]["accuracy_partner_visible"] for k in results]
        f1s = [results[k][name]["macro_f1"] for k in results]
        print(f"  {name:>13}: acc {np.nanmean(accs):.3f} | macroF1 {np.nanmean(f1s):.3f} "
              f"| acc(partner visible) {np.nanmean(vis):.3f}")
    print(f"[etm gate] wrote {out_path}")


if __name__ == "__main__":
    main()
