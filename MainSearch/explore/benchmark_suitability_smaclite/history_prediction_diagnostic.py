"""Test whether an observer's visible history helps predict a teammate's action.

Uses only the observer's own SMAClite observations. No target policy weights,
target observations, hidden role labels, or future data enter the predictor.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qmix_ns_checkpoint_diagnostic" / "trajectories.npz"
OUTPUT = HERE / "qmix_ns_checkpoint_diagnostic" / "history_prediction.json"
ALLY_START = 44  # 4 movement + 5 enemies * 8 features on SMAClite 2s3z.
ALLY_WIDTH = 8
HISTORY = 3


class Predictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 11))

    def forward(self, x):
        return self.net(x)


def make_examples(data):
    obs, actions, alive = data["obs"], data["actions"], data["alive"]
    steps, seeds, times = data["step"], data["seed"], data["time"]
    current, history, labels, train, pairs, checkpoint = [], [], [], [], [], []
    n_agents = obs.shape[1]
    for k in range(HISTORY, len(obs)):
        if times[k] < HISTORY or steps[k-HISTORY] != steps[k] or seeds[k-HISTORY] != seeds[k]:
            continue
        for observer in range(n_agents):
            if not alive[k, observer]:
                continue
            own = obs[k, observer]
            own_hist = [obs[k-j, observer] for j in range(1, HISTORY+1)]
            for target in range(n_agents):
                if target == observer or not alive[k, target]:
                    continue
                ally_index = target - int(target > observer)
                if own[ALLY_START + ALLY_WIDTH * ally_index] < 0.5:
                    continue
                ids = np.zeros(2*n_agents, dtype=np.float32)
                ids[observer] = ids[n_agents+target] = 1
                x = np.concatenate((own, ids))
                current.append(x)
                history.append(np.concatenate((x, *own_hist)))
                labels.append(actions[k, target])
                train.append(seeds[k] < 40)
                pairs.append(observer*n_agents + target)
                checkpoint.append(steps[k])
    return (np.asarray(current, dtype=np.float32), np.asarray(history, dtype=np.float32),
            np.asarray(labels, dtype=np.int64), np.asarray(train, dtype=bool),
            np.asarray(pairs, dtype=np.int16), np.asarray(checkpoint, dtype=np.int32))


def fit_predict(x, y, train_mask, test_mask, epochs, model_seed=0):
    torch.manual_seed(model_seed)
    model = Predictor(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x_train = torch.from_numpy(x[train_mask])
    y_train = torch.from_numpy(y[train_mask])
    x_test = torch.from_numpy(x[test_mask])
    rng = np.random.default_rng(model_seed)
    for _ in range(epochs):
        order = rng.permutation(len(x_train))
        model.train()
        for start in range(0, len(order), 1024):
            idx = order[start:start+1024]
            loss = nn.functional.cross_entropy(model(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(chunk).argmax(dim=1).numpy()
                               for chunk in x_test.split(2048)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    torch.set_num_threads(1)
    with np.load(args.source) as data:
        x_now, x_history, y, is_train, pair, checkpoint = make_examples(data)
    train_ids = np.flatnonzero(is_train)
    test_ids = np.flatnonzero(~is_train)
    rng = np.random.default_rng(123)
    if len(train_ids) > 40000:
        train_ids = rng.choice(train_ids, 40000, replace=False)
    if len(test_ids) > 12000:
        test_ids = rng.choice(test_ids, 12000, replace=False)
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_ids] = True
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_ids] = True
    if len(train_ids) == 0 or len(test_ids) == 0:
        raise RuntimeError("empty train or test split")
    pred_now = fit_predict(x_now, y, train_mask, test_mask, args.epochs)
    pred_history = fit_predict(x_history, y, train_mask, test_mask, args.epochs)
    truth = y[test_mask]
    out = {
        "status": "completed_development_probe",
        "train_events": len(train_ids), "test_events": len(test_ids),
        "train_seeds": "0-39", "test_seeds": "40-49", "checkpoints": sorted(set(checkpoint.tolist())),
        "current_only_accuracy": round(float(np.mean(pred_now == truth)), 4),
        "current_plus_3_observations_accuracy": round(float(np.mean(pred_history == truth)), 4),
        "attack_events": int(np.sum(truth >= 6)),
        "current_only_attack_accuracy": round(float(np.mean(pred_now[truth >= 6] == truth[truth >= 6])), 4),
        "history_attack_accuracy": round(float(np.mean(pred_history[truth >= 6] == truth[truth >= 6])), 4),
        "pair_results": [],
        "note": "Observer-local visible history only. Improvement would show temporal information, not by itself prove ETM-specific cognition or team reward gain. Single training seed."
    }
    for observer in range(5):
        for target in range(5):
            if observer == target:
                continue
            take = pair[test_mask] == observer*5+target
            if take.sum() >= 20:
                out["pair_results"].append({"observer": observer, "target": target,
                    "events": int(take.sum()), "current": round(float(np.mean(pred_now[take] == truth[take])), 4),
                    "history": round(float(np.mean(pred_history[take] == truth[take])), 4)})
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("train_events", "test_events", "current_only_accuracy",
                                                "current_plus_3_observations_accuracy", "attack_events",
                                                "current_only_attack_accuracy", "history_attack_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
