"""Development probe: does previous-game teammate evidence help action prediction?

Support is computed from the observer's own observations in games 0-9. Query
events come only from later games, and target actions are labels, not inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from history_prediction_diagnostic import ALLY_START, ALLY_WIDTH, fit_predict


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qmix_ns_checkpoint_diagnostic" / "trajectories.npz"
OUTPUT = HERE / "qmix_ns_checkpoint_diagnostic" / "cross_episode_prediction_v2.json"


def support_features(obs, alive, steps, seeds, times):
    result = {}
    for step in np.unique(steps):
        stage = np.flatnonzero((steps == step) & (seeds < 10))
        for observer in range(5):
            own = obs[stage, observer]
            own_alive = alive[stage, observer].astype(bool)
            for target in range(5):
                if target == observer:
                    continue
                ally_index = target - int(target > observer)
                base = ALLY_START + ALLY_WIDTH * ally_index
                visible = own_alive & (own[:, base] > 0.5)
                z = own[visible, base+1:base+5]
                if not len(z):
                    result[(int(step), observer, target)] = np.zeros(12, dtype=np.float32)
                    continue
                # Relative motion is used only when the same ally remained visible
                # in consecutive steps of the same previous episode.
                pos = stage[visible]
                prior = pos - 1
                valid = (times[pos] > 0) & (steps[prior] == steps[pos]) & (seeds[prior] == seeds[pos])
                valid &= (alive[prior, observer] == 1) & (obs[prior, observer, base] > 0.5)
                motion = np.abs(obs[pos[valid], observer, base+2:base+4] -
                                obs[prior[valid], observer, base+2:base+4])
                movement = motion.mean(axis=0) if len(motion) else np.zeros(2)
                result[(int(step), observer, target)] = np.asarray([
                    float(visible.sum() / max(1, own_alive.sum())),
                    *z.mean(axis=0), *z.std(axis=0),
                    float(z[:, 3].min()), *movement,
                ], dtype=np.float32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--model-seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    torch.set_num_threads(1)
    with np.load(args.source) as data:
        obs, actions, alive = data["obs"], data["actions"], data["alive"]
        steps, seeds, times = data["step"], data["seed"], data["time"]
    support = support_features(obs, alive, steps, seeds, times)
    now, correct, wrong, stale, labels, train, pairs = [], [], [], [], [], [], []
    earliest = int(steps.min())
    for k in np.flatnonzero(seeds >= 10):
        for observer in range(5):
            if not alive[k, observer]:
                continue
            own = obs[k, observer]
            for target in range(5):
                if target == observer:
                    continue
                ally_index = target - int(target > observer)
                if own[ALLY_START + ALLY_WIDTH * ally_index] < 0.5 or not alive[k, target]:
                    continue
                ids = np.zeros(10, dtype=np.float32)
                ids[observer] = ids[5+target] = 1
                x = np.concatenate((own, ids))
                key = (int(steps[k]), observer, target)
                other = next(j for j in range(5) if j != observer and j != target)
                now.append(x)
                correct.append(np.concatenate((x, support[key])))
                wrong.append(np.concatenate((x, support[(int(steps[k]), observer, other)])))
                stale.append(np.concatenate((x, support[(earliest, observer, target)])))
                labels.append(actions[k, target])
                train.append(seeds[k] < 40)
                pairs.append(observer*5+target)
    x_now = np.asarray(now, dtype=np.float32)
    x_correct = np.asarray(correct, dtype=np.float32)
    x_wrong = np.asarray(wrong, dtype=np.float32)
    x_stale = np.asarray(stale, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    pair = np.asarray(pairs, dtype=np.int16)
    is_train = np.asarray(train, dtype=bool)
    rng = np.random.default_rng(123)
    train_ids = np.flatnonzero(is_train)
    test_ids = np.flatnonzero(~is_train)
    if len(train_ids) > 40000:
        train_ids = rng.choice(train_ids, 40000, replace=False)
    if len(test_ids) > 12000:
        test_ids = rng.choice(test_ids, 12000, replace=False)
    training = np.zeros(len(y), dtype=bool)
    testing = np.zeros(len(y), dtype=bool)
    training[train_ids], testing[test_ids] = True, True
    if not training.any() or not testing.any():
        raise RuntimeError("empty train or test split")
    predictions = {
        "current": fit_predict(x_now, y, training, testing, args.epochs, args.model_seed),
        "correct_partner_support": fit_predict(x_correct, y, training, testing, args.epochs, args.model_seed),
        "wrong_partner_support": fit_predict(x_wrong, y, training, testing, args.epochs, args.model_seed),
        "stale_partner_support": fit_predict(x_stale, y, training, testing, args.epochs, args.model_seed),
    }
    truth = y[testing]
    result = {"status": "completed_development_probe", "model_seed": args.model_seed,
              "source_checkpoints": sorted(set(steps.tolist())),
              "support_seeds": "0-9", "train_query_seeds": "10-39", "test_query_seeds": "40-49",
              "train_events": int(training.sum()), "test_events": int(testing.sum()),
              "attack_events": int((truth >= 6).sum()), "accuracy": {}, "attack_accuracy": {},
              "pair_results": [],
              "note": "Support contains only observer-local visibility, relative position, HP, and relative motion from previous games. Frozen checkpoints are separate offline samples, not a simulated online-learning trajectory. Positive prediction alone would not show cooperation gain."}
    for name, pred in predictions.items():
        result["accuracy"][name] = round(float(np.mean(pred == truth)), 4)
        result["attack_accuracy"][name] = round(float(np.mean(pred[truth >= 6] == truth[truth >= 6])), 4)
    for observer in range(5):
        for target in range(5):
            if observer == target:
                continue
            take = pair[testing] == observer*5+target
            if take.sum() >= 20:
                row = {"observer": observer, "target": target, "events": int(take.sum())}
                row.update({name: round(float(np.mean(pred[take] == truth[take])), 4)
                            for name, pred in predictions.items()})
                result["pair_results"].append(row)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("train_events", "test_events", "attack_events", "accuracy", "attack_accuracy")}, indent=2))


if __name__ == "__main__":
    main()
