"""M2 data asset: episode persistence, loader, event-stream annotation,
capability/E statistics, and level-config derivation.

Layout note: research layouts are defined as ASCII grids (see TWO_POT below)
so geometry, stands and intent semantics stay fully under our control.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

from ocres.grid import World
from ocres.agents import CookAgent
from ocres.runner import run_episode

TWO_POT = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]

# E-level knob mapping (parallel_after_delay):
#   None = serial cook (one pot at a time) ... L0
#   >= 0 = allow starting a second pot after that many ticks of cooking
LEVEL_DEFS = {"L0": None, "L1": 12, "L2": 6, "Lk": 0}


def make_level_world(grid_rows, delay, horizon, seed):
    """Deterministic env for one (level, seed) episode."""
    np.random.seed(seed)
    w = World.make(grid_rows=grid_rows, horizon=horizon)
    # randomize start positions deterministically for cross-seed variety
    start_fn = w.grid.gw.get_random_start_state_fn(random_start_pos=True)
    w.env.reset(regen_mdp=True, outside_info={})
    # simpler determinism: fixed start per seed by re-seeding before each use
    w.start_fn = start_fn
    return w


def run_level_episode(grid_rows, alice_delay, partner_delay, horizon, seed):
    """Return (logs, metrics, game_stats, agent_ids)."""
    np.random.seed(seed)
    w = World.make(grid_rows=grid_rows, horizon=horizon)
    alice = CookAgent(w.grid, me=0, parallel_after_delay=alice_delay)
    partner = CookAgent(w.grid, me=1, parallel_after_delay=partner_delay)
    logs, metrics = run_episode(w, [alice, partner], horizon=horizon)
    stats = w.env.game_stats
    return logs, metrics, stats


# ---------------------------------------------------------------- persist
def persist_episode(out_dir, level, seed, logs, metrics, stats, alice_delay, horizon, layout_key):
    d = pathlib.Path(out_dir) / f"level_{level}"
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "layout": layout_key,
        "level": level,
        "alice_delay": alice_delay,
        "seed": seed,
        "horizon": horizon,
        "metrics": metrics,
        "game_stats": {k: v for k, v in stats.items()},
    }
    # per-tick arrays (stable column set; append-only in later stages)
    cols = {k: [] for k in logs[0].keys()}
    for l in logs:
        for k, v in l.items():
            cols[k].append(v)
    arr = {}
    for k, v in cols.items():
        arr[k] = np.array(v, dtype=object)
    np.savez_compressed(d / f"ep_seed{seed}.npz", **arr)
    with open(d / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)


def load_episode(out_dir, level, seed):
    d = pathlib.Path(out_dir) / f"level_{level}" / f"ep_seed{seed}.npz"
    z = np.load(d, allow_pickle=True)
    logs = []
    n = len(z["t"])
    keys = list(z.files)
    for i in range(n):
        logs.append({k: z[k][i] for k in keys})
    meta_p = pathlib.Path(out_dir) / f"level_{level}" / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    return logs, meta


# ------------------------------------------------- realized audit labels
EVENT_CLASS = {
    "onion_pickup": "FETCH",
    "potting_onion": "PLACE",
    "dish_pickup": "GET_DISH",
    "soup_pickup": "PICKUP",
    "soup_delivery": "DELIVER",
}


def event_timelines(stats):
    """agent -> {class: sorted [t, ...]} from env.game_stats."""
    out = {}
    for agent in (0, 1):
        tl = {}
        for key, cls in EVENT_CLASS.items():
            ts = stats.get(key, [[], []])[agent]
            tl[cls] = sorted(int(t) for t in ts)
        out[agent] = tl
    return out


def realized_labels(logs, stats, horizon, L=20):
    """Per-tick realized (future-event) label per agent, for audit."""
    tls = event_timelines(stats)
    labels = {0: [], 1: []}
    for l in logs:
        t = int(l["t"])
        for agent in (0, 1):
            best = None
            for cls, ts in tls[agent].items():
                for e in ts:
                    if t <= e <= t + L:
                        if best is None or e < best[1]:
                            best = (cls, e)
                        break
            labels[agent].append(best[0] if best else "IDLE")
    return labels


def audit_consistency(logs, labels, agent=0):
    """Agreement between primary logged intents and realized labels, only on
    event-visible classes (PRE/HOLD/COOK_START/IDLE are excluded by design)."""
    classes = set(EVENT_CLASS.values())
    n, agree, matrix = 0, 0, {}
    for l, lab in zip(logs, labels[agent]):
        prim = l[f"intent{agent}"]
        if prim not in classes:
            continue
        n += 1
        matrix[(prim, lab)] = matrix.get((prim, lab), 0) + 1
        if prim == lab:
            agree += 1
    return {"ticks": n, "agreement": agree / n if n else None, "confusion": {f"{a}->{b}": c for (a, b), c in matrix.items()}}


# ------------------------------------------------- capability + E stats
def summarize(logs, metrics):
    from collections import Counter

    c0 = Counter(l["intent0"] for l in logs)
    c1 = Counter(l["intent1"] for l in logs)
    total = max(len(logs), 1)
    dwell = Counter(l["p1"] for l in logs)
    return {
        "deliveries": metrics["deliveries"],
        "reward": metrics["reward"],
        "mean_gap": metrics["mean_gap"],
        "max_stay0": metrics["max_stay0"],
        "max_stay1": metrics["max_stay1"],
        "intent_frac0": {k: round(v / total, 3) for k, v in c0.items()},
        "intent_frac1": {k: round(v / total, 3) for k, v in c1.items()},
        "partner_dwell": {f"{x},{y}": round(c / total, 3) for (x, y), c in dwell.most_common()},
    }


def derive_next_delay(mean_gap):
    """Outcome-conditioned E rule: larger measured per-soup gaps teach a
    shorter delay before the cook starts a second pot."""
    if mean_gap is None or mean_gap > 36:
        return 12
    if mean_gap > 33:
        return 6
    if mean_gap > 30.5:
        return 2
    return 0
