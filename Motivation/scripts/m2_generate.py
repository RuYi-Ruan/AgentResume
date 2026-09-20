"""M2/M3: full level dataset + realized audit + matched-prefix flip stats."""
import json
import pathlib
import random
import sys

sys.path.insert(0, "src")

from ocres import data
from ocres.data import TWO_POT
from ocres.agents import CookAgent
from ocres.runner import run_episode
from ocres.grid import World

HORIZON = 900
SEEDS = [1, 2, 3]
OUT = pathlib.Path("data/twopot_v1")
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = [("L0", None), ("L1", 12), ("L2", 6), ("Lk", 0)]


def randomize_start(w, seed):
    rng = random.Random(1000 + seed)
    cells = sorted(w.grid.passable)
    a, b = rng.sample(cells, 2)
    s = w.env.state
    s.players[0].update_pos_and_or(a, (1, 0))
    s.players[1].update_pos_and_or(b, (1, 0))
    s.timestep = 0


def run_one(delay, seed):
    w = World.make(grid_rows=TWO_POT, horizon=HORIZON)
    randomize_start(w, seed)
    alice = CookAgent(w.grid, me=0, parallel_after_delay=delay)
    partner = CookAgent(w.grid, me=1, parallel_after_delay=None)
    logs, metrics = run_episode(w, [alice, partner], horizon=HORIZON)
    return logs, metrics, w.env.game_stats


episodes = {}
summary = {}
for label, delay in LEVELS:
    rows = []
    for seed in SEEDS:
        logs, metrics, stats = run_one(delay, seed)
        data.persist_episode(OUT, label, seed, logs, metrics, stats, delay, HORIZON, "twopot_v1")
        aud = data.audit_consistency(logs, data.realized_labels(logs, stats, HORIZON))
        summ = data.summarize(logs, metrics)
        rows.append({"seed": seed, "metrics": metrics, "audit": aud, "summary": summ})
        print(
            f"{label} seed{seed}: del={metrics['deliveries']} gap={metrics['mean_gap']} "
            f"audit_agree={aud['agreement']}"
        )
    gaps = [r["metrics"]["mean_gap"] for r in rows if r["metrics"]["mean_gap"]]
    summary[label] = {
        "delay": delay,
        "deliveries_mean": sum(r["metrics"]["deliveries"] for r in rows) / len(rows),
        "gap_mean": sum(gaps) / len(gaps) if gaps else None,
        "rows": rows,
    }
    episodes[label] = rows

with open(OUT / "summary.json", "w") as f:
    json.dump(
        {
            l: {
                "delay": s["delay"],
                "deliveries_mean": s["deliveries_mean"],
                "gap_mean": s["gap_mean"],
                "audit_mean": sum(r["audit"]["agreement"] for r in s["rows"] if r["audit"]["agreement"]) / len(s["rows"]),
            }
            for l, s in summary.items()
        },
        f,
        indent=1,
        default=str,
    )
print("\ncapability summary:")
for l, s in summary.items():
    print(f"  {l}: deliveries~{s['deliveries_mean']:.1f} gap~{s['gap_mean']:.2f}")

# ---- M3 matched-prefix flip stats across (L0, Lk) ----------------------
logs0_all = {seed: data.load_episode(OUT, "L0", seed)[0] for seed in SEEDS}
logsk_all = {seed: data.load_episode(OUT, "Lk", seed)[0] for seed in SEEDS}


def obs(l):
    return (l["p0"], l["p1"], l["held0"], l["held1"], l["pot"], tuple(l["a"]))


total_flips = 0
per_seed_flips = {}
for seed in SEEDS:
    logs0, logsk = logs0_all[seed], logsk_all[seed]
    n = min(len(logs0), len(logsk))
    flips = []
    i = 0
    while i < n:
        j = i
        while j < n and obs(logs0[j]) == obs(logsk[j]):
            j += 1
        if j - i >= 2 and j < n:
            li, lk = logs0[j], logsk[j]
            if (li["p0"], li["p1"], li["held0"], li["held1"], li["pot"]) == (
                lk["p0"],
                lk["p1"],
                lk["held0"],
                lk["held1"],
                lk["pot"],
            ) and li["intent0"] != lk["intent0"]:
                flips.append(
                    {
                        "t": li["t"],
                        "L0": (li["intent0"], str(li["target0"])),
                        "Lk": (lk["intent0"], str(lk["target0"])),
                    }
                )
        i = max(j, i + 1)
    per_seed_flips[seed] = flips
    total_flips += len(flips)
    print(f"seed{seed}: matched-prefix flips = {len(flips)}")

print(f"TOTAL matched-prefix flips (L0 vs Lk): {total_flips}")
for seed in SEEDS:
    for f in per_seed_flips[seed][:5]:
        print(f"  seed{seed} t={f['t']}: L0={f['L0']} vs Lk={f['Lk']}")
