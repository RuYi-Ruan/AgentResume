"""M3 analysis on persisted dataset: matched-prefix flips L0 vs Lk per seed."""
import json
import pathlib
import sys

sys.path.insert(0, "src")
from ocres import data

OUT = pathlib.Path("data/twopot_v1")
SEEDS = [1, 2, 3]


def obs(l):
    return (str(l["p0"]), str(l["p1"]), l["held0"], l["held1"], l["pot"], str(l["a"]))


logs0_all = {seed: data.load_episode(OUT, "L0", seed)[0] for seed in SEEDS}
logsk_all = {seed: data.load_episode(OUT, "Lk", seed)[0] for seed in SEEDS}
total = 0
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
            if (
                str(li["p0"]), str(li["p1"]), li["held0"], li["held1"], li["pot"]
            ) == (
                str(lk["p0"]), str(lk["p1"]), lk["held0"], lk["held1"], lk["pot"],
            ) and li["intent0"] != lk["intent0"]:
                flips.append((li["t"], (li["intent0"], str(li["target0"])), (lk["intent0"], str(lk["target0"]))))
        i = max(j, i + 1)
    total += len(flips)
    print(f"seed{seed}: flips={len(flips)}")
    for f in flips[:8]:
        print("   t=%d L0=%s Lk=%s" % f)
print("TOTAL flips:", total)
with open(OUT / "flips.json", "w") as fh:
    json.dump({"total": total}, fh)
