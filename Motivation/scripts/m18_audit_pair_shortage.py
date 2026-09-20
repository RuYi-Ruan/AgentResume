"""Read-only diagnosis of M18 source overlap and pairing funnel."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

root = Path("data/m18/v1_waitfix_20260915/group_0")
history = json.loads((root / "history/impressions.json").read_text(encoding="utf-8"))
hist_hashes = {r["event_hash"] for name in ("stale", "updated") for r in history[name]}
hist_families = {r["family"] for name in ("stale", "updated") for r in history[name]}


def pairs(partition):
    for phase in ("pre", "post"):
        for path in sorted((root / "pairs" / partition / phase).glob("*.json")):
            episode = json.loads(path.read_text(encoding="utf-8"))
            yield from episode["pairs"]


development, formal = list(pairs("development_sources")), list(pairs("A_sources"))
dev_hashes = {r["event_hash"] for r in development}
dev_families = {r["family"] for r in development}
for name, rows in (("development", development), ("formal", formal)):
    counts = Counter()
    for row in rows:
        h, f = row["event_hash"], row["family"]
        if h in hist_hashes:
            counts["exact_history"] += 1
        if f in hist_families:
            counts["near_family_history"] += 1
        if name == "formal":
            if h in dev_hashes:
                counts["exact_dev"] += 1
            if f in dev_families:
                counts["near_family_dev"] += 1
        counts["post_"+row["scoring_only"]["post"]["intent"]] += 1
    print(json.dumps({"partition": name, "raw": len(rows),
                      "unique_source_episodes": len({r["source_seed"] for r in rows}),
                      "unique_exact_public": len({r["event_hash"] for r in rows}),
                      "unique_near_families": len({r["family"] for r in rows}),
                      "only_exact_disjoint": len({r["event_hash"] for r in rows
                                                   if r["event_hash"] not in hist_hashes | dev_hashes
                                                   or name == "development" and r["event_hash"] not in hist_hashes}),
                      "counts": dict(counts)}, ensure_ascii=False))

natural = Counter()
for path in sorted((root / "pairs/development_sources/post").glob("*.json")):
    ep = json.loads(path.read_text(encoding="utf-8"))
    natural.update(row["scoring_only"]["intent"] for row in ep["decisions"])
print(json.dumps({"partition": "development_natural_visible", "visible_task_starts": sum(natural.values()),
                  "intent_counts": dict(natural)}, ensure_ascii=False))
