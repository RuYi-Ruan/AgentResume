"""Freeze ten clear/ordinary natural visible events for prompt calibration only."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
from ocres.m18_data import select_balanced
from m18_train import save


def main():
    root = Path("data/m18/v1_waitfix_20260915/group_0")
    destination = root / "development_D"
    if destination.exists():
        raise RuntimeError("development D already frozen")
    history = json.loads((root / "history/impressions.json").read_text(encoding="utf-8"))
    controlled = json.loads((root / "controlled_A/candidates.json").read_text(encoding="utf-8"))
    hashes = {r["event_hash"] for phase in ("stale", "updated") for r in history[phase]}
    families = {r["family"] for phase in ("stale", "updated") for r in history[phase]}
    hashes.update(r["event_hash"] for r in controlled)
    families.update(r["family"] for r in controlled)
    raw = []
    for source in sorted((root / "pairs/development_sources/post").glob("*.json")):
        episode = json.loads(source.read_text(encoding="utf-8"))
        raw.extend(episode["decisions"])
    eligible = [{**r, "scoring_only": {"post": r["scoring_only"]}} for r in raw
                if r["event_hash"] not in hashes and r["family"] not in families]
    selected = select_balanced(eligible, 10, 18941)
    if len(selected) != 10:
        raise RuntimeError("insufficient isolated natural development events")
    save(destination / "development_scoring.json", selected)
    save(destination / "development_public.json", [{"event_hash": r["event_hash"],
                                                     "family": r["family"], "public_event": r["event"]}
                                                    for r in selected])
    save(destination / "manifest.json", {"purpose": "prompt calibration, never formal evidence",
                                         "source": "M18 natural development post trajectories",
                                         "raw": len(raw), "eligible": len(eligible),
                                         "selected": len(selected),
                                         "classes": dict(Counter(r["scoring_only"]["post"]["intent"] for r in selected))})
    print(json.dumps({"selected": len(selected),
                      "classes": dict(Counter(r["scoring_only"]["post"]["intent"] for r in selected))}), flush=True)


if __name__ == "__main__":
    main()
