"""Create a non-destructive acceptance layer for the approved D-v3 capacity rule."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from m18_train import save


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    config_path = Path("configs/m18_d_v3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not (config["accept_actual_count_when_below_maximum"]
            and config["maximum_events_per_group"] == 50
            and config["source_starts_per_group"] == 100
            and config["preserve_v2_manifests"]):
        raise RuntimeError("D-v3 differs from approved capacity rule")
    groups, total = [], 0
    for group in range(9):
        directory = root / f"group_{group}" / "natural_D"
        manifest_path = directory / "manifest.json"
        public_path = directory / "formal_public.json"
        truth_path = directory / "formal_scoring.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        public = json.loads(public_path.read_text(encoding="utf-8"))
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        hashes = [row["event_hash"] for row in public]
        if (manifest["source_seed_count"] != config["source_starts_per_group"]
                or manifest["selected"] != len(public) or len(public) != len(truth)
                or hashes != [row["event_hash"] for row in truth]
                or len(hashes) != len(set(hashes))
                or not config["minimum_events_per_group"] <= len(public) <= config["maximum_events_per_group"]):
            raise RuntimeError(f"group {group} does not satisfy approved D-v3 bounds")
        row = {"group": group, "accepted": True, "actual_events": len(public),
               "original_v2_passed": manifest["passed"], "original_v2_required": manifest["required"],
               "source_episodes": manifest["selected_source_episodes"],
               "near_families": manifest["selected_near_families"],
               "intent_counts": manifest["intent_counts"], "v2_manifest_hash": sha(manifest_path),
               "public_hash": sha(public_path), "scoring_hash": sha(truth_path)}
        receipt = directory / "d_v3_acceptance.json"
        if receipt.exists():
            if json.loads(receipt.read_text(encoding="utf-8")) != row:
                raise RuntimeError(f"group {group} D-v3 acceptance differs")
        else:
            save(receipt, row)
        groups.append(row)
        total += len(public)
    report = {"version": config["version"], "passed": True, "groups": groups,
              "total_events": total, "expected_total": config["formal_total_expected_from_frozen_v2"],
              "config_hash": sha(config_path), "v2_manifests_preserved": True}
    if total != report["expected_total"]:
        raise RuntimeError(f"frozen D total {total} differs from reviewed total")
    destination = root / "D_v3_acceptance.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != report:
            raise RuntimeError("run-level D-v3 acceptance differs")
    else:
        save(destination, report)
    print(json.dumps({"D_v3_passed": True, "groups": len(groups), "total_events": total}), flush=True)


if __name__ == "__main__":
    main()
