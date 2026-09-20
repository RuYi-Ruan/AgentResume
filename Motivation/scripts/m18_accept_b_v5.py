"""Freeze the nine-group development gate for PublicBobV5."""
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
    root = Path("data/m18") / args.run_id
    config_path = Path("configs/m18_b_v5.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    groups, all_diffs = [], []
    for group in range(config["development_groups"]):
        report_path = root / f"group_{group}" / "B_controller_probe_v5/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["controller"] != config["controller"] or report["starts"] != config["development_starts_per_group"]:
            raise RuntimeError(f"group {group} v5 development report differs")
        source_hashes = report["source_hashes"]
        for path, expected in source_hashes.items():
            if sha(path) != expected:
                raise RuntimeError(f"group {group} v5 source changed: {path}")
        none_episodes = []
        sectors = set()
        for row in report["paired"]:
            episode_path = report_path.parent / "episodes" / f"{row['seed']}_none.json"
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            none_episodes.append(episode)
            for query in episode["public_queries"]:
                p = query["event"]["after"]["bob"]["position"]
                sectors.add((p[0] // 2, p[1] // 2))
        starts_with_query = sum(e["query_count"] > 0 for e in none_episodes)
        mean_queries = sum(e["query_count"] for e in none_episodes) / len(none_episodes)
        trigger_passed = starts_with_query >= 12 and mean_queries >= 3 and len(sectors) >= 2
        diffs = [(row["oracle_300"] - row["none_300"]) / 20 for row in report["paired"]]
        all_diffs.extend(diffs)
        groups.append({"group": group, "trigger_passed": trigger_passed,
                       "starts_with_query": starts_with_query, "mean_queries": mean_queries,
                       "sectors": len(sectors), "mean_oracle_gain_300_soups": sum(diffs) / len(diffs),
                       "positive": sum(x > 0 for x in diffs), "tie": sum(x == 0 for x in diffs),
                       "negative": sum(x < 0 for x in diffs), "report_hash": sha(report_path)})
    mean_gain = sum(all_diffs) / len(all_diffs)
    gate = config["oracle_gate"]
    nonnegative = sum(row["mean_oracle_gain_300_soups"] >= 0 for row in groups)
    trigger_groups = sum(row["trigger_passed"] for row in groups)
    passed = (mean_gain > gate["minimum_overall_mean_gain_soups"]
              and nonnegative >= gate["minimum_nonnegative_groups"]
              and trigger_groups >= gate["minimum_groups_with_public_trigger_gate"])
    result = {"version": config["version"], "passed": passed, "controller": config["controller"],
              "development_only": True, "groups": groups, "starts": len(all_diffs),
              "mean_oracle_gain_300_soups": mean_gain,
              "positive": sum(x > 0 for x in all_diffs), "tie": sum(x == 0 for x in all_diffs),
              "negative": sum(x < 0 for x in all_diffs), "nonnegative_groups": nonnegative,
              "trigger_groups_passed": trigger_groups, "config_hash": sha(config_path),
              "controller_hash": sha("src/ocres/m18_bob_v5.py")}
    if not passed:
        raise RuntimeError(f"PublicBobV5 development gate failed: {result}")
    destination = root / "B_v5_acceptance.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != result:
            raise RuntimeError("B-v5 acceptance differs")
    else:
        save(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}), flush=True)


if __name__ == "__main__":
    main()
