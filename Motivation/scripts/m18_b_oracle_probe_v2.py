"""Development-only oracle probe for the candidate PublicBobV2 controller."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import torch

from m18_train import emit, save
from ocres.m18_b import run_b_episode
from ocres.m18_bob_v2 import PublicBobV2
from ocres.m18_training import episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id
    source = root / f"group_{args.group}"
    if not json.loads((source / "result.json").read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("Alice growth gate failed")
    artifacts = Path("artifacts/m18") / args.run_id
    model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / "post.pt")
    model.eval()
    destination = source / "B_controller_probe_v2"
    if destination.exists():
        raise RuntimeError("B controller v2 probe already exists; do not overwrite")
    destination.mkdir(parents=True)
    unknown = lambda event, condition, seed, index: ({"intent": "unknown", "target_facility": "unknown"}, 0)
    paired = []
    for seed in episode_seeds(config, args.group, "development"):
        none = run_b_episode(group["rows"], seed, model, spec, config, "none_local_unknown",
                             unknown, bob_class=PublicBobV2)
        oracle = run_b_episode(group["rows"], seed, model, spec, config, "oracle",
                               bob_class=PublicBobV2)
        save(destination / "episodes" / f"{seed}_none.json", none)
        save(destination / "episodes" / f"{seed}_oracle.json", oracle)
        paired.append({"seed": seed, "none_300": none["score_first_300"],
                       "oracle_300": oracle["score_first_300"], "none_700": none["score_full_700"],
                       "oracle_700": oracle["score_full_700"], "none_queries": none["query_count"],
                       "oracle_queries": oracle["query_count"],
                       "none_blocks": none["bob_block_events"], "oracle_blocks": oracle["bob_block_events"]})
        emit(root / "progress.jsonl", {"stage": "B_controller_probe_v2", "group": args.group,
                                       "seed": seed, "none_300": none["score_first_300"],
                                       "oracle_300": oracle["score_first_300"]})
    differences = [row["oracle_300"] - row["none_300"] for row in paired]
    report = {"controller": PublicBobV2.version, "group": args.group, "starts": len(paired),
              "paired": paired, "mean_300_gain_soups": sum(differences) / len(differences) / 20,
              "positive_starts": sum(x > 0 for x in differences),
              "tie_starts": sum(x == 0 for x in differences),
              "negative_starts": sum(x < 0 for x in differences),
              "starts_with_public_query": sum(row["none_queries"] > 0 for row in paired),
              "mean_public_queries": sum(row["none_queries"] for row in paired) / len(paired),
              "mean_block_gain": (sum(row["none_blocks"] - row["oracle_blocks"] for row in paired)
                                  / len(paired)),
              "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in ("src/ocres/m18_b.py", "src/ocres/m18_bob_v2.py", __file__)}}
    save(destination / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("paired", "source_hashes")}), flush=True)


if __name__ == "__main__":
    main()
