"""Resume frozen B group 8 under the user-approved 29,000 request cap."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, "src")
import torch

from m18_train import emit, save
from ocres.m18_b import run_b_episode
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_live_qwen import LiveQwen
from ocres.m18_training import config_hash, episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


GROUP = 8
OLD_LIMIT = 28000
NEW_LIMIT = 29000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path("data/m18") / args.run_id
    source = root / f"group_{GROUP}"
    config = load_config()
    if config["llm"]["request_limit"] != OLD_LIMIT:
        raise RuntimeError("frozen request limit unexpectedly changed")
    options = {**config["llm"], "request_limit": NEW_LIMIT}
    original = source / "B_formal"
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != config_hash(config):
        raise RuntimeError("frozen B config differs")
    for path, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"B frozen source changed: {path}")
    if not (source / "B_formal_anomaly_recovery2_selection.json").exists():
        raise RuntimeError("network-outage recovery selection is missing")

    extension = {
        "stage": "B_formal_group8_resume",
        "reason": "user-approved allowance for outage recovery and final 21 episodes",
        "request_limit_before": OLD_LIMIT,
        "request_limit_after": NEW_LIMIT,
        "frozen_config_hash": config_hash(config),
    }
    extension_path = root / "request_cap_extension_29000.json"
    if extension_path.exists():
        if json.loads(extension_path.read_text(encoding="utf-8")) != extension:
            raise RuntimeError("request-cap extension record differs")
    else:
        save(extension_path, extension)

    group = group_spec(config, GROUP)
    history = json.loads((source / "history/impressions.json").read_text(encoding="utf-8"))
    model, spec, _ = load_policy_checkpoint(
        Path("artifacts/m18") / args.run_id / "checkpoints" / f"group_{GROUP}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)
    live = LiveQwen(group["rows"], history, options, original / "qwen",
                    root / "api_budget.jsonl", GROUP)
    order = [(seed, condition) for seed in manifest["seeds"]
             for condition in config["llm"]["conditions"] + ["oracle"]]
    random.Random(18988 + GROUP).shuffle(order)
    for seed, condition in order:
        target = original / "episodes" / f"{seed}_{condition}.json"
        if target.exists():
            continue
        result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                               predict=live.predict if condition != "oracle" else None,
                               bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": "B_formal_cap29000", "group": GROUP,
                                       "seed": seed, "condition": condition,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
        print(json.dumps({"group": GROUP, "seed": seed, "condition": condition,
                          "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)
    print(json.dumps({"group": GROUP, "B_formal_complete": True,
                      "episodes": len(order), "request_limit": NEW_LIMIT}), flush=True)


if __name__ == "__main__":
    main()
