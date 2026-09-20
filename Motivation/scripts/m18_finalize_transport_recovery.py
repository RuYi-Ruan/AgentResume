"""Validate M18 transport recovery artifacts and freeze their selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


D_TAG = "transport_recovery1"
B_TAG = "transport_recovery2"
TRANSPORT_ERRORS = {"URLError", "TimeoutError", "ConnectionError", "transport_exhausted"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failures(path: Path) -> list[dict]:
    return [row for row in rows(path) if row.get("kind") == "completion"
            and TRANSPORT_ERRORS.intersection(row.get("errors", []))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    d_original = root / "group_8/natural_D/qwen/formal_v2_mapfix/responses.jsonl"
    d_recovery = root / f"group_8/natural_D/qwen/formal_v2_mapfix_{D_TAG}"
    b_original = root / "group_0/B_formal/qwen/responses.jsonl"
    b_recovery = root / f"group_0/B_formal_{B_TAG}"

    d_completions = [row for row in rows(d_recovery / "responses.jsonl")
                     if row.get("kind") == "completion"]
    if len(d_completions) != 450 or failures(d_recovery / "responses.jsonl"):
        raise RuntimeError("D recovery is incomplete or still has transport failures")
    if not (d_recovery / "scored.json").exists():
        raise RuntimeError("D recovery score is absent")

    episodes = list((b_recovery / "episodes").glob("*.json"))
    if len(episodes) != 64:
        raise RuntimeError(f"B recovery is incomplete: {len(episodes)}/64 episodes")
    if failures(b_recovery / "qwen/responses.jsonl"):
        raise RuntimeError("B recovery still has transport failures")

    selection = {
        "recovery": {"D": D_TAG, "B": B_TAG},
        "reason": "2026-09-17 external API transport outage",
        "preserve_original_failures": True,
        "D": {
            "group": 8,
            "selected_directory": str(d_recovery).replace("\\", "/"),
            "selected_responses_hash": sha256(d_recovery / "responses.jsonl"),
            "original_directory": str(d_original.parent).replace("\\", "/"),
            "original_responses_hash": sha256(d_original),
            "original_transport_failures": len(failures(d_original)),
        },
        "B": {
            "group": 0,
            "selected_directory": str(b_recovery).replace("\\", "/"),
            "selected_responses_hash": sha256(b_recovery / "qwen/responses.jsonl"),
            "selected_episode_count": len(episodes),
            "original_directory": str(b_original.parent.parent).replace("\\", "/"),
            "original_responses_hash": sha256(b_original),
            "original_transport_failures": len(failures(b_original)),
        },
    }
    output = root / "transport_recovery_final_selection.json"
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != selection:
            raise RuntimeError("frozen recovery selection differs")
    else:
        output.write_text(json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"transport_recovery_selection": "frozen",
                      "path": str(output), "D_requests": 450,
                      "B_episodes": 64}), flush=True)


if __name__ == "__main__":
    main()
