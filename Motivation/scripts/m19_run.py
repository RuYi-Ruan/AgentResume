"""M19 wrappers: run the frozen M17 pipeline on other Alice checkpoint groups.

Usage (torch env for prepare/impressions; plain env for qwen):
  python scripts/m19_prepare.py --group 1 --dev 20 --history 40 --formal 50
  python scripts/m19_impressions.py --group 1
  python scripts/m19_qwen.py --group 1 --split development --events 12 --repeats 3 --live
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RUN = "v1_waitfix_20260915"
CKPT_DIR = ROOT / "artifacts" / "m18" / RUN / "checkpoints"
DATA_DIR = ROOT / "data" / "m19" / RUN


def group_map(group: int):
    res = json.loads((ROOT / "data" / "m18" / RUN / f"group_{group}" / "result.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs" / "m18_v1.json").read_text(encoding="utf-8"))
    return res["group"]["map"], list(cfg["maps"][res["group"]["map"]]), res["group"]["training_seed"]


def merged_old_audit(out_path: pathlib.Path):
    """Signatures already consumed by M16 pilot / M17 splits -> development only."""
    pairs = []
    m16 = ROOT / "data" / "m16_v2_checkpoint_audit.json"
    if m16.exists():
        data = json.loads(m16.read_text(encoding="utf-8"))
        pairs += data.get("strict_pairs", {}).get("pairs", [])
    for f in ("data/m17_fixed_event_splits.json",):
        p = ROOT / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        for split in ("development", "history_probe", "formal_test"):
            for row in d.get(split, {}).get("events", []):
                before = row["public_event"]["before"]
                pairs.append({
                    "cooking_pot_index": row["group"]["cooking_pot_index"],
                    "alice_position": before.get("alice_visible_position"),
                    "bob_position": before.get("bob_position"),
                    "first_action": row["group"]["action_delta"],
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"strict_pairs": {"pairs": pairs}}, ensure_ascii=False), encoding="utf-8")
    return len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("prepare", "impressions", "qwen"))
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--dev", type=int, default=20)
    ap.add_argument("--history", type=int, default=40)
    ap.add_argument("--formal", type=int, default=50)
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--split", choices=("development", "formal_test"), default="development")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--allow-formal", action="store_true")
    ap.add_argument("--prompt-variant", default="v1")
    args = ap.parse_args()

    map_name, rows, seed = group_map(args.group)
    out_dir = DATA_DIR / f"group_{args.group}"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = out_dir / "splits.json"
    impressions = out_dir / "impressions.json"
    pre = CKPT_DIR / f"group_{args.group}" / "pre.pt"
    post = CKPT_DIR / f"group_{args.group}" / "post.pt"
    print(f"[m19] group={args.group} map={map_name} seed={seed}")

    if args.mode == "prepare":
        import m17_prepare_fixed_events as mod

        mod.AMBIGUOUS_KITCHEN_V2 = tuple(rows)
        audit = out_dir / "old_audit.json"
        n = merged_old_audit(audit)
        print(f"[m19] merged exclusions (dev-only): {n}")
        sys.argv = ["m19_prepare", "--pre-checkpoint", str(pre), "--post-checkpoint", str(post),
                    "--old-audit", str(audit), "--development-events", str(args.dev),
                    "--history-events", str(args.history), "--formal-events", str(args.formal),
                    "--output", str(splits)]
        mod.main()
    elif args.mode == "impressions":
        import m17_build_paired_impressions as mod
        import m17_prepare_fixed_events as prep

        mod.AMBIGUOUS_KITCHEN_V2 = tuple(rows)
        prep.AMBIGUOUS_KITCHEN_V2 = tuple(rows)
        sys.argv = ["m19_impressions", "--splits", str(splits),
                    "--pre-checkpoint", str(pre), "--post-checkpoint", str(post),
                    "--output", str(impressions)]
        mod.main()
    else:
        import m17_qwen_fixed as mod

        mod.AMBIGUOUS_KITCHEN_V2 = tuple(rows)
        out = out_dir / f"{args.split}_{args.prompt_variant}_responses.json"
        sys.argv = ["m19_qwen", "--splits", str(splits), "--impressions", str(impressions),
                    "--split", args.split, "--events", str(args.events), "--repeats", str(args.repeats),
                    "--output", str(out)] + (["--live"] if args.live else []) + (["--allow-formal"] if args.allow_formal else [])
        mod.main()


if __name__ == "__main__":
    main()
