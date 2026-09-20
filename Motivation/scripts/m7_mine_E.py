"""Mine candidate experiences E from naive (no-experience) Alice trajectories.

Outcome-conditioned detectors on full-truth logs; each candidate reports the
ticks it would have saved. Candidates with enough evidence (>= GATE ticks)
become experience items in data/v2_E.json.
"""
import json
import pathlib
import re
import sys

import numpy as np

OUT = pathlib.Path("data/v2_naive")
GATE = 8


def load_eps():
    eps = []
    for p in sorted(OUT.glob("ep*.npz")):
        z = np.load(p, allow_pickle=True)
        n = len(z["t"])
        eps.append([{k: z[k][i] for k in z.files} for i in range(n)])
    return eps


def pot_flags(potstr):
    return {"cooking": "cooking" in potstr, "empty": "'empty'" in potstr,
            "items12": ("'items1'" in potstr or "'items2'" in potstr),
            "ready": "'ready'" in potstr}


def mine(eps):
    cands = {
        "L_fill": {"ticks": 0, "streak": 0},
        "L_dish": {"ticks": 0, "streak": 0},
        "L_ready": {"ticks": 0, "streak": 0},
        "L_double_dish": {"ticks": 0, "streak": 0},
    }
    for logs in eps:
        for l in logs:
            pf = pot_flags(l["pot"])
            # Alice empty-handed & HOLD while some pot accepts onions and the
            # partner is not carrying an onion to fill it -> wasted fill chance
            if (l["held0"] in (None, "None") and l["intent0"] == "HOLD"
                    and (pf["empty"] or pf["items12"]) and l["held1"] != "onion"):
                cands["L_fill"]["ticks"] += 1
                cands["L_fill"]["streak"] += 1
            else:
                cands["L_fill"]["streak"] = 0
            # dish held by Alice while nothing is cooking/ready -> premature
            if l["held0"] == "dish" and not (pf["cooking"] or pf["ready"]):
                cands["L_dish"]["ticks"] += 1
                cands["L_dish"]["streak"] += 1
            else:
                cands["L_dish"]["streak"] = 0
            # pot ready but nobody is picking up soon
            if pf["ready"] and l["held1"] != "soup" and l["intent1"] not in ("PICKUP", "DELIVER"):
                cands["L_ready"]["ticks"] += 1
            # both agents holding dishes at once
            if l["held0"] == "dish" and l["held1"] == "dish":
                cands["L_double_dish"]["ticks"] += 1
    return cands


def build_items(c):
    items = []
    if c["L_fill"]["ticks"] >= GATE:
        items.append({
            "id": "E1",
            "situation": "存在一口空锅或未满的锅(empty/items1/items2)，且你空手、伙伴没有正拿着洋葱去补",
            "decide": "选 FETCH 去取洋葱补锅，而不是 HOLD 空等",
            "why": "空等不产出；利用等待时间补料才能更快出汤",
            "evidence": {"wasted_ticks": c["L_fill"]["ticks"]},
        })
    if c["L_dish"]["ticks"] >= GATE or c["L_double_dish"]["ticks"] >= GATE:
        items.append({
            "id": "E2",
            "situation": "还没有任何锅在煮或已 ready 时，或你已看到伙伴拿盘",
            "decide": "先别 GET_DISH 取盘；避免两人同时持盘（持盘时拿不了洋葱）",
            "why": "过早/重复取盘导致拿盘者无法补料，实测浪费 %d tick" % c["L_dish"]["ticks"],
            "evidence": {"premature_dish_ticks": c["L_dish"]["ticks"], "double_dish_ticks": c["L_double_dish"]["ticks"]},
        })
    if c["L_ready"]["ticks"] >= GATE:
        items.append({
            "id": "E3",
            "situation": "有锅已 ready 而你（或伙伴）正拿盘",
            "decide": "尽快 PICKUP 取汤并 DELIVER，不要拖延",
            "why": "ready 的锅占着灶位，早取早开下一锅",
            "evidence": {"ready_idle_ticks": c["L_ready"]["ticks"]},
        })
    return items


eps = load_eps()
c = mine(eps)
print("detector ticks:", c)
items = build_items(c)
print("experience items:", len(items))
for it in items:
    print(" -", it["id"], "|", it["decide"])
with open("data/v2_E.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=1)
