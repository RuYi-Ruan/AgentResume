"""M5 matrix run: stale vs current x N episodes, with canonicalized guesses,
Bob-decision-epoch scoring windows, and a parallel-fill divergence subset."""
import sys
import time

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

FLAVORS = ["stale", "updated", "current"]
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
HORIZON = 260
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]

FLAVOR_CFG = {
    "stale": dict(role_card=cards.BOB_STALE, mode="static"),
    "updated": dict(role_card=cards.UPDATED_BASE, mode="auto"),
    "current": dict(role_card=cards.BOB_CURRENT, mode="static"),
}

SYN = {
    "COOK": "COOK_START",
    "COOK_START": "COOK_START",
    "FETCH": "FETCH",
    "PLACE": "PLACE",
    "GET_DISH": "GET_DISH",
    "PICKUP": "PICKUP",
    "PICK": "PICKUP",
    "DELIVER": "DELIVER",
    "SERVER": "DELIVER",
    "SERVE": "DELIVER",
    "PRE": "PRE",
    "PREP": "GET_DISH",
    "PLATE": "GET_DISH",
    "HOLD": "HOLD",
    "WAIT": "HOLD",
    "STAY": "HOLD",
    "IDLE": "HOLD",
}


def canon(g):
    if not g:
        return None
    s = str(g).strip().upper()
    return SYN.get(s)


def parallel_fill_ticks(logs):
    """ticks where a pot is cooking, another accepts onions, and Alice picks FETCH."""
    out = []
    for l in logs:
        pots = l["pot"]
        if "cooking" in pots and l["intent0"] == "FETCH":
            has_acc = any(k in pots for k in ("'empty'", "'items1'", "'items2'"))
            if has_acc:
                out.append(l["t"])
    return out


results = []
for flavor in FLAVORS:
    cfg = FLAVOR_CFG[flavor]
    for rep in range(REPS):
        model = RemoteModel()
        w = World.make(grid_rows=ROWS, horizon=HORIZON)
        alice = LLMCook(w.grid, me=0, model=model, role_card=cards.ALICE_LK, label="Alice", role_fixed="cook")
        bob = LLMBob(
            w.grid,
            me=1,
            model=model,
            role_card=cfg["role_card"],
            label="Bob",
            impression_mode=cfg["mode"],
        )
        t0 = time.time()
        logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
        dt = time.time() - t0

        # Bob decision epochs from fresh-guess rows
        epochs = []  # (t_start, t_end, guess)
        last_t = None
        guess_at = {}
        for l in logs:
            g = (l.get("info1") or {}).get("bob_guess")
            if g is not None and canon(g) is not None:
                guess_at[l["t"]] = canon(g)
        order = sorted(guess_at)
        for i, t in enumerate(order):
            end = order[i + 1] if i + 1 < len(order) else HORIZON
            epochs.append((t, end, guess_at[t]))

        exact = agree_w = total = 0
        early_w = late_w = early_n = late_n = 0
        for (a, b, g) in epochs:
            total += 1
            hit = any(x["intent0"] == g for x in logs[a:b])
            if logs[a]["intent0"] == g:
                exact += 1
            if hit:
                agree_w += 1
            mid = (a + b) / 2 < HORIZON / 2
            if mid:
                early_n += 1
                early_w += int(hit)
            else:
                late_n += 1
                late_w += int(hit)
        # divergence subset: Bob epochs covering parallel-fill ticks
        pft = parallel_fill_ticks(logs)
        div_hit = div_n = 0
        for (a, b, g) in epochs:
            if any(a <= q < b for q in pft):
                div_n += 1
                div_hit += int(g == "FETCH")
        results.append(
            {
                "flavor": flavor,
                "rep": rep,
                "deliveries": m["deliveries"],
                "gap": m["mean_gap"],
                "calls": (alice.call_count, bob.call_count),
                "epochs": total,
                "acc_exact": exact / total if total else None,
                "acc_window": agree_w / total if total else None,
                "acc_early": early_w / early_n if early_n else None,
                "acc_late": late_w / late_n if late_n else None,
                "parfill_ticks": len(pft),
                "div_guess_n": div_n,
                "div_guess_fetch": div_hit,
                "obs_samples": len(bob.obs) if cfg["mode"] == "auto" else None,
                "obs_rate": (sum(bob.obs) / len(bob.obs)) if cfg["mode"] == "auto" and bob.obs else None,
                "wall_s": round(dt),
            }
        )
        print(
            f"{flavor} rep{rep}: del={m['deliveries']} gap={m['mean_gap']} "
            f"acc_w={agree_w / total if total else None:.2%} parfill={len(pft)} "
            f"div(fetch_guess)={div_hit}/{div_n} calls={alice.call_count},{bob.call_count} wall={dt:.0f}s"
        )

import json  # noqa: E402

with open("data/llm_pilot_matrix.json", "w") as f:
    json.dump(results, f, indent=1)
for flavor in FLAVORS:
    rs = [r for r in results if r["flavor"] == flavor]
    a = [r["acc_window"] for r in rs if r["acc_window"] is not None]
    d = [r["deliveries"] for r in rs]
    pf = [r["parfill_ticks"] for r in rs]
    dh = [r["div_guess_fetch"] for r in rs]
    dn = [r["div_guess_n"] for r in rs]
    print(
        f"\n{flavor}: acc_window mean={sum(a) / len(a) if a else None:.2%} "
        f"deliveries={d} parfill={pf} div_fetch={dh}/{dn}"
    )
