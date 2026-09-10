"""Egocentric 3x3 sensor + static map prior + short-term seen-state memory.

Pot memory entries: {pot: {"kind": kind, "t": seen_tick}}. A pot last seen
'cooking' more than COOK_GUESS ticks ago is reported as possibly ready, which
lets an agent infer readiness without re-visiting (like a human counting).
"""
from __future__ import annotations

from ocres.recipes import agent_held

COOK_GUESS = 22  # cook lasts ~20 ticks; margin for pickup walk

TILE_NAMES = {
    "P": "锅", "O": "洋葱台", "T": "番茄台", "D": "盘台", "S": "出餐口", "X": "墙/台面",
}


def _kind_of(v):
    if isinstance(v, dict):
        return v["kind"]
    if isinstance(v, tuple):
        return v[0]
    return v


def visible_state(state, grid, me, mem=None):
    player = state.players[me]
    other = state.players[1 - me]
    px, py = player.position
    tiles = []
    partner_visible = None
    for dy in (-1, 0, 1):
        row = []
        for dx in (-1, 0, 1):
            x, y = px + dx, py + dy
            if x < 0 or y < 0 or y >= len(grid.gw.terrain_mtx) or x >= len(grid.gw.terrain_mtx[0]):
                row.append("边界")
                continue
            ch = grid.gw.terrain_mtx[y][x]
            if ch in ("1", "2"):
                ch = " "
            if (x, y) == tuple(other.position):
                row.append(f"伙伴({other.get_object().name if other.has_object() else '空手'})")
                partner_visible = (x, y)
                continue
            if ch == "P":
                if state.has_object((x, y)):
                    soup = state.get_object((x, y))
                    if soup.is_ready:
                        row.append("锅[已好]")
                    elif soup.is_cooking:
                        row.append("锅[煮中]")
                    else:
                        row.append(f"锅[{len(soup.ingredients)}料]")
                else:
                    row.append("锅[空]")
                continue
            row.append(TILE_NAMES.get(ch, "空地" if ch == " " else "?" + ch))
        tiles.append("|".join(row))
    # memory: keep old, refresh only pots in view
    seen = dict(mem) if mem else {}
    for p in grid.pot_locs:
        if max(abs(p[0] - px), abs(p[1] - py)) <= 1:
            if state.has_object(p):
                soup = state.get_object(p)
                if soup.is_ready:
                    seen[str(p)] = {"kind": "ready", "t": state.timestep}
                elif soup.is_cooking:
                    seen[str(p)] = {"kind": "cooking", "t": state.timestep}
                else:
                    seen[str(p)] = {"kind": f"items{len(soup.ingredients)}", "t": state.timestep}
            else:
                seen[str(p)] = {"kind": "empty", "t": state.timestep}
    return {
        "me": me,
        "pos": tuple(player.position),
        "orient": tuple(player.orientation),
        "held": agent_held(state, me),
        "tiles": tiles,
        "partner_visible": partner_visible,
        "seen_pots": seen,
    }


def pot_display(k, v, now=None):
    d = _kind_of(v)
    t = v.get("t") if isinstance(v, dict) else None
    txt = f"锅{k}:{d}"
    if d == "cooking" and now is not None and t is not None and now - t >= COOK_GUESS:
        txt += "(按时间推算可能已好,建议去确认PICKUP)"
    elif t is not None:
        txt += f"(上次确认于t={t})"
    return txt


def obs_text(o, now=None):
    lines = [
        f"你在{o['pos']}，面朝{o['orient']}，手拿：{o['held'] or '无'}",
        "你看到(3×3，以你为中心)：",
    ]
    lines += ["  " + r for r in o["tiles"]]
    lines.append(f"伙伴{'在视线内:' + str(o['partner_visible']) if o['partner_visible'] else '不在视线内'}")
    if o["seen_pots"]:
        lines.append("你记忆中的锅状态：" + ", ".join(pot_display(k, v, now) for k, v in sorted(o["seen_pots"].items())))
    return "\n".join(lines)


MAP_PRIOR = """厨房布局先验（你熟悉自己的厨房）：
行y=0: XXPXPXX（锅在(2,0)和(4,0)，中央(3,0)是墙）
行y=1: O     O（洋葱台在(0,1)和(6,1)）
行y=2: X 1 2 X（走道）
行y=3: X D S X（盘台(2,3)，出餐口(4,3)）
行y=4: XXXXXXX
坐标(x,y)；你只能看到周围3×3与记忆中的锅态；做汤：取洋葱×3→锅满后空手interact点火→煮约20tick→先拿盘再取汤→出餐口交付(+20)。
"""
