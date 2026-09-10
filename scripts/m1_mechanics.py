"""M1 mechanics probe v3: teleport-based pure interaction semantics.
Goal: pin down start-cooking trigger, cook duration, soup pickup (dish?),
and delivery reward without walking overhead."""
import sys

sys.path.insert(0, "src")
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402

STAY = (0, 0)
gw = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(gw, horizon=2000)
POT, ONION_L, DISH, SERVE = (2, 0), (0, 1), (1, 3), (3, 3)
STANDS = {POT: ((2, 1), (0, -1)), ONION_L: ((1, 1), (-1, 0)), DISH: ((1, 2), (0, 1)), SERVE: ((3, 2), (0, 1))}


def tele(i, pos, face):
    s = env.state
    s.players[i].update_pos_and_or(pos, face)


def act(a0, a1=STAY):
    s2, r, done, _ = env.step((a0, a1))
    return s2, r


def interact(target, i=0):
    stand, face = STANDS[target]
    tele(i, stand, face)
    s, r = act("interact")
    return s, r


def held(i=0):
    o = env.state.players[i].held_object
    return o.name if o else None


def pot():
    return env.state.get_object(POT) if env.state.has_object(POT) else None


def show(tag):
    s = env.state
    soup = pot()
    if soup is None:
        print(f"  t={s.timestep} {tag}: held0={held()} | pot empty")
        return
    try:
        rem = soup.cook_time_remaining
    except Exception:  # noqa: BLE001
        rem = "?"
    print(f"  t={s.timestep} {tag}: held0={held()} | soup[ingr={len(soup.ingredients)} cooking={soup.is_cooking} ready={soup.is_ready} rem={rem}]")


# 1) three onions in
for k in range(3):
    s, r = interact(ONION_L)
    s, r = interact(POT)
    show(f"place{k+1} r={r}")
# 2) try start cooking by empty-hand interact at pot
s, r = interact(POT)
show(f"start-cook interact r={r}")
print("cook_time attr:", pot().cook_time if pot() else None)
# 3) poll without moving (partner STAY at spawn, agent stays at pot stand)
soup = pot()
for tick in range(int((soup.cook_time or 30) * 2) + 10):
    s, r = act(STAY)
    soup = pot()
    if soup and soup.is_ready:
        print(f"READY at t={s.timestep}; reward so far {r}")
        break
show("ready check")
# 4) pickup empty-hand
s, r = interact(POT)
show(f"pickup attempt r={r}")
if held() is None:
    # try dish first
    s, r = interact(DISH)
    show(f"dish pickup r={r}")
    s, r = interact(POT)
    show(f"pot pickup with dish r={r}")
# 5) deliver
if held() is not None:
    s, r = interact(SERVE)
    show(f"serve r={r}")
    print("delivery events:", env.game_stats.get("soup_delivery"))
print("sparse cumulative:", env.game_stats["cumulative_sparse_rewards_by_agent"])
