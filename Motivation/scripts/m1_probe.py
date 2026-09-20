"""M1 step 1: geometry + planner/MLAM API + reward semantics probe."""
import inspect

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402

gw = OvercookedGridworld.from_layout_name("cramped_room")
env = OvercookedEnv.from_mdp(gw, horizon=400)
mp = env.mp
mlam = env.mlam

print("== geometry ==")
for name in [
    "get_pot_locations",
    "get_onion_dispenser_locations",
    "get_dish_dispenser_locations",
    "get_serving_locations",
    "get_counter_locations",
    "get_start_positions",
]:
    f = getattr(gw, name, None)
    if f:
        try:
            print(f"{name}: {f()}")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: ERR {e!r}")

print("terrain rows:")
for row in gw.terrain_mtx:
    print("  ", "".join(str(c) for c in row))

print("\n== state/object semantics ==")
print("env_params:", env.env_params)
s = env.state
print("state:", s)
print("players[0]:", s.players[0])
print("player fields:", [n for n in dir(s.players[0]) if not n.startswith("_")][:30])
print("player pos:", s.player_positions, "orients:", s.player_orientations)

print("\n== mp API ==")
print("mp methods:", [n for n in dir(mp) if not n.startswith("_")][:60])
print("mp class:", type(mp).__name__)
print("mp sig fields:", {k: v for k, v in vars(mp).items() if isinstance(v, (int, float, str, bool))})

print("\n== mlam API ==")
print("mlam methods:", [n for n in dir(mlam) if not n.startswith("_")][:80])
print("mlam class:", type(mlam).__name__)

print("\n== reward on delivery ==")
try:
    print("reward_shaping params:", gw.env_params if hasattr(gw, "env_params") else "n/a")
except Exception as e:  # noqa: BLE001
    print(e)
# cooking timing
try:
    print("cooking params:", {k: v for k, v in vars(gw).items() if "cook" in k.lower() or "num" in k.lower()})
except Exception as e:  # noqa: BLE001
    print("vars:", e)
