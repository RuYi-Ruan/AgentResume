import sys
sys.path.insert(0, "src")
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
for name in ["counter_circuit", "forced_coordination", "cramped_room_o_3orders"]:
    try:
        gw = OvercookedGridworld.from_layout_name(name)
        print(f"== {name} shape={gw.shape} ==")
        for row in gw.terrain_mtx:
            print("   ", "".join(str(c) for c in row))
        print("   pots:", gw.get_pot_locations())
        print("   onions:", gw.get_onion_dispenser_locations())
        print("   dishes:", gw.get_dish_dispenser_locations())
        print("   serve:", gw.get_serving_locations())
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        env = OvercookedEnv.from_mdp(gw, horizon=10)
        print("   start:", env.state.players_pos_and_or, "orders:", [tuple(o.ingredients) for o in env.state._all_orders])
        env.reset()
    except Exception as e:
        print(name, "ERR", repr(e))
