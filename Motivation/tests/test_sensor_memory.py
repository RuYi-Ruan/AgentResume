import unittest


class SensorMemoryTest(unittest.TestCase):
    def test_pot_memory_records_visible_provenance_and_timestamp(self):
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2
        from ocres.sensor import visible_state

        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        pot = sorted(world.grid.pot_locs)[0]
        stand, _ = world.grid.interact_stand_cells(pot)[0]
        world.env.state.players[1].update_pos_and_or(stand, (0, -1))
        observation = visible_state(world.env.state, world.grid, 1, {})
        memory = observation["seen_pots"][str(pot)]
        self.assertEqual(memory["source"], "visible")
        self.assertEqual(memory["t"], world.env.state.timestep)


if __name__ == "__main__":
    unittest.main()
