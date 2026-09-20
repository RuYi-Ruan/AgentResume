import unittest


class M16LayoutTest(unittest.TestCase):
    def test_layout_dimensions_features_and_reachability(self):
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V1

        self.assertEqual({len(row) for row in AMBIGUOUS_KITCHEN_V1}, {11})
        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V1, horizon=20)
        self.assertEqual(len(world.grid.pot_locs), 3)
        self.assertEqual(len(world.grid.onion_locs), 2)
        self.assertEqual(len(world.grid.dish_locs), 3)
        self.assertEqual(len(world.grid.serve_locs), 4)

    def test_v2_interleaves_onion_and_dish_facilities(self):
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2

        self.assertEqual({len(row) for row in AMBIGUOUS_KITCHEN_V2}, {11})
        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        self.assertEqual(len(world.grid.pot_locs), 3)
        self.assertEqual(len(world.grid.onion_locs), 2)
        self.assertEqual(len(world.grid.dish_locs), 3)
        self.assertEqual(len(world.grid.serve_locs), 4)
        self.assertTrue(all(world.grid.bfs(a, b) is not None for a in world.grid.passable for b in world.grid.passable))
        start = tuple(world.env.state.players[0].position)
        for feature in world.grid.pot_locs + world.grid.onion_locs + world.grid.dish_locs + world.grid.serve_locs:
            self.assertTrue(
                any(world.grid.bfs(start, stand, ()) is not None for stand, _ in world.grid.interact_stand_cells(feature))
            )


if __name__ == "__main__":
    unittest.main()
