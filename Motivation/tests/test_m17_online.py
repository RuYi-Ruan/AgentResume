import unittest


class M17OnlineEligibilityTest(unittest.TestCase):
    def test_requires_a_fresh_visible_growth_decision(self):
        from overcooked_ai_py.mdp.overcooked_mdp import SoupState
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2
        from scripts.m17_online_experiment import strict_growth_candidate

        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        state = world.env.state
        # Put both players together in local view and make one pot cook while
        # at least one other pot remains able to accept an onion.
        state.players[0].update_pos_and_or((1, 1), (1, 0))
        state.players[1].update_pos_and_or((1, 2), (1, 0))
        state.add_object(SoupState.get_soup(sorted(world.grid.pot_locs)[0], num_onions=3, cooking_tick=5))
        post = ((1, 0), "FETCH", (0, 1), {"decision_id": 1})
        pre = ((1, 0), "PRE", (2, 0), {"decision_id": 1})
        self.assertTrue(strict_growth_candidate(state, world.grid, post, pre))

        # Continuing a macro intent is never queried, even if the remaining
        # fields look like a strict pair.
        post_without_new_decision = (post[0], post[1], post[2], None)
        self.assertFalse(
            strict_growth_candidate(state, world.grid, post_without_new_decision, pre)
        )


if __name__ == "__main__":
    unittest.main()
