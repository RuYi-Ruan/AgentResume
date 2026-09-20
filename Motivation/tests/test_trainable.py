import pathlib
import tempfile
import unittest
from types import SimpleNamespace


def trainable_api():
    # Delay the official PyTorch wheel import until the NumPy-only tests have
    # initialized Conda MKL. unittest imports all test modules before running
    # any methods, so a module-level import would reverse that safe order.
    from ocres import trainable
    import numpy as np
    import torch

    return trainable, np, torch


class TrainablePolicyTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "t": 10,
            "p0": (5, 3),
            "p1": (1, 1),
            "held0": None,
            "held1": "dish",
            "pot": "[((2, 0), 'empty'), ((4, 0), 'cooking')]",
        }

    def test_observation_has_fixed_shape_and_no_label_input(self):
        api, np, _ = trainable_api()
        spec = api.ObservationSpec()
        encoded = api.encode_log_observation(self.row, deliveries=2, horizon=700, spec=spec)
        self.assertEqual(encoded.shape, (spec.dim,))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertAlmostEqual(float(encoded.sum()), 6.0 + 10 / 700 + 2 / 20, places=5)

    def test_physical_action_mask(self):
        api, _, _ = trainable_api()
        mask = api.legal_intent_mask(self.row)
        self.assertTrue(mask[api.INTENT_ID["FETCH"]])
        self.assertTrue(mask[api.INTENT_ID["GET_DISH"]])
        self.assertTrue(mask[api.INTENT_ID["PRE"]])
        self.assertTrue(mask[api.INTENT_ID["HOLD"]])
        self.assertFalse(mask[api.INTENT_ID["DELIVER"]])

    def test_masked_policy_never_chooses_an_illegal_intent(self):
        api, _, torch = trainable_api()
        spec = api.ObservationSpec()
        model = api.MacroIntentPolicy(spec.dim, hidden_dim=8)
        x = torch.from_numpy(api.encode_log_observation(self.row, spec=spec)).unsqueeze(0)
        mask = torch.from_numpy(api.legal_intent_mask(self.row)).unsqueeze(0)
        choice = int(model.choose(x, mask).item())
        self.assertTrue(bool(mask[0, choice]))

    def test_checkpoint_round_trip(self):
        api, _, torch = trainable_api()
        spec = api.ObservationSpec()
        model = api.MacroIntentPolicy(spec.dim, hidden_dim=8)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "policy.pt"
            api.save_policy_checkpoint(path, model, spec, extra={"seed": 3})
            restored, restored_spec, extra = api.load_policy_checkpoint(path)
        self.assertEqual(restored_spec, spec)
        self.assertEqual(extra["seed"], 3)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_historical_expert_labels_are_legal(self):
        api, _, _ = trainable_api()
        paths = []
        for level_dir in sorted(pathlib.Path("data/twopot_v1").glob("level_*")):
            first = next(iter(sorted(level_dir.glob("ep_seed*.npz"))), None)
            if first is not None:
                paths.append(first)
        if not paths:
            self.skipTest("historical NPZ data is not present")
        dataset = api.HistoricalIntentDataset(paths)
        self.assertGreater(len(dataset), 0)

    def test_yield_action_vacates_a_choke_point(self):
        api, _, _ = trainable_api()

        class DummyGrid:
            passable = {(3, 2), (4, 2), (5, 2), (4, 1)}
            onion_locs = []
            pot_locs = []
            dish_locs = []
            serve_locs = []

            @staticmethod
            def interact_stand_cells(_):
                return []

        model = api.MacroIntentPolicy(api.ObservationSpec().dim, hidden_dim=8)
        agent = api.TrainableMacroAgent(DummyGrid(), 0, model)
        state = SimpleNamespace(
            players=[SimpleNamespace(position=(4, 2)), SimpleNamespace(position=(4, 1))]
        )
        self.assertEqual(agent._yield_action(state), (-1, 0))


if __name__ == "__main__":
    unittest.main()
