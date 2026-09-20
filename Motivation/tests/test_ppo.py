import unittest


def ppo_api():
    from ocres import ppo, trainable
    import numpy as np
    import torch

    return ppo, trainable, np, torch


class PPOTest(unittest.TestCase):
    def test_value_network_shape(self):
        ppo, _, _, torch = ppo_api()
        network = ppo.ValueNetwork(5, hidden_dim=8)
        self.assertEqual(tuple(network(torch.zeros(3, 5)).shape), (3,))

    def test_small_ppo_update_is_finite(self):
        ppo, trainable, np, torch = ppo_api()
        actor = trainable.MacroIntentPolicy(4, hidden_dim=8)
        critic = ppo.ValueNetwork(4, hidden_dim=8)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
        records = []
        for index in range(8):
            records.append(
                {
                    "observation": np.asarray([index / 8, 0, 1, 0], dtype=np.float32),
                    "mask": np.ones(len(trainable.INTENTS), dtype=np.bool_),
                    "action": index % len(trainable.INTENTS),
                    "old_log_prob": -2.0,
                    "return": float(index % 2),
                    "advantage": float(index % 2) - 0.5,
                }
            )
        stats = ppo.ppo_update(
            actor, critic, optimizer, records, torch.device("cpu"), update_epochs=1, batch_size=8
        )
        self.assertTrue(all(np.isfinite(value) for value in stats.values()))


if __name__ == "__main__":
    unittest.main()

