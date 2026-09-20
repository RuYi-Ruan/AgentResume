import unittest

from ocres.m18_d_sampling import choose_natural_d


class DSamplingTests(unittest.TestCase):
    def test_exact_unique_but_family_variants_allowed(self):
        rows = [{"source_seed": i//2, "source_t": i, "event_hash": f"h{i}",
                 "family": "similar"} for i in range(8)]
        selected, _ = choose_natural_d(rows, 6, 2, 18933)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({r["event_hash"] for r in selected}), 6)
        self.assertEqual({r["family"] for r in selected}, {"similar"})
        self.assertTrue(all(sum(r["source_seed"] == seed for r in selected) <= 2
                            for seed in range(4)))

    def test_literal_public_duplicate_not_reused(self):
        rows = [{"source_seed": i, "source_t": 0, "event_hash": "same" if i < 3 else f"h{i}",
                 "family": f"f{i}"} for i in range(6)]
        selected, _ = choose_natural_d(rows, 6, 2, 18933)
        self.assertEqual(len(selected), 4)
        self.assertEqual(len({r["event_hash"] for r in selected}), 4)


if __name__ == "__main__":
    unittest.main()
