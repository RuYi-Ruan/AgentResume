import inspect
import unittest

from ocres.m18_b import run_b_episode


class BProtocolTests(unittest.TestCase):
    def test_predictor_contract_has_no_truth_parameter(self):
        source = inspect.getsource(run_b_episode)
        self.assertIn("predict(observed, condition, seed, query_index)", source)
        self.assertNotIn("predict(observed, truth", source)


if __name__ == "__main__":
    unittest.main()
