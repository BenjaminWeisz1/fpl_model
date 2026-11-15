import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import simulate_bonus
from test_pass_bands import match_players


def near_tied_match():
    """Four midfielders, no randomness: residual BPS 10.2, 9.8, 5, 4."""
    players = match_players(False)
    players["ResidualBPS_Mean"] = [10.2, 9.8, 5.0, 4.0]
    return players


class IntegerBpsTests(unittest.TestCase):
    def test_official_tie_rules(self):
        cases = {(20, 20, 10, 5): [3, 3, 1, 0], (20, 15, 15, 5): [3, 2, 2, 0],
                 (20, 15, 10, 10): [3, 2, 1, 1], (20, 15, 10, 5): [3, 2, 1, 0]}
        for bps, expected in cases.items():
            got = simulate_bonus.assign_bonus(np.array(bps, float)[:, None])[:, 0]
            np.testing.assert_array_equal(got, expected, err_msg=str(bps))

    def test_rounding_creates_ties(self):
        simulate_bonus.rng = np.random.default_rng(0)
        with patch.object(simulate_bonus, "INTEGER_BPS", True):
            bonus = simulate_bonus.simulate_match(near_tied_match(), 0.0, 0.0, "A", 20)[0]
        np.testing.assert_array_equal(bonus, [3, 3, 1, 0])

    def test_switch_off_restores_continuous_totals(self):
        simulate_bonus.rng = np.random.default_rng(0)
        with patch.object(simulate_bonus, "INTEGER_BPS", False):
            bonus = simulate_bonus.simulate_match(near_tied_match(), 0.0, 0.0, "A", 20)[0]
        np.testing.assert_array_equal(bonus, [3, 2, 1, 0])


if __name__ == "__main__":
    unittest.main()
