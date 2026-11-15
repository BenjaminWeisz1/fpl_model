"""xMins clock scale (CHANGELOG §177). Run: python -m unittest test_xmins_clock"""
import unittest

import pandas as pd

import build_simulation_features as sf


def rows(team_totals):
    return pd.DataFrame([{"Team": t, "Round": r, "xMins_Raw": v} for (t, r), v in team_totals.items()])


class ClockScale(unittest.TestCase):
    def test_scale_is_recorded_minutes_over_the_typical_team_total(self):
        self.assertEqual(sf.RECORDED_TEAM_MINUTES, 990)
        self.assertAlmostEqual(sf.recorded_clock_scale(rows({("a", 1): 1100, ("b", 1): 1100, ("c", 1): 1100})), 0.9)

    def test_a_double_gameweek_does_not_move_it(self):
        totals = {("a", 1): 1100, ("b", 1): 1100, ("c", 1): 1100, ("a", 2): 2200, ("b", 2): 1100}
        self.assertAlmostEqual(sf.recorded_clock_scale(rows(totals)), 0.9)

    def test_it_follows_a_change_in_the_clock(self):
        self.assertAlmostEqual(sf.recorded_clock_scale(rows({("a", 1): 990, ("b", 1): 990})), 1.0)


if __name__ == "__main__":
    unittest.main()
