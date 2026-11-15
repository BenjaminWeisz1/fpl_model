"""Tests for the §106 league-level adjustment (defender_model.py). Run: python -m fpl_model.defcon.test_level_adjust"""
import unittest

import numpy as np
import pandas as pd

from fpl_model.defcon.defender_model import (
    CenteredIsotonic,
    apply_level_factors,
    level_adjust_features,
    level_factors,
    season_levels,
)


def frame():
    return pd.DataFrame({
        "Player": ["A", "A", "A", "A", "B"],
        "Season": [2020, 2020, 2021, 2022, 2022],
        "Minutes": [90, 30, 90, 90, 90],
        "Per90Stat": [10.0, 99.0, 8.0, 6.0, 8.0],
        "AvgPer90": [5.0, 5.0, 5.0, 5.0, 5.0], "AvgRaw": [4.0] * 5, "RecentEWPer90": [6.0] * 5,
    })


class LevelAdjust(unittest.TestCase):
    def test_levels_exclude_cameos(self):
        self.assertEqual(season_levels(frame()), {2020: 10.0, 2021: 8.0, 2022: 7.0})

    def test_factors_by_hand(self):
        f = frame()
        lv = season_levels(f)
        plain, ew, latest = level_factors(f, lv, 2023, half_life=4.0)
        # first row: no earlier qualifying match; second: same-season match gets L(2019), absent -> 1
        self.assertEqual(plain[0], 1.0)
        self.assertEqual(plain[1], 1.0)
        # 2021 row: one earlier qualifying match in 2020 (level 10), current level L(2020) = 10 -> 1
        self.assertAlmostEqual(plain[2], 1.0)
        # 2022 row: earlier matches 2020 (10), 2021 (8); current L(2021) = 8 -> 8 / 9
        self.assertAlmostEqual(plain[3], 8 / 9)
        d = 2 ** (-1 / 4)
        self.assertAlmostEqual(ew[3], 8 * (d + 1) / (d * 10 + 8))
        # cameo (30') does not count as an earlier qualifying match
        # forecast 2023: matches 2020, 2021, 2022 (levels 10, 8, 7), current L(2022) = 7
        self.assertAlmostEqual(latest["A"][0], 7 / ((10 + 8 + 7) / 3))
        # B has no earlier match in 2022 (factor 1) and one 2022 match for the 2023 forecast
        self.assertEqual(plain[4], 1.0)
        self.assertAlmostEqual(latest["B"][0], 1.0)

    def test_apply_only_activity_columns(self):
        f = frame()
        out = apply_level_factors(f, np.full(5, 0.5), np.full(5, 2.0))
        self.assertTrue((out["AvgPer90"] == 2.5).all() and (out["AvgRaw"] == 2.0).all())
        self.assertTrue((out["RecentEWPer90"] == 12.0).all())
        self.assertTrue(out["Per90Stat"].equals(f["Per90Stat"]))
        feat = level_adjust_features({"AvgPer90": 4.0, "MatchesPlayed": 9}, (0.5, 2.0))
        self.assertEqual(feat, {"AvgPer90": 2.0, "MatchesPlayed": 9})


class CenteredIsotonicTest(unittest.TestCase):
    def test_interpolates_between_block_centres(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        y = np.array([0, 1, 0, 1, 1, 1])
        c = CenteredIsotonic().fit(x, y)
        # isotonic blocks: {0.1}=0, {0.2,0.3}=0.5, {0.4,0.5,0.6}=1 -> centres 0.1, 0.25, 0.5
        np.testing.assert_allclose(c.x_, [0.1, 0.25, 0.5])
        np.testing.assert_allclose(c.y_, [0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(c.predict([0.2])[0]), 1 / 3)
        self.assertEqual(float(c.predict([0.05])[0]), 0.0)   # clipped at the ends
        p = c.predict(np.linspace(0, 1, 50))
        self.assertTrue((np.diff(p) >= 0).all())


if __name__ == "__main__":
    unittest.main()
