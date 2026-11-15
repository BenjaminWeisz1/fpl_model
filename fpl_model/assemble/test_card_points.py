"""Card points (CHANGELOG §175). Run: python -m unittest test_card_points"""
import unittest

import numpy as np
import pandas as pd

from fpl_model.card_points import (
    SUSPENSION_THETA,
    K,
    decayed_multiplier,
    expected_card_points,
    fixture_yellows_90,
    forecast_cameo_share,
)


class CardPoints(unittest.TestCase):
    def test_points_scale_with_minutes(self):
        self.assertAlmostEqual(float(expected_card_points(0.2, 0.01, 90)), -0.23)
        self.assertAlmostEqual(float(expected_card_points(0.2, 0.01, 45)), -0.115)
        self.assertEqual(float(expected_card_points(0.2, 0.01, 0)), 0.0)

    def test_new_player_gets_the_position_rate(self):
        d = pd.DataFrame({"Player": ["a"], "y": [1.0], "e": [0.2]})
        self.assertEqual(decayed_multiplier(d, "y", "e", K)[0], 1.0)

    def test_multiplier_uses_only_earlier_matches_and_shrinks(self):
        d = pd.DataFrame({"Player": ["a", "a", "a"], "y": [1.0, 1.0, 0.0], "e": [0.2, 0.2, 0.2]})
        m, state = decayed_multiplier(d, "y", "e", K, final=True)
        self.assertEqual(m[0], 1.0)
        self.assertAlmostEqual(m[1], (1 + K) / (0.2 + K))
        self.assertAlmostEqual(m[2], (2 + K) / (0.4 + K))
        self.assertAlmostEqual(state["a"], (2 + K) / (0.6 + K))
        self.assertTrue(np.all(m < 2))

    def rates(self, season_yellows, share=np.nan):
        return pd.DataFrame({"Yellows_90": [0.2] * len(season_yellows), "Yellows_90_Start": [0.15] * len(season_yellows),
                             "Yellows_90_Cameo": [0.3] * len(season_yellows), "Cameo_Share": [share] * len(season_yellows),
                             "Season_Yellows": season_yellows})

    def test_role_mix_and_fallback(self):
        r = self.rates([0, 0])
        y = fixture_yellows_90(r, [10, 10], [0.25, np.nan])
        self.assertAlmostEqual(y[0], 0.75 * 0.15 + 0.25 * 0.3)
        self.assertAlmostEqual(y[1], 0.2)

    def test_suspension_threat_only_before_cut_off(self):
        r = self.rates([4, 4, 9, 9, 14, 3])
        y = fixture_yellows_90(r, [19, 20, 32, 33, 37, 10], [0.0] * 6)
        t = SUSPENSION_THETA * 0.15
        np.testing.assert_allclose(y, [t, 0.15, t, 0.15, t, 0.15])

    def test_forecast_cameo_share(self):
        s = forecast_cameo_share([0.9, 0.5, 0.0], [0.8, 0.0, 0.0], [80.0, 9.0, 0.0], 18.0)
        self.assertAlmostEqual(s[0], 0.1 * 18 / 80)
        self.assertAlmostEqual(s[1], 1.0)
        self.assertTrue(np.isnan(s[2]))


if __name__ == "__main__":
    unittest.main()
