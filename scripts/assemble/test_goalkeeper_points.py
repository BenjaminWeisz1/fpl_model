import unittest
import sys
from pathlib import Path

import numpy as np
from scipy.stats import poisson

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from build_predicted_points import (
    PENALTY_GOALS_FACTOR,
    SAVE_FIXTURE_BETA,
    expected_clean_sheet_points,
    expected_goalkeeper_goals_conceded_points,
    expected_goalkeeper_save_points,
    expected_goals_conceded_deduction,
    expected_points_per_three_saves,
    save_fixture_multiplier,
    save_fixture_reference,
)


class GoalkeeperPointsTests(unittest.TestCase):
    def test_expected_save_points_matches_poisson_sum(self):
        means = np.array([0.0, 3.0, 12.0])
        counts = np.arange(100)
        brute_force = np.array([
            np.dot(counts // 3, poisson.pmf(counts, mean))
            for mean in means
        ])
        np.testing.assert_allclose(
            expected_points_per_three_saves(means), brute_force, atol=1e-10
        )

    def test_expected_goals_conceded_deduction_matches_poisson_sum(self):
        lam = np.array([0.0, 0.5, 2.0, 5.0])
        counts = np.arange(100)
        brute_force = np.array([
            np.dot(counts // 2, poisson.pmf(counts, mean))
            for mean in lam
        ])
        np.testing.assert_allclose(
            expected_goals_conceded_deduction(lam), brute_force, atol=1e-10
        )

    def test_goalkeeper_exposure_scales_for_absence_cameo_and_full_match(self):
        minutes = np.array([0.0, 15.0, 90.0])
        saves = expected_goalkeeper_save_points(3.0, minutes)
        goals_conceded = expected_goalkeeper_goals_conceded_points(2.0, minutes)

        self.assertEqual(saves[0], 0)
        self.assertEqual(goals_conceded[0], 0)
        self.assertGreater(saves[1], 0)
        self.assertGreater(goals_conceded[1], goals_conceded[2])
        self.assertAlmostEqual(goals_conceded[2], -expected_goals_conceded_deduction(2.0))

    def test_save_fixture_multiplier_keeps_the_pooled_level(self):
        history = pd.DataFrame({
            "Season": [2024, 2025, 2026],
            "Home_Prediction": [9.0, 1.0, 2.0],
            "Away_Prediction": [9.0, 1.5, 0.5],
        })
        reference = save_fixture_reference(history)
        forecasts = np.array([1.0, 2.0, 1.5, 0.5])  # the 2024 row is outside the window
        multipliers = save_fixture_multiplier(forecasts * PENALTY_GOALS_FACTOR, reference)
        self.assertAlmostEqual(multipliers.mean(), 1.0)
        self.assertAlmostEqual(multipliers[1] / multipliers[0], 2 ** SAVE_FIXTURE_BETA)

    def test_clean_sheet_points_require_the_sixty_minute_probability(self):
        points = expected_clean_sheet_points(
            [0.5, 0.5, 0.5], [0.0, 0.5, 1.0], [4.0, 4.0, 4.0]
        )
        np.testing.assert_allclose(points, [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
