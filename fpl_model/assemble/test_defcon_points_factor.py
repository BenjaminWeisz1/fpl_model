"""§159: the defender defcon points factor touches defenders only and reverts with its switch."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]

from fpl_model.defcon_scaling import DEFENDER_POINTS_FACTOR, nb_sf, points_mu90, points_prob_at_90, prob_at_minutes


class DefenderPointsFactorTests(unittest.TestCase):
    def test_only_defenders_are_scaled(self):
        result = points_mu90([4.0, 4.0, 4.0, 4.0], ["D", "M", "F", "G"])
        np.testing.assert_allclose(result, [4.0 * DEFENDER_POINTS_FACTOR, 4.0, 4.0, 4.0])

    def test_defender_probability_rises(self):
        before = prob_at_minutes(np.array([7.0]), np.array([90.0]), np.array([10]))
        after = prob_at_minutes(points_mu90([7.0], ["D"]), np.array([90.0]), np.array([10]))
        self.assertGreater(after[0], before[0])

    def test_rating_matches_the_points_path(self):
        prob = float(nb_sf(8.0, 9))
        expected = float(nb_sf(8.0 * DEFENDER_POINTS_FACTOR, 9))
        self.assertAlmostEqual(float(points_prob_at_90([prob], ["D"])[0]), expected, places=3)

    def test_rating_leaves_other_positions_and_blanks(self):
        result = points_prob_at_90([0.3, 0.3, np.nan, np.nan], ["M", "F", "G", "D"])
        np.testing.assert_array_equal(result, [0.3, 0.3, np.nan, np.nan])

    def test_switch_reverts(self):
        code = "from fpl_model.defcon_scaling import points_mu90; print(float(points_mu90([4.0], ['D'])[0]))"
        env = {**os.environ, "DEFCON_POINTS_CALIBRATION": "0"}
        out = subprocess.run([sys.executable, "-c", code], cwd=SCRIPTS.parent, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(float(out.stdout), 4.0)


if __name__ == "__main__":
    unittest.main()
