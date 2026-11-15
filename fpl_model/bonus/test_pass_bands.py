import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


from fpl_model.bonus import pass_bands as pb
from fpl_model.bonus import simulate_bonus
from fpl_model.bonus.build_residual_bps_baseline import aggregate_residuals


def match_players(p6_first):
    base = {"Pos": "M", "Team": "A", "P1": 1.0, "P60": 1.0, "NPG90_scaled": 0.0, "Ast90_scaled": 0.0,
            "PenAttempts": 0.0, "Defcon_Mu90": 0.0, "ResidualBPS_Mean": 10.0, "ResidualBPS_Std": 0.0,
            "PB_P0": 1.0, "PB_P2": 0.0, "PB_P4": 0.0, "PB_P6": 0.0}
    rows = [dict(base) for _ in range(4)]
    if p6_first:
        rows[0].update(PB_P0=0.0, PB_P6=1.0)
    return pd.DataFrame(rows)


class PassBandTests(unittest.TestCase):
    def test_band_thresholds(self):
        total = [29, 30, 30, 30, 30, 0]
        accurate = [29, 20, 21, 24, 27, 0]
        np.testing.assert_array_equal(pb.pass_band(total, accurate), [0, 0, 2, 4, 6, 0])

    def test_tilt_moves_mean_and_zero_shift_is_identity(self):
        p = np.array(pb.PRIOR["D"])
        np.testing.assert_allclose(pb.tilt(p, 0.0), p, atol=1e-9)
        self.assertAlmostEqual(pb.tilt(p, 0.5) @ pb.BANDS, p @ pb.BANDS + 0.5, places=6)

    def test_no_history_gives_position_prior(self):
        np.testing.assert_allclose(pb.shrunk(np.zeros(4), "F"), pb.PRIOR["F"])

    def test_active_follows_sofascore_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            self.assertFalse(pb.active(data)[0])
            (data / "sofascore_status.json").write_text(json.dumps({"usable_for_model": True}))
            self.assertTrue(pb.active(data)[0])
            (data / "sofascore_status.json").write_text(json.dumps({"usable_for_model": False, "lag_rounds": 3}))
            self.assertFalse(pb.active(data)[0])

    def test_simulated_pass_band_decides_otherwise_tied_match(self):
        simulate_bonus.rng = np.random.default_rng(0)
        bonus, _, _ = simulate_bonus.simulate_match(match_players(True), 0.0, 0.0, "A", 50)
        self.assertEqual(bonus[0], 3.0)

    def test_inactive_pass_bands_leave_random_stream_unchanged(self):
        simulate_bonus.rng = np.random.default_rng(0)
        first = simulate_bonus.simulate_match(match_players(False), 1.0, 1.0, "A", 50)[0]
        simulate_bonus.rng = np.random.default_rng(0)
        second = simulate_bonus.simulate_match(match_players(False).drop(columns=[]), 1.0, 1.0, "A", 50)[0]
        np.testing.assert_array_equal(first, second)

    def test_position_specific_shrinkage(self):
        frame = pd.DataFrame({"Player": ["a"] * 4 + ["b"] * 4, "Pos": ["D"] * 4 + ["M"] * 4,
                              "residual": [4.0] * 4 + [8.0] * 4})
        frame = pd.concat([frame, pd.DataFrame({"Player": ["c", "d"], "Pos": ["D", "M"], "residual": [0.0, 0.0]})])
        out, _ = aggregate_residuals(frame, {"D": 4, "M": 0})
        means = out.set_index("Player")["ResidualBPS_Mean"]
        self.assertAlmostEqual(means["b"], 8.0)
        self.assertLess(means["a"], 4.0)


if __name__ == "__main__":
    unittest.main()
