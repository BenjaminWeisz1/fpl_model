import unittest

import numpy as np
import pandas as pd

from fpl_model.bonus.build_residual_bps_baseline import add_known_bps


def rows(**overrides):
    base = {
        "Pos": ["D", "D", "M", "F", "G", "G"],
        "Minutes": [90, 30, 90, 90, 90, 45],
        "NPG": [0] * 6, "PK": [0] * 6, "PKatt": [0] * 6, "Assists": [0] * 6,
        "Clean_Sheet": [0] * 6,
        "Goals_Conceded": [2, 1, 2, 2, 2, 1],
        "CBI": [7, 3, 4, 2, 2, 1],
        "Tkl": [3, 1, 2, 1, 0, 0],
        "Recov": [5, 2, 6, 3, 4, 1],
        "Saves": [0, 0, 0, 0, 3, 2],
        "Penalties_Saved": [0, 0, 0, 0, 1, 0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


class BpsAccountingTests(unittest.TestCase):
    def test_goals_conceded_for_defenders_and_keepers_at_any_minutes(self):
        out = add_known_bps(rows(), legacy=False)
        np.testing.assert_array_equal(out["gc_bps"], [-8, -4, 0, 0, -8, -4])

    def test_keeper_saves_only_on_60_plus_rows_at_scale(self):
        out = add_known_bps(rows(), legacy=False, save_scale=1.0)
        np.testing.assert_array_equal(out["save_bps"], [0, 0, 0, 0, 3 * 2 + 7, 0])
        half = add_known_bps(rows(), legacy=False, save_scale=0.5)
        self.assertAlmostEqual(half["save_bps"].iloc[4], 6.5)

    def test_defensive_term_is_defcon_count_div_3(self):
        out = add_known_bps(rows(), legacy=False)
        # D: CBI+Tkl; M/F: CBI+Tkl+Recov; G: none (no simulated defcon term)
        np.testing.assert_array_equal(out["defensive_bps"], [10 // 3, 4 // 3, 12 // 3, 6 // 3, 0, 0])

    def test_legacy_matches_pre_change_formula(self):
        frame = rows()
        out = add_known_bps(frame, legacy=True)
        expected = (np.where(frame["Minutes"] >= 60, 6, 3)
                    + frame["CBI"] // 3 + frame["Tkl"] * 2 + frame["Recov"] // 3)
        np.testing.assert_array_equal(out["known_bps"], expected)


if __name__ == "__main__":
    unittest.main()
