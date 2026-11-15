"""Tests for the Solio comparison's derived values. Run: python -m unittest test_compare_solio (from this folder)."""
import unittest

import pandas as pd

from fpl_model.solio import compare_solio as c


def solio_team(team="LIV", npg=1.7, assists=1.75):
    """Four players whose G/90 and A/90 follow Solio's identity; the first also takes penalties (+0.2 G/90)."""
    rows = []
    for name, mins, gs, as_, pen in (("A", 90, 0.4, 0.2, 0.2), ("B", 80, 0.3, 0.3, 0.0),
                                     ("C", 70, 0.2, 0.3, 0.0), ("D", 10, 0.1, 0.2, 0.0)):
        g90 = gs * npg * 90 / mins + pen
        a90 = as_ * assists * 90 / mins
        rows.append({"Web_Name": name, "Team_Short": team, "Solio_Mins": mins, "Solio_G90": g90 if mins >= 18 else 0.0,
                     "Solio_A90": a90 if mins >= 18 else 0.0, "Solio_Goal_Share": gs, "Solio_Assist_Share": as_,
                     "Solio_Per90_Suppressed": mins < 18, "Solio_Team_CS": 0.3})
    return pd.DataFrame(rows)


class SolioDerived(unittest.TestCase):
    def test_team_rates_ignore_the_penalty_taker(self):
        t = c.solio_team_rates(solio_team())
        self.assertAlmostEqual(t.loc["LIV", "Solio_Team_NPG"], 1.7, places=6)
        self.assertAlmostEqual(t.loc["LIV", "Solio_Team_Assists"], 1.75, places=6)
        self.assertGreater(t.loc["LIV", "Solio_Team_NPG_Max_Over"], 0.1)  # the taker sits above

    def test_goal_split_recovers_penalties(self):
        s = c.solio_split_goals(solio_team()).set_index("Web_Name")
        self.assertAlmostEqual(s.loc["A", "Solio_PenG90"], 0.2, places=6)
        self.assertAlmostEqual(s.loc["B", "Solio_PenG90"], 0.0, places=6)
        self.assertAlmostEqual(s.loc["B", "Solio_NPG90"], 0.3 * 1.7 * 90 / 80, places=6)
        self.assertTrue(pd.isna(s.loc["D", "Solio_NPG90"]))  # suppressed per-90 row stays blank


if __name__ == "__main__":
    unittest.main()
