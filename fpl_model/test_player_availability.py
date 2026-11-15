import unittest

import pandas as pd

from fpl_model.player_availability import status_conflicts, with_availability


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({
            "Web_Name": ["Gone", "Late", "Hurt", "M. Same", "Fit"],
            "Raw_Web_Name": ["Gone", "Late", "Hurt", "Same", "Fit"],
            "Team_Short": ["AVL", "AVL", "AVL", "MCI", "MCI"],
            "Status": ["u", "u", "i", "a", "a"],
        })
        self.minutes = pd.DataFrame({
            "Name": ["Gone", "Late", "Hurt", "Same", "Same"],
            "Team": ["AVL", "AVL", "AVL", "MCI", "LIV"],
            "9_xMins": [0, 30, 0, 90, 0],
            "10_xMins": [0, 0, 0, 90, 0],
        })

    def test_departed_needs_both_the_flag_and_no_minutes(self):
        frame = with_availability(self.players, self.minutes).set_index("Web_Name")
        self.assertEqual(frame.index[frame["Departed"]].tolist(), ["Gone"])

    def test_totals_join_on_raw_name_and_team(self):
        frame = with_availability(self.players, self.minutes).set_index("Web_Name")
        self.assertEqual(frame.loc["M. Same", "Total_xMins"], 180)
        self.assertEqual(frame.loc["Fit", "Total_xMins"], 0)  # absent from minutes.csv

    def test_conflict_is_a_flagged_player_with_minutes(self):
        self.assertEqual(status_conflicts(self.players, self.minutes)["Web_Name"].tolist(), ["Late"])


if __name__ == "__main__":
    unittest.main()
