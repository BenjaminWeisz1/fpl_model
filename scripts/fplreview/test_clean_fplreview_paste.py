"""Tests for the FPL Review paste cleaner. Run: python -m unittest test_clean_fplreview_paste (from this folder)."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import clean_fplreview_paste as c

H = c.HORIZON  # the pastes cover the pipeline's forecast window (forecast_horizon.py)
ROUNDS = tuple(range(11, 11 + H))


def paste(path, stat, players, rounds=ROUNDS):
    rows = [["FPL Review"], ["Settings"], [stat], ["Secondary Stat"], ["Own%"],
            ["Player", "Price", *[f"GW{r}" for r in rounds], "Total", "Elite Own%"]]
    for name, team, pos, price, values, xmins in players:
        rows.append([team, price, *values, round(sum(values), 2), 0.1])
        rows.append([f"{name}{team} • {pos}"])
        rows.append([xmins])
    pd.DataFrame(rows).to_excel(path, header=False, index=False)


class ParsePaste(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_three_row_blocks_and_dash_minutes(self):
        p = self.dir / "goals.xlsx"
        haaland = ([0.6, 0.8, 0.3, 0.7, 0.5] * H)[:H]
        paste(p, "Goals", [("Haaland", "MCI", "FWD", "£15.8mSV: 15.6▲", haaland, 93),
                           ("Martinelli", "ARS", "MID", "£6.3m", ([0, 0, 0.1, 0.1, 0.1] * H)[:H], "-")])
        rounds, rows = c.parse_paste(p, "Goals")
        self.assertEqual(rounds, list(ROUNDS))
        self.assertEqual(rows[0]["FPLR_Name"], "Haaland")
        self.assertEqual(rows[0]["Price"], 15.8)
        self.assertEqual(rows[0]["FPLR_Pos"], "F")
        self.assertEqual(rows[0]["values"][12], 0.8)
        self.assertTrue(pd.isna(rows[1]["FPLR_xMins_First"]))

    def test_wrong_stat_is_refused(self):
        p = self.dir / "goals.xlsx"
        paste(p, "Assists", [("Haaland", "MCI", "FWD", "£15.8m", [0.1] * H, 93)])
        with self.assertRaisesRegex(ValueError, "expected 'Goals'"):
            c.parse_paste(p, "Goals")

    def test_shifted_layout_is_refused(self):
        p = self.dir / "goals.xlsx"
        paste(p, "Goals", [("Haaland", "MCI", "FWD", "£15.8m", [0.1] * H, 93)])
        df = pd.read_excel(p, header=None)
        df = pd.concat([df, pd.DataFrame([["extra"]])], ignore_index=True)
        df.to_excel(p, header=False, index=False)
        with self.assertRaisesRegex(ValueError, "multiple of three"):
            c.parse_paste(p, "Goals")

    def test_other_horizon_is_refused(self):
        p = self.dir / "goals.xlsx"
        paste(p, "Goals", [("Haaland", "MCI", "FWD", "£15.8m", [0.1] * 5, 93)], rounds=(11, 12, 13, 14, 15))
        with self.assertRaisesRegex(ValueError, f"expected {H} gameweeks"):
            c.parse_paste(p, "Goals")

    def test_rounding_of_ten_values_is_accepted(self):
        p = self.dir / "points.xlsx"
        paste(p, "Points", [("Haaland", "MCI", "FWD", "£15.8m", [6.9] * H, 93)])
        df = pd.read_excel(p, header=None)
        df.iloc[6, 2 + H] = round(6.9 * H + 0.05 * H, 1)  # each shown value rounded down by up to 0.05
        df.to_excel(p, header=False, index=False)
        c.parse_paste(p, "Points")


def strengths(path, teams, blend=("MarketsDEFAULT", 0, "100%elevenify")):
    cells = ["FPL Review", "Settings", "Team Strengths", "Adjust ...", "Source Blend", *blend,
             "Team", "AttackGF", "GADefence", "GD"]
    for code, name, gf, gd, ga in teams:
        cells += [code, name, gf, gd, ga]
    cells += ["Data provided for free by", "elevenify.com", "DEF", "Player"]
    pd.DataFrame({0: cells}).to_excel(path, header=False, index=False)


class TeamStrengths(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.teams = [(f"T{chr(65 + i)}{chr(65 + i)}", f"Team {i}", 1.5, 0.1, 1.4) for i in range(20)]

    def test_values_are_gf_gd_ga(self):
        p = self.dir / "Team Strengths.xlsx"
        strengths(p, [("ARS", "Arsenal", 1.84, 1.05, 0.79)] + self.teams[1:])
        table, blend = c.parse_team_strengths(p)
        self.assertEqual(len(table), 20)
        self.assertEqual(table.iloc[0][["FPLR_GF", "FPLR_GA"]].tolist(), [1.84, 0.79])
        self.assertIn("elevenify", blend)

    def test_inconsistent_goal_difference_is_refused(self):
        p = self.dir / "Team Strengths.xlsx"
        strengths(p, [("ARS", "Arsenal", 1.84, 1.05, 0.90)] + self.teams[1:])
        with self.assertRaisesRegex(ValueError, "code, name, GF, GD, GA"):
            c.parse_team_strengths(p)

    def test_swapped_gd_and_ga_columns_are_refused(self):
        p = self.dir / "Team Strengths.xlsx"
        swapped = [(code, name, gf, ga, gd) for code, name, gf, gd, ga in self.teams]  # GF - GD = GA still holds
        strengths(p, swapped)
        with self.assertRaisesRegex(ValueError, "look swapped"):
            c.parse_team_strengths(p)


class LinkPlayers(unittest.TestCase):
    def test_shared_raw_name_is_split_by_team(self):
        players = pd.DataFrame({"Raw_Web_Name": ["Gabriel", "Gabriel"], "Team_Short": ["ARS", "CHE"],
                                "Web_Name": ["Gabriel", "Gabriel (CHE)"], "Pos": ["D", "M"]})
        frame = pd.DataFrame({"FPLR_Name": ["Gabriel", "Gabriel"], "Team_Short": ["CHE", "ARS"]})
        linked = c.link_players(frame, players)
        self.assertEqual(linked["Web_Name"].tolist(), ["Gabriel (CHE)", "Gabriel"])


if __name__ == "__main__":
    unittest.main()
