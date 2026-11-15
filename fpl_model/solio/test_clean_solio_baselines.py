"""Tests for the Solio baselines cleaner. Run: python -m unittest test_clean_solio_baselines (from this folder)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fpl_model.solio import clean_solio_baselines as c

TEAMS = [f"T{i:02d}" for i in range(1, 21)]


def world(root, gameweek=11, captured="2026-11-13T10:00:00+00:00", shown=11, drop_team=False, break_share=False):
    """20 teams x 2 players (a regular forward and a fringe defender under the minutes cut-off)."""
    data, raw = root / "data" / "current", root / "data" / "raw" / "solio"
    data.mkdir(parents=True)
    folder = raw / "2026" / f"GW{gameweek:02d}"
    folder.mkdir(parents=True)
    rows, codes, players = [], [], []
    for t, team in enumerate(TEAMS[:19] if drop_team else TEAMS, start=1):
        for k, (pos, mins, g90, share) in enumerate([("FWD", 85.0, 0.5, 0.8), ("DEF", 10.0, 0.0, 0.2)]):
            code = 1000 * t + k
            rows.append({"fpl_id": code, "fpl_code": code, "name": f"P{code}", "position": pos, "solio_team_id": t,
                         "now_cost": 60, "status": "available", "pts90": 4.0 - k, "mins_per_match": mins,
                         "g90": g90, "a90": 0.1 if k == 0 else 0.0, "g_pct": share + (0.1 if break_share and t == 1 else 0),
                         "a_pct": share, "cs_pct": 0.3, "bon90": 0.5 if k == 0 else 0.0, "dc90": 3.0 if k == 0 else 0.0})
            codes.append({"Opta_Code": code, "Player": f"Full {code}", "First_Seen_GW": 1})
            players.append({"Player": f"Full {code}", "Web_Name": f"P{code}", "Raw_Web_Name": f"P{code}",
                            "Pos": {"FWD": "F", "DEF": "D"}[pos], "Team": team, "Team_Short": team, "Price": 6.0, "Status": "a"})
    pd.DataFrame(rows).to_csv(folder / c.SOURCE_NAME, index=False)
    (folder / c.CAPTURE_NAME).write_text(json.dumps({"captured_utc": captured, "first_gameweek_shown": shown}))
    pd.DataFrame(codes).to_csv(data / "fpl_player_codes.csv", index=False)
    pd.DataFrame(players).to_csv(data / "players_current.csv", index=False)
    pd.DataFrame([{"Gameweek": gameweek, "Deadline_UTC": "2026-11-21T11:00:00+00:00"}]).to_csv(data / "gameweek_deadlines.csv", index=False)
    return data, raw


class Build(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())

    def run_build(self, **kw):
        data, raw = world(self.root, **kw)
        with patch.object(c, "DATA", data), patch.object(c, "RAW", raw), patch.object(c, "ROOT", self.root):
            return c.build(11, 2026)

    def test_clean_capture_links_and_blanks_suppressed_per90(self):
        table, status = self.run_build()
        self.assertTrue(status["usable_for_comparison"], status["reasons"])
        self.assertEqual(len(table), 40)
        self.assertEqual(status["linked_share"], 1.0)
        fringe = table[table["Solio_Per90_Suppressed"]]
        self.assertEqual(len(fringe), 20)
        self.assertTrue(fringe["Solio_G90"].isna().all() and fringe["Solio_DC90"].isna().all())
        self.assertTrue(fringe["Solio_Goal_Share"].notna().all())  # shares stay
        self.assertEqual(status["per90_suppression"]["max_mins_suppressed"], 10.0)
        self.assertEqual(status["per90_suppression"]["min_mins_shown"], 85.0)
        self.assertEqual(set(table["Solio_Team_Short"]), set(TEAMS))

    def test_capture_after_deadline_is_unusable(self):
        _, status = self.run_build(captured="2026-11-21T12:00:00+00:00")
        self.assertFalse(status["usable_for_comparison"])
        self.assertIn("after the GW11 deadline", status["reasons"][0])

    def test_capture_of_another_week_is_unusable(self):
        _, status = self.run_build(shown=12)
        self.assertFalse(status["usable_for_comparison"])
        self.assertIn("GW12", status["reasons"][0])

    def test_partial_capture_is_unusable(self):
        _, status = self.run_build(drop_team=True)
        self.assertFalse(status["usable_for_comparison"])
        self.assertIn("19 teams", status["reasons"][0])

    def test_share_not_summing_to_one_is_unusable(self):
        _, status = self.run_build(break_share=True)
        self.assertFalse(status["usable_for_comparison"])
        self.assertIn("team checks failed", status["reasons"][0])

    def test_missing_column_is_refused(self):
        data, raw = world(self.root)
        path = raw / "2026" / "GW11" / c.SOURCE_NAME
        pd.read_csv(path).drop(columns="dc90").to_csv(path, index=False)
        with patch.object(c, "DATA", data), patch.object(c, "RAW", raw), patch.object(c, "ROOT", self.root), \
                self.assertRaisesRegex(ValueError, "lacks columns"):
            c.build(11, 2026)


if __name__ == "__main__":
    unittest.main()
