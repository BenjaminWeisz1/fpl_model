"""penalty_log.csv reconciliation (CHANGELOG §73, §80); synthetic frames plus a check on live data."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from any working directory
import penalty_reconcile as pr  # noqa: E402


def log(rows):
    frame = pd.DataFrame(rows, columns=["Season", "HomeAway", "Player", "Team", "Opponent", "Scored"])
    frame["Player"] = frame["Player"].replace(pr.ALIASES)
    frame["is_scored"] = frame["Scored"].astype(str).str.upper() == "TRUE"
    return frame.rename(columns={"HomeAway": "Venue"})


def stats():
    return pd.DataFrame({
        "Season": [2026, 2025, 2025, 2024, 2025],
        "Player": ["A", "Mover", "Mover", "A", "Raúl Jiménez"],
        "Team": ["X", "Old", "New", "X", "Fulham"],
        "Opponent": ["Y", "Z", "Z", "Y", "Burnley"],
        "Venue": ["Home", "Away", "Away", "Home", "Home"],
        "Goals": [2, 1, 1, 1, 1], "PK": [0, 0, 0, 0, 0], "PKatt": [0, 0, 0, 0, 0], "NPG": [2, 1, 1, 1, 1],
        "Penalties_Missed": [0, 0, 0, 0, 0],
        "xG": [1.0, 0.3, 0.2, None, 0.9], "npxG": [1.0, 0.3, 0.2, 0.4, 0.9],
        "Data_Source": ["FPL_API", "FPL_API", "FPL_API", "fbref", "FPL_API"]})


def results():
    return pd.DataFrame({
        "Season": [2026, 2026, 2024], "Home": ["X", "P", "X"], "Away": ["Y", "Q", "Y"],
        "Home_Goals": [2, 1, 1], "Away_Goals": [0, 1, 0], "Home_NPG": [2, 1, 1], "Away_NPG": [0, 1, 0],
        "Home_PK": [0, 0, None], "Away_PK": [0, 0, None], "Home_PKatt": [0, 0, None], "Away_PKatt": [0, 0, None],
        "Home_NPxG_Adj": [2.5, 1.234567, 1.7], "Away_NPxG_Adj": [0.4, 0.9, 0.3]})


class PenaltyReconcileTests(unittest.TestCase):
    def test_late_logged_penalty_reaches_player_rows(self):
        out, report = pr.reconcile_player_stats(stats(), log([(2026, "Home", "A", "X", "Y", "TRUE")]))
        self.assertEqual(out.loc[0, ["PK", "PKatt", "NPG"]].tolist(), [1, 1, 1])
        self.assertEqual(report["rows_changed"], 1)
        pd.testing.assert_series_equal(out.loc[3], stats().loc[3])  # fbref row untouched (NaN-aware)

    def test_team_separates_a_mid_season_transfer(self):
        out, _ = pr.reconcile_player_stats(stats(), log([(2025, "Away", "Mover", "New", "Z", "TRUE")]))
        self.assertEqual(out.loc[1, "PK"], 0)
        self.assertEqual(out.loc[2, "PK"], 1)

    def test_alias_matches_fpl_name(self):
        out, report = pr.reconcile_player_stats(
            stats(), log([(2025, "Home", "Raúl Jiménez Rodríguez", "Fulham", "Burnley", "TRUE")]))
        self.assertEqual(out.loc[4, "PK"], 1)
        self.assertEqual(report["unmatched_log_rows"], [])

    def test_unmatched_log_row_is_reported(self):
        _, report = pr.reconcile_player_stats(stats(), log([(2026, "Home", "Nobody", "X", "Y", "TRUE")]))
        self.assertEqual(len(report["unmatched_log_rows"]), 1)

    def test_more_scored_penalties_than_goals_raises(self):
        rows = [(2026, "Home", "A", "X", "Y", "TRUE")] * 3
        with self.assertRaises(ValueError):
            pr.reconcile_player_stats(stats(), log(rows))

    def test_match_results_shift_npxg_adj_by_delta_only(self):
        out, report = pr.reconcile_match_results(results(), log([(2026, "Home", "A", "X", "Y", "TRUE")]))
        self.assertEqual(out.loc[0, ["Home_PK", "Home_PKatt", "Home_NPG"]].tolist(), [1, 1, 1])
        self.assertAlmostEqual(out.loc[0, "Home_NPxG_Adj"], 2.5 - 0.79)
        self.assertEqual(out.loc[1, "Home_NPxG_Adj"], 1.234567)  # unchanged match keeps its exact value
        self.assertTrue(pd.isna(out.loc[2, "Home_PK"]))  # pre-FPL season untouched
        self.assertEqual(report["rows_changed"], 1)

    def test_idempotent(self):
        lg = log([(2026, "Home", "A", "X", "Y", "TRUE")])
        once, _ = pr.reconcile_match_results(results(), lg)
        twice, report = pr.reconcile_match_results(once, lg)
        self.assertEqual(report["rows_changed"], 0)
        pd.testing.assert_frame_equal(once, twice)

    def test_live_data_is_reconciled(self):
        lg = pr.load_log()
        _, rs = pr.reconcile_player_stats(
            pd.read_csv(pr.PLAYER_STATS_FILE, encoding="utf-8-sig", low_memory=False), lg)
        _, rr = pr.reconcile_match_results(pd.read_csv(pr.MATCH_RESULTS_FILE, low_memory=False), lg)
        self.assertEqual((rs["rows_changed"], rr["rows_changed"]), (0, 0))
        self.assertEqual(rs["unmatched_log_rows"], [])

    def test_npxg_excludes_penalty_xg(self):
        out, report = pr.reconcile_player_stats(stats(), log([(2026, "Home", "A", "X", "Y", "TRUE")]))
        self.assertAlmostEqual(out.loc[0, "npxG"], 0.21)
        self.assertEqual(out.loc[0, "xG"], 1.0)  # raw FPL xG kept
        self.assertEqual(out.loc[3, "npxG"], 0.4)  # fbref npxG untouched
        self.assertEqual(report["rows_changed"], 1)

    def test_player_stats_idempotent(self):
        lg = log([(2026, "Home", "A", "X", "Y", "TRUE"), (2026, "Home", "A", "X", "Y", "FALSE")])
        once, _ = pr.reconcile_player_stats(stats(), lg)
        twice, report = pr.reconcile_player_stats(once, lg)
        self.assertAlmostEqual(once.loc[0, "npxG"], 0.0)  # 1.0 - 2 x 0.79, clipped at zero
        self.assertEqual(report["rows_changed"], 0)
        pd.testing.assert_frame_equal(once, twice)

    def test_rebuild_with_counts_already_right_still_fixes_npxg(self):
        # build_player_match_stats.py restores raw FPL npxG = xG with PKatt already correct;
        # a shift-by-change rule would miss this row, the recompute does not.
        raw = stats()
        raw.loc[0, ["PK", "PKatt", "NPG"]] = [1, 1, 1]
        out, _ = pr.reconcile_player_stats(raw, log([(2026, "Home", "A", "X", "Y", "TRUE")]))
        self.assertAlmostEqual(out.loc[0, "npxG"], 0.21)

    def test_xg_below_penalty_value_is_clipped_and_reported(self):
        frame = stats()
        frame.loc[0, "xG"] = 0.5
        out, report = pr.reconcile_player_stats(frame, log([(2026, "Home", "A", "X", "Y", "TRUE")]))
        self.assertEqual(out.loc[0, "npxG"], 0.0)
        self.assertEqual(len(report["xg_below_penalty_xg_clipped"]), 1)

    def test_missing_xg_raises(self):
        with self.assertRaises(ValueError):
            pr.reconcile_player_stats(stats().drop(columns="xG"), log([]))
        frame = stats()
        frame.loc[1, "xG"] = None
        with self.assertRaises(ValueError):
            pr.reconcile_player_stats(frame, log([]))

    def test_live_summed_player_npxg_matches_team_penalties(self):
        # Sum of player npxG = sum of player xG - 0.79 x the team's PKatt in match_results.csv.
        # (Not compared with NPxG_Adj itself, which holds the user's game-state adjustments.)
        ps = pd.read_csv(pr.PLAYER_STATS_FILE, encoding="utf-8-sig", low_memory=False)
        mr = pd.read_csv(pr.MATCH_RESULTS_FILE, low_memory=False)
        ps = ps[ps["Data_Source"].eq("FPL_API") & (ps["Season"] >= pr.FIRST_FPL_SEASON)]
        team = ps.groupby(["Season", "Team", "Opponent", "Venue"])[["npxG", "xG"]].sum()
        m = mr[mr["Season"] >= pr.FIRST_FPL_SEASON]
        for side, other in (("Home", "Away"), ("Away", "Home")):
            idx = pd.MultiIndex.from_arrays([m["Season"], m[side], m[other], pd.Series(side, index=m.index)])
            t = team.reindex(idx)
            self.assertFalse(t["npxG"].isna().any(), side)
            diff = t["npxG"].to_numpy() - (t["xG"].to_numpy() - pr.PEN_XG_VALUE * m[f"{side}_PKatt"].to_numpy())
            self.assertLessEqual(float(abs(diff).max()), 0.011, side)


if __name__ == "__main__":
    unittest.main()
