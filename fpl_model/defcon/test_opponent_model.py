"""Tests for the opponent factor (CHANGELOG §149). Run: python -m pytest scripts/defcon/test_opponent_model.py"""
import numpy as np
import pandas as pd

from fpl_model.defcon.opponent_model import opponent_factors, position_groups

POS = ["Pos_CB", "Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]


def frame(rows):
    """rows: (season, team, opponent, venue, per90)"""
    f = pd.DataFrame(rows, columns=["Season", "Team", "Opponent", "Venue", "Per90Stat"])
    f["Minutes"], f["MatchesPlayed"], f["AvgPer90"], f["Recent5AvgPer90"] = 90, 20, 8.0, 8.0
    f[POS] = 0
    f["Pos_CB"] = 1
    return f


MATCHES = [((2024, "A", "B", "Home"), (2024, "B", "A", "Away")),
           ((2024, "C", "B", "Home"), (2024, "B", "C", "Away")),
           ((2024, "D", "B", "Home"), (2024, "B", "D", "Away"))]
ROWS = [(2024, "A", "B", "Home", 8.0), (2024, "B", "A", "Away", 8.0),
        (2024, "C", "B", "Home", 16.0), (2024, "B", "C", "Away", 8.0),
        (2024, "D", "B", "Home", 8.0), (2024, "B", "D", "Away", 8.0)]


def test_first_match_is_neutral():
    row, _ = opponent_factors(frame(ROWS), "D", MATCHES)
    assert row[0] == 1.0 and row[1] == 1.0


def test_a_rows_own_match_never_enters_its_factor():
    base = opponent_factors(frame(ROWS), "D", MATCHES)[0]
    changed = list(ROWS)
    changed[4] = (2024, "D", "B", "Home", 30.0)   # the last match's outcome
    row = opponent_factors(frame(changed), "D", MATCHES)[0]
    np.testing.assert_array_equal(base[[0, 1, 2, 3, 4, 5]], row[[0, 1, 2, 3, 4, 5]])


def test_conceding_more_raises_the_factor_afterwards():
    row, latest = opponent_factors(frame(ROWS), "D", MATCHES)
    assert row[4] > 1.0 and latest[("B", "CB")] > 1.0


def test_unknown_slot_groups():
    f = frame(ROWS[:2])
    assert (position_groups(f, "D") == "CB").all()
    f["Pos_CB"], f["Pos_LB"] = 0, 1
    assert (position_groups(f, "D") == "WD").all()
