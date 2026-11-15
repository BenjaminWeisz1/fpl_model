"""python -m unittest test_fixture_rounds (from scripts/fetch)"""
import unittest

import pandas as pd

from fpl_model.fetch.fetch_fixtures_results import round_mismatches  # noqa: E402

TEAMS = {1: "A", 2: "B", 3: "C"}


def api(home, away, event):
    return {"team_h": home, "team_a": away, "event": event}


def rows(*r):
    return pd.DataFrame(r, columns=["Home", "Away", "Round"])


class Rounds(unittest.TestCase):
    def test_agreement(self):
        self.assertEqual(round_mismatches([api(1, 2, 5)], TEAMS, rows(("A", "B", 5))), [])

    def test_moved_fixture(self):
        self.assertEqual(round_mismatches([api(1, 2, 9)], TEAMS, rows(("A", "B", 5))), [("A", "B", 5, 9)])

    def test_postponed_needs_a_blank_round(self):
        self.assertEqual(round_mismatches([api(1, 2, None)], TEAMS, rows(("A", "B", 5))), [("A", "B", 5, None)])
        self.assertEqual(round_mismatches([api(1, 2, None)], TEAMS, rows(("A", "B", None))), [])

    def test_missing_row(self):
        self.assertEqual(round_mismatches([api(1, 3, 5)], TEAMS, rows(("A", "B", 5))), [("A", "C", "no row", 5)])


if __name__ == "__main__":
    unittest.main()
