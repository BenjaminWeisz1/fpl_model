"""python -m unittest test_get_next_round (from scripts/team_ratings)"""
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from get_next_round import get_next_round  # noqa: E402

FIXTURES = [(2026, 1, "A", "B"), (2026, 1, "C", "D"), (2026, 2, "A", "C"), (2026, 2, "B", "D"),
            (2026, 3, "A", "D"), (2026, 3, "B", "C")]


def next_round(played, fixtures=FIXTURES):
    with tempfile.TemporaryDirectory() as d:
        pd.DataFrame(fixtures, columns=["Season", "Round", "Home", "Away"]).to_csv(Path(d) / "fixtures.csv", index=False)
        pd.DataFrame([(2026, h, a) for h, a in played], columns=["Season", "Home", "Away"]).to_csv(
            Path(d) / "match_results.csv", index=False)
        return get_next_round(Path(d), 2026)


class NextRound(unittest.TestCase):
    def test_no_played_match(self):
        self.assertEqual(next_round([]), 1)

    def test_partly_played_round_stays_open(self):
        self.assertEqual(next_round([("A", "B")]), 1)

    def test_full_rounds(self):
        self.assertEqual(next_round([("A", "B"), ("C", "D")]), 2)

    def test_rearranged_match_is_not_skipped(self):
        self.assertEqual(next_round([("A", "B"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]), 1)

    def test_season_finished(self):
        self.assertEqual(next_round([(h, a) for _, _, h, a in FIXTURES]), 4)

    def test_blank_round_holds_nothing(self):
        fixtures = [(s, None if (h, a) == ("C", "D") else r, h, a) for s, r, h, a in FIXTURES]
        self.assertEqual(next_round([("A", "B")], fixtures), 2)

    def test_trailing_space_in_fixtures(self):
        fixtures = [(s, r, h + "\n", a) for s, r, h, a in FIXTURES]
        self.assertEqual(next_round([("A", "B"), ("C", "D")], fixtures), 2)

    def test_repeated_fixture_stops(self):
        with self.assertRaises(Exception):
            next_round([("A", "B")], FIXTURES + [(2026, 3, "A", "B")])


if __name__ == "__main__":
    unittest.main()
