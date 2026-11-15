"""Automatic penalty log from Understat shots (CHANGELOG §75); synthetic frames, no network."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fpl_model.fetch import penalty_log_understat as plu


def log(rows):
    return pd.DataFrame(rows, columns=["Season", "Venue", "Player", "Team", "Opponent", "Scored"])


def pens(rows):
    frame = pd.DataFrame(rows, columns=["Season", "Venue", "Player", "Team", "Opponent", "Scored"])
    return frame.assign(MatchID_Understat=1, Minute=range(10, 10 + len(frame)), Understat_Player=frame["Player"])


class PenaltyLogUnderstatTests(unittest.TestCase):
    def test_appends_only_missing_penalties(self):
        add, extra, _ = plu.compare(log([(2026, "Home", "A", "X", "Y", True)]),
                                    pens([(2026, "Home", "A", "X", "Y", True), (2026, "Away", "B", "Y", "X", False)]))
        self.assertEqual(add[["Player", "Scored"]].values.tolist(), [["B", False]])
        self.assertEqual(extra, [])

    def test_counts_repeats_within_a_match(self):
        add, _, _ = plu.compare(log([(2026, "Home", "A", "X", "Y", True)]),
                                pens([(2026, "Home", "A", "X", "Y", True)] * 2))
        self.assertEqual(len(add), 1)

    def test_manual_row_understat_lacks_is_kept_and_reported(self):
        add, extra, _ = plu.compare(log([(2026, "Home", "A", "X", "Y", True), (2026, "Home", "C", "X", "Y", True)]),
                                    pens([(2026, "Home", "A", "X", "Y", True)]))
        self.assertEqual(len(add), 0)
        self.assertEqual([e["Player"] for e in extra], ["C"])

    def test_alias_names_count_as_logged(self):
        add, extra, _ = plu.compare(log([(2025, "Home", "Raúl Jiménez Rodríguez", "Fulham", "Burnley", True)]),
                                    pens([(2025, "Home", "Raúl Jiménez", "Fulham", "Burnley", True)]))
        self.assertEqual((len(add), extra), (0, []))

    def test_unmapped_player_is_skipped(self):
        p = pens([(2026, "Home", "A", "X", "Y", True)])
        p.loc[0, "Player"] = None
        add, _, unmapped = plu.compare(log([]), p)
        self.assertEqual((len(add), len(unmapped)), (0, 1))

    def test_append_keeps_bom_and_crlf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "penalty_log.csv"
            path.write_bytes("﻿Season,HomeAway,Player,Team,Opponent,Scored\r\n2026,Home,A,X,Y,TRUE\r\n".encode("utf-8"))
            plu.append_to_log(pens([(2026, "Away", "Pascal Groß", "Y", "X", False)]), path)
            raw = path.read_bytes()
            self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
            self.assertTrue(raw.endswith("2026,Away,Pascal Groß,Y,X,FALSE\r\n".encode("utf-8")))
            self.assertEqual(raw.count(b"\r\n"), 3)


if __name__ == "__main__":
    unittest.main()
