"""FPL-only assist rates (CHANGELOG §92): frozen model, shrinkage, modes and the live rates file."""
import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fpl_model.bonus import fpl_only_assists as foa


class FplOnlyAssistTests(unittest.TestCase):
    def test_no_history_gives_the_prior(self):
        for pos in ("D", "M", "F"):
            self.assertAlmostEqual(foa.shrunk_rate(0.0, 0.0, pos, "rebound_rate"), foa.PRIOR["rebound_rate"][pos])

    def test_model_at_prior_rates_is_close_to_the_observed_position_rate(self):
        for pos in ("D", "M", "F"):
            r = foa.rate(pos, foa.PRIOR["rebound_rate"][pos], foa.PRIOR["pens_rate"][pos])
            self.assertLess(abs(r / foa.FALLBACK[pos] - 1), 0.25, pos)

    def test_more_rebound_shots_and_penalties_won_raise_the_rate(self):
        base = foa.rate("M", 0.9, 0.01)
        self.assertGreater(foa.rate("M", 1.5, 0.01), base)
        self.assertGreater(foa.rate("M", 0.9, 0.05), base)

    def test_a_full_share_taker_gets_no_penalty_part(self):
        taker, winner = foa.rate("F", 1.2, 0.08, share=1.0), foa.rate("F", 1.2, 0.08, share=0.0)
        self.assertAlmostEqual(taker, foa.rate("F", 1.2, 0.0))
        self.assertAlmostEqual(winner - taker, 0.08 * foa.C_PEN)

    def test_keepers_use_the_fallback(self):
        self.assertEqual(foa.rate("G", 5.0, 1.0), foa.FALLBACK["G"])

    def test_modes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(foa.mode(d)[0], "fallback")  # no status file
            (d / "sofascore_status.json").write_text(json.dumps({"usable_for_model": False}), encoding="utf-8")
            self.assertEqual(foa.mode(d)[0], "fallback")
            (d / "sofascore_status.json").write_text(json.dumps({"usable_for_model": True}), encoding="utf-8")
            self.assertEqual(foa.mode(d)[0], "model")
            os.environ["FPL_ONLY_ASSISTS"] = "0"
            try:
                self.assertEqual(foa.mode(d)[0], "off")
            finally:
                del os.environ["FPL_ONLY_ASSISTS"]

    def test_live_rates_cover_every_current_player(self):
        path = foa._DATA / "fpl_only_assist_rates.csv"
        if not path.exists():
            self.skipTest("rates file not built yet")
        rates = pd.read_csv(path, encoding="utf-8-sig")
        players = pd.read_csv(foa._DATA / "players_current.csv", encoding="utf-8-sig")
        players = players[players["Pos"].isin(["D", "M", "F", "G"])]
        self.assertEqual(set(players["Player"]), set(rates["Player"]))
        self.assertFalse(rates["FPLOnly_Ast90"].isna().any())
        self.assertTrue((rates["FPLOnly_Ast90"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
