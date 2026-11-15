"""FotMob match pages (CHANGELOG §197): page checks, parsing, slot mapping. Synthetic pages, no network."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fpl_model.fetch import fetch_fotmob_matches as ff


def page(match_id="100", league=47, finished=True, score="2 - 1", formation="4-2-3-1"):
    starters = [{"id": 1, "name": "Keeper One", "positionId": 11},
                {"id": 2, "name": "Back Two", "positionId": 32},
                {"id": 3, "name": "Pivot Three", "positionId": 64,
                 "performance": {"substitutionEvents": [{"time": 70, "type": "subOut"}]}}]
    subs = [{"id": 4, "name": "Sub Four", "performance": {"substitutionEvents": [{"time": 70, "type": "subIn"}]}},
            {"id": 5, "name": "Unused Five"}]
    away = [{"id": 6, "name": "Away Striker", "positionId": 115}]
    props = {"general": {"matchId": match_id, "leagueId": league, "finished": finished,
                         "matchTimeUTCDate": "2026-10-31T12:30:00.000Z"},
             "header": {"status": {"scoreStr": score}, "teams": [{"score": 2}, {"score": 1}]},
             "content": {"playerStats": {str(i): {"optaId": 1000 + i} for i in range(1, 7)},
                         "lineup": {"homeTeam": {"id": 9, "name": "Manchester United", "formation": formation,
                                                 "starters": starters, "subs": subs},
                                    "awayTeam": {"id": 8, "name": "Arsenal", "formation": "4-3-3",
                                                 "starters": away, "subs": []}},
                         "matchFacts": {"events": {"events": [
                             {"type": "Card", "card": "Yellow", "playerId": 2, "time": 30},
                             {"type": "Card", "card": "YellowRed", "playerId": 6, "time": 90, "overloadTime": 4}]}},
                         "shotmap": {"shots": [
                             {"teamId": 9, "playerId": 2, "playerName": "Back Two", "min": 45, "minAdded": 3,
                              "expectedGoals": 0.79, "situation": "Penalty", "eventType": "Goal", "isOwnGoal": False}]}}}
    return '<script id="__NEXT_DATA__" type="application/json">' + json.dumps({"props": {"pageProps": props}}) + "</script>"


class CheckPageTests(unittest.TestCase):
    def test_accepts_the_requested_match(self):
        ff.check_page(ff.page_props(page()), "100", "2 - 1")

    def test_rejects_another_meeting_of_the_same_clubs(self):
        with self.assertRaises(ValueError):
            ff.check_page(ff.page_props(page(match_id="999")), "100")

    def test_rejects_a_cup_tie(self):
        with self.assertRaises(ValueError):
            ff.check_page(ff.page_props(page(league=132)), "100")

    def test_rejects_unfinished_or_wrong_score(self):
        with self.assertRaises(ValueError):
            ff.check_page(ff.page_props(page(finished=False)), "100")
        with self.assertRaises(ValueError):
            ff.check_page(ff.page_props(page()), "100", "1 - 1")


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "100.html"
        self.path.write_text(page(), encoding="utf-8")
        self.players, self.shots = ff.parse_page(self.path, 2026)

    def tearDown(self):
        self.tmp.cleanup()

    def test_starters_and_used_subs_only(self):
        self.assertEqual(sorted(p["Player_FotMob"] for p in self.players),
                         ["Away Striker", "Back Two", "Keeper One", "Pivot Three", "Sub Four"])

    def test_project_team_names_venue_and_opta_ids(self):
        p = next(p for p in self.players if p["Player_FotMob"] == "Back Two")
        self.assertEqual((p["Team"], p["Opponent"], p["Venue"], p["Opta_ID"]), ("Manchester Utd", "Arsenal", "Home", "1002"))

    def test_substitution_and_red_card_minutes(self):
        by = {p["Player_FotMob"]: p for p in self.players}
        self.assertEqual((by["Pivot Three"]["Sub_Off_Minute"], by["Sub Four"]["Sub_On_Minute"]), (70, 70))
        self.assertAlmostEqual(by["Away Striker"]["Red_Minute"], 90.04)
        self.assertIsNone(by["Back Two"]["Red_Minute"])          # a single yellow is not a dismissal

    def test_shot_minute_carries_added_time(self):
        self.assertAlmostEqual(self.shots[0]["Minute"], 45.03)
        self.assertEqual(self.shots[0]["Situation"], "Penalty")


class SlotMappingTests(unittest.TestCase):
    def test_formation_specific_mapping_beats_the_slot_alone(self):
        starters = pd.DataFrame({"Formation": ["4-2-3-1"] * 3 + ["3-4-2-1"] * 2,
                                 "FotMob_Position_ID": [64] * 5,
                                 "Position_Understat": ["DMC", "DMC", "DMC", "MC", "MC"]})
        by_form, by_pid = ff.fit_slot_map(starters)
        self.assertEqual(ff.apply_slot_map("3-4-2-1", 64, by_form, by_pid), "MC")
        self.assertEqual(ff.apply_slot_map("4-4-2", 64, by_form, by_pid), "DMC")   # unseen formation: slot alone
        self.assertEqual(ff.apply_slot_map("4-4-2", None, by_form, by_pid), "Sub")

    def test_rule_for_unseen_codes(self):
        self.assertEqual([ff.rule_slot(c) for c in (11, 32, 35, 38, 62, 66, 72, 85, 87, 115)],
                         ["GK", "DR", "DC", "DL", "DMR", "DMC", "MR", "AMC", "AML", "FW"])

    def test_ties_break_alphabetically(self):
        _, by_pid = ff.fit_slot_map(pd.DataFrame({"Formation": ["x", "x"], "FotMob_Position_ID": [64, 64],
                                                  "Position_Understat": ["MC", "DMC"]}))
        self.assertEqual(by_pid[64], "DMC")

    def test_back_three_follows_understat_rule(self):
        fm = pd.DataFrame({"MatchID_FotMob": [1] * 4, "Team": ["A"] * 4, "Is_Starter": [True] * 4,
                           "Formation": ["3-4-3"] * 4, "FotMob_Position_ID": [33, 35, 37, 32]})
        out = ff.label_slots(fm, {}, {})
        self.assertTrue(out["Back_Three"].all())
        self.assertEqual(out.loc[3, "Position"], "WB")              # a wide defender in a back three


if __name__ == "__main__":
    unittest.main()
