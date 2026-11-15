"""Tests for export_site_data.py: the site's numbers must equal the CSVs they come from (website_plan.md).

    python -m unittest scripts/site_export/test_export_site_data.py
"""
import json
import math
import unittest
from unittest import mock

import pandas as pd

from fpl_model.site_export import export_site_data as e  # noqa: E402


def synthetic(horizon=3):
    rounds = list(range(11, 11 + horizon))
    points = pd.DataFrame([{"Player": "A", "Pos": "M", "Team": "AAA", "Price": 7.5,
                            **{str(r): 1.0 for r in rounds}, "Total": 3.0}])
    ratings = pd.DataFrame([{"Player": "A", "Pos": "M", "Team": "AAA", "Price": 7.5, "NPG": .3, "Assists": .2, "Defcon": .1},
                            {"Player": "B", "Pos": "D", "Team": "BBB", "Price": 4.5, "NPG": .1, "Assists": .1, "Defcon": .4}])
    players_current = pd.DataFrame([{"Web_Name": "A", "Raw_Web_Name": "Araw", "Team_Short": "AAA"},
                                    {"Web_Name": "B", "Raw_Web_Name": "B", "Team_Short": "BBB"}])
    def row(player, team, rnd, pts, opp):
        return {"Player": player, "Team": team, "Round": rnd, "Opponent": opp, "Venue": "Home", "TotalPts": pts,
                "xMins_Raw": 80, "Goals": .2, "Assists": .1, "CS_Prob": .3, "Defcon_Prob": .05}
    components = pd.DataFrame([
        row("A", "AAA", 11, 4.0, "X"), row("A", "AAA", 12, 3.0, "Y"), row("A", "AAA", 12, 2.5, "Z"),  # double in 12
        row("B", "BBB", 11, 2.0, "X"), row("B", "BBB", 13, 2.2, "W"),                                  # blank in 12
    ])
    shares = pd.DataFrame([{"Player": "Araw", "Team": "AAA", **{f"GW{i}": v for i, v in zip(range(1, horizon + 1), (1, .5, "x"))}}])
    return {"predicted_points.csv": points, "player_ratings.csv": ratings, "players_current.csv": players_current,
            "predicted_components.csv": components, "penalty_share.csv": shares}


class SyntheticTests(unittest.TestCase):
    def build(self):
        with mock.patch.object(e, "HORIZON", 3):
            return e.build_projections(synthetic())

    def test_double_and_blank_weeks(self):
        out = {p["name"]: p for p in self.build()["players"]}
        a, b = out["A"], out["B"]
        self.assertEqual(a["points"], [4.0, 5.5, None])
        self.assertEqual(a["fixtures"], [1, 2, 0])
        self.assertEqual(a["minutes"], [80, 160, None])
        self.assertEqual(a["opponents"][1], ["Y (H)", "Z (H)"])
        self.assertEqual(b["points"], [2.0, None, 2.2])
        self.assertEqual(b["fixtures"], [1, 0, 1])

    def test_penalty_shares_resolve_raw_names_and_are_not_summed(self):
        out = {p["name"]: p for p in self.build()["players"]}
        # A double gameweek shows the average of its games' shares (one share per week, so the week's share).
        self.assertEqual(out["A"]["pens"], [1.0, 0.5, None])  # blank week is null
        self.assertEqual(out["B"]["pens"], [0.0, None, 0.0])   # no share row: 0

    def test_rounds_must_match_horizon(self):
        with mock.patch.object(e, "HORIZON", 4), self.assertRaises(ValueError):
            e.build_projections(synthetic())

    def test_rated_player_without_forecast_rows_fails(self):
        frames = synthetic()
        frames["predicted_components.csv"] = frames["predicted_components.csv"].query("Player != 'B'")
        with mock.patch.object(e, "HORIZON", 3), self.assertRaises(ValueError):
            e.build_projections(frames)


class SyntheticRatingsTests(unittest.TestCase):
    def build(self):
        frames = synthetic()
        frames["players_current.csv"] = frames["players_current.csv"].assign(
            Player=["Alpha Full", "Beta Full"], Team=["Aaa FC", "Bbb FC"])
        frames["team_ratings_current.csv"] = pd.DataFrame([
            {"Team": "Aaa FC", "Attack_Rating": 1.5, "Defense_Rating": 1.0},
            {"Team": "Bbb FC", "Attack_Rating": 1.0, "Defense_Rating": 2.0}])
        frames["minutes.csv"] = pd.DataFrame([{"Name": "Araw", "Team": "AAA", "11_xMins": 90, "12_xMins": 60, "13_xMins": 0}])
        with mock.patch.object(e, "HORIZON", 3):
            return {p["name"]: p for p in e.build_ratings(frames, rates=(0.1, 0.8), conversion_rates={"Alpha Full": 0.9})["players"]}

    def test_values(self):
        a, b = self.build()["A"], self.build()["B"]
        w = e.week_weights(3)                                             # weighted mean of the window, raw name resolved
        self.assertAlmostEqual(a["minutes"], round((90 * w[0] + 60 * w[1]) / sum(w), 1))
        self.assertEqual(b["minutes"], 0.0)                            # no minutes.csv row
        self.assertAlmostEqual(a["share"], (1 + 0.5 + 0) / 3, places=4)  # unparseable share counts 0
        self.assertAlmostEqual(a["team_att90"], 1.5 * 0.1)              # Attack_Rating x attempts per NPG
        self.assertAlmostEqual(a["cs"], round(math.exp(-1.0 * (1 + 0.1 * 0.8)), 4))
        self.assertAlmostEqual(b["cs"], round(math.exp(-2.0 * (1 + 0.1 * 0.8)), 4))
        self.assertEqual(a["conversion"], 0.9)
        self.assertEqual(b["conversion"], 0.79)                        # no history: 0.79, as build_penalty_predictions
        self.assertEqual((a["npg"], a["assists"], a["defcon"]), (0.3, 0.2, 0.1))


@unittest.skipUnless((e.DATA / "predicted_components.csv").exists(), "no production outputs")
class ProductionTests(unittest.TestCase):
    """The export of the current data/current files agrees with the CSVs, value by value."""

    @classmethod
    def setUpClass(cls):
        cls.payload = e.build_projections()
        cls.rounds = cls.payload["meta"]["rounds"]
        cls.components = e.read("predicted_components.csv")
        cls.points = e.read("predicted_points.csv")

    def test_window_is_horizon(self):
        self.assertEqual(len(self.rounds), e.HORIZON)
        self.assertEqual(self.payload["meta"]["horizon"], e.HORIZON)

    def test_players_are_the_rated_players(self):
        ratings = e.read("player_ratings.csv")
        self.assertEqual({(p["name"], p["team"]) for p in self.payload["players"]}, set(zip(ratings.Player, ratings.Team)))

    def test_weekly_values_equal_summed_components(self):
        sums = self.components.groupby(["Player", "Team", "Round"])[list(e.SUMMED_STATS)].sum()
        for p in self.payload["players"]:
            for i, rnd in enumerate(self.rounds):
                key = (p["name"], p["team"], rnd)
                for column, stat in e.SUMMED_STATS.items():
                    if key in sums.index:
                        self.assertAlmostEqual(p[stat][i], sums.loc[key, column], delta=10 ** -e.DECIMALS[stat] / 2 + 1e-9)
                    else:
                        self.assertIsNone(p[stat][i])

    def test_clean_sheet_is_p60_times_team_chance(self):
        """Projections' clean sheet % is the player's: P60 x the team's chance, summed over a double (user, 2026-11-10)."""
        c = self.components.assign(_cs=self.components["P60"] * self.components["CS_Prob_Team"])
        sums = c.groupby(["Player", "Team", "Round"])["_cs"].sum()
        for p in self.payload["players"]:
            for i, rnd in enumerate(self.rounds):
                if p["cs"][i] is not None:
                    self.assertAlmostEqual(p["cs"][i], sums.loc[(p["name"], p["team"], rnd)], delta=2e-4)

    def test_points_total_matches_predicted_points(self):
        table = self.points.set_index(["Player", "Team"])
        for p in self.payload["players"]:
            total = sum(v or 0 for v in p["points"])
            self.assertAlmostEqual(total, table.loc[(p["name"], p["team"]), "Total"], delta=0.02)

    def test_penalty_shares_equal_penalty_share_csv(self):
        shares = e.read("penalty_share.csv")
        shares["Web_Name"] = e.resolve_names(shares, e.read("players_current.csv"), "Player")
        lookup = shares.drop_duplicates(["Web_Name", "Team"]).set_index(["Web_Name", "Team"])
        for p in self.payload["players"]:
            for i in range(len(self.rounds)):
                if p["pens"][i] is None:
                    continue
                key = (p["name"], p["team"])
                expected = float(lookup.loc[key, f"GW{i + 1}"]) if key in lookup.index else 0.0
                self.assertAlmostEqual(p["pens"][i], expected, places=4)

    def test_json_round_trip(self):
        self.assertEqual(json.loads(json.dumps(self.payload)), self.payload)


@unittest.skipUnless((e.DATA / "predicted_components.csv").exists(), "no production outputs")
class ProductionRatingsTests(unittest.TestCase):
    """ratings.json values against player_ratings.csv, minutes.csv, penalty_share.csv and the team ratings."""

    @classmethod
    def setUpClass(cls):
        cls.payload = e.build_ratings()
        cls.players = {(p["name"], p["team"]): p for p in cls.payload["players"]}
        cls.rounds = cls.payload["meta"]["rounds"]

    def test_ratings_copied(self):
        for r in e.read("player_ratings.csv").itertuples():
            p = self.players[(r.Player, r.Team)]
            self.assertAlmostEqual(p["npg"], r.NPG, places=4)
            self.assertAlmostEqual(p["assists"], r.Assists, places=4)
            if pd.isna(r.Defcon):
                self.assertIsNone(p["defcon"])
            else:
                self.assertAlmostEqual(p["defcon"], r.Defcon, places=4)

    def test_minutes_match_projections(self):
        projections = {(p["name"], p["team"]): p for p in e.build_projections()["players"]}
        for key, p in self.players.items():
            weekly = [v or 0 for v in projections[key]["minutes"]]
            w = e.week_weights(len(weekly))
            self.assertAlmostEqual(p["minutes"], sum(v * x for v, x in zip(weekly, w)) / sum(w), delta=0.06)

    def test_team_values_follow_team_ratings(self):
        attempts, conversion = e.penalty_rates()
        teams = e.read("team_ratings_current.csv")
        short = dict(zip(e.read("players_current.csv")["Team"], e.read("players_current.csv")["Team_Short"]))
        expected = {short[t.Team]: t for t in teams.itertuples()}
        for p in self.players.values():
            t = expected[p["team"]]
            self.assertAlmostEqual(p["team_att90"], t.Attack_Rating * attempts, places=5)
            self.assertAlmostEqual(p["cs"], math.exp(-t.Defense_Rating * (1 + attempts * conversion)), places=4)

    def test_team_att90_is_the_forecast_at_an_average_opponent(self):
        """Undo the opponent and venue in fixture_goal_predictions.csv: what is left is the team's NPG v an
        average opponent at a neutral venue, which must equal team_att90 / attempts per NPG."""
        from fpl_model.league_rates import mean_npg
        from fpl_model.team_ratings.update_team_ratings import home_advantage
        attempts, _ = e.penalty_rates()
        defence = e.read("team_ratings_current.csv").set_index("Team")["Defense_Rating"]
        pc = e.read("players_current.csv")
        short = dict(zip(pc["Team"], pc["Team_Short"]))
        att90 = {p["team"]: p["team_att90"] for p in self.players.values()}
        fixtures = e.read("fixture_goal_predictions.csv")
        for f in fixtures.itertuples():
            for team, opp, npg, venue in ((f.Home, f.Away, f.Home_Pred_NPG, home_advantage),
                                          (f.Away, f.Home, f.Away_Pred_NPG, 2 - home_advantage)):
                neutral = npg / (defence[opp] / mean_npg()) / venue
                self.assertAlmostEqual(att90[short[team]], neutral * attempts, places=4)

    def test_share_is_window_mean_of_projection_shares(self):
        projections = {(p["name"], p["team"]): p for p in e.build_projections()["players"]}
        for key, p in self.players.items():
            weekly = projections[key]["pens"]
            if None in weekly:
                continue  # a blank week: the projection hides that week's share
            self.assertAlmostEqual(p["share"], sum(weekly) / len(weekly), delta=1e-3)


class SyntheticTeamsTests(unittest.TestCase):
    def test_schedule_double_blank_and_strength(self):
        frames = synthetic()
        frames["players_current.csv"] = pd.DataFrame({"Team": ["Aaa FC", "Bbb FC", "Ccc FC"], "Team_Short": ["AAA", "BBB", "CCC"]})
        frames["team_ratings_current.csv"] = pd.DataFrame([
            {"Team": "Aaa FC", "Attack_Rating": 1.5, "Defense_Rating": 1.0, "Rating": 0.5},
            {"Team": "Bbb FC", "Attack_Rating": 1.0, "Defense_Rating": 2.0, "Rating": -1.0},
            {"Team": "Ccc FC", "Attack_Rating": 1.2, "Defense_Rating": 1.2, "Rating": 0.0}])
        frames["fixture_goal_predictions.csv"] = pd.DataFrame([
            {"MatchID": 1, "Round": 11, "Home": "Aaa FC", "Away": "Bbb FC", "Home_Pred_NPG": 2.0, "Away_Pred_NPG": 1.0},
            {"MatchID": 2, "Round": 12, "Home": "Ccc FC", "Away": "Aaa FC", "Home_Pred_NPG": 1.0, "Away_Pred_NPG": 1.5},
            {"MatchID": 3, "Round": 12, "Home": "Aaa FC", "Away": "Bbb FC", "Home_Pred_NPG": 1.0, "Away_Pred_NPG": 0.5}])
        frames["predicted_table.csv"] = pd.DataFrame([{"Team": "Aaa FC", "Played": 10, "Predicted_Points": 70.0}])
        # (opponent, venue of the team facing it) -> (goals for, goals against) of an average team
        strength = {(c, v): (gf, ga) for c, gf, ga in (("AAA", 1.0, 1.6), ("BBB", 1.8, 1.1), ("CCC", 1.3, 1.3))
                    for v in ("H", "A")}
        strength[("BBB", "A")] = (1.7, 1.2)
        with mock.patch.object(e, "HORIZON", 3):
            out = e.build_teams(frames, strength=strength)
        weeks = {t["team"]: t["weeks"] for t in out["fixtures"]}
        self.assertEqual(weeks["AAA"][0], [{"opp": "BBB", "venue": "H", "gf": 1.8, "ga": 1.1}])  # opponent's strength
        self.assertEqual(weeks["BBB"][0], [{"opp": "AAA", "venue": "A", "gf": 1.0, "ga": 1.6}])
        self.assertEqual([f["opp"] for f in weeks["AAA"][1]], ["CCC", "BBB"])                     # double, by MatchID
        self.assertEqual(weeks["AAA"][2], [])                                                    # blank
        self.assertEqual(weeks["CCC"][0], [])
        self.assertEqual(out["ratings"][0], {"team": "AAA", "name": "Aaa FC", "overall": 0.5, "attack": 1.5, "defense": 1.0})
        self.assertEqual(out["table"][0]["Predicted_Points"], 70.0)


@unittest.skipUnless((e.DATA / "predicted_components.csv").exists(), "no production outputs")
class ProductionTeamsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = e.build_teams()
        pc = e.read("players_current.csv")
        cls.short = dict(zip(pc["Team"], pc["Team_Short"]))

    def test_strength_is_an_average_team_v_the_opponent(self):
        """gf/ga follow the fixture ticker's formula: an average team's goals v that opponent at that venue."""
        from fpl_model.team_ratings.predict_fixture_goals import AVG_CONCEDED, home_advantage
        ratings = e.read("team_ratings_current.csv")
        ratings["Code"] = ratings["Team"].map(self.short)
        attack, defence = ratings["Attack_Rating"].mean(), ratings["Defense_Rating"].mean()
        by_code = ratings.set_index("Code")
        for t in self.payload["fixtures"]:
            for games in t["weeks"]:
                for f in games:
                    opp = by_code.loc[f["opp"]]
                    own = home_advantage if f["venue"] == "H" else 2 - home_advantage
                    self.assertAlmostEqual(f["gf"], attack * opp.Defense_Rating / AVG_CONCEDED * own, places=3)
                    self.assertAlmostEqual(f["ga"], opp.Attack_Rating * defence / AVG_CONCEDED * (2 - own), places=3)

    def test_schedule_matches_the_forecast(self):
        """Every team-fixture in the window, and only those, as in fixture_goal_predictions.csv."""
        fixtures = e.read("fixture_goal_predictions.csv")
        expected = sorted([(self.short[f.Home], int(f.Round), self.short[f.Away], "H") for f in fixtures.itertuples()] +
                          [(self.short[f.Away], int(f.Round), self.short[f.Home], "A") for f in fixtures.itertuples()])
        rounds = self.payload["meta"]["rounds"]
        got = sorted((t["team"], rounds[i], f["opp"], f["venue"]) for t in self.payload["fixtures"]
                     for i, games in enumerate(t["weeks"]) for f in games)
        self.assertEqual(got, expected)

    def test_ratings_and_table_copied(self):
        teams = e.read("team_ratings_current.csv")
        out = {r["name"]: r for r in self.payload["ratings"]}
        for t in teams.itertuples():
            self.assertAlmostEqual(out[t.Team]["overall"], t.Rating, places=4)
            self.assertAlmostEqual(out[t.Team]["overall"], t.Attack_Rating - t.Defense_Rating, places=3)
            self.assertAlmostEqual(out[t.Team]["defense"], t.Defense_Rating, places=4)
        table = e.read("predicted_table.csv")
        self.assertEqual([r["name"] for r in self.payload["table"]], table["Team"].tolist())
        for r, t in zip(self.payload["table"], table.itertuples()):
            self.assertAlmostEqual(r["Predicted_Points"], t.Predicted_Points, places=3)


if __name__ == "__main__":
    unittest.main()
