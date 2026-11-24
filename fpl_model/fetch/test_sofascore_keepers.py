"""Keepers' Sofascore player pages: import, penalty rates, non-penalty columns, chart table. Synthetic, no network."""
import unittest
from pathlib import Path

import pandas as pd

from fpl_model.eda import keeper_shot_stopping as ks
from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch import sofascore_keepers as k
from fpl_model.fetch import sofascore_set_pieces as sp
from fpl_model.fetch import sofascore_weekly as w
from fpl_model.fetch.player_identity import write_csv
from fpl_model.fetch.test_sofascore_weekly import Base

SEASONS = {'uniqueTournamentSeasons': [
    {'uniqueTournament': {'id': 17, 'name': 'Premier League'},
     'seasons': [{'year': '25/26', 'id': 76986}, {'year': '20/21', 'id': 29000}]}]}
STATS = {'statistics': {'appearances': 38, 'minutesPlayed': 3420, 'goalsConceded': 40,
                        'expectedGoalsOnTargetConceded': 45.0, 'penaltyFaced': 4, 'penaltySave': 1}}


def penalty(shot_type, xgot):
    return {'player': {'id': 1}, 'situation': 'penalty', 'shotType': shot_type, 'xgot': xgot}


class Helpers(unittest.TestCase):
    def test_season_year(self):
        self.assertEqual(k.season_year('21/22'), 2021)
        self.assertEqual(k.season_year('2024'), 2024)
        self.assertIsNone(k.season_year('Season'))

    def test_route_path(self):
        self.assertEqual(k.route_path('r', '/player/5/statistics/seasons')[0].as_posix(), 'r/players/5/seasons.json')
        self.assertEqual(k.route_path('r', '/player/5/unique-tournament/17/season/9/statistics/overall')[0].name,
                         '17_9.json')
        self.assertIsNone(k.route_path('r', '/event/5/lineups'))


class ImportAndBuild(Base):
    def test_player_pages_import_build_and_revise(self):
        self.capture('ps', f.BASE + '/player/7/statistics/seasons', SEASONS)
        self.capture('st', f.BASE + '/player/7/unique-tournament/17/season/76986/statistics/overall', STATS)
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures'], {})
        self.assertEqual(result['imported'], {'player_new': 2})
        # Penalty rates come from shot maps: 1 of 2 unsaved penalties missed; on-target xGOT averages 0.8.
        f.atomic_json(sp.feed_path(self.raw, 'shotmap', 2026, 1),
                      {'payload': {'shotmap': [penalty('goal', 0.9), penalty('save', 0.7), penalty('miss', 0.0)]},
                       'sha256': f.digest({'shotmap': [penalty('goal', 0.9), penalty('save', 0.7),
                                                       penalty('miss', 0.0)]}),
                       'retrieved_at': f.now(), 'url': 'x'})
        write_csv(self.data / 'sofascore_player_match_stats.csv',
                  [dict(Player_ID_Sofascore='7', Local_Player='Keeper Seven', Player_Sofascore='K. Seven',
                        Kickoff='2026-01-01', Position_Sofascore='G')],
                  ['Player_ID_Sofascore', 'Local_Player', 'Player_Sofascore', 'Kickoff', 'Position_Sofascore'])
        summary = k.build(self.raw, self.data)
        self.assertEqual(summary['penalty_miss_rate'], 0.5)
        out = pd.read_csv(self.data / k.OUT_FILE, encoding='utf-8-sig').iloc[0]
        # 3 unsaved penalties x 0.5 scored = 1.5 penalty goals; 4 faced - 1.5 missed = 2.5 on target x 0.8 xGOT.
        self.assertAlmostEqual(out['NPGC'], 40 - 1.5)
        self.assertAlmostEqual(out['NPxGOTC'], 45 - 2.5 * 0.8)
        self.assertEqual((out['Tournament'], out['Season_Start'], out['Local_Player']),
                         ('Premier League', 2025, 'Keeper Seven'))
        revised = {'statistics': {**STATS['statistics'], 'goalsConceded': 41}}
        self.capture('st2', f.BASE + '/player/7/unique-tournament/17/season/76986/statistics/overall', revised)
        self.assertEqual(w.import_inbox(self.raw, self.inbox)['imported'], {'player_revised': 1})
        self.assertEqual(len(list((self.raw / 'players' / '7').glob('17_76986.json.revision.*'))), 1)


class Chart(unittest.TestCase):
    def test_table_uses_seasons_with_xgot_from_first_season(self):
        seasons = pd.DataFrame([
            dict(Player_ID_Sofascore=7, Season_Start=2025, expectedGoalsOnTargetConceded=45.0, minutesPlayed=3420,
                 NPxGOTC=43.0, NPGC=38.5),
            dict(Player_ID_Sofascore=7, Season_Start=2024, expectedGoalsOnTargetConceded=None, minutesPlayed=900,
                 NPxGOTC=None, NPGC=10.0),
            dict(Player_ID_Sofascore=7, Season_Start=2019, expectedGoalsOnTargetConceded=30.0, minutesPlayed=900,
                 NPxGOTC=30.0, NPGC=10.0)])
        players = pd.DataFrame([dict(Player='Keeper Seven', Web_Name='Seven', Team_Short='ARS', Price=5.5)])
        t = ks.table(seasons, [('ARS', 'Keeper Seven', 7)], players).iloc[0]
        self.assertEqual(t[ks.MINUTES], 3420)
        self.assertAlmostEqual(t[ks.SCORE], (38.5 + 50) / (43.0 + 50))  # goals over xGOT: lower is better



class Finishing(unittest.TestCase):
    def test_finishing_table(self):
        from fpl_model.eda import player_finishing as pf
        seasons = pd.DataFrame([
            dict(Player_ID_Sofascore=9, Season_Start=2025, expectedGoals=20.0, minutesPlayed=3000, NPG=18, NPxG=15.0),
            dict(Player_ID_Sofascore=9, Season_Start=2024, expectedGoals=None, minutesPlayed=900, NPG=5, NPxG=None),
            dict(Player_ID_Sofascore=9, Season_Start=2020, expectedGoals=9.0, minutesPlayed=900, NPG=9, NPxG=9.0)])
        players = pd.DataFrame([dict(Player='Striker Nine', Web_Name='Nine', Pos='F', Team_Short='ARS', Price=8.0)])
        t = pf.table(seasons, [('ARS', 'Striker Nine', 9)], players).iloc[0]
        self.assertAlmostEqual(t[pf.SCORE], (18 + 50) / (15.0 + 50))  # only 2021+ seasons with xG count

    def test_top_defenders_need_predicted_minutes(self):
        import tempfile

        from fpl_model import paths
        from fpl_model.fetch import sofascore_finishing as sf
        with tempfile.TemporaryDirectory() as tmp:
            def put(name, frame):
                path = paths.current_file(name, root=Path(tmp))
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path, index=False, encoding='utf-8-sig')
            put('player_ratings.csv', pd.DataFrame(dict(Player=['A', 'B', 'C', 'D'], Pos=['D', 'D', 'D', 'M'],
                                                        Team=['ARS'] * 4, NPG=[0.3, 0.2, 0.1, 0.9])))
            put('players_current.csv', pd.DataFrame(dict(Raw_Web_Name=['A', 'B', 'C', 'D'], Web_Name=['A', 'B', 'C', 'D'],
                                                         Team_Short=['ARS'] * 4)))
            minutes = {'Name': ['A', 'B', 'C', 'D'], 'Team': ['ARS'] * 4}
            minutes.update({f'{gw}_xMins': [10, 90, 90, 90] for gw in range(11, 18)})
            put('minutes.csv', pd.DataFrame(minutes))
            # A has the best rating but only 50 minutes over the next five gameweeks; D is a midfielder.
            self.assertEqual(sf.top_defenders(Path(tmp), 2)['Player'].tolist(), ['B', 'C'])

    def test_skipped_competitions(self):
        self.assertTrue(k.skipped({'id': 19, 'name': 'FA Cup'}))
        self.assertTrue(k.skipped({'id': 999, 'name': 'Club Friendly Games'}))
        self.assertFalse(k.skipped({'id': 17, 'name': 'Premier League'}))


if __name__ == '__main__':
    unittest.main()
