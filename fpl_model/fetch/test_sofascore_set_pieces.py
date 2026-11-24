"""Set-piece tables from Sofascore commentary and shot maps; synthetic payloads, no network."""
import unittest

import pandas as pd

from fpl_model.eda import set_piece_takers as st
from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch import sofascore_set_pieces as sp
from fpl_model.fetch import sofascore_weekly as w
from fpl_model.fetch.test_sofascore import lineup
from fpl_model.fetch.test_sofascore_weekly import Base, event, round_url


def shot(pid, name, minute, situation, xg, home=True, added=None, shot_type='miss'):
    return {'player': {'id': pid, 'name': name}, 'time': minute, 'addedTime': added, 'situation': situation,
            'xg': xg, 'isHome': home, 'shotType': shot_type}


def comment(kind, minute, text, pid=None, home=True, **extra):
    return {'type': kind, 'time': minute, 'text': text, 'isHome': home,
            'player': {'id': pid} if pid else None, **extra}


class Parsing(unittest.TestCase):
    def test_assister_regex(self):
        texts = {'Attempt missed. A (X) header. Assisted by Enzo Le Fée with a cross following a corner.': 'Enzo Le Fée',
                 'Attempt saved. A (X) shot. Assisted by Pascal Groß.': 'Pascal Groß',
                 'Attempt missed. A (X) header following a set piece situation. Assisted by J. Doe following a set piece situation.': 'J. Doe'}
        for text, name in texts.items():
            self.assertEqual(sp.ASSIST.search(text).group(1), name)

    def test_match_name(self):
        team = {1: ('home', 'Estêvão', 'Estêvão', True), 2: ('home', 'Fatawu Issahaku', 'F. Issahaku', True),
                3: ('home', 'Danny Welbeck', 'D. Welbeck', True)}
        self.assertEqual(sp.match_name('Estevao', team), 1)
        self.assertEqual(sp.match_name('Abdul Fatawu', team), 2)
        self.assertEqual(sp.match_name('Welbeck', team), 3)
        self.assertIsNone(sp.match_name('Nobody Here', team))

    def test_pairing_is_one_to_one_and_uses_added_time(self):
        shots = [shot(7, 'A', 90, 'corner', 0.1, added=2), shot(7, 'A', 90, 'regular', 0.2, added=4)]
        comments = [comment('shotOffTarget', 94, 'second', 7), comment('shotSaved', 92, 'first', 7),
                    comment('shotSaved', 50, 'too far', 7)]
        pairs = sp.pair_shots(shots, comments)
        self.assertEqual(pairs[0]['text'], 'first')
        self.assertEqual(pairs[1]['text'], 'second')

    def test_on_pitch_spans(self):
        players = {1: ('home', 'A', 'A', True), 2: ('home', 'B', 'B', False), 3: ('home', 'C', 'C', True)}
        comments = [comment('substitution', 60, 'sub', 2, playerIn={'id': 2}, playerOut={'id': 1}),
                    comment('penaltyScored', 65, 'pen', 3),  # "penaltyScored" contains "red": not a red card
                    comment('redCard', 70, 'red', 3)]
        self.assertEqual(sp.on_pitch(players, comments, 95), {1: [0.0, 60], 2: [60, 95], 3: [0.0, 70]})


class FeedImport(Base):
    def test_comments_and_shotmap_import_then_refuse_revision(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('comments', f.BASE + '/event/11/comments', {'comments': [comment('matchStarted', 0, 'KO')]})
        self.capture('shotmap', f.BASE + '/event/11/shotmap', {'shotmap': [shot(5, 'A', 3, 'corner', 0.1)]})
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures'], {})
        self.assertEqual(result['imported'], {'round': 1, 'comments_new': 1, 'shotmap_new': 1})
        self.assertTrue(sp.feed_path(self.raw, 'shotmap', 2026, 11).exists())
        self.assertEqual(sp.cached_events(self.raw, 2026), {'11'})
        self.capture('shotmap2', f.BASE + '/event/11/shotmap', {'shotmap': [shot(5, 'A', 3, 'corner', 0.2)]})
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures']['sofascore_capture_shotmap2.json']['kind'], 'cache_revision')

    def test_plan_lists_missing_feeds_for_cached_lineups(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        self.fixtures([(9, 'Arsenal', 'Everton')])
        kinds = [c['kind'] for c in w.plan(self.raw, self.data)['capture']]
        self.assertEqual(kinds, ['comments', 'shotmap'])


class Chart(unittest.TestCase):
    def test_shares_count_only_shots_while_on_pitch(self):
        ev = pd.DataFrame([
            dict(Season=2026, Event_ID_Sofascore=1, Team='ARS', Event_Type='shot', Situation='corner', Minute=10,
                 Added_Time=None, Taker_ID_Sofascore=1, xG=0.3),
            dict(Season=2026, Event_ID_Sofascore=1, Team='ARS', Event_Type='shot', Situation='corner', Minute=80,
                 Added_Time=None, Taker_ID_Sofascore=2, xG=0.1),
            dict(Season=2026, Event_ID_Sofascore=1, Team='ARS', Event_Type='shot', Situation='free-kick', Minute=20,
                 Added_Time=None, Taker_ID_Sofascore=2, xG=0.05),
        ])
        pitch = pd.DataFrame([dict(Season=2026, Event_ID_Sofascore=1, Team='ARS', Player_ID_Sofascore=1,
                                   Local_Player='One', On=0, Off=60, Minutes_Played=None),
                              dict(Season=2026, Event_ID_Sofascore=1, Team='ARS', Player_ID_Sofascore=2,
                                   Local_Player='Two', On=0, Off=95, Minutes_Played=90)])
        players = pd.DataFrame([dict(Player='One', Web_Name='One', Pos='M', Team_Short='ARS', Price=5.0, Status='a'),
                                dict(Player='Two', Web_Name='Two', Pos='M', Team_Short='ARS', Price=5.0, Status='a')])
        t = st.table(ev, pitch, players, 2026).set_index('Player')
        self.assertAlmostEqual(t.loc['One', 'SP xA/90'], 0.3 * 90 / 60)
        self.assertEqual(t.loc['One', st.CORNER], 1.0)   # the 80th-minute corner came after he went off
        self.assertEqual(t.loc['Two', st.CORNER], 0.5)
        self.assertEqual(t.loc['Two', st.DIRECT_FK], 1.0)
        self.assertEqual(t.loc['One', st.IND_FK], 0.0)  # no indirect free-kick shots while on: 0%, not blank
        self.assertEqual(t.loc['One', st.TAKEN], 1)   # his corner shot
        self.assertEqual(t.loc['Two', st.TAKEN], 2)   # a corner shot set up and a direct free kick

    def test_eligible_needs_predicted_minutes_and_set_pieces_taken(self):
        minutes = pd.DataFrame({'Name': ['Raw1', 'Two', 'Three'], 'Team': ['ARS'] * 3,
                                **{f'{gw}_xMins': [30, 10, 30] for gw in range(11, 18)}})
        players = pd.DataFrame([dict(Raw_Web_Name='Raw1', Team_Short='ARS', Web_Name='One')])
        xmins = st.next_xmins(minutes, players)
        self.assertEqual(xmins[('One', 'ARS')], 150)   # raw name mapped; only the next five gameweeks count
        df = pd.DataFrame({'Player': ['One', 'Two', 'Three'], 'Team': ['ARS'] * 3, st.TAKEN: [5, 9, 4]})
        self.assertEqual(list(st.eligible(df, xmins)['Player']), ['One'])  # Two: 50 minutes; Three: 4 taken

    def test_team_table_counts_against_per_match_without_penalties(self):
        rows = [dict(Event_Type='corner_won', Situation='corner', Team='ARS', Opponent='EVE', xG=None),
                dict(Event_Type='free_kick_won', Situation='attacking half', Team='ARS', Opponent='EVE', xG=None),
                dict(Event_Type='free_kick_won', Situation='defensive half', Team='ARS', Opponent='EVE', xG=None),
                dict(Event_Type='shot', Situation='corner', Team='ARS', Opponent='EVE', xG=0.2),
                dict(Event_Type='shot', Situation='penalty', Team='ARS', Opponent='EVE', xG=0.79),
                dict(Event_Type='shot', Situation='throw-in-set-piece', Team='ARS', Opponent='EVE', xG=0.1)]
        ev = pd.DataFrame([{**r, 'Season': 2026, 'Event_ID_Sofascore': 1} for r in rows]
                          + [dict(Season=2026, Event_ID_Sofascore=2, Event_Type='corner_won', Situation='corner',
                                  Team='CHE', Opponent='EVE', xG=None)])
        t = st.team_table(ev, 2026).set_index('Team')
        self.assertEqual(t.loc['EVE', 'Matches'], 2)
        self.assertEqual(t.loc['EVE', st.CONCEDED], 1.5)   # two corners and one dangerous free kick, two matches
        self.assertEqual(t.loc['EVE', st.SHOTS], 0.5)
        self.assertAlmostEqual(t.loc['EVE', st.XGC], 0.1)
        self.assertEqual(t.loc['ARS', st.XGC], 0)


if __name__ == '__main__':
    unittest.main()
