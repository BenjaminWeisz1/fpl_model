"""Weekly browser-download import and status tests; all payloads synthetic, no network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_sofascore as f
from player_identity import write_csv
import sofascore_weekly as w
from test_sofascore import lineup

SID = 96668


def event(eid, home='Arsenal', away='Everton', ts=1790000000 + 0):
    return {'id': eid, 'homeTeam': {'name': home, 'id': 1}, 'awayTeam': {'name': away, 'id': 2},
            'status': {'type': 'finished'}, 'startTimestamp': ts,
            'tournament': {'uniqueTournament': {'id': 17}}, 'season': {'id': SID, 'year': '26/27'}}


def round_url(n, sid=SID):
    return f.BASE + f'/unique-tournament/17/season/{sid}/events/round/{n}'


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw, self.data, self.inbox = root / 'raw', root / 'current', root / 'inbox'
        for d in (self.data, self.inbox):
            d.mkdir(parents=True)
        f.atomic_json(self.raw / 'discovery' / '2026.json',
                      {'payload': {'events': [], 'season_id': SID},
                       'sha256': f.digest({'events': [], 'season_id': SID}),
                       'retrieved_at': f.now(), 'url': round_url(1)})

    def tearDown(self):
        self.tmp.cleanup()

    def capture(self, name, url, payload):
        text = payload if isinstance(payload, str) else json.dumps(payload)
        (self.inbox / f'sofascore_capture_{name}.json').write_text(json.dumps(
            {'capture_version': 1, 'url': url, 'retrieved_at': '2026-11-02T10:00:00.000Z',
             'transport': 'chrome_normal_navigation_download', 'dom_text': text}), encoding='utf-8')

    def fixtures(self, completed):
        write_csv(self.data / 'fixtures.csv', [dict(Season='2026', Round=r, Home=h, Away=a) for r, h, a in completed],
                  ['Season', 'Round', 'Home', 'Away'])
        write_csv(self.data / 'match_results.csv', [dict(Season='2026', Home=h, Away=a) for _, h, a in completed],
                  ['Season', 'Home', 'Away'])


class WeeklyTests(Base):
    def test_round_then_lineup_import_with_provenance(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures'], {})
        self.assertEqual(result['imported'], {'round': 1, 'lineup_new': 1})
        events, _ = w.discovery(self.raw)
        self.assertIn(11, events)
        env = json.loads((self.raw / 'matches/2026/11.json').read_text(encoding='utf-8'))
        self.assertEqual(env['transport'], 'chrome_normal_navigation_download')
        self.assertEqual(json.loads(Path(env['dom_path']).read_text(encoding='utf-8')), env['payload'])
        self.assertEqual(list(self.inbox.iterdir()), [])

    def test_identical_reimport_is_a_no_op(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        cache = self.raw / 'matches/2026/11.json'
        before = cache.read_bytes()
        self.capture('lineup2', f.BASE + '/event/11/lineups', lineup())
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['imported'], {'lineup_unchanged': 1})
        self.assertEqual(cache.read_bytes(), before)

    def test_follower_count_change_is_unchanged(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        cache = self.raw / 'matches/2026/11.json'
        before = cache.read_bytes()
        recaptured = lineup()
        recaptured['home']['players'][0]['player']['userCount'] = 1234
        self.capture('lineup2', f.BASE + '/event/11/lineups', recaptured)
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual((result['imported'], result['failures']), ({'lineup_unchanged': 1}, {}))
        self.assertEqual(cache.read_bytes(), before)

    def test_changed_lineup_for_frozen_event_is_rejected(self):
        self.capture('round', round_url(9), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        changed = lineup()
        changed['home']['players'][0]['statistics']['totalTackle'] = 3
        self.capture('lineup2', f.BASE + '/event/11/lineups', changed)
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures']['sofascore_capture_lineup2.json']['kind'], 'cache_revision')
        self.assertTrue((self.inbox / 'sofascore_capture_lineup2.json').exists())

    def test_lineup_without_discovered_event_stays_in_inbox(self):
        self.capture('lineup', f.BASE + '/event/12/lineups', lineup())
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['failures']['sofascore_capture_lineup.json']['kind'], 'event_not_discovered')
        self.assertFalse((self.raw / 'matches/2026/12.json').exists())

    def test_challenge_page_and_wrong_season_rejected(self):
        self.capture('challenge', f.BASE + '/event/11/lineups', '<html>Just a moment...</html>')
        self.capture('oldseason', round_url(9, sid=76986), {'events': []})
        failures = w.import_inbox(self.raw, self.inbox)['failures']
        self.assertEqual(failures['sofascore_capture_challenge.json']['kind'], 'non_json')
        self.assertEqual(failures['sofascore_capture_oldseason.json']['kind'], 'schema')

    def test_manual_saves_import(self):
        (self.inbox / 'sofascore__round__9.json').write_text(json.dumps({'events': [event(11)]}), encoding='utf-8')
        (self.inbox / 'sofascore__lineups__11.json').write_text(json.dumps(lineup()), encoding='utf-8')
        result = w.import_inbox(self.raw, self.inbox)
        self.assertEqual(result['imported'], {'round': 1, 'lineup_new': 1})
        env = json.loads((self.raw / 'matches/2026/11.json').read_text(encoding='utf-8'))
        self.assertEqual(env['transport'], 'manual_browser_save')

    def test_status_lag_and_staleness(self):
        self.fixtures([(1, 'Arsenal', 'Everton'), (2, 'Everton', 'Arsenal'), (3, 'Chelsea', 'Fulham'),
                       (4, 'Fulham', 'Chelsea')])
        self.capture('round', round_url(1), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        write_csv(self.data / 'sofascore_player_match_stats.csv',
                  [{'Season': '2026', 'Event_ID_Sofascore': '11'}], ['Season', 'Event_ID_Sofascore'])
        result = w.status(self.raw, self.data)
        self.assertEqual((result['sofascore_complete_through_round'], result['lag_rounds']), (1, 3))
        self.assertFalse(result['usable_for_model'])
        self.assertEqual(len(result['missing_fixtures']), 3)

    def test_status_flags_unbuilt_cache(self):
        self.fixtures([(1, 'Arsenal', 'Everton')])
        self.capture('round', round_url(1), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        write_csv(self.data / 'sofascore_player_match_stats.csv', [], ['Season', 'Event_ID_Sofascore'])
        result = w.status(self.raw, self.data)
        self.assertEqual(result['cached_not_built'], ['11'])
        self.assertFalse(result['usable_for_model'])


class PipelineTests(Base):
    def test_bad_capture_and_build_failure_never_raise(self):
        self.fixtures([(1, 'Arsenal', 'Everton')])
        self.capture('round', round_url(1), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        self.capture('challenge', f.BASE + '/event/12/lineups', '<html>Just a moment...</html>')
        write_csv(self.data / 'sofascore_player_match_stats.csv', [], ['Season', 'Event_ID_Sofascore'])
        with patch.object(w, 'build', side_effect=ValueError('Duplicate source player-event rows')):
            result = w.pipeline(self.raw, self.data, self.inbox)
        self.assertIn('build failed: ValueError: Duplicate source player-event rows', result['errors'])
        self.assertTrue(any('challenge' in m for m in result['warnings']))
        self.assertFalse(result['usable_for_model'])
        self.assertTrue((self.inbox / 'sofascore_capture_challenge.json').exists())

    def test_build_runs_only_when_output_is_stale(self):
        self.fixtures([(1, 'Arsenal', 'Everton')])
        self.capture('round', round_url(1), {'events': [event(11)]})
        self.capture('lineup', f.BASE + '/event/11/lineups', lineup())
        w.import_inbox(self.raw, self.inbox)
        write_csv(self.data / 'sofascore_player_match_stats.csv',
                  [{'Season': '2026', 'Event_ID_Sofascore': '11'}], ['Season', 'Event_ID_Sofascore'])
        with patch.object(w, 'build') as build:
            result = w.pipeline(self.raw, self.data, self.inbox)
        build.assert_not_called()
        self.assertTrue(result['usable_for_model'])


if __name__ == '__main__':
    unittest.main()
