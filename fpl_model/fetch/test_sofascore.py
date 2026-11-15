"""Fault-injection and identity tests; all payloads synthetic, no network."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch.player_identity import Registry, write_csv


def lineup():
    return {side: {'formation': '4-4-2', 'players': [
        {'player': {'id': offset + i, 'name': f'Player {offset+i}', 'position': 'G' if i == 0 else 'D'},
         'substitute': i >= 11, 'statistics': {'minutesPlayed': 90 if i < 11 else 0,
                                              'totalTackle': 0, 'wonTackle': 0}}
        for i in range(12)]} for side, offset in [('home', 1), ('away', 101)]}


def event(eid=1):
    return {'id': eid, 'homeTeam': {'name': 'Arsenal'}, 'awayTeam': {'name': 'Liverpool'},
            'status': {'type': 'finished'}, 'startTimestamp': 1502496000}


def response(payload=None, status=200, headers=None):
    r = Mock(status_code=status, headers=headers or {}, content=b'synthetic')
    r.json.return_value = payload
    if status >= 400:
        r.raise_for_status.side_effect = requests.HTTPError(str(status))
    return r


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw, self.data = self.root/'raw', self.root/'current'
        self.data.mkdir()
        self.session, self.sleep = Mock(), Mock()
        self.c = f.Client(self.raw, spacing=0, attempts=2, session=self.session, sleep=self.sleep)

    def tearDown(self):
        self.tmp.cleanup()

    def test_timeout_then_success(self):
        self.session.get.side_effect = [requests.Timeout('slow'), response(lineup())]
        self.assertTrue(self.c.match(2017, event()))
        self.assertEqual(self.c.requests, 2)

    def test_persistent_match_failure_does_not_kill_others(self):
        self.session.get.side_effect = [response(status=404), response(lineup()), response(status=404)]
        failures, successes, stopped = f.download(self.c, {2017: [event(), event(2)]}, retry_passes=1)
        self.assertEqual(successes, 1)
        self.assertFalse(stopped)
        self.assertEqual(failures['2017:1']['kind'], 'unavailable')
        self.assertTrue((self.raw/'queue.json').exists())

    def test_deferred_then_restart_skip_cache_and_success_limit(self):
        self.session.get.side_effect = [requests.Timeout(), requests.Timeout(), response(lineup()), response(lineup())]
        failures, successes, _ = f.download(self.c, {2017: [event(), event(2)]}, retry_passes=1)
        self.assertFalse(failures)
        self.assertEqual(successes, 2)
        requests_before = self.c.requests
        self.assertFalse(self.c.match(2017, event()))
        self.assertEqual(self.c.requests, requests_before)
        self.session.get.side_effect = [response(lineup())]
        failures, successes, _ = f.download(self.c, {2017: [event(), event(3)]}, limit=1)
        self.assertEqual(successes, 1)
        self.assertFalse(failures)

    def test_systemic_outage_stops_after_three_distinct_matches(self):
        self.session.get.side_effect = requests.ConnectionError('offline')
        failures, _, stopped = f.download(self.c, {2017: [event(i) for i in range(10)]})
        self.assertTrue(stopped)
        self.assertEqual(len(failures), 3)
        self.assertEqual(self.c.requests, 6)

    def test_rate_limit_retry_after(self):
        self.session.get.side_effect = [response(status=429, headers={'Retry-After': '20'}), response(lineup())]
        self.c.match(2017, event())
        self.sleep.assert_any_call(20)

    def test_access_not_retried_and_non_json_not_cached(self):
        self.session.get.return_value = response(status=403)
        with self.assertRaises(f.FetchError) as cm:
            self.c.match(2017, event())
        self.assertEqual(cm.exception.kind, 'access')
        self.assertEqual(self.c.requests, 1)
        r = response()
        r.json.side_effect = ValueError('html challenge')
        self.session.get.return_value = r
        with self.assertRaises(f.FetchError) as cm:
            self.c.match(2017, event())
        self.assertEqual(cm.exception.kind, 'non_json')
        self.assertFalse((self.raw/'matches/2017/1.json').exists())

    def test_invalid_schema_and_duplicate_player(self):
        p = lineup()
        p['away']['players'][0]['player']['id'] = 1
        self.session.get.return_value = response(p)
        with self.assertRaises(f.FetchError):
            self.c.match(2017, event())
        p = lineup()
        p['home']['players'][0]['statistics']['saves'] = -1
        with self.assertRaises(f.FetchError):
            f.validate_lineup(p)

    def test_truncated_cache_recovery_frozen_payload_and_invalid_build(self):
        path = self.raw/'matches/2017/1.json'
        path.parent.mkdir(parents=True)
        path.write_text('{truncated')
        self.session.get.return_value = response(lineup())
        self.assertTrue(self.c.match(2017, event()))
        self.assertEqual(len(list(path.parent.glob('*.invalid.*'))), 1)
        with self.assertRaises(f.FetchError):
            self.c.store(path, {'changed': True}, 'synthetic')
        path.write_text('{bad')
        with self.assertRaises(f.FetchError):
            f.build(self.c, Registry(self.data), self.data)
        with self.assertRaises(f.FetchError):
            f.build(self.c, Registry(self.data), self.data)

    def test_weekly_refresh_finds_new_event_and_preserves_revision(self):
        listing = {'rounds': [{'round': 1}]}
        self.session.get.side_effect = [response(listing), response({'events': [event()]}),
                                       response(listing), response({'events': [event(), event(2)]})]
        with patch.object(f, 'CURRENT_SEASON', 2017), patch.object(f, 'DATA', self.data):
            old = f.discover(self.c, 2017, 22, refresh=True)
            new = f.discover(self.c, 2017, 22, refresh=True)
        self.assertEqual(len(old), 1)
        self.assertEqual(len(new), 2)
        self.assertTrue(list((self.raw/'discovery').glob('*.revision.*')))

    def test_pagination_and_discovery_gap(self):
        self.session.get.side_effect = [response({'rounds': [{'round': 1}]}),
                                       response({'events': [event()], 'hasNextPage': True}),
                                       response({'events': [event(2)], 'hasNextPage': False})]
        with patch.object(f, 'CURRENT_SEASON', 2017), patch.object(f, 'DATA', self.data):
            self.assertEqual(len(f.discover(self.c, 2017, 22)), 2)
        self.assertIn('?page=1', self.session.get.call_args.args[0])
        self.session.get.side_effect = [response({'rounds': [{'round': 1}]}), response({'events': []})]
        with self.assertRaises(f.FetchError):
            f.discover(self.c, 2016, 21)
        self.assertFalse((self.raw/'discovery/2016.json').exists())

    def test_build_all_seasons_nulls_unused_bench_and_provenance(self):
        self.session.get.side_effect = [response(lineup()), response(lineup())]
        self.c.match(2017, event())
        self.c.match(2018, event(2))
        reg = Registry(self.data)
        result = f.build(self.c, reg, self.data)
        self.assertEqual(result['rows'], 48)
        rows = f.read_csv(self.data/'sofascore_player_match_stats.csv')
        self.assertEqual({r['Season'] for r in rows}, {'2017', '2018'})
        self.assertEqual(sum(r['Appearance_Status'] == 'unused_bench' for r in rows), 4)
        self.assertTrue(all(r['Sofascore_keyPass'] == '' and r['Sofascore_totalTackle'] == '0' for r in rows))
        self.assertTrue(all(r['Raw_SHA256'] and r['Source_URL'] for r in rows))
        first = (self.data/'sofascore_player_match_stats.csv').read_bytes()
        f.build(self.c, Registry(self.data), self.data)
        self.assertEqual(first, (self.data/'sofascore_player_match_stats.csv').read_bytes())

    def test_limit_does_not_count_cache_reads_or_repair_above_cap(self):
        self.session.get.side_effect = [response(lineup())]
        self.c.match(2017, event())
        broken = self.raw/'matches/2017/2.json'
        broken.write_text('{bad')
        failures, successes, _ = f.download(self.c, {2017: [event(), event(2), event(3)]}, limit=0)
        self.assertEqual(successes, 0)
        self.assertEqual(self.c.requests, 1)
        self.assertEqual(set(failures), {'2017:2', '2017:3'})

    def test_missing_discovery_cannot_be_complete(self):
        with patch.object(f, 'DATA', self.data):
            report = f.manifest(self.c, [2017], {2017: '403'}, {}, {'player_rows': {}, 'rows': 0})
        self.assertFalse(report['complete'])
        self.assertIsNone(report['seasons']['2017']['expected'])
        self.assertEqual(report['seasons']['2017']['discovery_unverified'], 1)

    def test_invalid_metadata_not_cached(self):
        self.session.get.return_value = response({'rounds': []})
        with self.assertRaises(f.FetchError):
            self.c.cached_get('/unique-tournament/17/season/22/rounds', 'metadata/rounds.json')
        self.assertFalse((self.raw/'metadata/rounds.json').exists())

    def test_unknown_bench_minutes_are_not_zero(self):
        p = lineup()
        del p['home']['players'][-1]['statistics']['minutesPlayed']
        self.session.get.return_value = response(p)
        self.c.match(2017, event())
        f.build(self.c, Registry(self.data), self.data)
        rows = f.read_csv(self.data/'sofascore_player_match_stats.csv')
        row = next(r for r in rows if r['Player_ID_Sofascore'] == '12')
        self.assertEqual(row['Appearance_Status'], 'bench_unknown')
        self.assertEqual(row['Sofascore_minutesPlayed'], '')


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.r = Registry(self.data)
        self.cid = self.r.ensure('understat', 7, 'John Smith')
        self.r.add_alias('legacy_pms', '', self.cid, 'John Smith', 2017, 'Arsenal', 'accepted')

    def tearDown(self):
        self.tmp.cleanup()

    def test_provider_reuse_transfer_and_stable_rerun(self):
        ids = self.r.roster('sofascore', [(88, 'John Smith')], 2017, 'Arsenal', ['John Smith'])
        self.assertEqual(ids['88'][0], self.cid)
        self.r.save()
        reg = Registry(self.data)
        self.assertEqual(reg.roster('sofascore', [(88, 'Changed Name')], 2018, 'Liverpool', [])['88'][0], self.cid)

    def test_duplicate_names_remain_unresolved(self):
        result = self.r.roster('sofascore', [(88, 'John Smith'), (89, 'John Smith')], 2017, 'Arsenal', ['John Smith'])
        self.assertTrue(all(status == 'unresolved_cross_provider' for _, status in result.values()))
        self.assertNotEqual(result['88'][0], result['89'][0])

    def test_context_season_and_team_no_global_name_join(self):
        for season, team in [(2018, 'Arsenal'), (2017, 'Liverpool')]:
            result = self.r.roster('sofascore', [(88, 'John Smith')], season, team, ['John Smith'])
            self.assertEqual(result['88'][1], 'unresolved_cross_provider')

    def test_alias_collision_and_roster_collision_rejected(self):
        other = self.r.ensure('understat', 9, 'Other')
        with self.assertRaises(ValueError):
            self.r.add_alias('legacy_pms', '', other, 'John Smith', 2017, 'Arsenal')
        self.r.add_alias('sofascore', 88, self.cid)
        self.r.add_alias('sofascore', 89, self.cid)
        with self.assertRaises(ValueError):
            self.r.roster('sofascore', [(88, 'John'), (89, 'John')], 2017, 'Arsenal', [])

    def test_fixtures_distinguish_events_reschedule_preserves_id(self):
        a = self.r.fixture('understat', 10, 2017, 'Arsenal', 'Liverpool', '2017-08-01')
        b = self.r.fixture('sofascore', 20, 2017, 'Arsenal', 'Liverpool', '2017-08-02')
        self.assertEqual(a, b)
        self.assertNotEqual(a, self.r.fixture('sofascore', 21, 2017, 'Liverpool', 'Arsenal', '2017-08-02'))
        self.assertNotEqual(a, self.r.fixture('sofascore', 22, 2017, 'Arsenal', 'Liverpool', '2017-10-02'))
        with self.assertRaises(ValueError):
            self.r.fixture('sofascore', 20, 2017, 'Other', 'Liverpool', '2017-08-02')

    def test_understat_context_reuse_without_fuzzy_assignment(self):
        key = (2017, 'Arsenal', 'Liverpool', 'Home')
        assignments = self.r.understat_assignments(key, [7], ['John Smith'], ['John Smith'])
        self.assertEqual(assignments, [(0, 'John Smith', 1.0)])
        self.assertEqual(self.r.understat_assignments(key, [8, 9], ['John Smith', 'John Smith'], ['John Smith']), [])

    def test_reviewed_override_resolves_source_only_alias(self):
        self.r.roster('sofascore', [(88, 'Unknown')], 2017, 'Liverpool', [])
        self.r.tables['identity_overrides'].append({'Provider': 'sofascore', 'Provider_Player_ID': '88',
                                                   'Canonical_Player_ID': self.cid, 'Reason': 'Reviewed roster evidence'})
        self.r.save()
        reg = Registry(self.data)
        result = reg.roster('sofascore', [(88, 'John Smith')], 2017, 'Arsenal', [])
        self.assertEqual(result['88'], (self.cid, 'provider_id'))

    def test_unused_substitute_is_retried_once_he_plays(self):
        # First fixture: on the bench, no local row -> source-only, not frozen.
        first = self.r.roster('sofascore', [(88, 'John Smith')], 2017, 'Arsenal', [], minutes={'88': None})
        self.assertEqual(first['88'][1], 'unresolved_cross_provider')
        second = self.r.roster('sofascore', [(88, 'John Smith')], 2017, 'Arsenal', ['John Smith'],
                               minutes={'88': 90}, local_minutes={'John Smith': 90})
        self.assertEqual(second['88'], (self.cid, 'roster_accepted'))
        self.assertEqual(self.r.promoted[('sofascore', '88')], self.cid)
        self.r.save()
        reg = Registry(self.data)
        self.assertEqual(reg.roster('sofascore', [(88, 'X')], 2018, 'Liverpool', [])['88'], (self.cid, 'provider_id'))
        self.assertFalse(any(a['Evidence'] == 'unresolved_source_only' for a in reg.tables['player_identity_aliases']
                             if a['Provider_Player_ID'] == '88'))

    def test_bench_player_cannot_claim_namesakes_local_row(self):
        # Kadan Young (bench) must not take Ashley Young's identity.
        result = self.r.roster('sofascore', [(88, 'Jack Smith')], 2017, 'Arsenal', ['John Smith'],
                               minutes={'88': None}, local_minutes={'John Smith': 90})
        self.assertEqual(result['88'][1], 'unresolved_cross_provider')
        self.assertNotIn(('sofascore', self.cid), self.r.holders)

    def test_minutes_disagreement_blocks_match(self):
        result = self.r.roster('sofascore', [(88, 'John Smith')], 2017, 'Arsenal', ['John Smith'],
                               minutes={'88': 20}, local_minutes={'John Smith': 90})
        self.assertEqual(result['88'][1], 'unresolved_cross_provider')

    def test_short_name_contained_in_full_name(self):
        cid = self.r.ensure('understat', 8, 'Alisson Becker')
        self.r.add_alias('legacy_pms', '', cid, 'Alisson Becker', 2018, 'Liverpool', 'accepted')
        result = self.r.roster('sofascore', [(90, 'Alisson')], 2018, 'Liverpool', ['Alisson Becker'],
                               minutes={'90': 90}, local_minutes={'Alisson Becker': 90})
        self.assertEqual(result['90'], (cid, 'roster_accepted'))

    def test_shared_surnames_resolved_by_whole_assignment(self):
        ids = {}
        for uid, name in [(11, 'Bernardo Silva'), (12, 'David Silva')]:
            ids[name] = self.r.ensure('understat', uid, name)
            self.r.add_alias('legacy_pms', '', ids[name], name, 2017, 'Manchester City', 'accepted')
        result = self.r.roster('sofascore', [(91, 'Bernardo Silva'), (92, 'David Silva')], 2017, 'Manchester City',
                               ['Bernardo Silva', 'David Silva'], minutes={'91': 90, '92': 90},
                               local_minutes={'Bernardo Silva': 90, 'David Silva': 90})
        self.assertEqual(result['91'][0], ids['Bernardo Silva'])
        self.assertEqual(result['92'][0], ids['David Silva'])

    def test_identity_held_by_other_provider_id_is_not_taken(self):
        self.r.add_alias('sofascore', 77, self.cid, 'John Smith', 2017, 'Arsenal', 'fixture_roster:1.0000')
        result = self.r.roster('sofascore', [(88, 'John Smith')], 2018, 'Arsenal', ['John Smith'],
                               minutes={'88': 90}, local_minutes={'John Smith': 90})
        self.assertEqual(result['88'][1], 'unresolved_cross_provider')


class BuildRepointTests(unittest.TestCase):
    def test_earlier_bench_rows_follow_later_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, data = root/'raw', root/'current'
            data.mkdir()
            reg = Registry(data)
            cid = reg.ensure('understat', 7, 'Player 12')
            reg.add_alias('legacy_pms', '', cid, 'Player 12', 2017, 'Arsenal', 'accepted')
            reg.save()
            write_csv(data/'player_match_stats.csv',
                      [{'Season': '2017', 'Team': 'Arsenal', 'Opponent': 'Liverpool', 'Venue': 'Home',
                        'Player': 'Player 12', 'Minutes': '30'}],
                      ['Season', 'Team', 'Opponent', 'Venue', 'Player', 'Minutes'])
            session = Mock()
            c = f.Client(raw, spacing=0, attempts=1, session=session, sleep=Mock())
            bench, played = lineup(), lineup()
            played['home']['players'][-1]['statistics']['minutesPlayed'] = 30
            session.get.side_effect = [response(bench), response(played)]
            c.match(2017, event(1))
            c.match(2017, {**event(2), 'startTimestamp': 1502496000 + 86400 * 30})
            f.build(c, Registry(data), data)
            rows = [r for r in f.read_csv(data/'sofascore_player_match_stats.csv') if r['Player_ID_Sofascore'] == '12']
            self.assertEqual({r['Canonical_Player_ID'] for r in rows}, {cid})
            self.assertEqual(len(rows), 2)


if __name__ == '__main__':
    unittest.main()
