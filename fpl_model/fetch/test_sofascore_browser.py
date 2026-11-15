"""Offline browser-import recovery checks, using explicitly synthetic fixtures."""
import json
import tempfile
import unittest
from pathlib import Path

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch import sofascore_browser_pilot as b
from fpl_model.fetch.test_sofascore import event, lineup


class BrowserImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.source = self.base / 'test.json'
        self.source.write_text(json.dumps(lineup()))
        e = event()
        e.update(season={'id': 13380, 'year': '17/18'},
                 tournament={'uniqueTournament': {'id': 17}})
        self.item = {'event': e, 'season': 2017, 'season_id': 13380}
        self.obs = {'url': f.BASE + '/event/1/lineups', 'outcome': 'parsed_json', 'end': f.now()}
        self.client = f.Client(raw=self.base/'raw')

    def tearDown(self):
        self.tmp.cleanup()

    def test_import_restart_skip_preserves_bytes(self):
        self.assertTrue(b.import_dom(self.client, self.base, self.item, self.source, self.obs))
        path = self.base/'raw/matches/2017/1.json'
        before = path.read_bytes()
        self.source.write_text('interrupted DOM read')
        self.assertFalse(b.import_dom(f.Client(raw=self.base/'raw'), self.base, self.item, self.source, self.obs))
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(self.client.requests, 0)

    def test_bad_dom_and_wrong_observation_never_cached(self):
        self.source.write_text('<html>challenge</html>')
        with self.assertRaises(ValueError):
            b.import_dom(self.client, self.base, self.item, self.source, self.obs)
        self.source.write_text(json.dumps(lineup()))
        with self.assertRaises(ValueError):
            b.import_dom(self.client, self.base, self.item, self.source, dict(self.obs, url='wrong'))
        self.assertFalse((self.base/'raw/matches/2017/1.json').exists())

    def test_competition_season_and_schema_checks(self):
        for field, value in [('season', 2018), ('season_id', 99)]:
            with self.assertRaises(ValueError):
                b.import_dom(self.client, self.base, dict(self.item, **{field:value}), self.source, self.obs)
        p = lineup()
        p['home']['players'][0]['statistics']['saves'] = -1
        self.source.write_text(json.dumps(p))
        with self.assertRaises(f.FetchError):
            b.import_dom(self.client, self.base, self.item, self.source, self.obs)

    def test_signed_unexported_statistics_are_preserved(self):
        p = lineup()
        p['home']['players'][0]['statistics']['goalsPrevented'] = -0.5
        self.source.write_text(json.dumps(p))
        b.import_dom(self.client, self.base, self.item, self.source, self.obs)
        env = self.client.load(self.base/'raw/matches/2017/1.json', f.validate_lineup)
        self.assertEqual(env['payload'], p)


if __name__ == '__main__':
    unittest.main()
