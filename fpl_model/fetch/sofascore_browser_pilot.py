"""Offline adapter for the bounded, tool-driven Chrome pilot; never sends HTTP.

Supported browser tooling writes original DOM text and navigation.json. This CLI
validates and imports it through Client.store, checkpoints the missing selection,
and builds only in the pilot's shadow current directory. See pilot README.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch.player_identity import Registry

DEFAULT = f.ROOT / 'data/validation/sofascore_browser_pilot_20261026'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected():
    paths = set()
    for folder in ('data/current', 'data/raw', 'data/snapshots', 'fpl_model', 'cache'):
        for p in (f.ROOT / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',):
                paths.add(p)
    paths.update(f.ROOT.glob('run_*.py'))
    # The new pilot adapter/tests are permitted implementation changes.
    return {str(p.relative_to(f.ROOT)): sha(p) for p in sorted(paths)
            if p.name not in ('sofascore_browser_pilot.py', 'test_sofascore_browser.py', 'SOFASCORE.md')}


def prepare(base):
    base.mkdir(parents=True, exist_ok=True)
    baseline = base / 'preservation_before.json'
    if baseline.exists():
        raise ValueError('Baseline already exists; do not reset preservation evidence')
    f.atomic_json(baseline, protected())
    data = base / 'current'
    data.mkdir(exist_ok=True)
    for name in (*Registry.SCHEMAS, 'player_match_stats', 'understat_fpl_crosswalk',
                 'player_match_positions', 'understat_match_positions', 'players_current'):
        source = f.DATA / (name + '.csv')
        if source.exists():
            shutil.copy2(source, data / source.name)


def validate_event(event, season, season_id):
    f.validate_metadata({'events': [event]}, '/events/round/1')
    if (event['status']['type'] != 'finished'
            or event['tournament']['uniqueTournament']['id'] != 17
            or event['season']['id'] != season_id
            or f.season_year(event['season']) != season):
        raise ValueError('Wrong competition/season or unfinished fixture')


def import_dom(client, base, item, source, observation):
    event, season = item['event'], item['season']
    validate_event(event, season, item['season_id'])
    path = client.raw / 'matches' / str(season) / f'{event["id"]}.json'
    if client.load(path, f.validate_lineup):
        return False
    payload = json.loads(source.read_text(encoding='utf-8'))
    url = f.BASE + f'/event/{event["id"]}/lineups'
    if observation['url'] != url or observation.get('outcome') != 'parsed_json':
        raise ValueError('Missing successful matching browser observation')
    client.store(path, payload, url, f.validate_lineup, season=season, event=event,
                 event_id=event['id'], transport='chrome_normal_navigation_dom',
                 retrieved_at=observation['end'], http_status=None,
                 dom_path=str(source.relative_to(base)), dom_sha256=sha(source),
                 navigation=observation)
    return True


def sync(base):
    selection = json.loads((base / 'selection.json').read_text(encoding='utf-8'))
    if len(selection) != 30 or len({x['event']['id'] for x in selection}) != 30:
        raise ValueError('Pilot requires exactly 30 distinct selected fixtures')
    if any(sum(x['season'] == y for x in selection) != 3 for y in range(2017, 2027)):
        raise ValueError('Pilot requires three fixtures per season')
    client = f.Client(raw=base / 'raw')
    observations = json.loads((base / 'navigation.json').read_text(encoding='utf-8'))
    old = f.ROOT / 'data/validation/sofascore_browser_test_20261026'
    prior = json.loads((old / 'results.json').read_text(encoding='utf-8'))
    imports = {x['event_id']: x for x in prior['lineups']}
    imported = new = 0
    missing = []
    for item in selection:
        eid = item['event']['id']
        if eid in imports:
            src = old / imports[eid]['saved_file']
            if sha(src) != imports[eid]['saved_text_sha256']:
                raise ValueError('Original saved response hash mismatch')
            dest = base / 'dom' / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            obs = dict(url=imports[eid]['url'], outcome='parsed_json', end=prior['tested_at'],
                       provenance='saved complete response from section 57; original hash checked')
            imported += import_dom(client, base, item, dest, obs)
        else:
            src = base / 'dom' / f'{eid}.lineups.json'
            obs = next((o for o in reversed(observations) if o.get('file') == src.name), None)
            if src.exists() and obs and obs['outcome'] == 'parsed_json':
                new += import_dom(client, base, item, src, obs)
        path = client.raw / 'matches' / str(item['season']) / f'{eid}.json'
        if not client.load(path, f.validate_lineup):
            missing.append(item)
    f.atomic_json(base / 'queue.json', missing)
    result = {'time': f.now(), 'imported_this_sync': imported, 'new_this_sync': new,
              'cached': 30-len(missing), 'missing': len(missing)}
    f.atomic_json(base / 'checkpoint.json', result)
    return result


def build(base):
    registry = Registry(data=base / 'current')
    registry.seed()
    return f.build(f.Client(raw=base / 'raw'), registry, data=base / 'current')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'sync', 'build', 'verify-preservation'])
    parser.add_argument('--pilot', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.pilot)
        result = {'prepared': str(args.pilot)}
    elif args.action == 'sync':
        result = sync(args.pilot)
    elif args.action == 'build':
        result = build(args.pilot)
    else:
        before = json.loads((args.pilot / 'preservation_before.json').read_text())
        after = protected()
        result = {'checked': len(before), 'changed_or_missing': [k for k,v in before.items() if after.get(k) != v],
                  'added': sorted(set(after)-set(before))}
        f.atomic_json(args.pilot / 'preservation_results.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
