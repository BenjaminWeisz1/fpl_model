"""Weekly Sofascore collection from browser downloads; never makes HTTP requests.

Ordinary requests are refused (CHANGELOG §56), so each gameweek's pages are captured
by normal Chrome navigation and saved as downloads (CHANGELOG §66):

  plan    lists the round-metadata and lineup URLs still needed for completed FPL
          fixtures, with the filename to use when saving a page by hand.
  import  validates captures in the inbox (default ~/Downloads) into the frozen raw
          cache: round metadata first (refreshing current-season discovery), then
          lineups for discovered finished events. Valid files move to
          data/raw/sofascore/weekly/imported/; invalid ones stay and are reported.
  build   rebuilds sofascore_player_match_stats.csv and coverage from the whole cache.
  status  read-only lag check; writes sofascore_status.json and always exits 0.
          Model stages may use Sofascore only when usable_for_model is true
          (built, valid, at most MAX_LAG rounds behind).
  pipeline  the run_pre_deadline.py stage: import, build only when cached lineups
          are missing from the output, then status. Never halts; problems are
          recorded in sofascore_status.json.

Capture files: `sofascore_capture_*.json` holding {"capture_version": 1, "url",
"retrieved_at", "transport", "dom_text"}, where dom_text is the page's exact
document.body.innerText. Manual saves: the raw page JSON named
`sofascore__round__<n>.json` or `sofascore__lineups__<event>.json`. Collect at least
~2 days after a round's last match; stats were stable from 36 h onward
(data/validation/sofascore_revision_check_20261028). See SOFASCORE.md.

Run locally:
    python sofascore_weekly.py plan | import | build | status
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch.player_identity import Registry, read_csv
from fpl_model.fetch.sofascore_browser_backfill import metadata_path
from fpl_model.fetch.sofascore_browser_pilot import validate_event

INBOX = Path.home() / 'Downloads'
MAX_LAG = 2
CAPTURE_GLOB = 'sofascore_capture_*.json'
MANUAL = re.compile(r'^sofascore__(?:round__(\d+)|lineups__(\d+))\.json$')
ROUND_ROUTE = re.compile(r'^/unique-tournament/17/season/(\d+)/events/round/(\d+)(?:\?page=(\d+))?$')
LINEUP_ROUTE = re.compile(r'^/event/(\d+)/lineups$')
VOLATILE_KEYS = {'userCount'}  # follower counts; observed changing on GW8 re-captures (§66)


def weekly_dir(raw):
    return raw / 'weekly'


def log(raw, result):
    path = weekly_dir(raw) / 'log.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as h:
        h.write(json.dumps({'time': f.now(), **result}) + '\n')
    return result


def discovery(raw):
    """Current-season discovered events and season ID from the frozen discovery index."""
    path = raw / 'discovery' / f'{f.CURRENT_SEASON}.json'
    env = json.loads(path.read_text(encoding='utf-8'))
    if f.digest(env['payload']) != env['sha256']:
        raise ValueError(f'Discovery hash mismatch: {path}')
    return {e['id']: e for e in env['payload']['events']}, env['payload']['season_id']


def local_teams(event):
    return tuple(f.TEAM_NAMES.get(event[s]['name'], event[s]['name']) for s in ('homeTeam', 'awayTeam'))


def fpl_fixtures(data):
    """Completed current-season FPL fixtures (Home, Away) -> Round."""
    rounds = {(r['Home'], r['Away']): int(r['Round']) for r in read_csv(data / 'fixtures.csv')
              if r['Season'] == str(f.CURRENT_SEASON)}
    completed = {(r['Home'], r['Away']) for r in read_csv(data / 'match_results.csv')
                 if r['Season'] == str(f.CURRENT_SEASON)}
    return {fx: rounds.get(fx) for fx in completed}


def cached_fixtures(raw):
    """(Home, Away) -> event ID for every cached current-season lineup (hash-checked, never quarantined)."""
    out = {}
    for path in (raw / 'matches' / str(f.CURRENT_SEASON)).glob('*.json'):
        env = json.loads(path.read_text(encoding='utf-8'))
        if f.digest(env['payload']) != env['sha256']:
            raise ValueError(f'Lineup cache hash mismatch: {path}')
        out[local_teams(env['event'])] = env['event_id']
    return out


def plan(raw, data):
    events, sid = discovery(raw)
    cached = cached_fixtures(raw)
    needed = {fx: rnd for fx, rnd in fpl_fixtures(data).items() if fx not in cached}
    known = {local_teams(e): eid for eid, e in events.items()}
    capture = []
    for rnd in sorted({r for r in needed.values() if r is not None}):
        capture.append({'kind': 'round', 'round': rnd,
                        'url': f.BASE + f'/unique-tournament/17/season/{sid}/events/round/{rnd}',
                        'manual_filename': f'sofascore__round__{rnd}.json'})
    for fx, rnd in sorted(needed.items(), key=lambda kv: (kv[1] or 0, kv[0])):
        if fx in known:
            capture.append({'kind': 'lineup', 'round': rnd, 'fixture': list(fx),
                            'url': f.BASE + f'/event/{known[fx]}/lineups',
                            'manual_filename': f'sofascore__lineups__{known[fx]}.json'})
    return {'season_id': sid, 'missing_fixtures': [dict(home=h, away=a, round=r) for (h, a), r in sorted(needed.items())],
            'capture': capture,
            'note': 'Lineup URLs for a round appear once its round page is imported; '
                    'when capturing in Chrome, read finished event IDs from the round page instead.'}


def read_inbox(inbox):
    """Yield (source file, route, retrieved_at, transport, dom bytes); route None marks a bad file."""
    for path in sorted(inbox.glob(CAPTURE_GLOB)):
        try:
            env = json.loads(path.read_text(encoding='utf-8'))
            if env.get('capture_version') != 1 or not str(env.get('url', '')).startswith(f.BASE):
                raise ValueError('not a version-1 Sofascore capture')
            datetime.fromisoformat(env['retrieved_at'].replace('Z', '+00:00'))
            yield (path, env['url'][len(f.BASE):], env['retrieved_at'],
                   env.get('transport', 'chrome_normal_navigation_download'), env['dom_text'].encode('utf-8'))
        except (ValueError, KeyError, AttributeError) as exc:
            yield path, None, None, None, str(exc)
    for path in sorted(inbox.glob('sofascore__*.json')):
        m = MANUAL.match(path.name)
        if not m:
            continue
        retrieved = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
        route = (f'/unique-tournament/17/season/{{sid}}/events/round/{m.group(1)}' if m.group(1)
                 else f'/event/{m.group(2)}/lineups')
        yield path, route, retrieved, 'manual_browser_save', path.read_bytes()


def import_inbox(raw, inbox):
    client = f.Client(raw=raw)
    events, sid = discovery(raw)
    captures, failures, imported = [], {}, Counter()
    for path, route, retrieved, transport, dom in read_inbox(inbox):
        if route is None:
            failures[path.name] = {'route': None, 'kind': 'capture_invalid', 'error': dom}
            continue
        captures.append((path, route.replace('{sid}', str(sid)), retrieved, transport, dom))
    captures.sort(key=lambda c: (LINEUP_ROUTE.match(c[1]) is not None, c[1]))
    added = {}
    done = []
    for path, route, retrieved, transport, dom in captures:
        try:
            try:
                payload = json.loads(dom.decode('utf-8'))
            except ValueError as exc:
                raise f.FetchError('non_json', 'Page text is not JSON (challenge or error page?)') from exc
            dom_path = weekly_dir(raw) / 'dom' / (hashlib.sha256(dom).hexdigest() + '.json')
            meta = dict(transport=transport, retrieved_at=retrieved, http_status=None,
                        dom_path=str(dom_path.resolve()), dom_sha256=hashlib.sha256(dom).hexdigest())
            if m := ROUND_ROUTE.match(route):
                if int(m.group(1)) != sid:
                    raise f.FetchError('schema', f'Season {m.group(1)} is not current season {sid}')
                f.validate_metadata(payload, route)
                for event in payload['events']:
                    if event['status']['type'] == 'finished' and event['startTimestamp'] <= time.time():
                        validate_event(event, f.CURRENT_SEASON, sid)
                        added[event['id']] = event
                write_dom(dom_path, dom)
                client.store(metadata_path(raw, route), payload, f.BASE + route,
                             lambda p: f.validate_metadata(p, route), refresh=True, **meta)  # noqa: B023 (stored now, this iteration)
                imported['round'] += 1
            elif m := LINEUP_ROUTE.match(route):
                eid = int(m.group(1))
                event = added.get(eid) or events.get(eid)
                if event is None:
                    raise f.FetchError('event_not_discovered', f'Event {eid}: import its round page first')
                validate_event(event, f.CURRENT_SEASON, sid)
                f.validate_lineup(payload)
                path_cache = raw / 'matches' / str(f.CURRENT_SEASON) / f'{eid}.json'
                old = client.load(path_cache, f.validate_lineup)
                if old:
                    # Follower counts change daily; any other difference is a real revision.
                    if without_volatile(old['payload']) != without_volatile(payload):
                        raise f.FetchError('cache_revision', 'Lineup or statistics differ from the frozen cache')
                    imported['lineup_unchanged'] += 1
                else:
                    write_dom(dom_path, dom)
                    client.store(path_cache, payload, f.BASE + route, f.validate_lineup,
                                 event=event, season=f.CURRENT_SEASON, event_id=eid, **meta)
                    imported['lineup_new'] += 1
            else:
                raise f.FetchError('schema', f'Unsupported route {route}')
            done.append(path)
        except (f.FetchError, ValueError, KeyError) as exc:
            failures[path.name] = {'route': route, 'kind': getattr(exc, 'kind', 'schema'), 'error': str(exc)}
    new = {eid: e for eid, e in added.items() if eid not in events}
    for eid, e in added.items():
        old = events.get(eid)
        if old and (old['homeTeam']['id'], old['awayTeam']['id'], old['startTimestamp']) != (
                e['homeTeam']['id'], e['awayTeam']['id'], e['startTimestamp']):
            raise ValueError(f'Conflicting rediscovered event {eid}; review before importing')
    if new:
        merged = sorted({**events, **new}.values(), key=lambda e: (e['startTimestamp'], e['id']))
        client.store(raw / 'discovery' / f'{f.CURRENT_SEASON}.json', {'events': merged, 'season_id': sid},
                     f.BASE + f'/unique-tournament/17/season/{sid}/rounds', refresh=True,
                     transport='derived_from_browser_round_envelopes')
    archive = weekly_dir(raw) / 'imported' / datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    for path in done:
        archive.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(archive / path.name))
    return log(raw, {'action': 'import', 'imported': dict(imported), 'events_discovered': len(new),
                     'failures': failures, 'archived_to': str(archive) if done else None})


def without_volatile(value):
    """Payload without fields that change after the match without being match data."""
    if isinstance(value, dict):
        return {k: without_volatile(v) for k, v in value.items() if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [without_volatile(v) for v in value]
    return value


def write_dom(path, dom):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_bytes(dom)
    tmp.replace(path)


def build(raw, data):
    registry = Registry(data)
    registry.seed()
    return log(raw, {'action': 'build', **f.build(f.Client(raw=raw), registry, data)})


def unbuilt_events(raw, data):
    built = {r['Event_ID_Sofascore'] for r in read_csv(data / 'sofascore_player_match_stats.csv')
             if r['Season'] == str(f.CURRENT_SEASON)}
    return sorted(str(e) for e in cached_fixtures(raw).values() if str(e) not in built)


def pipeline(raw, data, inbox):
    """import -> build (only when cached lineups are missing from the output) -> status; never raises.

    Failures are recorded in sofascore_status.json, so run_pre_deadline.py continues and
    models fall back to the §64 baseline whenever usable_for_model is false.
    """
    errors, warnings = [], []
    try:
        result = import_inbox(raw, inbox)
        for name, failure in result['failures'].items():
            warnings.append(f"{name} left in inbox: {failure['kind']}: {failure['error']}")
        print(f"Sofascore import: {result['imported'] or 'nothing new'}")
    except Exception as exc:  # supplementary data must never stop the forecast
        errors.append(f'import failed: {type(exc).__name__}: {exc}')
    try:
        if unbuilt_events(raw, data):
            summary = build(raw, data)
            print(f"Sofascore build: {summary['rows']} player rows")
        else:
            print('Sofascore build: outputs already current, skipped')
    except Exception as exc:  # build raises before replacing any good output
        errors.append(f'build failed: {type(exc).__name__}: {exc}')
    return status(raw, data, errors, warnings)


def status(raw, data, errors=(), warnings=()):
    result = {'checked_at': f.now(), 'max_lag_rounds': MAX_LAG, 'errors': list(errors),
              'warnings': list(warnings)}
    try:
        fixtures = fpl_fixtures(data)
        cached = cached_fixtures(raw)
        rounds = sorted({r for r in fixtures.values() if r is not None})
        # Round 0 marks a completed fixture with no round mapping: reported, never "covered".
        missing = sorted((r or 0, h, a) for (h, a), r in fixtures.items() if (h, a) not in cached)
        covered = 0
        for rnd in rounds:
            if any(r == rnd for r, _, _ in missing):
                break
            covered = rnd
        unbuilt = unbuilt_events(raw, data)
        result.update(fpl_latest_round=rounds[-1] if rounds else None, sofascore_complete_through_round=covered,
                      lag_rounds=(rounds[-1] - covered) if rounds else 0,
                      missing_fixtures=[dict(round=r, home=h, away=a) for r, h, a in missing],
                      cached_not_built=unbuilt)
        if unbuilt:
            result['errors'].append('Cached lineups missing from sofascore_player_match_stats.csv: run build')
    except (OSError, ValueError, KeyError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    result['usable_for_model'] = not result['errors'] and result.get('lag_rounds', MAX_LAG + 1) <= MAX_LAG
    f.atomic_json(data / 'sofascore_status.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['plan', 'import', 'build', 'status', 'pipeline'])
    p.add_argument('--raw', type=Path, default=f.RAW)
    p.add_argument('--data', type=Path, default=f.DATA)
    p.add_argument('--inbox', type=Path, default=INBOX)
    a = p.parse_args(argv)
    if a.action == 'plan':
        result = plan(a.raw, a.data)
    elif a.action == 'import':
        result = import_inbox(a.raw, a.inbox)
    elif a.action == 'build':
        result = build(a.raw, a.data)
    else:
        result = pipeline(a.raw, a.data, a.inbox) if a.action == 'pipeline' else status(a.raw, a.data)
        lag = result.get('lag_rounds')
        print(f"Sofascore: complete through round {result.get('sofascore_complete_through_round')}, "
              f"FPL through {result.get('fpl_latest_round')} (lag {lag}); "
              f"usable for model: {result['usable_for_model']}")
        for message in result['errors'] + result['warnings']:
            print('WARNING: ' + message)
        if lag and lag > MAX_LAG:
            print(f'WARNING: Sofascore is {lag} rounds behind; models fall back to the §64 baseline')
        return 0
    print(json.dumps(result, indent=2))
    return 1 if result.get('failures') else 0


if __name__ == '__main__':
    raise SystemExit(main())
