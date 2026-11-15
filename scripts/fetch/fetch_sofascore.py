"""Supplementary EPL lineup statistics, 2017 onward; never a model input.

Atomic validated raw envelopes, bounded retries/deferred queue, frozen historical
discovery and fresh weekly metadata. Requires documented permitted access for bulk
network runs. --probe makes only the four agreed lineup checks plus seasons metadata.
--build-only needs no network. Exit 2 = incomplete, 1 = integrity/schema error;
--weekly tolerates access/incomplete outcomes only. See SOFASCORE.md.
"""
import sys
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
from pathlib import Path
import time

import requests

from player_identity import Registry, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/raw/sofascore'
DATA = ROOT / 'data/current'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()
BASE = 'https://www.sofascore.com/api/v1'
FIELDS = ('minutesPlayed totalPass accuratePass totalCross accurateCross bigChanceCreated '
          'bigChanceMissed totalContest wonContest totalClearance clearanceOffLine '
          'outfielderBlock interceptionWon ballRecovery totalTackle wonTackle '
          'onTargetScoringAttempt shotOffTarget blockedScoringAttempt hitWoodwork '
          'fouls wasFouled totalOffside errorLeadToAShot errorLeadToAGoal dispossessed '
          'keyPass saves savedShotsFromInsideTheBox penaltyConceded penaltySave '
          'penaltyMiss penaltyWon goals goalAssist ownGoals expectedGoals expectedAssists').split()


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class FetchError(Exception):
    def __init__(self, kind, message, retryable=False):
        super().__init__(message)
        self.kind, self.retryable = kind, retryable


def validate_lineup(payload):
    if not isinstance(payload, dict):
        raise FetchError('schema', 'Lineup must be an object')
    seen = set()
    for side in ('home', 'away'):
        players = payload.get(side, {}).get('players')
        if not isinstance(players, list) or not players:
            raise FetchError('unavailable', f'{side} lineup absent/empty')
        for row in players:
            p = row.get('player', {})
            if not isinstance(p.get('id'), int) or not p.get('name') or not isinstance(row.get('statistics', {}), dict):
                raise FetchError('schema', 'Invalid player identity/statistics')
            if not isinstance(row.get('substitute'), bool):
                raise FetchError('schema', 'Missing substitute flag')
            if p['id'] in seen:
                raise FetchError('schema', 'Duplicate player in source event')
            seen.add(p['id'])
            for key in FIELDS:
                value = row.get('statistics', {}).get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    raise FetchError('schema', f'Invalid numeric field {key}')
        if sum(not p['substitute'] for p in players) != 11:
            raise FetchError('schema', f'{side} must have 11 starters')


def validate_metadata(payload, route):
    if route.endswith('/seasons'):
        values = payload.get('seasons')
        if not isinstance(values, list) or not values or any(not isinstance(s.get('id'), int) or season_year(s) is None for s in values):
            raise FetchError('schema', 'Invalid seasons list')
    elif route.endswith('/rounds'):
        values = payload.get('rounds')
        if not isinstance(values, list) or not values or any(not isinstance(r.get('round'), int) for r in values):
            raise FetchError('schema', 'Invalid rounds list')
    elif '/events/round/' in route:
        if not isinstance(payload.get('events'), list):
            raise FetchError('schema', 'Invalid event list')
        for event in payload['events']:
            if (not isinstance(event.get('id'), int) or not isinstance(event.get('startTimestamp'), (int, float))
                    or not event.get('homeTeam', {}).get('name') or not event.get('awayTeam', {}).get('name')
                    or not event.get('status', {}).get('type')):
                raise FetchError('schema', 'Invalid fixture metadata')


class Client:
    def __init__(self, raw=RAW, spacing=6, attempts=4, session=None, sleep=time.sleep):
        self.raw, self.spacing, self.attempts = Path(raw), spacing, attempts
        self.session, self.sleep = session or requests.Session(), sleep
        self.last = None
        self.requests = 0
        self.observations = []

    def get(self, route):
        url = BASE + route
        for attempt in range(self.attempts):
            if self.last is not None:
                self.sleep(max(0, self.spacing - (time.monotonic() - self.last)))
            self.last = time.monotonic()
            self.requests += 1
            start = time.monotonic()
            try:
                response = self.session.get(url, timeout=(10, 40))
            except (requests.Timeout, requests.ConnectionError) as exc:
                error = FetchError('transport', str(exc), True)
                self.observations.append({'url': url, 'time': now(), 'kind': 'transport', 'error': str(exc),
                                          'elapsed_seconds': time.monotonic() - start})
            else:
                self.observations.append({'url': url, 'time': now(), 'status': response.status_code,
                                          'content_type': response.headers.get('Content-Type'),
                                          'elapsed_seconds': time.monotonic() - start,
                                          'body_sha256': hashlib.sha256(response.content).hexdigest()})
                status = response.status_code
                if status in (401, 403):
                    raise FetchError('access', f'HTTP {status}: {url}')
                if status == 429 or status in (408, 500, 502, 503, 504):
                    error = FetchError('rate_limit' if status == 429 else 'server', f'HTTP {status}: {url}', True)
                    retry = response.headers.get('Retry-After')
                    if retry and attempt + 1 < self.attempts:
                        try:
                            delay = float(retry)
                        except ValueError:
                            delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                        self.sleep(max(0, delay))
                else:
                    try:
                        response.raise_for_status()
                    except requests.HTTPError as exc:
                        raise FetchError('unavailable' if status in (404, 410) else 'http', str(exc)) from exc
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise FetchError('non_json', f'Non-JSON/challenge: {url}') from exc
                    if not isinstance(payload, dict):
                        raise FetchError('schema', f'Expected object: {url}')
                    return payload, url
            if attempt + 1 < self.attempts:
                self.sleep(min(60, 2 ** (attempt + 1)))
        raise error

    def load(self, path, validator=None):
        if not path.exists():
            return None
        try:
            env = json.loads(path.read_text(encoding='utf-8'))
            if digest(env['payload']) != env['sha256'] or not env['retrieved_at'] or not env['url']:
                raise ValueError('Missing provenance/hash mismatch')
            if validator:
                validator(env['payload'])
            return env
        except (ValueError, KeyError, FetchError) as exc:
            # Preserve corrupt bytes for diagnosis. Never trust a partial cache.
            quarantine = path.with_name(path.name + '.invalid.' + str(time.time_ns()))
            path.replace(quarantine)
            raise FetchError('cache_invalid', f'{path}: {exc}; quarantined at {quarantine}') from exc

    def store(self, path, payload, url, validator=None, refresh=False, **metadata):
        if validator:
            validator(payload)
        env = {'payload': payload, 'sha256': digest(payload), 'retrieved_at': now(), 'url': url, **metadata}
        if path.exists():
            old = self.load(path, validator)
            if old['sha256'] == env['sha256']:
                return old
            if not refresh:
                raise FetchError('cache_revision', 'Refusing to overwrite frozen successful response')
            # Explicit metadata refresh preserves dated original response bytes.
            path.replace(path.with_name(path.name + '.revision.' + str(time.time_ns())))
        atomic_json(path, env)
        return env

    def cached_get(self, route, relative, refresh=False):
        path = self.raw / relative
        validator = lambda p: validate_metadata(p, route)
        old = self.load(path, validator)
        if old and not refresh:
            return old['payload']
        payload, url = self.get(route)
        self.store(path, payload, url, validator, refresh=refresh)
        return payload

    def match(self, season, event):
        path = self.raw / 'matches' / str(season) / f'{event["id"]}.json'
        try:
            old = self.load(path, validate_lineup)
        except FetchError as exc:
            if exc.kind != 'cache_invalid':
                raise
            old = None  # download repairs quarantined truncated caches
        if old:
            return False
        payload, url = self.get(f'/event/{event["id"]}/lineups')
        self.store(path, payload, url, validate_lineup, event_id=event['id'], season=season, event=event)
        return True


def season_year(season):
    label = str(season.get('year', season.get('name', '')))
    first = label.split('/')[0].strip()
    try:
        value = int(first)
        return value + 2000 if value < 100 else value
    except ValueError:
        return None


def discover(client, season, season_id, refresh=False):
    index = client.raw / 'discovery' / f'{season}.json'
    old = client.load(index)
    if old and not refresh:
        return old['payload']['events']
    rounds = client.cached_get(f'/unique-tournament/17/season/{season_id}/rounds',
                               f'metadata/{season}/rounds.json', refresh)
    if not isinstance(rounds.get('rounds'), list) or not rounds['rounds']:
        raise FetchError('schema', 'No discoverable rounds')
    events = {}
    for r in rounds['rounds']:
        number = r.get('round')
        if not isinstance(number, int):
            raise FetchError('schema', 'Unsupported round identifier')
        page = 0
        while True:
            route = f'/unique-tournament/17/season/{season_id}/events/round/{number}'
            if page:
                route += f'?page={page}'
            batch = client.cached_get(route, f'metadata/{season}/round_{number}_{page}.json', refresh)
            if not isinstance(batch.get('events'), list):
                raise FetchError('schema', 'Missing round events array')
            for event in batch['events']:
                if not all(k in event for k in ('id', 'homeTeam', 'awayTeam', 'status', 'startTimestamp')):
                    raise FetchError('schema', 'Invalid fixture metadata')
                if event['status'].get('type') == 'finished':
                    if event['startTimestamp'] > time.time():
                        raise FetchError('schema', 'Finished fixture has future kickoff')
                    previous = events.get(event['id'])
                    if previous and previous != event:
                        raise FetchError('schema', 'Conflicting event enumeration')
                    events[event['id']] = event
            if batch.get('hasNextPage') is not True:
                break
            page += 1
            if page > 100:
                raise FetchError('schema', 'Pagination did not terminate')
    result = sorted(events.values(), key=lambda e: e['id'])
    # This independent local archive establishes the known minimum, never source IDs.
    local = {r['MatchID_Understat'] for r in read_csv(DATA / 'understat_match_positions.csv')
             if r['Season'] == str(season)}
    minimum = 380 if season < CURRENT_SEASON else len(local)
    if len(result) < minimum:
        raise FetchError('discovery_gap', f'{len(result)} enumerated versus known minimum {minimum}')
    client.store(index, {'events': result, 'season_id': season_id}, BASE + f'/unique-tournament/17/season/{season_id}/rounds', refresh=refresh)
    return result


def download(client, events_by_season, limit=None, retry_passes=2, threshold=3):
    failures, successes, consecutive, stopped = {}, 0, 0, False
    queue = [(s, e) for s, events in events_by_season.items() for e in events]
    for pass_number in range(retry_passes + 1):
        pending = []
        for season, event in queue:
            key = f'{season}:{event["id"]}'
            path = client.raw / 'matches' / str(season) / f'{event["id"]}.json'
            if limit is not None and successes >= limit:
                try:
                    valid = client.load(path, validate_lineup)
                except FetchError:
                    valid = None
                if not valid:
                    failures[key] = {'kind': 'limit', 'message': 'New-success limit reached'}
                    continue
            try:
                new = client.match(season, event)
                successes += int(new)
                consecutive = 0
                failures.pop(key, None)
            except FetchError as exc:
                failures[key] = {'kind': exc.kind, 'message': str(exc), 'pass': pass_number}
                if exc.retryable or exc.kind == 'unavailable':
                    pending.append((season, event))
                if exc.kind in ('transport', 'server', 'rate_limit', 'access', 'non_json'):
                    consecutive += 1
                else:
                    consecutive = 0
                atomic_json(client.raw / 'queue.json', failures)
                if consecutive >= threshold:
                    stopped = True
                    break
        if stopped or not pending:
            break
        queue = pending
    atomic_json(client.raw / 'queue.json', failures)
    return failures, successes, stopped


TEAM_NAMES = {'Liverpool FC': 'Liverpool', 'West Ham United': 'West Ham',
              'Sheffield United': 'Sheffield Utd',
              'Huddersfield Town': 'Huddersfield',
              'Manchester United': 'Manchester Utd', 'Newcastle United': 'Newcastle Utd',
              'Nottingham Forest': "Nott'ham Forest", 'Tottenham Hotspur': 'Tottenham',
              'Wolverhampton': 'Wolves', 'Wolverhampton Wanderers': 'Wolves',
              'Brighton & Hove Albion': 'Brighton', 'Leicester': 'Leicester City',
              'West Bromwich Albion': 'West Brom', 'Leeds': 'Leeds United'}


def build(client, registry, data=DATA):
    """All cached seasons, preserving omitted/null statistics and unused bench rows."""
    for quarantined in (client.raw / 'matches').glob('*/*.json.invalid.*'):
        original = quarantined.with_name(quarantined.name.split('.invalid.')[0])
        if not original.exists():
            raise FetchError('cache_invalid', f'Unrepaired quarantined cache: {quarantined}')
    rows, present, eligible, season_rows = [], Counter(), Counter(), Counter()
    local, local_minutes = defaultdict(list), defaultdict(dict)
    for r in read_csv(data / 'player_match_stats.csv'):
        side = (r['Season'], r['Team'], r['Opponent'], r['Venue'])
        local[side].append(r['Player'])
        if r.get('Minutes') not in (None, ''):
            local_minutes[side][r['Player']] = float(r['Minutes'])
    for path in sorted((client.raw / 'matches').glob('*/*.json')):
        env = client.load(path, validate_lineup)
        payload, event, season = env['payload'], env['event'], env['season']
        if str(event['id']) != path.stem or str(season) != path.parent.name:
            raise ValueError('Cache event/season path mismatch')
        home = TEAM_NAMES.get(event['homeTeam']['name'], event['homeTeam']['name'])
        away = TEAM_NAMES.get(event['awayTeam']['name'], event['awayTeam']['name'])
        kickoff = datetime.fromtimestamp(event['startTimestamp'], timezone.utc).isoformat()
        fixture = registry.fixture('sofascore', event['id'], season, home, away, kickoff)
        for side, team, opp, venue in [('home', home, away, 'Home'), ('away', away, home, 'Away')]:
            players = payload[side]['players']
            side_key = (str(season), team, opp, venue)
            identities = registry.roster('sofascore', [(p['player']['id'], p['player']['name']) for p in players],
                                         season, team, local[side_key],
                                         minutes={str(p['player']['id']): p.get('statistics', {}).get('minutesPlayed')
                                                  for p in players},
                                         local_minutes=local_minutes[side_key])
            for p in players:
                stats, player = p.get('statistics', {}), p['player']
                cid, status = identities[str(player['id'])]
                minutes = stats.get('minutesPlayed')
                # Missing bench minutes alone does not establish an unused substitute.
                state = ('played' if minutes is not None and minutes > 0 else
                         'unused_bench' if p['substitute'] and minutes == 0 else
                         'bench_unknown' if p['substitute'] else 'starter_minutes_unknown')
                aliases = [n for n in local[(str(season), team, opp, venue)]
                           if registry.context.get((str(season), team, n)) == cid]
                row = {'Season': season, 'Event_ID_Sofascore': event['id'], 'Player_ID_Sofascore': player['id'],
                       'Canonical_Player_ID': cid, 'Canonical_Fixture_ID': fixture, 'Player_Sofascore': player['name'],
                       'Local_Player': aliases[0] if len(set(aliases)) == 1 else '', 'Identity_Status': status,
                       'Team': team, 'Opponent': opp, 'Venue': venue, 'Kickoff': kickoff,
                       'Is_Starter': not p['substitute'], 'Appearance_Status': state,
                       'Position_Sofascore': player.get('position'), 'Formation_Sofascore': payload[side].get('formation'),
                       'Raw_Path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                       'Raw_SHA256': env['sha256'], 'Retrieved_At': env['retrieved_at'], 'Source_URL': env['url'],
                       'Present_Fields': json.dumps(sorted(stats))}
                for key in FIELDS:
                    row['Sofascore_' + key] = stats.get(key)
                    eligible[(season, state, key)] += 1
                    present[(season, state, key)] += int(key in stats and stats[key] is not None)
                rows.append(row)
                season_rows[season] += 1
    # A source-only player resolved in a later fixture re-points its earlier rows.
    for row in rows:
        cid = registry.promoted.get(('sofascore', str(row['Player_ID_Sofascore'])))
        if cid and row['Canonical_Player_ID'] != cid:
            names = [n for n in local[(str(row['Season']), row['Team'], row['Opponent'], row['Venue'])]
                     if registry.context.get((str(row['Season']), row['Team'], n)) == cid]
            row.update(Canonical_Player_ID=cid, Identity_Status='provider_id',
                       Local_Player=names[0] if len(set(names)) == 1 else '')
    keys = [(r['Event_ID_Sofascore'], r['Player_ID_Sofascore']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate source player-event rows')
    canonical_keys = [(r['Event_ID_Sofascore'], r['Canonical_Player_ID']) for r in rows]
    if len(canonical_keys) != len(set(canonical_keys)):
        raise ValueError('Duplicate canonical player across source event sides')
    # Integrity failures above occur before any good output is replaced.
    if rows:
        registry.save()
        write_csv(data / 'sofascore_player_match_stats.csv', rows, list(rows[0]))
    coverage = [{'Season': s, 'Appearance_Status': status, 'Field': field,
                 'Present_Nonnull': present[(s, status, field)], 'Rows': count}
                for (s, status, field), count in sorted(eligible.items())]
    if coverage or not (data / 'sofascore_field_coverage.csv').exists():
        write_csv(data / 'sofascore_field_coverage.csv', coverage,
                  ['Season', 'Appearance_Status', 'Field', 'Present_Nonnull', 'Rows'])
    return {'player_rows': dict(season_rows), 'identity_status': dict(Counter(r['Identity_Status'] for r in rows)),
            'rows': len(rows), 'unique_player_events': len(set(keys))}


def manifest(client, seasons, discovery_errors, failures, summary):
    counts = {}
    local = defaultdict(set)
    for r in read_csv(DATA / 'understat_match_positions.csv'):
        local[int(r['Season'])].add(r['MatchID_Understat'])
    for season in range(2017, CURRENT_SEASON + 1):
        path = client.raw / 'discovery' / f'{season}.json'
        env = client.load(path)
        events = env['payload']['events'] if env else []
        cached_ids = {p.stem for p in (client.raw / 'matches' / str(season)).glob('*.json')
                      if client.load(p, validate_lineup)}
        ids = {str(e['id']) for e in events}
        unavailable = sum(v['kind'] == 'unavailable' for k, v in failures.items() if k.startswith(f'{season}:'))
        counts[str(season)] = {'expected': len(ids) if env else None, 'cached_valid': len(cached_ids),
                              'known_local_played': len(local[season]),
                              'missing': len(ids - cached_ids) if env else None,
                              'deferred': len(ids - cached_ids) - unavailable if env else None,
                              'unsupported_unavailable': unavailable, 'discovery_unverified': int(not env or season in discovery_errors),
                              'discovery_error': discovery_errors.get(season),
                              'player_rows': summary['player_rows'].get(season, 0)}
    complete = all(c['discovery_unverified'] == 0 and c['missing'] == 0 for c in counts.values())
    result = {'updated_at': now(), 'complete': complete, 'seasons': counts, 'summary': summary,
              'requests_this_run': client.requests, 'request_observations': client.observations,
              'resume_command': 'python -u -X utf8 scripts/fetch/fetch_sofascore.py --access-evidence PATH'}
    atomic_json(client.raw / 'manifest.json', result)
    return result


def probe(client):
    observations = []
    for event in (7437077, 14025013, 14025059, 16363891):
        start = time.monotonic()
        try:
            payload, url = client.get(f'/event/{event}/lineups')
            validate_lineup(payload)
            client.store(client.raw / 'pilots' / f'{event}.json', payload, url, validate_lineup, event_id=event)
            observations.append({'event_id': event, 'result': 'valid_cached'})
        except FetchError as exc:
            observations.append({'event_id': event, 'kind': exc.kind, 'error': str(exc)})
        observations[-1]['elapsed_seconds'] = time.monotonic() - start
    try:
        payload, url = client.get('/unique-tournament/17/seasons')
        client.store(client.raw / 'pilots/seasons.json', payload, url, refresh=True)
    except FetchError as exc:
        observations.append({'route': 'seasons', 'kind': exc.kind, 'error': str(exc)})
    atomic_json(client.raw / 'access_probe.json', {'observations': observations, 'requests': client.observations,
                                                'request_count': client.requests, 'time': now()})
    return observations


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--season', type=int, action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--weekly', action='store_true')
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--spacing', type=float, default=6)
    parser.add_argument('--retry-passes', type=int, default=2)
    parser.add_argument('--refresh-metadata', action='store_true')
    parser.add_argument('--access-evidence', type=Path, help='JSON: permitted_method, permission_reference, unattended_pilot_passed=true')
    args = parser.parse_args(argv)
    if args.spacing < 0 or args.retry_passes < 0 or (args.limit is not None and args.limit < 0):
        parser.error('Spacing, retry passes and limit must be nonnegative')
    seasons = [CURRENT_SEASON] if args.weekly else args.season or list(range(2017, CURRENT_SEASON + 1))
    if any(s < 2017 or s > CURRENT_SEASON for s in seasons):
        parser.error('Season outside supported range')
    client = Client()
    client.spacing = args.spacing
    if args.probe:
        observations = probe(client)
        print(json.dumps(observations, indent=2))
        return 2 if any('error' in r for r in observations) else 0
    start = time.monotonic()
    registry = Registry()
    registry.seed()
    registry.save()
    # Offline rebuilds must not erase the last network pass's discovery failures.
    prior_path = client.raw / 'manifest.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8')) if prior_path.exists() else {}
    errors = {int(s): c['discovery_error'] for s, c in prior.get('seasons', {}).items() if c.get('discovery_error')}
    queue_path = client.raw / 'queue.json'
    failures = json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else {}
    if not args.build_only:
        evidence = json.loads(args.access_evidence.read_text(encoding='utf-8')) if args.access_evidence else {}
        if not (evidence.get('permitted_method') and evidence.get('permission_reference')
                and evidence.get('unattended_pilot_passed') is True):
            errors = {s: 'Access not established: no documented permitted unattended method' for s in seasons}
        else:
            discovered = {}
            try:
                listing = client.cached_get('/unique-tournament/17/seasons', 'metadata/seasons.json', args.weekly or args.refresh_metadata)
                ids = {season_year(s): s['id'] for s in listing['seasons']}
            except (FetchError, KeyError) as exc:
                errors = {s: str(exc) for s in seasons}
                ids = {}
            consecutive = 0
            for season in seasons:
                if season not in ids:
                    errors[season] = errors.get(season, 'Season missing from provider listing')
                    continue
                try:
                    discovered[season] = discover(client, season, ids[season], args.refresh_metadata or season == CURRENT_SEASON)
                    errors.pop(season, None)
                    consecutive = 0
                except FetchError as exc:
                    errors[season] = f'{exc.kind}: {exc}'
                    consecutive += 1
                    if consecutive >= 3:
                        for remaining in seasons:
                            if remaining not in discovered:
                                errors.setdefault(remaining, 'Discovery stopped after 3 consecutive failures')
                        break
            new_failures, _, _ = download(client, discovered, args.limit, args.retry_passes)
            failures = {k: v for k, v in failures.items() if int(k.split(':')[0]) not in discovered}
            failures.update(new_failures)
            atomic_json(client.raw / 'queue.json', failures)
            if any(v['kind'] in ('schema', 'cache_revision') for v in new_failures.values()):
                atomic_json(client.raw / 'integrity_failure.json', {'time': now(), 'failures': new_failures})
                raise ValueError('Source schema/integrity failure; previous enrichment retained; inspect queue.json')
    summary = build(client, registry)
    report = manifest(client, seasons, errors, failures, summary)
    report['elapsed_seconds'] = time.monotonic() - start
    atomic_json(client.raw / 'manifest.json', report)
    if not report['complete']:
        print('INCOMPLETE: ' + json.dumps(report['seasons']))
        print('Resume: ' + report['resume_command'])
    print(f'{summary["rows"]} player rows; {client.requests} requests; {report["elapsed_seconds"]:.2f}s')
    return 0 if args.weekly or report['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
