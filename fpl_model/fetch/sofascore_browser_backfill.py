"""Offline coordinator for full-history tool-driven Chrome collection (no HTTP).

prepare checkpoints protected files externally before calling this module. import-pilot
retains original envelopes and DOM provenance; plan/sync turn saved browser responses
into validated frozen cache, complete discovery indexes, and a durable dynamic queue.
The connected Chrome collector consumes jobs.json, one normal navigation at a time.
Explicit --raw/--data/--run paths permit shadow validation and reliable resume.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch.player_identity import Registry, read_csv
from fpl_model.fetch.sofascore_browser_pilot import sha, validate_event

DEFAULT_RUN = f.ROOT / 'data/validation/sofascore_full_backfill_20261026'


def route_key(route):
    return hashlib.sha256(route.encode()).hexdigest()[:24]


def metadata_path(raw, route):
    return raw / 'browser_metadata' / (route_key(route) + '.json')


def log(run, result):
    with (run / 'progress.jsonl').open('a', encoding='utf-8') as h:
        h.write(json.dumps({'time': f.now(), **result}) + '\n')
    return result


def prepare(run, data):
    run.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    for name in (*Registry.SCHEMAS, 'player_match_stats', 'understat_fpl_crosswalk',
                 'player_match_positions', 'understat_match_positions', 'players_current'):
        src = f.DATA / (name + '.csv')
        dst = data / src.name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    for name, default in [('navigation.json', []), ('passes.json', [])]:
        if not (run / name).exists():
            f.atomic_json(run / name, default)
    return log(run, {'action': 'prepare', 'data': str(data)})


def import_pilot(raw, data, run, pilot):
    """Validate originals before copying; merge additions without changing accepted IDs."""
    client = f.Client(raw=raw)
    selection = json.loads((pilot / 'selection.json').read_text(encoding='utf-8'))
    if len(selection) != 30 or len({r['event']['id'] for r in selection}) != 30:
        raise ValueError('Expected original 30-fixture pilot')
    imported = 0
    evidence = []
    for item in selection:
        event, season = item['event'], item['season']
        validate_event(event, season, item['season_id'])
        src = pilot / 'raw/matches' / str(season) / f'{event["id"]}.json'
        # Use non-mutating validation to preserve irreplaceable pilot evidence.
        env = json.loads(src.read_text(encoding='utf-8'))
        if env['sha256'] != f.digest(env['payload']) or env['event'] != event:
            raise ValueError('Pilot envelope content/metadata mismatch')
        f.validate_lineup(env['payload'])
        dom = pilot / env['dom_path']
        if sha(dom) != env['dom_sha256'] or json.loads(dom.read_text(encoding='utf-8')) != env['payload']:
            raise ValueError('Pilot DOM mismatch')
        dest_dom = raw / 'dom/imported_pilot' / dom.name
        dest_dom.parent.mkdir(parents=True, exist_ok=True)
        if dest_dom.exists() and sha(dest_dom) != sha(dom):
            raise ValueError('Frozen imported DOM conflict')
        if not dest_dom.exists():
            shutil.copy2(dom, dest_dom)
        dest = raw / 'matches' / str(season) / src.name
        old = client.load(dest, f.validate_lineup)
        if old and old['sha256'] != env['sha256']:
            raise ValueError('Frozen imported payload conflict')
        if not old:
            copied = dict(env, dom_path=str(dest_dom.resolve()),
                          imported_from=str(src.resolve()), original_envelope_sha256=sha(src))
            f.atomic_json(dest, copied)
            imported += 1
        evidence.append({'season': season, 'event_id': event['id'], 'source_envelope_sha256': sha(src),
                         'payload_sha256': env['sha256'], 'dom_sha256': sha(dom)})
    registry, prior = Registry(data), Registry(pilot / 'current')
    for name, rows in prior.tables.items():
        if name == 'player_identity_registry':
            for row in rows:
                cid = row['Canonical_Player_ID']
                if cid not in registry.players:
                    registry.players[cid] = dict(row)
                    registry.tables[name].append(dict(row))
        elif name == 'player_identity_aliases':
            for row in rows:
                if row['Provider'] == 'sofascore':
                    registry.add_alias(row['Provider'], row['Provider_Player_ID'], row['Canonical_Player_ID'],
                                       row['Name'], row['Season'], row['Team'], row['Evidence'])
        elif name == 'fixture_identity_registry':
            for row in rows:
                if row['Provider'] == 'sofascore':
                    existing=registry.fixtures.get(('sofascore',row['Event_ID']))
                    if existing:
                        def names(r):
                            return tuple(f.TEAM_NAMES.get(r[k],r[k]) for k in ('Home','Away'))
                        if existing['Season']!=row['Season'] or names(existing)!=names(row):
                            raise ValueError('Pilot fixture membership conflict requires review')
                        cid=existing['Canonical_Fixture_ID']
                    else:
                        cid = registry.fixture(row['Provider'], row['Event_ID'], row['Season'],
                                               row['Home'], row['Away'], row['Kickoff'])
                    if cid != row['Canonical_Fixture_ID']:
                        raise ValueError('Pilot fixture identity migration requires review')
    registry.save()
    f.atomic_json(run / 'pilot_import.json', evidence)
    return log(run, {'action': 'import-pilot', 'imported_this_run': imported, 'validated': len(evidence)})


def normalize_fixture_aliases(raw,data,run):
    """Reviewed team-label normalization only; never replace an accepted project ID."""
    registry=Registry(data)
    changes=[]
    for row in registry.tables['fixture_identity_registry']:
        if row['Provider']!='sofascore':
            continue
        normalized={k:f.TEAM_NAMES.get(row[k],row[k]) for k in ('Home','Away')}
        if all(row[k]==v for k,v in normalized.items()):
            continue
        path=raw/'matches'/row['Season']/(row['Event_ID']+'.json')
        env=f.Client(raw=raw).load(path,f.validate_lineup)
        if not env:
            raise ValueError('Alias review requires actual raw event')
        event=env['event']
        actual={k:f.TEAM_NAMES.get(event[s]['name'],event[s]['name']) for k,s in [('Home','homeTeam'),('Away','awayTeam')]}
        if actual!=normalized or str(env['season'])!=row['Season']:
            raise ValueError('Alias review source membership mismatch')
        candidates=[dict(r) for r in registry.tables['fixture_identity_registry']
                    if r['Provider']=='understat' and r['Season']==row['Season']
                    and all(r[k]==v for k,v in normalized.items())]
        changes.append({'before':dict(row),'after':dict(row,**normalized),'raw_path':str(path),
                        'raw_sha256':env['sha256'],'local_fixture_candidates':candidates,
                        'decision':'Normalize known team spelling; retain accepted Sofascore fixture ID and all Understat IDs. No ID merge.'})
        row.update(normalized)
    if changes:
        report=run/'fixture_alias_review.json'
        prior=json.loads(report.read_text(encoding='utf-8')) if report.exists() else []
        f.atomic_json(report,prior+changes)
        registry.save()
    return log(run,{'action':'normalize-fixture-aliases','reviewed_changes':len(changes),'accepted_ids_changed':0})


def sync(raw, run):
    client = f.Client(raw=raw)
    index_path=run/'validated_cache_index.json'
    index=json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else {}
    def remember(path,env,observation_end=None):
        stat=path.stat()
        index[str(path)]={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                          'retrieved_at':env['retrieved_at'],'payload_sha256':env['sha256'],
                          'observation_end':observation_end}
    def validated_cached(path,validator):
        if path.exists():
            stat=path.stat()
            record=index.get(str(path))
            if record and (record['size'],record['mtime_ns'])==(stat.st_size,stat.st_mtime_ns):
                return record
        env=client.load(path,validator)
        if env:
            remember(path,env)
        return env
    jobs = json.loads((run / 'jobs.json').read_text(encoding='utf-8')) if (run / 'jobs.json').exists() else []
    observations = json.loads((run / 'navigation.json').read_text(encoding='utf-8'))
    count = 0
    failure_path=run/'validation_failures.json'
    failures=json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
    for job in jobs:
        path = Path(job['cache'])
        validator = f.validate_lineup if job['kind'] == 'lineup' else lambda p: f.validate_metadata(p, job['route'])  # noqa: B023 (used this iteration)
        old = validated_cached(path, validator)
        obs = next((o for o in reversed(observations)
                    if o.get('route') == job['route'] and o.get('outcome') == 'parsed_json'), None)
        if not obs:
            continue
        if obs.get('url') != f.BASE + job['route'] or not obs.get('end'):
            raise ValueError('Browser observation URL/time does not match queued route')
        if old and (old.get('observation_end')==obs['end'] or old.get('retrieved_at', '') >= obs['end']):
            continue
        src = Path(obs['file'])
        payload = json.loads(src.read_text(encoding='utf-8'))
        metadata = dict(transport='chrome_normal_navigation_dom', retrieved_at=obs['end'],
                        http_status=None, dom_path=str(src.resolve()), dom_sha256=sha(src), navigation=obs)
        if job['kind'] == 'lineup':
            validate_event(job['event'], job['season'], job['season_id'])
            metadata.update(event=job['event'], season=job['season'], event_id=job['event']['id'])
        try:
            validator(payload)
        except f.FetchError as exc:
            failures[job['route']]={'kind':exc.kind,'error':str(exc),'observation_end':obs['end'],
                                    'dom_path':str(src),'time':f.now()}
            continue
        stored=client.store(path, payload, f.BASE + job['route'], validator,
                            refresh=job.get('refresh', False), **metadata)
        remember(path,stored,obs['end'])
        failures.pop(job['route'],None)
        count += 1
    f.atomic_json(failure_path,failures)
    f.atomic_json(index_path,index)
    return log(run, {'action': 'sync', 'responses_imported': count,'validation_failures':len(failures)})


def plan(raw, data, run, refresh_current=False):
    """Discover from actual listing/rounds/pages; never infer a complete season from local counts."""
    client, jobs, discoveries = f.Client(raw=raw), [], {}
    previous = json.loads((run / 'jobs.json').read_text(encoding='utf-8')) if (run / 'jobs.json').exists() else []
    previous_by_route={j['route']:j for j in previous}
    observations=json.loads((run/'navigation.json').read_text(encoding='utf-8'))
    failure_path=run/'validation_failures.json'
    failures=json.loads(failure_path.read_text(encoding='utf-8')) if failure_path.exists() else {}
    generation=f.now() if refresh_current else None
    if refresh_current:
        old_listing=client.load(metadata_path(raw,'/unique-tournament/17/seasons'))
        current_id=next((s['id'] for s in old_listing['payload']['seasons']
                         if f.season_year(s)==f.CURRENT_SEASON),None) if old_listing else None
        for job in previous:
            if job['kind']=='metadata' and (job['route'].endswith('/seasons') or
                    current_id is not None and f'/season/{current_id}/' in job['route']):
                job.update(refresh=True,refresh_after=generation)
    def read(route, refresh=False):
        path = metadata_path(raw, route)
        env = client.load(path, lambda p: f.validate_metadata(p, route))
        old_job=previous_by_route.get(route,{})
        after=generation if refresh else old_job.get('refresh_after') if old_job.get('refresh') else None
        fresh=next((o for o in reversed(observations) if o.get('route')==route and o.get('outcome')=='parsed_json'),None)
        fulfilled=bool(after and fresh and route not in failures and
                       datetime.fromisoformat(fresh['end'])>=datetime.fromisoformat(after))
        pending=bool(after and not fulfilled)
        job=dict(kind='metadata',route=route,cache=str(path.resolve()),refresh=pending)
        if pending:
            job['refresh_after']=after
        jobs.append(job)
        # Old successful data remains frozen, but it cannot establish the refresh.
        return env['payload'] if env and not pending else None
    listing = read('/unique-tournament/17/seasons', refresh_current)
    context_path=run/'run_context.json'
    context=json.loads(context_path.read_text(encoding='utf-8'))
    if refresh_current:
        context['cutoff_at']=f.now()
        f.atomic_json(context_path,context)
    cutoff = datetime.fromisoformat(context.get('cutoff_at',context['started_at'])).timestamp()
    ids = {f.season_year(s): s['id'] for s in listing['seasons']} if listing else {}
    for season in range(2017, f.CURRENT_SEASON + 1):
        if season not in ids:
            discoveries[season] = {'verified': False, 'reason': 'season listing absent/missing season'}
            continue
        sid, all_events, complete = ids[season], {}, True
        refresh = refresh_current and season == f.CURRENT_SEASON
        rounds = read(f'/unique-tournament/17/season/{sid}/rounds', refresh)
        if not rounds:
            discoveries[season] = {'verified': False, 'reason': 'round listing absent'}
            continue
        for r in rounds['rounds']:
            page = 0
            while True:
                route = f'/unique-tournament/17/season/{sid}/events/round/{r["round"]}'
                if page:
                    route += f'?page={page}'
                batch = read(route, refresh)
                if batch is None:
                    complete = False
                    break
                for event in batch['events']:
                    if event['status']['type'] != 'finished' or event['startTimestamp'] > cutoff:
                        continue
                    validate_event(event, season, sid)
                    old = all_events.get(event['id'])
                    if old and (old['homeTeam']['id'],old['awayTeam']['id'],old['startTimestamp']) != (event['homeTeam']['id'],event['awayTeam']['id'],event['startTimestamp']):
                        raise ValueError('Conflicting repeated discovery event')
                    all_events[event['id']] = event
                if batch.get('hasNextPage') is not True:
                    break
                page += 1
                if page > 100:
                    raise ValueError('Pagination did not terminate')
        events = sorted(all_events.values(), key=lambda e: (e['startTimestamp'],e['id']))
        discoveries[season] = {'verified': complete, 'enumerated_completed': len(events), 'season_id': sid}
        if complete:
            path = raw / 'discovery' / f'{season}.json'
            client.store(path, {'events': events, 'season_id': sid}, f.BASE + f'/unique-tournament/17/season/{sid}/rounds',
                         refresh=season == f.CURRENT_SEASON, transport='derived_from_browser_round_envelopes')
        for event in events:
            jobs.append(dict(kind='lineup', route=f'/event/{event["id"]}/lineups', season=season, season_id=sid,
                             event=event, cache=str((raw/'matches'/str(season)/f'{event["id"]}.json').resolve())))
    # Retain previously discovered work while new metadata is incomplete.
    routes = {j['route'] for j in jobs}
    jobs.extend(j for j in previous if j['kind']=='lineup' and j['route'] not in routes)
    jobs.extend(j for j in previous if j['kind']=='metadata' and j.get('refresh') and j['route'] not in routes)
    jobs.sort(key=lambda j: (j['kind']=='lineup', '/events/round/' in j['route'], j.get('season',0), j.get('event',{}).get('startTimestamp',0),j['route']))
    f.atomic_json(run/'jobs.json', jobs)
    f.atomic_json(run/'discovery_status.json', discoveries)
    pending=[]
    for j in jobs:
        env=client.load(Path(j['cache']), f.validate_lineup if j['kind']=='lineup' else None)
        if not env or j.get('refresh'):
            pending.append(j)
    f.atomic_json(run/'queue.json',pending)
    return log(run, {'action':'plan','jobs':len(jobs),'pending':len(pending),
                     'pending_by_kind':dict(Counter(j['kind'] for j in pending)), 'discoveries':discoveries})


def build(raw,data,run):
    registry=Registry(data)
    registry.seed()
    result=f.build(f.Client(raw=raw),registry,data)
    f.atomic_json(run/'build_summary.json',result)
    return log(run,dict(action='build',**result))


def checkpoint(raw,data,run):
    sync(raw,run)
    state_path=run/'discovery_status.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    jobs=json.loads((run/'jobs.json').read_text(encoding='utf-8')) if (run/'jobs.json').exists() else []
    # Once independently discovered, historical metadata/jobs are frozen. sync
    # validates all existing cache envelopes above; avoid re-deriving and rewriting
    # the entire fixture universe for each small lineup batch. Explicit metadata
    # refresh jobs still force the discovery planner after their responses arrive.
    if (set(state)=={str(s) for s in range(2017,f.CURRENT_SEASON+1)}
            and all(d['verified'] for d in state.values()) and not any(j.get('refresh') for j in jobs)):
        pending=[j for j in jobs if not Path(j['cache']).exists()]
        f.atomic_json(run/'queue.json',pending)
        planned={'discoveries':state,'pending_by_kind':dict(Counter(j['kind'] for j in pending))}
    else:
        planned=plan(raw,data,run)
    status=planned['discoveries']
    failures=json.loads((run/'validation_failures.json').read_text(encoding='utf-8'))
    counts={}
    collection_complete=True
    for s,d in status.items():
        cached={p.stem for p in (raw/'matches'/str(s)).glob('*.json')}
        discovery_path=raw/'discovery'/f'{s}.json'
        expected={str(e['id']) for e in json.loads(discovery_path.read_text())['payload']['events']} if d['verified'] else set()
        missing=expected-cached
        collection_complete &= d['verified'] and cached==expected
        unavailable={eid for eid in missing if failures.get(f'/event/{eid}/lineups',{}).get('kind')=='unavailable'}
        counts[str(s)]=dict(expected=len(expected) if d['verified'] else None,
                           enumerated_so_far=d.get('enumerated_completed',0),cached_valid=len(cached),
                           missing=len(missing) if d['verified'] else None,
                           deferred=len(missing-unavailable) if d['verified'] else None,
                           unsupported_unavailable=len(unavailable),unavailable_ids=sorted(unavailable),
                           discovery_unverified=int(not d['verified']))
    nav=json.loads((run/'navigation.json').read_text(encoding='utf-8'))
    # This manifest describes source collection. The run completion audit also
    # verifies derived output, idempotence, compatible publication/preservation.
    collection_complete = bool(collection_complete and not failures and
                               not any(planned['pending_by_kind'].values()))
    result=dict(updated_at=f.now(),complete=collection_complete,
                collection_complete=collection_complete,seasons=counts,
                navigations=len(nav),navigation_outcomes=dict(Counter(o['outcome'] for o in nav)),
                imported_pilot=30,new_lineup_navigations=sum('/lineups' in o['route'] for o in nav),
                pending=planned['pending_by_kind'],validation_failures=len(failures),
                validation_failure_kinds=dict(Counter(v['kind'] for v in failures.values())))
    f.atomic_json(raw/'manifest.json',result)
    f.atomic_json(run/'checkpoint.json',result)
    snapshots=run/'checkpoints'
    snapshots.mkdir(exist_ok=True)
    f.atomic_json(snapshots/(str(len(nav))+'.json'),result)
    return log(run,dict(action='checkpoint',**result))


def reconcile(raw,data,run):
    local={}
    for row in read_csv(data/'understat_match_positions.csv'):
        key=(int(row['Season']),row['Home'],row['Away'])
        local[key]=dict(date=row['Date'],event_id=row['MatchID_Understat'])
    results={}
    for season in range(2017,f.CURRENT_SEASON+1):
        path=raw/'discovery'/f'{season}.json'
        if not path.exists():
            results[season]={'verified':False,'reason':'discovery incomplete'}
            continue
        events=f.Client(raw=raw).load(path)['payload']['events']
        source={}
        for e in events:
            home=f.TEAM_NAMES.get(e['homeTeam']['name'],e['homeTeam']['name'])
            away=f.TEAM_NAMES.get(e['awayTeam']['name'],e['awayTeam']['name'])
            key=(season,home,away)
            if key in source:
                raise ValueError('Duplicate season/home/away; review fixture identities')
            source[key]={'date':datetime.fromtimestamp(e['startTimestamp'],UTC).date().isoformat(),
                         'event_id':e['id']}
        expected={k:v for k,v in local.items() if k[0]==season}
        only_source=[dict(season=k[0],home=k[1],away=k[2],**source[k]) for k in source.keys()-expected.keys()]
        only_local=[dict(season=k[0],home=k[1],away=k[2],**expected[k]) for k in expected.keys()-source.keys()]
        dates=[dict(season=k[0],home=k[1],away=k[2],source=source[k],local=expected[k])
               for k in source.keys()&expected.keys() if source[k]['date']!=expected[k]['date']]
        results[season]=dict(verified=not(only_source or only_local or dates),source_count=len(source),
                             local_count=len(expected),only_source=only_source,only_local=only_local,date_differences=dates)
    f.atomic_json(run/'fixture_reconciliation.json',results)
    return log(run,dict(action='reconcile',seasons=results))


def audit(raw,data,run):
    """Validate every raw/DOM/derived row, not a sample or count-only manifest."""
    def raw_snapshot():
        return {str(p):(p.stat().st_size,p.stat().st_mtime_ns)
                for p in (raw/'matches').glob('*/*.json')}
    before=raw_snapshot()
    rows=read_csv(data/'sofascore_player_match_stats.csv')
    by_key={(r['Event_ID_Sofascore'],r['Player_ID_Sofascore']):r for r in rows}
    if len(by_key)!=len(rows):
        raise ValueError('Duplicate derived source identity')
    canonical={(r['Event_ID_Sofascore'],r['Canonical_Player_ID']) for r in rows}
    if len(canonical)!=len(rows):
        raise ValueError('Duplicate derived canonical identity')
    client=f.Client(raw=raw)
    seen=set()
    seasons={}
    raw_hashes={}
    for path in sorted((raw/'matches').glob('*/*.json')):
        env=client.load(path,f.validate_lineup)
        event=env['event']
        season=env['season']
        validate_event(event,season,event['season']['id'])
        if str(event['id'])!=path.stem or str(season)!=path.parent.name:
            raise ValueError('Raw path mismatch')
        dom=Path(env['dom_path'])
        if sha(dom)!=env['dom_sha256'] or json.loads(dom.read_text(encoding='utf-8'))!=env['payload']:
            raise ValueError('Raw/DOM provenance mismatch')
        raw_hashes[str(path.relative_to(raw))]=sha(path)
        report=seasons.setdefault(season,dict(cached=0,rows=0,played=0,unresolved=0,starting_keepers=0))
        report['cached']+=1
        for side in ('home','away'):
            keepers=[p for p in env['payload'][side]['players'] if not p['substitute'] and p['player'].get('position')=='G']
            if len(keepers)!=1:
                raise ValueError('Starting keeper absent/ambiguous')
            report['starting_keepers']+=1
            for p in env['payload'][side]['players']:
                key=(str(event['id']),str(p['player']['id']))
                seen.add(key)
                if key not in by_key:
                    raise ValueError('Dropped source row')
                row=by_key[key]
                if not row['Canonical_Player_ID'] or not row['Canonical_Fixture_ID']:
                    raise ValueError('Missing project identity')
                if row['Player_Sofascore']!=p['player']['name'] or row['Is_Starter']!=str(not p['substitute']):
                    raise ValueError('Source metadata changed')
                if json.loads(row['Present_Fields'])!=sorted(p.get('statistics',{})):
                    raise ValueError('Presence metadata changed')
                for field in f.FIELDS:
                    value=p.get('statistics',{}).get(field)
                    out=row['Sofascore_'+field]
                    if (value is None and out!='') or (value is not None and (out=='' or float(out)!=value)):
                        raise ValueError(f'Source statistic changed: {key}/{field}')
                report['rows']+=1
                report['played']+=int(row['Appearance_Status']=='played')
                report['unresolved']+=int(row['Identity_Status']=='unresolved_cross_provider')
    if seen!=set(by_key):
        raise ValueError('Derived rows without raw source')
    registry=Registry(data)
    for base in (f.DATA, f.ROOT/'data/validation/sofascore_browser_pilot_20261026/current'):
        accepted=Registry(base)
        for key,cid in accepted.alias.items():
            if registry.alias.get(key)!=cid:
                raise ValueError(f'Changed accepted player identity: {key}')
        for key,row in accepted.fixtures.items():
            if key in registry.fixtures and registry.fixtures[key]['Canonical_Fixture_ID']!=row['Canonical_Fixture_ID']:
                raise ValueError(f'Changed accepted fixture identity: {key}')
    unresolved=[r for r in rows if r['Identity_Status']=='unresolved_cross_provider']
    if rows:
        from fpl_model.fetch.player_identity import write_csv
        write_csv(run/'unresolved_identities.csv',unresolved,list(rows[0]))
    reconciliation=reconcile(raw,data,run)['seasons']
    discovery=json.loads((run/'discovery_status.json').read_text(encoding='utf-8'))
    complete=True
    for season in range(2017,f.CURRENT_SEASON+1):
        report=seasons.setdefault(season,dict(cached=0,rows=0,played=0,unresolved=0,starting_keepers=0))
        state=discovery[str(season)]
        expected_env=client.load(raw/'discovery'/f'{season}.json')
        ids={str(e['id']) for e in expected_env['payload']['events']} if expected_env else set()
        cached={p.stem for p in (raw/'matches'/str(season)).glob('*.json')}
        report.update(expected=len(ids) if state['verified'] else None,
                      missing=sorted(ids-cached),unexpected_cache=sorted(cached-ids),
                      discovery_verified=state['verified'],local_reconciliation=reconciliation[season]['verified'])
        complete &= state['verified'] and not(ids-cached or cached-ids) and reconciliation[season]['verified']
    result=dict(time=f.now(),collection_complete=bool(complete),rows=len(rows),seasons=seasons,
                raw_dom_stat_equality=True,canonical_unique=True,accepted_identities_preserved=True,
                unresolved_players=len({r['Player_ID_Sofascore'] for r in unresolved}))
    if raw_snapshot()!=before:
        raise ValueError('Raw cache changed during audit; hold imports and rerun build/audit')
    f.atomic_json(run/'raw_hashes_audit.json',raw_hashes)
    f.atomic_json(run/'data_audit.json',result)
    return log(run,dict(action='audit',**result))


def main():
    started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','import-pilot','sync','plan','build','checkpoint','reconcile','audit','normalize-fixture-aliases'])
    p.add_argument('--raw',type=Path,default=f.RAW)
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--data',type=Path,default=DEFAULT_RUN/'current')
    p.add_argument('--pilot',type=Path,default=f.ROOT/'data/validation/sofascore_browser_pilot_20261026')
    p.add_argument('--refresh-current',action='store_true')
    a=p.parse_args()
    if a.action=='prepare':
        result=prepare(a.run,a.data)
    elif a.action=='import-pilot':
        result=import_pilot(a.raw,a.data,a.run,a.pilot)
    elif a.action=='sync':
        result=sync(a.raw,a.run)
    elif a.action=='plan':
        result=plan(a.raw,a.data,a.run,a.refresh_current)
    elif a.action=='checkpoint':
        result=checkpoint(a.raw,a.data,a.run)
    elif a.action=='reconcile':
        result=reconcile(a.raw,a.data,a.run)
    elif a.action=='audit':
        result=audit(a.raw,a.data,a.run)
    elif a.action=='normalize-fixture-aliases':
        result=normalize_fixture_aliases(a.raw,a.data,a.run)
    else:
        result=build(a.raw,a.data,a.run)
    log(a.run,{'command':sys.argv,'elapsed_seconds':time.monotonic()-started})
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
