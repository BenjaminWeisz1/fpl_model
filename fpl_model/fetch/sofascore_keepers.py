"""Goalkeepers' career shot-stopping from Sofascore player pages (offline; never makes HTTP requests).

Sofascore's player pages give, per competition and season, expected goals on target conceded (xGOTC), goals
conceded, goals prevented and penalties faced and saved, wherever Sofascore has xG (top leagues and European cups,
from about 2021/22). Two pages per keeper and season are captured in Chrome (SOFASCORE.md) and frozen by
`sofascore_weekly.py import` under data/raw/sofascore/players/<player id>/:

  seasons.json          /player/{id}/statistics/seasons (every competition and season played)
  <ut>_<season>.json    /player/{id}/unique-tournament/{ut}/season/{season}/statistics/overall

A changed re-capture (the current season, a new competition) replaces the file and keeps the old one as
`.revision.<ns>`, so past values stay traceable.

Tracked keepers: each team's starting keeper, the goalkeeper with the most FPL Review xMins over the next
NEXT_GAMEWEEKS gameweeks in minutes.csv. `plan` lists the pages to capture for them: the seasons page, any
season from FIRST_SEASON on that is not cached, and current-season pages, which change every week.

`build` writes data/current/sourced/sofascore_keeper_seasons.csv, one row per keeper, competition and season,
with non-penalty columns. The season pages count penalties in both goals and xGOT conceded but give only
penalties faced and saved, so penalties are taken out with two rates measured each run from the cached match
shot maps: the share of unsaved penalties that miss the target, and the mean xGOT of a penalty on target.

Run locally:
    python -m fpl_model.fetch.sofascore_keepers plan | build
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic
from fpl_model.fetch import fetch_sofascore as f

FIRST_SEASON = 2021  # Sofascore has keeper xGOT from about 2021/22
NEXT_GAMEWEEKS = 5
# Competitions with no xG in any captured page (§258): domestic cups, qualifiers, youth, friendlies, super cups.
SKIP_TOURNAMENTS = {11, 19, 21, 24, 25, 27, 295, 328, 334, 339, 346, 454, 1295}
SKIP_NAME = re.compile(r'friendl|\bU\d{2}\b|youth|Premier League 2|All.?Star|Troph|Super ?Cup|Supercup|Shield|Qual',
                       re.IGNORECASE)
SEASONS_ROUTE = re.compile(r'^/player/(\d+)/statistics/seasons$')
STATS_ROUTE = re.compile(r'^/player/(\d+)/unique-tournament/(\d+)/season/(\d+)/statistics/overall$')
OUT_FILE = 'sofascore_keeper_seasons.csv'
STAT_FIELDS = ['appearances', 'minutesPlayed', 'goalsConceded', 'expectedGoalsOnTargetConceded', 'goalsPrevented',
               'penaltyFaced', 'penaltySave', 'saves']


def validate_seasons(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('uniqueTournamentSeasons'), list):
        raise f.FetchError('schema', 'Player seasons must hold a uniqueTournamentSeasons list')
    for u in payload['uniqueTournamentSeasons']:
        if not isinstance(u.get('uniqueTournament', {}).get('id'), int) or not isinstance(u.get('seasons'), list):
            raise f.FetchError('schema', 'Invalid player seasons entry')


def validate_stats(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('statistics'), dict):
        raise f.FetchError('schema', 'Player season statistics must hold a statistics object')


def season_year(text):
    """'21/22' -> 2021, '2024' -> 2024, anything else -> None."""
    if m := re.fullmatch(r'(\d{2})/(\d{2})', str(text)):
        return 2000 + int(m.group(1))
    return int(text) if re.fullmatch(r'\d{4}', str(text)) else None


def route_path(raw, route):
    """(cache path, validator) for a player route, or None if the route is not a player page."""
    if m := SEASONS_ROUTE.match(route):
        return Path(raw) / 'players' / m.group(1) / 'seasons.json', validate_seasons
    if m := STATS_ROUTE.match(route):
        return Path(raw) / 'players' / m.group(1) / f'{m.group(2)}_{m.group(3)}.json', validate_stats
    return None


def store(client, raw, route, payload, **meta):
    """Freeze one player page; a changed page replaces the file and keeps the old one as a revision."""
    path, validator = route_path(raw, route)
    old = client.load(path, validator) if path.exists() else None
    client.store(path, payload, f.BASE + route, validator, refresh=True, **meta)
    if old is None:
        return 'player_new'
    return 'player_unchanged' if f.digest(old['payload']) == f.digest(payload) else 'player_revised'


def load(path):
    env = json.loads(path.read_text(encoding='utf-8'))
    if f.digest(env['payload']) != env['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    return env['payload']


def starting_keepers(data=f.DATA):
    """Each team's starting keeper: (Team_Short, Local_Player, Sofascore id), most xMins over the next gameweeks."""
    minutes = pd.read_csv(paths.current_file('minutes.csv', root=data), encoding='utf-8-sig')
    cols = sorted((c for c in minutes.columns if c.endswith('_xMins')), key=lambda c: int(c.split('_')[0]))
    gk = minutes[minutes['Pos'] == 'GKP'].assign(xMins=lambda d: d[cols[:NEXT_GAMEWEEKS]].sum(axis=1))
    top = gk.sort_values('xMins', ascending=False).groupby('Team').head(1)
    players = pd.read_csv(paths.current_file('players_current.csv', root=data), encoding='utf-8-sig')
    top = top.merge(players[['Raw_Web_Name', 'Team_Short', 'Player']], left_on=['Name', 'Team'],
                    right_on=['Raw_Web_Name', 'Team_Short'], how='left')
    return link_ids(top.rename(columns={'Team': 'Team_Code'}), data, team='Team_Code')


def link_ids(frame, data=f.DATA, team='Team', name='Player'):
    """[(team, full name, Sofascore id or None)] for a frame of players_current full names."""
    sofa = pd.read_csv(paths.current_file('sofascore_player_match_stats.csv', root=data), encoding='utf-8-sig',
                       usecols=['Player_ID_Sofascore', 'Local_Player', 'Player_Sofascore', 'Kickoff'])
    sofa = sofa.sort_values('Kickoff')
    # Fallback: Sofascore's own name, when unique, for players whose identity was never resolved (e.g. Gboho).
    by_source = sofa.drop_duplicates('Player_ID_Sofascore', keep='last')
    by_source = by_source[~by_source['Player_Sofascore'].duplicated(keep=False)]
    ids = dict(zip(by_source['Player_Sofascore'], by_source['Player_ID_Sofascore'], strict=True))
    linked = sofa.dropna(subset=['Local_Player']).drop_duplicates('Local_Player', keep='last')
    ids.update(zip(linked['Local_Player'], linked['Player_ID_Sofascore'], strict=True))
    return [(t, p, None if pd.isna(ids.get(p)) else int(ids[p])) for t, p in zip(frame[team], frame[name],
                                                                                  strict=True)]


def skipped(tournament):
    """Competitions Sofascore keeps no xG for (domestic cups, qualifiers, youth, friendlies): never captured."""
    return tournament.get('id') in SKIP_TOURNAMENTS or bool(SKIP_NAME.search(tournament.get('name', '')))


def plan_pages(raw, people, xg_field):
    """Pages to capture for [(team, name, id)]: the seasons page for anyone not cached yet, competition-seasons
    from FIRST_SEASON not cached, and current-season pages that carry *xg_field* (they change weekly)."""
    raw = Path(raw)
    capture, unlinked = [], []
    for team, name, pid in people:
        if pid is None:
            unlinked.append(f'{team}: {name}')
            continue
        seasons_path = raw / 'players' / str(pid) / 'seasons.json'
        if not seasons_path.exists():
            capture.append({'kind': 'player_seasons', 'player': name,
                            'url': f'{f.BASE}/player/{pid}/statistics/seasons'})
            continue  # stats pages are listed once the seasons page is imported
        for u in load(seasons_path)['uniqueTournamentSeasons']:
            if skipped(u['uniqueTournament']):
                continue
            for s in u['seasons']:
                year = season_year(s.get('year'))
                if year is None or year < FIRST_SEASON:
                    continue
                route = f"/player/{pid}/unique-tournament/{u['uniqueTournament']['id']}/season/{s['id']}/statistics/overall"
                path = route_path(raw, route)[0]
                refresh = path.exists() and year >= f.CURRENT_SEASON and \
                    load(path)['statistics'].get(xg_field) is not None
                if not path.exists() or refresh:
                    capture.append({'kind': 'player_stats', 'player': name, 'url': f.BASE + route})
    return {'capture': capture, 'unlinked': unlinked}


def plan(raw=f.RAW, data=f.DATA):
    """Weekly pages for the starting keepers. A new competition shows up only after his seasons page is
    recaptured: delete or recapture `seasons.json` at the start of a season."""
    return plan_pages(raw, starting_keepers(data), 'expectedGoalsOnTargetConceded')


def season_rows(raw, data, fields, keep=None):
    """One row per cached player, competition and season, with *fields* from the statistics page."""
    raw = Path(raw)
    names = pd.read_csv(paths.current_file('sofascore_player_match_stats.csv', root=data), encoding='utf-8-sig',
                        usecols=['Player_ID_Sofascore', 'Local_Player', 'Player_Sofascore', 'Kickoff'])
    names = names.sort_values('Kickoff').drop_duplicates('Player_ID_Sofascore', keep='last').set_index(
        'Player_ID_Sofascore')
    rows = []
    for folder in sorted((raw / 'players').glob('*')):
        pid = int(folder.name)
        seasons_path = folder / 'seasons.json'
        if (keep is not None and pid not in keep) or not seasons_path.exists():
            continue
        labels = {(u['uniqueTournament']['id'], s['id']): (u['uniqueTournament'].get('name', ''), s.get('year'))
                  for u in load(seasons_path)['uniqueTournamentSeasons'] for s in u['seasons']}
        for path in sorted(folder.glob('*_*.json')):
            ut, sid = (int(x) for x in path.stem.split('_'))
            stats = load(path)['statistics']
            tournament, year = labels.get((ut, sid), ('', None))
            rows.append({'Player_ID_Sofascore': pid,
                         'Player_Sofascore': names['Player_Sofascore'].get(pid, ''),
                         'Local_Player': names['Local_Player'].get(pid, ''),
                         'Tournament_ID': ut, 'Tournament': tournament, 'Season_ID': sid, 'Season_Label': year,
                         'Season_Start': season_year(year), **{k: stats.get(k) for k in fields}})
    return pd.DataFrame(rows)


def goalkeeper_ids(data):
    pos = pd.read_csv(paths.current_file('sofascore_player_match_stats.csv', root=data), encoding='utf-8-sig',
                      usecols=['Player_ID_Sofascore', 'Position_Sofascore'])
    return set(pos.loc[pos['Position_Sofascore'] == 'G', 'Player_ID_Sofascore'].astype(int))


def penalty_rates(raw):
    """(share of unsaved penalties that miss the target, mean xGOT of a penalty on target) from cached shot maps."""
    shots = [s for p in (Path(raw) / 'shotmap').glob('*/*.json') for s in load(p)['shotmap']
             if s.get('situation') == 'penalty']
    on_target = [s['xgot'] for s in shots if s.get('shotType') in ('goal', 'save') and s.get('xgot') is not None]
    unsaved = [s for s in shots if s.get('shotType') != 'save']
    if not on_target or not unsaved:
        raise ValueError('No cached penalty shots to measure penalty rates from')
    missed = sum(s.get('shotType') in ('miss', 'post') for s in unsaved)
    return missed / len(unsaved), sum(on_target) / len(on_target)


def build(raw=f.RAW, data=f.DATA):
    raw = Path(raw)
    p_miss, xgot_pen = penalty_rates(raw)
    out = season_rows(raw, data, STAT_FIELDS, keep=goalkeeper_ids(data))  # outfield players' pages live alongside
    if out.empty:
        return {'rows': 0}
    faced = out['penaltyFaced'].fillna(0)
    unsaved = faced - out['penaltySave'].fillna(0)
    pens_scored = unsaved * (1 - p_miss)
    out['Penalty_Goals_Est'] = pens_scored
    out['NPGC'] = out['goalsConceded'] - pens_scored
    out['NPxGOTC'] = out['expectedGoalsOnTargetConceded'] - (faced - unsaved * p_miss) * xgot_pen
    out['Penalty_Miss_Rate'] = p_miss
    out['Penalty_xGOT_On_Target'] = xgot_pen
    to_csv_atomic(out.sort_values(['Player_ID_Sofascore', 'Season_Start', 'Tournament_ID']),
                  paths.current_file(OUT_FILE, root=data), index=False, encoding='utf-8-sig')
    return {'rows': len(out), 'keepers': out['Player_ID_Sofascore'].nunique(),
            'with_xgot': int(out['expectedGoalsOnTargetConceded'].notna().sum()),
            'penalty_miss_rate': round(p_miss, 3), 'penalty_xgot_on_target': round(xgot_pen, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['plan', 'build'])
    a = p.parse_args(argv)
    result = plan() if a.action == 'plan' else build()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
