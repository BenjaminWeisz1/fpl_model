"""Outfield players' career finishing from Sofascore player pages (offline; never makes HTTP requests).

Same pages as the keepers' (`sofascore_keepers.py`): `/player/{id}/statistics/seasons` and one statistics page per
competition and season, frozen under data/raw/sofascore/players/<id>/ by `sofascore_weekly.py import`. Tracked
players: the top TOP_N midfielders and forwards by predicted points (predicted_points.csv Total), plus the top
TOP_DEFENDERS defenders by non-penalty goal rating (player_ratings.csv NPG) among those with more than MIN_NEXT_XMINS
FPL Review xMins over the next NEXT_GAMEWEEKS gameweeks (user decision, 2026-11-19).

`build` writes data/current/sourced/sofascore_player_finishing_seasons.csv, one row per player, competition and
season, with NPG = goals - penaltyGoals and NPxG = expectedGoals - penaltiesTaken x the mean xG of a penalty in the
cached match shot maps (the season pages' xG includes penalties; checked against 2026/27 shot maps, §259).

Run locally:
    python -m fpl_model.fetch.sofascore_finishing plan | build
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic
from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch import sofascore_keepers as pages

TOP_N = 100
POSITIONS = ('M', 'F')  # user decision: midfielders and forwards only
TOP_DEFENDERS = 10  # user decision: plus the best defenders by NPG rating who are expected to play
MIN_NEXT_XMINS = 100
NEXT_GAMEWEEKS = pages.NEXT_GAMEWEEKS
OUT_FILE = 'sofascore_player_finishing_seasons.csv'
STAT_FIELDS = ['appearances', 'minutesPlayed', 'goals', 'penaltyGoals', 'penaltiesTaken', 'expectedGoals',
               'totalShots']


def next_xmins(data=f.DATA):
    """(Web_Name, Team_Short) -> FPL Review xMins over the next NEXT_GAMEWEEKS gameweeks (minutes.csv names are raw)."""
    minutes = pd.read_csv(paths.current_file('minutes.csv', root=data), encoding='utf-8-sig')
    cols = sorted((c for c in minutes.columns if c.endswith('_xMins')), key=lambda c: int(c.split('_')[0]))
    total = minutes[cols[:NEXT_GAMEWEEKS]].sum(axis=1)
    players = pd.read_csv(paths.current_file('players_current.csv', root=data), encoding='utf-8-sig')
    web = {(r.Raw_Web_Name, r.Team_Short): r.Web_Name for r in players.itertuples()}
    return {(web.get((n, t), n), t): x for n, t, x in zip(minutes['Name'], minutes['Team'], total, strict=True)}


def top_defenders(data=f.DATA, n=TOP_DEFENDERS):
    """Web_Name/Team rows of the top *n* defenders by NPG rating with over MIN_NEXT_XMINS predicted minutes."""
    ratings = pd.read_csv(paths.current_file('player_ratings.csv', root=data), encoding='utf-8-sig')
    xmins = next_xmins(data)
    d = ratings[ratings['Pos'] == 'D']
    d = d[[xmins.get((p, t), 0.0) > MIN_NEXT_XMINS for p, t in zip(d['Player'], d['Team'], strict=True)]]
    return d.sort_values('NPG', ascending=False).head(n)


def top_outfield(data=f.DATA, n=TOP_N, defenders=TOP_DEFENDERS, positions=POSITIONS):
    """[(team, full name, Sofascore id)]: the top *n* midfielders and forwards by predicted points, then the
    top *defenders* defenders (`top_defenders`)."""
    pp = pd.read_csv(paths.current_file('predicted_points.csv', root=data), encoding='utf-8-sig')
    top = pp[pp['Pos'].isin(positions)].sort_values('Total', ascending=False).head(n)
    if defenders:
        top = pd.concat([top[['Player', 'Team']], top_defenders(data, defenders)[['Player', 'Team']]])
    players = pd.read_csv(paths.current_file('players_current.csv', root=data), encoding='utf-8-sig')
    top = top.merge(players[['Web_Name', 'Team_Short', 'Player']].rename(columns={'Player': 'Full_Name'}),
                    left_on=['Player', 'Team'], right_on=['Web_Name', 'Team_Short'], how='left')
    return pages.link_ids(top, data, name='Full_Name')


def plan(raw=f.RAW, data=f.DATA):
    return pages.plan_pages(raw, top_outfield(data), 'expectedGoals')


def penalty_xg(raw):
    """Mean Sofascore xG of a penalty in the cached match shot maps."""
    xg = [s['xg'] for p in (Path(raw) / 'shotmap').glob('*/*.json') for s in pages.load(p)['shotmap']
          if s.get('situation') == 'penalty' and s.get('xg') is not None]
    if not xg:
        raise ValueError('No cached penalty shots to measure penalty xG from')
    return sum(xg) / len(xg)


def build(raw=f.RAW, data=f.DATA):
    pen_xg = penalty_xg(raw)
    out = pages.season_rows(raw, data, STAT_FIELDS, keep=None)
    out = out[~out['Player_ID_Sofascore'].isin(pages.goalkeeper_ids(data))] if not out.empty else out
    if out.empty:
        return {'rows': 0}
    out = out.assign(NPG=out['goals'] - out['penaltyGoals'].fillna(0),
                     NPxG=out['expectedGoals'] - out['penaltiesTaken'].fillna(0) * pen_xg,
                     Penalty_xG=pen_xg)
    to_csv_atomic(out.sort_values(['Player_ID_Sofascore', 'Season_Start', 'Tournament_ID']),
                  paths.current_file(OUT_FILE, root=data), index=False, encoding='utf-8-sig')
    return {'rows': len(out), 'players': int(out['Player_ID_Sofascore'].nunique()),
            'with_xg': int(out['expectedGoals'].notna().sum()), 'penalty_xg': round(pen_xg, 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['plan', 'build'])
    a = p.parse_args(argv)
    print(json.dumps(plan() if a.action == 'plan' else build(), indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
