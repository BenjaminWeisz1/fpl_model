"""Set-piece events from Sofascore match commentary and shot maps (offline; never makes HTTP requests).

Sofascore's commentary (`/event/{id}/comments`, Opta's text feed) names the player who set up each shot and
says whether it followed a corner or a set piece; its shot map (`/event/{id}/shotmap`) gives every shot's
situation and xG. Neither names who took a corner or free kick that produced no shot, nor the side a corner
came from: commentary lists only the player who conceded the corner or won the free kick. So the taker
is known for set pieces that ended in a shot (the assister for corners and indirect free kicks, the
shooter for direct free kicks and penalties); every corner and free kick won is still recorded, with its
team and minute.

Both feeds are captured in Chrome with the lineups (SOFASCORE.md) and frozen by `sofascore_weekly.py
import` under data/raw/sofascore/{comments,shotmap}/<season>/<event>.json. `build` writes:

  sofascore_set_piece_events.csv  one row per corner won, free kick won and set-piece shot
                                  (corner, set-piece, free-kick, throw-in-set-piece, penalty)
  sofascore_player_on_pitch.csv   one row per player appearance: minute on, minute off on the commentary
                                  clock (starters from the lineup; substitutions, red cards and injuries
                                  with no subs left from commentary), and Sofascore's minutesPlayed

Shots pair with commentary lines by shooter and minute. Assisters are matched by name to the
shooter's team's lineup. Unpaired shots and unmatched assisters are counted in the build summary.

Run locally:
    python -m fpl_model.fetch.sofascore_set_pieces
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic
from fpl_model.fetch import fetch_sofascore as f

FEEDS = ('comments', 'shotmap')
SET_PIECE_SITUATIONS = ('corner', 'set-piece', 'free-kick', 'throw-in-set-piece', 'penalty')
SHOT_COMMENT_TYPES = {'shotOffTarget', 'shotSaved', 'shotBlocked', 'post', 'scoreChange',
                      'penaltyScored', 'penaltyMissed', 'penaltySaved'}
LEAVES_FIELD = {'redCard', 'yellowRedCard', 'playerRetired'}  # playerRetired: off injured, no subs left
MAX_PAIR_GAP = 3  # minutes between a shot-map shot and its commentary line
ASSIST = re.compile(r'Assisted by (.+?)(?: with | following |\.\s*$)')
CORNER_TEAM = re.compile(r'^Corner,\s*(.+?)\.\s*Conceded by')
FREE_KICK_ZONE = re.compile(r'wins a free kick (?:in|on) the (.+?)\.')
EVENTS_FILE = 'sofascore_set_piece_events.csv'
ON_PITCH_FILE = 'sofascore_player_on_pitch.csv'


def validate_comments(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('comments'), list):
        raise f.FetchError('schema', 'Commentary must be an object with a comments list')
    for c in payload['comments']:
        if not isinstance(c, dict) or not isinstance(c.get('type'), str):
            raise f.FetchError('schema', 'Invalid commentary entry')


def validate_shotmap(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('shotmap'), list):
        raise f.FetchError('schema', 'Shot map must be an object with a shotmap list')
    for s in payload['shotmap']:
        if not isinstance(s.get('player', {}).get('id'), int) or not isinstance(s.get('situation'), str):
            raise f.FetchError('schema', 'Invalid shot-map entry')


VALIDATORS = {'comments': validate_comments, 'shotmap': validate_shotmap}


def feed_path(raw, kind, season, event_id):
    return Path(raw) / kind / str(season) / f'{event_id}.json'


def load(path):
    env = json.loads(path.read_text(encoding='utf-8'))
    if f.digest(env['payload']) != env['sha256']:
        raise ValueError(f'Hash mismatch: {path}')
    return env


def norm(name):
    text = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', '', text.casefold()).strip()


def lineup_players(lineup):
    """Sofascore player id -> (side, name, short name, starter) from a lineups payload."""
    out = {}
    for side in ('home', 'away'):
        for row in lineup[side]['players']:
            p = row['player']
            out[p['id']] = (side, p['name'], p.get('shortName', ''), not row['substitute'],
                            (row.get('statistics') or {}).get('minutesPlayed'))
    return out


def match_name(name, candidates):
    """Sofascore id for a commentary name among one team's lineup, or None."""
    key = norm(name)
    for field in (1, 2):
        hits = [pid for pid, v in candidates.items() if norm(v[field]) == key]
        if len(hits) == 1:
            return hits[0]
    last = key.split(' ')[-1] if key else ''
    hits = [pid for pid, v in candidates.items() if norm(v[1]).split(' ')[-1:] == [last]]
    if len(hits) == 1:
        return hits[0]
    tokens = {t for t in key.split(' ') if len(t) > 2}  # e.g. "Abdul Fatawu" v "Fatawu Issahaku"
    hits = [pid for pid, v in candidates.items() if tokens & set(norm(v[1]).split(' '))]
    return hits[0] if len(hits) == 1 else None


def pair_shots(shots, comments):
    """Shot-map index -> commentary entry, one to one, same shooter, nearest minute within MAX_PAIR_GAP."""
    pool = [c for c in comments if c['type'] in SHOT_COMMENT_TYPES and c.get('player') and 'time' in c]
    used, pairs = set(), {}
    options = []
    for i, s in enumerate(shots):
        t = s['time'] + (s.get('addedTime') or 0)
        for j, c in enumerate(pool):
            if c['player'].get('id') == s['player']['id']:
                gap = abs(c['time'] - t)
                if gap <= MAX_PAIR_GAP:
                    options.append((gap, i, j))
    for _gap, i, j in sorted(options):
        if i not in pairs and j not in used:
            pairs[i] = pool[j]
            used.add(j)
    return pairs


def on_pitch(players, comments, end):
    """Sofascore id -> (on, off) minutes for everyone who played."""
    spans = {pid: [0.0, end] for pid, v in players.items() if v[3]}
    for c in sorted((c for c in comments if 'time' in c), key=lambda c: c['time']):
        if c['type'] == 'substitution':
            pin, pout = (c.get('playerIn') or {}).get('id'), (c.get('playerOut') or {}).get('id')
            if pout in spans:
                spans[pout][1] = min(spans[pout][1], c['time'])
            if pin is not None:
                spans[pin] = [c['time'], end]
        elif c['type'] in LEAVES_FIELD and (c.get('player') or {}).get('id') in spans:
            spans[c['player']['id']][1] = min(spans[c['player']['id']][1], c['time'])
    return spans


def local_names(data):
    """(event id, Sofascore id) -> Local_Player (players_current full name) from the built lineup table."""
    s = pd.read_csv(paths.current_file('sofascore_player_match_stats.csv', root=data), encoding='utf-8-sig',
                    usecols=['Event_ID_Sofascore', 'Player_ID_Sofascore', 'Local_Player'], dtype=str)
    s = s.dropna(subset=['Local_Player'])
    return {(e, p): n for e, p, n in s.itertuples(index=False)}


def match_rows(season, lineup_env, comments, shots, names):
    event = lineup_env['event']
    eid = str(lineup_env['event_id'])
    home, away = (f.TEAM_NAMES.get(event[s]['name'], event[s]['name']) for s in ('homeTeam', 'awayTeam'))
    team_of = {'home': home, 'away': away}
    source_team = {event['homeTeam']['name']: 'home', event['awayTeam']['name']: 'away'}
    players = lineup_players(lineup_env['payload'])
    kickoff = pd.Timestamp(event['startTimestamp'], unit='s', tz='UTC').isoformat()

    def who(pid):
        return '' if pid is None else names.get((eid, str(pid)), '')

    def base(side, minute, added=None):
        return dict(Season=season, Event_ID_Sofascore=eid, Kickoff=kickoff, Team=team_of[side],
                    Opponent=team_of['away' if side == 'home' else 'home'],
                    Venue='Home' if side == 'home' else 'Away', Minute=minute, Added_Time=added)

    rows, stats = [], {'shots': 0, 'unpaired_shots': 0, 'assists': 0, 'unmatched_assisters': []}
    for c in comments:
        text = c.get('text', '')
        if c['type'] == 'cornerKick':
            m = CORNER_TEAM.match(text)
            side = source_team.get(m.group(1)) if m else None
            if side is None:  # the conceder's side is isHome; the corner goes to the other team
                side = 'away' if c.get('isHome') else 'home'
            rows.append({**base(side, c['time']), 'Event_Type': 'corner_won', 'Situation': 'corner',
                         'Named_Player_ID_Sofascore': (c.get('player') or {}).get('id'),
                         'Named_Player_Role': 'conceded_by', 'Comment': text})
        elif c['type'] == 'freeKickWon':
            m = FREE_KICK_ZONE.search(text)
            side = 'home' if c.get('isHome') else 'away'
            pid = (c.get('player') or {}).get('id')
            rows.append({**base(side, c['time']), 'Event_Type': 'free_kick_won',
                         'Situation': m.group(1) if m else '', 'Named_Player_ID_Sofascore': pid,
                         'Named_Player_Local': who(pid), 'Named_Player_Role': 'won_by', 'Comment': text})
    pairs = pair_shots(shots, comments)
    for i, s in enumerate(shots):
        if s['situation'] not in SET_PIECE_SITUATIONS:
            continue
        stats['shots'] += 1
        side = 'home' if s['isHome'] else 'away'
        comment = pairs.get(i)
        text = comment.get('text', '') if comment else ''
        stats['unpaired_shots'] += comment is None
        taker = taker_name = None
        role = 'unknown'
        if s['situation'] in ('free-kick', 'penalty'):
            taker, taker_name, role = s['player']['id'], s['player']['name'], 'shooter'
        elif (m := ASSIST.search(text)):
            taker_name = m.group(1).strip()
            team = {pid: v for pid, v in players.items() if v[0] == side}
            taker, role = match_name(taker_name, team), 'assister'
            stats['assists'] += 1
            if taker is None:
                stats['unmatched_assisters'].append(f'{eid}: {taker_name}')
        rows.append({**base(side, s['time'], s.get('addedTime')), 'Event_Type': 'shot', 'Situation': s['situation'],
                     'Taker_ID_Sofascore': taker, 'Taker_Sofascore': taker_name, 'Taker_Local': who(taker),
                     'Taker_Role': role, 'Shooter_ID_Sofascore': s['player']['id'],
                     'Shooter_Sofascore': s['player']['name'], 'Shooter_Local': who(s['player']['id']),
                     'xG': s.get('xg'), 'Shot_Outcome': s.get('shotType'), 'Is_Goal': s.get('shotType') == 'goal',
                     'Comment': text})
    end = max([90.0] + [c['time'] for c in comments if 'time' in c])
    spans = on_pitch(players, comments, end)
    pitch = [dict(Season=season, Event_ID_Sofascore=eid, Kickoff=kickoff, Team=team_of[players[pid][0]],
                  Player_ID_Sofascore=pid, Player_Sofascore=players[pid][1], Local_Player=who(pid),
                  On=on, Off=off, Match_End=end, Minutes_Played=players[pid][4])
             for pid, (on, off) in spans.items() if pid in players]
    return rows, pitch, stats


EVENT_COLUMNS = ['Season', 'Event_ID_Sofascore', 'Kickoff', 'Team', 'Opponent', 'Venue', 'Minute', 'Added_Time',
                 'Event_Type', 'Situation', 'Taker_ID_Sofascore', 'Taker_Sofascore', 'Taker_Local', 'Taker_Role',
                 'Shooter_ID_Sofascore', 'Shooter_Sofascore', 'Shooter_Local', 'xG', 'Shot_Outcome', 'Is_Goal',
                 'Named_Player_ID_Sofascore', 'Named_Player_Local', 'Named_Player_Role', 'Comment']


def build(raw=f.RAW, data=f.DATA):
    """Rebuild both outputs from every match with lineups, commentary and shot map cached."""
    raw = Path(raw)
    names = local_names(data)
    events, pitch = [], []
    summary = {'matches': 0, 'set_piece_shots': 0, 'unpaired_shots': 0, 'assisted': 0, 'unmatched_assisters': []}
    for path in sorted((raw / 'comments').glob('*/*.json')):
        season, eid = path.parent.name, path.stem
        lineup_path = raw / 'matches' / season / f'{eid}.json'
        shot_path = feed_path(raw, 'shotmap', season, eid)
        if not (lineup_path.exists() and shot_path.exists()):
            continue
        rows, spans, stats = match_rows(int(season), load(lineup_path), load(path)['payload']['comments'],
                                        load(shot_path)['payload']['shotmap'], names)
        events += rows
        pitch += spans
        summary['matches'] += 1
        summary['set_piece_shots'] += stats['shots']
        summary['unpaired_shots'] += stats['unpaired_shots']
        summary['assisted'] += stats['assists']
        summary['unmatched_assisters'] += stats['unmatched_assisters']
    if not events:
        return summary
    ev = pd.DataFrame(events).reindex(columns=EVENT_COLUMNS)
    ev = ev.sort_values(['Kickoff', 'Event_ID_Sofascore', 'Minute', 'Event_Type'], kind='stable')
    to_csv_atomic(ev, paths.current_file(EVENTS_FILE, root=data), index=False, encoding="utf-8-sig")
    to_csv_atomic(pd.DataFrame(pitch).sort_values(['Kickoff', 'Event_ID_Sofascore', 'Team', 'On'], kind='stable'),
                  paths.current_file(ON_PITCH_FILE, root=data), index=False, encoding="utf-8-sig")
    summary['rows'] = len(ev)
    return summary


def cached_events(raw, season):
    """Event IDs with both commentary and shot map frozen for *season*."""
    have = [{p.stem for p in (Path(raw) / kind / str(season)).glob('*.json')} for kind in FEEDS]
    return have[0] & have[1]


def main():
    summary = build()
    print(json.dumps({**summary, 'unmatched_assisters': summary['unmatched_assisters'][:20]}, indent=2,
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
