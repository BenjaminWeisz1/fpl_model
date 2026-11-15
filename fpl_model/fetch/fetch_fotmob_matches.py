"""
FotMob match pages: per-match positions, shots and red cards from 2026 on (CHANGELOG §197).

By user decision (2026-11-03) FotMob replaces Understat as the weekly source of per-match positions, penalty
shots and red-card timing from the 2026 season on. Understat stopped answering at GW9 of 2026; its frozen cache
in data/raw/understat/ stays the source for 2017-2025. FotMob's xG is Opta's, the same provider as the FPL xG the
models use, and only shot timing and relative sizes are taken from it.

How the data is reached (FotMob's robots.txt disallows /api/* only; match and league pages are allowed, §115):

  * league fixtures -- GET https://www.fotmob.com/leagues/47/fixtures/premier-league. Its __NEXT_DATA__ lists
      the season's 380 matches with FotMob ids, clubs, kick-off and score. A dated copy is saved per run.
  * match page      -- GET https://www.fotmob.com/match/<id>. Never use the league page's /matches/<slug> link:
      the slug is shared by every meeting of the two clubs (cup ties included) and the #<id> part never reaches
      the server, so it serves whichever meeting FotMob picks (§197). Every saved page is checked: matchId = the
      requested id, leagueId 47, finished, score = match_results.csv.

Pages are cached verbatim under data/raw/fotmob/matches/<season>/<id>.html and never rewritten. Project matches
come from match_results.csv and are paired with FotMob by (home, away) within the season, checked on the score.
FotMob's `round` is not FPL's gameweek and is never used.

Identity: playerStats[].optaId = FPL's `code` (fpl_player_codes.csv, written by fetch_bootstrap.py), and FPL
first + second name is the project Player. A player missing there (left the league before his code was saved)
falls back to an exact accent-free name match within his team's player_match_stats rows for that fixture.

Positions: FotMob gives each starter a `positionId` (tens = line, units = side: 32 RB ... 38 LB, 64/66 a
holding pair, 115 striker) and the team a formation. They are rendered in Understat's slot vocabulary (DC, DR,
DMC, AMC, FW, ...) by a modal mapping fitted at run time on the matches both sources hold (2026 GW1-8, frozen in
data/raw/fotmob/understat_overlap_2026.csv by `--freeze-overlap` before the switch):
(formation, positionId) first, then positionId alone, then a fixed line/side rule for codes never seen. The
mapping reproduces Understat's conventions, e.g. 3-4-2-1's central pair is MC there and DM in FotMob.
Back_Three and the FBref-style Position follow fetch_understat.py exactly (3+ DC starters; _mapped_position).

Outputs (data/current/, rebuilt from the cache every run):
  * fotmob_match_positions.csv -- one row per player who played (starters + substitutes who came on)
  * fotmob_shots.csv           -- every shot, with minute (+ added time / 100), xG, situation, result

The stage stops (non-zero exit) when a played project match of the season has no valid page, so a FotMob
outage cannot silently drop a round as the Understat outage did. FOTMOB_ALLOW_MISSING=1 lets it continue.

    python -m fpl_model.fetch.fetch_fotmob_matches             # fetch new pages (7 s apart), then rebuild the outputs
    python -m fpl_model.fetch.fetch_fotmob_matches --offline   # rebuild from the cache only
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from fpl_model import (
    config,
    league_rates,  # noqa: E402
    paths,
)
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.fetch.fetch_understat import _mapped_position  # noqa: E402

_ROOT = paths.ROOT
_DATA = paths.CURRENT
RAW = paths.RAW / "fotmob" / "matches"
CURRENT_SEASON = league_rates.current_season()
FIRST_SEASON = 2026          # first season served by FotMob instead of Understat (user decision, 2026-11-03)
LEAGUE_ID = 47
LEAGUE_URL = "https://www.fotmob.com/leagues/47/fixtures/premier-league"
MATCH_URL = "https://www.fotmob.com/match/{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
THROTTLE_S = 7
MIN_OVERLAP_STARTERS = 500   # the slot mapping needs at least this many starters seen by both sources

POSITIONS_OUT = paths.CURRENT / "fotmob_match_positions.csv"
SHOTS_OUT = paths.CURRENT / "fotmob_shots.csv"
CODES_FILE = paths.CURRENT / "fpl_player_codes.csv"
OVERLAP_FILE = paths.RAW / "fotmob" / "understat_overlap_2026.csv"   # frozen; see freeze_overlap()
RESULTS_FILE = paths.CURRENT / "match_results.csv"
PMS_FILE = paths.CURRENT / "player_match_stats.csv"

TEAM_NAMES = {"Manchester United": "Manchester Utd", "Nottingham Forest": "Nott'ham Forest",
              "AFC Bournemouth": "Bournemouth", "Brighton & Hove Albion": "Brighton",
              "Tottenham Hotspur": "Tottenham", "Newcastle United": "Newcastle Utd",
              "Wolverhampton Wanderers": "Wolves", "West Ham United": "West Ham",
              "Sheffield United": "Sheffield Utd"}
KEY = ["Season", "Team", "Opponent", "Venue", "Player"]


def team_name(name):
    return TEAM_NAMES.get(name, name)


def norm(s):
    s = "".join(ch for ch in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace("-", " ")).split()


def page_props(html):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("no __NEXT_DATA__ block")
    return json.loads(m.group(1))["props"]["pageProps"]


def season_dir(season):
    return RAW / str(season)


# ── fetching ────────────────────────────────────────────────────────────────

def league_fixtures(season, offline=False):
    """The season's FotMob fixture list, from a fresh league page or the newest saved copy."""
    folder = season_dir(season)
    html = None
    if not offline:
        try:
            resp = requests.get(LEAGUE_URL, headers=HEADERS, timeout=40)
            resp.raise_for_status()
            html = resp.text
            props = page_props(html)
            selected = str(props["details"].get("selectedSeason", ""))
            if not selected.startswith(str(season)):
                raise ValueError(f"league page shows season {selected!r}, expected {season}")
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            (folder / f"league_47_fixtures_{stamp}.html").write_text(html, encoding="utf-8")
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"WARNING: FotMob league page unavailable ({exc}); using the newest saved copy")
            html = None
    if html is None:
        saved = sorted(folder.glob("league_47_fixtures*.html"), key=lambda p: p.stat().st_mtime)
        if not saved:
            return []
        html = saved[-1].read_text(encoding="utf-8")
    matches = page_props(html)["fixtures"]["allMatches"]
    return [{"id": str(m["id"]), "home": team_name(m["home"]["name"]), "away": team_name(m["away"]["name"]),
             "score": (m.get("status") or {}).get("scoreStr"), "finished": bool((m.get("status") or {}).get("finished"))}
            for m in matches]


def played_project_matches(season):
    r = pd.read_csv(RESULTS_FILE, encoding="utf-8-sig")
    r = r[(r["Season"] == season) & r["Home_Goals"].notna() & r["Away_Goals"].notna()]
    return [(h, a, f"{int(hg)} - {int(ag)}") for h, a, hg, ag in zip(r["Home"], r["Away"], r["Home_Goals"], r["Away_Goals"])]


def check_page(props, match_id, score=None):
    """Raise ValueError unless the page is the finished Premier League match requested."""
    g = props.get("general") or {}
    if str(g.get("matchId")) != str(match_id):
        raise ValueError(f"page is match {g.get('matchId')}, not {match_id}")
    if int(g.get("leagueId") or 0) != LEAGUE_ID:
        raise ValueError(f"match {match_id} is league {g.get('leagueId')}, not the Premier League")
    if not g.get("finished") or not (props.get("content") or {}).get("playerStats"):
        raise ValueError(f"match {match_id} is not finished or has no player stats yet")
    if score is not None:
        got = (props.get("header") or {}).get("status", {}).get("scoreStr")
        if got != score:
            raise ValueError(f"match {match_id} score {got!r} differs from match_results {score!r}")


def fetch_page(match_id, season, score):
    resp = requests.get(MATCH_URL.format(match_id), headers=HEADERS, timeout=40)
    resp.raise_for_status()
    check_page(page_props(resp.text), match_id, score)
    folder = season_dir(season)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{match_id}.part"
    tmp.write_text(resp.text, encoding="utf-8")
    tmp.replace(folder / f"{match_id}.html")


def sync(season, offline=False):
    """Fetch every played project match without a valid cached page. Returns {(home, away): id} and the gaps."""
    fixtures = {(f["home"], f["away"]): f for f in league_fixtures(season, offline)}
    ids, missing, fetched = {}, [], 0
    for home, away, score in played_project_matches(season):
        f = fixtures.get((home, away))
        if f is None:
            missing.append(f"{home} v {away}: not on the FotMob fixture list")
            continue
        path = season_dir(season) / f"{f['id']}.html"
        if path.exists():
            ids[(home, away)] = f["id"]
            continue
        if offline:
            missing.append(f"{home} v {away}: no cached page (offline)")
            continue
        if fetched:
            time.sleep(THROTTLE_S)
        try:
            fetch_page(f["id"], season, score)
            fetched += 1
            ids[(home, away)] = f["id"]
            print(f"  fetched {home} v {away} ({f['id']})")
        except (requests.RequestException, ValueError) as exc:
            missing.append(f"{home} v {away}: {exc}")
    print(f"FotMob {season}: {fetched} new page(s); {len(ids)} matches with a page; {len(missing)} missing")
    return ids, missing


# ── parsing ─────────────────────────────────────────────────────────────────

def minute(t, added):
    return float(t) + float(added or 0) / 100.0


def parse_page(path, season):
    """One cached page -> (player rows, shot rows). Player names are FotMob's; identity is resolved later."""
    props = page_props(Path(path).read_text(encoding="utf-8"))
    check_page(props, Path(path).stem)
    g, c = props["general"], props["content"]
    opta = {str(k): str(v.get("optaId")) for k, v in c["playerStats"].items()}
    lineup = c["lineup"]
    home, away = team_name(lineup["homeTeam"]["name"]), team_name(lineup["awayTeam"]["name"])
    hs, as_ = (props["header"]["teams"][0]["score"], props["header"]["teams"][1]["score"])
    reds = {}
    for e in c["matchFacts"]["events"]["events"]:
        if e.get("type") == "Card" and e.get("card") in ("Red", "YellowRed"):
            reds[str(e.get("playerId") or (e.get("player") or {}).get("id"))] = minute(e["time"], e.get("overloadTime"))
    players, side_of = [], {}
    for key, venue in (("homeTeam", "Home"), ("awayTeam", "Away")):
        t = lineup[key]
        team, opp = (home, away) if venue == "Home" else (away, home)
        side_of[str(t["id"])] = (team, opp, venue)
        for starter, plist in ((True, t.get("starters") or []), (False, t.get("subs") or [])):
            for pl in plist:
                ev = (pl.get("performance") or {}).get("substitutionEvents") or []
                on = [x["time"] for x in ev if x.get("type") == "subIn"]
                off = [x["time"] for x in ev if x.get("type") == "subOut"]
                if not starter and not on:
                    continue
                pid = str(pl["id"])
                players.append({"Season": season, "MatchID_FotMob": int(g["matchId"]), "Date": g["matchTimeUTCDate"][:10],
                                "Home": home, "Away": away, "Home_Goals": hs, "Away_Goals": as_,
                                "Team": team, "Opponent": opp, "Venue": venue, "Player_FotMob": pl["name"],
                                "Player_ID_FotMob": int(pid), "Opta_ID": opta.get(pid),
                                "FotMob_Position_ID": pl.get("positionId") if starter else None,
                                "Formation": t.get("formation") or "", "Is_Starter": starter,
                                "Sub_On_Minute": on[0] if on else None, "Sub_Off_Minute": off[0] if off else None,
                                "Red_Minute": reds.get(pid)})
    shots = []
    for s in c["shotmap"]["shots"]:
        team, opp, venue = side_of[str(s["teamId"])]
        shots.append({"Season": season, "MatchID_FotMob": int(g["matchId"]), "Team": team, "Opponent": opp,
                      "Venue": venue, "Player_FotMob": s.get("playerName"), "Player_ID_FotMob": s.get("playerId"),
                      "Minute": minute(s["min"], s.get("minAdded")), "Period": s.get("period"),
                      "xG": s.get("expectedGoals"), "Situation": s.get("situation"), "Event_Type": s.get("eventType"),
                      "Own_Goal": bool(s.get("isOwnGoal"))})
    return players, shots


def resolve_identity(players):
    """Project Player for every FotMob row: Opta ID -> FPL code, else an exact name match within the fixture."""
    codes = pd.read_csv(CODES_FILE, encoding="utf-8-sig", dtype={"Opta_Code": str}) if CODES_FILE.exists() \
        else pd.DataFrame(columns=["Opta_Code", "Player"])
    by_code = dict(zip(codes["Opta_Code"], codes["Player"]))
    pms = pd.read_csv(PMS_FILE, encoding="utf-8-sig", low_memory=False, usecols=["Season", "Team", "Opponent", "Venue", "Player"])
    names = pms.groupby(["Season", "Team", "Opponent", "Venue"])["Player"].apply(list).to_dict()
    out, methods = [], Counter()
    for r in players:
        name = by_code.get(str(r["Opta_ID"]))
        method = "opta_code"
        if name is None:
            cand = [p for p in names.get((r["Season"], r["Team"], r["Opponent"], r["Venue"]), []) if norm(p) == norm(r["Player_FotMob"])]
            name, method = (cand[0], "exact_name") if len(cand) == 1 else (None, "unmapped")
        methods[method] += 1
        out.append(r | {"Player": name, "Identity_Method": method})
    return out, methods


# ── slot mapping ────────────────────────────────────────────────────────────

def rule_slot(pid):
    """Fixed fallback for a positionId never seen in the overlap: line from the leading digits, side from the last."""
    pid = int(pid)
    if pid == 11:
        return "GK"
    line, side = pid // 10, pid % 10
    right, left = side <= 2, side >= 8
    if line >= 10:
        return "FW"
    if line in (3, 4, 5):
        return "DR" if right else "DL" if left else "DC"
    if line == 6:
        return "DMR" if right else "DML" if left else "DMC"
    if line == 7:
        return "MR" if right else "ML" if left else "MC"
    return "AMR" if side <= 3 else "AML" if side >= 7 else "AMC"


def _modal(pairs):
    """{key: most common label}, ties broken alphabetically so the fit is deterministic."""
    counts = defaultdict(Counter)
    for k, v in pairs:
        counts[k][v] += 1
    return {k: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, c in counts.items()}


def fit_slot_map(starters):
    """starters: frame with Formation, FotMob_Position_ID, Position_Understat (Understat's slot)."""
    s = starters.dropna(subset=["FotMob_Position_ID", "Position_Understat"])
    s = s[s["Position_Understat"] != "Sub"]
    by_form = _modal(zip(zip(s["Formation"], s["FotMob_Position_ID"].astype(int)), s["Position_Understat"]))
    by_pid = _modal(zip(s["FotMob_Position_ID"].astype(int), s["Position_Understat"]))
    return by_form, by_pid


def apply_slot_map(formation, pid, by_form, by_pid):
    if pid is None or pd.isna(pid):
        return "Sub"
    pid = int(pid)
    return by_form.get((formation, pid)) or by_pid.get(pid) or rule_slot(pid)


def freeze_overlap(path=OVERLAP_FILE):
    """One-off: save Understat's FIRST_SEASON slots, keyed to project names, before the crosswalk stops making them.

    Source: player_match_positions.csv as built from Understat (all its FIRST_SEASON rows are Understat's at that
    point). Refuses to overwrite: the file is frozen evidence the slot mapping is fitted on every run.
    """
    if path.exists():
        sys.exit(f"{path.name} already exists; it is frozen and never rewritten")
    cw = pd.read_csv(_DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    if "Source" in cw.columns:
        sys.exit("player_match_positions.csv already carries FotMob rows; the overlap must be frozen before the switch")
    cw = cw[cw["Season"] >= FIRST_SEASON][KEY + ["Player_ID_Understat", "Position_Understat", "Is_Starter"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    cw.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"froze {len(cw)} Understat rows ({cw.groupby(['Team', 'Opponent', 'Venue']).ngroups // 2} matches) -> {path}")


def overlap_starters(fm, overlap_file=OVERLAP_FILE):
    """FotMob starters joined to Understat's slot for the same player-match (the matches both sources hold)."""
    if not overlap_file.exists():
        sys.exit(f"missing {overlap_file}; run `fetch_fotmob_matches.py --freeze-overlap` once, before the switch")
    u = pd.read_csv(overlap_file, encoding="utf-8-sig")
    st = fm[fm["Is_Starter"] & fm["Player"].notna()]
    return st.merge(u[KEY + ["Position_Understat"]], on=KEY, how="inner")


def label_slots(fm, by_form, by_pid):
    fm = fm.copy()
    fm["Position_Understat"] = [apply_slot_map(f, p, by_form, by_pid) if s else "Sub"
                                for f, p, s in zip(fm["Formation"], fm["FotMob_Position_ID"], fm["Is_Starter"])]
    dc = fm[fm["Is_Starter"] & (fm["Position_Understat"] == "DC")].groupby(["MatchID_FotMob", "Team"]).size()
    fm["Back_Three"] = [dc.get((m, t), 0) >= 3 for m, t in zip(fm["MatchID_FotMob"], fm["Team"])]
    fm["Position"] = [_mapped_position(r, b) for r, b in zip(fm["Position_Understat"], fm["Back_Three"])]
    return fm


# ── build ───────────────────────────────────────────────────────────────────

def load_season(season):
    players, shots = [], []
    for path in sorted(season_dir(season).glob("*.html")):
        if not path.stem.isdigit():
            continue
        p, s = parse_page(path, season)
        players += p
        shots += s
    return players, shots


def build(seasons):
    players, shots = [], []
    for season in seasons:
        p, s = load_season(season)
        players += p
        shots += s
    players, methods = resolve_identity(players)
    fm = pd.DataFrame(players)
    overlap = overlap_starters(fm)
    if len(overlap) < MIN_OVERLAP_STARTERS:
        sys.exit(f"slot mapping: only {len(overlap)} starters held by both FotMob and Understat "
                 f"(need {MIN_OVERLAP_STARTERS}); cannot render FotMob positions")
    by_form, by_pid = fit_slot_map(overlap)
    fm = label_slots(fm, by_form, by_pid)
    check = overlap.merge(fm[KEY + ["Position_Understat"]].rename(columns={"Position_Understat": "_fm"}), on=KEY)
    agree = float((check["_fm"] == check["Position_Understat"]).mean())
    ids = {(r["Season"], r["MatchID_FotMob"], r["Player_ID_FotMob"]): r["Player"] for r in players}
    sh = pd.DataFrame(shots)
    if len(sh):
        sh["Player"] = [ids.get((s, m, p)) for s, m, p in zip(sh["Season"], sh["MatchID_FotMob"], sh["Player_ID_FotMob"])]
    cols = ["Season", "MatchID_FotMob", "Date", "Home", "Away", "Home_Goals", "Away_Goals", "Team", "Opponent", "Venue",
            "Player", "Player_FotMob", "Player_ID_FotMob", "Opta_ID", "Identity_Method", "FotMob_Position_ID",
            "Formation", "Position_Understat", "Position", "Is_Starter", "Back_Three", "Sub_On_Minute",
            "Sub_Off_Minute", "Red_Minute"]
    fm = fm[cols].sort_values(["Season", "Date", "MatchID_FotMob", "Venue", "Is_Starter", "Player_FotMob"],
                              ascending=[True, True, True, False, False, True])
    dup = fm.dropna(subset=["Player"]).duplicated(KEY).sum()
    if dup:
        sys.exit(f"internal error: {dup} duplicate (Season, Team, Opponent, Venue, Player) keys")
    to_csv_atomic(fm, POSITIONS_OUT, index=False, encoding="utf-8-sig")
    shot_cols = ["Season", "MatchID_FotMob", "Team", "Opponent", "Venue", "Player", "Player_FotMob", "Player_ID_FotMob",
                 "Minute", "Period", "xG", "Situation", "Event_Type", "Own_Goal"]
    to_csv_atomic(sh[shot_cols] if len(sh) else pd.DataFrame(columns=shot_cols), SHOTS_OUT, index=False, encoding="utf-8-sig")
    print(f"  identity: {dict(methods)}")
    print(f"  slot mapping fitted on {len(overlap)} starters held by both sources: {len(by_form)} (formation, slot) "
          f"pairs, {len(by_pid)} slots; reproduces Understat on {agree:.1%} of them")
    print(f"  wrote {len(fm):,} rows ({fm['MatchID_FotMob'].nunique()} matches) -> {POSITIONS_OUT.name}; "
          f"{len(sh):,} shots -> {SHOTS_OUT.name}")
    unmapped = fm[fm["Player"].isna()]
    for r in unmapped.itertuples():
        print(f"WARNING: no project player for FotMob {r.Player_FotMob} ({r.Team} v {r.Opponent}, {r.Date})")
    return fm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="rebuild from the cache, no network")
    parser.add_argument("--freeze-overlap", action="store_true", help="one-off: freeze Understat's 2026 slots")
    args = parser.parse_args()
    if args.freeze_overlap:
        freeze_overlap()
        return
    seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
    missing = []
    for season in seasons:
        _, gaps = sync(season, offline=args.offline)
        missing += [f"{season}: {g}" for g in gaps]
    print("Building fotmob_match_positions.csv and fotmob_shots.csv from the cache...")
    build(seasons)
    if missing:
        for m in missing:
            print(f"MISSING FotMob page -- {m}")
        if not config.get("FOTMOB_ALLOW_MISSING"):
            sys.exit(f"{len(missing)} played match(es) have no FotMob page: positions, penalties and red cards "
                     f"would be incomplete. Rerun when FotMob answers, or set FOTMOB_ALLOW_MISSING=1 to continue.")


if __name__ == "__main__":
    main()
