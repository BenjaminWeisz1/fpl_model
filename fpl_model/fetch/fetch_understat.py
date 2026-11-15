"""
Understat per-match player positions (AGENDA §4, cold-start data sources §3).

Understat is the one free source that labels the *formation slot* every player
occupied in every Premier League match, consistently from 2014/15 onward
(GK, DR, DC, DL, DMR/DMC/DML, MR, MC, ML, AMR/AMC/AML, FWR/FW/FWL, Sub). The
FPL API only gives GK/DEF/MID/FWD and the frozen FBref archive's per-match
`Pos` column stops at 2024, so this is what fills the 2025+ gap and lets the
model detect role changes (e.g. a centre-back drifting to full-back).

How the data is reached (Understat moved its rosters behind an XHR endpoint;
they are no longer inline in the match HTML):

  * league metadata -- GET https://understat.com/getLeagueData/EPL/<season>
      returns {teams, players, dates}; `dates` is the season's fixture list,
      one entry per match with its Understat match id, both clubs, the
      datetime, the score and whether it has been played (`isResult`).
  * match rosters   -- GET https://understat.com/getMatchData/<match_id>
      (needs the X-Requested-With: XMLHttpRequest header) returns
      {rosters: {h, a}, shots, tmpl}; each roster player carries `player`,
      `player_id`, `team_id`, `position`, `time` (minutes), `roster_in/out`
      and Understat's own xG/xA.

Both raw responses are cached verbatim under data/raw/understat/ and never
rewritten once present, so the ~3,800-match backfill is resumable: delete a
cache file to force a re-download. The tidy, pipeline-facing output is
rebuilt from the cache every run into data/current/understat_match_positions.csv.

Discipline (CLAUDE.md §2):
  * one fetcher per source, plain requests + raise_for_status(), paths from
    Path(__file__), CURRENT_SEASON bumped by hand;
  * append-only raw cache -- historical pages are frozen;
  * Understat xG/xA stay in their own columns (Understat_xG / Understat_xA),
    never spliced with the FPL/Opta or archived FBref xG models;
  * throttle to Understat's courtesy ~1 request / 6 s (override with the
    UNDERSTAT_THROTTLE env var), keep use personal and non-commercial.

Position labels are stored twice: `Position_Understat` is the lossless raw
slot, `Position` is a deterministic FBref-style rendering (see BASE_POS_MAP
and the back-three wing-back rule below). Because the raw label is kept, any
mapping can be recomputed later without re-scraping.

Run locally:
    python fetch_understat.py                 # full backfill (resumable), then rebuild CSV
    python fetch_understat.py --build-only    # rebuild CSV from the cache, no downloads
    python fetch_understat.py --season 2026   # one season only
    python fetch_understat.py --limit 50      # cap new match downloads (testing)
    python fetch_understat.py --weekly        # current season only, tolerant of network errors
"""
from fpl_model import paths
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

_ROOT = paths.ROOT
_DATA = paths.CURRENT
RAW_DIR = paths.RAW / "understat"
LEAGUE_CACHE = paths.RAW / "understat" / "league"
MATCH_CACHE = paths.RAW / "understat" / "matches"
MATCH_RESULTS_FILE = paths.CURRENT / "match_results.csv"
OUT_FILE = paths.CURRENT / "understat_match_positions.csv"

from fpl_model.league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()
FIRST_SEASON = 2017     # per-match positions are wanted from 2017/18 onward
LEAGUE = "EPL"

BASE_URL = "https://understat.com"
LEAGUE_URL = BASE_URL + "/getLeagueData/{league}/{season}"
MATCH_URL = BASE_URL + "/getMatchData/{match_id}"
THROTTLE_S = float(os.environ.get("UNDERSTAT_THROTTLE", "6"))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) personal non-commercial FPL research",
    "X-Requested-With": "XMLHttpRequest",
}
MAX_RETRIES = 4

# Understat club title -> match_results.csv (fbref-style) name. Unlisted clubs
# are identical on both sides; the coverage check in build_positions() catches
# any miss and exits non-zero (except in --weekly mode, which only warns).
UNDERSTAT_TO_FBREF_TEAM_NAME = {
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Nottingham Forest": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "Wolverhampton Wanderers": "Wolves",
    "West Bromwich Albion": "West Brom",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
    "Norwich": "Norwich City",
    "Cardiff": "Cardiff City",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
}

# Raw Understat slot -> FBref-style label. The wide-defender slots (DR/DL) and
# the defensive-mid-wide slots (DMR/DML) are re-rendered as wing-backs (WB)
# when the side started three centre-backs -- see _mapped_position().
BASE_POS_MAP = {
    "GK": "GK",
    "DC": "CB",
    "DR": "RB",
    "DL": "LB",
    "DMC": "DM",
    "DMR": "DM",
    "DML": "DM",
    "MC": "CM",
    "MR": "RM",
    "ML": "LM",
    "AMC": "AM",
    "AMR": "RW",
    "AML": "LW",
    "FWR": "RW",
    "FWL": "LW",
    "FW": "FW",
    "Sub": "Sub",
}
_WINGBACK_SLOTS = {"DR", "DL", "DMR", "DML"}


def fbref_team_name(understat_title):
    return UNDERSTAT_TO_FBREF_TEAM_NAME.get(understat_title, understat_title)


# ── raw download + cache ───────────────────────────────────────────────────────

def _get_json(url, throttle=True):
    """GET a JSON endpoint with retries/backoff. Never returns an error page."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=40)
            resp.raise_for_status()
            data = resp.json()          # refuses to accept a non-JSON error page
            if throttle:
                time.sleep(THROTTLE_S)
            return data
        except Exception as exc:        # noqa: BLE001 - transient network/parse errors
            last = exc
            wait = THROTTLE_S * (attempt + 2)
            print(f"    retry {attempt + 1}/{MAX_RETRIES} for {url} ({exc}); sleep {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def league_raw_path(season):
    return LEAGUE_CACHE / f"{LEAGUE}_{season}.json"


def match_raw_path(match_id):
    return MATCH_CACHE / f"{match_id}.json"


def load_league(season, refresh=False):
    """The season's fixture list (list of `dates` entries), from cache if present."""
    path = league_raw_path(season)
    if path.exists() and not refresh:
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _get_json(LEAGUE_URL.format(league=LEAGUE, season=season))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data["dates"]


def load_match(match_id):
    """One match's getMatchData payload, downloading + caching it if absent.

    Returns None if it was already cached (nothing new downloaded)."""
    path = match_raw_path(match_id)
    if path.exists():
        return None
    data = _get_json(MATCH_URL.format(match_id=match_id))
    if "rosters" not in data:
        raise RuntimeError(f"match {match_id}: no rosters in response")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def read_cached_match(match_id):
    path = match_raw_path(match_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ── parsing ────────────────────────────────────────────────────────────────────

def _to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapped_position(raw, back_three):
    """Render a raw Understat slot as an FBref-style label.

    In a back three/five (three or more centre-backs started) the wide
    defenders/wing-back slots become WB; otherwise DR/DL are full-backs and
    DMR/DML are read as defensive midfielders. Substitutes keep `Sub` here --
    resolving a sub to the slot they filled is a downstream step (AGENDA §4.6).
    """
    if back_three and raw in _WINGBACK_SLOTS:
        return "WB"
    return BASE_POS_MAP.get(raw, raw)


def parse_match(match_id, fixture):
    """Turn one cached match into per-player position rows.

    `fixture` is the season's `dates` entry (supplies clubs, date, score).
    Returns a list of row dicts.
    """
    data = read_cached_match(match_id)
    if data is None:
        return []
    rosters = data.get("rosters", {})

    home_title = fixture["h"]["title"]
    away_title = fixture["a"]["title"]
    home_fbref = fbref_team_name(home_title)
    away_fbref = fbref_team_name(away_title)
    date = str(fixture.get("datetime", ""))[:10]
    goals = fixture.get("goals", {}) or {}
    xg = fixture.get("xG", {}) or {}

    rows = []
    for side, opp_fbref, opp_title in (("h", away_fbref, away_title), ("a", home_fbref, home_title)):
        team_fbref = home_fbref if side == "h" else away_fbref
        team_title = home_title if side == "h" else away_title
        roster = rosters.get(side, {}) or {}
        # starters are everyone not listed as a substitute
        starters = [p for p in roster.values() if p.get("position") != "Sub"]
        back_three = sum(1 for p in starters if p.get("position") == "DC") >= 3
        for rec in roster.values():
            raw_pos = rec.get("position")
            is_starter = raw_pos != "Sub"
            rows.append({
                "Season": int(fixture.get("season", _season_from_date(date))),
                "MatchID_Understat": int(match_id),
                "Date": date,
                "Home": home_fbref,
                "Away": away_fbref,
                "Player": rec.get("player"),
                "Player_ID_Understat": _to_int(rec.get("player_id")),
                "Team": team_fbref,
                "Team_Understat": team_title,
                "Opponent": opp_fbref,
                "Venue": "Home" if side == "h" else "Away",
                "Position_Understat": raw_pos,
                "Position": _mapped_position(raw_pos, back_three),
                "Is_Starter": is_starter,
                "Back_Three": back_three,
                "Minutes": _to_int(rec.get("time")),
                "Goals": _to_int(rec.get("goals")),
                "Assists": _to_int(rec.get("assists")),
                "Understat_xG": _to_float(rec.get("xG")),
                "Understat_xA": _to_float(rec.get("xA")),
                "Home_Goals": _to_int(goals.get("h")),
                "Away_Goals": _to_int(goals.get("a")),
                "Home_xG_Understat": _to_float(xg.get("h")),
                "Away_xG_Understat": _to_float(xg.get("a")),
            })
    return rows


def _season_from_date(date_str):
    """Fallback season (start year) from a YYYY-MM-DD date string."""
    try:
        year = int(date_str[:4])
        month = int(date_str[5:7])
        return year if month >= 7 else year - 1
    except (ValueError, TypeError):
        return None


# ── build the tidy CSV from the cache ──────────────────────────────────────────

def build_positions(seasons, weekly=False):
    """Parse every cached match for `seasons` into understat_match_positions.csv.

    Attaches the fbref MatchID from match_results.csv on (Season, Home, Away)
    where available, and runs a team-name coverage check.
    """
    all_rows = []
    n_matches = 0
    for season in seasons:
        try:
            fixtures = load_league(season)
        except Exception as exc:        # noqa: BLE001
            print(f"  {season}: league metadata unavailable ({exc}); skipped in build")
            continue
        for fx in fixtures:
            if not fx.get("isResult"):
                continue                # only played matches have rosters
            mid = fx["id"]
            if not match_raw_path(mid).exists():
                continue
            rows = parse_match(mid, {**fx, "season": season})
            if rows:
                n_matches += 1
                all_rows.extend(rows)

    if not all_rows:
        print("  No cached matches to build from -- nothing written.")
        return None

    df = pd.DataFrame(all_rows)

    # Attach the fbref MatchID so this joins straight onto match_results.csv.
    if MATCH_RESULTS_FILE.exists():
        mr = pd.read_csv(MATCH_RESULTS_FILE, encoding="utf-8-sig")
        mr_keys = mr[["MatchID", "Season", "Home", "Away"]].drop_duplicates(
            subset=["Season", "Home", "Away"])
        df = df.merge(mr_keys, on=["Season", "Home", "Away"], how="left")
        matched = df["MatchID"].notna().mean()
        print(f"  Joined to match_results MatchID on {matched:.1%} of rows")

    # Team-name coverage check (skip in weekly mode, which only warns).
    seen = set(df["Team_Understat"])
    mapped_ok = {t for t in seen if fbref_team_name(t) in set(_fbref_names())}
    missing = sorted(seen - mapped_ok)
    if missing:
        msg = f"Unmapped Understat team names: {missing}"
        if weekly:
            print(f"  WARNING: {msg}")
        else:
            sys.exit(msg + " -- add them to UNDERSTAT_TO_FBREF_TEAM_NAME and re-run.")

    cols = ["Season", "MatchID", "MatchID_Understat", "Date", "Home", "Away",
            "Player", "Player_ID_Understat", "Team", "Team_Understat", "Opponent",
            "Venue", "Position_Understat", "Position", "Is_Starter", "Back_Three",
            "Minutes", "Goals", "Assists",
            "Understat_xG", "Understat_xA", "Home_Goals", "Away_Goals",
            "Home_xG_Understat", "Away_xG_Understat"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["Season", "MatchID_Understat", "Venue", "Is_Starter"],
                              ascending=[True, True, True, False])
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  Wrote {len(df):,} rows ({n_matches} matches, "
          f"{df['Season'].nunique()} seasons) -> {OUT_FILE.name}")
    return df


def _fbref_names():
    if not MATCH_RESULTS_FILE.exists():
        return set()
    mr = pd.read_csv(MATCH_RESULTS_FILE, encoding="utf-8-sig")
    return set(mr["Home"]) | set(mr["Away"])


# ── download driver ────────────────────────────────────────────────────────────

def download_seasons(seasons, limit=None, weekly=False, refresh_league=False,
                     retry_passes=2):
    """Cache league metadata + every played match's rosters for `seasons`.

    Resilient by design: a match that still fails after _get_json's own
    retry/backoff is logged and deferred, never fatal -- a multi-hour backfill
    must not be lost to one transient DNS/connection blip. Deferred matches get
    `retry_passes` further attempts at the end (blips usually clear). Returns
    (downloaded, failed_ids); anything left in failed_ids is simply picked up on
    the next run, since cached matches are always skipped.

    In --weekly mode the season's league metadata is ALWAYS re-fetched, even
    though a cached copy exists: the cached `dates` list is frozen at whatever
    gameweek had been played when it was written, so trusting it would hide the
    latest gameweek's newly-finished matches and download nothing. Historical
    (non-weekly) seasons keep their frozen league cache -- those fixtures never
    change.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    failed = []
    stop = False
    league_refresh = refresh_league or weekly
    for season in seasons:
        if stop:
            break
        try:
            fixtures = load_league(season, refresh=league_refresh)
        except Exception as exc:        # noqa: BLE001
            print(f"{season}: league metadata unavailable ({exc}); skipping season")
            continue
        played = [fx for fx in fixtures if fx.get("isResult")]
        todo = [fx for fx in played if not match_raw_path(fx["id"]).exists()]
        print(f"{season}: {len(played)} played, {len(played) - len(todo)} cached, {len(todo)} to download")
        for fx in todo:
            if limit is not None and downloaded >= limit:
                print(f"  reached --limit {limit}; stopping downloads")
                stop = True
                break
            try:
                load_match(fx["id"])
                downloaded += 1
                if downloaded % 25 == 0:
                    print(f"  ...downloaded {downloaded} new matches")
            except Exception as exc:    # noqa: BLE001
                print(f"  WARNING: match {fx['id']} failed ({exc}); deferring to retry pass")
                failed.append(fx["id"])

    # Retry deferred matches; a transient outage normally clears within a pass.
    for attempt in range(1, retry_passes + 1):
        if not failed:
            break
        print(f"Retry pass {attempt}/{retry_passes}: {len(failed)} deferred match(es)")
        still_failed = []
        for mid in failed:
            if limit is not None and downloaded >= limit:
                still_failed.append(mid)
                continue
            try:
                load_match(mid)
                downloaded += 1
            except Exception as exc:    # noqa: BLE001
                print(f"  still failing: match {mid} ({exc})")
                still_failed.append(mid)
            time.sleep(THROTTLE_S)
        failed = still_failed

    return downloaded, failed


def main():
    ap = argparse.ArgumentParser(description="Fetch Understat per-match player positions.")
    ap.add_argument("--season", type=int, help="only this season (start year)")
    ap.add_argument("--limit", type=int, help="cap the number of new match downloads")
    ap.add_argument("--build-only", action="store_true", help="rebuild the CSV from cache, no downloads")
    ap.add_argument("--refresh-league", action="store_true", help="re-download league metadata")
    ap.add_argument("--weekly", action="store_true",
                    help="current season only; tolerant of network errors (for the pipeline stage)")
    args = ap.parse_args()

    if args.season is not None:
        seasons = [args.season]
    elif args.weekly:
        seasons = [CURRENT_SEASON]
    else:
        seasons = list(range(FIRST_SEASON, CURRENT_SEASON + 1))

    failed = []
    if not args.build_only:
        n, failed = download_seasons(seasons, limit=args.limit, weekly=args.weekly,
                                     refresh_league=args.refresh_league)
        print(f"Downloaded {n} new match pages.")

    print("Building understat_match_positions.csv from cache...")
    build_positions(list(range(FIRST_SEASON, CURRENT_SEASON + 1)), weekly=args.weekly)

    if failed:
        print("\n" + "=" * 60)
        print(f"INCOMPLETE: {len(failed)} match(es) could not be downloaded "
              f"(network): {failed[:20]}{' ...' if len(failed) > 20 else ''}")
        print("The run is resumable -- just re-run this script to retry the gaps "
              "(cached matches are skipped).")
        print("=" * 60)



if __name__ == "__main__":
    main()
