"""
Understat season data for the other top leagues (AGENDA "OVERNIGHT, new-player priors", step 3).

New-to-the-PL players' goal/assist priors need their previous seasons' output. Understat
publishes, for La Liga, Bundesliga, Serie A, Ligue 1 and the Russian Premier League, one
`getLeagueData/<league>/<season>` payload per season: {teams, players, dates}. `players`
holds season totals (games, time, goals, npg, xG, npxG, assists, xA, position, team_title);
`teams` holds every club's per-match xG / xGA history, used for source-team strength.

Frozen one-off download: data/raw/understat/league/<LEAGUE>_<season>.json (the EPL files'
naming), never rewritten; files that already exist are skipped. SHA256 of every file is
written to data/raw/understat/league/manifest_leagues.json. Season = the starting year
(2014 = 2014/15), as Understat and the EPL cache use it.

Discipline (CLAUDE.md §2): plain requests + raise_for_status(), ~6 s between requests,
personal non-commercial use. A 403 or 429 stops the run at once (no retries, no
workarounds). Understat xG stays its own model: it is only ever used through fitted
league translation factors, never spliced into PL match xG.

Run locally:
    python fetch_understat_leagues.py            # fetch what is missing, then write the manifest
    python fetch_understat_leagues.py --manifest # manifest only, no downloads
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[2]
LEAGUE_CACHE = _ROOT / "data" / "raw" / "understat" / "league"
MANIFEST = LEAGUE_CACHE / "manifest_leagues.json"

LEAGUES = ["La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
FIRST_SEASON = 2014
LAST_SEASON = 2025      # last completed season; bump manually each summer
URL = "https://understat.com/getLeagueData/{league}/{season}"
THROTTLE_S = float(os.environ.get("UNDERSTAT_THROTTLE", "6"))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) personal non-commercial FPL research",
    "X-Requested-With": "XMLHttpRequest",
}


def raw_path(league, season):
    return LEAGUE_CACHE / f"{league}_{season}.json"


def fetch_one(league, season):
    resp = requests.get(URL.format(league=league, season=season), headers=HEADERS, timeout=40)
    if resp.status_code in (403, 429):
        raise SystemExit(f"BLOCKED: {resp.status_code} for {league} {season}; stopping (no workarounds)")
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "players" not in data or "teams" not in data:
        raise RuntimeError(f"{league} {season}: unexpected payload keys")
    return data


def write_manifest():
    entries = {}
    for league in LEAGUES:
        for season in range(FIRST_SEASON, LAST_SEASON + 1):
            p = raw_path(league, season)
            if p.exists():
                entries[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps(entries, indent=1, sort_keys=True), encoding="utf-8")
    print(f"manifest: {len(entries)} files -> {MANIFEST}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="write the manifest only")
    args = ap.parse_args()
    LEAGUE_CACHE.mkdir(parents=True, exist_ok=True)
    if not args.manifest:
        n_new = 0
        for league in LEAGUES:
            for season in range(FIRST_SEASON, LAST_SEASON + 1):
                p = raw_path(league, season)
                if p.exists():
                    continue
                data = fetch_one(league, season)
                p.write_text(json.dumps(data), encoding="utf-8")
                n_new += 1
                print(f"{league} {season}: {len(data['players'])} players, {len(data['teams'])} teams", flush=True)
                time.sleep(THROTTLE_S)
        print(f"downloaded {n_new} new files")
    write_manifest()


if __name__ == "__main__":
    sys.exit(main())
