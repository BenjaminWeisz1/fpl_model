"""
One-off backfill of ClubElo preseason snapshots (AGENDA §2, CHANGELOG §33):
every club's Elo on the day before each Premier League season's first match,
2017/18 onward -- a non-market team-strength signal for promoted teams.

api.clubelo.com/<YYYY-MM-DD> returns all clubs' ratings on that date as CSV
(Rank, Club, Country, Level, Elo, From, To). Each snapshot is saved verbatim
to data/raw/clubelo/clubelo_<Season>.csv and never rewritten once present.
The snapshot date comes from the season's first fixture in the
football-data.co.uk files (fetch_football_data.py must have run first).

load_season_elo() maps ClubElo names to match_results.csv names and checks
that every team of that season has a rating, exiting non-zero otherwise.

Since CHANGELOG §250 (2026-11-15) the API is down for good, so load_season_elo()
reads the frozen mirror data/raw/clubelo_mirror/EloRatings.csv (GitHub
xgabora/Club-Football-Match-Data, MIT; ClubElo's own snapshots to 2025-06-01,
the author's continuation after) at 1 June of the season's starting year.
CLUBELO_SOURCE=api restores the original API snapshots (day before the first match).

Run locally:
    python fetch_clubelo.py           # mirror: coverage check only
    CLUBELO_SOURCE=api python fetch_clubelo.py
"""
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from fetch_football_data import FIRST_SEASON, load_season

_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = _ROOT / "data" / "raw" / "clubelo"
MIRROR_FILE = _ROOT / "data" / "raw" / "clubelo_mirror" / "EloRatings.csv"
SOURCE = os.environ.get("CLUBELO_SOURCE", "mirror")
MATCH_RESULTS_FILE = _ROOT / "data" / "current" / "match_results.csv"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()
URL = "http://api.clubelo.com/{date}"
THROTTLE_S = 3
HEADERS = {"User-Agent": "personal non-commercial FPL research"}

# ClubElo name -> match_results.csv (fbref-style) name. Unlisted = identical;
# the coverage check catches misses.
CLUBELO_TO_FBREF_TEAM_NAME = {
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Newcastle": "Newcastle Utd",
    "Forest": "Nott'ham Forest",
    "Nott'm Forest": "Nott'ham Forest",
    "Nottm Forest": "Nott'ham Forest",  # ClubElo's name 2022-2024; see load_season_elo for 2026 duplicates
    "Nottingham": "Nott'ham Forest",
    "Sheffield United": "Sheffield Utd",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Stoke": "Stoke City",
    "Swansea": "Swansea City",
    "Cardiff": "Cardiff City",
    "Norwich": "Norwich City",
    "Luton": "Luton Town",
    "Ipswich": "Ipswich Town",
    "Coventry": "Coventry City",
    "Hull": "Hull City",
}


def snapshot_date(season):
    """Day before the season's first fixture (football-data.co.uk dates)."""
    d = pd.to_datetime(load_season(season)["Date"], dayfirst=True)
    return (d.min() - pd.Timedelta(days=1)).date().isoformat()


def raw_path(season):
    return RAW_DIR / f"clubelo_{season}.csv"


def download(season):
    date = snapshot_date(season)
    resp = requests.get(URL.format(date=date), headers=HEADERS, timeout=40)
    resp.raise_for_status()
    pd.read_csv(StringIO(resp.text))  # refuse to save an error page
    raw_path(season).write_text(resp.text, encoding="utf-8")
    return date, len(resp.content)


_MIRROR = None
LAST_OWN_SNAPSHOT = "2025-06-01"  # the mirror's README: ClubElo's own data to here, continuation after


def mirror_date(season):
    """1 June of the season's starting year (§250): after the play-offs, ClubElo's own to 2025."""
    return f"{season}-06-01"


def load_season_elo(season):
    """{match_results team name: Elo} for every English club in the snapshot."""
    global _MIRROR
    if SOURCE == "mirror":
        if _MIRROR is None:
            _MIRROR = pd.read_csv(MIRROR_FILE, encoding="utf-8-sig")
            _MIRROR = _MIRROR[_MIRROR["country"] == "ENG"]
        df = _MIRROR[_MIRROR["date"] == mirror_date(season)].rename(columns={"club": "Club", "elo": "Elo"})
        if df.empty:
            sys.exit(f"No mirror snapshot on {mirror_date(season)}")
        # The continuation (after 2025-06-01) can hold two aliases of one club, e.g. a frozen
        # "Nottm Forest" stub beside "Nott'm Forest". Keep the alias in ClubElo's last own snapshot (§250).
        last_own = set(_MIRROR.loc[_MIRROR["date"] == LAST_OWN_SNAPSHOT, "club"])
        df = df.assign(_name=[CLUBELO_TO_FBREF_TEAM_NAME.get(c, c) for c in df["Club"]],
                       _own=df["Club"].isin(last_own))
        df = df.sort_values("_own", ascending=False).drop_duplicates("_name")
        names = list(df["_name"])
        if len(set(names)) != len(names):
            sys.exit(f"{season}: two mirror clubs map to one name")
        return {n: float(e) for n, e in zip(names, df["Elo"])}
    df = pd.read_csv(raw_path(season), encoding="utf-8-sig")
    df = df[df["Country"] == "ENG"]
    return {CLUBELO_TO_FBREF_TEAM_NAME.get(c, c): float(e) for c, e in zip(df["Club"], df["Elo"])}


def main():
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1) if SOURCE == "api" else ():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        if raw_path(season).exists():
            print(f"{season}: present, skipped")
            continue
        date, n = download(season)
        print(f"{season}: {date} downloaded {n:,} bytes -> {raw_path(season).name}")
        time.sleep(THROTTLE_S)

    mr = pd.read_csv(MATCH_RESULTS_FILE, encoding="utf-8-sig")
    bad = False
    for season, g in mr.groupby("Season"):
        if not (FIRST_SEASON <= season <= CURRENT_SEASON):
            continue
        elo = load_season_elo(season)
        missing = sorted((set(g["Home"]) | set(g["Away"])) - set(elo))
        print(f"{season}: {len(elo)} English clubs rated | missing PL teams: {missing or 'none'}")
        bad |= bool(missing)
    if bad:
        sys.exit("Add the missing names to CLUBELO_TO_FBREF_TEAM_NAME and re-run.")


if __name__ == "__main__":
    main()
