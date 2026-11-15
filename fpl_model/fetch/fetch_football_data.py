"""
One-off backfill of football-data.co.uk Premier League match CSVs (E0), the
source for AGENDA §2's promoted-team odds priors: pre-match 1X2 and
over/under 2.5 odds per fixture, 2017/18 onward.

Each season's CSV is saved verbatim to data/raw/football_data/E0_<Season>.csv
(Season = start year, matching match_results.csv) and never rewritten once
present -- delete a file to force a re-download. Requests are throttled.

After downloading, every (Season, HomeTeam, AwayTeam) is mapped to
match_results.csv team names via FD_TO_FBREF_TEAM_NAME and checked: every
football-data fixture must map onto exactly one match_results.csv row, or the
script exits non-zero naming the unmapped teams.

Run locally:
    python -m fpl_model.fetch.fetch_football_data
"""
import sys
import time

import pandas as pd
import requests

from fpl_model import paths

_ROOT = paths.ROOT
_DATA = paths.CURRENT
RAW_DIR = paths.RAW / "football_data"
MATCH_RESULTS_FILE = paths.CURRENT / "match_results.csv"

from fpl_model.league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()
FIRST_SEASON = 2017
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
THROTTLE_S = 3

# football-data.co.uk name -> match_results.csv (fbref-style) name.
# Names not listed are assumed identical; the coverage check catches misses.
FD_TO_FBREF_TEAM_NAME = {
    "Man United": "Manchester Utd",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle Utd",
    "Nott'm Forest": "Nott'ham Forest",
    "Leeds": "Leeds United",
    "Sheffield United": "Sheffield Utd",
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


def season_code(season):
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def raw_path(season):
    return RAW_DIR / f"E0_{season}.csv"


def download(season):
    resp = requests.get(URL.format(code=season_code(season)), timeout=30)
    resp.raise_for_status()
    raw_path(season).write_bytes(resp.content)
    return len(resp.content)


def load_season(season):
    # Newer files are UTF-8 with a BOM, older ones latin-1; some carry
    # trailing empty columns/rows.
    try:
        df = pd.read_csv(raw_path(season), encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(raw_path(season), encoding="latin-1")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(subset=["HomeTeam", "AwayTeam"])
    df.insert(0, "Season", season)
    for side in ("HomeTeam", "AwayTeam"):
        df[side] = df[side].str.strip().map(lambda t: FD_TO_FBREF_TEAM_NAME.get(t, t))
    return df


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        if raw_path(season).exists():
            print(f"{season}: present, skipped")
            continue
        n = download(season)
        print(f"{season}: downloaded {n:,} bytes -> {raw_path(season).name}")
        time.sleep(THROTTLE_S)

    # Coverage check against match_results.csv team names.
    mr = pd.read_csv(MATCH_RESULTS_FILE, encoding="utf-8-sig")
    mr_keys = set(zip(mr["Season"], mr["Home"], mr["Away"]))
    mr_teams = {s: set(g["Home"]) | set(g["Away"]) for s, g in mr.groupby("Season")}
    bad = False
    for season in range(FIRST_SEASON, CURRENT_SEASON + 1):
        fd = load_season(season)
        fd_teams = set(fd["HomeTeam"]) | set(fd["AwayTeam"])
        unknown = sorted(fd_teams - mr_teams.get(season, set()))
        keys = set(zip(fd["Season"], fd["HomeTeam"], fd["AwayTeam"]))
        mr_season = {k for k in mr_keys if k[0] == season}
        print(f"{season}: {len(fd)} fixtures | matched to match_results {len(keys & mr_season)}"
              f" / {len(mr_season)} played")
        if unknown:
            bad = True
            print(f"   UNMAPPED football-data team names: {unknown}")
    if bad:
        sys.exit("Add the unmapped names to FD_TO_FBREF_TEAM_NAME and re-run.")


if __name__ == "__main__":
    main()
