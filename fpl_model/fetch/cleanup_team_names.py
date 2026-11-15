"""
One-time cleanup: applies the FPL->fbref team name correction to
players_current.csv, match_results.csv, and player_match_stats.csv --
needed because fetch_bootstrap.py, fetch_gameweek_stats.py, and
fetch_fixtures_results.py were run once BEFORE the team_name_map.py fix
existed, and will have written FPL-style names ("Spurs", "Man Utd",
"Newcastle", "Nott'm Forest", "Leeds") into these files' TEAM columns for
any team-name field, not just the ones observed in match_results.csv.

Safe to run multiple times -- only ever renames FPL-style names TO fbref
names, never the reverse, so re-running after the fix is already applied
does nothing.

Run this ONCE, immediately, before running any fetch script again or
proceeding with the rest of the pipeline.

Run locally:
    python cleanup_team_names.py
"""
from fpl_model import paths
import pandas as pd
from pathlib import Path
from fpl_model.fetch.team_name_map import to_fbref_name

_DATA = paths.CURRENT

FILES_AND_TEAM_COLS = {
    "players_current.csv": ["Team"],
    "match_results.csv": ["Home", "Away"],
    "player_match_stats.csv": ["Team", "Opponent"],
}


def main():
    for filename, cols in FILES_AND_TEAM_COLS.items():
        path = _DATA / filename
        df = pd.read_csv(path, encoding="utf-8-sig")
        total_changed = 0
        for col in cols:
            before = df[col].copy()
            df[col] = df[col].apply(to_fbref_name)
            n_changed = (df[col] != before).sum()
            total_changed += n_changed
            if n_changed > 0:
                changed_values = sorted(set(before[df[col] != before]))
                print(f"{filename} [{col}]: {n_changed} rows corrected. Renamed: {changed_values}")
        if total_changed == 0:
            print(f"{filename}: no FPL-style names found, nothing to fix.")
        df.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()