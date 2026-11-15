"""
Refreshes the current season's money priors when historical_money.csv's
Wages/Squad_Value have changed (e.g. after transfer-window activity) --
re-runs the ridge regression and REPLACES (not appends) that season's rows
in money_priors_by_season.csv, so update_team_ratings.py picks up the change
on its next full re-run.

Run manually whenever historical_money.csv has been updated -- not part of
the routine weekly cycle, since squad value/wages don't change every
gameweek the way match results do.
"""
import subprocess
import sys

import pandas as pd

from fpl_model import paths

_DATA = paths.CURRENT
_COMPUTE_SCRIPT = paths.ARCHIVE / "compute_money_ratings.py"
CURRENT_SEASON = 2026

def main():
    if not _COMPUTE_SCRIPT.exists():
        raise FileNotFoundError(f"Money-rating producer not found: {_COMPUTE_SCRIPT}")

    # The recovered producer uses relative paths. Run it in data/current so
    # it reads the live historical_money.csv and writes its output there.
    subprocess.run(
        [sys.executable, str(_COMPUTE_SCRIPT)],
        cwd=_DATA,
        check=True,
    )

    fresh = pd.read_csv(_DATA / "money_ratings_2026.csv")
    required = {"Team", "Money_Attack_Rating", "Money_Defense_Rating"}
    missing = required - set(fresh.columns)
    if missing:
        raise ValueError(f"Generated money ratings missing columns: {sorted(missing)}")
    if fresh["Team"].isna().any() or fresh["Team"].duplicated().any():
        raise ValueError("Generated money ratings must contain one non-null row per team")
    if fresh["Money_Attack_Rating"].isna().any() or fresh["Money_Defense_Rating"].isna().any():
        raise ValueError("Generated money ratings contain null rating values")

    fresh_rows = fresh.rename(columns={
        "Money_Attack_Rating": "Money_GF", "Money_Defense_Rating": "Money_GA"})
    fresh_rows["Season"] = CURRENT_SEASON

    lookup = pd.read_csv(_DATA / "money_priors_by_season.csv")
    lookup["Season"] = pd.to_numeric(lookup["Season"], errors="raise").astype(int)
    lookup = lookup[lookup["Season"] != CURRENT_SEASON]  # drop stale season rows
    lookup = pd.concat([lookup, fresh_rows[["Season", "Team", "Money_GF", "Money_GA"]]],
                        ignore_index=True)
    lookup.to_csv(_DATA / "money_priors_by_season.csv", index=False)
    print(f"Refreshed {len(fresh_rows)} rows for Season {CURRENT_SEASON}. "
          f"Re-run update_team_ratings.py to propagate.")

if __name__ == "__main__":
    main()
