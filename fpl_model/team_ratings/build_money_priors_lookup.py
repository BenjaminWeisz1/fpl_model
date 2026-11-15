"""
One-time extraction: pulls (Season, Team) -> (Money_GF_per_match, Money_GA_per_match)
directly out of the existing all_matchlogs_with_money_pred.csv, preserving the
exact values already in use. This becomes the permanent lookup table the new
team-ratings script reads from -- eliminating the need for a separate
all_matchlogs_with_money.csv intermediate file. Future seasons get NEW ROWS
APPENDED to this same file (e.g. from money_ratings_2027.csv once that's
built), rather than a new standalone money_ratings_YYYY.csv file each year.

Run locally (once):
    python build_money_priors_lookup.py

DO NOT RERUN (CHANGELOG §200-§202): the source's 2017-2024 values were fitted in-sample; the lookup's 2017-2024
rows are now past-only (rebuild_money_priors_past_only.py), and rerunning this would bring the leak back.
The source folder (Archive/) no longer exists anyway.
"""
from fpl_model import paths
import pandas as pd
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[3]
_DATA = paths.CURRENT

SOURCE_FILE = _PROJECT / "Archive" / "all_matchlogs_with_money_pred.csv"
OUT_FILE = paths.CURRENT / "money_priors_by_season.csv"


def main():
    df = pd.read_csv(SOURCE_FILE)

    home_side = df[["Season", "Home", "Home_Money_GF", "Home_Money_GA"]].rename(
        columns={"Home": "Team", "Home_Money_GF": "Money_GF", "Home_Money_GA": "Money_GA"})
    away_side = df[["Season", "Away", "Away_Money_GF", "Away_Money_GA"]].rename(
        columns={"Away": "Team", "Away_Money_GF": "Money_GF", "Away_Money_GA": "Money_GA"})

    combined = pd.concat([home_side, away_side], ignore_index=True)
    # one row per (Season, Team) -- values should already be identical across
    # all of a team's matches within a season, since they're a season-level
    # prior, not match-specific; drop_duplicates confirms rather than assumes
    lookup = combined.drop_duplicates(subset=["Season", "Team"])

    dupe_check = combined.groupby(["Season", "Team"])[["Money_GF", "Money_GA"]].nunique()
    inconsistent = dupe_check[(dupe_check["Money_GF"] > 1) | (dupe_check["Money_GA"] > 1)]
    if len(inconsistent) > 0:
        print(f"WARNING: {len(inconsistent)} (Season, Team) pairs have inconsistent "
              f"Money_GF/GA across different matches -- investigate before trusting this lookup:")
        print(inconsistent)

    lookup = lookup.sort_values(["Season", "Team"]).reset_index(drop=True)
    lookup.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(lookup)} (Season, Team) rows to {OUT_FILE}")
    print(lookup.head(10))


if __name__ == "__main__":
    main()