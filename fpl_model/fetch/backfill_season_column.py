"""
One-time backfill: adds an explicit Season column to player_match_stats.csv.

Necessary because the old implicit derivation (Season = 2017 + MatchID // 380)
only worked because fbref-era MatchIDs followed a fixed round-based offset --
that assumption breaks once new, live-fetched rows use a simple running
counter instead (needed because FPL's own gameweek numbering resets to 1
every season, so round-based offsets can't stay unique across seasons).

Run this ONCE, before running fetch_gameweek_stats.py for the first time.

Run locally:
    python backfill_season_column.py
"""
import pandas as pd
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
FILE = _DATA / "player_match_stats.csv"

# MatchID cutover point where the old fbref round-based scheme ends and the
# 2025/26 FPL-API offset scheme begins -- matches the MATCHID_OFFSET originally
# used when player_match_stats.csv was first built.
FBREF_MATCHID_MAX = 3039  # 8 seasons (2017-2024) x 380 matches - 1


def main():
    df = pd.read_csv(FILE, encoding="utf-8-sig")
    if "Season" in df.columns:
        print("Season column already exists -- nothing to do.")
        return

    is_fbref = df["MatchID"] <= FBREF_MATCHID_MAX
    df["Season"] = None
    df.loc[is_fbref, "Season"] = 2017 + (df.loc[is_fbref, "MatchID"] // 380)
    # everything else (the original 2025/26 FPL-API rows) is a single season
    df.loc[~is_fbref, "Season"] = 2025

    print("Season value counts:")
    print(df["Season"].value_counts().sort_index())

    df.to_csv(FILE, index=False, encoding="utf-8-sig")
    print(f"\nUpdated {FILE} with Season column.")


if __name__ == "__main__":
    main()