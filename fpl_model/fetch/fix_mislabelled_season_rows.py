"""
One-off maintenance: remove the current-season rows that `player_match_stats.csv`
carries under `Season = CURRENT_SEASON - 1`.

The bug. The FPL-era block of `player_match_stats.csv` keys rows two different
ways. Rows that belong to a completed season sit under a per-GAMEWEEK MatchID
(38 ids, ~300 rows each), while `fetch_gameweek_stats.py` writes one
sequentially-numbered MatchID PER ROW. When `backfill_season_column.py` added
the `Season` column it stamped everything then in the file with the previous
season, including the 341 per-row entries that had already been fetched for the
current season (MatchID 3079-3419, immediately below the 3420 boundary where the
Season column flips to 2026).

Those rows then became invisible to the fetcher's dedup key
`(Season, Player, Opponent, Venue)` -- it looked for `(2026, ...)` and found
`(2025, ...)` -- so the next fetch appended the same matches again under the
correct season. Result: 337 of the 341 are exact duplicates of a Season-2026
row (336 stat-for-stat identical, the 337th differing only in a late stat
correction), and every affected player's history double-counts those matches.

Why it matters beyond the duplication: the stray rows also (a) land in the
Season-2025 half, so they pollute the defcon model's Split A validation set
with current-season matches, (b) sort AFTER the whole real previous season
because their ids are higher than every gameweek id, which corrupts the
last-5 form window, and (c) sit before their player's transfer-prior arrival
season, so prior blending skips them.

What this script does, idempotently:
  * find FPL_API rows in the previous season whose MatchID holds a single row
    (the per-row id signature -- gameweek ids hold hundreds);
  * DELETE the ones that already exist under the current season for the same
    (Player, Opponent, Venue);
  * RELABEL the handful with no current-season counterpart to the current
    season, since they are real matches that were simply never re-fetched.

A backup of the file is written to data/archive/ first. Safe to re-run: once
clean it finds nothing. After this runs, the fetcher's dedup key matches the
rows again, so the duplication cannot recur.

    python scripts/fetch/fix_mislabelled_season_rows.py [--dry-run]
"""
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / "data" / "current"
_ARCHIVE = _ROOT / "data" / "archive"

CURRENT_SEASON = 2026  # bump manually each new season
STATS_FILE = _DATA / "player_match_stats.csv"

# A MatchID holding at most this many rows is a per-row id, not a gameweek id.
# Real gameweek ids in the completed-season block hold ~300 rows each.
SINGLETON_MAX_ROWS = 50


def main() -> int:
    dry_run = "--dry-run" in sys.argv
    prev_season = CURRENT_SEASON - 1

    df = pd.read_csv(STATS_FILE, encoding="utf-8-sig", low_memory=False)
    print(f"{STATS_FILE.name}: {len(df)} rows")

    fpl_prev = df[(df["Data_Source"] == "FPL_API") & (df["Season"] == prev_season)]
    if fpl_prev.empty:
        print(f"No FPL_API rows in season {prev_season} -- nothing to do.")
        return 0

    sizes = fpl_prev.groupby("MatchID").size()
    singleton_ids = set(sizes[sizes <= SINGLETON_MAX_ROWS].index)
    gameweek_ids = set(sizes[sizes > SINGLETON_MAX_ROWS].index)
    stray = fpl_prev[fpl_prev["MatchID"].isin(singleton_ids)]
    print(f"Season {prev_season} FPL block: {len(gameweek_ids)} gameweek ids "
          f"({fpl_prev['MatchID'].isin(gameweek_ids).sum()} rows), "
          f"{len(singleton_ids)} per-row ids ({len(stray)} rows)")
    if stray.empty:
        print("No stray per-row entries -- nothing to do.")
        return 0

    current_keys = set(zip(*(df[df["Season"] == CURRENT_SEASON][c]
                             for c in ("Player", "Opponent", "Venue"))))
    is_dupe = pd.Series(
        [(r.Player, r.Opponent, r.Venue) in current_keys for r in stray.itertuples()],
        index=stray.index)

    to_delete = stray.index[is_dupe]
    to_relabel = stray.index[~is_dupe]
    print(f"  duplicates of a Season-{CURRENT_SEASON} row -> DELETE : {len(to_delete)}")
    print(f"  no counterpart                             -> RELABEL: {len(to_relabel)}")
    if len(to_relabel):
        print(df.loc[to_relabel, ["MatchID", "Player", "Team", "Opponent",
                                  "Venue", "Minutes"]].to_string(index=False))

    affected = df.loc[to_delete, "Player"].nunique()
    print(f"  players whose history stops double-counting matches: {affected}")

    if dry_run:
        print("\n--dry-run: no changes written.")
        return 0

    _ARCHIVE.mkdir(parents=True, exist_ok=True)
    backup = _ARCHIVE / f"player_match_stats_pre_season_fix_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy2(STATS_FILE, backup)
    print(f"\nBackup written: {backup.relative_to(_ROOT)}")

    df.loc[to_relabel, "Season"] = CURRENT_SEASON
    if "Season.1" in df.columns:
        df.loc[to_relabel, "Season.1"] = CURRENT_SEASON
    out = df.drop(index=to_delete)
    out.to_csv(STATS_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows ({len(df) - len(out)} removed) to {STATS_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
