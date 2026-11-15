# Data migrations

One-off repairs that were run once against the local data files and are kept as a record of how
those files reached their current state. Rerunning them is never part of the weekly pipeline.

| Script | What it repaired |
|---|---|
| `backfill_season_column.py` | Added an explicit `Season` column to `player_match_stats.csv`, replacing a derivation from MatchID that broke for live-fetched rows |
| `cleanup_team_names.py` | Applied the FPL-to-fbref team-name correction to three files written before `team_name_map.py` existed |
| `fix_mislabelled_season_rows.py` | Removed current-season rows that `player_match_stats.csv` carried under the previous season |
| `rebuild_2025_team_xg.py` | Rebuilt 2025 team xG in `match_results.csv` from summed FPL/Opta player xG, replacing Understat's |
| `merge_defcon_into_transfers.py` | Merged prior-league defensive-contribution data for new arrivals into `transfers.csv` |
