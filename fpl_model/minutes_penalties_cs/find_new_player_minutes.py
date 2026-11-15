"""
One-off helper (pre-season 2026): career Premier League minutes for each name in new_players_2026.csv.

Reads the `Players` column of data/current/new_players_2026.csv, looks each name up in player_match_stats.csv and
rewrites the file as `Player, Mins` (0 for a player with no PL minutes). Not a pipeline stage. Because it replaces its
own input's column, run it on a fresh list only.

    python -m fpl_model.minutes_penalties_cs.find_new_player_minutes
"""
import pandas as pd

from fpl_model import paths

NEW_PLAYERS_FILE = paths.CURRENT / "new_players_2026.csv"


def career_minutes(names, player_match_stats: pd.DataFrame) -> pd.DataFrame:
    """Total PL minutes per name, 0 for a name without a row; sorted by name, one row per distinct name."""
    totals = player_match_stats.groupby("Player")["Minutes"].sum()
    names = sorted(set(names))
    return pd.DataFrame({"Player": names, "Mins": [totals.get(n, 0) for n in names]})


def main() -> None:
    new_players = pd.read_csv(NEW_PLAYERS_FILE)
    stats = pd.read_csv(paths.CURRENT / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    out = career_minutes(new_players["Players"], stats)
    print(len(out))
    out.to_csv(NEW_PLAYERS_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
