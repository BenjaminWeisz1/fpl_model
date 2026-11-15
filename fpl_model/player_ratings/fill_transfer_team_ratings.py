"""
Fills in Team_Attack_Rating for new rows in transfers.csv:
  - Season == 2026 rows: uses team_ratings_preseason_2026.csv (the blended
    pre-season rating)
  - Season == 2025 rows (mid-season debuts): uses the team's rating from
    team_ratings_current.csv -- this is the END of 2025/26,
    an approximation for "their team's rating at the point they debuted"
    since per-round historical snapshots aren't readily available. Flag
    this with the user if precision matters more than this.

transfers.csv doesn't store Team directly -- it's looked up from
players_current.csv by Player name first.

Only fills BLANK Team_Attack_Rating cells -- leaves any already-populated
values untouched, so this is safe to re-run as you add more players.

Run locally:
    python fill_transfer_team_ratings.py
"""
import pandas as pd
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"

TRANSFERS_FILE = _DATA / "transfers.csv"
TRANSFERS_PLAYER_COL = "Player"
TRANSFERS_SEASON_COL = "Season"

CURRENT_PLAYERS_FILE = _DATA / "players_current.csv"
CURRENT_PLAYERS_NAME_COL = "Player"
CURRENT_PLAYERS_TEAM_COL = "Team"

PRESEASON_FILE = _DATA / "team_ratings_preseason_2026.csv"
PRESEASON_TEAM_COL = "Team"
PRESEASON_ATTACK_COL = "Attack_Rating"

CURRENT_FILE = _DATA / "team_ratings_current.csv"
CURRENT_TEAM_COL = "Team"
CURRENT_ATTACK_COL = "Attack_Rating"


def main():
    transfers = pd.read_csv(TRANSFERS_FILE, encoding="utf-8-sig")
    players_current = pd.read_csv(CURRENT_PLAYERS_FILE, encoding="utf-8-sig")
    preseason = pd.read_csv(PRESEASON_FILE)
    current = pd.read_csv(CURRENT_FILE)

    player_team_lookup = dict(zip(players_current[CURRENT_PLAYERS_NAME_COL],
                                   players_current[CURRENT_PLAYERS_TEAM_COL]))
    preseason_lookup = dict(zip(preseason[PRESEASON_TEAM_COL], preseason[PRESEASON_ATTACK_COL]))
    current_lookup = dict(zip(current[CURRENT_TEAM_COL], current[CURRENT_ATTACK_COL]))

    to_fill = transfers["Team_Attack_Rating"].isna()
    print(f"{to_fill.sum()} rows need Team_Attack_Rating filled in")

    filled_2026, filled_2025, missing = 0, 0, []

    for idx in transfers[to_fill].index:
        season = transfers.at[idx, TRANSFERS_SEASON_COL]
        player = transfers.at[idx, TRANSFERS_PLAYER_COL]

        team = player_team_lookup.get(player)
        if team is None:
            missing.append((player, None, season, "player not found in players_current.csv"))
            continue

        if season == 2026:
            if team in preseason_lookup:
                transfers.at[idx, "Team_Attack_Rating"] = preseason_lookup[team]
                filled_2026 += 1
            else:
                missing.append((player, team, season, "team not in preseason file"))
        elif season == 2025:
            if team in current_lookup:
                transfers.at[idx, "Team_Attack_Rating"] = current_lookup[team]
                filled_2025 += 1
            else:
                missing.append((player, team, season, "team not in current-ratings file"))
        else:
            missing.append((player, team, season, f"unexpected season {season}"))

    print(f"Filled {filled_2026} rows from preseason 2026 ratings")
    print(f"Filled {filled_2025} rows from end-of-2025/26 ratings (approximation, see note above)")
    if missing:
        print(f"\n{len(missing)} rows could NOT be filled -- check player/team-name spelling:")
        for player, team, season, reason in missing:
            print(f"  {player} ({team}, season {season}): {reason}")

    transfers.to_csv(TRANSFERS_FILE, index=False, encoding="utf-8-sig")
    print(f"\nUpdated {TRANSFERS_FILE} in place.")


if __name__ == "__main__":
    main()