"""
Assembles player_ratings.csv -- the weekly one-row-per-player ratings table
(the second output of the weekly pipeline, alongside predicted_points.csv).

Columns:
    Player  -- Web_Name (same identity as predicted_points.csv)
    Pos     -- D / M / F / G
    Team    -- 3-letter team code (Team_Short from players_current.csv)
    Price   -- FPL price
    NPG     -- NPG/90 from player_goal_assist_ratings.csv
    Assists -- Assists/90 from player_goal_assist_ratings.csv
    Defcon  -- P(reach the defcon threshold in a full 90-minute match vs an
               average opponent), home and away averaged. Produced by
               scripts/defcon/predict.py as defcon_average_probabilities.csv;
               defenders then carry the points factor (CHANGELOG §159), so this
               is a real FPL hit chance; blank for goalkeepers (out of scope
               for the defcon model).

Players who have left the league are left out (scripts/player_availability.py).

Identity gotcha: player_goal_assist_ratings.csv's "Player" column holds
Web_Name values, NOT full names, while defcon_average_probabilities.csv uses
full player names -- players_current.csv bridges the two, keyed on
(Web_Name, Team) because Web_Name is not unique league-wide (shared
Web_Names exist, e.g. Wesley/Malick Fofana).

Run locally:
    python export_player_ratings.py
"""
import sys
import time

import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defcon_scaling import points_prob_at_90          # noqa: E402
from player_availability import with_availability     # noqa: E402

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"

RATINGS_FILE = _DATA / "player_goal_assist_ratings.csv"
DEFCON_FILE = _DATA / "defcon_average_probabilities.csv"
PLAYERS_CURRENT_FILE = _DATA / "players_current.csv"
MINUTES_FILE = _DATA / "minutes.csv"
OUT_FILE = _DATA / "player_ratings.csv"


def _require(path, producer):
    if not path.exists():
        sys.exit(f"ERROR: {path.name} not found in {path.parent} -- "
                 f"run {producer} first.")
    return path


def _mtime(path):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))


def main():
    ratings = pd.read_csv(
        _require(RATINGS_FILE, "scripts/player_ratings/build_player_ratings.py"),
        encoding="utf-8-sig")
    defcon = pd.read_csv(_require(DEFCON_FILE, "scripts/defcon/predict.py"))
    players = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")

    print(f"Inputs: player_goal_assist_ratings.csv (updated {_mtime(RATINGS_FILE)}), "
          f"defcon_average_probabilities.csv (updated {_mtime(DEFCON_FILE)})")

    # Bridge Web_Name (ratings identity) -> full name (defcon identity) and
    # the 3-letter team code. Web_Name is NOT unique across the league
    # (2026-09-03: Wesley/Malick Fofana, Bobby/Sorba Thomas share Web_Names
    # after the promoted-squads update), so bridge on (Web_Name, Team) --
    # both files carry the full Team string. Rows whose (Web_Name, Team)
    # pair is absent from players_current (stale ratings vs a squad update)
    # fall back to a Web_Name-only match when that name is unambiguous.
    # Price also comes from players_current: the Price column carried by
    # player_goal_assist_ratings.csv is frozen at that (manually-run)
    # stage's last run, so it goes stale as FPL prices move.
    bridge = players[["Web_Name", "Team", "Player", "Team_Short", "Price"]]
    pair_key = {(r.Web_Name, r.Team): (r.Player, r.Team_Short, r.Price)
                for r in bridge.itertuples()}
    name_counts = players["Web_Name"].value_counts()
    name_key = {r.Web_Name: (r.Player, r.Team_Short, r.Price)
                for r in bridge.itertuples() if name_counts[r.Web_Name] == 1}
    defcon_map = dict(zip(defcon["Player"], defcon["Defcon"]))

    def _resolve(web_name, team):
        hit = pair_key.get((web_name, team)) or name_key.get(web_name)
        return hit if hit else (None, None, None)

    out = ratings.rename(columns={"NPG/90": "NPG", "Assists/90": "Assists"}).copy()
    resolved = [_resolve(w, t) for w, t in zip(out["Player"], out["Team"])]
    out["_full_name"] = [r[0] for r in resolved]
    out["Defcon"] = points_prob_at_90(out["_full_name"].map(defcon_map), out["Pos"]).round(4)
    out["Team"] = [r[1] for r in resolved]
    # Unresolved players keep their stale ratings-file price rather than blank.
    out["Price"] = [r[2] if r[2] is not None else p
                    for r, p in zip(resolved, out["Price"])]

    # Non-GK players without a defcon value would mean a coverage gap
    # between players_current and the defcon artifact -- GKs are expected
    # (the defcon model only covers D/M/F).
    bad = out[out["Defcon"].isna() & (out["Pos"] != "G")]
    if len(bad) > 0:
        print(f"WARNING: {len(bad)} non-GK players have no defcon value "
              f"(missing from defcon_average_probabilities.csv):")
        print(bad[["Player", "Pos", "Team"]].head(10).to_string(index=False))
    if out["Team"].isna().any():
        print(f"WARNING: {int(out['Team'].isna().sum())} players could not be "
              f"mapped to a team code")

    # This is a list of options, so players who have left the league come off it (§168).
    # They stay in players_current.csv and in everything that reads their history.
    frame = with_availability(players, pd.read_csv(MINUTES_FILE, encoding="utf-8-sig"))
    departed = set(zip(frame.loc[frame["Departed"], "Web_Name"], frame.loc[frame["Departed"], "Team_Short"]))
    gone = [pair in departed for pair in zip(out["Player"], out["Team"])]
    print(f"Dropped {sum(gone)} departed players (status u, no expected minutes)")
    out = out[[not g for g in gone]]

    n_gk = int((out["Pos"] == "G").sum())
    out = out[["Player", "Pos", "Team", "Price", "NPG", "Assists", "Defcon"]]
    out = out.sort_values("NPG", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nWrote {len(out)} players to {OUT_FILE} "
          f"({n_gk} goalkeepers with blank Defcon -- out of scope for the defcon model)")
    print("Mean Defcon by position:")
    print(out.groupby("Pos")["Defcon"].agg(["count", "mean"]).round(4).to_string())
    print("\nTop 10 by NPG:")
    print(out.head(10).to_string(index=False))
    print("\nTop 5 defenders by Defcon:")
    print(out[out["Pos"] == "D"].nlargest(5, "Defcon").to_string(index=False))


if __name__ == "__main__":
    main()
