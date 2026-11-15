"""
Must run BEFORE fetch_fixtures_results.py
"""
import sys
import requests
import pandas as pd
from pathlib import Path
from team_name_map import to_fbref_name
import penalty_reconcile

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
LIVE_URL = "https://fantasy.premierleague.com/api/event/{event_id}/live/"

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
PLAYER_STATS_FILE = _DATA / "player_match_stats.csv"
PENALTY_LOG_FILE = _DATA / "penalty_log.csv"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from league_rates import current_season  # noqa: E402
from atomic_write import to_csv_atomic  # noqa: E402

CURRENT_SEASON = current_season()


def fetch_json(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main():
    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)

    team_lookup = {t["id"]: to_fbref_name(t["name"]) for t in bootstrap["teams"]}
    player_lookup = {p["id"]: {
        "Player": f'{p["first_name"]} {p["second_name"]}'.strip(),
        "TeamID": p["team"],
    } for p in bootstrap["elements"]}

    existing = pd.read_csv(PLAYER_STATS_FILE, encoding="utf-8-sig")
    if "Season" not in existing.columns:
        raise RuntimeError(
            "player_match_stats.csv has no Season column -- run backfill_season_column.py "
            "once before running this script for the first time."
        )
    existing_fpl = existing[existing["Data_Source"] == "FPL_API"]
    # A fixture is fetched once, for both teams together. Only then can bootstrap's team be trusted:
    # it is today's club, so a past gameweek would give a player who has moved a false row (§183).
    fetched = set(zip(
        existing_fpl["Season"], existing_fpl["Team"], existing_fpl["Opponent"], existing_fpl["Venue"]))

    penalty_log = pd.read_csv(PENALTY_LOG_FILE, encoding="utf-8-sig")
    pk_lookup = {}
    pkatt_lookup = {}
    for _, row in penalty_log.iterrows():
        key = (row["Season"], row["Player"], row["Opponent"], row["HomeAway"])
        pkatt_lookup[key] = pkatt_lookup.get(key, 0) + 1
        if str(row["Scored"]).strip().upper() == "TRUE":
            pk_lookup[key] = pk_lookup.get(key, 0) + 1

    gws_to_fetch = sorted(set(f["event"] for f in fixtures if f["finished_provisional"] and f["event"] is not None))
    print(f"Finished gameweeks with fixtures: {gws_to_fetch}")

    next_match_id = existing["MatchID"].max() + 1
    new_rows = []
    fixtures_by_gw = {}
    for f in fixtures:
        if f["finished_provisional"]:
            fixtures_by_gw.setdefault(f["event"], []).append(f)

    for gw in gws_to_fetch:
        # team_id -> (opponent_team_id, venue) for this gameweek's finished fixtures not yet fetched
        team_fixture_map = {}
        fixture_by_id = {}
        for f in fixtures_by_gw[gw]:
            home_id, away_id = f["team_h"], f["team_a"]
            fixture_by_id[f["id"]] = {home_id, away_id}
            if (CURRENT_SEASON, team_lookup[home_id], team_lookup[away_id], "Home") not in fetched:
                team_fixture_map[home_id] = (away_id, "Home")
            if (CURRENT_SEASON, team_lookup[away_id], team_lookup[home_id], "Away") not in fetched:
                team_fixture_map[away_id] = (home_id, "Away")
        if not team_fixture_map:
            continue

        live_data = fetch_json(LIVE_URL.format(event_id=gw))

        for el in live_data["elements"]:
            player_id = el["id"]
            stats = el["stats"]
            if stats["minutes"] <= 0:
                continue
            if player_id not in player_lookup:
                continue
            info = player_lookup[player_id]
            team_id = info["TeamID"]
            if team_id not in team_fixture_map:
                continue  # this player's team didn't feature in a finished fixture this GW
            opp_id, venue = team_fixture_map[team_id]

            player_name = info["Player"]
            played_in = [fixture_by_id[e["fixture"]] for e in el.get("explain", []) if e["fixture"] in fixture_by_id]
            if played_in and not any(team_id in teams for teams in played_in):
                # moved between the match and this fetch: bootstrap has no record of his club then
                print(f"WARNING: {player_name} played GW{gw} for another club and has moved since; "
                      f"row not added, add it by hand")
                continue
            team_name = team_lookup[team_id]
            opp_name = team_lookup[opp_id]

            key = (CURRENT_SEASON, player_name, opp_name, venue)
            pkatt = pkatt_lookup.get(key, 0)
            pk = pk_lookup.get(key, 0)
            npg = stats["goals_scored"] - pk

            new_rows.append({
                "MatchID": next_match_id,
                "Player": player_name,
                "Team": team_name,
                "Opponent": opp_name,
                "Venue": venue,
                "Position": pd.NA,
                "Minutes": stats["minutes"],
                "PK": pk,
                "PKatt": pkatt,
                "CrdY": stats["yellow_cards"],
                "CrdR": stats["red_cards"],
                # FPL xG includes penalties; reconcile_player_stats derives npxG from it
                "npxG": stats.get("expected_goals", 0.0),
                "xG": stats.get("expected_goals", 0.0),
                "xAG": stats.get("expected_assists", 0.0),
                "Goals": stats["goals_scored"],
                "Assists": stats["assists"],
                "Tkl": stats.get("tackles"),
                "Recov": stats.get("recoveries"),
                "CBI": stats.get("clearances_blocks_interceptions"),
                "Defensive_Contribution": stats.get("defensive_contribution"),
                "Clean_Sheet": stats["clean_sheets"],
                "Goals_Conceded": stats["goals_conceded"],
                "Own_Goals": stats["own_goals"],
                "Saves": stats["saves"],
                "Bonus": stats["bonus"],
                "BPS": stats["bps"],
                "Penalties_Missed": stats["penalties_missed"],
                "Penalties_Saved": stats["penalties_saved"],
                "NPG": npg,
                "Data_Source": "FPL_API",
                "Season": CURRENT_SEASON,
            })
            next_match_id += 1

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True, sort=False) if new_rows else existing

    # Re-apply penalty_log.csv to every FPL-era row, so penalties logged or corrected after
    # their match was fetched still reach PK/PKatt/NPG (CHANGELOG §73).
    combined, report = penalty_reconcile.reconcile_player_stats(combined, penalty_reconcile.load_log())
    penalty_reconcile.warn(report, "player_match_stats")

    if not new_rows and not report["rows_changed"]:
        print("No new player-match rows to add (nothing newly finished, or already fetched).")
        return

    to_csv_atomic(combined, PLAYER_STATS_FILE, index=False, encoding="utf-8-sig")
    print(f"Added {len(new_df)} new player-match rows to {PLAYER_STATS_FILE}; "
          f"penalty_log reconciliation changed {report['rows_changed']} rows")
    if new_rows:
        print(new_df[["Player", "Team", "Opponent", "Venue", "Goals", "PK", "NPG"]].head(10))


if __name__ == "__main__":
    main()