"""
Run locally (after fetch_gameweek_stats.py):
"""
import sys
import requests
import pandas as pd
from pathlib import Path
from team_name_map import to_fbref_name
import penalty_reconcile

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
MATCH_RESULTS_FILE = _DATA / "match_results.csv"
PLAYER_STATS_FILE = _DATA / "player_match_stats.csv"
PENALTY_LOG_FILE = _DATA / "penalty_log.csv"
FIXTURES_FILE = _DATA / "fixtures.csv"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from league_rates import current_season  # noqa: E402
from atomic_write import to_csv_atomic  # noqa: E402

CURRENT_SEASON = current_season()
PEN_XG_VALUE = 0.79


def fetch_json(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def round_mismatches(api_fixtures, team_lookup, file_rows):
    """Where fixtures.csv and FPL disagree: (Home, Away, file Round, FPL event).

    A blank Round matches a null event (a postponed match with no date yet).
    """
    rounds = {(r.Home.strip(), r.Away.strip()): r.Round for r in file_rows.itertuples(index=False)}
    bad = []
    for f in api_fixtures:
        home, away = team_lookup[f["team_h"]], team_lookup[f["team_a"]]
        key = (home, away)
        if key not in rounds:
            bad.append((home, away, "no row", f["event"]))
            continue
        file_round = None if pd.isna(rounds[key]) else int(rounds[key])
        if file_round != f["event"]:
            bad.append((home, away, file_round, f["event"]))
    return bad


def check_rounds(api_fixtures, team_lookup):
    """Stop if fixtures.csv is out of step with FPL: every forecast takes its rounds from that file (§186)."""
    file_rows = pd.read_csv(FIXTURES_FILE)
    bad = round_mismatches(api_fixtures, team_lookup, file_rows[file_rows["Season"] == CURRENT_SEASON])
    if bad:
        lines = "\n".join(f"  {h} v {a}: fixtures.csv Round {r}, FPL event {e}" for h, a, r, e in bad)
        raise SystemExit(f"fixtures.csv disagrees with FPL on {len(bad)} fixture(s):\n{lines}\n"
                         f"Set Round to FPL's event by hand, blank if FPL has none, and rerun; nothing was written.")


def main():
    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)
    team_lookup = {t["id"]: to_fbref_name(t["name"]) for t in bootstrap["teams"]}
    check_rounds(fixtures, team_lookup)

    match_results = pd.read_csv(MATCH_RESULTS_FILE)
    player_stats = pd.read_csv(PLAYER_STATS_FILE, encoding="utf-8-sig")
    penalty_log = pd.read_csv(PENALTY_LOG_FILE, encoding="utf-8-sig")

    existing_keys = set(zip(
        match_results["Season"], match_results["Home"], match_results["Away"]))

    # penalty attempts per (Season, Team, Opponent, Venue) -- for the NPxG_Adj
    # subtraction, we need attempts by TEAM, not by individual player
    penalty_log["is_scored"] = penalty_log["Scored"].astype(str).str.strip().str.upper() == "TRUE"
    pkatt_by_team = penalty_log.groupby(
        ["Season", "Team", "Opponent", "HomeAway"]).size()

    next_match_id = match_results["MatchID"].max() + 1
    new_rows = []

    finished_fixtures = [f for f in fixtures if f["finished_provisional"]]
    print(f"{len(finished_fixtures)} finished fixtures found via the API")

    for f in finished_fixtures:
        home_name = team_lookup[f["team_h"]]
        away_name = team_lookup[f["team_a"]]
        key = (CURRENT_SEASON, home_name, away_name)
        if key in existing_keys:
            continue  # already have this result

        home_goals = f["team_h_score"]
        away_goals = f["team_a_score"]

        home_pkatt = pkatt_by_team.get((CURRENT_SEASON, home_name, away_name, "Home"), 0)
        away_pkatt = pkatt_by_team.get((CURRENT_SEASON, away_name, home_name, "Away"), 0)
        home_pk = int(penalty_log[
            (penalty_log["Season"] == CURRENT_SEASON) & (penalty_log["Team"] == home_name) &
            (penalty_log["Opponent"] == away_name) & (penalty_log["HomeAway"] == "Home") &
            penalty_log["is_scored"]
        ].shape[0])
        away_pk = int(penalty_log[
            (penalty_log["Season"] == CURRENT_SEASON) & (penalty_log["Team"] == away_name) &
            (penalty_log["Opponent"] == home_name) & (penalty_log["HomeAway"] == "Away") &
            penalty_log["is_scored"]
        ].shape[0])

        home_npg = home_goals - home_pk
        away_npg = away_goals - away_pk

        # team-level summed expected_goals from the already-fetched player stats
        home_players = player_stats[
            (player_stats["Season"] == CURRENT_SEASON) & (player_stats["Team"] == home_name) &
            (player_stats["Opponent"] == away_name) & (player_stats["Venue"] == "Home")]
        away_players = player_stats[
            (player_stats["Season"] == CURRENT_SEASON) & (player_stats["Team"] == away_name) &
            (player_stats["Opponent"] == home_name) & (player_stats["Venue"] == "Away")]

        if len(home_players) == 0 or len(away_players) == 0:
            print(f"WARNING: skipping {home_name} vs {away_name} -- no player stats found yet "
                  f"(did fetch_gameweek_stats.py run first?)")
            continue

        # penalty-inclusive player xG: player npxG already excludes penalties (CHANGELOG §80)
        home_npxg_adj = home_players["xG"].sum() - PEN_XG_VALUE * home_pkatt
        away_npxg_adj = away_players["xG"].sum() - PEN_XG_VALUE * away_pkatt

        new_rows.append({
            "MatchID": next_match_id,
            "Season": CURRENT_SEASON,
            "Home": home_name,
            "Away": away_name,
            "Home_Goals": home_goals,
            "Away_Goals": away_goals,
            "Home_NPG": home_npg,
            "Away_NPG": away_npg,
            "Home_PK": home_pk,
            "Away_PK": away_pk,
            "Home_PKatt": home_pkatt,
            "Away_PKatt": away_pkatt,
            "Home_NPxG_Adj": round(home_npxg_adj, 4),
            "Away_NPxG_Adj": round(away_npxg_adj, 4),
        })
        next_match_id += 1

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([match_results, new_df], ignore_index=True, sort=False) if new_rows else match_results

    # Re-apply penalty_log.csv to every FPL-era match, so penalties logged or corrected after
    # the match was fetched still reach PK/PKatt/NPG and NPxG_Adj (CHANGELOG §73).
    combined, report = penalty_reconcile.reconcile_match_results(combined, penalty_reconcile.load_log())
    penalty_reconcile.warn(report, "match_results")

    if not new_rows and not report["rows_changed"]:
        print("No new match results to add.")
        return

    to_csv_atomic(combined, MATCH_RESULTS_FILE, index=False)
    print(f"Added {len(new_df)} new match results to {MATCH_RESULTS_FILE}; "
          f"penalty_log reconciliation changed {report['rows_changed']} rows")
    if new_rows:
        print(new_df[["Home", "Away", "Home_Goals", "Away_Goals", "Home_NPxG_Adj", "Away_NPxG_Adj"]])


if __name__ == "__main__":
    main()