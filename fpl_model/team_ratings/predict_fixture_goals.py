"""
Predicts non-penalty goals for each team in upcoming fixtures, using the
same formula the exponential-smoothing script uses for past matches, applied
forward with current team ratings. This is a building block other facets
(penalty goals, clean sheets) both consume -- not a standalone deliverable.

Run locally:
    python predict_fixture_goals.py
"""
import sys
import pandas as pd
from pathlib import Path
from fpl_model.team_ratings.get_next_round import get_next_round

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"


FIXTURES_FILE = _DATA / "fixtures.csv"
TEAM_RATINGS_FILE = _DATA / "team_ratings_current.csv"  # latest current ratings
OUT_FILE = _DATA / "fixture_goal_predictions.csv"

from fpl_model.forecast_horizon import HORIZON as NEXT_N_ROUNDS  # noqa: E402  (set in one place)
from fpl_model.league_rates import current_season  # noqa: E402

from fpl_model.league_rates import mean_npg  # noqa: E402
from fpl_model.team_ratings.update_team_ratings import home_advantage  # noqa: E402

CURRENT_SEASON = current_season()
AVG_CONCEDED = mean_npg()


def main():
    fixtures = pd.read_csv(FIXTURES_FILE)
    ratings = pd.read_csv(TEAM_RATINGS_FILE)
    rating_lookup = ratings.set_index("Team")[["Attack_Rating", "Defense_Rating"]].to_dict("index")

    next_round = get_next_round(_DATA, CURRENT_SEASON)
    target_rounds = list(range(next_round, next_round + NEXT_N_ROUNDS))
    fixtures = fixtures[(fixtures["Season"] == CURRENT_SEASON) & fixtures["Round"].isin(target_rounds)].copy()
    print(f"Next round from match_results.csv: {next_round}")
    print(f"Filtered to rounds {target_rounds}: {len(fixtures)} fixtures")

    rows = []
    missing_teams = set()
    for _, row in fixtures.iterrows():
        home, away = row["Home"], row["Away"]
        if home not in rating_lookup:
            missing_teams.add(home)
            continue
        if away not in rating_lookup:
            missing_teams.add(away)
            continue
        h_atk = rating_lookup[home]["Attack_Rating"]
        h_def = rating_lookup[home]["Defense_Rating"]
        a_atk = rating_lookup[away]["Attack_Rating"]
        a_def = rating_lookup[away]["Defense_Rating"]

        pred_home_goals = h_atk * (a_def / AVG_CONCEDED) * home_advantage
        pred_away_goals = a_atk * (h_def / AVG_CONCEDED) * (2 - home_advantage)

        rows.append({
            "MatchID": row["MatchID"], "Round": row["Round"],
            "Home": home, "Away": away,
            "Home_Pred_NPG": pred_home_goals, "Away_Pred_NPG": pred_away_goals,
        })

    if missing_teams:
        print(f"WARNING: {len(missing_teams)} teams not found in {TEAM_RATINGS_FILE}, "
              f"their fixtures were skipped: {sorted(missing_teams)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} fixture predictions to {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()