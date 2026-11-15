"""
Team ratings update -- consolidated version. Reads match_results.csv (raw
results only) and money_priors_by_season.csv (the standalone lookup table,
built once via build_money_priors_lookup.py) directly, merging money priors
in-memory rather than requiring a separately-saved all_matchlogs_with_money.csv
intermediate file. Outputs team_ratings_history.csv (replaces
all_matchlogs_with_money_pred.csv) and team_ratings_current.csv (replaces
exponential_smoothing_ratings_2026.csv -- renamed since "2026" doesn't
correspond to a Season value and has caused genuine confusion before).

To add a new season's money priors: append rows to money_priors_by_season.csv
(Season, Team, Money_GF, Money_GA) rather than creating a new standalone file.

Run locally:
    python update_team_ratings.py
"""
from fpl_model import paths
import pandas as pd
from pathlib import Path

_DATA = paths.CURRENT

alpha = 0.045
beta = 0.8
home_advantage = 1.084340351
money_weight = 0.3

MATCH_RESULTS_FILE = paths.CURRENT / "match_results.csv"
MONEY_PRIORS_FILE = paths.CURRENT / "money_priors_by_season.csv"
OUT_FILE = paths.CURRENT / "team_ratings_history.csv"
CURRENT_RATINGS_FILE = paths.CURRENT / "team_ratings_current.csv"


def main():
    df = pd.read_csv(MATCH_RESULTS_FILE)
    money_priors = pd.read_csv(MONEY_PRIORS_FILE)

    ave_goals = pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean()
    print(f"Dynamically computed ave_goals: {ave_goals:.5f}")

    prior_map = {
        (row["Season"], row["Team"]): (row["Money_GF"], row["Money_GA"])
        for _, row in money_priors.iterrows()
    }

    ratings = pd.DataFrame(columns=["Team", "Attack_Rating", "Defense_Rating"])
    current_season = None
    total_error = 0.0
    error_count = 0
    home_preds, away_preds = [], []
    home_attack_rating, home_defense_rating, away_attack_rating, away_defense_rating = [], [], [], []

    for _, row in df.iterrows():
        season = row["Season"]
        if season != current_season:
            current_season = season
            season_teams = set(df[df["Season"] == season]["Home"]) | set(df[df["Season"] == season]["Away"])
            ratings = ratings[ratings["Team"].isin(season_teams)].reset_index(drop=True)
            for i in ratings.index:
                team = ratings.at[i, "Team"]
                if (season, team) in prior_map:
                    money_atk, money_def = prior_map[(season, team)]
                    ratings.at[i, "Attack_Rating"] = (
                        (1 - money_weight) * ratings.at[i, "Attack_Rating"] + money_weight * money_atk
                    )
                    ratings.at[i, "Defense_Rating"] = (
                        (1 - money_weight) * ratings.at[i, "Defense_Rating"] + money_weight * money_def
                    )
        for team in (row["Home"], row["Away"]):
            if team not in ratings["Team"].values:
                money_atk, money_def = prior_map[(season, team)]
                ratings.loc[len(ratings)] = [team, money_atk, money_def]

        home = row["Home"]
        away = row["Away"]
        h_atk = ratings.loc[ratings["Team"] == home, "Attack_Rating"].iloc[0]
        a_atk = ratings.loc[ratings["Team"] == away, "Attack_Rating"].iloc[0]
        h_def = ratings.loc[ratings["Team"] == home, "Defense_Rating"].iloc[0]
        a_def = ratings.loc[ratings["Team"] == away, "Defense_Rating"].iloc[0]

        pred_home_goals = h_atk * (a_def / ave_goals) * home_advantage
        pred_away_goals = a_atk * (h_def / ave_goals) * (2 - home_advantage)

        home_preds.append(pred_home_goals)
        away_preds.append(pred_away_goals)
        home_attack_rating.append(h_atk)
        home_defense_rating.append(h_def)
        away_attack_rating.append(a_atk)
        away_defense_rating.append(a_def)

        total_error += abs(pred_home_goals - row["Home_NPG"])
        total_error += abs(pred_away_goals - row["Away_NPG"])
        error_count += 2

        h_npxg, a_npxg = row["Home_NPxG_Adj"], row["Away_NPxG_Adj"]
        h_npg, a_npg = row["Home_NPG"], row["Away_NPG"]
        eps = 1e-6

        h_atk *= (1 - alpha + alpha * ((h_npxg * beta + h_npg * (1 - beta)) / max(pred_home_goals, eps)))
        a_atk *= (1 - alpha + alpha * ((a_npxg * beta + a_npg * (1 - beta)) / max(pred_away_goals, eps)))
        h_def *= (1 - alpha + alpha * ((a_npxg * beta + a_npg * (1 - beta)) / max(pred_away_goals, eps)))
        a_def *= (1 - alpha + alpha * ((h_npxg * beta + h_npg * (1 - beta)) / max(pred_home_goals, eps)))

        ratings.loc[ratings["Team"] == home, "Attack_Rating"] = h_atk
        ratings.loc[ratings["Team"] == home, "Defense_Rating"] = h_def
        ratings.loc[ratings["Team"] == away, "Attack_Rating"] = a_atk
        ratings.loc[ratings["Team"] == away, "Defense_Rating"] = a_def

    df["Home_Prediction"] = home_preds
    df["Away_Prediction"] = away_preds
    df["Home_Attack_Rating"] = home_attack_rating
    df["Home_Defense_Rating"] = home_defense_rating
    df["Away_Attack_Rating"] = away_attack_rating
    df["Away_Defense_Rating"] = away_defense_rating
    df.to_csv(OUT_FILE, index=False)

    ratings["Rating"] = ratings["Attack_Rating"] - ratings["Defense_Rating"]
    ratings = ratings.sort_values("Rating", ascending=False).reset_index(drop=True)
    ratings.to_csv(CURRENT_RATINGS_FILE, index=False)
    print(ratings[["Team", "Attack_Rating", "Defense_Rating", "Rating"]])
    print(f"\nMean absolute error per team match: {total_error / error_count:.4f}")
    print(f"\nSaved match-level predictions to: {OUT_FILE}")
    print(f"Saved current team ratings to: {CURRENT_RATINGS_FILE}")


if __name__ == "__main__":
    main()