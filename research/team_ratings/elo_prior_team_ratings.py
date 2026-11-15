"""
Promoted-team priors from preseason ClubElo (AGENDA §2, CHANGELOG §33).

Same experiment design as odds_prior_team_ratings.py (§32) -- entry prior
(1 - w) * money + w * elo_implied for promoted teams only, rolling-origin w,
two tiers + early window, cluster bootstrap -- with a different signal:

  elo_implied for season s comes from two log-linear regressions fitted on
  ALL team-seasons of seasons before s (no leakage; continuing teams are a
  test bed only, never re-rated in production):
      ln(target_for)  = a + b * elo_c
      ln(target_agst) = c + d * elo_c
  elo_c = the team's preseason ClubElo minus that season's PL-team mean (so
  league-wide Elo drift cancels); target = the team's per-match season mean
  of beta * NPxG_Adj + (1 - beta) * NPG -- the model's own update target, in
  the NPG-per-match units of Attack/Defense_Rating and the money priors.

Run (after scripts/fetch/fetch_clubelo.py):
    python elo_prior_team_ratings.py
Writes data/current/team_ratings_oot/elo_prior_report.txt + _summary.json.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
from research.team_ratings.evaluate_team_ratings_oot import load_inputs, beta
from research.team_ratings.odds_prior_team_ratings import evaluate, promoted_by_season
from fpl_model.fetch.fetch_clubelo import load_season_elo


def season_targets(df):
    """Per team-season mean per-match target for / against."""
    t = beta
    home = pd.DataFrame({"Season": df["Season"], "Team": df["Home"],
                         "For": t * df["Home_NPxG_Adj"] + (1 - t) * df["Home_NPG"],
                         "Agst": t * df["Away_NPxG_Adj"] + (1 - t) * df["Away_NPG"]})
    away = pd.DataFrame({"Season": df["Season"], "Team": df["Away"],
                         "For": t * df["Away_NPxG_Adj"] + (1 - t) * df["Away_NPG"],
                         "Agst": t * df["Home_NPxG_Adj"] + (1 - t) * df["Home_NPG"]})
    return pd.concat([home, away]).groupby(["Season", "Team"], as_index=False)[["For", "Agst"]].mean()


def elo_signal(df, seasons, promoted):
    targets = season_targets(df)
    rows = []
    for s in seasons:
        elo = load_season_elo(s)
        teams = sorted(set(df.loc[df["Season"] == s, "Home"]) | set(df.loc[df["Season"] == s, "Away"]))
        mean_elo = np.mean([elo[t] for t in teams])
        rows += [{"Season": s, "Team": t, "Elo_c": elo[t] - mean_elo} for t in teams]
    table = pd.DataFrame(rows).merge(targets, on=["Season", "Team"], how="left")
    fits = {}

    def signal(s, ctx):
        train = table[table["Season"] < s]
        bf = np.polyfit(train["Elo_c"], np.log(train["For"]), 1)
        ba = np.polyfit(train["Elo_c"], np.log(train["Agst"]), 1)
        fits[s] = {"n": len(train), "for": bf.tolist(), "agst": ba.tolist()}
        cur = table[table["Season"] == s].set_index("Team")["Elo_c"]
        return {t: ((float(np.exp(np.polyval(bf, cur[t]))), float(np.exp(np.polyval(ba, cur[t])))),
                    {"elo_c": float(cur[t])})
                for t in sorted(promoted.get(s, ()))}
    return signal, table, fits


def main():
    df, prior_map, _ = load_inputs()
    seasons, promoted = promoted_by_season(df)
    signal, table, fits = elo_signal(df, seasons, promoted)

    # Signal-strength diagnostics on the full table (descriptive only; never used for selection).
    full = table.dropna()
    r_for = np.corrcoef(full["Elo_c"], np.log(full["For"]))[0, 1]
    r_agst = np.corrcoef(full["Elo_c"], np.log(full["Agst"]))[0, 1]
    header = [f"  team-seasons with Elo + target: {len(full)} | corr(Elo_c, ln target) "
              f"for {r_for:+.3f}, against {r_agst:+.3f} (descriptive, all seasons)",
              "  promoted Elo_c by season: " + "; ".join(
                  f"{s}: " + ", ".join(f"{t} {table[(table.Season == s) & (table.Team == t)].Elo_c.iloc[0]:+.0f}"
                                       for t in sorted(promoted[s])) for s in seasons[1:])]
    evaluate(df, prior_map, promoted, seasons, signal,
             "PROMOTED-TEAM PRIORS FROM PRESEASON CLUBELO (CHANGELOG §33)", "elo_prior", "elo",
             header=header, prereg=(0.001, 0.005))


if __name__ == "__main__":
    main()
