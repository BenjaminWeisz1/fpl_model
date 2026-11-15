"""
Computes pre-season 2026/27 team ratings, before any matches have been played --
needed because the main exponential-smoothing script only blends toward money
priors as a side effect of processing a match row, and no 2026/27 rows exist yet.

Replicates the exact blending logic from the main script:
  - Teams that played in 2025/26 and remain in the league: blend
    (1 - money_weight) * old_rating + money_weight * money_rating
  - Newly promoted teams (no 2025/26 rating to blend from): use the
    money-implied rating directly, unblended -- same as how the main
    script handles a team's first-ever appearance.

Run locally:
    python compute_preseason_ratings.py
"""

import pandas as pd

from fpl_model import paths
from fpl_model.league_rates import current_season  # noqa: E402

_DATA = paths.CURRENT

money_weight = 0.3  # must match the constant in the main exponential-smoothing script

PREV_RATINGS_FILE = paths.CURRENT / "team_ratings_current.csv"  # end-of-2025/26 state
MONEY_FILE = paths.CURRENT / "money_priors_by_season.csv"  # consolidated money priors lookup
CURRENT_SEASON = current_season()
OUT_FILE = _DATA / f"team_ratings_preseason_{CURRENT_SEASON}.csv"


def main():
    prev = pd.read_csv(PREV_RATINGS_FILE)
    money_all = pd.read_csv(MONEY_FILE)
    money = money_all[money_all["Season"] == CURRENT_SEASON].copy()

    prev_lookup = prev.set_index("Team")[["Attack_Rating", "Defense_Rating"]]

    rows = []
    for _, r in money.iterrows():
        team = r["Team"]
        money_atk, money_def = r["Money_GF"], r["Money_GA"]

        if team in prev_lookup.index:
            old_atk = prev_lookup.at[team, "Attack_Rating"]
            old_def = prev_lookup.at[team, "Defense_Rating"]
            atk = (1 - money_weight) * old_atk + money_weight * money_atk
            defn = (1 - money_weight) * old_def + money_weight * money_def
            source = "blended"
        else:
            atk, defn = money_atk, money_def
            source = "money_only (newly promoted / no prior rating)"

        rows.append({"Team": team, "Attack_Rating": atk, "Defense_Rating": defn, "Source": source})

    out = pd.DataFrame(rows)
    out["Rating"] = out["Attack_Rating"] - out["Defense_Rating"]
    out = out.sort_values("Rating", ascending=False).reset_index(drop=True)

    print(out.to_string(index=False))

    promoted = out[out["Source"] != "blended"]
    if len(promoted) > 0:
        print(f"\n{len(promoted)} team(s) had no 2025/26 rating and used money-only ratings:")
        for t in promoted["Team"]:
            print(f"  {t}")

    out[["Team", "Attack_Rating", "Defense_Rating"]].to_csv(OUT_FILE, index=False)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
