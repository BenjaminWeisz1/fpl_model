"""Predicted final league table -> images/team_ratings/predicted_table.png (+ data/current/predicted_table.csv).

Points = current points from the season's results in match_results.csv (3 a win, 1 a draw) + expected points from
every remaining fixture: 3 x P(win) + 1 x P(draw).

Remaining fixtures come live from FPL (/api/fixtures/, as the fixture ticker), so moved and postponed matches are
included; a fixture counts as remaining when match_results.csv has no result for it. Each team's goals in a fixture
are Poisson, independent, with the production means:
  non-penalty goals  = attack x (opponent defence / league mean) x venue factor   (predict_fixture_goals.py)
  all goals          = non-penalty goals x (1 + penalty attempts per goal x conversion)
                                                       (league_rates.penalty_rates(), as clean sheets use)
Ratings are today's team_ratings_current.csv for every remaining match (no updating along the way), so the table is
an expected value, not a simulation of how ratings would move.

    python -u -X utf8 scripts/team_ratings/predicted_table.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_DATA = _ROOT / "data" / "current"
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "fetch"))
sys.path.insert(0, str(_HERE))

from league_rates import current_season, mean_npg, penalty_rates  # noqa: E402
from team_name_map import to_fbref_name  # noqa: E402
from update_team_ratings import home_advantage  # noqa: E402

BASE_URL = "https://fantasy.premierleague.com/api"
OUT_CSV = _DATA / "predicted_table.csv"
IMAGE = "team_ratings/predicted_table.png"
MAX_GOALS = 15          # Poisson tail beyond this is < 1e-9 at these means


def fetch(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def outcome_probs(home_mean, away_mean):
    """(P home win, P draw, P away win) for independent Poisson goals."""
    k = np.arange(MAX_GOALS + 1)
    joint = np.outer(poisson.pmf(k, home_mean), poisson.pmf(k, away_mean))
    return float(np.tril(joint, -1).sum()), float(np.trace(joint)), float(np.triu(joint, 1).sum())


def current_points(results, teams):
    table = {t: {"Played": 0, "Points": 0} for t in teams}
    for h, a, hg, ag in zip(results["Home"], results["Away"], results["Home_Goals"], results["Away_Goals"]):
        for team, gf, ga in ((h, hg, ag), (a, ag, hg)):
            table[team]["Played"] += 1
            table[team]["Points"] += 3 if gf > ga else 1 if gf == ga else 0
    return table


def build():
    season = current_season()
    fpl = fetch(f"{BASE_URL}/bootstrap-static/")
    fixtures = fetch(f"{BASE_URL}/fixtures/")
    name = {t["id"]: to_fbref_name(t["name"]) for t in fpl["teams"]}
    teams = sorted(name.values())

    m = pd.read_csv(_DATA / "match_results.csv", encoding="utf-8-sig")
    played = m[(m["Season"] == season) & m["Home_Goals"].notna() & m["Away_Goals"].notna()]
    done = set(zip(played["Home"], played["Away"]))
    assert set(played["Home"]) | set(played["Away"]) <= set(teams), "match_results team not in FPL's list"

    all_pairs = [(name[f["team_h"]], name[f["team_a"]]) for f in fixtures]
    assert len(all_pairs) == len(set(all_pairs)) == len(teams) * (len(teams) - 1), "FPL fixture list is not a full season"
    remaining = [p for p in all_pairs if p not in done]
    assert len(remaining) + len(done) == len(all_pairs), "a result in match_results.csv is not on FPL's fixture list"

    ratings = pd.read_csv(_DATA / "team_ratings_current.csv", encoding="utf-8-sig").set_index("Team")
    missing = set(teams) - set(ratings.index)
    assert not missing, f"no current rating for {missing}"
    avg = mean_npg()
    attempts_per_npg, conversion = penalty_rates()
    pen_factor = 1 + attempts_per_npg * conversion

    table = current_points(played, teams)
    for t in teams:
        table[t].update({"Exp_Points_Remaining": 0.0, "Exp_W": 0.0, "Exp_D": 0.0, "Exp_L": 0.0, "Remaining": 0})
    for home, away in remaining:
        h, a = ratings.loc[home], ratings.loc[away]
        home_goals = h["Attack_Rating"] * (a["Defense_Rating"] / avg) * home_advantage * pen_factor
        away_goals = a["Attack_Rating"] * (h["Defense_Rating"] / avg) * (2 - home_advantage) * pen_factor
        pw, pd_, pl = outcome_probs(home_goals, away_goals)
        for team, w, l in ((home, pw, pl), (away, pl, pw)):
            row = table[team]
            row["Exp_Points_Remaining"] += 3 * w + pd_
            row["Exp_W"] += w
            row["Exp_D"] += pd_
            row["Exp_L"] += l
            row["Remaining"] += 1

    out = pd.DataFrame.from_dict(table, orient="index").rename_axis("Team").reset_index()
    out["Predicted_Points"] = out["Points"] + out["Exp_Points_Remaining"]
    out = out.sort_values("Predicted_Points", ascending=False).reset_index(drop=True)
    assert (out["Played"] + out["Remaining"] == 2 * (len(teams) - 1)).all()
    next_gw = next((e["id"] for e in fpl["events"] if e["is_next"]), None)
    return out, len(remaining), next_gw, pen_factor


def draw(out):
    """Same renderer and style as the team-ratings images (export_weekly_heatmaps), team names as they use."""
    import seaborn as sns
    from export_weekly_heatmaps import _STYLE, _render_heatmap
    sns.set_style(_STYLE)
    df = out[["Team", "Predicted_Points"]].rename(columns={"Predicted_Points": "Points"})
    _render_heatmap(df, sort_col="Points", display_cols=["Points"], meta_cols=["Team"], title="Predicted Final Table",
                    filename=IMAGE, precision=1, fig_height_per_row=0.48, show_legend=False)


def main():
    out, n_remaining, next_gw, pen_factor = build()
    cols = ["Team", "Played", "Points", "Remaining", "Exp_W", "Exp_D", "Exp_L",
            "Exp_Points_Remaining", "Predicted_Points"]
    out[cols].round(3).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    draw(out)
    print(f"{n_remaining} remaining fixtures; penalty factor {pen_factor:.4f}; "
          f"total predicted points {out['Predicted_Points'].sum():.1f}")
    for r in out.itertuples():
        print(f"  {r.Team:18s} {r.Points:3d} + {r.Exp_Points_Remaining:5.1f} = {r.Predicted_Points:5.1f}")
    print(f"Wrote images/{IMAGE} and {OUT_CSV.name}")


if __name__ == "__main__":
    main()
