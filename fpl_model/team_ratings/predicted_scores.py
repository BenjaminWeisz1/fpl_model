"""Next gameweek's predicted scores -> images/next_gameweek/next_gw_predicted_scores.png (part 1).

Every fixture of FPL's next gameweek (fixtures.csv rows with that Round), each team's expected goals, listed by
kickoff, soonest first. fixtures.csv carries no dates, so kickoff times come live from FPL's fixtures API on every
run (always current, including moved matches); equal kickoffs keep fixtures.csv order, and a fixture FPL has no
kickoff for, or doesn't list in that gameweek, goes last with a warning (2026-11-05, user request).
  non-penalty goals = attack x (opponent defence / league mean) x venue factor   (predict_fixture_goals.py)
  all goals         = non-penalty goals x (1 + penalty attempts per goal x conversion)   (league_rates.penalty_rates())
from team_ratings_current.csv. Moved here from part 2's export_weekly_heatmaps.py (2026-11-04, user request), whose
version added each team's penalty goals from penalty_predictions.csv; that needs minutes.csv and the penalty share,
which only part 2 has, so part 1 uses the league penalty rate, as clean sheets and the predicted table do.

    python -u -X utf8 scripts/team_ratings/predicted_scores.py
"""
import sys
from pathlib import Path

import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parents[1] / "data" / "current"

from fpl_model.league_rates import current_season, mean_npg, penalty_rates  # noqa: E402
from fpl_model.team_ratings.update_team_ratings import home_advantage  # noqa: E402


FPL_API = "https://fantasy.premierleague.com/api"


def fetch(path):
    r = requests.get(f"{FPL_API}/{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def next_gameweek(bootstrap):
    return next(e["id"] for e in bootstrap["events"] if e["is_next"])


def kickoff_times(bootstrap, round_num):
    """(Home, Away) in project team names -> FPL kickoff (UTC Timestamp, or NaT when FPL has none yet)."""
    players = pd.read_csv(_DATA / "players_current.csv", encoding="utf-8-sig")
    team_of_code = players.drop_duplicates("Team_Short").set_index("Team_Short")["Team"]
    name = {t["id"]: team_of_code.get(t["short_name"]) for t in bootstrap["teams"]}
    unknown = [t["short_name"] for t in bootstrap["teams"] if name[t["id"]] is None]
    assert not unknown, f"FPL team codes not in players_current.csv: {unknown}"
    return {(name[f["team_h"]], name[f["team_a"]]): pd.to_datetime(f["kickoff_time"], utc=True)
            for f in fetch(f"fixtures/?event={round_num}")}


def sort_by_kickoff(fx, kickoffs):
    fx = fx.copy()
    fx["Kickoff"] = [kickoffs.get((h, a), pd.NaT) for h, a in zip(fx["Home"], fx["Away"])]
    absent = [f"{h} v {a}" for h, a in zip(fx["Home"], fx["Away"]) if (h, a) not in kickoffs]
    if absent:
        print(f"WARNING: not in FPL's gameweek, listed last: {absent}")
    extra = sorted(set(kickoffs) - set(zip(fx["Home"], fx["Away"])))
    if extra:
        print(f"WARNING: FPL lists fixtures that fixtures.csv doesn't give this round: {extra}")
    undated = [f"{h} v {a}" for h, a, k in zip(fx["Home"], fx["Away"], fx["Kickoff"]) if (h, a) in kickoffs and pd.isna(k)]
    if undated:
        print(f"WARNING: no kickoff time from FPL yet, listed last: {undated}")
    return fx.sort_values("Kickoff", kind="stable", na_position="last").reset_index(drop=True)


def predicted_scores(round_num):
    fx = pd.read_csv(_DATA / "fixtures.csv", encoding="utf-8-sig")
    fx = fx[(fx["Season"] == current_season()) & (pd.to_numeric(fx["Round"], errors="coerce") == round_num)].copy()
    if fx.empty:
        sys.exit(f"fixtures.csv has no rows for round {round_num}")
    ratings = pd.read_csv(_DATA / "team_ratings_current.csv", encoding="utf-8-sig").set_index("Team")
    missing = (set(fx["Home"]) | set(fx["Away"])) - set(ratings.index)
    assert not missing, f"no current rating for {missing}"
    avg = mean_npg()
    attempts_per_npg, conversion = penalty_rates()
    pen = 1 + attempts_per_npg * conversion
    h, a = ratings.loc[fx["Home"]].reset_index(drop=True), ratings.loc[fx["Away"]].reset_index(drop=True)
    fx = fx.reset_index(drop=True)
    fx["Home_Goals"] = h["Attack_Rating"] * (a["Defense_Rating"] / avg) * home_advantage * pen
    fx["Away_Goals"] = a["Attack_Rating"] * (h["Defense_Rating"] / avg) * (2 - home_advantage) * pen
    return fx[["Home", "Away", "Home_Goals", "Away_Goals"]]


def main():
    import seaborn as sns
    from fpl_model.export_weekly_heatmaps import _STYLE, _TEAM_DISPLAY, _render_scores
    bootstrap = fetch("bootstrap-static/")
    round_num = next_gameweek(bootstrap)
    fx = sort_by_kickoff(predicted_scores(round_num), kickoff_times(bootstrap, round_num))
    for side in ("Home", "Away"):
        fx[side] = fx[side].replace(_TEAM_DISPLAY)
    sns.set_style(_STYLE)
    _render_scores(fx, title=f"GW{round_num} Predicted Scores", filename="next_gameweek/next_gw_predicted_scores.png")
    for r in fx.itertuples():
        when = r.Kickoff.strftime("%a %d %b %H:%M UTC") if pd.notna(r.Kickoff) else "kickoff TBC"
        print(f"  {when}  {r.Home} {r.Home_Goals:.2f} - {r.Away_Goals:.2f} {r.Away}")


if __name__ == "__main__":
    main()
