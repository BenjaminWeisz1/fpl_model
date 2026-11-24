"""EDA chart: shot-stopping of each team's starting goalkeeper, over every competition Sofascore has xG for.

Reads sofascore_keeper_seasons.csv (`fpl_model/fetch/sofascore_keepers.py`) and draws
images/goalkeepers/shot_stopping.png in the weekly heatmaps' look, one row per starting keeper (the goalkeeper
with the most FPL Review xMins over the next five gameweeks). Its one column, the shot-stopping rating, is
(NPGC + PRIOR) / (NPxGOTC + PRIOR) over the competition-seasons with xGOT from FIRST_SEASON: non-penalty goals
conceded per non-penalty expected goal on target conceded (user decision, 2026-11-19; it was the inverse). Below 1
means fewer goals than the shots on target were worth, so lower is better: the chart lists the lowest first and
colours low green. PRIOR (50, the user's choice) pulls short records towards 1. `table` keeps minutes and both
totals for checking.

Penalties are removed with rates measured from Sofascore shot maps (sofascore_keepers.penalty_rates).
The chart is exploratory and feeds no model. It skips (exit 0) when the table is missing.

Run locally:
    python -m fpl_model.eda.keeper_shot_stopping
"""
import pandas as pd

from fpl_model import export_weekly_heatmaps as hm
from fpl_model import paths
from fpl_model.fetch import sofascore_keepers as keepers

PRIOR = 50
OUT = "goalkeepers/shot_stopping.png"
MINUTES, XGOTC, GC, SCORE = "Minutes", "NPxGOT\nconceded", "NP goals\nconceded", "Shot-stopping\nrating"


def table(seasons, starters, players):
    """One row per starting keeper: minutes, NPxGOTC, NPGC and the shot-stopping ratio."""
    rows = seasons[seasons["expectedGoalsOnTargetConceded"].notna()
                   & (seasons["Season_Start"] >= keepers.FIRST_SEASON)]
    ids = {pid: name for _, name, pid in starters if pid is not None}
    rows = rows[rows["Player_ID_Sofascore"].isin(ids)]
    out = rows.groupby("Player_ID_Sofascore").agg(**{MINUTES: ("minutesPlayed", "sum"), XGOTC: ("NPxGOTC", "sum"),
                                                     GC: ("NPGC", "sum")})
    out[SCORE] = (out[GC] + PRIOR) / (out[XGOTC] + PRIOR)
    out["_Order"] = -out[SCORE]  # lowest (best) first
    out["Local_Player"] = out.index.map(ids)
    pc = players[["Player", "Web_Name", "Team_Short", "Price"]]
    out = out.merge(pc, left_on="Local_Player", right_on="Player", how="inner")
    return out.drop(columns=["Player"]).rename(columns={"Web_Name": "Player", "Team_Short": "Team"})


def main():
    try:
        seasons = pd.read_csv(paths.current_file(keepers.OUT_FILE), encoding="utf-8-sig")
    except FileNotFoundError as exc:
        print(f"  [SKIP] shot-stopping chart: {exc.filename} missing (collect keepers' Sofascore player pages)")
        return 0
    players = pd.read_csv(paths.current_file("players_current.csv"), encoding="utf-8-sig")
    df = table(seasons, keepers.starting_keepers(), players)
    first = keepers.FIRST_SEASON
    hm._render_heatmap(
        df, sort_col="_Order", display_cols=[SCORE], invert_cols=(SCORE,),
        meta_cols=["Player", "Team", "Price"],
        title="Goalkeepers by shot-stopping ability",
        subtitle=(f"All competitions since {first}/{str(first + 1)[-2:]}, penalties excluded. "
                  f"Shot-stopping = (NP goals conceded + {PRIOR}) / (NPxGOT conceded + {PRIOR}). "
                  "Rating represents predicted goals conceded per xG on target conceded. "
                  "Source: Sofascore."),
        filename=OUT, precision=2, fig_height_per_row=0.48,
        show_legend=False,  # one narrow column: the legend would cover the title
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
