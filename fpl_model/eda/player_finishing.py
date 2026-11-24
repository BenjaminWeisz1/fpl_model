"""EDA chart: the 30 best finishers among the top 100 midfielders and forwards by predicted points and the top 10
defenders by non-penalty goal rating expected to play (`sofascore_finishing.top_outfield`).

Reads sofascore_player_finishing_seasons.csv (`fpl_model/fetch/sofascore_finishing.py`) and draws
images/finishing/finishing_top30.png in the weekly heatmaps' look. Its one column, the finishing rating, is
(NPG + PRIOR) / (NPxG + PRIOR) over every competition-season with Sofascore xG from FIRST_SEASON: predicted
non-penalty goals per xG. Above 1 means more goals than the chances were worth; PRIOR (50, the user's choice)
pulls short records towards 1. `table` keeps minutes and both totals for checking.

The chart is exploratory and feeds no model. It skips (exit 0) when the table is missing.

Run locally:
    python -m fpl_model.eda.player_finishing
"""
import pandas as pd

from fpl_model import export_weekly_heatmaps as hm
from fpl_model import paths
from fpl_model.fetch import sofascore_finishing as finishing
from fpl_model.fetch import sofascore_keepers as pages

PRIOR = 50
OUT = "finishing/finishing_top30.png"
OUT_MID = "finishing/finishing_top_midfielders.png"
OUT_FWD = "finishing/finishing_top_forwards.png"
OUT_MID50 = "finishing/finishing_top50_midfielders.png"
SHOWN = 30
MID50 = 50  # user request: the 50 midfielders with most predicted points, in two columns of 25
FWD_N = 20  # user request: the 20 forwards with most predicted points
SCORE = "Finishing\nrating"


def table(seasons, tracked, players):
    """One row per tracked player: minutes, NPG, NPxG and the finishing rating."""
    rows = seasons[seasons["expectedGoals"].notna() & (seasons["Season_Start"] >= pages.FIRST_SEASON)]
    ids = {pid: name for _, name, pid in tracked if pid is not None}
    rows = rows[rows["Player_ID_Sofascore"].isin(ids)]
    out = rows.groupby("Player_ID_Sofascore").agg(Minutes=("minutesPlayed", "sum"), NPG=("NPG", "sum"),
                                                  NPxG=("NPxG", "sum"))
    out[SCORE] = (out["NPG"] + PRIOR) / (out["NPxG"] + PRIOR)
    out["Local_Player"] = out.index.map(ids)
    pc = players[["Player", "Web_Name", "Pos", "Team_Short", "Price"]]
    out = out.merge(pc, left_on="Local_Player", right_on="Player", how="inner")
    return out.drop(columns=["Player"]).rename(columns={"Web_Name": "Player", "Team_Short": "Team"})


def render(df, out, title, scope, scale, meta=("Player", "Pos", "Team", "Price"), shown=SHOWN, panels=1):
    """Draw the best *shown* rows of *df* by finishing rating. The subtitle says how to read the rating; the footnote
    holds *scope* (which players were considered), the method and the source. *scale* is the (lo, hi) colour range,
    shared by every finishing chart so a colour means the same rating in each."""
    first = pages.FIRST_SEASON
    hm._render_heatmap(
        df.sort_values(SCORE, ascending=False).head(shown), sort_col=SCORE, display_cols=[SCORE],
        meta_cols=list(meta), title=title,
        subtitle="Predicted non-penalty goals per xG",
        footnote=(f"{scope}\n"
                  f"All competitions since {first}/{str(first + 1)[-2:]}, penalties excluded. "
                  f"Rating = (NP goals + {PRIOR}) / (NPxG + {PRIOR}).\nSource: Sofascore."),
        filename=out, precision=2, fig_height_per_row=0.42, color_range={SCORE: scale},
        top_n=shown, panels=panels,
        show_legend=False,  # one narrow column: the legend would cover the title
    )


def limited(n, label):
    return f"Limited to the {n} {label} with the most predicted points."


def main():
    try:
        seasons = pd.read_csv(paths.current_file(finishing.OUT_FILE), encoding="utf-8-sig")
    except FileNotFoundError as exc:
        print(f"  [SKIP] finishing chart: {exc.filename} missing (collect players' Sofascore pages)")
        return 0
    players = pd.read_csv(paths.current_file("players_current.csv"), encoding="utf-8-sig")
    df = table(seasons, finishing.top_outfield(), players)
    scale = (df[SCORE].min(), df[SCORE].max())
    render(df, OUT, "Players by finishing ability",
           limited(finishing.TOP_N, "midfielders and forwards")[:-1]
           + f", and the {finishing.TOP_DEFENDERS} defenders with the best non-penalty goal rating predicted to play "
           f"over {finishing.MIN_NEXT_XMINS} minutes in the next {finishing.NEXT_GAMEWEEKS} gameweeks.", scale)
    one_pos = ("Player", "Team", "Price")
    # The midfielders (forwards) among the top N midfielders and forwards are exactly the top-k of their position.
    mids = df[df["Pos"] == "M"]
    top_fwds = finishing.top_outfield(n=FWD_N, defenders=0, positions=("F",))
    fwds = df[df["Local_Player"].isin({name for _, name, _ in top_fwds})]  # all within the collected top N
    render(mids, OUT_MID, "Midfielders by finishing ability", limited(len(mids), "midfielders"), scale, meta=one_pos)
    render(fwds, OUT_FWD, "Forwards by finishing ability", limited(FWD_N, "forwards"), scale,
           meta=one_pos)
    top_mids = finishing.top_outfield(n=MID50, defenders=0, positions=("M",))
    mid50 = df[df["Local_Player"].isin({name for _, name, _ in top_mids})]  # all within the collected top N
    render(mid50, OUT_MID50, "Midfielders by finishing ability", limited(MID50, "midfielders"), scale,
           meta=one_pos, shown=MID50, panels=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
