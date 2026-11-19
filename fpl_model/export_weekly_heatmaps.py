"""Weekly heatmap exports -- team ratings, player ratings, predicted points
and next-gameweek attacking/defcon leaderboards.

Generates colour-scaled PNGs (green = good, red = bad) from the weekly
pipeline's CSV artifacts, grouped into one images/ subfolder per family.

The charts split along the same line as the pipeline itself, so each
orchestrator renders only what its own stages have just refreshed:

  --part 1  (run_post_gameweek.py)   team ratings + player ratings
  --part 2  (run_pre_deadline.py)    predicted points + next gameweek
  --part all (default, standalone)   everything

Outputs:
  images/team_ratings/team_ratings_overall.png
  images/team_ratings/team_ratings_attack.png      (same table, sorted by attack)
  images/team_ratings/team_ratings_defense.png     (same table, best defense first)
  images/player_ratings/player_ratings_all_npg_assists.png
  images/player_ratings/player_ratings_mid_npg_assists.png
  images/player_ratings/player_ratings_fwd_npg_assists.png
  images/player_ratings/player_ratings_def_npg_assists.png
  images/player_ratings/player_ratings_def_defcon.png
  images/player_ratings/player_ratings_mid_defcon.png
  images/predicted_points/predicted_points_all.png
  images/predicted_points/predicted_points_gk.png
  images/predicted_points/predicted_points_fwd.png
  images/predicted_points/predicted_points_mid.png
  images/predicted_points/predicted_points_def.png
  images/next_gameweek/next_gw_goal_involvement.png
  images/next_gameweek/next_gw_defcon.png
  images/next_gameweek/next_gw_predicted_points.png
  (next_gw_predicted_scores.png is drawn in part 1 by fpl_model/team_ratings/predicted_scores.py)

"""
import argparse
import re
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

from fpl_model import paths

matplotlib.use("Agg")            # headless -- no GUI event loop

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from fpl_model.defcon_scaling import points_mu90, prob_at_minutes  # noqa: E402

# ── Configuration ──────────────────────────────────────────────────────────────
_OUT_DIR = paths.IMAGES
_TOP_N = 30                      # players per chart (all 20 teams shown)
# Excel-style 3-colour scale: red (low) → yellow (middle) → green (high),
# yellow anchored at the normalised midpoint (0.5).
_CMAP = LinearSegmentedColormap.from_list(
    "excel_ryg",
    [(0.0, "#F8696B"), (0.5, "#FFEB84"), (1.0, "#63BE7B")],
    N=511,          # odd table size so 0.5 hits a lut entry exactly
)
_STYLE = "whitegrid"

# Font sizes (pt) -- cell values are the point of these images, keep them large.
_FONTSIZE_CELL = 12
_FONTSIZE_HEADER = 13          # data-column headers (bold black)
_FONTSIZE_METAHDR = 11         # meta-column headers (uppercase, muted grey)
_FONTSIZE_TITLE = 15
_FONTSIZE_SUBTITLE = 10.5
_FONTSIZE_RANK = 11
_FONTSIZE_LEGEND = 8

# Greys for the editorial chrome (team-ratings.png reference look).
_C_SUBTITLE = "#666666"        # subtitle text
_C_MUTED    = "#8A8A8A"        # meta headers, legend labels
_C_RANK     = "#999999"        # rank numbers
_C_RULE     = "#BBBBBB"        # header separator rule
_C_ROWLINE  = "#DDDDDD"        # meta-row separators

_DATA = paths.CURRENT
_TEAM_RATINGS    = paths.current_file("team_ratings_current.csv", root=paths.CURRENT)
_PLAYER_RATINGS  = paths.current_file("player_ratings.csv", root=paths.CURRENT)
_PREDICTED_PTS   = paths.current_file("predicted_points.csv", root=paths.CURRENT)
_MINUTES         = paths.current_file("minutes.csv", root=paths.CURRENT)
_PLAYERS_CURRENT = paths.current_file("players_current.csv", root=paths.CURRENT)
ACTIVE_XMINS_PER_GW = 20  # charts show players averaging more than this per forecast gameweek
_SIM_FEATURES    = paths.current_file("simulation_features.csv", root=paths.CURRENT)
_PENALTIES       = paths.current_file("penalty_predictions.csv", root=paths.CURRENT)
_FIXTURE_GOALS   = paths.current_file("fixture_goal_predictions.csv", root=paths.CURRENT)
GAMEWEEK_HORIZON = 5

# Which input files each part needs (checked up front in main()).
_PART_INPUTS = {
    1: (_TEAM_RATINGS, _PLAYER_RATINGS, _MINUTES, _PLAYERS_CURRENT),
    2: (_PREDICTED_PTS, _SIM_FEATURES, _PENALTIES, _MINUTES, _PLAYERS_CURRENT),
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _active_players() -> set:
    """Return (Web_Name, Team) pairs averaging > ACTIVE_XMINS_PER_GW xMins per gameweek in minutes.csv.

    minutes.csv names players by Raw_Web_Name (e.g. "Palmer"), while
    player_ratings.csv / predicted_points.csv use Web_Name (e.g.
    "C. Palmer").  players_current.csv provides the
    (Raw_Web_Name, Team_Short) -> Web_Name bridge; pairs not found in
    the bridge are kept under their raw name as a fallback.
    """
    mins = pd.read_csv(_MINUTES, encoding="utf-8-sig")
    xmins_cols = [c for c in mins.columns if c.endswith("_xMins")]
    mins["_total"] = mins[xmins_cols].sum(axis=1)
    # 20 xMins per gameweek on average (was > 100 over five rounds), whatever the horizon.
    active_raw = mins[mins["_total"] > ACTIVE_XMINS_PER_GW * len(xmins_cols)][["Name", "Team"]]

    pc = pd.read_csv(_PLAYERS_CURRENT, encoding="utf-8-sig")
    raw_to_web = {(r.Raw_Web_Name, r.Team_Short): r.Web_Name
                  for r in pc.itertuples()}

    active = set()
    for name, team in active_raw.values:
        web = raw_to_web.get((name, team))
        active.add((web if web is not None else name, team))
    return active


def _norm(series: pd.Series) -> pd.Series:
    """Min-max normalise to [0, 1]; degenerate (flat) column → 0.5."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _fmt(v: float, precision: int) -> str:
    """Format a number for display at a fixed number of decimals.

    Never switches to scientific notation: these are leaderboard cells read at
    a glance, and a tiny-but-non-zero value (e.g. a forward's 0.0009 defcon)
    is "0.00" to the reader, not "9.00e-04". Exact zero stays "0".
    """
    if pd.isna(v):
        return ""
    if v == 0.0:
        return "0"
    return f"{v:.{precision}f}"


def _wrap_subtitle(text: str, width: int) -> list:
    """Wrap a subtitle to *width* chars, preferring sentence boundaries."""
    if not text:
        return []
    parts = [p for p in re.split(r"(?<=\.)\s+", text) if p]
    lines, cur = [], ""
    for part in parts:
        pieces = (textwrap.wrap(part, width) if len(part) > width
                  else [part])
        for piece in pieces:
            if cur and len(cur) + 1 + len(piece) > width:
                lines.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        lines.append(cur)
    return lines


def _render_heatmap(df: pd.DataFrame, sort_col: str, display_cols: list,
                    meta_cols: list, title: str, filename: str,
                    subtitle: str = "",
                    invert_cols: tuple = (), precision: int = 3,
                    fig_height_per_row: float = 0.34,
                    bold_max_cols: list = None,
                    pct_cols: tuple = (),
                    show_legend: bool = True) -> None:
    """Render a single leaderboard-style heatmap PNG (team-ratings.png look).

    Layout: bold left-aligned title + grey subtitle + separator rule, a
    WEAKER→STRONGER gradient legend top-right, then the table.  The first
    meta column renders as rank (muted grey) + name (bold) on a white
    background with light-grey row separators; the remaining meta columns
    are centred on white; data columns are colour-scaled cells.

    meta_cols    -- text-only columns (Player/Team, Pos, Team, Price)
                    placed before the colour-scaled data columns.
    display_cols -- columns shown AND colour-scaled.  If *sort_col* is a
                    derived aggregate not in *display_cols* it is used
                    only for ordering.
    invert_cols -- columns where LOWER raw value is BETTER (e.g.
                    Defense).  The colour-scale value is flipped so the
                    good end of the scale (green) still means good; the
                    cell annotation keeps the real number.
    pct_cols   -- columns whose value is a 0-1 probability rendered as a
                    whole-number percentage (e.g. 0.63 -> "63%").  Only the
                    annotation text changes; the colour scale still uses the
                    raw probability.
    subtitle    -- grey explainer line under the title.
    """
    (_OUT_DIR / filename).parent.mkdir(parents=True, exist_ok=True)
    sorted_df = (df.dropna(subset=[sort_col])
                   .sort_values(sort_col, ascending=False)
                   .reset_index(drop=True))
    n_full = len(sorted_df)
    if n_full == 0:
        print(f"  [SKIP] {filename} -- empty dataframe")
        return

    # Build the full display dataframe: meta columns + data columns.
    all_cols = list(meta_cols) + list(display_cols)
    show_df = sorted_df[all_cols].copy()

    # Normalise the data columns on the FULL eligible population so the
    # colour scale is comparable across charts, then slice to top N.
    norm_all = show_df[display_cols].apply(_norm)
    for col in invert_cols:
        if col in norm_all.columns:
            norm_all[col] = 1.0 - norm_all[col]

    # Track per-column maximums (on the full population) so specific
    # columns can bold them in the annotation loop.
    _bold_set = set(bold_max_cols or [])
    col_maxes = {}
    for col in _bold_set:
        if col in show_df.columns:
            col_maxes[col] = show_df[col].max()

    # Now slice to top N.
    show_df = show_df.head(_TOP_N).reset_index(drop=True)
    norm_df = norm_all.head(_TOP_N).reset_index(drop=True)
    n = len(show_df)

    # ── Variable-width columns via pcolormesh ──────────────────────────
    # Meta columns are sized to fit their content; data columns share
    # the remaining space equally.  Char widths use realistic DejaVu
    # Sans metrics (~0.68 em mixed case at cell size, ~0.75 em for
    # bold caps headers) so text never spills into the next column.
    cmap = _CMAP
    n_meta = len(meta_cols)
    n_all = len(all_cols)

    _CELL_CW = _FONTSIZE_CELL * 0.68 / 72
    _HDR_CW  = _FONTSIZE_METAHDR * 0.75 / 72
    _RANK_W  = 2 * _FONTSIZE_RANK * 0.64 / 72 + 0.06
    _DATA_W = max(
        4 * _CELL_CW + 0.7,
        max((len(str(c)) for c in display_cols), default=0)
        * (_FONTSIZE_HEADER * 0.70 / 72) + 0.4,
    )
    meta_widths = []
    for j, mc in enumerate(meta_cols):
        vals = show_df[mc].astype(str)
        longest = max(vals.str.len().max(), len(mc))
        content = max(longest * _CELL_CW, len(str(mc)) * _HDR_CW)
        if j == 0:                                  # rank + name column
            meta_widths.append(0.08 + _RANK_W + content + 0.10)
        else:
            meta_widths.append(0.12 + content + 0.12)
    data_widths = [_DATA_W] * len(display_cols)
    widths = meta_widths + data_widths
    total_w = sum(widths)

    x_edges = np.concatenate([[0], np.cumsum(widths)])
    y_edges = np.arange(n + 1)

    # The header band (title/subtitle/rule/legend) lives in data
    # coordinates above the table: 1 y-unit = fig_height_per_row inches,
    # and _hy(pt) is the y position of a point `pt` above the table top.
    _row_pt = fig_height_per_row * 72.0
    def _hy(pts: float) -> float:
        return -pts / _row_pt

    # Wrap the subtitle to the table width (sentence boundaries first)
    # so it never overflows the axes.
    sub_chars = max(20, int(total_w / (_FONTSIZE_SUBTITLE * 0.52 / 72)))
    sub_lines = _wrap_subtitle(subtitle, sub_chars)
    _SUB_LINE_PT = 13.0
    sub_bottom, sub_top = 37.0, 37.0 + _SUB_LINE_PT * len(sub_lines)
    title_bottom = sub_top + 5.0
    title_top = title_bottom + 15.0
    header_pt = title_top + 6.0

    # Figure sized EXACTLY to the content with the axes filling it, so
    # one x-unit is exactly one inch and one y-unit exactly one row.
    # (plt.tight_layout must not be used here: an ax.text wider than the
    # axes -- e.g. a long subtitle -- makes it squeeze the axes box and
    # every column with it.)
    fig_w = total_w
    fig_h = (n + header_pt / _row_pt) * fig_height_per_row
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Data columns: colour-scaled cells with white gridlines.
    ax.pcolormesh(x_edges[n_meta:], y_edges, norm_df.values,
                  cmap=cmap, vmin=0.0, vmax=1.0,
                  edgecolors="white", linewidths=0.5)

    # Meta area: white background, light-grey separators between rows.
    if n_meta:
        for r in range(1, n):
            ax.plot([x_edges[0], x_edges[n_meta]], [r, r],
                    color=_C_ROWLINE, lw=0.8, zorder=1)

    # Place text annotations in each cell.
    name_x = x_edges[0] + 0.08
    for r in range(n):
        cy = r + 0.5
        # First meta column: rank (muted grey) + name (bold), left-aligned.
        if n_meta:
            ax.text(name_x, cy, str(r + 1), ha="left", va="center",
                    fontsize=_FONTSIZE_RANK, color=_C_RANK)
            ax.text(name_x + _RANK_W, cy, str(show_df.iloc[r, 0]),
                    ha="left", va="center", fontsize=_FONTSIZE_CELL,
                    fontweight="bold", color="black")
            for c in range(1, n_meta):                  # other meta -- centred
                cx = (x_edges[c] + x_edges[c + 1]) / 2
                ax.text(cx, cy, str(show_df.iloc[r, c]),
                        ha="center", va="center",
                        fontsize=_FONTSIZE_CELL, color="black")
        # Data columns: formatted value, text colour by cell luma.
        for c in range(n_meta, n_all):
            cx = (x_edges[c] + x_edges[c + 1]) / 2
            raw = show_df.iloc[r, c]
            v = float(raw)
            if all_cols[c] in pct_cols:  # noqa: SIM108 (a nested conditional expression would read worse)
                txt = "" if pd.isna(raw) else f"{v * 100:.0f}%"
            else:
                txt = _fmt(v, precision)
            rgba = cmap(norm_df.iloc[r, c - n_meta])
            luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            is_col_max = (all_cols[c] in col_maxes
                          and show_df.iloc[r, c] == col_maxes[all_cols[c]])
            ax.text(cx, cy, txt, ha="center", va="center",
                    fontsize=_FONTSIZE_CELL,
                    fontweight="bold" if is_col_max else "normal",
                    color="white" if luma < 0.5 else "black")

    # Column headers: meta (uppercase, muted grey) and data (bold black).
    hdr_y = _hy(15)
    if n_meta:
        ax.text(name_x, hdr_y, str(all_cols[0]).upper(), ha="left",
                va="center", fontsize=_FONTSIZE_METAHDR,
                fontweight="bold", color=_C_MUTED)
        for c in range(1, n_meta):
            cx = (x_edges[c] + x_edges[c + 1]) / 2
            ax.text(cx, hdr_y, str(all_cols[c]).upper(), ha="center",
                    va="center", fontsize=_FONTSIZE_METAHDR,
                    fontweight="bold", color=_C_MUTED)
    for c in range(n_meta, n_all):
        cx = (x_edges[c] + x_edges[c + 1]) / 2
        ax.text(cx, hdr_y, str(all_cols[c]), ha="center", va="center",
                fontsize=_FONTSIZE_HEADER, fontweight="bold", color="black")

    # Title / wrapped subtitle / separator rule.
    if sub_lines:
        ax.text(x_edges[0], _hy(sub_bottom), "\n".join(sub_lines),
                ha="left", va="bottom", fontsize=_FONTSIZE_SUBTITLE,
                color=_C_SUBTITLE, linespacing=1.25)
    ax.text(x_edges[0], _hy(title_top), title, ha="left", va="top",
            fontsize=_FONTSIZE_TITLE, fontweight="bold", color="black")
    ax.plot([x_edges[0], x_edges[-1]], [_hy(33), _hy(33)],
            color=_C_RULE, lw=1.0)

    if show_legend:
        # WEAKER → STRONGER gradient legend, vertically centred on the title.
        _lw = 1.15
        _lx1 = x_edges[-1]
        _lx0 = _lx1 - _lw
        _ly = _hy((title_bottom + title_top) / 2)
        _lh = 7.0 / _row_pt
        for i in range(64):
            ax.add_patch(Rectangle((_lx0 + i * _lw / 64, _ly - _lh / 2),
                                   _lw / 64, _lh,
                                   facecolor=cmap(i / 63), edgecolor="none"))
        ax.text(_lx0 - 0.07, _ly, "WEAKER", ha="right", va="center",
                fontsize=_FONTSIZE_LEGEND, color=_C_MUTED)
        ax.text(_lx1 + 0.07, _ly, "STRONGER", ha="left", va="center",
                fontsize=_FONTSIZE_LEGEND, color=_C_MUTED)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(n, _hy(header_pt))                     # header band at top
    ax.set_axis_off()
    path = _OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(_OUT_DIR.parent).as_posix()}"
          f"  ({n} rows, {len(all_cols)} cols)")


def _render_scores(df: pd.DataFrame, title: str, filename: str,
                   subtitle: str = "") -> None:
    """Render a fixture list "Home [1.5] [1.4] Away" in the same editorial look (bold title, optional grey
    subtitle, rule, light row separators). Each score is its own cell on the heatmaps' red-yellow-green scale,
    home and away goals sharing one min-max scale (more goals = greener).
    df columns: Home, Away, Home_Goals, Away_Goals (in display order)."""
    (_OUT_DIR / filename).parent.mkdir(parents=True, exist_ok=True)
    n = len(df)
    if n == 0:
        print(f"  [SKIP] {filename} -- no fixtures")
        return
    row_h = 0.42
    _CELL_CW = _FONTSIZE_CELL * 0.68 / 72
    name_w = max(df["Home"].str.len().max(), df["Away"].str.len().max()) * _CELL_CW + 0.25
    score_w = 1.5                      # two score cells side by side
    cell_w = score_w / 2
    goals = pd.concat([df["Home_Goals"], df["Away_Goals"]], ignore_index=True)
    lo, hi = goals.min(), goals.max()
    shade = (lambda v: _CMAP(0.5)) if hi == lo else (lambda v: _CMAP((v - lo) / (hi - lo)))
    total_w = max(2 * name_w + score_w, 4.5)
    name_w = (total_w - score_w) / 2
    _row_pt = row_h * 72.0
    def _hy(pts: float) -> float:
        return -pts / _row_pt
    sub_lines = _wrap_subtitle(subtitle, max(20, int(total_w / (_FONTSIZE_SUBTITLE * 0.52 / 72))))
    sub_bottom, sub_top = 14.0, 14.0 + 13.0 * len(sub_lines)
    title_bottom = sub_top + 5.0
    title_top = title_bottom + 15.0
    header_pt = title_top + 6.0

    fig, ax = plt.subplots(figsize=(total_w, (n + header_pt / _row_pt) * row_h))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for r in range(1, n):
        ax.plot([0, total_w], [r, r], color=_C_ROWLINE, lw=0.8)
    for r, row in enumerate(df.itertuples(index=False)):
        cy = r + 0.5
        ax.text(name_w - 0.12, cy, row.Home, ha="right", va="center",
                fontsize=_FONTSIZE_CELL, fontweight="bold", color="black")
        for i, v in enumerate((row.Home_Goals, row.Away_Goals)):
            x0 = name_w + i * cell_w
            ax.add_patch(plt.Rectangle((x0, r), cell_w, 1, facecolor=shade(v), edgecolor="white", lw=1.5))
            ax.text(x0 + cell_w / 2, cy, f"{v:.2f}", ha="center", va="center",
                    fontsize=_FONTSIZE_CELL + 1, color="black")
        ax.text(name_w + score_w + 0.12, cy, row.Away, ha="left", va="center",
                fontsize=_FONTSIZE_CELL, fontweight="bold", color="black")
    if sub_lines:
        ax.text(0, _hy(sub_bottom), "\n".join(sub_lines), ha="left", va="bottom",
                fontsize=_FONTSIZE_SUBTITLE, color=_C_SUBTITLE, linespacing=1.25)
    ax.text(0, _hy(title_top), title, ha="left", va="top",
            fontsize=_FONTSIZE_TITLE, fontweight="bold", color="black")
    ax.plot([0, total_w], [_hy(8), _hy(8)], color=_C_RULE, lw=1.0)
    ax.set_xlim(0, total_w)
    ax.set_ylim(n, _hy(header_pt))
    ax.set_axis_off()
    path = _OUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.relative_to(_OUT_DIR.parent).as_posix()}  ({n} fixtures)")


# Display names for the predicted-scores chart (data files use fbref-style short forms).
_TEAM_DISPLAY = {"Manchester Utd": "Manchester United", "Newcastle Utd": "Newcastle United",
                 "Nott'ham Forest": "Nottingham Forest"}


# ── Charts ─────────────────────────────────────────────────────────────────────
def team_ratings_chart() -> None:
    print("\n── Team ratings ──")
    df = pd.read_csv(_TEAM_RATINGS)
    df = df.dropna(subset=["Team"])
    # Shorter headers for the image ("Attack" / "Defense" instead of
    # "Attack_Rating" / "Defense_Rating").
    df = df.rename(columns={"Attack_Rating": "Attack",
                            "Defense_Rating": "Defense"})
    df["Overall"] = df["Attack"] - df["Defense"]
    df["_DefenseOrder"] = -df["Defense"]      # lower conceded = better = first

    # Same table three ways: by overall rating, by attack, by defense.
    for sort_col, title, fname in (
            ("Overall", "Team Ratings", "team_ratings_overall.png"),
            ("Attack", "Team Ratings by Attack", "team_ratings_attack.png"),
            ("_DefenseOrder", "Team Ratings by Defense", "team_ratings_defense.png")):
        _render_heatmap(
            df, sort_col=sort_col,
            display_cols=["Attack", "Defense", "Rating"],
            meta_cols=["Team"],
            invert_cols=("Defense",),             # lower is better defensively
            title=title,
            subtitle=("Predicted goals and goals conceded vs average opponent. "
                      "Rating is attack minus defense."),
            filename=f"team_ratings/{fname}",
            precision=2, fig_height_per_row=0.48,
        )


def player_ratings_charts() -> None:
    print("\n── Player ratings ──")
    df = pd.read_csv(_PLAYER_RATINGS, encoding="utf-8-sig")
    df = df[df["Pos"].isin(["D", "M", "F"])].copy()   # drop goalkeepers
    # Exclude players with zero expected minutes (injured, on loan, etc.).
    active = _active_players()
    df = df[df[["Player", "Team"]].apply(tuple, axis=1).isin(active)].copy()
    df["Defcon"] = pd.to_numeric(df["Defcon"], errors="coerce").fillna(0.0)
    df["NPG+Assists"] = df["NPG"] + df["Assists"]

    # (sort_col, pos_filter, filename, display_cols)
    specs = [
        # NPG+Assists views
        ("NPG+Assists", None, "player_ratings/player_ratings_all_npg_assists.png",
         ["NPG", "Assists", "NPG+Assists"]),
        ("NPG+Assists", "M",  "player_ratings/player_ratings_mid_npg_assists.png",
         ["NPG", "Assists", "NPG+Assists"]),
        ("NPG+Assists", "F",  "player_ratings/player_ratings_fwd_npg_assists.png",
         ["NPG", "Assists", "NPG+Assists"]),
        ("NPG+Assists", "D",  "player_ratings/player_ratings_def_npg_assists.png",
         ["NPG", "Assists", "NPG+Assists"]),
        # Defcon views
        ("Defcon",      "D",  "player_ratings/player_ratings_def_defcon.png",
         ["Defcon", "NPG", "Assists"]),
        ("Defcon",      "M",  "player_ratings/player_ratings_mid_defcon.png",
         ["Defcon", "NPG", "Assists"]),
    ]
    _POS_LABEL = {None: "All positions", "M": "Midfielders",
                  "F": "Forwards", "D": "Defenders"}

    for sort_col, pos, fname, cols in specs:
        subset = df[df["Pos"] == pos] if pos is not None else df
        if pos is None:
            title = f"Player Ratings by {sort_col}"
            # Only the all-positions chart carries a subtitle; the
            # position-specific ones are self-explanatory from the title.
            subtitle = ("Predicted non-penalty goals and assists in a "
                        "90 minute match vs the average opponent.")
        else:
            title = f"{_POS_LABEL[pos]} by {sort_col}"
            subtitle = ""
        meta = ["Player"] + (["Pos"] if pos is None else []) + ["Team", "Price"]
        _render_heatmap(subset, sort_col=sort_col, display_cols=cols,
                        meta_cols=meta, title=title, subtitle=subtitle,
                        filename=fname, precision=2)


def predicted_points_charts() -> None:
    print("\n── Predicted points ──")
    df = pd.read_csv(_PREDICTED_PTS, encoding="utf-8-sig")
    df = df[df["Pos"].isin(["G", "D", "M", "F"])].copy()
    # Exclude players with zero expected minutes (injured, on loan, etc.).
    active = _active_players()
    df = df[df[["Player", "Team"]].apply(tuple, axis=1).isin(active)].copy()

    # Detect gameweek columns dynamically: any integer-typed column
    # between Price and Total is a gameweek (GW number = the next N
    # rounds after the last completed match).
    cols = list(df.columns)
    price_idx = cols.index("Price")
    total_idx = cols.index("Total")
    gw_cols = sorted((c for c in cols[price_idx + 1 : total_idx]
                      if str(c).isdigit()), key=int)[:GAMEWEEK_HORIZON]
    print(gw_cols)
    if gw_cols:
        # The input Total covers the full forecast horizon.
        df["Total"] = df[gw_cols].sum(axis=1)

    # (pos_filter, filename)
    specs = [
        (None, "predicted_points/predicted_points_all.png"),
        ("G",  "predicted_points/predicted_points_gk.png"),
        ("F",  "predicted_points/predicted_points_fwd.png"),
        ("M",  "predicted_points/predicted_points_mid.png"),
        ("D",  "predicted_points/predicted_points_def.png"),
    ]
    _POS_LABEL = {None: "All positions", "G": "Goalkeepers", "M": "Midfielders",
                  "F": "Forwards", "D": "Defenders"}

    for pos, fname in specs:
        subset = df[df["Pos"] == pos] if pos is not None else df
        display = list(gw_cols) + ["Total"]
        # Names are digit strings, so compare as ints ("10" < "6" lexically).
        gw_range = (f"GW{int(gw_cols[0])}\u2013{int(gw_cols[-1])}"
                    if gw_cols else "\u2014")
        if pos is None:
            title = f"Predicted Points ({gw_range})"
            # Only the all-positions chart carries a subtitle; the
            # position-specific ones are self-explanatory from the title.
            subtitle = "FPL predicted points per gameweek, sorted by total."
        else:
            title = f"Predicted Points ({gw_range}) \u2014 {_POS_LABEL[pos]}"
            subtitle = ""
        meta = ["Player"] + (["Pos"] if pos is None else []) + ["Team", "Price"]
        _render_heatmap(subset, sort_col="Total", display_cols=display,
                        meta_cols=meta, title=title, subtitle=subtitle,
                        filename=fname, precision=2,
                        fig_height_per_row=0.38,
                        bold_max_cols=gw_cols if pos is None else None)


def next_gameweek_charts() -> None:
    """Top-30 leaderboards for the NEXT gameweek only (not the multi-week total).

    Three heatmap leaderboards, each with player, opponent and fixture venue,
    but without a legend or a "-- Top N" suffix in the title:

      next_gw_goal_involvement.png -- Goals / Assists / G+A
      next_gw_defcon.png           -- Defcon probability only, as a whole %
      next_gw_predicted_points.png -- next-gameweek predicted points only

    Goals / assists / G+A / defcon are read off simulation_features.csv, the
    same per-player, per-gameweek table the bonus simulation and the points
    assembly consume, so these charts cannot drift from predicted_points.csv:

      Goals   = NPG90_scaled * xMins/90  +  expected penalty goals
      Assists = Ast90_scaled * xMins/90
      G+A     = Goals + Assists
      Defcon  = P(defcon stat >= position threshold) at this week's xMins,
                re-scaled from the unscaled 90-minute rate (see
                defcon_scaling.py for why it is re-scaled here, rather than
                taken from predictions_defensive.csv's scaled column).

    The predicted-points chart reads the next-gameweek column straight out of
    predicted_points.csv so it is identical to the per-week column there.
    """
    print("\n── Next gameweek ──")
    all_sim = pd.read_csv(_SIM_FEATURES, encoding="utf-8-sig")
    sim = all_sim[all_sim["Pos"].isin(["D", "M", "F"])].copy()
    sim = sim[sim["xMins"] > 0].copy()
    if sim.empty:
        print("  [SKIP] simulation_features.csv has no D/M/F rows")
        return

    # GW is the 1..HORIZON label of the forecast rounds, so the next gameweek is
    # GW == 1; the Round column carries the real round number for the title.
    next_gw = int(sim["GW"].min())
    sim = sim[sim["GW"] == next_gw].copy()
    round_num = int(sim["Round"].mode().iloc[0])
    all_next_gw = all_sim[all_sim["GW"] == next_gw].copy()

    # Display opponent teams with the same FPL short codes used elsewhere.
    pc = pd.read_csv(_PLAYERS_CURRENT, encoding="utf-8-sig")
    team_short = dict(zip(pc["Team"], pc["Team_Short"]))
    all_next_gw["Opponent"] = all_next_gw["Opponent"].map(team_short).fillna(
        all_next_gw["Opponent"]
    )

    venue_code = all_next_gw["Venue"].map({"Home": "H", "Away": "A"})
    invalid_venue = all_next_gw["Opponent"].notna() & venue_code.isna()
    if invalid_venue.any():
        values = sorted(all_next_gw.loc[invalid_venue, "Venue"].astype(str).unique())
        raise ValueError(f"Unknown fixture venue value(s): {values}")
    all_next_gw["Opponent"] = np.where(
        all_next_gw["Opponent"].notna(),
        all_next_gw["Opponent"].astype(str) + " (" + venue_code.fillna("") + ")",
        "—",
    )

    # Expected penalty goals for the same gameweek, keyed on the full player
    # name (both files carry it alongside Web_Name).
    pens = pd.read_csv(_PENALTIES, encoding="utf-8-sig")
    pen_col = f"PenGoals_GW{next_gw}"
    if pen_col in pens.columns:
        pen_lookup = dict(zip(pens["Player"], pens[pen_col]))
    else:
        print(f"  WARNING: {_PENALTIES.name} has no {pen_col} column -- "
              f"penalty goals treated as zero")
        pen_lookup = {}
    pen_goals = sim["Player"].map(pen_lookup).fillna(0.0)

    minutes_frac = sim["xMins"] / 90.0
    sim["Goals"] = sim["NPG90_scaled"] * minutes_frac + pen_goals
    sim["Assists"] = sim["Ast90_scaled"] * minutes_frac
    sim["G+A"] = sim["Goals"] + sim["Assists"]
    sim["Defcon"] = prob_at_minutes(points_mu90(sim["Defcon_Mu90"].fillna(0.0), sim["Pos"]),
                                    sim["xMins"], sim["Threshold"])

    # Display player names in the same convention as the other charts.
    sim["Player"] = sim["Web_Name"]
    sim["Opponent"] = sim["Opponent"].map(team_short).fillna(sim["Opponent"])

    # 1. Goal involvement -- goals, assists and G+A (defcon dropped).
    sim["Opponent"] = (
        sim["Opponent"].astype(str) + " ("
        + sim["Venue"].map({"Home": "H", "Away": "A"}).fillna("") + ")"
    )
    _render_heatmap(
        sim, sort_col="G+A", display_cols=["Goals", "Assists", "G+A"],
        meta_cols=["Player", "Opponent"], title=f"GW{round_num} Goal Involvement",
        subtitle="", filename="next_gameweek/next_gw_goal_involvement.png",
        precision=2, show_legend=False,
    )

    # 2. Defensive contributions -- probability only, whole-number percent.
    _render_heatmap(
        sim, sort_col="Defcon", display_cols=["Defcon"],
        meta_cols=["Player", "Opponent"], title=f"GW{round_num} Defensive Contributions",
        subtitle="", filename="next_gameweek/next_gw_defcon.png",
        precision=2, pct_cols=("Defcon",), show_legend=False,
    )

    # 3. Predicted points -- next gameweek column only.
    pts = pd.read_csv(_PREDICTED_PTS, encoding="utf-8-sig")
    pts = pts[pts["Pos"].isin(["G", "D", "M", "F"])].copy()
    active = _active_players()
    pts = pts[pts[["Player", "Team"]].apply(tuple, axis=1).isin(active)].copy()
    fixture_context = all_next_gw[["Web_Name", "Team", "Opponent"]].copy()
    fixture_context["Team"] = fixture_context["Team"].map(team_short).fillna(
        fixture_context["Team"]
    )
    fixture_context = fixture_context.rename(columns={"Web_Name": "Player"})
    pts = pts.merge(fixture_context, on=["Player", "Team"], how="left",
                    validate="many_to_one")
    pts["Opponent"] = pts["Opponent"].fillna("—")
    pcols = list(pts.columns)
    price_idx = pcols.index("Price")
    total_idx = pcols.index("Total")
    gw_cols = sorted((c for c in pcols[price_idx + 1 : total_idx]
                      if str(c).isdigit()), key=int)
    if not gw_cols:
        print("  [SKIP] next_gw_predicted_points -- no gameweek column found")
    else:
        # The first gameweek column is the next round (matches round_num).
        pts = pts.rename(columns={gw_cols[0]: "Points"})
        _render_heatmap(
            pts, sort_col="Points", display_cols=["Points"],
            meta_cols=["Player", "Opponent"], title=f"GW{round_num} Predicted Points",
            subtitle="", filename="next_gameweek/next_gw_predicted_points.png",
            precision=2, show_legend=False,
        )
    # Predicted scores moved to part 1: fpl_model/team_ratings/predicted_scores.py (2026-11-04).


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render the weekly chart PNGs into images/.")
    parser.add_argument(
        "--part", choices=["1", "2", "all"], default="all",
        help="1 = team + player ratings (post-gameweek); "
             "2 = predicted points + next gameweek (pre-deadline); "
             "all = both (default)")
    args = parser.parse_args()
    parts = [1, 2] if args.part == "all" else [int(args.part)]

    required = [f for part in parts for f in _PART_INPUTS[part]]
    missing = sorted({f.name for f in required if not f.exists()})
    if missing:
        print(f"Missing input files: {', '.join(missing)}")
        print(f"Expected in: {_DATA}")
        return 1

    sns.set_style(_STYLE)

    if 1 in parts:
        team_ratings_chart()
        player_ratings_charts()
    if 2 in parts:
        predicted_points_charts()
        next_gameweek_charts()
    print(f"\nDone — {_OUT_DIR}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
