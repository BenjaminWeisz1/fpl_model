"""Draw three fixture tickers for the next 8 gameweeks to images/fdr/:

    fixture_ticker_GW<nn>.png          (Overall) expected goal difference
    fixture_ticker_GW<nn>_attack.png   (Attack)  expected goals scored: the opponent's defence, for attackers
    fixture_ticker_GW<nn>_defense.png  (Defense) expected goals conceded, lower is easier: the opponent's attack

    python -u -X utf8 scripts/optimizer/fixture_ticker.py

The gameweek comes from FPL, so it needs no edits week to week. Each of the 40 possible fixtures (20 opponents, home or
away) gets a difficulty: the expected goal difference an average team would have in it, from team_ratings_current.csv
and the same formula as predict_fixture_goals.py. The 40 are grouped into six bands by k-means, which puts the breaks
where the natural gaps are. Fixtures come live from FPL, so moved games, blanks and doubles show as they stand.

Attack and Defense are the two halves of that goal difference (an average team's goals for, and goals against with the
sign flipped so higher is easier); each ticker has its own six k-means bands and its own team order.

Teams are sorted easiest run first, by total difficulty with each week's fixtures weighted 0.85 ^ (weeks ahead): a blank
adds nothing and a double counts both games. data/optimizer/GW<nn>/fixture_difficulty.csv holds the 40 values of each.
"""
from fpl_model import paths
import json
import os
import sys
from pathlib import Path

import matplotlib

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")  # joblib otherwise warns: Windows 11 has no wmic
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = paths.ROOT
_DATA = paths.CURRENT
_IMAGES = paths.IMAGES / "fdr"

from fpl_model.team_ratings.predict_fixture_goals import AVG_CONCEDED, home_advantage  # noqa: E402

BASE_URL = "https://fantasy.premierleague.com/api"
WEEKS = 8
LAST_GW = 38
DECAY = 0.85
BANDS = 6
KMEANS_RESTARTS = 50
KMEANS_SEED = 42

BACKGROUND = "#0b1622"
TEXT = "#d8e4ee"
MUTED = "#7f95a8"
CELL_TEXT = "#0b1622"
DARK_BAND_TEXT = "#ffffff"
DARK_BAND = 1
BLANK = "#1a2a3a"
TICKERS = (  # (title suffix, score column, file suffix); every score is higher = easier
    ("Overall", "Goal_Diff", ""),
    ("Attack", "Goals_For", "_attack"),
    ("Defense", "Goals_Against_Easiness", "_defense"),
)
BAND_COLOURS = {1: "#15894f", 2: "#54c078", 3: "#b4e3a0", 4: "#f2e394", 5: "#f5a35c", 6: "#e0483e"}

TEAM_WIDTH = 0.7  # inches, as are the next three
CELL_WIDTH = 1.15
CELL_HEIGHT = 0.36
HEADER_HEIGHT = 0.75
CELL_GAP = 0.012
DPI = 200


def fetch(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def build_difficulty(codes):
    """One row per (Opponent, Venue); Venue is where the team facing that opponent plays."""
    ratings = pd.read_csv(_DATA / "team_ratings_current.csv", encoding="utf-8-sig").set_index("Team")
    current = pd.read_csv(_DATA / "players_current.csv", encoding="utf-8-sig")
    code_of = current.drop_duplicates("Team").set_index("Team")["Team_Short"]
    ratings["Code"] = ratings.index.map(code_of)
    ratings = ratings[ratings["Code"].isin(codes)]
    assert set(ratings["Code"]) == set(codes), set(codes) - set(ratings["Code"])

    attack, defence = ratings["Attack_Rating"].mean(), ratings["Defense_Rating"].mean()
    rows = []
    for row in ratings.itertuples():
        for venue, own, theirs in (("H", home_advantage, 2 - home_advantage), ("A", 2 - home_advantage, home_advantage)):
            scored = attack * (row.Defense_Rating / AVG_CONCEDED) * own
            conceded = row.Attack_Rating * (defence / AVG_CONCEDED) * theirs
            rows.append({"Opponent": row.Code, "Venue": venue, "Goals_For": scored, "Goals_Against": conceded,
                         "Goal_Diff": scored - conceded})
    table = pd.DataFrame(rows)
    assert len(table) == 2 * len(codes)
    table["Goals_Against_Easiness"] = -table["Goals_Against"]
    for _, column, _ in TICKERS:
        model = KMeans(BANDS, n_init=KMEANS_RESTARTS, random_state=KMEANS_SEED).fit(table[[column]])
        easiest_first = model.cluster_centers_.ravel().argsort()[::-1]  # k-means numbers its clusters arbitrarily
        table[f"Band_{column}"] = pd.Series(model.labels_).map({c: b for b, c in enumerate(easiest_first, start=1)})
    return table.sort_values("Goal_Diff", ascending=False).reset_index(drop=True)


def draw(path, title, gws, teams, grid, lookup, band_column):
    """Axes are in inches from the top left, so cells keep their size whatever the number of gameweeks."""
    width = TEAM_WIDTH + CELL_WIDTH * len(gws) + 0.15
    height = HEADER_HEIGHT + CELL_HEIGHT * len(teams) + 0.15
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    fig = plt.figure(figsize=(width, height), facecolor=BACKGROUND)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(height, 0))
    ax.axis("off")

    ax.text(TEAM_WIDTH, 0.27, title, color=TEXT, fontsize=15, fontweight="bold", va="center")
    for col, gw in enumerate(gws):
        ax.text(TEAM_WIDTH + CELL_WIDTH * (col + 0.5), HEADER_HEIGHT - 0.14, f"GW{gw}", color=MUTED, fontsize=10.5, fontweight="bold", ha="center", va="center")

    for row, team in enumerate(teams):
        top = HEADER_HEIGHT + CELL_HEIGHT * row
        ax.text(TEAM_WIDTH - 0.1, top + CELL_HEIGHT / 2, team, color=TEXT, fontsize=11, fontweight="bold", ha="right", va="center")
        for col, gw in enumerate(gws):
            games = grid[team, gw]
            band = min((getattr(lookup[g], band_column) for g in games), default=None)  # a double takes its easier game's colour
            colour = BAND_COLOURS.get(band, BLANK)
            ink = DARK_BAND_TEXT if band == DARK_BAND else CELL_TEXT if games else MUTED
            ax.add_patch(FancyBboxPatch(
                (TEAM_WIDTH + CELL_WIDTH * col + CELL_GAP, top + CELL_GAP), CELL_WIDTH - 2 * CELL_GAP, CELL_HEIGHT - 2 * CELL_GAP,
                boxstyle="round,pad=0,rounding_size=0.03", facecolor=colour, edgecolor="none",
            ))
            label = "  ".join(f"{opponent} ({venue})" for opponent, venue in games) or "-"
            ax.text(TEAM_WIDTH + CELL_WIDTH * (col + 0.5), top + CELL_HEIGHT / 2, label, color=ink,
                    fontsize=10.5 if len(games) < 2 else 7.5, fontweight="bold", ha="center", va="center")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BACKGROUND)
    plt.close(fig)


def main():
    fpl = fetch(f"{BASE_URL}/bootstrap-static/")
    fixtures = fetch(f"{BASE_URL}/fixtures/")
    next_gw = next(e["id"] for e in fpl["events"] if e["is_next"])
    gws = list(range(next_gw, min(next_gw + WEEKS, LAST_GW + 1)))
    code = {t["id"]: t["short_name"] for t in fpl["teams"]}

    difficulty = build_difficulty(sorted(code.values()))
    lookup = {(r.Opponent, r.Venue): r for r in difficulty.itertuples()}

    grid = {(c, gw): [] for c in code.values() for gw in gws}
    for f in fixtures:
        if f["event"] in gws:
            home, away = code[f["team_h"]], code[f["team_a"]]
            grid[home, f["event"]].append((away, "H"))
            grid[away, f["event"]].append((home, "A"))

    def run_score(team, column):
        return sum(DECAY ** (gw - next_gw) * getattr(lookup[g], column) for gw in gws for g in grid[team, gw])

    orders, paths = {}, []
    for name, column, suffix in TICKERS:
        teams = sorted(code.values(), key=lambda t: run_score(t, column), reverse=True)
        path = _IMAGES / f"fixture_ticker_GW{next_gw:02d}{suffix}.png"
        draw(path, f"Fixture ticker GW{gws[0]}-{gws[-1]} ({name})", gws, teams, grid, lookup, f"Band_{column}")
        orders[name] = ", ".join(f"{t} {run_score(t, column):+.2f}" for t in teams)
        paths.append(path)

    out_dir = _ROOT / "data" / "optimizer" / f"GW{next_gw:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    difficulty.to_csv(out_dir / "fixture_difficulty.csv", index=False, encoding="utf-8")
    counts = {gw: sum(len(grid[c, gw]) for c in code.values()) // 2 for gw in gws}
    print(json.dumps({"fixtures_per_gw": counts}))
    for name, order in orders.items():
        print(f"{name}: {order}")
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
