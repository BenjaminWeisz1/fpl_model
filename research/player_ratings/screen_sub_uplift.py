"""
Screen: substitute uplift in per-90 npxG/xAG (diagnostic, CHANGELOG §100). Nothing here changes a model.

Players who come off the bench face tired defences and open game states, so their per-90 output
in cameos may not carry over to starts. For each appearance type this compares the actual npxG
(xAG) with what the same player would have produced at his own per-90 rate in full starts
(85+ minutes): uplift = sum actual / sum (player's start rate x minutes / 90), over players with
at least 900 start minutes. Types: substitute appearances by minutes band, starters taken off
early, starters playing 60-84. Start/sub from player_match_positions.csv (Understat Is_Starter).

Run locally:
    python screen_sub_uplift.py
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.VALIDATION / "sub_uplift_20261028"
KEY = ["Season", "Team", "Opponent", "Venue", "Player"]


def load():
    ps = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    ps["Season"] = np.where(ps["Data_Source"] == "fbref", 2017 + ps["MatchID"] // 380, ps["Season"])
    ps = ps[ps["Minutes"] > 0]
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    pos = pos.drop_duplicates(KEY)[KEY + ["Is_Starter", "Position_Resolved"]]
    m = ps.merge(pos, on=KEY, how="inner")
    m["Is_Starter"] = m["Is_Starter"].astype(str).str.lower().eq("true")
    m["npxG"], m["xAG"] = m["npxG"].fillna(0), m["xAG"].fillna(0)
    return m, len(ps)


def appearance_type(m):
    mins, start = m["Minutes"], m["Is_Starter"]
    return np.select(
        [start & (mins >= 85), start & (mins >= 60), start, ~start & (mins <= 15), ~start & (mins <= 30), ~start & (mins <= 45), ~start],
        ["start_full", "start_60_84", "start_off_before_60", "sub_1_15", "sub_16_30", "sub_31_45", "sub_46_plus"],
        default="other")


def main():
    m, n_rows = load()
    m["type"] = appearance_type(m)
    full = m[m["type"] == "start_full"].groupby("Player").agg(mins=("Minutes", "sum"), npxG=("npxG", "sum"), xAG=("xAG", "sum"))
    full = full[full["mins"] >= 900]
    rate = (full[["npxG", "xAG"]].div(full["mins"], axis=0) * 90)
    r = m[m["Player"].isin(rate.index)].copy()
    for t in ("npxG", "xAG"):
        r[f"exp_{t}"] = r["Player"].map(rate[t]) * r["Minutes"] / 90
    pos_map = {"FW": "F", "W": "M", "AM": "M", "WM": "M", "CM": "M", "DM": "M", "WB": "D", "FB": "D", "CB": "D", "GK": "G"}
    r["Pos"] = r["Position_Resolved"].map(lambda s: pos_map.get(str(s).split(",")[0], None))
    out = {"rows_with_start_flag": int(len(m)), "rows_total": n_rows, "players_with_900_full_start_minutes": int(len(rate)),
           "by_type": {}, "by_type_and_position": {}, "by_type_and_era": {}}
    agg = lambda g: {"rows": int(len(g)), "minutes": int(g["Minutes"].sum()),
                     "npxG_uplift": round(float(g["npxG"].sum() / g["exp_npxG"].sum()), 3),
                     "xAG_uplift": round(float(g["xAG"].sum() / g["exp_xAG"].sum()), 3)}
    for t, g in r.groupby("type"):
        out["by_type"][t] = agg(g)
    for (t, p), g in r[r["Pos"].isin(["D", "M", "F"])].groupby(["type", "Pos"]):
        out["by_type_and_position"][f"{t}|{p}"] = agg(g)
    r["era"] = np.where(r["Data_Source"] == "fbref", "fbref", "FPL")
    for (t, e), g in r.groupby(["type", "era"]):
        out["by_type_and_era"][f"{t}|{e}"] = agg(g)
    # bootstrap over players for the pooled substitute uplift (all sub bands)
    sub = r[~r["Is_Starter"]]
    g = sub.groupby("Player")[["npxG", "exp_npxG", "xAG", "exp_xAG"]].sum().to_numpy()
    idx = np.random.default_rng(42).integers(0, len(g), size=(5000, len(g)))
    s = g[idx].sum(axis=1)
    out["sub_pooled"] = {"players": int(len(g)), "npxG_uplift": round(float(g[:, 0].sum() / g[:, 1].sum()), 3),
                         "npxG_ci95": np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5]).round(3).tolist(),
                         "xAG_uplift": round(float(g[:, 2].sum() / g[:, 3].sum()), 3),
                         "xAG_ci95": np.percentile(s[:, 2] / s[:, 3], [2.5, 97.5]).round(3).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("rows_with_start_flag", "rows_total", "players_with_900_full_start_minutes", "by_type", "sub_pooled")}, indent=2))
    print(pd.DataFrame(out["by_type_and_position"]).T.to_string())
    print(pd.DataFrame(out["by_type_and_era"]).T.to_string())


if __name__ == "__main__":
    main()
