"""
Diagnostic: are goal/assist ratings too high after a period of inactivity? (CHANGELOG §108). Nothing here
changes a model.

Scores 60+ minute starts (players with 10+ earlier logged appearances) from a production pre-match log, grouped by
(A) cameos since the last 60+ start, (B) team matches missed just before the appearance, and (C) a gap of two or
more seasons since the previous appearance. Ratio = sum predicted / sum actual per bin; CIs by player-cluster
bootstrap on the difference from bin 0.

Run locally (after the logged run):
    python diagnose_inactivity.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.ROOT
DATA = paths.CURRENT
DIR = paths.VALIDATION / "inactivity_shrink_20261029"
LOG = paths.VALIDATION / "inactivity_shrink_20261029" / "baseline_log.csv"
KEY = ["Season", "Team", "Opponent", "Venue", "Player"]
TARGETS = {"npxG": ("npxg90", "pred_goals_90"), "xAG": ("xag90", "pred_ast_90")}
POS = {"FW": "F", "W": "M", "AM": "M", "WM": "M", "CM": "M", "DM": "M", "WB": "D", "FB": "D", "CB": "D", "GK": "G"}
N_BOOT, SEED = 5000, 42


def build():
    log = pd.read_csv(LOG, encoding="utf-8-sig")
    log["order"] = np.arange(len(log))
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    pos = pos.drop_duplicates(KEY)[KEY + ["Is_Starter", "Position_Resolved"]]
    f = log.merge(pos, on=KEY, how="left").sort_values("order").reset_index(drop=True)
    f["minutes"] = f["weight"] * 90
    f["start"] = f["Is_Starter"].astype(str).str.lower().eq("true")
    f["pos"] = f["Position_Resolved"].map(lambda s: POS.get(str(s).split(",")[0]))
    f["era"] = np.where(f["Data_Source"] == "fbref", "fbref", "FPL")
    # team fixture index within the season, in log order
    fx = f.groupby(["Season", "Team", "Opponent", "Venue"])["order"].min().reset_index()
    fx["team_idx"] = fx.sort_values("order").groupby(["Season", "Team"]).cumcount()
    f = f.merge(fx[["Season", "Team", "Opponent", "Venue", "team_idx"]], on=["Season", "Team", "Opponent", "Venue"])
    f = f.sort_values("order").reset_index(drop=True)

    g = f.groupby("Player")
    f["prior_apps"] = g.cumcount()
    full = f["start"] & (f["minutes"] >= 60)
    # (A) appearances since the last 60+ start (none yet: counted from the first appearance)
    since = np.zeros(len(f), dtype=int)
    count = {}
    for i, (p, is_full) in enumerate(zip(f["Player"], full)):
        since[i] = count.get(p, 0)
        count[p] = 0 if is_full else count.get(p, 0) + 1
    f["since_full"] = since
    # (B) team matches missed just before this appearance (same season and team)
    prev_season, prev_team, prev_idx = g["Season"].shift(), g["Team"].shift(), g["team_idx"].shift()
    same = (prev_season == f["Season"]) & (prev_team == f["Team"])
    f["missed"] = np.where(same, f["team_idx"] - prev_idx - 1, np.where(prev_season < f["Season"], f["team_idx"], np.nan))
    # (C) previous appearance two or more seasons earlier
    f["season_gap"] = f["Season"] - prev_season
    tr = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    ss = tr[pd.to_numeric(tr["Seasons_Missed"], errors="coerce").fillna(0) > 0]
    ss_keys = set(zip(ss["Player"], pd.to_numeric(ss["Season"], errors="coerce")))
    f["second_season_row"] = [(p, s) in ss_keys for p, s in zip(f["Player"], f["Season"])]

    t = f[full & (f["prior_apps"] >= 10)].copy()
    for k, (act, pred) in TARGETS.items():
        t[f"{k}_act"], t[f"{k}_pred"] = t[act] * t["weight"], t[pred] * t["weight"]
    t["A"] = pd.cut(t["since_full"], [-1, 0, 4, 9, 10**6], labels=["0", "1-4", "5-9", "10+"])
    t["B"] = pd.cut(t["missed"], [-1, 0, 3, 9, 10**6], labels=["0", "1-3", "4-9", "10+"])
    t["C"] = np.select([t["season_gap"] >= 2, t["season_gap"].notna()],
                       [np.where(t["second_season_row"], "gap2+_with_row", "gap2+_no_row"), "0-1"], "first")
    med = t.groupby(["era", "pos"])["NPG_Rating_Pre"].transform("median")
    t["high_rated"] = t["NPG_Rating_Pre"] > med
    return f, t


def compare(t, col, base):
    out = {}
    for b, grp in t.groupby(col, observed=True):
        out[str(b)] = {"rows": int(len(grp)), "players": int(grp["Player"].nunique())}
        for k in TARGETS:
            out[str(b)][f"{k}_ratio"] = round(float(grp[f"{k}_pred"].sum() / grp[f"{k}_act"].sum()), 3)
        if str(b) == base:
            continue
        sub = t[t[col].astype(str).isin([base, str(b)])]
        cols = []
        for k in TARGETS:
            for lab in (base, str(b)):
                m = sub[col].astype(str) == lab
                cols += [sub[f"{k}_pred"].where(m, 0), sub[f"{k}_act"].where(m, 0)]
        agg = pd.concat(cols, axis=1).groupby(sub["Player"]).sum().to_numpy()
        idx = np.random.default_rng(SEED).integers(0, len(agg), size=(N_BOOT, len(agg)))
        s = agg[idx].sum(axis=1)
        for j, k in enumerate(TARGETS):
            o = 4 * j
            diff = s[:, o + 2] / s[:, o + 3] - s[:, o] / s[:, o + 1]
            out[str(b)][f"{k}_diff_vs_{base}_ci95"] = np.percentile(diff, [2.5, 97.5]).round(3).tolist()
    return out


def main():
    f, t = build()
    out = {"log_rows": int(len(f)), "start_flag_coverage": float(f["Is_Starter"].notna().mean()), "target_rows": int(len(t)),
           "season_monotonic_per_player": bool((f.groupby("Player")["Season"].diff().fillna(0) >= 0).all())}
    for col, base in (("A", "0"), ("B", "0"), ("C", "0-1")):
        out[col] = {"all": compare(t, col, base),
                    "high_rated": compare(t[t["high_rated"]], col, base),
                    **{f"era_{e}": compare(g, col, base) for e, g in t.groupby("era")}}
    named = f[f["Player"].isin(["Taiwo Awoniyi", "Callum Wilson"])]
    out["named_last_rows"] = named.groupby("Player").tail(3)[["Player", "Season", "Team", "minutes", "start",
                                                              "NPG_Rating_Pre", "since_full", "missed"]].round(3).to_dict("records")
    dec = {}
    for col, b in (("A", "10+"), ("B", "10+"), ("C", "gap2+_no_row"), ("C", "gap2+_with_row")):
        r = out[col]["all"].get(b)
        if r:
            base = out[col]["all"]["0" if col != "C" else "0-1"]["npxG_ratio"]
            dec[f"{col}:{b}"] = {"npxG_ratio_vs_base_pct": round(100 * (r["npxG_ratio"] / base - 1), 1),
                                 "ci_excludes_zero": r["npxG_diff_vs_" + ("0" if col != "C" else "0-1") + "_ci95"][0] > 0,
                                 "goes_forward": r["npxG_ratio"] / base - 1 >= 0.05 and r["npxG_diff_vs_" + ("0" if col != "C" else "0-1") + "_ci95"][0] > 0}
    out["decision"] = dec
    (DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
