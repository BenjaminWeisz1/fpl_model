"""
Diagnostic: which expected-assists source predicts future assists best? (CHANGELOG §111). Nothing here
changes a model.

Per era (fbref 2017-2024, FPL 2025+), each played player-match whose player has 900+ earlier minutes in the
same era over the current and previous season is predicted from his per-90 rate over those appearances, for
native xA (fbref xAG / FPL expected assists), Understat xA and a 50/50 blend, each with one calibration
constant. Scored by MSE against conventional assists (Understat) and FPL assists.

Run locally:
    python compare_xa_sources.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.ROOT
DATA = paths.CURRENT
OUT = paths.VALIDATION / "xa_sources_20261029"
K = ["Season", "Team", "Opponent", "Venue"]
MIN_PRIOR = 900
N_BOOT, SEED = 5000, 42


def load():
    ps = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    ps["Season"] = np.where(ps["Data_Source"] == "fbref", 2017 + ps["MatchID"] // 380, ps["Season"])
    ps = ps[ps["Minutes"] > 0]
    pos = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    pos = pos.drop_duplicates(K + ["Player"])[K + ["Player", "Player_ID_Understat"]]
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False)
    u = u[K + ["Player_ID_Understat", "Date", "Assists", "Understat_xA"]].rename(columns={"Assists": "conv_ast"})
    m = ps.merge(pos, on=K + ["Player"], how="left").merge(u, on=K + ["Player_ID_Understat"], how="inner")
    m["era"] = np.where(m["Data_Source"] == "fbref", "fbref", "FPL")
    m = m.rename(columns={"xAG": "native_xa", "Assists": "fpl_ast"})
    m["native_xa"] = m["native_xa"].fillna(0)
    return m.sort_values(["Date", "MatchID"]).reset_index(drop=True)


def prior_rates(g):
    """Per-90 rates from each player's earlier appearances in the current and previous season (same era)."""
    rows = []
    hist = {}
    for r in g.itertuples(index=False):
        h = hist.setdefault(r.Player, [])
        prev = [x for x in h if x[0] >= r.Season - 1]
        mins = sum(x[1] for x in prev)
        if mins >= MIN_PRIOR:
            rows.append((r.Index_, mins, sum(x[2] for x in prev) / mins * 90, sum(x[3] for x in prev) / mins * 90))
        h.append((r.Season, r.Minutes, r.native_xa, r.Understat_xA))
    return pd.DataFrame(rows, columns=["Index_", "prior_mins", "rate_native", "rate_understat"]).set_index("Index_")


def score(t, target):
    out, preds = {}, {}
    for arm in ("native", "understat"):
        raw = t[f"rate_{arm}"] * t["Minutes"] / 90
        c = t[target].sum() / raw.sum()
        preds[arm] = raw * c
        out[f"{arm}_constant"] = round(float(c), 4)
    preds["blend"] = (preds["native"] + preds["understat"]) / 2
    se = {a: (p - t[target]) ** 2 for a, p in preds.items()}
    for a in preds:
        out[f"mse_{a}"] = float(se[a].mean())
    g = pd.DataFrame(se).groupby(t["Player"]).sum()
    arr = g[["native", "understat", "blend"]].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(arr), size=(N_BOOT, len(arr)))
    s = arr[idx].sum(axis=1)
    for j, a in ((1, "understat"), (2, "blend")):
        rel = 100 * (s[:, j] / s[:, 0] - 1)
        out[f"{a}_vs_native_pct"] = round(float(100 * (se[a].mean() / se["native"].mean() - 1)), 3)
        out[f"{a}_vs_native_ci95"] = np.percentile(rel, [2.5, 97.5]).round(3).tolist()
    rel = 100 * (s[:, 2] / s[:, 1] - 1)
    out["blend_vs_understat_ci95"] = np.percentile(rel, [2.5, 97.5]).round(3).tolist()
    return out


def main():
    m = load()
    out = {"rows_joined": int(len(m))}
    for era, g in m.groupby("era"):
        g = g.reset_index().rename(columns={"index": "Index_"})
        t = g.set_index("Index_").join(prior_rates(g), how="inner")
        t = t[t["conv_ast"].notna()]
        res = {"rows": int(len(t)), "players": int(t["Player"].nunique()),
               "conventional_assists": score(t, "conv_ast")}
        if era == "FPL":
            res["fpl_assists"] = score(t, "fpl_ast")
        out[era] = res
    p, s = out["FPL"]["conventional_assists"], out["FPL"]["fpl_assists"]
    out["decision"] = {a: {"primary_better_ci_excludes_zero": p[f"{a}_vs_native_ci95"][1] < 0,
                           "fpl_assists_not_worse": s[f"{a}_vs_native_pct"] <= 0} for a in ("understat", "blend")}
    for a in out["decision"]:
        out["decision"][a]["goes_forward"] = all(out["decision"][a].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
