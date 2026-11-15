"""Defender experience bands on real FPL hits (CHANGELOG §192). Protocol: data/validation/defender_experience_20261102/protocol.json.
Reads item 6's arm A predictions. No production writes. Run: python -u -X utf8 scripts/defcon/diagnose_defender_experience.py
"""
from fpl_model import paths
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = paths.ROOT
from fpl_model.defcon_scaling import DEFENDER_POINTS_FACTOR, nb_sf, prob_at_minutes

SRC = paths.VALIDATION / "defcon_team_aggregates_20261102" / "predictions_D.csv.gz"
OUT = paths.VALIDATION / "defender_experience_20261102"
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]
N_BOOT, SEED, THR = 5000, 175, 10


def ratio(g, y, f):
    u, inv = np.unique(g.Player.values, return_inverse=True)
    a, b = np.bincount(inv, g[y].values, len(u)), np.bincount(inv, g[f].values, len(u))
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    v = a[ix].sum(1) / b[ix].sum(1)
    return dict(rows=len(g), players=len(u), hits=round(float(a.sum()), 1), forecast=round(float(b.sum()), 1), ratio=round(float(a.sum() / b.sum()), 3),
                lo=round(float(np.percentile(v, 2.5)), 3), hi=round(float(np.percentile(v, 97.5)), 3))


def tab(d, y, f):
    out = []
    for (e, a), g in d.groupby(["exp", "act"], observed=True):
        out.append({"exp": str(e), "act": str(a), **ratio(g, y, f)})
    for e, g in d.groupby("exp", observed=True):
        out.append({"exp": str(e), "act": "all", **ratio(g, y, f)})
    mid = d[d.exp.astype(str).isin(["51-150", "151-250"])]
    out.append({"exp": "51-250", "act": "all", **ratio(mid, y, f)})
    out.append({"exp": "all", "act": "all", **ratio(d, y, f)})
    return out


def main():
    v = pd.read_csv(SRC, encoding="utf-8-sig", low_memory=False)
    v["exp"] = pd.cut(v.MatchesPlayed, [-1, 50, 150, 250, 10000], labels=["<=50", "51-150", "151-250", ">250"])
    v["act"] = pd.cut(v.AvgPer90, [-1, 8, 10, 1000], right=False, labels=["<8", "8-10", ">=10"])
    p = pd.read_csv(ROOT / "data" / "current" / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p.Data_Source == "FPL_API"][KEY + ["Defensive_Contribution"]].drop_duplicates(KEY)
    r = v[v.Season >= 2025].drop_duplicates(KEY).merge(p, on=KEY, how="inner", validate="one_to_one")
    r["hit"] = (r.Defensive_Contribution >= THR).astype(float)
    grid = np.linspace(.05, 40, 800)
    mu = np.interp(r.p_A.values, nb_sf(grid, THR - 1), grid) * DEFENDER_POINTS_FACTOR
    r["f"] = np.asarray(prob_at_minutes(mu, r.Minutes.values, THR), dtype=float)
    res = {"real_60plus": tab(r[r.Minutes >= 60], "hit", "f"), "real_all_minutes": tab(r, "hit", "f"),
           "real_2026_60plus": tab(r[(r.Minutes >= 60) & (r.Season == 2026)], "hit", "f"),
           "target_2022_24": tab(v[v.Season <= 2024], "Target", "p_A")}
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    for k, t in res.items():
        print(k)
        print(pd.DataFrame(t).to_string())


if __name__ == "__main__":
    main()
