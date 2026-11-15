"""§153: scale each head's implied 90-minute defcon mean so scaled probabilities match FPL's real Defensive_Contribution hits.
One factor per head, fitted on one FPL-era season's held-out rows, tested on the other. No production writes.
Run: python -u -X utf8 scripts/defcon/evaluate_points_calibration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from defcon_scaling import nb_sf, prob_at_minutes

OUT = HERE.parents[1] / "data" / "validation" / "defcon_overnight_20261031"
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]
HEADS = {"D": ("p_control", 10), "M": ("p_count", 12), "F": ("p_control", 12)}


def rows(h):
    col, thr = HEADS[h]
    p = pd.read_csv(ROOT / "data/current/player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    p = p[p.Data_Source == "FPL_API"][KEY + ["Defensive_Contribution"]].drop_duplicates(KEY)
    d = pd.read_csv(OUT / f"item2_predictions_{h}.csv", encoding="utf-8-sig")
    d = d[d.Season >= 2025].drop_duplicates(KEY).merge(p, on=KEY, how="inner", validate="one_to_one")
    grid = np.linspace(.05, 40, 800)
    d["mu"] = np.interp(d[col].values, nb_sf(grid, thr - 1), grid)
    d["hit"] = (d.Defensive_Contribution >= thr).astype(float)
    return d, thr


def prob(d, thr, c):
    return np.clip(np.asarray(prob_at_minutes(d.mu.values * c, d.Minutes.values, thr), dtype=float), 1e-6, 1 - 1e-6)


def fit(d, thr):
    return brentq(lambda c: prob(d, thr, c).mean() - d.hit.mean(), .2, 3)


def boot(d, p0, p1, n=2000):
    y = d.hit.values
    x = -(y * (np.log(p1) - np.log(p0)) + (1 - y) * (np.log1p(-p1) - np.log1p(-p0)))
    codes, _ = pd.factorize(d.Player)
    k = codes.max() + 1
    s, c = np.bincount(codes, x, k), np.bincount(codes, minlength=k).astype(float)
    w = np.random.default_rng(0).multinomial(k, np.ones(k) / k, size=n)
    b = (w * s).sum(1) / (w * c).sum(1)
    return float(x.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


out = {}
for h in HEADS:
    d, thr = rows(h)
    res = {}
    for train, test in ((2025, 2026), (2026, 2025)):
        tr, te = d[d.Season == train], d[d.Season == test]
        c = fit(tr, thr)
        p0, p1 = prob(te, thr, 1.0), prob(te, thr, c)
        ch = boot(te, p0, p1)
        res[f"fit{train}_test{test}"] = {"c": c, "n": len(te), "hits": int(te.hit.sum()), "ll0": log_loss(te.hit, p0), "ll1": log_loss(te.hit, p1),
                                         "d_ll": ch[0], "lo": ch[1], "hi": ch[2], "mean_p0": p0.mean(), "mean_p1": p1.mean(), "actual": te.hit.mean(),
                                         "pts_mse0": float(((2 * p0 - 2 * te.hit) ** 2).mean()), "pts_mse1": float(((2 * p1 - 2 * te.hit) ** 2).mean())}
    res["all_seasons_c"] = fit(d, thr)
    out[h] = res
    print(h, json.dumps(res, indent=0, default=float).replace("\n", " "))
(OUT / "item4b_calibration.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
