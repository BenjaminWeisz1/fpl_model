"""Correlated scorelines (CHANGELOG §157). Walk-forward Dixon-Coles rho on pre-match forecasts; no production writes.
python -u -X utf8 scripts/team_ratings/validate_dixon_coles.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

ROOT = paths.ROOT
PEN = 1 + 792 / 7734 * 0.79


def tau(h, a, lh, la, rho):
    t = np.ones(len(h))
    t = np.where((h == 0) & (a == 0), 1 - lh * la * rho, t)
    t = np.where((h == 0) & (a == 1), 1 + lh * rho, t)
    t = np.where((h == 1) & (a == 0), 1 + la * rho, t)
    t = np.where((h == 1) & (a == 1), 1 - rho, t)
    return np.clip(t, 1e-6, None)


def ll(d, rho):
    p = poisson.pmf(d.Home_Goals, d.lh) * poisson.pmf(d.Away_Goals, d.la)
    return -(np.log(p) + np.log(tau(d.Home_Goals.values, d.Away_Goals.values, d.lh.values, d.la.values, rho))).mean()


def main():
    h = pd.read_csv(ROOT / "data" / "current" / "team_ratings_history.csv", encoding="utf-8-sig")
    h["lh"], h["la"] = h.Home_Prediction * PEN, h.Away_Prediction * PEN
    h = h[h.Season >= 2018]
    rows = []
    for s in range(2021, 2027):
        tr, te = h[h.Season < s], h[h.Season == s]
        r = minimize_scalar(lambda x: ll(tr, x), bounds=(-.3, .3), method="bounded").x
        rows.append({"Season": s, "rho": r, "ll0": ll(te, 0.0), "ll1": ll(te, r), "n": len(te)})
    t = pd.DataFrame(rows)
    print(t.round(5).to_string())
    res = {"pooled_d_ll": float(((t.ll1 - t.ll0) * t.n).sum() / t.n.sum())}
    full = minimize_scalar(lambda x: ll(h, x), bounds=(-.3, .3), method="bounded").x
    grid = np.linspace(-.2, .2, 161)
    dev = np.array([2 * len(h) * (ll(h, g) - ll(h, full)) for g in grid])
    inside = grid[dev <= 3.84]
    res.update({"rho_full": float(full), "rho_ci": [float(inside.min()), float(inside.max())], "seasons": rows})
    z = (h.Home_Goals == 0) & (h.Away_Goals == 0)
    pz = np.exp(-h.lh - h.la)
    res["nil_nil"] = {"actual": float(z.mean()), "independent": float(pz.mean()), "with_rho": float((pz * (1 - h.lh * h.la * full)).mean())}
    (ROOT / "data" / "validation" / "dixon_coles_20261031.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "seasons"})


if __name__ == "__main__":
    main()
