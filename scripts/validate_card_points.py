"""Card forecasts: walk-forward test (CHANGELOG §175).

Protocol, frozen before any forecast was scored: data/validation/card_points_20261101/protocol.json.
Phases:  tune (seasons to 2023)  ->  test (2024-2026, once).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from card_points import DATA, build_frame, decayed_multiplier, trailing_level

OUT = DATA.parent / "validation" / "card_points_20261101"
KS = [2, 5, 10, 20, 40, 80]
HLS = [None, 12, 25, 50, 100, 200]
KFS = [5, 10, 20, 40, 80]
EXPS = [0, .25, .5, .75, 1]
N_BOOT, SEED = 5000, 175


def prepare():
    d = build_frame()
    d["_all"] = "all"
    d["lvl_y"] = trailing_level(d, "yellow")
    d["lvl_r"] = trailing_level(d, "red")
    d["lvl_f"] = trailing_level(d, "fouls")
    d["flat_y"] = trailing_level(d, "yellow", by=("_all",))
    d["flat_r"] = trailing_level(d, "red", by=("_all",))
    d["e_y"] = d.lvl_y * d.n90
    d["e_f"] = d.lvl_f * d.n90
    h = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Home", "Away", "Home_Prediction", "Away_Prediction"]).drop_duplicates(["Season", "Home", "Away"])
    home = d.Venue == "Home"
    d["_h"], d["_a"] = d.Team.where(home, d.Opponent), d.Opponent.where(home, d.Team)
    d = d.merge(h, left_on=["Season", "_h", "_a"], right_on=["Season", "Home", "Away"], how="left")
    d["gap"] = np.where(d.Venue == "Home", d.Home_Prediction - d.Away_Prediction, d.Away_Prediction - d.Home_Prediction)
    print(f"rows {len(d)}; fixture gap coverage {d.gap.notna().mean():.3f}; fouls coverage {d.fouls.notna().mean():.3f}")
    d["gap"] = d.gap.fillna(0)
    d["away"] = (d.Venue == "Away").astype(float)
    return d


def pts(y_rate, r_rate, n90):
    return -(y_rate + 3 * r_rate) * n90


def mse(pred, a):
    return float(((a - pred) ** 2).mean())


def tune():
    d = prepare()
    tr = d.Season.between(2018, 2023).values
    a = d.pts.values
    res = {}
    best = {}
    for k in KS:
        m = decayed_multiplier(d, "yellow", "e_y", k)
        best[k] = mse(pts(d.lvl_y * m, d.lvl_r, d.n90)[tr], a[tr])
    res["k_grid"] = best
    k = min(best, key=best.get)
    h_scores = {}
    for h in HLS:
        m = decayed_multiplier(d, "yellow", "e_y", k, h)
        h_scores[str(h)] = mse(pts(d.lvl_y * m, d.lvl_r, d.n90)[tr], a[tr])
    res["half_life_grid"] = h_scores
    hl = min(h_scores, key=h_scores.get)
    h = None if hl == "None" else int(hl)
    m = decayed_multiplier(d, "yellow", "e_y", k, h).clip(1e-3)
    f_scores = {}
    for kf in KFS:
        mf = decayed_multiplier(d, "fouls", "e_f", kf).clip(1e-3)
        for x in EXPS:
            for y in EXPS:
                f_scores[f"{kf}|{x}|{y}"] = mse(pts(d.lvl_y * m ** x * mf ** y, d.lvl_r, d.n90)[tr], a[tr])
    fk = min(f_scores, key=f_scores.get)
    kf, ex, ey = [float(v) for v in fk.split("|")]
    res["foul_best"] = dict(kf=kf, a=ex, b=ey, mse=f_scores[fk], no_fouls=h_scores[hl])
    mf = decayed_multiplier(d, "fouls", "e_f", int(kf)).clip(1e-3)
    base = (d.lvl_y * m ** ex * mf ** ey * d.n90).values[tr]
    y, aw, gp = d.yellow.values[tr], d.away.values[tr], d.gap.values[tr]

    def nll(c):
        mu = base * np.exp(c[0] * aw + c[1] * gp - c[2])
        return (mu - y * np.log(mu)).mean()

    c = minimize(nll, [0, 0, 0], method="BFGS").x
    # The intercept keeps the level; its value is frozen with the other two.
    res.update(k=k, half_life=h, kf=int(kf), a=ex, b=ey, c_away=float(c[0]), c_gap=float(c[1]), c_0=float(c[2]))
    (OUT / "tuned.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


def cluster(prev, new, a, pl):
    se0, se1 = (a - prev) ** 2, (a - new) ** 2
    u, inv = np.unique(pl, return_inverse=True)
    t0, t1 = np.bincount(inv, se0), np.bincount(inv, se1)
    ix = np.random.default_rng(SEED).integers(0, len(u), size=(N_BOOT, len(u)))
    v = t1[ix].sum(1) / t0[ix].sum(1) - 1
    return round(float(se1.sum() / se0.sum() - 1) * 100, 3), [round(float(x) * 100, 3) for x in np.percentile(v, [2.5, 97.5])]


def test():
    t = json.loads((OUT / "tuned.json").read_text(encoding="utf-8"))
    d = prepare()
    k, h = t["k"], t["half_life"]
    mb = decayed_multiplier(d, "yellow", "e_y", k).clip(1e-3)
    mr = decayed_multiplier(d, "yellow", "e_y", k, h).clip(1e-3)
    mf = decayed_multiplier(d, "fouls", "e_f", t["kf"]).clip(1e-3)
    n = d.n90
    rate = {"F": d.flat_y, "P": d.lvl_y, "B": d.lvl_y * mb, "R": d.lvl_y * mr, "O": d.lvl_y * mr ** t["a"] * mf ** t["b"]}
    rate["X"] = rate["O"] * np.exp(t["c_away"] * d.away + t["c_gap"] * d.gap - t["c_0"])
    pred = {s: pts(rate[s], d.flat_r if s == "F" else d.lvl_r, n).values for s in rate}
    te = (d.Season >= 2024).values
    a, pl = d.pts.values, d.Player.values
    w24, w25 = (d.Season == 2024).values, (d.Season >= 2025).values
    res = {"rows": int(te.sum()), "players": int(d[te].Player.nunique()), "steps": {}}
    kept = "F"
    for s in ["P", "B", "R", "O", "X"]:
        ch, ci = cluster(pred[kept][te], pred[s][te], a[te], pl[te])
        c24 = (mse(pred[s][w24], a[w24]) / mse(pred[kept][w24], a[w24]) - 1) * 100
        c25 = (mse(pred[s][w25], a[w25]) / mse(pred[kept][w25], a[w25]) - 1) * 100
        ok = ci[1] < 0 and c24 < 0 and c25 < 0
        res["steps"][s] = dict(against=kept, change_pct=ch, ci=ci, s2024=round(c24, 3), s2025_26=round(c25, 3), kept=bool(ok))
        print(s, res["steps"][s])
        if ok:
            kept = s
    res["final"] = kept
    res["vs_flat"] = cluster(pred["F"][te], pred[kept][te], a[te], pl[te])
    print("final", kept, "v flat", res["vs_flat"], "| mean pts actual", round(a[te].mean(), 4), "forecast", round(pred[kept][te].mean(), 4))

    y = d.yellow.values[te]
    for s in ("F", kept):
        p = np.clip((rate[s] * n).values[te], 1e-4, .9)
        res[f"yellow_ll_{s}"] = round(float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), 5)
    q = pd.DataFrame(dict(p=(rate[kept]).values[te], n=n.values[te], y=y))
    q["bin"] = pd.qcut(q.p, 10, labels=False, duplicates="drop")
    g = q.groupby("bin").apply(lambda x: pd.Series(dict(rows=len(x), forecast_90=x.p.mean(), actual_90=x.y.sum() / x.n.sum(),
                                                       ratio=x.y.sum() / (x.p * x.n).sum())), include_groups=False).round(3)
    print(g.to_string())
    res["deciles"] = g.reset_index().to_dict("records")
    q["band"] = pd.cut(q.n * 90, [0, 30, 60, 89, 90, 200], labels=["1-30", "31-60", "61-89", "90", "90+"])
    b = q.groupby("band", observed=True).apply(lambda x: pd.Series(dict(rows=len(x), ratio=x.y.sum() / (x.p * x.n).sum())), include_groups=False).round(3)
    print(b.to_string())
    res["minutes_bands"] = b.reset_index().to_dict("records")
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    {"tune": tune, "test": test}[sys.argv[1]]()
