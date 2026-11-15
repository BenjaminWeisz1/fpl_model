"""CHANGELOG §161 amendment, Arm C: a slow team-level ratio from past matches only, with diagnostics of Arm A and A-pre.

Test only; no production writes.
Run: python -u -X utf8 scripts/assemble/validate_team_coherence_trailing.py
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from research.assemble import validate_team_coherence as V

K, WINDOW = 8, 19


def trailing(t, order):
    """R for each team-match: past log(T/S) in this season, shrunk toward zero. Empty past gives zero."""
    x = t.merge(order, on="Match_Key", how="left", validate="many_to_one").sort_values(["Season", "Team", "Order"]).copy()
    x["r"], x["ra"] = np.log(x["T"] / x.S), np.log(x["T"] / x.Sa)
    for col, out in (("r", "R"), ("ra", "Ra")):
        g = x.groupby(["Season", "Team"])[col]
        x[out] = g.transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).sum()).fillna(0)
    x["n"] = x.groupby(["Season", "Team"]).cumcount().clip(upper=WINDOW)
    x["R"], x["Ra"] = x.R / (x.n + K), x.Ra / (x.n + K)
    return x


def fit(t, y, s, r):
    m = sm.GLM(t[y], sm.add_constant(t[r].rename("g")), family=sm.families.Poisson(), offset=np.log(t[s])).fit()
    return float(m.params["const"]), float(m.params["g"]), float(m.bse["g"])


def match_order(f):
    """Rounds where known; the 2025 windows have none, so the file order of their first rows stands in."""
    o = f.assign(i=np.arange(len(f))).groupby("Match_Key").agg(Order=("Round", "first"), i=("i", "first")).reset_index()
    o["Order"] = o.Order.fillna(o.i / 1e6)
    return o[["Match_Key", "Order"]]


def split(x, lam, alam):
    y1, y2 = x.NPG * x.gp, x.Ast * 3
    a0, a1 = V.assist_level(x, "alam") * 3, V.assist_level(x, alam) * 3
    return {"goals_pct": float(100 * (((y1 - x[lam] * x.gp) ** 2).mean() / ((y1 - x.lam * x.gp) ** 2).mean() - 1)),
            "assists_pct": float(100 * (((y2 - a1) ** 2).mean() / ((y2 - a0) ** 2).mean() - 1))}


def matched_mse(v, col, base, arm):
    a = v[col].mean() / v[arm].mean()
    b = v[col].mean() / v[base].mean()
    return {"inc": float(((v[col] - v[base] * b) ** 2).mean()), "arm": float(((v[col] - v[arm] * a) ** 2).mean()), "rows": len(v)}


def main():
    pd.set_option("display.width", 220, "display.max_columns", 40)
    f = V.frame()
    t = trailing(V.team_table(f), match_order(f))
    h = t[t.Block == "2018-2024"]
    fits, rows = {}, []
    for s in V.TEST:
        tr = h[h.Season < s]
        fits[s] = fit(tr, "G", "S", "R")[:2] + fit(tr, "A", "Sa", "Ra")[:2]
        rows.append({"Season": s, "c": fits[s][0], "gamma": fits[s][1], "c_ast": fits[s][2], "gamma_ast": fits[s][3]})
    all_fit = fit(h, "G", "S", "R")[:2] + fit(h, "A", "Sa", "Ra")[:2]
    fits[2025] = fits[2026] = all_fit
    rows.append({"Season": "2025-26", "c": all_fit[0], "gamma": all_fit[1], "c_ast": all_fit[2], "gamma_ast": all_fit[3]})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    x = f[f.Season >= 2019].merge(t[["Match_Key", "Team", "T", "S", "Sa", "R", "Ra"]], on=["Match_Key", "Team"], how="left", validate="many_to_one").reset_index(drop=True)
    x["lam_c"] = x.lam * np.exp(x.Season.map(lambda s: fits[s][0]) + x.Season.map(lambda s: fits[s][1]) * x.R)
    x["alam_c"] = x.alam * np.exp(x.Season.map(lambda s: fits[s][2]) + x.Season.map(lambda s: fits[s][3]) * x.Ra)
    x["Q"] = pd.qcut(x.Team_Pred_NPG, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    x["cluster"] = x.Season.astype(str) + x.Team

    keep = V.pre_match_sum(f[f.Season <= 2024])[["Match_Key", "Team"]]
    o = x[x.Season <= 2024].merge(keep, on=["Match_Key", "Team"], how="inner", validate="many_to_one").reset_index(drop=True)
    pooled = V.score(o, clusters=o.cluster)[0]
    by = {int(s): V.score(g.reset_index(drop=True))[0] for s, g in o.groupby("Season")}
    print("pooled", pooled)
    print(pd.DataFrame(by).T[["rows", "mse_inc", "mse_arm", "pct"]].round(4))
    print("split", split(o, "lam_c", "alam_c"))
    xg = {"npxG": matched_mse(o[o.npxG.notna()], "npxG", "lam", "lam_c"), "xAG": matched_mse(o[o.xAG.notna()], "xAG", "alam", "alam_c")}
    print(xg)
    g0, g1 = V.gap(o, "lam", "alam"), V.gap(o, "lam_c", "alam_c")
    print(pd.DataFrame({"incumbent": g0, "arm": g1}).round(4))
    pos = {}
    for p, g in o.groupby("Pos"):
        g = g.reset_index(drop=True)
        y = g.NPG * g.gp + g.Ast * 3
        pos[p] = {"inc": float((y - (g.lam * g.gp + V.assist_level(g, "alam") * 3)).mean()), "arm": float((y - (g.lam_c * g.gp + V.assist_level(g, "alam_c") * 3)).mean())}
    print(pos)
    rec = x[x.Season >= 2025].reset_index(drop=True)
    recent = V.score(rec, clusters=rec.cluster)[0]
    print("2025-26", recent)
    print(pd.DataFrame({"incumbent": V.gap(rec, "lam", "alam"), "arm": V.gap(rec, "lam_c", "alam_c")}).round(4))
    rule = {"i": pooled["hi"] < 0, "ii": sum(v["diff"] < 0 for v in by.values()) >= 4, "iii": all(v["arm"] <= v["inc"] for v in xg.values()),
            "iv": (g1.Q4 - g1.Q1) <= .5 * (g0.Q4 - g0.Q1), "v": all(abs(v["arm"]) - abs(v["inc"]) <= .02 for v in pos.values()), "vii": recent["diff"] <= 0}
    rule = {k: bool(v) for k, v in rule.items()}
    rule["adopt_worthy"] = all(rule.values())
    res = {"fits": rows, "pooled": pooled, "by_season": by, "xg": xg, "gap_inc": g0.to_dict(), "gap_arm": g1.to_dict(), "position": pos, "recent": recent, "rule": rule}
    (V.OUT / "results_trailing.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rule, indent=1))

    # Diagnostics of the first run, no decision attached.
    z = V.apply_arm(f[(f.Season >= 2019) & (f.Season <= 2024)].reset_index(drop=True), V.team_table(f), {s: fits_a(h0 := V.team_table(f), s) for s in V.TEST})
    print("Arm A split", split(z, "lam_c", "alam_c"))
    print("Arm A level-matched", {"npxG": matched_mse(z[z.npxG.notna()], "npxG", "lam", "lam_c"), "xAG": matched_mse(z[z.xAG.notna()], "xAG", "alam", "alam_c")})
    print("xAG mean", round(float(z.xAG.mean()), 4), "alam", round(float(z.alam.mean()), 4), "alam_c", round(float(z.alam_c.mean()), 4))


def fits_a(t, s):
    h = t[(t.Block == "2018-2024") & (t.Season < s)]
    c, g, _ = V.fit_coherence(h, "G", "S")
    ca, ga, _ = V.fit_coherence(h, "A", "Sa")
    return c, g, ca, ga


if __name__ == "__main__":
    main()
