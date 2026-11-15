"""CHANGELOG §162, after the first run: why A gains 9.4% in 2019-2024 and 0.3% in 2025-26.

Not part of the adoption rule; a disclosed follow-up. Checks the saves data against Sofascore, then scores
the later block with each season's level matched, so the control's stale level doesn't colour the result.
Run: python -u -X utf8 -m research.assemble.validate_keeper_saves_later
"""
import json

import numpy as np
import pandas as pd

from research.assemble import validate_keeper_saves as K


def sofascore_check(x):
    s = pd.read_csv(K.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=K.KEY + ["Sofascore_saves"])
    s = s.groupby(K.KEY, as_index=False).Sofascore_saves.sum()
    f = x.groupby(K.KEY, as_index=False).Saves.sum()
    m = f.merge(s, on=K.KEY, how="left", validate="one_to_one")
    m["Same"] = m.Saves == m.Sofascore_saves
    t = m.groupby("Season").agg(Matches=("Saves", "size"), FPL=("Saves", "mean"), Sofascore=("Sofascore_saves", "mean"), Corr=("Saves", lambda c: c.corr(m.loc[c.index, "Sofascore_saves"])),
                                Same=("Same", "mean"), Var_FPL=("Saves", "var"), Var_Sofa=("Sofascore_saves", "var"))
    print(t.round(3).to_string())
    return t.round(4).reset_index().to_dict("records")


def level_matched(t, beta):
    """Spread only: each arm rescaled to the season's actual saves. The scale uses hindsight, equally for both arms."""
    out = t.copy()
    base = out.P * out.O ** beta
    for name, r in (("ctrl", out.P), ("a", base)):
        lam = r * out.Exposure
        out["r_" + name] = r * (out.groupby("Season").Saves.transform("sum") / lam.groupby(out.Season).transform("sum"))
        lam = out["r_" + name] * out.Exposure
        out["pts_" + name] = K.expected_points_per_three_saves(lam.values)
        out["ll_" + name] = K.poisson_ll(out.Saves.values, lam.values)
    return out


def main():
    x = K.frame()
    res = {"sofascore": sofascore_check(x)}
    t, _ = K.walk_forward(x)
    t = K.add_points(t)
    for label, seasons in (("2019-2024", K.TEST), ("2025-26", K.LATER), ("2025", [2025]), ("2026", [2026])):
        g = t[t.Season.isin(seasons)].reset_index(drop=True)
        res[label] = {}
        for beta in (0.66, 0.50):
            p = K.paired(level_matched(g, beta), "ctrl", "a")
            res[label][str(beta)] = p
            print(label, "rows", len(g), "beta", beta, "level-matched MSE", round(p["mse_base"], 4), "->", round(p["mse_arm"], 4),
                  f"{p['mse_pct']:+.2f}% ({p['mse_lo']:+.2f}, {p['mse_hi']:+.2f}) LL {p['ll_gain']:+.4f} ({p['ll_lo']:+.4f}, {p['ll_hi']:+.4f})")
    # Gap by team strength in the later block, at level.
    g = level_matched(t[t.Season.isin(K.LATER)].reset_index(drop=True), 0.66)
    g["Q"] = pd.qcut(g.Strength, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    res["later_gap"] = {a: K.spread(g, a) for a in ("ctrl", "a")}
    print(json.dumps(res["later_gap"], indent=1))
    # Production level: the 2025+ rate before each 2026 match, as goalkeeper_event_rates would give it.
    y = x[x.Season >= 2025].copy()
    m = y.groupby("MatchID").agg(s=("Saves", "sum"), e=("Exposure", "sum")).sort_index()
    y["P"] = y.MatchID.map(m.s.cumsum().shift(1) / m.e.cumsum().shift(1))
    y = y[y.Season == 2026].reset_index(drop=True)
    ref = float(np.average(y.O ** .66, weights=y.Exposure))  # hindsight in the reference only: level, not spread
    for name, r in (("ctrl", y.P), ("a", y.P * y.O ** .66 / ref)):
        lam = (r * y.Exposure).values
        y["pts_" + name], y["ll_" + name] = K.expected_points_per_three_saves(lam), K.poisson_ll(y.Saves.values, lam)
    y["Cluster"] = y.Team + "_" + y.Season.astype(str)
    res["2026_production_level"] = K.paired(y, "ctrl", "a")
    print("2026 at production level", len(y), {k: round(v, 4) for k, v in res["2026_production_level"].items()})
    (K.OUT / "results_later.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
