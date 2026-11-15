"""CHANGELOG §164: does non-penalty xGOT improve career goal conversion? Test only; no production writes.

Needs a pre-match log first:
  GOAL_MODEL_PREMATCH_LOG=data/validation/conversion_xgot_20261101/log.csv python build_player_ratings.py
Run: python -u -X utf8 scripts/player_ratings/validate_conversion_xgot.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "current"
OUT = ROOT / "data" / "validation" / "conversion_xgot_20261101"
KEY = ["Season", "Team", "Opponent", "Venue"]
C = 30
SCORED_XGOT, MISSED_XGOT = .86, .70
ALPHAS = [.25, .5, .75, 1.0]
GRID = np.arange(0, 1.51, .25)
TEST = [2023, 2024, 2025, 2026]
REPS = 2000


def match_order():
    H = pd.read_csv(DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    a = H.assign(Team=H.Home, Opponent=H.Away, Venue="Home")
    b = H.assign(Team=H.Away, Opponent=H.Home, Venue="Away")
    return pd.concat([a, b])[KEY + ["MatchID"]]


def xgot_rows():
    e = pd.read_csv(DATA / "sofascore_player_extra.csv.gz", low_memory=False, usecols=["Season", "Event_ID_Sofascore", "Player_ID_Sofascore", "expectedGoalsOnTarget"])
    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=KEY + ["Event_ID_Sofascore", "Player_ID_Sofascore", "Local_Player"])
    e = e[e.Season >= 2022].merge(s, on=["Season", "Event_ID_Sofascore", "Player_ID_Sofascore"], how="left", validate="one_to_one")
    e = e.rename(columns={"Local_Player": "Player", "expectedGoalsOnTarget": "RawX"})
    return e[KEY + ["Player", "RawX"]].dropna(subset=["Player"]).drop_duplicates(KEY + ["Player"])


def frame():
    p = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=KEY + ["Player", "Position", "Minutes", "Goals", "PK", "PKatt", "npxG"])
    p = p[p.Minutes.fillna(0) > 0].drop_duplicates(KEY + ["Player"]).copy()
    for c in ("Goals", "PK", "PKatt", "npxG"):
        p[c] = p[c].fillna(0.0)
    p["G"] = (p.Goals - p.PK).clip(lower=0)
    p = p.merge(match_order(), on=KEY, how="left", validate="many_to_one")
    assert p.MatchID.notna().all()
    x = xgot_rows()
    p = p.merge(x, on=KEY + ["Player"], how="left", validate="one_to_one")
    kept = float(p.RawX.sum() / x.RawX.sum())
    print("xGOT kept after the join", round(kept, 4), "total", round(float(x.RawX.sum()), 1))
    assert kept > .97
    p["X"] = (p.RawX.fillna(0) - SCORED_XGOT * p.PK - MISSED_XGOT * (p.PKatt - p.PK)).clip(lower=0)
    post = p.Season >= 2022
    p["X"] = p.X.where(post, 0.0)
    p["G2"], p["xG2"] = p.G.where(post, 0.0), p.npxG.where(post, 0.0)
    p = p.sort_values(["MatchID"]).reset_index(drop=True)
    # Career to date: earlier matches only.
    for src, dst in (("G", "Gc"), ("npxG", "xGc"), ("X", "Xc"), ("G2", "G2c"), ("xG2", "xG2c")):
        p[dst] = p.groupby("Player")[src].transform(lambda s: s.cumsum().shift(1)).fillna(0.0)
    print(p[p.Season >= 2022].groupby("Season")[["G", "npxG", "X"]].sum().round(1).to_string())
    return p


def conv_alpha(p, a):
    return ((p.Gc + a * (p.Xc - p.G2c) + C) / (p.xGc + C)).clip(lower=.2)


def conv_split(p, b1, b2):
    S = p.Xc - p.xG2c
    F = (p.Gc - p.xGc) - S
    return (1 + (b1 * S + b2 * F) / (p.xGc + C)).clip(lower=.2)


def ll(y, lam):
    lam = np.maximum(lam, .01)
    return y * np.log(lam) - lam - gammaln(y + 1)


def paired(t, base, arm, y="G"):
    codes, _ = pd.factorize(t.Player)
    k = codes.max() + 1
    w = np.random.default_rng(1).multinomial(k, np.ones(k) / k, size=REPS)
    e0, e1 = (t[y] - t[base]) ** 2, (t[y] - t[arm]) ** 2
    s0, s1 = np.bincount(codes, e0.values, k), np.bincount(codes, e1.values, k)
    n = np.bincount(codes, minlength=k).astype(float)
    pct = 100 * ((w * s1).sum(1) / (w * s0).sum(1) - 1)
    d = ll(t[y].values, t[arm].values) - ll(t[y].values, t[base].values)
    b = (w * np.bincount(codes, d, k)).sum(1) / (w * n).sum(1)
    return {"mse_pct": float(100 * (e1.mean() / e0.mean() - 1)), "lo": float(np.quantile(pct, .025)), "hi": float(np.quantile(pct, .975)),
            "ll": float(d.mean()), "ll_lo": float(np.quantile(b, .025)), "ll_hi": float(np.quantile(b, .975))}


def judge(t, arm):
    p = paired(t, "f_ctrl", "f_" + arm)
    by = {int(s): paired(g, "f_ctrl", "f_" + arm)["mse_pct"] for s, g in t.groupby("Season")}
    r = t[t.r_ctrl.notna()]
    q = paired(r, "r_ctrl", "r_" + arm)
    bias = {}
    for pos, g in t.groupby("Line"):
        b0, b1 = abs((g.G - g.f_ctrl).mean()), abs((g.G - g["f_" + arm]).mean())
        bias[pos] = {"control": float(b0), "arm": float(b1)}
    checks = {"i": p["hi"] < 0, "ii": p["ll_lo"] > 0, "iii": sum(by[s] < 0 for s in by if s < 2026) >= 2, "iv": q["mse_pct"] <= 0,
              "v": all(v["arm"] <= v["control"] + .005 for v in bias.values())}
    return {"pooled": p, "by_season": by, "production_form": q, "bias": bias, "checks": checks, "passes": all(checks.values())}


def persistence(p):
    """Season-to-season: per-xG, shrunk by c, players with 15+ npxG in the pooled seasons since 2022."""
    a = p[p.Season.between(2022, 2025)].groupby(["Player", "Season"]).agg(G=("G", "sum"), xG=("npxG", "sum"), X=("X", "sum")).reset_index()
    a = a[a.xG >= 3]
    tot = a.groupby("Player").xG.transform("sum")
    a = a[tot >= 15].copy()
    a["S"], a["F"], a["T"] = (a.X - a.xG) / (a.xG + C), (a.G - a.X) / (a.xG + C), (a.G - a.xG) / (a.xG + C)
    n = a.merge(a.assign(Season=a.Season - 1), on=["Player", "Season"], suffixes=("", "2"))
    out = {"pairs": len(n), "players": int(a.Player.nunique()), "r_S_S": float(n.S.corr(n.S2)), "r_F_F": float(n.F.corr(n.F2)), "r_T_T": float(n["T"].corr(n.T2)), "r_S_T": float(n.S.corr(n.T2)), "r_F_T": float(n.F.corr(n.T2))}
    X = np.column_stack([np.ones(len(n)), n.S, n.F])
    beta, *_ = np.linalg.lstsq(X, n.T2.values, rcond=None)
    res = n.T2.values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=3))
    out["next_T_on_S"], out["next_T_on_F"], out["se_S"], out["se_F"] = float(beta[1]), float(beta[2]), float(se[1]), float(se[2])
    return out


def main():
    p = frame()
    p["Line"] = p.Position.astype(str).str.extract(r"(GK|FW|ST|CF|LW|RW|W|M|AM|CM|DM|D|B)")[0].map(lambda v: "F" if v in ("FW", "ST", "CF", "LW", "RW", "W") else "M" if v in ("M", "AM", "CM", "DM") else "D" if v in ("D", "B") else "Unlisted")
    log = pd.read_csv(OUT / "log.csv", encoding="utf-8-sig", low_memory=False).drop_duplicates(KEY + ["Player"])
    p = p.merge(log[KEY + ["Player", "pred_goals_90"]], on=KEY + ["Player"], how="left", validate="one_to_one")
    arms = {"ctrl": conv_alpha(p, 0.0)}
    arms.update({f"a{int(a * 100)}": conv_alpha(p, a) for a in ALPHAS})
    for n, c in arms.items():
        p["f_" + n] = p.npxG * c
        p["r_" + n] = p.pred_goals_90 * p.Minutes / 90 * c
    assert (conv_alpha(p, 0) - conv_split(p, 1, 1)).abs().max() < 1e-9
    t = p[p.Season.isin(TEST)].reset_index(drop=True)
    print("test rows", len(t), "with pre-match rating", int(t.r_ctrl.notna().sum()), "players", t.Player.nunique())
    print("conversion SD, control", round(float(arms["ctrl"][p.Season.isin(TEST)].std()), 4), "a50 less control SD", round(float((arms["a50"] - arms["ctrl"])[p.Season.isin(TEST)].std()), 4))
    res = {"rows": len(t), "mse_control": float(((t.G - t.f_ctrl) ** 2).mean())}
    for a in ALPHAS:
        res[f"a{int(a * 100)}"] = judge(t, f"a{int(a * 100)}")

    # D: fitted on earlier seasons, tested 2024-2026.
    picks, parts = [], []
    for s in [2024, 2025, 2026]:
        tr = p[p.Season.between(2023, s - 1)]
        best = max(((b1, b2) for b1 in GRID for b2 in GRID), key=lambda b: ll(tr.G.values, (tr.npxG * conv_split(tr, *b)).values).mean())
        g = p[p.Season == s].copy()
        c = conv_split(g, *best)
        g["f_d"], g["r_d"] = g.npxG * c, g.pred_goals_90 * g.Minutes / 90 * c
        parts.append(g)
        picks.append({"Season": s, "b1": float(best[0]), "b2": float(best[1])})
    d = pd.concat(parts, ignore_index=True)
    res["d_picks"] = picks
    res["d"] = judge(d, "d")
    res["d_v_a50"] = paired(d, "f_a50", "f_d")
    res["a100_v_a50"] = paired(t, "f_a50", "f_a100")
    # Where the change could matter: players with real evidence since 2022.
    heavy = t[t.xG2c >= 10].reset_index(drop=True)
    res["heavy_rows"] = len(heavy)
    res["heavy"] = {a: paired(heavy, "f_ctrl", "f_" + a) for a in ("a50", "a100")}
    res["persistence"] = persistence(p)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))
    cur = p.sort_values("MatchID").groupby("Player").tail(1)
    cur = cur[(cur.Season == 2026) & (cur.xGc >= 15)]
    v = pd.DataFrame({"Player": cur.Player, "xG": cur.xGc.round(1), "Control": conv_alpha(cur, 0).round(3), "A50": conv_alpha(cur, .5).round(3), "A100": conv_alpha(cur, 1).round(3)})
    v["Move"] = v.A50 - v.Control
    print(v.sort_values("Move").head(8).to_string(index=False)); print(v.sort_values("Move").tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
