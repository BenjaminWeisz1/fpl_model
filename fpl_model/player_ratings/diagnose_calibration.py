"""
Goal/assist calibration by rating decile, cameo share and opponent tercile, jointly (CHANGELOG §121).
Diagnostic only; reads a GOAL_MODEL_PREMATCH_LOG log with the Opp_Factor and Is_Starter columns.

    python diagnose_calibration.py [log.csv]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "validation" / "calibration_diag_20261030"
LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "log.csv"
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"), "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
N_BOOT, SEED = 200, 42
warnings.filterwarnings("ignore")


def frame():
    log = pd.read_csv(LOG, encoding="utf-8-sig")
    log["k"] = log.groupby("Player").cumcount()
    sub = log["Is_Starter"].astype(str).str.lower().eq("false").astype(float)
    # share of the previous 20 appearances that were substitute appearances
    log["cameo"] = sub.groupby(log["Player"]).transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean()).fillna(0)
    t = pd.read_csv(ROOT / "data" / "current" / "transfers.csv", encoding="utf-8-sig")
    gk = set(t.loc[t["Pos"] == "GK", "Player"])
    d = log[(log["Season"] >= 2018) & (log["k"] >= 10) & ~log["Player"].isin(gk)].copy()
    d["era"] = np.where(d["Season"] >= 2025, "2025+", "2018-24")
    d["cameo_bin"] = pd.cut(d["cameo"], [-0.01, 0.0, 0.25, 0.5, 1.0], labels=["0", "<=.25", "<=.5", ">.5"])
    d["opp"] = d.groupby("era")["Opp_Factor"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["strong", "middle", "weak"]))
    for t_, (_, _, r) in TARGETS.items():
        d[f"dec_{t_}"] = d.groupby("era")[r].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    return d


def glm(s, t_):
    a, p, _ = TARGETS[t_]
    y = s[a] * s["weight"]
    mu = (s[p] * s["weight"]).clip(lower=1e-9)
    X = pd.get_dummies(pd.DataFrame({
        "dec": pd.Categorical(s[f"dec_{t_}"], categories=[5] + [i for i in range(1, 11) if i != 5]),
        "cameo": pd.Categorical(s["cameo_bin"], categories=["0", "<=.25", "<=.5", ">.5"]),
        "opp": pd.Categorical(s["opp"], categories=["middle", "strong", "weak"])}), drop_first=True).astype(float)
    X = sm.add_constant(X)
    res = sm.GLM(y.to_numpy(), X.to_numpy(), family=sm.families.Poisson(), offset=np.log(mu.to_numpy())).fit()
    return dict(zip(X.columns, np.exp(res.params)))


def marginal(s, t_, col):
    a, p, _ = TARGETS[t_]
    g = s.assign(y=s[a] * s["weight"], m=s[p] * s["weight"]).groupby(col, observed=True)
    return {str(k): round(float(v.y.sum() / v.m.sum()), 3) for k, v in g}


def main():
    d = frame()
    out = {"rows": {e: int(n) for e, n in d["era"].value_counts().items()}}
    rng = np.random.default_rng(SEED)
    for era, s in d.groupby("era"):
        for t_ in TARGETS:
            point = glm(s, t_)
            players = s["Player"].unique()
            idx = {p: np.flatnonzero(s["Player"].to_numpy() == p) for p in players}
            draws = []
            for _ in range(N_BOOT):
                pick = rng.choice(players, len(players))
                draws.append(glm(s.iloc[np.concatenate([idx[p] for p in pick])], t_))
            ci = {k: [round(float(np.percentile([b[k] for b in draws], 2.5)), 3),
                      round(float(np.percentile([b[k] for b in draws], 97.5)), 3)] for k in point}
            out[f"{era} {t_}"] = {
                "marginal": {c: marginal(s, t_, c) for c in (f"dec_{t_}", "cameo_bin", "opp")},
                "joint": {k: {"ratio": round(float(v), 3), "ci95": ci[k]} for k, v in point.items()}}
            print(era, t_, {k: (round(float(v), 3), ci[k]) for k, v in point.items() if k != "const"}, flush=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
