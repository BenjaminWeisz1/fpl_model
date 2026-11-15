"""
Money prior fitted without outlier promoted teams (pre-registered in CHANGELOG §208; user idea).

Production's past-only OLS (§202: season GF and GA on Wages_pct and Squad_Value_pct from historical_money.csv,
seasons < s) refitted after dropping training rows whose residual exceeds a multiple of the training residual SD:
promoted rows only at 2 SD (P2, primary) or 1.5 SD (P15), or any row at 2 SD (A2). C0 (no exclusion) must reproduce
production money_priors_by_season.csv for 2017-2025. Stage 1 scores the priors on 2017-2025 team-seasons; stage 2
(only if stage 1 passes) runs §206's update_team_ratings.py comparison.

Run locally:
    python -m research.team_ratings.validate_money_prior_outliers
"""
import json
import sys

import numpy as np
import pandas as pd

from research.team_ratings import validate_money_priors_pooled as vmp

OUT = vmp.ROOT / "data" / "validation" / "money_prior_outliers_20261107"
FEATURES = ["Wages_pct", "Squad_Value_pct"]
TEST = vmp.TEST
ARMS = {"C0": None, "P2": ("promoted", 2.0), "P15": ("promoted", 1.5), "A2": ("all", 2.0)}


def load():
    df = pd.read_csv(vmp.CUR / "historical_money.csv", encoding="utf-8-sig")
    for col in ("Wages", "Squad_Value"):
        df[f"{col}_pct"] = df.groupby("Season")[col].transform(lambda x: x / x.sum())
    prev = {s: set(df.loc[df["Season"] == s - 1, "Team"]) for s in df["Season"].unique()}
    df["promoted"] = [t not in prev[s] if prev[s] else False for s, t in zip(df["Season"], df["Team"])]
    # 2013 has no earlier row in the file: take its promoted clubs from Transfermarkt's 2012 games
    tm = pd.read_csv(vmp.OUT / "team_seasons.csv", encoding="utf-8-sig")
    tm13 = tm[(tm["competition_id"] == "GB1") & (tm["season"] == 2013) & tm["promoted"]]["name"]
    names13 = {vmp.TM_TO_PROJECT.get(n, n) for n in tm13}
    df.loc[df["Season"] == 2013, "promoted"] = df.loc[df["Season"] == 2013, "Team"].isin(names13)
    assert df.loc[df["Season"] == 2013, "promoted"].sum() == 3, names13
    per = df[df["GF"].notna()].groupby("Season")["promoted"].sum()
    assert (per.loc[2014:] == 3).all(), per
    return df


def fit_predict(train, test, target, rule):
    X, y = np.column_stack([np.ones(len(train)), train[FEATURES]]), train[target].to_numpy()
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    dropped = []
    if rule is not None:
        who, k = rule
        r = y - X @ c
        flag = np.abs(r) > k * r.std(ddof=X.shape[1])
        if who == "promoted":
            flag &= train["promoted"].to_numpy()
        dropped = [(int(s), t, round(float(v), 1)) for s, t, v in zip(train["Season"][flag], train["Team"][flag], r[flag])]
        c, *_ = np.linalg.lstsq(X[~flag], y[~flag], rcond=None)
    pred = (np.column_stack([np.ones(len(test)), test[FEATURES]]) @ c) / 38
    return pred, dropped


def predictions(df):
    known = df[df["GF"].notna()]
    rows, drops = [], {a: {} for a in ARMS}
    for s in TEST:
        train, test = known[known["Season"] < s], known[known["Season"] == s].copy()
        for arm, rule in ARMS.items():
            for target in ("GF", "GA"):
                test[f"{arm}_{target}"], d = fit_predict(train, test, target, rule)
                drops[arm][f"{s}_{target}"] = d
        rows.append(test)
    p = pd.concat(rows, ignore_index=True)
    p["GF"], p["GA"] = p["GF"] / 38, p["GA"] / 38
    return p, drops


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    p, drops = predictions(df)
    prod = pd.read_csv(vmp.CUR / "money_priors_by_season.csv", encoding="utf-8-sig")
    chk = p.merge(prod, on=["Season", "Team"], how="left")
    assert chk["Money_GF"].notna().all()
    err = max((chk["C0_GF"] - chk["Money_GF"]).abs().max(), (chk["C0_GA"] - chk["Money_GA"]).abs().max())
    assert err < 1e-8, f"C0 doesn't reproduce production ({err})"
    p.to_csv(OUT / "stage1_predictions.csv", index=False, encoding="utf-8-sig")

    arms = list(ARMS)
    pr = p[p["promoted"]]
    res = {"c0_max_abs_diff_vs_production": float(err),
           "all": vmp.score(p, arms), "promoted": vmp.score(pr, arms),
           "ci95_vs_C0": {a: {"all": vmp.boot(p, a, "C0"), "promoted": vmp.boot(pr, a, "C0")} for a in arms[1:]},
           "seasons_won": {a: int(sum(vmp.sq_err(g, a).mean() < vmp.sq_err(g, "C0").mean() for _, g in p.groupby("Season")))
                           for a in arms[1:]},
           "rel_pct": {a: {k: round(100 * (res_mse(d, a) / res_mse(d, "C0") - 1), 2) for k, d in (("all", p), ("promoted", pr))}
                       for a in arms[1:]},
           # distinct team-seasons ever dropped, with the last-seen residual (season goals)
           "dropped": {a: sorted({(s, t) for v in drops[a].values() for s, t, _ in v}) for a in arms[1:]},
           "dropped_by_window": {a: drops[a] for a in arms[1:]}}
    a = "P2"
    res["P2_stage1_pass"] = bool(res["all"][a]["mse"] < res["all"]["C0"]["mse"] and res["seasons_won"][a] >= 5
                                 and (res["ci95_vs_C0"][a]["all"][1] < 0 or res["ci95_vs_C0"][a]["promoted"][1] < 0))
    summary = {k: res[k] for k in ("all", "promoted", "ci95_vs_C0", "seasons_won", "rel_pct", "dropped", "P2_stage1_pass")}
    print(json.dumps(summary, indent=1, default=str), flush=True)
    if res["P2_stage1_pass"]:
        vmp.OUT = OUT
        cand = p.rename(columns={"Season": "season"}).assign(competition_id="GB1")
        res["stage2_P2"] = vmp.stage2(cand, "P2")
        print(json.dumps(res["stage2_P2"], indent=1), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def res_mse(d, a):
    return float(vmp.sq_err(d, a).mean())


if __name__ == "__main__":
    sys.exit(main())
