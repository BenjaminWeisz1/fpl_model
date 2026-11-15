"""Recency-weighted defender training, out of time (CHANGELOG §105).

Control: the stored §46 production-configuration fits (feature_experiment_20261020/history_no_std_ew_<S>),
whose predictions must reproduce validation_no_std_ew_<S>.csv. Candidate: the same configuration with
training rows weighted 0.5^((S_last - season) / HALF_LIFE) in the GBM fit, the OOF predictions and the
isotonic calibrator. No production writes. Run: python -u -X utf8 scripts/defcon/evaluate_defender_recency.py
"""
from fpl_model import paths
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gzip
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from fpl_model.defcon.defender_model import POSITIVE_EVIDENCE, monotonic_params
from fpl_model.defcon.evaluate_defender_monotonic import metrics, bootstrap, predict
from fpl_model.defcon.recent_defcon import feature_columns

HERE = Path(__file__).resolve().parent
FEAT = paths.VALIDATION / "defcon_feature_experiment_20261020"
OUT = paths.VALIDATION / "defcon_recency_experiment_20261029"
OUT.mkdir(exist_ok=True)
BASE_COLS = ["AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed", "Recent5AvgPer90", "Recent5AvgRaw", "PctAboveThreshold",
             "OpponentAvgStat", "TeamAvgStat", "VenueAvg", "IsHome", "Is_FPL_API", "Pos_CB", "Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]
COLS = feature_columns(BASE_COLS, "no_std_ew")
HALF_LIFE = 2.0
YEARS = (2022, 2023, 2024, 2025, 2026)
NAMED = ["James Tarkowski", "Ethan Pinnock", "Chris Richards", "James Hill", "Jannik Schuster", "Tarik Muharemović"]


def params():
    p = monotonic_params(COLS)
    p["monotonic_cst"] = [int(c in POSITIVE_EVIDENCE or c == "RecentEWPer90") for c in COLS]
    return p


def recency_weights(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def fit_weighted(train, year):
    path = OUT / f"recency_{year}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as s:
            return pickle.load(s)
    t0 = time.monotonic()
    x, y = train[COLS].values, train.Target.values
    w = recency_weights(train.Season.values.astype(float))
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    model = HistGradientBoostingClassifier(random_state=42, **params())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, xs, y, cv=cv, method="predict_proba", n_jobs=1,
                            params={"sample_weight": w})[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(np.clip(oof, 1e-6, 1 - 1e-6), y, sample_weight=w)
    model.fit(xs, y, sample_weight=w)
    payload = {"base_model": model, "calibrator": cal, "scaler": scaler, "half_life": HALF_LIFE}
    with gzip.open(path, "wb") as s:
        pickle.dump(payload, s)
    print(f"fitted recency {year}: {len(y)} rows, {time.monotonic() - t0:.0f}s", flush=True)
    return payload


def control(year):
    with gzip.open(FEAT / f"history_no_std_ew_{year}.pkl.gz", "rb") as s:
        return pickle.load(s)


def bands(frame, col):
    f = frame.assign(exp=pd.cut(frame.MatchesPlayed, [-1, 50, 150, 250, 1e9], labels=["<=50", "51-150", "151-250", ">250"]),
                     act=pd.cut(frame.AvgPer90, [-1, 8, 10, 1e9], labels=["<8", "8-10", ">=10"]))
    g = f.groupby(["exp", "act"], observed=True).agg(n=("Target", "size"), actual=("Target", "mean"), pred=(col, "mean"))
    g["ratio"] = g.actual / g.pred
    return g


def band_error(g):
    g = g[g.n >= 75]
    return float((g.n * np.abs(np.log(g.actual / g.pred))).sum() / g.n.sum())


def violations(payload, val):
    x = val[COLS].values
    base = predict(payload, x)
    n = 0
    for c in [c for c in COLS if c in POSITIVE_EVIDENCE or c == "RecentEWPer90"]:
        s = x.copy(); j = COLS.index(c)
        s[:, j] = np.minimum(s[:, j] + .001, 1) if c == "PctAboveThreshold" else s[:, j] + .1
        n += int((predict(payload, s) < base - 1e-12).sum())
    return n


def main():
    frame = pd.read_pickle(FEAT / "frame_with_ew.pkl")
    rows, out = [], {"half_life": HALF_LIFE, "features": COLS, "splits": {}}
    for year in YEARS:
        train, val = frame[frame.Season < year], frame[frame.Season == year].copy()
        p0 = predict(control(year), val[COLS].values)
        ref = pd.read_csv(FEAT / f"validation_no_std_ew_{year}.csv", encoding="utf-8-sig")
        np.testing.assert_allclose(p0, ref["probability"].values, atol=1e-12, rtol=0)
        cand = fit_weighted(train, year)
        p1 = predict(cand, val[COLS].values)
        val["control"], val["recency"] = p0, p1
        out["splits"][str(year)] = {"control": metrics(val.Target.values, p0), "recency": metrics(val.Target.values, p1),
                                    "violations_recency": violations(cand, val)}
        print(year, {k: round(v["log_loss"], 5) for k, v in out["splits"][str(year)].items() if isinstance(v, dict)}, flush=True)
        rows.append(val)
    v = pd.concat(rows, ignore_index=True)
    out["pooled"] = {"control": metrics(v.Target.values, v.control.values),
                     "recency": metrics(v.Target.values, v.recency.values),
                     "paired_player_bootstrap": bootstrap(v, v.control.values, v.recency.values)}
    gc, gr = bands(v, "control"), bands(v, "recency")
    out["bands"] = {"control": gc.reset_index().astype({"exp": str, "act": str}).to_dict("records"),
                    "recency": gr.reset_index().astype({"exp": str, "act": str}).to_dict("records")}
    out["band_error"] = {"control": band_error(gc), "recency": band_error(gr)}
    out["named"] = {p: {str(s): {"n": int(len(g)), "actual": float(g.Target.mean()), "control": float(g.control.mean()),
                                 "recency": float(g.recency.mean())} for s, g in v[v.Player == p].groupby("Season")}
                    for p in NAMED}
    pc, pr = out["pooled"]["control"], out["pooled"]["recency"]
    out["adopt"] = bool(pr["log_loss"] <= pc["log_loss"] and pr["brier"] <= pc["brier"]
                        and out["band_error"]["recency"] < out["band_error"]["control"]
                        and all(s["violations_recency"] == 0 for s in out["splits"].values()))
    v[["Player", "Season", "MatchID", "Target", "MatchesPlayed", "AvgPer90", "control", "recency"]].to_csv(
        OUT / "validation_rows.csv", index=False, encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"pooled": out["pooled"], "band_error": out["band_error"], "adopt": out["adopt"]}, indent=1))
    print(gc.round(3)); print(gr.round(3))
    for p, d in out["named"].items():
        print(p, {s: (x["n"], round(x["actual"], 2), round(x["control"], 2), round(x["recency"], 2)) for s, x in d.items()})


if __name__ == "__main__":
    main()
