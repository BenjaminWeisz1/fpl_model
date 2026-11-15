"""Defender activity features at the current league level, out of time (CHANGELOG §106).

Same frame, control and scoring as evaluate_defender_recency.py (§105); the candidate rescales AvgPer90,
AvgRaw and RecentEWPer90 by L_(s-1) / (the league levels of the player's earlier qualifying matches).
No production writes. Run: python -u -X utf8 scripts/defcon/evaluate_defender_level.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gzip
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from research.defcon import evaluate_defender_recency as R
from research.defcon.evaluate_defender_monotonic import metrics, bootstrap, predict
from fpl_model.defcon.recent_defcon import EW_HALF_LIFE

OUT = R.HERE.parents[1] / "data" / "validation" / "defcon_level_experiment_20261029"
OUT.mkdir(exist_ok=True)


def season_levels(frame):
    q = frame[frame.Minutes >= 60]
    return q.groupby("Season").Per90Stat.mean().to_dict()


def level_factors(frame, levels):
    """Per-row factors for the career (plain mean) and recent (EW) features; frame in production order."""
    plain, ew = np.ones(len(frame)), np.ones(len(frame))
    decay = 2.0 ** (-1.0 / EW_HALF_LIFE)
    for _, pos in frame.groupby("Player", sort=False).indices.items():
        g = frame.iloc[pos]
        seen = []  # seasons of earlier qualifying matches
        for j, s, mins in zip(pos, g.Season.values, g.Minutes.values):
            s = int(s)
            if seen and (s - 1) in levels:
                cur = levels[s - 1]
                lv = np.array([levels[x] if x < s else cur for x in seen])
                w = decay ** np.arange(len(lv) - 1, -1, -1)
                plain[j] = cur / lv.mean()
                ew[j] = cur / (w * lv).sum() * w.sum()
            if mins >= 60:
                seen.append(s)
    return plain, ew


def adjusted(frame):
    levels = season_levels(frame)
    plain, ew = level_factors(frame, levels)
    f = frame.copy()
    f["AvgPer90"] = f.AvgPer90 * plain
    f["AvgRaw"] = f.AvgRaw * plain
    f["RecentEWPer90"] = f.RecentEWPer90 * ew
    f["_plain"], f["_ew"] = plain, ew
    return f, levels


def fit(train, year):
    path = OUT / f"level_{year}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as s:
            return pickle.load(s)
    t0 = time.monotonic()
    x, y = train[R.COLS].values, train.Target.values
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    model = HistGradientBoostingClassifier(random_state=42, **R.params())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, xs, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(np.clip(oof, 1e-6, 1 - 1e-6), y)
    model.fit(xs, y)
    payload = {"base_model": model, "calibrator": cal, "scaler": scaler}
    with gzip.open(path, "wb") as s:
        pickle.dump(payload, s)
    print(f"fitted level {year}: {len(y)} rows, {time.monotonic() - t0:.0f}s", flush=True)
    return payload


def main():
    raw = pd.read_pickle(R.FEAT / "frame_with_ew.pkl")
    frame, levels = adjusted(raw)
    print("levels", {k: round(v, 3) for k, v in levels.items()})
    print("factor quantiles", np.quantile(frame._plain, [.05, .5, .95]).round(3), np.quantile(frame._ew, [.05, .5, .95]).round(3))
    rows, out = [], {"levels": levels, "splits": {}}
    for year in R.YEARS:
        val_raw = raw[raw.Season == year]
        p0 = predict(R.control(year), val_raw[R.COLS].values)
        train, val = frame[frame.Season < year], frame[frame.Season == year].copy()
        cand = fit(train, year)
        p1 = predict(cand, val[R.COLS].values)
        val["control"], val["level"] = p0, p1
        val["AvgPer90"] = val_raw.AvgPer90.values  # bands on the unadjusted career rate, as in §105
        out["splits"][str(year)] = {"control": metrics(val.Target.values, p0), "level": metrics(val.Target.values, p1),
                                    "violations_level": R.violations(cand, frame[frame.Season == year])}
        print(year, round(out["splits"][str(year)]["control"]["log_loss"], 5), round(out["splits"][str(year)]["level"]["log_loss"], 5), flush=True)
        rows.append(val)
    v = pd.concat(rows, ignore_index=True)
    out["pooled"] = {"control": metrics(v.Target.values, v.control.values), "level": metrics(v.Target.values, v.level.values),
                     "paired_player_bootstrap": bootstrap(v, v.control.values, v.level.values)}
    gc, gl = R.bands(v, "control"), R.bands(v, "level")
    out["bands"] = {k: g.reset_index().astype({"exp": str, "act": str}).to_dict("records") for k, g in [("control", gc), ("level", gl)]}
    out["band_error"] = {"control": R.band_error(gc), "level": R.band_error(gl)}
    out["named"] = {p: {str(s): {"n": int(len(g)), "actual": float(g.Target.mean()), "control": float(g.control.mean()),
                                 "level": float(g.level.mean())} for s, g in v[v.Player == p].groupby("Season")} for p in R.NAMED}
    pc, pl = out["pooled"]["control"], out["pooled"]["level"]
    out["adopt"] = bool(pl["log_loss"] <= pc["log_loss"] and pl["brier"] <= pc["brier"]
                        and out["band_error"]["level"] < out["band_error"]["control"]
                        and all(s["violations_level"] == 0 for s in out["splits"].values()))
    v[["Player", "Season", "MatchID", "Target", "MatchesPlayed", "AvgPer90", "control", "level"]].to_csv(
        OUT / "validation_rows.csv", index=False, encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"pooled": out["pooled"], "band_error": out["band_error"], "adopt": out["adopt"]}, indent=1))
    print(gc.round(3)); print(gl.round(3))
    for p, d in out["named"].items():
        print(p, {s: (x["n"], round(x["actual"], 2), round(x["control"], 2), round(x["level"], 2)) for s, x in d.items()})


if __name__ == "__main__":
    main()
