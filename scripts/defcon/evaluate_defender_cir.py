"""Centred isotonic calibration for the defender head, out of time (CHANGELOG §107).

Reuses the §106 level-adjusted frame and fits (level_experiment_20261029/level_<S>.pkl.gz): the base models are
identical; only the calibrator changes. The OOF scores are recomputed with the same deterministic CV and checked
against the stored isotonic calibrator. No production writes. Run: python -u -X utf8 scripts/defcon/evaluate_defender_cir.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import evaluate_defender_recency as R
import evaluate_defender_level as L
from defender_model import CenteredIsotonic
from evaluate_defender_monotonic import metrics, bootstrap, predict

OUT = R.HERE / "cir_experiment_20261029"
OUT.mkdir(exist_ok=True)


def main():
    raw = pd.read_pickle(R.FEAT / "frame_with_ew.pkl")
    frame, _ = L.adjusted(raw)
    rows, out = [], {"splits": {}}
    for year in R.YEARS:
        train, val = frame[frame.Season < year], frame[frame.Season == year].copy()
        payload = L.fit(train, year)
        xs = payload["scaler"].transform(train[R.COLS].values)
        y = train.Target.values
        model = HistGradientBoostingClassifier(random_state=42, **R.params())
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.clip(cross_val_predict(model, xs, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1], 1e-6, 1 - 1e-6)
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y)
        np.testing.assert_allclose(iso.y_thresholds_, payload["calibrator"].y_thresholds_, atol=1e-12)
        cir = CenteredIsotonic().fit(oof, y)
        xv = val[R.COLS].values
        p0 = predict(payload, xv)
        raw_v = predict(payload, xv, raw=True)
        p1 = cir.predict(raw_v)
        val["control"], val["cir"] = p0, p1
        val["AvgPer90"] = raw.loc[val.index, "AvgPer90"]  # bands on the unadjusted career rate, as in §105/§106
        cand = {**payload, "calibrator": cir}
        out["splits"][str(year)] = {"control": metrics(val.Target.values, p0), "cir": metrics(val.Target.values, p1),
                                    "distinct_control": int(np.unique(p0.round(9)).size),
                                    "distinct_cir": int(np.unique(p1.round(9)).size),
                                    "violations_cir": R.violations(cand, val)}
        s = out["splits"][str(year)]
        print(year, round(s["control"]["log_loss"], 5), round(s["cir"]["log_loss"], 5), s["distinct_control"], s["distinct_cir"], len(val), flush=True)
        rows.append(val)
    v = pd.concat(rows, ignore_index=True)
    out["pooled"] = {"control": metrics(v.Target.values, v.control.values), "cir": metrics(v.Target.values, v.cir.values),
                     "paired_player_bootstrap": bootstrap(v, v.control.values, v.cir.values)}
    out["band_error"] = {"control": R.band_error(R.bands(v, "control")), "cir": R.band_error(R.bands(v, "cir"))}
    pc, pr = out["pooled"]["control"], out["pooled"]["cir"]
    out["adopt"] = bool(pr["log_loss"] <= pc["log_loss"] and pr["brier"] <= pc["brier"]
                        and all(s["violations_cir"] == 0 for s in out["splits"].values()))
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"pooled": {k: (v if k == "paired_player_bootstrap" else {m: v[m] for m in ("log_loss", "brier", "auc")})
                                 for k, v in out["pooled"].items()}, "band_error": out["band_error"], "adopt": out["adopt"]}, indent=1))


if __name__ == "__main__":
    main()
