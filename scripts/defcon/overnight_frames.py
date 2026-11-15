"""Shared harness for the overnight defcon experiments (AGENDA 2026-10-31, CHANGELOG §149+). No production writes.

`extract()` runs predict.py's source up to its out-of-time section (nothing is trained or written there) and
saves the three current training frames plus a dated, sequenced copy of the match log. predict.py is not edited.
`fit_head` / `score` reproduce the production heads: monotonic HGB with centred isotonic for defenders, random
forests with step isotonic for midfielders and forwards, calibrated on 5-fold out-of-fold predictions.

Run: python -u -X utf8 scripts/defcon/overnight_frames.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import ast
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from defender_model import CenteredIsotonic, monotonic_params

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "overnight_20261031"
OUT.mkdir(exist_ok=True)
FRAMES = OUT / "frames.pkl"
RF_MF = dict(n_estimators=400, max_depth=10, min_samples_leaf=3, max_features="sqrt")
RF_FWD = dict(n_estimators=200, max_depth=8, min_samples_leaf=5, max_features="log2")


def extract():
    source = (HERE / "predict.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_MODEL_CONFIGS" for t in node.targets):
            break
        body.append(node)
    tree.body = body
    scope = {"__file__": str(HERE / "predict.py"), "__name__": "predict_frames"}
    exec(compile(tree, str(HERE / "predict.py"), "exec"), scope)
    keep = {k: scope[k] for k in ("df_def", "df_mid", "df_fwd", "DEF_FEATURE_COLS", "MID_FEATURE_COLS", "FWD_FEATURE_COLS")}
    keep["matchlogs"] = scope["matchlogs"]
    keep["predict_md5"] = hashlib.md5(source.encode()).hexdigest()
    with FRAMES.open("wb") as s:
        pickle.dump(keep, s)
    return keep


def frames():
    if FRAMES.exists():
        return pd.read_pickle(FRAMES)
    return extract()


def fit_head(head, x, y, cols, seed=42):
    """head: D, M or F. Returns (scaler, base model, calibrator)."""
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    if head == "D":
        model = HistGradientBoostingClassifier(random_state=seed, **monotonic_params(cols))
        njobs = 1
    else:
        model = RandomForestClassifier(random_state=seed, n_jobs=1, **(RF_MF if head == "M" else RF_FWD))
        njobs = 5
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, xs, y, cv=cv, method="predict_proba", n_jobs=njobs)[:, 1]
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    cal = (CenteredIsotonic() if head == "D" else IsotonicRegression(out_of_bounds="clip")).fit(oof, y)
    model.fit(xs, y)
    return scaler, model, cal


def predict(fit, x):
    scaler, model, cal = fit
    return np.clip(cal.predict(model.predict_proba(scaler.transform(x))[:, 1]), 1e-6, 1 - 1e-6)


def score(y, p):
    return {"n": int(len(y)), "hits": int(np.sum(y)), "ll": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)), "auc": float(roc_auc_score(y, p)) if 0 < np.sum(y) < len(y) else None,
            "mean_p": float(np.mean(p)), "mean_y": float(np.mean(y))}


def paired_bootstrap(players, y, p0, p1, n=2000, seed=0):
    """Player-cluster bootstrap of the log-loss change p1 - p0 (negative favours p1)."""
    p0, p1 = np.clip(p0, 1e-6, 1 - 1e-6), np.clip(p1, 1e-6, 1 - 1e-6)
    d = -(y * (np.log(p1) - np.log(p0)) + (1 - y) * (np.log1p(-p1) - np.log1p(-p0)))
    codes, _ = pd.factorize(players)
    k = codes.max() + 1
    s, c = np.bincount(codes, d, k), np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(k, np.ones(k) / k, size=n)
    stats = (w * s).sum(1) / (w * c).sum(1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(stats, .025)), "hi": float(np.quantile(stats, .975))}


if __name__ == "__main__":
    f = extract()
    for k in ("df_def", "df_mid", "df_fwd"):
        print(k, f[k].shape, f[k].groupby("Season").Target.agg(["size", "sum"]).T.to_dict())
    print(f["DEF_FEATURE_COLS"])
    print(json.dumps({"predict_md5": f["predict_md5"]}))
