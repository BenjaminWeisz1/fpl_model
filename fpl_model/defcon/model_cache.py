"""Defcon model: the calibrated-model wrapper, the trained-model cache and tune_and_calibrate.

Split out of predict.py (2026-11-16 cleanup). The cache key hashes the source of every module that shapes a trained
model (MODEL_SOURCES below), so an edit to any of them retrains once instead of serving a stale model.
"""
import contextlib
import gzip
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from fpl_model import config, paths
from fpl_model.defcon.defender_model import CenteredIsotonic

_DATA = paths.CURRENT
_DEFCON_DIR = Path(__file__).resolve().parent
# Every module whose code shapes a trained model or its training frame. defender_model.py is hashed separately
# (_DEFENDER_SOURCE_MD5, §189), as before the split.
MODEL_SOURCES = ["predict.py", "features.py", "model_cache.py", "recent_defcon.py", "opponent_model.py"]


def _centered_calibration(label):
    """§107: the defender head's calibrator interpolates between isotonic block centres instead of returning steps
    (which gave many defenders identical probabilities)."""
    return label.startswith("Defender")


# ── Calibration wrapper ──────────────────────────────────────────────
class CalibratedModel:
    """Base model + standalone isotonic calibrator.
    predict_proba applies scaler (if set) then calibrates the raw output.
    Replaces CalibratedClassifierCV with an explicit cross_val_predict
    pipeline so calibration is a single isotonic fit on OOF predictions
    from the *tuned* model, not an ensemble of 5 internally-refitted
    copies."""
    def __init__(self, base_model, calibrator, scaler=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.scaler = scaler

    def predict_proba(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ── Trained-model cache + parallel RF fits ──────────────────────────
# Weekly runs retrained all 9 models even when nothing that determines
# them had changed. tune_and_calibrate now consults a disk cache keyed
# on a content hash of EVERYTHING that determines the artifact:
#   * md5 of every model module (MODEL_SOURCES) and of defender_model.py --
#     any code edit invalidates the cache
#   * sklearn/numpy versions
#   * model class, label, tuning path (do_tune), default params, and
#     the random_state/cv/n_iter call arguments
#   * the training frame itself (X and y bytes) -- NOT the input-file
#     md5s: fixtures.csv, minutes_probabilities.csv and match_results.csv
#     change weekly but feed only the prediction loops, never training
#     rows, so keying on them would force pointless retrains; conversely
#     any edit to player_match_stats.csv / players_current.csv /
#     transfers.csv that changes a frame's contents changes X and
#     invalidates naturally. Input-file md5s are still written to
#     input_fingerprints.json each run for diagnostics ("why did the
#     cache miss?").
# The payload stores base model + calibrator + scaler + best_params as
# plain sklearn objects (not the CalibratedModel wrapper, which lives
# in __main__ and would be unpicklable elsewhere). Stale entries from
# earlier weeks are pruned once all nine trainings are done.
# DEFCON_MODEL_CACHE=0 disables the cache entirely (forced retrain);
# DEFCON_RF_N_JOBS overrides the RF fit parallelism (default -1 = all
# cores; with random_state fixed, n_jobs is result-invariant).
_CACHE_ENABLED = config.get("DEFCON_MODEL_CACHE")
_CACHE_DIR = paths.CACHE / "defcon"
_PREDICT_SOURCE_MD5 = hashlib.md5(b"".join((_DEFCON_DIR / f).read_bytes() for f in MODEL_SOURCES)).hexdigest()
# The defender head's parameters and calibration live here, so an edit must retrain too (§189).
_DEFENDER_SOURCE_MD5 = hashlib.md5((_DEFCON_DIR / "defender_model.py").read_bytes()).hexdigest()
_RF_N_JOBS = config.get("DEFCON_RF_N_JOBS")
_CACHE_USED_KEYS = set()
_CACHE_INITIALISED = False


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _initialise_model_cache():
    """Create the cache dir and record input-file md5s once per run
    (diagnostics only -- cache keys are content-based, see above)."""
    global _CACHE_INITIALISED
    if _CACHE_INITIALISED or not _CACHE_ENABLED:
        return
    _CACHE_INITIALISED = True
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprints = {name: _md5_file(_DATA / name)
                    for name in ("player_match_stats.csv", "players_current.csv",
                                 "transfers.csv", "fixtures.csv",
                                 "minutes_probabilities.csv", "match_results.csv")
                    if (_DATA / name).exists()}
    payload = {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "predict_source_md5": _PREDICT_SOURCE_MD5,
               "files": fingerprints}
    tmp = _CACHE_DIR / "input_fingerprints.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, _CACHE_DIR / "input_fingerprints.json")


def _model_cache_key(X, y, model_cls, label, do_tune, default_params,
                     random_state, cv, n_iter):
    """md5 over everything that determines the trained artifact."""
    h = hashlib.md5()
    h.update(_PREDICT_SOURCE_MD5.encode())
    h.update(_DEFENDER_SOURCE_MD5.encode())
    h.update(f"sklearn={sklearn.__version__};numpy={np.__version__}".encode())
    h.update(f"{model_cls.__name__}|{label}|tune={int(bool(do_tune))}".encode())
    h.update(repr(sorted((default_params or {}).items())).encode())
    h.update(f"rs={random_state}|cv={cv}|n_iter={n_iter}|cir={int(_centered_calibration(label))}".encode())
    h.update(f"X{X.shape}|{X.dtype}|".encode())
    h.update(X.tobytes())
    h.update(f"y{y.shape}|{y.dtype}|".encode())
    h.update(y.tobytes())
    return h.hexdigest()


def _cache_path_for(label, key):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return _CACHE_DIR / f"{slug}__{key}.pkl.gz"


def _load_cached_model(path, label):
    """Return (calibrated, scaler, best_params) from the cache, or None
    on any failure (corrupt entry, unpicklable payload) -- cache
    problems must never break a run, only cost a retrain."""
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        return (CalibratedModel(payload["base_model"], payload["calibrator"],
                                 payload["scaler"]),
                payload["scaler"], payload["best_params"])
    except Exception as exc:
        print(f"  [{label}] cache entry unreadable ({exc}) -- retraining")
        with contextlib.suppress(OSError):
            path.unlink()
        return None


def _store_cached_model(path, label, production, calibrator, scaler,
                        best_params):
    """Persist a freshly trained artifact atomically (tmp + rename) so a
    crash mid-write can never leave a corrupt cache entry behind."""
    payload = {"base_model": production, "calibrator": calibrator,
               "scaler": scaler, "best_params": best_params}
    tmp = path.with_name(path.name + ".tmp")
    try:
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"  [{label}] cache write failed ({exc}) -- continuing uncached")
        with contextlib.suppress(OSError):
            tmp.unlink()


def prune_model_cache():
    """Delete entries whose keys the current run did not produce: last
    week's training frames are strictly superseded, so their cached
    models can never be hit again (and the cache sits in OneDrive-synced
    storage, so dead weight is real sync churn). Runs after the last of
    the nine trainings."""
    if not (_CACHE_ENABLED and _CACHE_DIR.exists()):
        return
    kept = removed = 0
    freed = 0
    for f in sorted(_CACHE_DIR.glob("*.pkl.gz")):
        if f.name in _CACHE_USED_KEYS:
            kept += 1
        else:
            freed += f.stat().st_size
            f.unlink()
            removed += 1
    for f in sorted(_CACHE_DIR.glob("*.tmp")):
        f.unlink()
    print(f"Model cache: {kept} entr{'y' if kept == 1 else 'ies'} kept, "
          f"{removed} stale pruned ({freed / 1e6:.1f} MB freed)")


def tune_and_calibrate(X, y, model_cls, param_grid, label, cv=5,
                       random_state=42, n_iter=30, do_tune=True,
                       default_params=None):
    """Tune hyperparameters via RandomizedSearchCV, then calibrate the
    tuned model with cross_val_predict + isotonic regression.

    Steps:
      1. Tune: RandomizedSearchCV(n_iter, cv, scoring=neg_log_loss) to
         find best hyperparameters on the training data.
      2. Calibrate: cross_val_predict with the same fold structure to
         generate out-of-fold probabilities; fit IsotonicRegression on
         OOF p vs y.
      3. Final model: train the tuned model on ALL training data; attach
         the calibrator for inference.

    Returns (CalibratedModel, scaler, best_params).
    The CalibratedModel wraps scaler + base_model + calibrator so
    predict_proba accepts raw (unscaled) feature arrays.

    Results are served from the model cache when the content key
    matches a stored artifact (see the model-cache section above); RF
    fits run with n_jobs=_RF_N_JOBS, which with random_state fixed is
    result-invariant and purely faster."""
    _initialise_model_cache()
    cache_path = None
    if _CACHE_ENABLED:
        key = _model_cache_key(X, y, model_cls, label, do_tune,
                               default_params, random_state, cv, n_iter)
        cache_path = _cache_path_for(label, key)
        if cache_path.exists():
            cached = _load_cached_model(cache_path, label)
            if cached is not None:
                _CACHE_USED_KEYS.add(cache_path.name)
                print(f"  [{label}] cache hit -- skipping training "
                      f"(key {key[:12]})")
                return cached
    t_start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Step 1: Tune hyperparameters
    if do_tune and param_grid:
        search = RandomizedSearchCV(
            model_cls(random_state=random_state),
            param_grid, n_iter=n_iter, cv=skf,
            scoring="neg_log_loss", random_state=random_state, n_jobs=4,
        )
        search.fit(X_scaled, y)
        best_params = search.best_params_
        print(f"  [{label}] tuning best: {best_params}")
    else:
        best_params = default_params or {}
        print(f"  [{label}] skipping tuning (using defaults)")

    # RF fits parallelise across trees (result-invariant with the fixed
    # random_state). The search estimator above stays single-threaded so
    # its n_jobs=4 candidate parallelism is not oversubscribed.
    def _make(params):
        p = dict(params)
        p["random_state"] = random_state
        if model_cls is RandomForestClassifier:
            p["n_jobs"] = _RF_N_JOBS
        return model_cls(**p)

    # Step 2: OOF calibration on the tuned model
    base_for_cv = _make(best_params)
    oof_proba = cross_val_predict(
        base_for_cv, X_scaled, y, cv=skf, method="predict_proba",
    )
    oof_p = np.clip(oof_proba[:, 1], 1e-6, 1 - 1e-6)
    calibrator = CenteredIsotonic() if _centered_calibration(label) else IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_p, y)

    # Step 3: Train final production model on ALL training data
    production = _make(best_params)
    production.fit(X_scaled, y)

    print(f"  [{label}] trained on {len(y)} records")
    if cache_path is not None:
        _store_cached_model(cache_path, label, production, calibrator,
                            scaler, best_params)
        _CACHE_USED_KEYS.add(cache_path.name)
        print(f"  [{label}] cached for next run (key {key[:12]})")
    print(f"  [{label}] training wall time {time.time() - t_start:.1f}s")
    return CalibratedModel(production, calibrator, scaler), scaler, best_params
