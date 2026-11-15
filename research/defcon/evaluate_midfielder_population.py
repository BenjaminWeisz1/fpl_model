"""Registered, diagnostic-only test of the RF_MF training population.

The production data-preparation statements are executed from predict.py only
through MID_FEATURE_COLS. This reproduces its row, target, feature, prior,
source-scale, and position construction without running production fits,
touching the model cache, or writing serving outputs.
"""
import argparse
import ast
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from fpl_model import paths

_ROOT = paths.ROOT
_DATA = paths.CURRENT
_OUT = paths.VALIDATION / "midfielder_defcon_diagnostics_20261022"
_PREDICT = Path(__file__).with_name("predict.py")
_INPUTS = (
    "player_match_stats.csv",
    "players_current.csv",
    "transfers.csv",
    "player_match_positions.csv",
)
_PRODUCTION_OUTPUTS = (
    "predictions_defensive.csv",
    "defcon_average_probabilities.csv",
    "player_ratings.csv",
)
_KEYS = ["Season", "MatchID", "Player", "Team", "Opponent", "Venue"]
_ROW_KEYS = _KEYS + ["_RowOccurrence"]
_SPLITS = (2022, 2023, 2024, 2025, 2026)
_FEATURE_FIELDS = (
    "AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed", "Recent5AvgPer90",
    "Recent5AvgRaw", "PctAboveThreshold", "OpponentAvgStat", "TeamAvgStat",
    "VenueAvg", "IsHome", "Is_FPL_API",
)
_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
}
_CV_SEED = 42
_BOOTSTRAP_REPLICATES = 2000
_BUCKETS = (
    ("0-9", -0.5, 9.5),
    ("10-29", 9.5, 29.5),
    ("30-79", 29.5, 79.5),
    ("80-149", 79.5, 149.5),
    ("150+", 149.5, np.inf),
)
_RF_N_JOBS = int(os.environ.get("DEFCON_RF_N_JOBS", "-1"))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _input_hashes():
    paths = {name: _DATA / name for name in _INPUTS}
    paths.update({
        "predict.py": _PREDICT,
        "recent_defcon.py": _PREDICT.with_name("recent_defcon.py"),
        "defender_model.py": _PREDICT.with_name("defender_model.py"),
    })
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required experiment inputs are missing: {missing}")
    return {name: sha256(path) for name, path in paths.items()}


def preserve_baseline():
    baseline = _OUT / "baseline"
    baseline.mkdir(parents=True, exist_ok=True)
    manifest_path = baseline / "preservation_manifest.json"
    inputs = _input_hashes()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["input_sha256"] != inputs:
            raise RuntimeError("Frozen experiment inputs changed after baseline preservation.")
        return manifest

    frozen_outputs = {}
    for name in _PRODUCTION_OUTPUTS:
        source = _DATA / name
        if not source.is_file():
            raise FileNotFoundError(f"Cannot preserve missing production output: {source}")
        destination = baseline / name
        shutil.copy2(source, destination)
        frozen_outputs[name] = sha256(destination)

    source_dir = _PREDICT.parent
    for name in ("predict.py", "recent_defcon.py", "defender_model.py"):
        shutil.copy2(source_dir / name, baseline / name)

    cache_dir = _ROOT / "cache" / "defcon"
    cache_files = {}
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.pkl.gz")):
            cache_files[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        fingerprint_file = cache_dir / "input_fingerprints.json"
        if fingerprint_file.is_file():
            cache_files["input_fingerprints.json"] = {
                "sha256": sha256(fingerprint_file),
                "bytes": fingerprint_file.stat().st_size,
            }

    manifest = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_sha256": inputs,
        "production_output_sha256": frozen_outputs,
        "preserved_source_sha256": {
            name: sha256(baseline / name)
            for name in ("predict.py", "recent_defcon.py", "defender_model.py")
        },
        "production_model_cache_inventory": cache_files,
        "cache_modified_by_experiment": False,
        "production_outputs_modified_by_experiment": False,
    }
    json_write(manifest_path, manifest)
    return manifest


def prepare_frames():
    source = _PREDICT.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_PREDICT))
    end = None
    for index, node in enumerate(tree.body):
        targets = []
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == "MID_FEATURE_COLS"
            for target in targets
        ):
            end = index
            break
    if end is None:
        raise RuntimeError("Could not locate the production MID_FEATURE_COLS assignment.")

    prep_body = []
    for node in tree.body[:end + 1]:
        names = {
            child.id for child in ast.walk(node)
            if isinstance(child, ast.Name)
        }
        if names.intersection({"df_def", "df_fwd"}):
            continue
        prep_body.append(node)
    prep = ast.Module(body=prep_body, type_ignores=[])
    namespace = {"__file__": str(_PREDICT), "__name__": "_midfielder_diagnostic_prep"}
    exec(compile(prep, str(_PREDICT), "exec"), namespace)
    required = {"matchlogs", "df_mid", "MID_FEATURE_COLS"}
    if not required.issubset(namespace):
        raise RuntimeError(f"Production data preparation omitted {required - namespace.keys()}")

    matchlogs = namespace["matchlogs"]
    frame = namespace["df_mid"].copy()
    features = list(namespace["MID_FEATURE_COLS"])
    if features != list(_FEATURE_FIELDS) + [
        col for col in features if col.startswith("Pos_")
    ]:
        raise AssertionError(f"Unexpected production RF_MF feature schema: {features}")
    if "IsForward" not in frame or not set(frame["IsForward"].unique()).issubset({0, 1}):
        raise AssertionError("Production midfielder/forward frame has an invalid IsForward flag.")
    frame["_RowOccurrence"] = frame.groupby(_KEYS, dropna=False, sort=False).cumcount()

    source_rows = matchlogs[
        matchlogs["IsMidfielderOrForward"]
        & ~matchlogs["IsForward"]
        & (matchlogs["Minutes"] >= 1)
    ].copy()
    source_rows["Target"] = (
        (source_rows["Per90_Mid"] >= 12)
        & (source_rows["Minutes"] >= 60)
    ).astype(int)
    source_rows = source_rows[_KEYS + [
        "Data_Source", "Minutes", "CBI", "Tkl", "Recov", "Target", "Per90_Mid",
    ]]
    source_rows["_RowOccurrence"] = source_rows.groupby(
        _KEYS, dropna=False, sort=False
    ).cumcount()

    midfield_frame = frame[frame["IsForward"] == 0].copy()
    if len(midfield_frame) != len(source_rows):
        raise AssertionError(
            f"Midfielder filter row mismatch: frame={len(midfield_frame)}, source={len(source_rows)}"
        )
    comparison = midfield_frame[_ROW_KEYS + ["Target"]].merge(
        source_rows[_ROW_KEYS + ["Target"]],
        on=_ROW_KEYS,
        how="outer",
        suffixes=("_frame", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    if not comparison["_merge"].eq("both").all():
        raise AssertionError("Midfielder match keys differ between built frame and source rows.")
    if not comparison["Target_frame"].eq(comparison["Target_source"]).all():
        raise AssertionError("Midfielder target labels differ from production source construction.")

    raw = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    scale_checks = {}
    raw_keys = raw[_KEYS + ["Data_Source", "CBI", "Tkl", "Recov"]].copy()
    raw_keys["_RowOccurrence"] = raw_keys.groupby(
        _KEYS, dropna=False, sort=False
    ).cumcount()
    raw_keys = raw_keys.rename(
        columns={"CBI": "CBI_raw", "Tkl": "Tkl_raw", "Recov": "Recov_raw"}
    )
    merged = source_rows.merge(
        raw_keys,
        on=_ROW_KEYS + ["Data_Source"],
        how="left",
        validate="one_to_one",
    )
    api = merged["Data_Source"].eq("FPL_API")
    if merged.loc[api, ["CBI_raw", "Tkl_raw", "Recov_raw"]].isna().any().any():
        raise AssertionError("Raw FPL_API source values failed to join to in-memory rows.")
    np.testing.assert_allclose(
        merged.loc[api, "CBI"].to_numpy(),
        merged.loc[api, "CBI_raw"].to_numpy() * 1.36,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        merged.loc[api, "Recov"].to_numpy(),
        merged.loc[api, "Recov_raw"].to_numpy() * 1.16,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        merged.loc[api, "Tkl"].to_numpy(),
        merged.loc[api, "Tkl_raw"].to_numpy(),
        rtol=0,
        atol=0,
    )
    scale_checks["FPL_API_midfielder_rows"] = int(api.sum())
    scale_checks["CBI_factor_applied_once"] = 1.36
    scale_checks["Recov_factor_applied_once"] = 1.16
    scale_checks["Tkl_unchanged"] = True
    scale_checks["raw_source_csv_unchanged"] = True

    frame_path = _OUT / "midfielder_frame.pkl"
    frame.to_pickle(frame_path)
    source_rows.to_pickle(_OUT / "midfielder_source_rows.pkl")
    json_write(_OUT / "frame_manifest.json", {
        "frame_sha256": sha256(frame_path),
        "source_rows_sha256": sha256(_OUT / "midfielder_source_rows.pkl"),
        "features": features,
        "row_count_shared_frame": len(frame),
        "row_count_midfielder": len(midfield_frame),
        "unique_midfielder_players": int(midfield_frame["Player"].nunique()),
        "target_positive_count_midfielder": int(midfield_frame["Target"].sum()),
        "target_positive_rate_midfielder": float(midfield_frame["Target"].mean()),
        "key_columns": _ROW_KEYS,
        "key_target_equality": True,
        "duplicate_base_keys_in_shared_frame": int(frame.duplicated(_KEYS).sum()),
        "duplicate_base_keys_in_midfielder_cohort": int(midfield_frame.duplicated(_KEYS).sum()),
        "duplicate_base_keys_in_source_midfielder_rows": int(source_rows.duplicated(_KEYS).sum()),
        "scale_checks": scale_checks,
        "input_sha256": _input_hashes(),
    })
    return frame, source_rows, features


def support_table(source_rows):
    data = source_rows.copy()
    data["CBI_per90"] = np.where(data["Minutes"] > 0, data["CBI"] * 90 / data["Minutes"], np.nan)
    data["Source"] = data["Data_Source"]
    records = []
    for year in sorted(data["Season"].unique()):
        for role, mask in (
            ("training", data["Season"] < year),
            ("validation", data["Season"] == year),
        ):
            selected = data[mask]
            if selected.empty:
                continue
            for (season, source), group in selected.groupby(["Season", "Source"], dropna=False):
                qualifying = group[group["Minutes"] >= 60]
                record = {
                    "validation_season": int(year),
                    "cohort": role,
                    "season": int(season),
                    "source": str(source),
                    "rows": int(len(group)),
                    "players": int(group["Player"].nunique()),
                    "positive_targets": int(group["Target"].sum()),
                    "target_rate": float(group["Target"].mean()),
                    "qualifying_minute_rows": int(len(qualifying)),
                    "qualifying_minute_coverage": float(len(qualifying) / len(group)),
                    "qualifying_cbi_zero_rows": int((qualifying["CBI"] == 0).sum()),
                    "qualifying_cbi_zero_share": (
                        float((qualifying["CBI"] == 0).mean()) if len(qualifying) else None
                    ),
                    "qualifying_cbi_per90_mean": (
                        float(qualifying["CBI_per90"].mean()) if len(qualifying) else None
                    ),
                    "qualifying_cbi_per90_q10": (
                        float(qualifying["CBI_per90"].quantile(0.10)) if len(qualifying) else None
                    ),
                    "qualifying_cbi_per90_median": (
                        float(qualifying["CBI_per90"].median()) if len(qualifying) else None
                    ),
                    "qualifying_cbi_per90_q90": (
                        float(qualifying["CBI_per90"].quantile(0.90)) if len(qualifying) else None
                    ),
                }
                records.append(record)
    return pd.DataFrame(records)


def assert_preserved(manifest):
    if _input_hashes() != manifest["input_sha256"]:
        raise RuntimeError("An experiment input changed after baseline preservation.")
    for name, expected in manifest["production_output_sha256"].items():
        if sha256(_DATA / name) != expected:
            raise RuntimeError(f"Production output changed during the diagnostic: {name}")
    cache_dir = _ROOT / "cache" / "defcon"
    actual_cache = {}
    if cache_dir.is_dir():
        for path in sorted(cache_dir.glob("*.pkl.gz")):
            actual_cache[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
        fingerprint_file = cache_dir / "input_fingerprints.json"
        if fingerprint_file.is_file():
            actual_cache["input_fingerprints.json"] = {
                "sha256": sha256(fingerprint_file),
                "bytes": fingerprint_file.stat().st_size,
            }
    if actual_cache != manifest["production_model_cache_inventory"]:
        raise RuntimeError("The production defcon model cache changed during the diagnostic.")


def _cluster_bootstrap(frame, values, seed=_CV_SEED, replicates=_BOOTSTRAP_REPLICATES):
    """Player-cluster percentile interval for the mean of row-level values."""
    values = np.asarray(values, dtype=float)
    players = frame["Player"].to_numpy()
    unique_players, inverse = np.unique(players, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_players), size=(replicates, len(unique_players)))
    sample_sum = sums[sampled].sum(axis=1)
    sample_count = counts[sampled].sum(axis=1)
    draws = sample_sum / sample_count
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def _metrics(frame, probability):
    y = frame["Target"].to_numpy(dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    ll_rows = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    brier_rows = (p - y) ** 2
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = None
    ece = 0.0
    for index in range(10):
        mask = (p >= index / 10) & (p < (index + 1) / 10)
        if index == 9:
            mask |= p == 1.0
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return {
        "rows": int(len(frame)),
        "players": int(frame["Player"].nunique()),
        "positive_targets": int(y.sum()),
        "observed_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": auc,
        "ece_10_equal_width": float(ece),
        "row_log_loss": ll_rows,
        "row_brier": brier_rows,
    }


def _calibration_buckets(frame, probability):
    data = frame[["Player", "Target", "MatchesPlayed"]].copy()
    data["probability"] = np.asarray(probability, dtype=float)
    output = {}
    for label, low, high in _BUCKETS:
        selected = data[(data["MatchesPlayed"] > low) & (data["MatchesPlayed"] <= high)]
        if selected.empty:
            continue
        gap = selected["probability"].to_numpy() - selected["Target"].to_numpy()
        output[label] = {
            "rows": int(len(selected)),
            "players": int(selected["Player"].nunique()),
            "positive_targets": int(selected["Target"].sum()),
            "observed_rate": float(selected["Target"].mean()),
            "mean_prediction": float(selected["probability"].mean()),
            "calibration_gap_prediction_minus_observed": float(gap.mean()),
            "calibration_gap_player_cluster_95ci": _cluster_bootstrap(selected, gap),
            "supported_for_decision": bool(
                len(selected) >= 100
                and selected["Player"].nunique() >= 30
                and selected["Target"].sum() >= 10
            ),
        }
    return output


def _fit_and_calibrate(train, val, features):
    x = train[features].to_numpy(dtype=float)
    y = train["Target"].to_numpy(dtype=int)
    xv = val[features].to_numpy(dtype=float)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=_CV_SEED)
    params = dict(_PARAMS, random_state=_CV_SEED, n_jobs=_RF_N_JOBS)
    estimator = RandomForestClassifier(**params)
    started = time.monotonic()
    oof = cross_val_predict(
        estimator,
        x_scaled,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.clip(oof, 1e-6, 1 - 1e-6), y)
    model = RandomForestClassifier(**params)
    model.fit(x_scaled, y)
    raw = model.predict_proba(scaler.transform(xv))[:, 1]
    probability = calibrator.predict(raw)
    return probability, time.monotonic() - started


def _save_development_gate(candidate_dir, protocol_bytes, manifest, results, candidate_rows, thresholds):
    predictions = pd.concat(candidate_rows, ignore_index=True)
    control = _metrics(predictions, predictions["control_probability"].to_numpy())
    candidate = _metrics(predictions, predictions["candidate_probability"].to_numpy())
    ll_delta = candidate.pop("row_log_loss") - control.pop("row_log_loss")
    brier_delta = candidate.pop("row_brier") - control.pop("row_brier")
    year_gates = {
        year: (
            results[str(year)]["delta_candidate_minus_control"]["log_loss"]
            <= thresholds["development_year_max_ll_regression"]
            and results[str(year)]["delta_candidate_minus_control"]["brier"]
            <= thresholds["development_year_max_brier_regression"]
        )
        for year in (2022, 2023, 2024)
    }
    passed = (
        ll_delta.mean() <= -thresholds["development_min_ll_improvement"]
        and brier_delta.mean() <= thresholds["development_max_brier_regression"]
        and candidate["auc"] is not None
        and control["auc"] is not None
        and candidate["auc"] - control["auc"]
        >= -thresholds["development_max_auc_regression"]
        and all(year_gates.values())
    )
    predictions.to_csv(
        candidate_dir / "development_candidate_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    json_write(candidate_dir / "development_gate.json", {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "splits": {str(year): results[str(year)] for year in (2022, 2023, 2024)},
        "development": {
            "control": control,
            "candidate": candidate,
            "delta_candidate_minus_control": {
                "log_loss": float(ll_delta.mean()),
                "brier": float(brier_delta.mean()),
                "auc": float(candidate["auc"] - control["auc"]),
                "log_loss_player_cluster_95ci": _cluster_bootstrap(predictions, ll_delta),
                "brier_player_cluster_95ci": _cluster_bootstrap(predictions, brier_delta),
            },
        },
        "development_year_guardrails": year_gates,
        "development_gate_passed": bool(passed),
        "confirmation_fits_performed": False,
        "production_proposal_gate_passed": False,
        "deployment_authorized": False,
        "production_changed": False,
        "cache_modified": False,
    })
    assert_preserved(manifest)
    return passed


def run_audit():
    preservation = preserve_baseline()
    if not _input_hashes() == preservation["input_sha256"]:
        raise RuntimeError("An input changed after preservation; stop before fitting.")
    frame, source_rows, features = prepare_frames()
    source_rows.to_csv(_OUT / "midfielder_source_support_rows.csv", index=False, encoding="utf-8-sig")
    support = support_table(source_rows)
    support.to_csv(_OUT / "support_by_season_source.csv", index=False, encoding="utf-8-sig")

    model_results = {}
    all_control_rows = []
    for year in _SPLITS:
        train = frame[frame["Season"] < year].copy()
        val = frame[(frame["Season"] == year) & (frame["IsForward"] == 0)].copy()
        if train.empty or val.empty:
            raise AssertionError(f"Required chronological split {year} has no training or validation rows.")
        train_mid = train[train["IsForward"] == 0].copy()
        if train_mid.empty or train["Target"].nunique() != 2 or train_mid["Target"].nunique() != 2:
            raise AssertionError(f"Split {year} has insufficient target classes to fit both arms.")
        if val[_ROW_KEYS].duplicated().any():
            raise AssertionError(f"Split {year} validation keys are not unique.")
        p, elapsed = _fit_and_calibrate(train, val, features)
        val["control_probability"] = p
        result = _metrics(val, p)
        result.pop("row_log_loss")
        result.pop("row_brier")
        result["runtime_seconds"] = float(elapsed)
        result["training_rows_combined"] = int(len(train))
        result["training_rows_midfielder_only"] = int(len(train_mid))
        result["training_players_combined"] = int(train["Player"].nunique())
        result["training_players_midfielder_only"] = int(train_mid["Player"].nunique())
        result["calibration_by_matches_played"] = _calibration_buckets(val, p)
        model_results[str(year)] = result
        all_control_rows.append(val[_ROW_KEYS + [
            "Target", "MatchesPlayed", "IsForward", "control_probability",
        ]])
        print(
            f"CONTROL {year}: rows={result['rows']} players={result['players']} "
            f"positives={result['positive_targets']} LL={result['log_loss']:.6f} "
            f"Brier={result['brier']:.6f} runtime={elapsed:.1f}s",
            flush=True,
        )

    predictions = pd.concat(all_control_rows, ignore_index=True)
    predictions.to_csv(_OUT / "baseline" / "control_predictions.csv", index=False, encoding="utf-8-sig")
    json_write(_OUT / "baseline" / "control_results.json", {
        "input_sha256": preservation["input_sha256"],
        "frame_sha256": sha256(_OUT / "midfielder_frame.pkl"),
        "feature_columns": features,
        "model": {
            "class": "RandomForestClassifier",
            "parameters": dict(_PARAMS, random_state=_CV_SEED, n_jobs=_RF_N_JOBS),
            "scaler": "StandardScaler fit on the chronological training rows",
            "calibration": "5-fold shuffled StratifiedKFold OOF isotonic; seed 42",
            "tuning": "disabled; locked §19 RF_MF parameters",
        },
        "splits": model_results,
        "production_changed": False,
        "cache_modified": False,
    })
    assert_preserved(preservation)
    print(f"Saved control baseline and audit under {_OUT}", flush=True)


def run_candidate():
    protocol_path = _OUT / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError("Freeze protocol.json after reviewing baseline audit before candidate scoring.")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if protocol.get("status") != "frozen_before_candidate":
        raise RuntimeError("Protocol status must be frozen_before_candidate before candidate scoring.")
    protocol_data = protocol.get("data", {})
    baseline = _OUT / "baseline"
    baseline_results = json.loads((baseline / "control_results.json").read_text(encoding="utf-8"))
    manifest = json.loads((baseline / "preservation_manifest.json").read_text(encoding="utf-8"))
    if protocol_data.get("input_sha256") != manifest["input_sha256"]:
        raise RuntimeError("Protocol input fingerprints differ from the preserved baseline.")
    if _input_hashes() != manifest["input_sha256"]:
        raise RuntimeError("A source input changed after control reconstruction; aborting candidate fit.")
    if protocol_data.get("control_results_sha256") != sha256(baseline / "control_results.json"):
        raise RuntimeError("Protocol does not fingerprint the reconstructed control results.")
    if protocol_data.get("frame_sha256") != sha256(_OUT / "midfielder_frame.pkl"):
        raise RuntimeError("Protocol does not fingerprint the audited feature frame.")
    if protocol_data.get("experiment_script_sha256") != sha256(Path(__file__)):
        raise RuntimeError("Experiment script changed after protocol preregistration.")
    if protocol_data.get("preservation_manifest_sha256") != sha256(
        baseline / "preservation_manifest.json"
    ):
        raise RuntimeError("Preservation manifest changed after protocol preregistration.")
    candidate_dir = _OUT / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_path = candidate_dir / "results.json"
    if result_path.exists():
        raise FileExistsError("Candidate results already exist; refusing to overwrite frozen outputs.")

    frame = pd.read_pickle(_OUT / "midfielder_frame.pkl")
    features = baseline_results["feature_columns"]
    controls = pd.read_csv(baseline / "control_predictions.csv", encoding="utf-8-sig")
    results = {}
    candidate_rows = []
    for year in _SPLITS:
        checkpoint = candidate_dir / f"split_{year}.json"
        checkpoint_rows = candidate_dir / f"split_{year}_predictions.csv"
        if checkpoint.exists() or checkpoint_rows.exists():
            if not checkpoint.is_file() or not checkpoint_rows.is_file():
                raise RuntimeError(f"Split {year} has an incomplete resume checkpoint.")
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("protocol_sha256") != hashlib.sha256(protocol_bytes).hexdigest():
                raise RuntimeError(f"Split {year} checkpoint belongs to a different protocol.")
            split_rows = pd.read_csv(checkpoint_rows, encoding="utf-8-sig")
            expected = controls[controls["Season"] == year]
            pd.testing.assert_frame_equal(
                split_rows[_ROW_KEYS + ["Target", "MatchesPlayed"]]
                .sort_values(_KEYS).reset_index(drop=True),
                expected[_ROW_KEYS + ["Target", "MatchesPlayed"]]
                .sort_values(_KEYS).reset_index(drop=True),
                check_dtype=False,
            )
            results[str(year)] = saved["result"]
            candidate_rows.append(split_rows)
            print(f"RESUMED {year} from frozen split checkpoint", flush=True)
            if year == 2024 and not _save_development_gate(
                candidate_dir,
                protocol_bytes,
                manifest,
                results,
                candidate_rows,
                protocol["decision_thresholds"],
            ):
                print("Development gate failed; confirmation fits were not started.", flush=True)
                return
            continue

        train_all = frame[frame["Season"] < year].copy()
        train = train_all[train_all["IsForward"] == 0].copy()
        val = frame[(frame["Season"] == year) & (frame["IsForward"] == 0)].copy()
        control = controls[controls["Season"] == year].copy()
        keys = _ROW_KEYS + ["Target", "MatchesPlayed"]
        pd.testing.assert_frame_equal(
            val[keys].sort_values(_KEYS).reset_index(drop=True),
            control[keys].sort_values(_KEYS).reset_index(drop=True),
            check_dtype=False,
        )
        p, elapsed = _fit_and_calibrate(train, val, features)
        paired = val.merge(
            control[_ROW_KEYS + ["Target", "MatchesPlayed", "control_probability"]],
            on=_ROW_KEYS,
            how="left",
            suffixes=("", "_control"),
            validate="one_to_one",
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            raise AssertionError(f"Split {year} failed exact control/candidate row pairing.")
        if not paired["Target"].eq(paired["Target_control"]).all():
            raise AssertionError(f"Split {year} control/candidate targets differ.")
        if not paired["MatchesPlayed"].eq(paired["MatchesPlayed_control"]).all():
            raise AssertionError(f"Split {year} evidence counts differ between arms.")
        val = val.copy()
        val["candidate_probability"] = p
        val["control_probability"] = paired["control_probability"].to_numpy()
        control_p = val["control_probability"].to_numpy()
        control_metrics = _metrics(val, control_p)
        candidate_metrics = _metrics(val, p)
        ll_delta = candidate_metrics.pop("row_log_loss") - control_metrics.pop("row_log_loss")
        brier_delta = candidate_metrics.pop("row_brier") - control_metrics.pop("row_brier")
        result = {
            "control": control_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_control": {
                "log_loss": float(ll_delta.mean()),
                "brier": float(brier_delta.mean()),
                "auc": (
                    float(candidate_metrics["auc"] - control_metrics["auc"])
                    if candidate_metrics["auc"] is not None and control_metrics["auc"] is not None
                    else None
                ),
                "mean_prediction": float(
                    candidate_metrics["mean_prediction"] - control_metrics["mean_prediction"]
                ),
                "ece_10_equal_width": float(
                    candidate_metrics["ece_10_equal_width"] - control_metrics["ece_10_equal_width"]
                ),
                "log_loss_player_cluster_95ci": _cluster_bootstrap(val, ll_delta),
                "brier_player_cluster_95ci": _cluster_bootstrap(val, brier_delta),
            },
            "runtime_seconds": float(elapsed),
            "training_rows_candidate": int(len(train)),
            "training_rows_control": int(len(train_all)),
            "calibration_by_matches_played": _calibration_buckets(val, p),
        }
        control_buckets = baseline_results["splits"][str(year)]["calibration_by_matches_played"]
        calibration_comparison = {}
        for bucket, candidate_bucket in result["calibration_by_matches_played"].items():
            control_bucket = control_buckets.get(bucket)
            if control_bucket is None:
                continue
            calibration_comparison[bucket] = {
                "supported_for_decision": candidate_bucket["supported_for_decision"]
                and control_bucket["supported_for_decision"],
                "control_abs_gap": abs(control_bucket["calibration_gap_prediction_minus_observed"]),
                "candidate_abs_gap": abs(candidate_bucket["calibration_gap_prediction_minus_observed"]),
                "candidate_minus_control_abs_gap": (
                    abs(candidate_bucket["calibration_gap_prediction_minus_observed"])
                    - abs(control_bucket["calibration_gap_prediction_minus_observed"])
                ),
            }
        result["calibration_bucket_comparison"] = calibration_comparison
        results[str(year)] = result
        split_rows = val[_ROW_KEYS + [
            "Target", "MatchesPlayed", "candidate_probability", "control_probability",
        ]]
        candidate_rows.append(split_rows)
        split_rows.to_csv(checkpoint_rows, index=False, encoding="utf-8-sig")
        json_write(checkpoint, {
            "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "result": result,
        })
        print(
            f"CANDIDATE {year}: LL delta={ll_delta.mean():+.6f} "
            f"Brier delta={brier_delta.mean():+.6f} runtime={elapsed:.1f}s",
            flush=True,
        )
        if year == 2024:
            if not _save_development_gate(
                candidate_dir,
                protocol_bytes,
                manifest,
                results,
                candidate_rows,
                protocol["decision_thresholds"],
            ):
                print("Development gate failed; confirmation fits were not started.", flush=True)
                return

    predictions = pd.concat(candidate_rows, ignore_index=True)
    predictions.to_csv(candidate_dir / "candidate_predictions.csv", index=False, encoding="utf-8-sig")
    pooled_results = {}
    for label, years in (
        ("development", (2022, 2023, 2024)),
        ("confirmation", (2025, 2026)),
    ):
        subset = predictions[predictions["Season"].isin(years)].copy()
        control_metrics = _metrics(subset, subset["control_probability"].to_numpy())
        candidate_metrics = _metrics(subset, subset["candidate_probability"].to_numpy())
        ll_delta = (
            candidate_metrics.pop("row_log_loss") - control_metrics.pop("row_log_loss")
        )
        brier_delta = (
            candidate_metrics.pop("row_brier") - control_metrics.pop("row_brier")
        )
        pooled_results[label] = {
            "control": control_metrics,
            "candidate": candidate_metrics,
            "delta_candidate_minus_control": {
                "log_loss": float(ll_delta.mean()),
                "brier": float(brier_delta.mean()),
                "auc": (
                    float(candidate_metrics["auc"] - control_metrics["auc"])
                    if candidate_metrics["auc"] is not None and control_metrics["auc"] is not None
                    else None
                ),
                "log_loss_player_cluster_95ci": _cluster_bootstrap(subset, ll_delta),
                "brier_player_cluster_95ci": _cluster_bootstrap(subset, brier_delta),
            },
            "calibration_by_matches_played": _calibration_buckets(
                subset, subset["candidate_probability"].to_numpy()
            ),
        }

    thresholds = protocol["decision_thresholds"]
    development = pooled_results["development"]["delta_candidate_minus_control"]
    confirmation = pooled_results["confirmation"]["delta_candidate_minus_control"]
    development_year_gates = {
        year: {
            "log_loss_within_guardrail": (
                results[str(year)]["delta_candidate_minus_control"]["log_loss"]
                <= thresholds["development_year_max_ll_regression"]
            ),
            "brier_within_guardrail": (
                results[str(year)]["delta_candidate_minus_control"]["brier"]
                <= thresholds["development_year_max_brier_regression"]
            ),
        }
        for year in (2022, 2023, 2024)
    }
    confirmation_year_gates = {
        year: {
            "log_loss_within_guardrail": (
                results[str(year)]["delta_candidate_minus_control"]["log_loss"]
                <= thresholds["confirmation_year_max_ll_regression"]
            ),
            "brier_within_guardrail": (
                results[str(year)]["delta_candidate_minus_control"]["brier"]
                <= thresholds["confirmation_year_max_brier_regression"]
            ),
            "auc_within_guardrail": (
                results[str(year)]["delta_candidate_minus_control"]["auc"] is not None
                and results[str(year)]["delta_candidate_minus_control"]["auc"]
                >= -thresholds["confirmation_year_max_auc_regression"]
            ),
        }
        for year in (2025, 2026)
    }
    supported_bucket_regressions = []
    for year in _SPLITS:
        for bucket, comparison in results[str(year)]["calibration_bucket_comparison"].items():
            if (
                comparison["supported_for_decision"]
                and comparison["candidate_minus_control_abs_gap"]
                > thresholds["material_calibration_gap_regression"]
            ):
                supported_bucket_regressions.append({"season": year, "bucket": bucket})
    development_gate = (
        development["log_loss"] <= -thresholds["development_min_ll_improvement"]
        and development["brier"] <= thresholds["development_max_brier_regression"]
        and development["auc"] is not None
        and development["auc"] >= -thresholds["development_max_auc_regression"]
        and all(all(gate.values()) for gate in development_year_gates.values())
    )
    confirmation_robustness_gate = all(
        all(gate.values()) for gate in confirmation_year_gates.values()
    )
    production_proposal_gate = (
        development_gate
        and confirmation_robustness_gate
        and confirmation["log_loss"] <= -thresholds["confirmation_min_ll_improvement"]
        and confirmation["log_loss_player_cluster_95ci"][1] < 0
        and not supported_bucket_regressions
    )
    json_write(result_path, {
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "experiment_script_sha256": sha256(Path(__file__)),
        "frame_sha256": sha256(_OUT / "midfielder_frame.pkl"),
        "splits": results,
        "pooled": pooled_results,
        "decision_gates": {
            "development_years": development_year_gates,
            "confirmation_years": confirmation_year_gates,
            "supported_calibration_bucket_regressions": supported_bucket_regressions,
            "development_gate_passed": bool(development_gate),
            "confirmation_robustness_gate_passed": bool(confirmation_robustness_gate),
            "production_proposal_gate_passed": bool(production_proposal_gate),
            "deployment_authorized": False,
            "deployment_requires_separate_user_decision": True,
        },
        "production_changed": False,
        "cache_modified": False,
    })
    assert_preserved(manifest)
    print(f"Saved candidate scores under {_OUT}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "candidate"))
    args = parser.parse_args()
    if args.stage == "audit":
        run_audit()
    else:
        run_candidate()


if __name__ == "__main__":
    main()
