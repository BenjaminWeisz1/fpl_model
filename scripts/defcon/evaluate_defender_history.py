"""Additional registered chronological comparison to make the §43 decision now.

Fits the two already-locked configurations on earlier seasons, validates on
2022, 2023 and 2024, and preserves existing 2025/26 evidence separately. No
production writes, retuning, future-outcome claims or changes to the GW8–12 study.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import ast
import gzip
import hashlib
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import sklearn

from defender_model import monotonic_params
from evaluate_defender_monotonic import OUT, freeze_control, metrics, predict, audit, bootstrap


def parameters():
    # Read the preserved exact legacy specification, rather than duplicate it.
    source = (OUT / "baseline" / "predict.py").read_text(encoding="utf-8-sig")
    scope = {"GradientBoostingClassifier": GradientBoostingClassifier}
    # The GBM entry is an AST literal dict in the preserved _MODEL_CONFIGS.
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_MODEL_CONFIGS" for t in node.targets):
            for key, value in zip(node.value.keys, node.value.values):
                if ast.literal_eval(key) == "GBM":
                    return ast.literal_eval(value.elts[2])
    raise ValueError("Legacy configuration not found")


def fit(train, cols, name, params, year):
    x, y = train[cols].values, train.Target.values
    cls = GradientBoostingClassifier if name == "legacy" else HistGradientBoostingClassifier
    key = hashlib.sha256(x.tobytes()+y.tobytes()+json.dumps(params, sort_keys=True).encode()
        +f"{cls.__name__}|sklearn={sklearn.__version__}|numpy={np.__version__}|cv=5|rs=42".encode()).hexdigest()
    path = OUT / f"history_{name}_{year}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as stream:
            payload = pickle.load(stream)
        assert payload["history_key"] == key
        return payload
    started = time.monotonic()
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    model = cls(random_state=42, **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, scaled, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(np.clip(oof, 1e-6, 1-1e-6), y)
    model.fit(scaled, y)
    payload = {"base_model": model, "calibrator": cal, "scaler": scaler,
               "best_params": params, "history_key": key}
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream)
    print(f"Fitted {name}, validate {year}: {len(y)} training rows, {time.monotonic()-started:.1f}s", flush=True)
    return payload


def summarize(val, old, new):
    return {"legacy": metrics(val.Target.values, old), "updated": metrics(val.Target.values, new),
            "paired_player_bootstrap": bootstrap(val, old, new)}


def main():
    freeze_control()
    registration = OUT / "historical_preregistration.json"
    protocol = json.loads(registration.read_text(encoding="utf-8"))
    frame_path = OUT / "defender_frame.pkl"
    frame = pd.read_pickle(frame_path)
    cols = ["AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed", "Recent5AvgPer90",
            "Recent5AvgRaw", "PctAboveThreshold", "OpponentAvgStat", "TeamAvgStat",
            "VenueAvg", "IsHome", "Is_FPL_API", "Pos_CB", "Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]
    output = {"preregistration_sha256": hashlib.sha256(registration.read_bytes()).hexdigest(),
              "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
              "legacy_params": parameters(), "updated_params": monotonic_params(cols), "splits": {}}
    all_rows = []
    for year in protocol["additional_validation_seasons"]:
        train, val = frame[frame.Season < year], frame[frame.Season == year].copy()
        assert train.Season.max() < val.Season.min()
        print(f"Chronological split: train {int(train.Season.min())}–{int(train.Season.max())}, validate {year} ({len(val)} rows)", flush=True)
        legacy = fit(train, cols, "legacy", output["legacy_params"], year)
        updated = fit(train, cols, "updated", output["updated_params"], year)
        old, new = predict(legacy, val[cols].values), predict(updated, val[cols].values)
        cell = summarize(val, old, new)
        cell["legacy_audit"], cell["updated_audit"] = audit(legacy, val, cols), audit(updated, val, cols)
        output["splits"][str(year)] = cell
        rows = val[["Player", "Season", "MatchID", "Target", "MatchesPlayed"]].copy()
        rows["legacy"], rows["updated"] = old, new
        rows.to_csv(OUT / f"history_validation_{year}.csv", index=False, encoding="utf-8-sig")
        all_rows.append(rows)
        print(json.dumps({year: cell}, indent=2), flush=True)
    pooled = pd.concat(all_rows, ignore_index=True)
    output["pooled_additional"] = summarize(pooled, pooled.legacy.values, pooled.updated.values)
    old, new = output["pooled_additional"]["legacy"], output["pooled_additional"]["updated"]
    ci = output["pooled_additional"]["paired_player_bootstrap"]
    deterioration = ((new["log_loss"]-old["log_loss"] > .002 and ci["log_loss_delta_95ci"][0] > 0) or
                     (new["brier"]-old["brier"] > .002 and ci["brier_delta_95ci"][0] > 0))
    invariant_failure = any(any(v["raw"] or v["calibrated"] for v in s["updated_audit"]["single_feature_violations"].values())
                            or s["updated_audit"]["backwards_joint_updates"] for s in output["splits"].values())
    output["decision"] = "review_rollback_now_no_parameter_search" if deterioration or invariant_failure else "keep_update_now_future_check_is_monitoring"
    output["clear_deterioration"], output["invariant_failure"] = bool(deterioration), bool(invariant_failure)
    output["existing_modern_evidence"] = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["splits"]
    output["production_changed"] = False
    (OUT / "historical_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({"pooled": output["pooled_additional"], "decision": output["decision"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
