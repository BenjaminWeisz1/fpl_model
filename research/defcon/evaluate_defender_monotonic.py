"""Evaluate the pre-registered 2026-10-20 defender replacement in isolation.

Reuses production ingestion/features and saved incumbent models without running
the pipeline or writing production outputs. Freezes the control before any edit
to predict.py. Run with python -u -X utf8 -m research.defcon.evaluate_defender_monotonic.
"""
import os

from fpl_model import paths

os.environ.setdefault("OMP_NUM_THREADS", "2")
import ast
import gzip
import hashlib
import json
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from fpl_model.defcon.defender_model import POSITIVE_EVIDENCE, monotonic_params

HERE = Path(__file__).resolve().parent
ROOT = paths.ROOT
OUT = paths.VALIDATION / "defcon_experiment_20261020"
DATA = paths.CURRENT
OUT.mkdir(exist_ok=True)
BASELINE = paths.VALIDATION / "defcon_experiment_20261020" / "baseline"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_control():
    if BASELINE.exists():
        manifest = json.loads((BASELINE / "manifest.json").read_text())
        assert digest(OUT / "preregistration.json") == manifest["preregistration_sha256"], "Pre-registration changed"
        for name, expected in manifest["inputs"].items():
            assert digest(paths.current_file(name, root=DATA)) == expected, f"Input changed: {name}"
        return
    BASELINE.mkdir()
    shutil.copy2(HERE / "predict.py", BASELINE / "predict.py")
    for name in ("predictions_defensive.csv", "defcon_average_probabilities.csv", "player_ratings.csv"):
        shutil.copy2(paths.current_file(name, root=DATA), BASELINE / name)
    models = BASELINE / "models"
    models.mkdir()
    for p in (ROOT / "cache" / "defcon").glob("*.pkl.gz"):
        shutil.copy2(p, models / p.name)
    inputs = ["player_match_stats.csv", "players_current.csv", "transfers.csv",
              "player_match_positions.csv", "fixtures.csv", "match_results.csv",
              "minutes_probabilities.csv"]
    manifest = {"inputs": {n: digest(paths.current_file(n, root=DATA)) for n in inputs},
                "outputs": {p.name: digest(p) for p in BASELINE.glob("*.csv")},
                "source_sha256": digest(BASELINE / "predict.py"),
                "preregistration_sha256": digest(OUT / "preregistration.json")}
    (BASELINE / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_scope():
    source = (BASELINE / "predict.py").read_text(encoding="utf-8-sig")
    scope = {"__file__": str(HERE / "predict.py"), "__name__": "defender_experiment"}
    exec(compile(source.split('print("Building defender dataset...")')[0], str(HERE / "predict.py"), "exec"), scope)
    required = {"BASE_FEATURE_COLS", "DEF_FEATURE_COLS", "GBM_PARAM_GRID", "RF_PARAM_GRID",
                "_MODEL_CONFIGS", "_model_cache_key", "_PREDICT_SOURCE_MD5"}
    for node in ast.parse(source).body:
        names = ({node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else
                 {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set())
        if names & required:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<baseline definitions>", "exec"), scope)
    scope["_PREDICT_SOURCE_MD5"] = hashlib.md5((BASELINE / "predict.py").read_bytes()).hexdigest()
    return scope


def predict(payload, x, raw=False):
    scaled = payload["scaler"].transform(x)
    p = payload["base_model"].predict_proba(scaled)[:, 1]
    return p if raw else payload["calibrator"].predict(p)


def incumbent(scope, train, label):
    cols = scope["DEF_FEATURE_COLS"]
    cls, _, params = scope["_MODEL_CONFIGS"]["GBM"]
    key = scope["_model_cache_key"](train[cols].values, train.Target.values,
        cls, label, False, params, 42, 5, 30)
    paths = list((BASELINE / "models").glob(f"*__{key}.pkl.gz"))
    assert len(paths) == 1, f"Control cache mismatch: {label}, {key}"
    with gzip.open(paths[0], "rb") as f:
        return pickle.load(f)


def fit_candidate(train, cols, name):
    path = OUT / f"candidate_{name}.pkl.gz"
    x, y = train[cols].values, train.Target.values
    key = hashlib.sha256(x.tobytes() + y.tobytes() +
        json.dumps(monotonic_params(cols), sort_keys=True).encode()).hexdigest()
    if path.exists():
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        assert payload["experiment_key"] == key
        return payload
    start = time.monotonic()
    scaler = StandardScaler().fit(x)
    scaled = scaler.transform(x)
    params = monotonic_params(cols)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = HistGradientBoostingClassifier(random_state=42, **params)
    oof = cross_val_predict(model, scaled, y, cv=cv, method="predict_proba")[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(np.clip(oof, 1e-6, 1-1e-6), y)
    model.fit(scaled, y)
    payload = {"base_model": model, "scaler": scaler, "calibrator": cal,
               "best_params": params, "experiment_key": key}
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Candidate {name}: {len(y)} rows, {time.monotonic()-start:.1f}s", flush=True)
    return payload


def metrics(y, p):
    ece = 0.0
    bins = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return {"n": len(y), "hits": int(y.sum()), "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p),
            "ece": float(ece), "mean_prediction": float(p.mean()), "observed": float(y.mean())}


def audit(payload, val, cols):
    x = val[cols].values
    base = predict(payload, x)
    raw = predict(payload, x, raw=True)
    violations = {}
    active_evidence = [c for c in POSITIVE_EVIDENCE if c in cols]
    for col in active_evidence:
        shifted = x.copy()
        j = cols.index(col)
        shifted[:, j] += 0.001 if col == "PctAboveThreshold" else 0.1
        if col == "PctAboveThreshold":
            shifted[:, j] = np.minimum(shifted[:, j], 1)
        violations[col] = {"calibrated": int((predict(payload, shifted) < base - 1e-12).sum()),
                           "raw": int((predict(payload, shifted, raw=True) < raw - 1e-12).sum())}
    ordered = val.sort_values(["Player", "MatchID"]).reset_index(drop=True)
    next_evidence = ordered.groupby("Player")[active_evidence].shift(-1)
    good = next_evidence.notna().all(axis=1)
    old = ordered.loc[good, cols].copy()
    updated = old.copy()
    updated.loc[:, active_evidence] = next_evidence.loc[good].values
    delta = predict(payload, updated.values) - predict(payload, old.values)
    fallen = (updated[active_evidence].values <= old[active_evidence].values + 1e-12).all(axis=1)
    return {"single_feature_violations": violations, "weekly_pairs": len(delta),
            "weekly_abs_p50": float(np.quantile(abs(delta), .5)),
            "weekly_abs_p95": float(np.quantile(abs(delta), .95)),
            "weekly_abs_max": float(abs(delta).max()),
            "all_evidence_fell_pairs": int(fallen.sum()),
            "backwards_joint_updates": int(((delta > 1e-12) & fallen).sum())}


def bootstrap(val, p_control, p_candidate):
    # Paired player-cluster bootstrap: player histories are not independent rows.
    y = val.Target.to_numpy()
    eps = np.finfo(float).eps
    pc, pn = np.clip(p_control, eps, 1-eps), np.clip(p_candidate, eps, 1-eps)
    diffs = pd.DataFrame({"Player": val.Player.to_numpy(),
        "ll": -(y*np.log(pn)+(1-y)*np.log1p(-pn)) + (y*np.log(pc)+(1-y)*np.log1p(-pc)),
        "brier": (y-pn)**2-(y-pc)**2, "n": 1}).groupby("Player").sum().values
    rng = np.random.default_rng(42)
    sampled = diffs[rng.integers(0, len(diffs), size=(2000, len(diffs)))].sum(axis=1)
    return {name: np.quantile(sampled[:, j]/sampled[:, 2], [.025, .975]).tolist()
            for j, name in enumerate(["log_loss_delta_95ci", "brier_delta_95ci"])}


def tarkowski(scope, control, candidate):
    logs = scope["matchlogs"]
    hist = logs[logs.Player.eq("James Tarkowski")].sort_values("MatchID")
    feats = []
    for frame in (logs.drop(hist.tail(1).index), logs):
        scope["matchlogs"] = frame
        feats.append(scope["precompute_player_features"](frame.IsDefender, "DefStats_Def", "Per90_Def", 10,
            scope["DEFENDER_DETAILED_POS"], priors=True, prior_season=2026,
            std_pct_estimator=scope["def_std_estimator"])["James Tarkowski"])
    scope["matchlogs"] = logs
    opp, team, venue = scope["def_final_opp"].copy(), scope["def_final_team"], scope["def_final_venue"]
    for t in {"Hull City", "Coventry City", "Ipswich Town"}:
        if t not in opp:
            opp[t] = np.mean(list(opp.values()))/1.067
    population = np.mean([opp.get(t, np.mean(list(opp.values()))) for t in scope["players_current"].Team.unique()])
    cols = scope["DEF_FEATURE_COLS"]
    def x(feat):
        return pd.DataFrame([{**feat, "OpponentAvgStat": population, "TeamAvgStat": team["Everton"],
            "VenueAvg": venue[v], "IsHome": int(v == "Home")} for v in ("Home", "Away")])[cols].values
    before, after = feats
    evidence_after = {**before, **{c: after[c] for c in POSITIVE_EVIDENCE if c in after}}
    result = {}
    for name, payload in (("control", control), ("candidate", candidate)):
        result[name] = {"before_personal": float(predict(payload, x(before)).mean()),
            "after_personal": float(predict(payload, x(after)).mean()),
            "after_evidence_only": float(predict(payload, x(evidence_after)).mean())}
    assert round(result["control"]["after_personal"], 4) == .695, result
    return result


def main():
    freeze_control()
    scope = load_scope()
    cols = scope["DEF_FEATURE_COLS"]
    dataset = OUT / "defender_frame.pkl"
    if dataset.exists():
        frame = pd.read_pickle(dataset)
    else:
        print("Building unchanged defender training frame...", flush=True)
        frame = scope["build_dataset"](scope["matchlogs"], ["IsDefender"], "DefStats_Def", "Per90_Def", 10,
            scope["def_prior_opp"], scope["def_prior_team"], scope["def_prior_venue"],
            scope["DEFENDER_DETAILED_POS"], std_pct_estimator=scope["def_std_estimator"])
        frame.to_pickle(dataset)
    result = {"preregistration_sha256": digest(OUT / "preregistration.json"), "splits": {}}
    for year in (2025, 2026):
        train, val = frame[frame.Season < year], frame[frame.Season == year]
        label = f"Defender (Season {year}" + (", priors ON)" if year == 2026 else ")")
        control = incumbent(scope, train, label)
        candidate = fit_candidate(train, cols, str(year))
        p_control, p_candidate = predict(control, val[cols].values), predict(candidate, val[cols].values)
        scores = {"control": metrics(val.Target.values, p_control), "candidate": metrics(val.Target.values, p_candidate),
                  "control_audit": audit(control, val, cols), "candidate_audit": audit(candidate, val, cols),
                  "paired_player_bootstrap": bootstrap(val, p_control, p_candidate)}
        result["splits"][str(year)] = scores
        rows = val[["Player", "MatchID", "Season", "Target", "MatchesPlayed"]].copy()
        rows["control"], rows["candidate"] = p_control, p_candidate
        rows.to_csv(OUT / f"validation_{year}.csv", index=False, encoding="utf-8-sig")
        print(json.dumps({year: scores}, indent=2), flush=True)
    control = incumbent(scope, frame, "Defender")
    candidate = fit_candidate(frame, cols, "production")
    result["tarkowski"] = tarkowski(scope, control, candidate)
    cells = list(result["splits"].values())
    gates = {
        "log_loss_improvement": any(s["candidate"]["log_loss"] <= s["control"]["log_loss"]-.001 for s in cells),
        "log_loss_guardrail": all(s["candidate"]["log_loss"] <= s["control"]["log_loss"]+.002 for s in cells),
        "brier_guardrail": all(s["candidate"]["brier"] <= s["control"]["brier"]+.002 for s in cells),
        "auc_guardrail": all(s["candidate"]["auc"] >= s["control"]["auc"]-.01 for s in cells),
        "monotonicity": all(all(v["raw"] == v["calibrated"] == 0 for v in s["candidate_audit"]["single_feature_violations"].values())
                            and s["candidate_audit"]["backwards_joint_updates"] == 0 for s in cells),
        "tarkowski_evidence": result["tarkowski"]["candidate"]["after_evidence_only"] <= result["tarkowski"]["candidate"]["before_personal"]+1e-12}
    result["gates"], result["ship"] = gates, all(gates.values())
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({"tarkowski": result["tarkowski"], "gates": gates, "ship": result["ship"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
