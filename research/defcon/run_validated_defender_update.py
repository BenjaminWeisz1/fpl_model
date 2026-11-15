"""Run predict.py, reusing the exact fits already preserved/validated in §41.

This one-time experiment runner verifies X/y/parameters/version cache keys for
unchanged random forests and experiment training hashes for the three fitted
defender candidates. It primes ordinary production cache entries on demand.
Subsequent weekly runs use predict.py normally. No monkeypatch is persisted.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import ast
import gzip
import hashlib
import inspect
import json
import pickle
from pathlib import Path

from fpl_model.defcon.defender_model import monotonic_params
from research.defcon.evaluate_defender_monotonic import BASELINE, OUT, freeze_control

HERE = Path(__file__).resolve().parent


def main():
    freeze_control()
    results = json.loads((OUT / "results.json").read_text())
    assert results["ship"] and all(results["gates"].values())
    assert os.environ.get("DEFCON_MONOTONIC_DEF", "1") == "1"
    assert os.environ.get("DEFCON_TUNE", "0") == "0"
    assert os.environ.get("DEFCON_MODEL_CACHE", "1") == "1"
    baseline_hash = hashlib.md5((BASELINE / "predict.py").read_bytes()).hexdigest()
    scope = {"__file__": str(HERE / "predict.py"), "__name__": "__main__"}
    reused = []

    def install_reuse():
        original = scope["tune_and_calibrate"]
        signature = inspect.signature(original)

        def fit_or_reuse(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            assert not a["do_tune"]
            cls, params = a["model_cls"], a["default_params"]
            key_args = [a[k] for k in ("X", "y", "model_cls", "label", "do_tune", "default_params", "random_state", "cv", "n_iter")]
            current_key = scope["_model_cache_key"](*key_args)
            current_path = scope["_cache_path_for"](a["label"], current_key)
            if not current_path.exists():
                if cls is scope["RandomForestClassifier"]:
                    current_hash = scope["_PREDICT_SOURCE_MD5"]
                    try:
                        scope["_PREDICT_SOURCE_MD5"] = baseline_hash
                        old_key = scope["_model_cache_key"](*key_args)
                    finally:
                        scope["_PREDICT_SOURCE_MD5"] = current_hash
                    paths = list((BASELINE / "models").glob(f"*__{old_key}.pkl.gz"))
                    assert len(paths) == 1, f"Unchanged-head training inputs differ: {a['label']}"
                    with gzip.open(paths[0], "rb") as f:
                        payload = pickle.load(f)
                    assert type(payload["base_model"]) is cls
                else:
                    assert cls is scope["HistGradientBoostingClassifier"]
                    assert params == monotonic_params(scope["DEF_FEATURE_COLS"])
                    assert (a["random_state"], a["cv"], a["n_iter"]) == (42, 5, 30)
                    tag = "2025" if "2025" in a["label"] else "2026" if "2026" in a["label"] else "production"
                    with gzip.open(OUT / f"candidate_{tag}.pkl.gz", "rb") as f:
                        payload = pickle.load(f)
                    expected = hashlib.sha256(a["X"].tobytes() + a["y"].tobytes() +
                        json.dumps(params, sort_keys=True).encode()).hexdigest()
                    assert payload["experiment_key"] == expected, f"Candidate inputs differ: {a['label']}"
                # Save plain sklearn objects using the standard production writer.
                scope["_initialise_model_cache"]()
                scope["_store_cached_model"](current_path, a["label"], payload["base_model"],
                    payload["calibrator"], payload["scaler"], payload["best_params"])
                reused.append({"label": a["label"], "new_key": current_key,
                               "kind": "unchanged_RF" if cls is scope["RandomForestClassifier"] else "validated_defender"})
                print(f"Verified existing fit reused for {a['label']}", flush=True)
            return original(*args, **kwargs)

        scope["tune_and_calibrate"] = fit_or_reuse

    scope["_install_verified_reuse"] = install_reuse
    tree = ast.parse((HERE / "predict.py").read_text(encoding="utf-8-sig"))
    body = []
    for node in tree.body:
        body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name == "tune_and_calibrate":
            body.extend(ast.parse("_install_verified_reuse()").body)
    tree.body = body
    exec(compile(ast.fix_missing_locations(tree), str(HERE / "predict.py"), "exec"), scope)
    assert len(reused) == 9, f"Expected nine verified fits: {reused}"
    actual = scope["avg_out"].set_index("Player").loc["James Tarkowski", "Defcon"]
    assert actual == round(results["tarkowski"]["candidate"]["after_personal"], 4)
    old = __import__("pandas").read_csv(BASELINE / "defcon_average_probabilities.csv", encoding="utf-8-sig")
    new = scope["avg_out"]
    old_mf = old[old.Pos.isin(["M", "F"])].sort_values(["Pos", "Player"]).reset_index(drop=True)
    new_mf = new[new.Pos.isin(["M", "F"])].sort_values(["Pos", "Player"]).reset_index(drop=True)
    __import__("pandas").testing.assert_frame_equal(old_mf, new_mf)
    old_fixture = __import__("pandas").read_csv(BASELINE / "predictions_defensive.csv", encoding="utf-8-sig")
    new_fixture = __import__("pandas").read_csv(scope["_DATA"] / "predictions_defensive.csv", encoding="utf-8-sig")
    old_fixture = old_fixture[old_fixture.Pos.isin(["M", "F"])].reset_index(drop=True)
    new_fixture = new_fixture[new_fixture.Pos.isin(["M", "F"])].reset_index(drop=True)
    __import__("pandas").testing.assert_frame_equal(old_fixture, new_fixture)
    (OUT / "integration.json").write_text(json.dumps({"reused": reused,
        "tarkowski": float(actual), "midfield_forward_outputs_unchanged": True,
        "production_source_sha256": hashlib.sha256((HERE / "predict.py").read_bytes()).hexdigest()}, indent=2))
    print("Integration verified: defender rating matches experiment; midfielder/forward outputs identical.")


if __name__ == "__main__":
    main()
