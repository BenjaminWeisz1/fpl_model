"""Freeze and score the §42 prospective defender comparison, GW8–12 of 2026.

freeze: preserve both already-fitted models and identical-input forecasts before
any study outcomes enter local data. Existing snapshots are never overwritten.
score: read the frozen forecasts and newly completed local match data, report
paired scores, and never fit, retune, replace forecasts, or change production.

The production model continues its normal weekly updates separately. This is a
fixed-fit/fixed-feature forecast test, not a weekly-refit comparison. All paths
resolve from this script. No network access or scheduler is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "current"
EXPERIMENT = Path(__file__).resolve().parent / "experiment_20261020"
STUDY = ROOT / "data" / "validation" / "defcon_forward_20261020"
MATCH_KEY = ["Season", "Home", "Away"]
PLAYER_KEY = ["Season", "Team", "Opponent", "Venue", "Player"]
FEATURES = ["AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed", "Recent5AvgPer90",
            "Recent5AvgRaw", "PctAboveThreshold", "OpponentAvgStat", "TeamAvgStat",
            "VenueAvg", "IsHome", "Is_FPL_API", "Pos_CB", "Pos_FB", "Pos_WB", "Pos_LB", "Pos_RB"]

# Registered before looking at any GW8–12 outcome. No score-based stopping or
# parameter selection is allowed. See the generated immutable protocol.json.
PROTOCOL = {
    "study_id": "defcon_forward_20261020",
    "season": 2026, "training_cutoff_round": 7, "rounds": [8, 9, 10, 11, 12],
    "design": "Freeze both fitted models, calibration maps, identical inputs and all five rounds of predictions now. No future refitting inside the study.",
    "production": "Monotonic update remains default; its normal weekly refits are separate from these frozen forecasts.",
    "expected": "Monotonic minus legacy log loss -0.001 to -0.010; Brier no worse by more than 0.002. A noisy or inconclusive five-round result is plausible.",
    "primary_population": "Frozen defender cohort with an observed positive-minute appearance in a completed study fixture; no appearance is excluded, not scored as a failed threshold.",
    "primary_target": "Exact training target: Minutes >= 60 and (CBI + Tkl) * 90 / Minutes >= 10. Defender source reconciliation leaves CBI and Tkl unchanged.",
    "secondary_population": "Same forecasts restricted to Minutes >= 60, using the same target.",
    "not_scored": "The stale minutes-scaled Probability column, non-appearances, and raw-count FPL points are not the primary target.",
    "metrics": ["log_loss_primary", "brier", "auc_if_two_classes", "fixed_width_10_bin_ECE", "calibration_bins", "by_round_scores"],
    "uncertainty": "2000 paired cluster bootstrap resamples, random_state 42; separately by player and fixture. Both intervals must support a direction before calling it convincing. Neither separately models every crossed dependency.",
    "assessment": {
        "window": "All 50 frozen fixtures complete, their appearance data available, with >=300 scored appearances and >=20 positives. Otherwise final result is insufficient evidence.",
        "better": "LL delta <= -0.001, upper 95% LL bounds <0 in BOTH cluster analyses, and Brier delta <= +0.002.",
        "worse": "LL delta >= +0.002 with lower 95% LL bounds >0 in BOTH analyses, OR Brier delta >+0.002 with lower 95% Brier bounds >0 in BOTH analyses.",
        "otherwise": "Inconclusive; retain the update on structural grounds. No automatic rollback or retuning, even after a worse result; flag it for review.",
        "interim": "Descriptive only; no early stopping, extending the window, or changing the rule based on interim scores."
    },
    "integrity": "SHA256-verified immutable snapshot; refuse to create after any study fixture appears in results OR player stats. This verifies local data cutoff, not independent kickoff timestamps.",
    "limitations": ["Five rounds can be underpowered", "The models share historical calibration assumptions", "Appearance selection and per90 target are inherited from the model; this does not validate the minutes model or realized FPL points", "Players outside the frozen cohort are not evaluated", "A fixed-fit five-round test does not measure weekly-refit behaviour"]
}


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_teams(frame):
    frame = frame.copy()
    for col in ("Team", "Opponent", "Home", "Away"):
        if col in frame:
            frame[col] = frame[col].str.strip()
    return frame


def fixture_legs(fixtures):
    home = fixtures.rename(columns={"Home": "Team", "Away": "Opponent"}).assign(Venue="Home")
    away = fixtures.rename(columns={"Away": "Team", "Home": "Opponent"}).assign(Venue="Away")
    return pd.concat([home, away], ignore_index=True)


def verify_snapshot(study):
    manifest = json.loads((study / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["frozen_sha256"].items():
        if sha256(study / name) != expected:
            raise ValueError(f"Frozen study file changed: {name}")
    return manifest


def frozen_model_predict(path, features):
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    x = payload["scaler"].transform(features[FEATURES].to_numpy())
    raw = payload["base_model"].predict_proba(x)[:, 1]
    return payload["calibrator"].predict(raw)


def validate_unplayed(fixtures, played, stats):
    if len(fixtures.merge(played[MATCH_KEY], on=MATCH_KEY)):
        raise ValueError("Study fixture already in results; retrospective freeze refused")
    if len(fixture_legs(fixtures).merge(stats, on=["Season", "Team", "Opponent", "Venue"])):
        raise ValueError("Study appearance already in stats; retrospective freeze refused")


def freeze(data=DATA, study=STUDY, experiment=EXPERIMENT):
    if (study / "manifest.json").exists():
        manifest = verify_snapshot(study)
        print(f"Existing immutable snapshot verified: {manifest['forecast_rows']} forecasts; no changes.")
        return manifest
    if study.exists():
        raise FileExistsError(f"Incomplete study directory exists; refusing to overwrite: {study}")
    # Prove both source forecasts and models came from the same GW7 inputs.
    baseline = experiment / "baseline"
    original_manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in original_manifest["inputs"].items():
        if sha256(data / name) != expected:
            raise ValueError(f"Source inputs changed since matched-model experiment: {name}")
    results = json.loads((experiment / "results.json").read_text(encoding="utf-8"))
    if not results["ship"]:
        raise ValueError("Replacement did not pass its registered historical comparison")
    legacy = clean_teams(read_csv(baseline / "predictions_defensive.csv"))
    updated = clean_teams(read_csv(data / "predictions_defensive.csv"))
    legacy = legacy[legacy.Pos.eq("D")].reset_index(drop=True)
    updated = updated[updated.Pos.eq("D")].reset_index(drop=True)
    pd.testing.assert_frame_equal(legacy.drop(columns=["Probability", "Probability_at_90min"]),
                                  updated.drop(columns=["Probability", "Probability_at_90min"]))
    fixtures = clean_teams(read_csv(data / "fixtures.csv"))
    fixtures = fixtures[fixtures.Season.eq(PROTOCOL["season"]) & fixtures.Round.isin(PROTOCOL["rounds"])].copy()
    if len(fixtures) != 50 or fixtures.duplicated(MATCH_KEY).any() or fixtures.MatchID.duplicated().any():
        raise ValueError("Study requires 50 unambiguous fixture identities")
    played = clean_teams(read_csv(data / "match_results.csv"))
    stats = clean_teams(read_csv(data / "player_match_stats.csv"))
    validate_unplayed(fixtures, played, stats)
    identity = updated[["MatchID", "Player", "Team", "Opponent", "Venue"]].merge(
        fixtures, on="MatchID", how="left", validate="many_to_one")
    if identity.Round.isna().any() or identity.duplicated(PLAYER_KEY).any():
        raise ValueError("Forecast identity mismatch")
    if not ((identity.Venue.eq("Home") & identity.Team.eq(identity.Home) & identity.Opponent.eq(identity.Away)) |
            (identity.Venue.eq("Away") & identity.Team.eq(identity.Away) & identity.Opponent.eq(identity.Home))).all():
        raise ValueError("Forecast team/venue does not match the fixture")
    legacy_paths = list((baseline / "models").glob("Defender__*.pkl.gz"))
    if len(legacy_paths) != 1:
        raise ValueError("Expected one preserved legacy production model")
    model_paths = {"legacy_model.pkl.gz": legacy_paths[0],
                   "updated_model.pkl.gz": experiment / "candidate_production.pkl.gz"}
    for name, path in model_paths.items():
        p = frozen_model_predict(path, updated)
        source = legacy if name.startswith("legacy") else updated
        if not np.allclose(p.round(4), source.Probability_at_90min.values, rtol=0, atol=1e-12):
            raise ValueError(f"Preserved model does not reproduce source forecasts: {name}")
        identity["P_legacy" if name.startswith("legacy") else "P_updated"] = p
    # All validation precedes creation; exclusive writes prevent silent replacement.
    study.mkdir(parents=True)
    for name, path in model_paths.items():
        shutil.copy2(path, study / name)
    identity.to_csv(study / "forecasts.csv", index=False, encoding="utf-8-sig")
    fixtures.to_csv(study / "fixtures.csv", index=False, encoding="utf-8-sig")
    updated[["MatchID", "Player", *FEATURES]].to_csv(study / "features.csv", index=False, encoding="utf-8-sig")
    with (study / "protocol.json").open("x", encoding="utf-8") as stream:
        json.dump({**PROTOCOL, "registered_utc": utc_now()}, stream, indent=2)
    frozen = ["forecasts.csv", "fixtures.csv", "features.csv", "protocol.json", *model_paths]
    manifest = {"created_utc": utc_now(), "forecast_rows": len(identity),
                "defenders": identity.Player.nunique(), "fixture_count": len(fixtures),
                "frozen_sha256": {name: sha256(study / name) for name in frozen},
                "source_inputs_sha256": original_manifest["inputs"],
                "source_old_forecasts_sha256": sha256(baseline / "predictions_defensive.csv"),
                "source_new_forecasts_sha256": sha256(data / "predictions_defensive.csv"),
                "scorer_source_sha256_at_freeze": sha256(Path(__file__))}
    with (study / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Frozen {len(identity)} paired forecasts, {manifest['defenders']} defenders, GW8–12 ({len(fixtures)} fixtures).")
    return manifest


def attach_outcomes(forecasts, fixtures, results, stats):
    """Exact natural-key joins; no FPL MatchID or current-squad name assumptions."""
    forecasts, fixtures, results, stats = map(clean_teams, (forecasts, fixtures, results, stats))
    if forecasts.duplicated(PLAYER_KEY).any() or fixtures.duplicated(MATCH_KEY).any():
        raise ValueError("Duplicate frozen identities")
    results = results[results.Season.isin(fixtures.Season.unique())].copy()
    results = results[results.Home_Goals.notna() & results.Away_Goals.notna()]
    completed = fixtures.merge(results[MATCH_KEY].drop_duplicates(), on=MATCH_KEY, validate="one_to_one")
    legs = fixture_legs(fixtures)[["Season", "Team", "Opponent", "Venue"]]
    relevant = stats.merge(legs, on=["Season", "Team", "Opponent", "Venue"], validate="many_to_one")
    relevant = relevant[relevant.Minutes.gt(0)].copy()
    if relevant.duplicated(PLAYER_KEY).any():
        raise ValueError("Duplicate player outcomes; refusing merge fan-out")
    # Results can arrive before a complete player-stat pull. Do not label every
    # missing row as a non-appearance while a side has fewer than 11 rows.
    counts = relevant.groupby(["Season", "Team", "Opponent", "Venue"]).size().rename("side_rows").reset_index()
    ready = fixture_legs(completed).merge(counts, on=["Season", "Team", "Opponent", "Venue"], how="left")
    ready["side_ready"] = ready.side_rows.fillna(0).ge(11)
    ready_keys = ready.groupby("MatchID").side_ready.all()
    completed["stats_ready"] = completed.MatchID.map(ready_keys).fillna(False).astype(bool)
    available = completed[completed.stats_ready].copy()
    eligible = forecasts.merge(available[MATCH_KEY], on=MATCH_KEY, how="inner", validate="many_to_one")
    eligible = eligible.merge(relevant[PLAYER_KEY + ["Minutes", "CBI", "Tkl"]], on=PLAYER_KEY,
                              how="left", validate="one_to_one", indicator=True)
    eligible["OutcomeStatus"] = np.where(eligible._merge.eq("both"), "appearance", "no_appearance_row")
    invalid = eligible._merge.eq("both") & eligible[["Minutes", "CBI", "Tkl"]].isna().any(axis=1)
    invalid |= eligible._merge.eq("both") & ((eligible.CBI < 0) | (eligible.Tkl < 0))
    eligible.loc[invalid, "OutcomeStatus"] = "invalid_stats"
    scored = eligible[eligible.OutcomeStatus.eq("appearance")].copy()
    scored["Per90"] = (scored.CBI + scored.Tkl) * 90 / scored.Minutes
    scored["Target"] = (scored.Minutes.ge(60) & scored.Per90.ge(10)).astype(int)
    # No unseen defender is silently added to the frozen cohort.
    return scored.drop(columns="_merge"), {
        "completed_fixtures": len(completed), "stats_ready_fixtures": len(available),
        "pending_fixtures": len(fixtures)-len(completed),
        "stats_pending_fixtures": int((~completed.stats_ready).sum()),
        "no_appearance_rows": int(eligible.OutcomeStatus.eq("no_appearance_row").sum()),
        "invalid_stat_rows": int(invalid.sum()), "scored_appearances": len(scored),
        "completed_rounds": [int(r) for r, g in fixtures.groupby("Round")
                             if set(g.MatchID).issubset(set(available.MatchID))]
    }


def losses(y, p):
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(p, dtype=float), eps, 1-eps)
    return -(y*np.log(p) + (1-y)*np.log1p(-p))


def metrics(frame, probability):
    if frame.empty:
        return None
    from sklearn.metrics import roc_auc_score
    y, p = frame.Target.to_numpy(), frame[probability].to_numpy()
    if not (np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()):
        raise ValueError("Invalid frozen probabilities")
    bins = np.minimum((p*10).astype(int), 9)
    calibration = []
    for b in range(10):
        mask = bins == b
        if mask.any():
            calibration.append({"bin": b, "n": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return {"n": len(frame), "hits": int(y.sum()), "log_loss": float(losses(y, p).mean()),
            "brier": float(((y-p)**2).mean()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "mean_prediction": float(p.mean()), "observed": float(y.mean()),
            "ece": sum(b["n"]*abs(b["predicted"]-b["observed"]) for b in calibration)/len(frame),
            "calibration": calibration}


def paired_bootstrap(frame, group):
    if frame.empty:
        return None
    y = frame.Target.to_numpy()
    diff = pd.DataFrame({"group": frame[group].to_numpy(),
        "ll": losses(y, frame.P_updated.to_numpy())-losses(y, frame.P_legacy.to_numpy()),
        "brier": (y-frame.P_updated.to_numpy())**2-(y-frame.P_legacy.to_numpy())**2,
        "n": 1}).groupby("group").sum().to_numpy()
    if len(diff) < 2:
        return {"clusters": len(diff), "log_loss_delta_95ci": None, "brier_delta_95ci": None}
    rng = np.random.default_rng(42)
    total = diff[rng.integers(0, len(diff), size=(2000, len(diff)))].sum(axis=1)
    return {"clusters": len(diff),
            "log_loss_delta_95ci": np.quantile(total[:, 0]/total[:, 2], [.025, .975]).tolist(),
            "brier_delta_95ci": np.quantile(total[:, 1]/total[:, 2], [.025, .975]).tolist()}


def compare(frame):
    if frame.empty:
        return {"legacy": None, "updated": None, "delta": None, "bootstrap": None}
    old, new = metrics(frame, "P_legacy"), metrics(frame, "P_updated")
    return {"legacy": old, "updated": new,
            "delta": {c: new[c]-old[c] for c in ("log_loss", "brier", "ece")},
            "bootstrap": {c: paired_bootstrap(frame, c) for c in ("Player", "MatchID")}}


def assessment(comparison, coverage, fixture_count):
    if coverage["completed_fixtures"] < fixture_count or coverage["stats_ready_fixtures"] < fixture_count:
        return "waiting_for_results" if coverage["scored_appearances"] == 0 else "interim_descriptive_only"
    if coverage["invalid_stat_rows"]:
        return "data_quality_review_required"
    old = comparison["legacy"]
    if old is None or old["n"] < 300 or old["hits"] < 20:
        return "final_insufficient_evidence"
    d, bs = comparison["delta"], comparison["bootstrap"]
    def both_bound(metric, bound, positive):
        intervals = [x[f"{metric}_delta_95ci"] for x in bs.values()]
        return all(x is not None and (x[bound] > 0 if positive else x[bound] < 0) for x in intervals)
    if ((d["log_loss"] >= .002 and both_bound("log_loss", 0, True)) or
            (d["brier"] > .002 and both_bound("brier", 0, True))):
        return "final_worse_flag_for_review_keep_production_pending_review"
    if d["log_loss"] <= -.001 and both_bound("log_loss", 1, False) and d["brier"] <= .002:
        return "final_supports_update"
    return "final_inconclusive_retain_update"


def score(data=DATA, study=STUDY):
    if not (study / "manifest.json").exists():
        print("No registered DEFCON forward check; nothing to score.")
        return None
    manifest = verify_snapshot(study)
    forecasts, fixtures = read_csv(study / "forecasts.csv"), read_csv(study / "fixtures.csv")
    rows, coverage = attach_outcomes(forecasts, fixtures, read_csv(data / "match_results.csv"), read_csv(data / "player_match_stats.csv"))
    primary = compare(rows)
    result = {"as_of_utc": utc_now(), "study_id": PROTOCOL["study_id"],
              "manifest_sha256": sha256(study / "manifest.json"), "coverage": coverage,
              "scorer_source_sha256": sha256(Path(__file__)),
              "assessment": assessment(primary, coverage, manifest["fixture_count"]),
              "primary": primary, "secondary_60plus": compare(rows[rows.Minutes.ge(60)]),
              "by_round": {str(int(r)): compare(g) for r, g in rows.groupby("Round")},
              "production_changed": False}
    report = study / "reports"
    report.mkdir(exist_ok=True)
    rows.to_csv(report / "scored_appearances.csv", index=False, encoding="utf-8-sig")
    (report / "latest.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    # Retain each new outcome state, so a later correction is visible rather than
    # silently replacing the only record. The immutable forecast is always reused.
    state = hashlib.sha256(rows.to_csv(index=False).encode()+json.dumps(coverage, sort_keys=True).encode()
                           +result["scorer_source_sha256"].encode()).hexdigest()[:16]
    snapshot = report / f"state_{state}.json"
    if not snapshot.exists():
        with snapshot.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    summary = {"assessment": result["assessment"], **coverage,
               "primary_delta": primary["delta"], "report": str(report / "latest.json")}
    print(json.dumps(summary, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "score"], nargs="?", default="score")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        score()


if __name__ == "__main__":
    main()
