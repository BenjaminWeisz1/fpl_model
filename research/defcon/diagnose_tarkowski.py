"""Read-only reproduction of Tarkowski's weekly feature sensitivity."""
import ast
import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

source_path = Path(__file__).with_name("predict.py")
source = source_path.read_text(encoding="utf-8-sig")
scope = {"__file__": str(source_path), "__name__": "defcon_diagnostic"}
# Run ingestion and feature definitions, stopping before dataset construction,
# fitting, cache maintenance, or production writes.
exec(compile(source.split('print("Building defender dataset...")')[0], str(source_path), "exec"), scope)
for node in ast.parse(source).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        {"BASE_FEATURE_COLS", "LEGACY_DEF_FEATURE_COLS", "DEF_FEATURE_COLS"} for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), "exec"), scope)

player = "James Tarkowski"
logs = scope["matchlogs"]
history = logs[logs.Player.eq(player)].sort_values("MatchID")
print(history.tail(8)[["Season", "MatchID", "Opponent", "Minutes", "DefStats_Def", "Per90_Def", "Position_Detailed"]].to_string(index=False))

cache = source_path.parents[2] / "cache" / "defcon"
with gzip.open(next(cache.glob("Defender__*.pkl.gz")), "rb") as f:
    payload = pickle.load(f)

def features(frame):
    scope["matchlogs"] = frame
    feat = scope["precompute_player_features"](
        frame.IsDefender, "DefStats_Def", "Per90_Def", 10,
        scope["DEFENDER_DETAILED_POS"], priors=True,
        prior_season=scope["MAX_SEASON"], std_pct_estimator=scope["def_std_estimator"])[player]
    if "RecentEWPer90" in scope["DEF_FEATURE_COLS"]:
        h=frame[frame.IsDefender & frame.Player.eq(player) & frame.Minutes.ge(1)].sort_values("MatchID").copy()
        h["RawStat"]=h.DefStats_Def
        h["AvgPer90"]=feat["AvgPer90"]
        _,latest=scope["attach_recent_rates"](h,scope["_PRIORS_BY_PLAYER"],scope["MAX_SEASON"])
        feat["RecentEWPer90"]=latest.get(player,feat["AvgPer90"])
    return feat

def context(frame):
    scope["matchlogs"] = frame
    *_, opp, team, venue = scope["compute_season_aggregates"](frame.IsDefender, "DefStats_Def", "Per90_Def")
    promoted = {"Hull City", "Coventry City", "Ipswich Town"}
    for t in promoted:
        if t not in opp:
            opp[t] = np.mean(list(opp.values())) / 1.067
    teams = scope["players_current"].Team.unique()
    return {"OpponentAvgStat": float(np.mean([opp.get(t, np.mean(list(opp.values()))) for t in teams])),
            "TeamAvgStat": team["Everton"], "venue": venue}

def score(feat, ctx):
    rows = [{**feat, **{k: v for k, v in ctx.items() if k != "venue"},
             "VenueAvg": ctx["venue"][v], "IsHome": int(v == "Home")} for v in ("Home", "Away")]
    x = pd.DataFrame(rows)[scope["DEF_FEATURE_COLS"]].to_numpy()
    raw = payload["base_model"].predict_proba(payload["scaler"].transform(x))[:, 1]
    cal = payload["calibrator"].predict(raw)
    return {"raw": raw.tolist(), "calibrated": cal.tolist(), "mean": float(cal.mean())}

before = logs.drop(history.tail(1).index)
old_feat, new_feat = features(before), features(logs)
ctx = context(logs)
print("FEATURES", json.dumps({k: [old_feat[k], new_feat[k]] for k in old_feat}, indent=2))
print("BEFORE_PERSONAL_CURRENT_MODEL", score(old_feat, ctx))
print("CURRENT", score(new_feat, ctx))
for k in new_feat:
    if old_feat[k] != new_feat[k]:
        print("ONLY_CHANGE", k, score({**old_feat, k: new_feat[k]}, ctx))
print("CONTEXT", ctx)
print("CALIBRATOR_HIGH_END", list(zip(payload["calibrator"].X_thresholds_[-20:].tolist(), payload["calibrator"].y_thresholds_[-20:].tolist())))
print("BOTH_RECENT_ONLY", score({**old_feat, "Recent5AvgRaw": new_feat["Recent5AvgRaw"], "Recent5AvgPer90": new_feat["Recent5AvgPer90"]}, ctx))
for rate in [0.610, 0.615, 0.618, 0.61846, 0.619, 0.620, 0.62037, 0.621, 0.625, 0.630]:
    print("HIT_RATE_SWEEP", rate, score({**new_feat, "PctAboveThreshold": rate}, ctx))
model = payload["base_model"]
scaler = payload["scaler"]
cols = scope["DEF_FEATURE_COLS"]
j = cols.index("PctAboveThreshold")
splits = []
for i, estimator in enumerate(getattr(model, "estimators_", np.empty((0, 1), dtype=object))[:, 0]):
    tree = estimator.tree_
    for n in np.flatnonzero(tree.feature == j):
        cut = tree.threshold[n] * scaler.scale_[j] + scaler.mean_[j]
        if new_feat["PctAboveThreshold"] < cut < old_feat["PctAboveThreshold"]:
            splits.append({"tree": i, "threshold": float(cut), "samples": int(tree.n_node_samples[n])})
print("CROSSED_HIT_RATE_SPLITS", json.dumps(splits))
