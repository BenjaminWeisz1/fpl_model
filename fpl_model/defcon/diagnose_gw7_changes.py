"""Reconstruct the GW6->GW7 update under the new defender model on both sides.

Read-only with respect to production. Refit the locked new head using only
pre-GW7 training rows; recompute prior estimators/features/context as of GW6.
Report model-refit, personal-feature and context effects in a stated sequence.
This reconstructs a counterfactual old rating, not a saved live GW6 output.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import ast
import hashlib
import json
import gzip
import pickle

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import nbinom

from fpl_model.defcon.defender_model import POSITIVE_EVIDENCE, DEFAULT_FEATURE_MODE
from fpl_model.defcon.recent_defcon import attach_recent_rates, feature_columns
from fpl_model.defcon.evaluate_defender_monotonic import OUT, DATA, freeze_control, load_scope, fit_candidate, predict


def context(scope, aggregates):
    _, _, _, opp, team, venue = aggregates
    opp, team, venue = opp.copy(), team.copy(), venue.copy()
    team_base = np.mean(list(team.values()))
    # Same promoted-team rules and iteration set as production.
    for name in {"Hull City", "Coventry City", "Ipswich Town"}:
        if name not in team:
            team[name] = team_base * 1.067
    for name in {"Hull City", "Coventry City", "Ipswich Town"}:
        if name not in opp:
            opp[name] = np.mean(list(opp.values())) / 1.067
    teams = sorted(scope["players_current"].Team.unique())
    return {"opp": opp, "team": team, "venue": venue,
            "opp_mean": float(np.mean([opp.get(t, np.mean(list(opp.values()))) for t in teams]))}


def design(feat, ctx, team, cols, opponent=None, venue=None):
    rows = []
    for v in ([venue] if venue is not None else ["Home", "Away"]):
        rows.append({**feat, "OpponentAvgStat": ctx["opp_mean"] if opponent is None else ctx["opp"].get(opponent, np.mean(list(ctx["opp"].values()))),
                     "TeamAvgStat": ctx["team"].get(team, np.mean(list(ctx["team"].values()))),
                     "VenueAvg": ctx["venue"][v], "IsHome": int(v == "Home")})
    return pd.DataFrame(rows)[cols].values


def score(model, feat, ctx, team, cols):
    return float(predict(model, design(feat, ctx, team, cols)).mean())


def main():
    freeze_control()
    production_hash = hashlib.sha256((DATA / "predictions_defensive.csv").read_bytes()).hexdigest()
    scope = load_scope()
    # Extract only the already-defined transfer/youth inference helpers.
    source = (OUT / "baseline" / "predict.py").read_text(encoding="utf-8-sig")
    for node in ast.parse(source).body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        names = {t.id for t in node.targets if isinstance(t, ast.Name)} if isinstance(node, ast.Assign) else set()
        if name == "make_transfer_features" or "DEF_DEFAULTS" in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<inference helper>", "exec"), scope)
    mode = os.environ.get("DEFCON_DEFENDER_FEATURES", DEFAULT_FEATURE_MODE) if os.environ.get("DEFCON_MONOTONIC_DEF", "1") == "1" else "baseline"
    base_cols = scope["DEF_FEATURE_COLS"]
    cols = feature_columns(base_cols, mode)
    evidence_cols = [c for c in POSITIVE_EVIDENCE if c in cols]
    personal_cols = [c for c in base_cols if c not in {"OpponentAvgStat", "TeamAvgStat", "VenueAvg", "IsHome"}]
    if "RecentEWPer90" in cols:
        personal_cols.append("RecentEWPer90")
    logs = scope["matchlogs"]
    fixtures = pd.read_csv(DATA / "fixtures.csv", encoding="utf-8-sig")
    gw = fixtures[fixtures.Season.eq(2026) & fixtures.Round.eq(7)].copy()
    for c in ("Home", "Away"):
        gw[c] = gw[c].str.strip()
    legs = pd.concat([gw.rename(columns={"Home": "Team", "Away": "Opponent"}).assign(Venue="Home"),
                      gw.rename(columns={"Away": "Team", "Home": "Opponent"}).assign(Venue="Away")])
    latest = logs.merge(legs[["Season", "Team", "Opponent", "Venue"]], on=["Season", "Team", "Opponent", "Venue"], validate="many_to_one")
    keys = ["Season", "Player", "Team", "Opponent", "Venue", "MatchID"]
    latest_keys = pd.MultiIndex.from_frame(latest[keys])
    before_logs = logs.loc[~pd.MultiIndex.from_frame(logs[keys]).isin(latest_keys)].copy()
    print(f"Removed {len(latest)} GW7 appearances; reconstructing the new model before GW7.", flush=True)
    after_ctx = context(scope, scope["def_agg"])
    current_features = pd.read_csv(DATA / "predictions_defensive.csv", encoding="utf-8-sig")
    current_features = current_features[current_features.Pos.eq("D")].drop_duplicates("Player").set_index("Player")
    model_files = list((DATA.parents[1] / "cache" / "defcon").glob("Defender__*.pkl.gz"))
    assert len(model_files) == 1
    with gzip.open(model_files[0], "rb") as stream:
        after_model = pickle.load(stream)

    scope["matchlogs"] = before_logs
    before_agg = scope["compute_season_aggregates"](before_logs.IsDefender, "DefStats_Def", "Per90_Def")
    before_ctx = context(scope, before_agg)
    pure = scope["precompute_player_features"](before_logs.IsDefender, "DefStats_Def", "Per90_Def", 10, scope["DEFENDER_DETAILED_POS"])
    before_estimator = scope["estimate_std_and_pct"](pure)
    before_features = scope["precompute_player_features"](before_logs.IsDefender, "DefStats_Def", "Per90_Def", 10,
        scope["DEFENDER_DETAILED_POS"], priors=True, prior_season=2026, std_pct_estimator=before_estimator)
    frame = pd.read_pickle(OUT / "defender_frame.pkl")
    before_train = frame.loc[~pd.MultiIndex.from_frame(frame[keys]).isin(latest_keys)].copy()
    # Non-prior history features and prior-season contexts for retained rows are
    # identical after truncation. Rebuild every prior-bearing player's rows:
    # the nearest-neighbour prior estimator itself changes when GW7 is removed.
    prior_players = set(scope["_PRIORS_BY_PLAYER"])
    rebuild = scope["build_dataset"](before_logs[before_logs.Player.isin(prior_players)], ["IsDefender"], "DefStats_Def", "Per90_Def", 10,
        before_agg[0], before_agg[1], before_agg[2], scope["DEFENDER_DETAILED_POS"], std_pct_estimator=before_estimator)
    indexed = before_train.set_index(keys)
    rebuilt = rebuild.set_index(keys)
    assert indexed.index.is_unique and rebuilt.index.is_unique
    expected = before_train[before_train.Player.isin(prior_players)].set_index(keys)
    assert expected.index.sort_values().equals(rebuilt.index.sort_values())
    pd.testing.assert_series_equal(expected.Target.sort_index(), rebuilt.Target.sort_index())
    indexed.loc[rebuilt.index, base_cols] = rebuilt[base_cols]
    before_train = indexed.reset_index()
    if "RecentEWPer90" in cols:
        before_train["RecentEWPer90"], before_ew = attach_recent_rates(before_train, scope["_PRIORS_BY_PLAYER"], 2026)
        for player, feat in before_features.items():
            feat["RecentEWPer90"] = before_ew.get(player, feat["AvgPer90"])
    before_model = fit_candidate(before_train, cols, f"before_gw7_{mode}")

    played = latest[latest.IsDefender & latest.Player.isin(current_features.index)].copy()
    players = scope["players_current"].set_index("Player")
    rows, details = [], {}
    for _, appearance in played.iterrows():
        player, team = appearance.Player, appearance.Team
        if player in before_features:
            old = before_features[player]
        else:
            prior = scope["prior_for_season"](player, 2026)
            old = (scope["make_transfer_features"](prior["per90"], prior["matches"], before_estimator, scope["DEFENDER_DETAILED_POS"])
                   if prior is not None else scope["DEF_DEFAULTS"].copy())
        if "RecentEWPer90" in cols and "RecentEWPer90" not in old:
            old = {**old, "RecentEWPer90": old["AvgPer90"]}
        new = current_features.loc[player, personal_cols].to_dict()
        p0 = score(before_model, old, before_ctx, team, cols)
        p1 = score(after_model, old, before_ctx, team, cols)
        p2 = score(after_model, new, before_ctx, team, cols)
        p3 = score(after_model, new, after_ctx, team, cols)
        only_evidence = {**old, **{c: new[c] for c in evidence_cols}}
        pe = score(after_model, only_evidence, before_ctx, team, cols)
        pre_fixture = float(predict(before_model, design(old, before_ctx, team, cols, appearance.Opponent, appearance.Venue))[0])
        implied_mean = brentq(lambda mu: nbinom.sf(9, 3*mu, .75)-pre_fixture, 1e-8, 60) if 0 < pre_fixture < 1 else np.nan
        record = {"Player": player, "Web_Name": players.loc[player, "Web_Name"], "Team": team,
            "Opponent": appearance.Opponent, "Venue": appearance.Venue,
            "Minutes": float(appearance.Minutes), "Actions": float(appearance.DefStats_Def), "Per90": float(appearance.Per90_Def),
            "PreCareerPer90": old["AvgPer90"], "ExpectedActionsCareer": old["AvgPer90"]*appearance.Minutes/90,
            "PreFixtureProbability": pre_fixture, "PreFixtureImpliedNBMean90": implied_mean,
            "PreRecent5": old["Recent5AvgPer90"], "PostRecent5": new["Recent5AvgPer90"],
            "PreHitRate": old["PctAboveThreshold"], "PostHitRate": new["PctAboveThreshold"],
            "RatingBeforeReconstructed": p0, "RatingAfter": p3, "Delta": p3-p0,
            "RefitEffect": p1-p0, "PersonalEffect": p2-p1, "ContextEffect": p3-p2,
            "EvidenceOnlyEffect": pe-p1, "SpreadExperienceRoleEffectAfterEvidence": p2-pe,
            "UnderCareerReference": appearance.Per90_Def < old["AvgPer90"]-1e-9,
            "OverCareerReference": appearance.Per90_Def > old["AvgPer90"]+1e-9,
            "MissedThreshold": not (appearance.Minutes >= 60 and appearance.Per90_Def >= 10),
            "AllEvidenceFell": all(new[c] <= old[c]+1e-12 for c in evidence_cols),
            "AllEvidenceRose": all(new[c] >= old[c]-1e-12 for c in evidence_cols)}
        assert abs((record["RefitEffect"]+record["PersonalEffect"]+record["ContextEffect"])-record["Delta"]) < 1e-12
        if record["AllEvidenceFell"]:
            assert record["EvidenceOnlyEffect"] <= 1e-12
        if record["AllEvidenceRose"]:
            assert record["EvidenceOnlyEffect"] >= -1e-12
        rows.append(record)
        details[player] = {"before": old, "after": new,
            "one_feature_effects_current_model_old_context": {c: score(after_model, {**old, c: new[c]}, before_ctx, team, cols)-p1
                for c in old if c in new and old[c] != new[c]}}
    table = pd.DataFrame(rows)
    current_avg = pd.read_csv(DATA / "defcon_average_probabilities.csv", encoding="utf-8-sig").set_index("Player")
    for _, row in table.iterrows():
        assert round(row.RatingAfter, 4) == current_avg.loc[row.Player, "Defcon"]
    full = table[table.Minutes.ge(60)].copy()
    rose = full.RatingAfter.round(4) > full.RatingBeforeReconstructed.round(4)
    fell = full.RatingAfter.round(4) < full.RatingBeforeReconstructed.round(4)
    opposite = full[(full.UnderCareerReference & rose) | (full.OverCareerReference & fell)].copy()
    opposite = opposite.assign(AbsDelta=opposite.Delta.abs()).sort_values("AbsDelta", ascending=False)
    prefix = f"gw7_{mode}"
    table.to_csv(OUT / f"{prefix}_rating_changes.csv", index=False, encoding="utf-8-sig")
    opposite.to_csv(OUT / f"{prefix}_opposite_rating_changes.csv", index=False, encoding="utf-8-sig")
    summary = {"scope": "Current FPL defenders who appeared in GW7; headline counts use >=60-minute appearances",
        "comparison": "Counterfactual reconstruction of the new model fitted through GW6 versus actual new model through GW7; not last week's published legacy ratings",
        "expected_definition": "Pre-match prior-blended career actions per90, scaled to observed minutes; classifier probability is not itself an action-count forecast",
        "decomposition": "Order: old fit/features/context -> new fit -> new personal features -> new context. Interactions make attribution order-dependent.",
        "limitations": "Historical position crosswalk and current squad/transfer tables are held as currently stored; no authentic saved GW6 run of the new model exists",
        "appearances": len(table), "full_appearances": len(full), "cameos_excluded_from_headline": int(table.Minutes.lt(60).sum()),
        "under_reference_rating_rose": int((full.UnderCareerReference & rose).sum()),
        "over_reference_rating_fell": int((full.OverCareerReference & fell).sum()),
        "threshold_miss_rating_rose": int((full.MissedThreshold & rose).sum()),
        "threshold_hit_rating_fell": int((~full.MissedThreshold & fell).sum()),
        "fixed_model_personal_under_rose": int((full.UnderCareerReference & (full.PersonalEffect > 5e-5)).sum()),
        "fixed_model_personal_over_fell": int((full.OverCareerReference & (full.PersonalEffect < -5e-5)).sum()),
        "largest_opposites": json.loads(opposite.head(15).to_json(orient="records")),
        "feature_mode": mode, "production_changed": False}
    (OUT / f"{prefix}_rating_change_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (OUT / f"{prefix}_rating_change_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert hashlib.sha256((DATA / "predictions_defensive.csv").read_bytes()).hexdigest() == production_hash
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
