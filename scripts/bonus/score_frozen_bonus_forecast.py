"""
Scores the immutable 2026 GW8 bonus forecast against actual GW8 awards
(prospective monitoring, CHANGELOG §49/§65). Never used to retune anything.

The frozen forecast predates keeper competition (§47), so its inputs are
outfield-only. As a labelled counterfactual, GW8 is also re-simulated from the
same frozen simulation features under the legacy and §64 accounting, with the
residual refitted on 2026 GW1-7 only. The frozen files are hash-checked and
never modified.

Run locally:
    python score_frozen_bonus_forecast.py
"""
import hashlib
import json

import numpy as np
import pandas as pd

import build_residual_bps_baseline as residual
import simulate_bonus
import validate_bonus_calibration as calibration

SNAPSHOT = calibration.FROZEN_FORECAST.parents[1]
OUT_DIR = calibration.ROOT / "data" / "validation" / "bonus_gw8_forecast_20261028"
ROUND = 8
BOOTSTRAP_REPLICATES = 2000
SEED = 42
FROZEN_HASHES = {
    "gameweek/bonus_predictions.csv": calibration.FROZEN_FORECAST_SHA256,
    "gameweek/simulation_features.csv": "923094bcd9a539fe96c3ed48016ba18346e6ff3c2e29f2a324f0b6c5313d1669",
    "gameweek/fixture_goal_predictions.csv": "f72f88b97b3676031a4c83f06447f3c9014bb4e9c83dad2b9aa09534afca8528",
}


def check_hashes():
    return {name: hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == expected
            for name, expected in FROZEN_HASHES.items()}


def refit_residuals(rows, legacy):
    fit = rows[(rows["Season"] == 2026) & (rows["Round"] <= ROUND - 1)]
    fit = residual.add_known_bps(fit, legacy=legacy,
                                 save_scale=0.35 if legacy else 1.0)
    fit["residual"] = (fit["BPS"] - fit["known_bps"]).astype(float)
    return residual.aggregate_residuals(fit)


def resimulate(features, fixture_goals, rows, legacy):
    player_res, pos_defaults = refit_residuals(rows, legacy)
    defaults = pos_defaults.set_index("Pos_")
    feats = features.drop(columns=["ResidualBPS_Mean", "ResidualBPS_Std"]).merge(
        player_res[["Player", "ResidualBPS_Mean", "ResidualBPS_Std"]], on="Player", how="left")
    feats["ResidualBPS_Mean"] = feats["ResidualBPS_Mean"].fillna(feats["Pos"].map(defaults["pos_mean"]))
    feats["ResidualBPS_Std"] = feats["ResidualBPS_Std"].fillna(feats["Pos"].map(defaults["pos_std"]))

    simulate_bonus.COHERENCE = False  # the GW8 counterfactuals predate §145
    simulate_bonus.LEGACY_ACCOUNTING = legacy
    simulate_bonus.GK_SAVE_BPS_SCALE = 0.35 if legacy else 1.0
    simulate_bonus.rng = np.random.default_rng(SEED)
    pen_per_npg = 792 / 7734  # as simulate_bonus.main
    out = []
    for _, fx in fixture_goals.iterrows():
        match = feats[(feats["Round"] == fx["Round"]) & feats["Team"].isin([fx["Home"], fx["Away"]])]
        bonus, _, _ = simulate_bonus.simulate_match(
            match, fx["Away_Pred_NPG"] * (1 + pen_per_npg * 0.79),
            fx["Home_Pred_NPG"] * (1 + pen_per_npg * 0.79), fx["Home"], simulate_bonus.N_DRAWS)
        out.append(pd.DataFrame({"Player": match["Player"].values, "Team": match["Team"].values,
                                 "Expected_Bonus": bonus}))
    return pd.concat(out, ignore_index=True)


def score(frame, col):
    err = frame[col] - frame["Bonus"]
    by_pos = frame.assign(err=err).groupby("Pos")
    return {
        "bias_by_pos": by_pos["err"].mean().round(4).to_dict(),
        "mae": float(err.abs().mean()),
        "predicted_total": float(frame[col].sum()),
    }


def top3_hits(frame, col):
    """Share of actual bonus recipients who were in the forecast's top three for that match."""
    hits, recipients = 0, 0
    for _, match in frame.groupby("MatchID"):
        top = set(match.nlargest(3, col)["Player"])
        actual = set(match.loc[match["Bonus"] > 0, "Player"])
        hits += len(top & actual)
        recipients += len(actual)
    return hits / recipients


def main():
    hashes_before = check_hashes()
    if not all(hashes_before.values()):
        raise ValueError(f"Frozen GW8 snapshot changed: {hashes_before}")

    rows = calibration.build_match_rows()
    players_current = pd.read_csv(calibration.DATA / "players_current.csv", encoding="utf-8-sig")
    rows["Pos"] = rows["Player"].map(dict(zip(players_current["Player"], players_current["Pos"])))
    rows["Pos"] = rows["Pos"].fillna(rows["Position_Mapped"].map(calibration.SLOT_POSITION))
    actual = rows[(rows["Season"] == 2026) & (rows["Round"] == ROUND)]
    fit_rows = rows[rows["Minutes"] > 0]

    frozen = pd.read_csv(calibration.FROZEN_FORECAST, encoding="utf-8-sig")
    features = pd.read_csv(SNAPSHOT / "gameweek" / "simulation_features.csv", encoding="utf-8-sig")
    fixture_goals = pd.read_csv(SNAPSHOT / "gameweek" / "fixture_goal_predictions.csv", encoding="utf-8-sig")

    legacy = resimulate(features, fixture_goals, fit_rows, legacy=True)
    fixed = resimulate(features, fixture_goals, fit_rows, legacy=False)
    preds = frozen[["Player", "Pos", "Team", "Expected_Bonus"]].rename(
        columns={"Expected_Bonus": "frozen"})
    preds = preds.merge(legacy.rename(columns={"Expected_Bonus": "resim_legacy"}), on=["Player", "Team"])
    preds = preds.merge(fixed.rename(columns={"Expected_Bonus": "resim_fixed"}), on=["Player", "Team"])

    # Forecast players who did not appear score zero; appearances outside the forecast are reported.
    scored = preds.merge(actual[["Player", "Team", "MatchID", "Minutes", "Bonus"]],
                         on=["Player", "Team"], how="left")
    team_match = actual.drop_duplicates("Team").set_index("Team")["MatchID"]
    scored["MatchID"] = scored["MatchID"].fillna(scored["Team"].map(team_match))
    scored[["Minutes", "Bonus"]] = scored[["Minutes", "Bonus"]].fillna(0)
    unforecast = actual[~actual.set_index(["Player", "Team"]).index.isin(
        preds.set_index(["Player", "Team"]).index) & (actual["Minutes"] > 0)]

    results = {
        "n_forecast_rows": int(len(scored)),
        "n_matches": int(scored["MatchID"].nunique()),
        "actual_bonus_total": float(actual["Bonus"].sum()),
        "actual_bonus_to_forecast_outfield_rows": float(scored["Bonus"].sum()),
        "actual_bonus_to_goalkeepers": float(actual.loc[actual["Pos"] == "G", "Bonus"].sum()),
        "unforecast_outfield_appearances": int((unforecast["Pos"] != "G").sum()),
        "unforecast_outfield_bonus": float(unforecast.loc[unforecast["Pos"] != "G", "Bonus"].sum()),
        "actual_bonus_by_pos_on_forecast_rows": scored.groupby("Pos")["Bonus"].sum().to_dict(),
    }
    keys = np.sort(scored["MatchID"].unique())
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))
    for col in ["frozen", "resim_legacy", "resim_fixed"]:
        results[col] = score(scored, col)
        results[col]["top3_share_of_actual_recipients"] = top3_hits(scored, col)
        g = scored.assign(err=scored[col] - scored["Bonus"]).groupby(["MatchID", "Pos"])["err"]
        s = g.sum().unstack(fill_value=0).reindex(keys).to_numpy()
        n = g.size().unstack(fill_value=0).reindex(keys).to_numpy()
        pos = g.sum().unstack(fill_value=0).columns
        bs = np.stack([s[p].sum(axis=0) / n[p].sum(axis=0) for p in picks])
        results[col]["bias_ci95_by_pos"] = {
            p: np.percentile(bs[:, i], [2.5, 97.5]).round(4).tolist() for i, p in enumerate(pos)}
    results["frozen_hashes_unchanged"] = check_hashes()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(OUT_DIR / "gw8_scored.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
