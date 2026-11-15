"""
Experiment C (pre-registered in CHANGELOG §70): round simulated total BPS to an
integer so ties occur, against the §68 production incumbent.

Same conditional hindcast, windows and random stream as validate_pass_bands.py;
the incumbent reproduces §68 arm A exactly and the candidate rounds each draw's
total BPS before the official tie rules. First checks that assign_bonus applied
to actual BPS reproduces actual FPL bonus.

Run locally:
    python validate_integer_bps.py
"""
import hashlib
import json

import numpy as np
import pandas as pd

import build_residual_bps_baseline as residual
import simulate_bonus
import validate_bonus_accounting as acc
import validate_bonus_calibration as calibration
import validate_pass_bands as vpb

OUT_DIR = calibration.ROOT / "data" / "validation" / "integer_bps_20261028"
PASS_BANDS_RESULTS = calibration.ROOT / "data" / "validation" / "pass_bands_20261028" / "hindcast_predictions.csv"
POSITIONS = acc.POSITIONS
N_DRAWS = acc.N_DRAWS
SEED = acc.SEED
ARMS = ["incumbent", "integer"]


def simulate(test, normals, uniforms):
    """Expected bonus, any-bonus probability and share of draws paying > 6, per arm."""
    out = {arm: (np.zeros(len(test)), np.zeros(len(test))) for arm in ARMS}
    extra = {arm: [] for arm in ARMS}
    for key, idx in test.groupby("Match_Key").indices.items():
        sub = test.iloc[idx]
        cum = np.cumsum(np.stack(sub["p_tilt"].to_numpy()), axis=1)
        pb = vpb.BANDS[(uniforms[key][:, :, None] > cum[:, None, :]).sum(axis=2).clip(max=3)]
        pb = np.where(sub["Minutes"].to_numpy()[:, None] >= 60, pb, 0)
        total = (sub["Explicit_BPS"].to_numpy()[:, None] + pb + sub["rem_m"].to_numpy()[:, None]
                 + sub["rem_s"].to_numpy()[:, None] * normals[key])
        for arm, t in (("incumbent", total), ("integer", np.rint(total))):
            bonus = simulate_bonus.assign_bonus(t)
            out[arm][0][idx], out[arm][1][idx] = bonus.mean(axis=1), (bonus > 0).mean(axis=1)
            extra[arm].append(float((bonus.sum(axis=0) > 6).mean()))
    return out, {arm: float(np.mean(v)) for arm, v in extra.items()}


def summarize(frame):
    per_match = frame.groupby("Match_Key")
    out = {"n_player_matches": int(len(frame)), "n_matches": int(per_match.ngroups),
           "actual_bonus_per_match": float(per_match["Bonus"].sum().mean()),
           "actual_share_matches_over_6": float((per_match["Bonus"].sum() > 6).mean())}
    for arm in ARMS:
        err = frame[f"pred_{arm}"] - frame["Bonus"]
        out[arm] = {"overall_bias": float(err.mean()), "mse": float((err ** 2).mean()),
                    "mae": float(err.abs().mean()),
                    "brier_any_bonus": float(((frame[f"pany_{arm}"] - (frame["Bonus"] > 0)) ** 2).mean()),
                    "bonus_per_match": float(per_match[f"pred_{arm}"].sum().mean()),
                    "position_bias": err.groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict()}
    return out


def bootstrap(frame):
    keys = np.sort(frame["Match_Key"].unique())
    g = frame.groupby("Match_Key")
    n = g.size().reindex(keys).to_numpy()
    err = {a: (frame[f"pred_{a}"] - frame["Bonus"]).groupby(frame["Match_Key"]).sum().reindex(keys).to_numpy() for a in ARMS}
    sq = {a: ((frame[f"pred_{a}"] - frame["Bonus"]) ** 2).groupby(frame["Match_Key"]).sum().reindex(keys).to_numpy() for a in ARMS}
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(p, minlength=len(keys))
                  for p in rng.integers(0, len(keys), size=(acc.BOOTSTRAP_REPLICATES, len(keys)))])
    bias = {a: (w @ err[a]) / (w @ n) for a in ARMS}
    mse = {a: (w @ sq[a]) / (w @ n) for a in ARMS}
    ci = lambda x: np.percentile(x, [2.5, 97.5]).round(5).tolist()
    return {"incumbent_bias_ci95": ci(bias["incumbent"]), "integer_bias_ci95": ci(bias["integer"]),
            "integer_minus_incumbent_mse_ci95": ci(mse["integer"] - mse["incumbent"])}


def tie_rule_check(test):
    """assign_bonus on actual BPS vs actual bonus, over every played row of each test match."""
    agree = []
    for _, idx in test.groupby("Match_Key").indices.items():
        sub = test.iloc[idx]
        got = simulate_bonus.assign_bonus(sub["BPS"].to_numpy(float)[:, None])[:, 0]
        agree.append(got == sub["Bonus"].to_numpy())
    return float(np.concatenate(agree).mean())


def main():
    frozen_before = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    rows, excluded = acc.load_rows()
    so = vpb.load_sofascore()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    join = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    rows = rows.merge(join[key + ["Canonical_Player_ID", "Kickoff", "band"]], on=key, how="left")
    rows["PB_actual"] = rows["band"].fillna(0)
    rows["Canonical_Player_ID"] = rows["Canonical_Player_ID"].fillna("unjoined:" + rows["Player"])
    known = residual.add_known_bps(rows, legacy=False, save_scale=1.0)
    rows["r_rem"] = (rows["BPS"] - known["known_bps"]).astype(float) - rows["PB_actual"]
    rows["Explicit_BPS"] = known["known_bps"].to_numpy(float)
    const = vpb.constants(so)

    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    k_rem = vpb.split_half_k(wins["A_2025"][0], "r_rem")
    results = {"excluded_played_rows_without_position": excluded, "k_rem": k_rem, "windows": {}}
    # Played rows without a position are excluded from the hindcast, so the check uses every
    # played row of the match from the raw history.
    allrows = calibration.build_match_rows()
    allrows = allrows[allrows["Minutes"] > 0].assign(
        Match_Key=lambda d: d["Season"].astype(int).astype(str) + "_" + d["MatchID"].astype(str))
    test_keys = set().union(*[set(t["Match_Key"]) for _, t, _ in wins.values()])
    results["tie_rule_agreement"] = tie_rule_check(allrows[allrows["Match_Key"].isin(test_keys)])
    print("Tie-rule agreement on actual BPS:", results["tie_rule_agreement"])
    if results["tie_rule_agreement"] < 0.99:
        results["stopped"] = "tie rule does not reproduce actual bonus"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        return

    tested = []
    for name, (fit, test, _) in wins.items():
        test = test.copy()
        table, pos = vpb.player_residuals(fit, "r_rem", k_rem)
        vpb.attach(test, table, pos, "rem")
        _, tilted, _ = vpb.band_distributions(test, so, const, int(test["Season"].iloc[0]), test["Kickoff"].min())
        test["p_tilt"] = list(tilted)
        rng = np.random.default_rng(SEED)
        normals, uniforms = {}, {}
        for k_, idx in sorted(test.groupby("Match_Key").indices.items()):
            normals[k_] = rng.standard_normal((len(idx), N_DRAWS))
            uniforms[k_] = rng.random((len(idx), N_DRAWS))
        out, over6 = simulate(test, normals, uniforms)
        for arm in ARMS:
            test[f"pred_{arm}"], test[f"pany_{arm}"] = out[arm]
        results["windows"][name] = summarize(test)
        results["windows"][name]["simulated_share_draws_over_6"] = over6
        tested.append(test.assign(Window=name))
        print(f"{name}: done")

    pooled = pd.concat(tested, ignore_index=True)
    ref = pd.read_csv(PASS_BANDS_RESULTS, encoding="utf-8-sig")
    chk = pooled.merge(ref[["Season", "MatchID", "Player", "pred_A"]], on=["Season", "MatchID", "Player"], how="left")
    results["incumbent_reproduces_68_arm_A_max_abs_diff"] = float((chk["pred_incumbent"] - chk["pred_A"]).abs().max())
    results["pooled"] = summarize(pooled)
    results["pooled"]["bootstrap"] = bootstrap(pooled)
    p, b = results["pooled"], results["pooled"]["bootstrap"]
    inc, cand = p["incumbent"], p["integer"]
    ci = b["integer_bias_ci95"]
    rule = {"i_abs_bias_halves": abs(cand["overall_bias"]) <= 0.5 * abs(inc["overall_bias"]),
            "i_bias_interval_contains_zero": ci[0] <= 0 <= ci[1],
            "ii_mse_not_worse_0.5pct": cand["mse"] <= 1.005 * inc["mse"],
            "iii_brier_not_worse_1pct": cand["brier_any_bonus"] <= 1.01 * inc["brier_any_bonus"],
            "iv_no_position_abs_bias_up_003": all(
                abs(cand["position_bias"][q]) - abs(inc["position_bias"][q]) <= 0.03 for q in POSITIONS)}
    rule["adopt"] = all(rule.values())
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["Window", "Season", "Round", "MatchID", "Player", "Pos", "Team", "Minutes", "BPS", "Bonus"] + [
        c for c in pooled if c.startswith(("pred_", "pany_"))]
    pooled[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "windows"}, indent=2, default=float))
    for name, w in results["windows"].items():
        print(name, {a: (round(w[a]["mse"], 5), round(w[a]["overall_bias"], 4), round(w[a]["bonus_per_match"], 3))
                     for a in ARMS}, "actual/match", round(w["actual_bonus_per_match"], 3),
              "over6 sim", w["simulated_share_draws_over_6"], "actual", round(w["actual_share_matches_over_6"], 3))


if __name__ == "__main__":
    main()
