"""
Conditional hindcast of the §64 BPS accounting fix (pre-registered in CHANGELOG §64).

Historical pre-match inputs are not retained, so each test match's explicit BPS
terms are computed from realized minutes and events using each arm's simulator
definitions; only the residual is random. This tests component accounting, not
the complete pre-match bonus forecast.

Arms: incumbent (legacy accounting, save scale 0.35), candidate (consistent
accounting, scale 1.0) and a sensitivity arm (consistent accounting, scale 0.35).
Window A fits 2025/26 on its first 266 fixtures and tests the last 114; window B
fits 2026/27 GW1-4 and tests GW5-7. GW8 (fit GW1-7) is descriptive only.

Run locally:
    python validate_bonus_accounting.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.bonus import build_residual_bps_baseline as residual
from fpl_model.bonus import simulate_bonus
from fpl_model.bonus import validate_bonus_calibration as calibration

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "validation" / "bonus_accounting_20261028"

N_DRAWS = 2000
SEED = 42
BOOTSTRAP_REPLICATES = 2000
POSITIONS = ["G", "D", "M", "F"]
ARMS = {
    "incumbent": {"legacy": True, "save_scale": 0.35},
    "candidate": {"legacy": False, "save_scale": 1.0},
    "candidate_scale035": {"legacy": False, "save_scale": 0.35},
}


def load_rows():
    rows = calibration.build_match_rows()
    players_current = pd.read_csv(calibration.DATA / "players_current.csv", encoding="utf-8-sig")
    rows["Pos"] = rows["Player"].map(dict(zip(players_current["Player"], players_current["Pos"])))
    rows["Pos"] = rows["Pos"].fillna(rows["Position_Mapped"].map(calibration.SLOT_POSITION))
    rows = rows[rows["Minutes"] > 0].copy()
    excluded = int(rows["Pos"].isna().sum())
    rows = rows[rows["Pos"].isin(POSITIONS)].copy()
    rows["Match_Key"] = rows["Season"].astype(int).astype(str) + "_" + rows["MatchID"].astype(str)
    return rows.reset_index(drop=True), excluded


def simulated_explicit_bps(frame, legacy, save_scale):
    """The simulator's explicit terms evaluated at realized minutes and events."""
    known = residual.add_known_bps(frame, legacy=False, save_scale=save_scale)
    if not legacy:
        return known["known_bps"].to_numpy(float)
    # Incumbent simulator: defcon count // 3, D-only goals conceded on 60+ draws,
    # saves on 60+ keeper draws at the given scale; no keeper goals-conceded term.
    full = frame["Minutes"].to_numpy() >= 60
    pos = frame["Pos"].to_numpy()
    gc = np.where((pos == "D") & full, frame["Goals_Conceded"].to_numpy() * residual.GC_BPS, 0)
    return (known["known_bps"] - known["gc_bps"]).to_numpy(float) + gc


def windows(rows):
    s25 = rows[rows["Season"] == 2025]
    ids25 = np.sort(s25["MatchID"].unique())
    cut = ids25[265]
    s26 = rows[rows["Season"] == 2026]
    return {
        "A_2025": (s25[s25["MatchID"] <= cut], s25[s25["MatchID"] > cut], True),
        "B_2026": (s26[s26["Round"] <= 4], s26[s26["Round"].between(5, 7)], True),
        "GW8_descriptive": (s26[s26["Round"] <= 7], s26[s26["Round"] == 8], False),
    }


def predict_window(fit, test, arm):
    fit = residual.add_known_bps(fit, legacy=arm["legacy"], save_scale=arm["save_scale"])
    fit["residual"] = (fit["BPS"] - fit["known_bps"]).astype(float)
    player_res, pos_defaults = residual.aggregate_residuals(fit)
    defaults = pos_defaults.set_index("Pos_")
    test = test.merge(player_res[["Player", "Pos", "ResidualBPS_Mean", "ResidualBPS_Std"]],
                      on=["Player", "Pos"], how="left")
    test["ResidualBPS_Mean"] = test["ResidualBPS_Mean"].fillna(test["Pos"].map(defaults["pos_mean"]))
    test["ResidualBPS_Std"] = test["ResidualBPS_Std"].fillna(test["Pos"].map(defaults["pos_std"]))
    test["Explicit_BPS"] = simulated_explicit_bps(test, arm["legacy"], arm["save_scale"])
    return test


def simulate(test, normals):
    """Expected bonus and any-bonus probability per row; `normals` keyed by match."""
    exp_bonus = np.zeros(len(test))
    p_any = np.zeros(len(test))
    for match_key, idx in test.groupby("Match_Key").indices.items():
        sub = test.iloc[idx]
        total = (sub["Explicit_BPS"].to_numpy()[:, None] + sub["ResidualBPS_Mean"].to_numpy()[:, None]
                 + sub["ResidualBPS_Std"].to_numpy()[:, None] * normals[match_key])
        bonus = simulate_bonus.assign_bonus(total)
        exp_bonus[idx] = bonus.mean(axis=1)
        p_any[idx] = (bonus > 0).mean(axis=1)
    return exp_bonus, p_any


def position_sums(frame, pred_col):
    """Per match x position sums of (pred - actual) and counts, for bootstrapping."""
    err = frame[pred_col] - frame["Bonus"]
    g = frame.assign(err=err).groupby(["Match_Key", "Pos"]).agg(s=("err", "sum"), n=("err", "size"))
    s = g["s"].unstack(fill_value=0).reindex(columns=POSITIONS, fill_value=0)
    n = g["n"].unstack(fill_value=0).reindex(columns=POSITIONS, fill_value=0)
    return s, n


def summarize(frame, arms):
    out = {"n_player_matches": int(len(frame)), "n_matches": int(frame["Match_Key"].nunique()),
           "support": frame["Pos"].value_counts().reindex(POSITIONS).fillna(0).astype(int).to_dict()}
    for name in arms:
        pred, pany, bps = frame[f"pred_{name}"], frame[f"pany_{name}"], frame[f"bps_{name}"]
        bias = (pred - frame["Bonus"]).groupby(frame["Pos"]).mean().reindex(POSITIONS)
        bps_bias = (bps - frame["BPS"]).groupby(frame["Pos"]).mean().reindex(POSITIONS)
        out[name] = {
            "primary_mean_abs_position_bias": float(bias.abs().mean()),
            "position_bias": bias.round(4).to_dict(),
            "position_bps_bias": bps_bias.round(3).to_dict(),
            "mae": float((pred - frame["Bonus"]).abs().mean()),
            "brier_any_bonus": float(((pany - (frame["Bonus"] > 0)) ** 2).mean()),
            "predicted_bonus_total": float(pred.sum()),
            "gk_bonus_share_predicted": float(pred[frame["Pos"] == "G"].sum() / pred.sum()),
        }
    out["actual_bonus_total"] = float(frame["Bonus"].sum())
    out["gk_bonus_share_actual"] = float(frame.loc[frame["Pos"] == "G", "Bonus"].sum() / frame["Bonus"].sum())
    return out


def bootstrap(frame, arms):
    keys = np.sort(frame["Match_Key"].unique())
    sums = {name: [m.reindex(keys).to_numpy() for m in position_sums(frame, f"pred_{name}")]
            for name in arms}
    rng = np.random.default_rng(SEED)
    picks = rng.integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))
    weights = np.stack([np.bincount(p, minlength=len(keys)) for p in picks])
    res = {}
    stats = {}
    for name, (s, n) in sums.items():
        bs, bn = weights @ s, weights @ n
        score = np.abs(bs / bn).mean(axis=1)
        stats[name] = score
        res[name] = {"primary_ci95": np.percentile(score, [2.5, 97.5]).round(4).tolist(),
                     "position_bias_ci95": {p: np.percentile(bs[:, i] / bn[:, i], [2.5, 97.5]).round(4).tolist()
                                            for i, p in enumerate(POSITIONS)}}
    for name in arms:
        if name != "incumbent":
            diff = stats[name] - stats["incumbent"]
            res[f"{name}_minus_incumbent_primary_ci95"] = np.percentile(diff, [2.5, 97.5]).round(4).tolist()
    return res


def main():
    frozen = calibration.FROZEN_FORECAST
    before = hashlib.sha256(frozen.read_bytes()).hexdigest()
    rows, excluded = load_rows()
    results = {"excluded_played_rows_without_position": excluded, "windows": {}}
    tested = []
    for wname, (fit, test, decisive) in windows(rows).items():
        rng = np.random.default_rng(SEED)
        normals = {k: rng.standard_normal((len(idx), N_DRAWS))
                   for k, idx in sorted(test.groupby("Match_Key").indices.items())}
        test = test.copy()
        for name, arm in ARMS.items():
            pred = predict_window(fit, test, arm)
            exp_bonus, p_any = simulate(pred, normals)
            test[f"pred_{name}"] = exp_bonus
            test[f"pany_{name}"] = p_any
            test[f"bps_{name}"] = (pred["Explicit_BPS"] + pred["ResidualBPS_Mean"]).to_numpy()
            print(f"{wname} {name}: done")
        test["Window"] = wname
        test["Decisive"] = decisive
        results["windows"][wname] = summarize(test, ARMS)
        tested.append(test)

    allrows = pd.concat(tested, ignore_index=True)
    pooled = allrows[allrows["Decisive"]]
    results["pooled_decisive"] = summarize(pooled, ARMS)
    results["pooled_decisive"]["bootstrap"] = bootstrap(pooled, ARMS)

    inc, cand = results["pooled_decisive"]["incumbent"], results["pooled_decisive"]["candidate"]
    diff_ci = results["pooled_decisive"]["bootstrap"]["candidate_minus_incumbent_primary_ci95"]
    rule = {
        "i_primary_falls_25pct": cand["primary_mean_abs_position_bias"]
        <= 0.75 * inc["primary_mean_abs_position_bias"],
        "i_difference_interval_below_zero": diff_ci[1] < 0,
        "ii_mae_not_worse_2pct": cand["mae"] <= 1.02 * inc["mae"],
        "ii_brier_not_worse_2pct": cand["brier_any_bonus"] <= 1.02 * inc["brier_any_bonus"],
        "iii_no_position_abs_bias_up_003": all(
            abs(cand["position_bias"][p]) - abs(inc["position_bias"][p]) <= 0.03 for p in POSITIONS),
    }
    rule["adopt"] = all(rule.values())
    ci = results["pooled_decisive"]["bootstrap"]
    g10, g035 = ci["candidate"]["position_bias_ci95"]["G"], ci["candidate_scale035"]["position_bias_ci95"]["G"]
    rule["scale_switch_to_035"] = bool(g10[0] > 0 and g035[0] <= 0 <= g035[1])
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        before == hashlib.sha256(frozen.read_bytes()).hexdigest() == calibration.FROZEN_FORECAST_SHA256)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["Window", "Decisive", "Season", "Round", "MatchID", "Player", "Pos", "Team", "Minutes",
            "BPS", "Bonus"] + [c for c in allrows.columns if c.startswith(("pred_", "pany_", "bps_"))]
    allrows[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
