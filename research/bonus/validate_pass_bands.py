"""
Experiment A (pre-registered in CHANGELOG §68): pass bands as an explicit,
discrete bonus component, against the §64 incumbent.

Conditional hindcast as validate_bonus_accounting.py: realized minutes and the
other explicit events are kept; pass bands and the residual are drawn from
pre-match information only. Arms: incumbent (§64, k=10), B0 (position-specific
residual shrinkage), A_noopp (pass bands, no tilt), A (pass bands tilted by
opponent and venue effects). Common random numbers across arms.

Run locally:
    python validate_pass_bands.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import optimize

from fpl_model.bonus import build_residual_bps_baseline as residual
from fpl_model.bonus import simulate_bonus
from research.bonus import validate_bonus_accounting as acc
from research.bonus import validate_bonus_calibration as calibration

OUT_DIR = calibration.ROOT / "data" / "validation" / "pass_bands_20261028"
BANDS = np.array([0, 2, 4, 6])
HISTORY = 38
CONSTANT_SEASONS = range(2018, 2025)
POSITIONS = acc.POSITIONS
N_DRAWS = acc.N_DRAWS
SEED = acc.SEED
ARMS = ["incumbent", "B0", "A_noopp", "A"]


def pass_band(total, accurate):
    share = accurate / total.where(total > 0)
    return np.where(total >= 30, np.select([share >= .9, share >= .8, share >= .7], [6, 4, 2], 0), 0)


CONTROLS = {  # 2026/27 BPS weights; controls for the amended definition check (§68)
    "keyPass": 1, "bigChanceCreated": 3, "bigChanceMissed": -3, "accurateCross": 1, "wonContest": 1,
    "onTargetScoringAttempt": 2, "shotOffTarget": -1, "wasFouled": 1, "fouls": -1, "totalOffside": -1,
    "errorLeadToAShot": -1, "errorLeadToAGoal": -3, "clearanceOffLine": 9, "penaltyConceded": -3, "wonTackle": 2,
}


def load_sofascore():
    so = pd.read_csv(calibration.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Canonical_Player_ID", "Local_Player", "Team", "Opponent", "Venue", "Kickoff",
                              "Appearance_Status", "Position_Sofascore", "Sofascore_minutesPlayed",
                              "Sofascore_totalPass", "Sofascore_accuratePass"]
                     + ["Sofascore_" + k for k in CONTROLS])
    so = so[so["Appearance_Status"].eq("played")].copy()
    for k, w in CONTROLS.items():
        so["c_" + k] = so["Sofascore_" + k].fillna(0) * w
    so["Kickoff"] = pd.to_datetime(so["Kickoff"], utc=True)
    so["band"] = pass_band(so["Sofascore_totalPass"].fillna(0), so["Sofascore_accuratePass"].fillna(0))
    so["full"] = so["Sofascore_minutesPlayed"] >= 60
    so["match"] = so["Season"].astype(str) + "_" + so["Kickoff"].astype(str) + "_" + so[["Team", "Opponent"]].min(axis=1)
    return so


def constants(so):
    """Band priors, k_pb, venue effects and k_opp from 2018-2024 only."""
    hist = so[so["Season"].isin(CONSTANT_SEASONS) & so["full"]].copy()
    prior = {p: np.array([(g["band"] == b).mean() for b in BANDS]) for p, g in hist.groupby("Position_Sofascore")}
    k_pb = {}
    ps = hist.groupby(["Position_Sofascore", "Canonical_Player_ID", "Season"])["band"]
    stats = pd.DataFrame({"mean": ps.mean(), "var": ps.var(), "n": ps.size()}).reset_index()
    for p, g in stats[stats["n"] >= 10].groupby("Position_Sofascore"):
        nxt = g.assign(Season=g["Season"] - 1)[["Canonical_Player_ID", "Season", "mean"]]
        pair = g.merge(nxt, on=["Canonical_Player_ID", "Season"], suffixes=("", "_n"))
        tau2 = np.cov(pair["mean"], pair["mean_n"])[0, 1]
        k_pb[p] = float(np.clip(g["var"].mean() / tau2, 0.5, 100)) if tau2 > 0 else 100.0
    hist["dev"] = hist["band"] - hist.groupby(["Canonical_Player_ID", "Season"])["band"].transform("mean")
    venue = hist.groupby("Venue")["dev"].mean().to_dict()
    match_dev = hist.groupby(["Opponent", "Season", "match"])["dev"].mean().reset_index()
    match_dev["half"] = match_dev.groupby(["Opponent", "Season"]).cumcount() % 2
    halves = match_dev.groupby(["Opponent", "Season", "half"])["dev"].mean().unstack().dropna()
    tau2_opp = np.cov(halves[0], halves[1])[0, 1]
    within = match_dev.groupby(["Opponent", "Season"])["dev"].var().mean()
    k_opp = float(np.clip(within / tau2_opp, 1, 200)) if tau2_opp > 0 else 200.0
    return {"prior": prior, "k_pb": k_pb, "venue": venue, "k_opp": k_opp}


def split_half_k(fit, col):
    """Position-specific shrinkage (matches) by split-half method of moments."""
    ids = np.sort(fit["MatchID"].unique())
    fit = fit.assign(half=np.where(fit["MatchID"] <= ids[len(ids) // 2], 0, 1))
    out = {}
    for p, g in fit.groupby("Pos"):
        means = g.groupby(["Player", "half"])[col].agg(["mean", "size"]).unstack()
        means = means[(means["size"][0] >= 8) & (means["size"][1] >= 8)]
        tau2 = np.cov(means["mean"][0], means["mean"][1])[0, 1] if len(means) > 2 else 0
        within = g.groupby("Player")[col].var().mean()
        out[p] = float(np.clip(within / tau2, 1, 100)) if tau2 > 0 else 100.0
    return out


def player_residuals(fit, col, k):
    """Shrunken residual mean/std per player, production std rule, k by position."""
    g = fit.groupby(["Player", "Pos"])[col].agg(["mean", "std", "count"]).reset_index()
    pos = fit.groupby("Pos")[col].agg(pos_mean="mean", pos_std="std")
    g = g.merge(pos, left_on="Pos", right_index=True)
    kk = g["Pos"].map(k) if isinstance(k, dict) else k
    g["m"] = (g["count"] * g["mean"] + kk * g["pos_mean"]) / (g["count"] + kk)
    g["s"] = np.where(g["count"] >= residual.MIN_MATCHES_FOR_OWN_STD, g["std"].fillna(g["pos_std"]), g["pos_std"])
    return g.set_index(["Player", "Pos"])[["m", "s"]], pos


def attach(test, table, pos, prefix):
    idx = pd.MultiIndex.from_arrays([test["Player"], test["Pos"]])
    test[prefix + "_m"] = table["m"].reindex(idx).to_numpy()
    test[prefix + "_s"] = table["s"].reindex(idx).to_numpy()
    test[prefix + "_m"] = test[prefix + "_m"].fillna(test["Pos"].map(pos["pos_mean"]))
    test[prefix + "_s"] = test[prefix + "_s"].fillna(test["Pos"].map(pos["pos_std"]))


def tilt(p, shift):
    target = float(np.clip(p @ BANDS + shift, 0.02, 5.98))
    f = lambda lam: (p * np.exp(lam * BANDS)) @ BANDS / (p * np.exp(lam * BANDS)).sum() - target
    lam = optimize.brentq(f, -10, 10)
    q = p * np.exp(lam * BANDS)
    return q / q.sum()


def band_distributions(test, so, const, season, cutoff):
    hist = so[so["full"] & (so["Kickoff"] < cutoff)].sort_values("Kickoff")
    recent = hist.groupby("Canonical_Player_ID").tail(HISTORY)
    counts = recent.groupby("Canonical_Player_ID")["band"].value_counts().unstack(fill_value=0).reindex(columns=BANDS, fill_value=0)
    pmean = recent.groupby("Canonical_Player_ID")["band"].mean()
    cur = hist[hist["Season"] == season].copy()
    cur["dev"] = cur["band"] - cur["Canonical_Player_ID"].map(pmean)
    md = cur.groupby(["Opponent", "match"])["dev"].mean().groupby("Opponent").agg(["sum", "size"])
    opp = (md["sum"] / (md["size"] + const["k_opp"])).to_dict()
    plain, tilted = [], []
    for row in test.itertuples():
        prior = const["prior"][row.Pos]
        c = counts.loc[row.Canonical_Player_ID].to_numpy() if row.Canonical_Player_ID in counts.index else np.zeros(4)
        p = (c + const["k_pb"][row.Pos] * prior) / (c.sum() + const["k_pb"][row.Pos])
        plain.append(p)
        tilted.append(tilt(p, opp.get(row.Opponent, 0.0) + const["venue"].get(row.Venue, 0.0)))
    return np.array(plain), np.array(tilted), opp


def simulate(test, normals, uniforms, arm):
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    for key, idx in test.groupby("Match_Key").indices.items():
        sub = test.iloc[idx]
        if arm in ("incumbent", "B0"):
            prefix, pb = ("inc" if arm == "incumbent" else "b0"), 0
        else:
            prefix = "rem"
            probs = np.stack(sub["p_tilt" if arm == "A" else "p_plain"].to_numpy())
            cum = np.cumsum(probs, axis=1)
            pb = BANDS[(uniforms[key][:, :, None] > cum[:, None, :]).sum(axis=2).clip(max=3)]
            pb = np.where(sub["Minutes"].to_numpy()[:, None] >= 60, pb, 0)
        total = (sub["Explicit_BPS"].to_numpy()[:, None] + pb + sub[prefix + "_m"].to_numpy()[:, None]
                 + sub[prefix + "_s"].to_numpy()[:, None] * normals[key])
        bonus = simulate_bonus.assign_bonus(total)
        exp_bonus[idx], p_any[idx] = bonus.mean(axis=1), (bonus > 0).mean(axis=1)
    return exp_bonus, p_any


def summarize(frame):
    out = {"n_player_matches": int(len(frame)), "n_matches": int(frame["Match_Key"].nunique())}
    for arm in ARMS:
        err = frame[f"pred_{arm}"] - frame["Bonus"]
        out[arm] = {"mse": float((err ** 2).mean()), "mae": float(err.abs().mean()),
                    "brier_any_bonus": float(((frame[f"pany_{arm}"] - (frame["Bonus"] > 0)) ** 2).mean()),
                    "position_bias": err.groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict()}
    return out


def bootstrap(frame):
    keys = np.sort(frame["Match_Key"].unique())
    sq = {arm: ((frame[f"pred_{arm}"] - frame["Bonus"]) ** 2).groupby(frame["Match_Key"]).sum().reindex(keys).to_numpy()
          for arm in ARMS}
    n = frame.groupby("Match_Key").size().reindex(keys).to_numpy()
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(p, minlength=len(keys))
                  for p in rng.integers(0, len(keys), size=(acc.BOOTSTRAP_REPLICATES, len(keys)))])
    mse = {arm: (w @ sq[arm]) / (w @ n) for arm in ARMS}
    pairs = [("A", "incumbent"), ("B0", "incumbent"), ("A_noopp", "B0"), ("A", "A_noopp")]
    return {f"{a}_minus_{b}_mse_ci95": np.percentile(mse[a] - mse[b], [2.5, 97.5]).round(5).tolist() for a, b in pairs}


def main():
    frozen_before = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    rows, excluded = acc.load_rows()
    so = load_sofascore()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    join = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    controls = ["c_" + k for k in CONTROLS]
    rows = rows.merge(join[key + ["Canonical_Player_ID", "Kickoff", "band"] + controls], on=key, how="left")
    joined = float(rows["Canonical_Player_ID"].notna().mean())
    rows["PB_actual"] = rows["band"].fillna(0)
    rows["Canonical_Player_ID"] = rows["Canonical_Player_ID"].fillna("unjoined:" + rows["Player"])
    known = residual.add_known_bps(rows, legacy=False, save_scale=1.0)
    rows["r_full"] = (rows["BPS"] - known["known_bps"]).astype(float)
    rows["r_rem"] = rows["r_full"] - rows["PB_actual"]
    rows["Explicit_BPS"] = known["known_bps"].to_numpy(float)
    const = constants(so)

    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    fitA = wins["A_2025"][0]
    k_full, k_rem = split_half_k(fitA, "r_full"), split_half_k(fitA, "r_rem")
    results = {"join_rate": joined, "excluded_played_rows_without_position": excluded,
               "constants": {"k_pb": const["k_pb"], "venue": const["venue"], "k_opp": const["k_opp"],
                             "prior": {p: v.round(4).tolist() for p, v in const["prior"].items()},
                             "k_full": k_full, "k_rem": k_rem},
               "definition_check": {}, "windows": {}}

    tested = []
    for name, (fit, test, _) in wins.items():
        # Amended check (§68): pass-band coefficient with the other Sofascore components controlled.
        d = fit[fit["Canonical_Player_ID"].str.startswith("unjoined:") == False]
        g = d.groupby("Player")
        y = (d["r_full"] - g["r_full"].transform("mean")).to_numpy(float)
        x = np.column_stack([d[c] - g[c].transform("mean") for c in ["PB_actual"] + controls]).astype(float)
        coef = np.linalg.lstsq(x, y, rcond=None)[0]
        simple = float((x[:, 0] * y).sum() / (x[:, 0] ** 2).sum())
        results["definition_check"][name] = float(coef[0])
        results["definition_check"][name + "_uncontrolled_slope"] = simple
    slopes = [v for k, v in results["definition_check"].items() if not k.endswith("_uncontrolled_slope")]
    if not all(0.7 <= s <= 1.3 for s in slopes):
        results["stopped"] = "definition check failed"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(json.dumps(results, indent=2, default=float))
        return

    for name, (fit, test, _) in wins.items():
        test = test.copy()
        for prefix, col, k in [("inc", "r_full", residual.SHRINKAGE_MATCHES), ("b0", "r_full", k_full),
                               ("rem", "r_rem", k_rem)]:
            table, pos = player_residuals(fit, col, k)
            attach(test, table, pos, prefix)
        season = int(test["Season"].iloc[0])
        cutoff = test["Kickoff"].min()
        plain, tilted, opp = band_distributions(test, so, const, season, cutoff)
        test["p_plain"], test["p_tilt"] = list(plain), list(tilted)
        rng = np.random.default_rng(SEED)
        normals, uniforms = {}, {}
        for k_, idx in sorted(test.groupby("Match_Key").indices.items()):
            normals[k_] = rng.standard_normal((len(idx), N_DRAWS))
            uniforms[k_] = rng.random((len(idx), N_DRAWS))
        for arm in ARMS:
            test[f"pred_{arm}"], test[f"pany_{arm}"] = simulate(test, normals, uniforms, arm)
            print(f"{name} {arm}: done")
        test["E_band_plain"], test["E_band_tilt"] = plain @ BANDS, tilted @ BANDS
        results["windows"][name] = summarize(test)
        results["windows"][name]["cutoff"] = str(cutoff)
        results["windows"][name]["opponent_effects"] = {k_: round(v, 3) for k_, v in opp.items()}
        tested.append(test.assign(Window=name))

    pooled = pd.concat(tested, ignore_index=True)
    results["pooled"] = summarize(pooled)
    results["pooled"]["bootstrap"] = bootstrap(pooled)
    inc, a = results["pooled"]["incumbent"], results["pooled"]["A"]
    ci = results["pooled"]["bootstrap"]["A_minus_incumbent_mse_ci95"]
    rule = {"i_mse_improves_1pct": a["mse"] <= 0.99 * inc["mse"], "i_interval_below_zero": ci[1] < 0,
            "ii_brier_not_worse_1pct": a["brier_any_bonus"] <= 1.01 * inc["brier_any_bonus"],
            "iii_no_position_abs_bias_up_003": all(abs(a["position_bias"][p]) - abs(inc["position_bias"][p]) <= 0.03
                                                   for p in POSITIONS)}
    rule["adopt_A"] = all(rule.values())
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["Window", "Season", "Round", "MatchID", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "BPS",
            "Bonus", "PB_actual", "E_band_plain", "E_band_tilt"] + [c for c in pooled if c.startswith(("pred_", "pany_"))]
    pooled[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "windows"}, indent=2, default=float))
    for name, w in results["windows"].items():
        print(name, {arm: round(w[arm]["mse"], 5) for arm in ARMS})


if __name__ == "__main__":
    main()
