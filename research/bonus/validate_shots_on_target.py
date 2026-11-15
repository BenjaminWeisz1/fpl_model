"""
Experiment B (pre-registered in CHANGELOG §69): shots on target as an explicit
bonus component, against the §68 production incumbent.

Conditional hindcast as validate_pass_bands.py: realized minutes and the other
explicit events are kept; pass bands, non-goal shots on target and the residual
are drawn from pre-match information. Arms: incumbent (§68: tilted pass bands,
K_REM residual), C_goal (+2 BPS per realized goal), C_noopp (+ Poisson non-goal
shots on target from the player's shrunk rate), C (+ opponent and venue
multipliers). A guard reruns every arm with realized non-penalty goals replaced by
draws from lagged pre-match rates. Common random numbers across arms; the
incumbent reuses the §68 random stream, so it reproduces §68's arm A exactly.

Run locally:
    python validate_shots_on_target.py
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats

from fpl_model.bonus import build_residual_bps_baseline as residual
from fpl_model.bonus import pass_bands, simulate_bonus
from research.bonus import validate_bonus_accounting as acc
from research.bonus import validate_bonus_calibration as calibration
from research.bonus import validate_pass_bands as vpb

OUT_DIR = calibration.ROOT / "data" / "validation" / "shots_on_target_20261028"
PASS_BANDS_RESULTS = calibration.ROOT / "data" / "validation" / "pass_bands_20261028" / "hindcast_predictions.csv"
HISTORY = 38
CONSTANT_SEASONS = range(2018, 2025)
POSITIONS = acc.POSITIONS
N_DRAWS = acc.N_DRAWS
SEED = acc.SEED
SOT_BPS = 2
BASE_ARMS = ["incumbent", "C_goal", "C_noopp", "C"]
ARMS = BASE_ARMS + [a + "_drawn" for a in BASE_ARMS]


def load_sofascore():
    """§68 Sofascore rows plus non-goal shots on target, NP goals and 90s played."""
    so = vpb.load_sofascore()
    extra = pd.read_csv(calibration.DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig",
                        low_memory=False, usecols=["Sofascore_goals"])
    so["goals"] = extra.loc[so.index, "Sofascore_goals"].fillna(0)
    pms = pd.read_csv(calibration.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Season", "Player", "Team", "Opponent", "Venue", "PK"])
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    pk = so.rename(columns={"Local_Player": "Player"})[key].merge(pms.drop_duplicates(key), on=key, how="left")
    so["PK_pms"] = pk["PK"].fillna(0).to_numpy()
    so["ng"] = (so["Sofascore_onTargetScoringAttempt"].fillna(0) - so["goals"]).clip(lower=0)
    so["npg"] = (so["goals"] - so["PK_pms"]).clip(lower=0)
    so["n90"] = so["Sofascore_minutesPlayed"].fillna(0) / 90
    return so


def mom_k(hist, col):
    """Shrinkage (appearances) by position: within-player-season variance / persistent variance."""
    out = {}
    for p, g in hist.groupby("Position_Sofascore"):
        ps = g.groupby(["Canonical_Player_ID", "Season"])[col]
        st = pd.DataFrame({"mean": ps.mean(), "var": ps.var(), "n": ps.size()}).reset_index()
        st = st[st["n"] >= 10]
        nxt = st.assign(Season=st["Season"] - 1)[["Canonical_Player_ID", "Season", "mean"]]
        pair = st.merge(nxt, on=["Canonical_Player_ID", "Season"], suffixes=("", "_n"))
        tau2 = np.cov(pair["mean"], pair["mean_n"])[0, 1] if len(pair) > 2 else 0
        out[p] = float(np.clip(st["var"].mean() / tau2, 0.5, 200)) if tau2 > 0 else 200.0
    return out


def constants(so):
    """Priors, k_ng, k_g, venue multipliers and k_opp from 2018-2024 60+ appearances."""
    hist = so[so["Season"].isin(CONSTANT_SEASONS) & so["full"]].copy()
    by_pos = hist.groupby("Position_Sofascore")
    prior_ng = (by_pos["ng"].sum() / by_pos["n90"].sum()).to_dict()
    prior_g = (by_pos["npg"].sum() / by_pos["n90"].sum()).to_dict()
    ps = hist.groupby(["Canonical_Player_ID", "Season"])
    hist["E"] = (ps["ng"].transform("sum") / ps["n90"].transform("sum")) * hist["n90"]
    venue = (hist.groupby("Venue")["ng"].sum() / hist.groupby("Venue")["E"].sum()).to_dict()
    md = hist.groupby(["Opponent", "Season", "match"])[["ng", "E"]].sum().reset_index()
    md["half"] = md.groupby(["Opponent", "Season"]).cumcount() % 2
    halves = md.groupby(["Opponent", "Season", "half"])[["ng", "E"]].sum()
    ratio = (halves["ng"] / halves["E"]).unstack().dropna()
    tau2 = np.cov(ratio[0], ratio[1])[0, 1]
    k_opp = float(np.clip(1 / tau2, 1, 1000)) if tau2 > 0 else 1000.0
    return {"prior_ng": prior_ng, "prior_g": prior_g, "k_ng": mom_k(hist, "ng"), "k_g": mom_k(hist, "npg"),
            "venue": venue, "k_opp": k_opp}


def shot_rates(test, so, const, season, cutoff):
    """Pre-match per-90 non-goal SoT rate, opponent x venue multiplier and NP-goal rate per test row."""
    hist = so[so["full"] & (so["Kickoff"] < cutoff)].sort_values("Kickoff")
    recent = hist.groupby("Canonical_Player_ID").tail(HISTORY)
    sums = recent.groupby("Canonical_Player_ID")[["ng", "npg", "n90"]].sum()
    pmean = sums["ng"] / sums["n90"]
    cur = hist[hist["Season"] == season].copy()
    cur["E"] = cur["Canonical_Player_ID"].map(pmean) * cur["n90"]
    oe = cur.groupby("Opponent")[["ng", "E"]].sum()
    opp = ((oe["ng"] + const["k_opp"]) / (oe["E"] + const["k_opp"])).to_dict()
    s = sums.reindex(test["Canonical_Player_ID"]).fillna(0)
    pos = test["Pos"].to_numpy()
    k_ng, k_g = test["Pos"].map(const["k_ng"]).to_numpy(), test["Pos"].map(const["k_g"]).to_numpy()
    rate_ng = (s["ng"].to_numpy() + k_ng * pd.Series(pos).map(const["prior_ng"]).to_numpy()) / (s["n90"].to_numpy() + k_ng)
    rate_g = (s["npg"].to_numpy() + k_g * pd.Series(pos).map(const["prior_g"]).to_numpy()) / (s["n90"].to_numpy() + k_g)
    mult = test["Opponent"].map(opp).fillna(1.0).to_numpy() * test["Venue"].map(const["venue"]).to_numpy()
    return rate_ng, mult, rate_g, opp


def poisson(u, mu):
    return np.nan_to_num(stats.poisson.ppf(u, np.clip(mu, 0, None)[:, None]))


def simulate(test, draws, arm):
    drawn = arm.endswith("_drawn")
    base = arm.removesuffix("_drawn")
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    for key, idx in test.groupby("Match_Key").indices.items():
        sub = test.iloc[idx]
        normals, u_pb, u_ng, u_g = draws[key]
        minutes = sub["Minutes"].to_numpy()
        cum = np.cumsum(np.stack(sub["p_tilt"].to_numpy()), axis=1)
        pb = vpb.BANDS[(u_pb[:, :, None] > cum[:, None, :]).sum(axis=2).clip(max=3)]
        pb = np.where(minutes[:, None] >= 60, pb, 0)
        explicit = sub["Explicit_BPS"].to_numpy()[:, None]
        npg = sub["NPG"].to_numpy()[:, None]
        if drawn:
            npg_draw = poisson(u_g, sub["rate_g"].to_numpy() * minutes / 90)
            explicit = explicit + sub["goal_rate_bps"].to_numpy()[:, None] * (npg_draw - npg)
            npg = npg_draw
        prefix = {"incumbent": "inc", "C_goal": "goal", "C_noopp": "sot", "C": "sot"}[base]
        total = explicit + pb + sub[prefix + "_m"].to_numpy()[:, None] + sub[prefix + "_s"].to_numpy()[:, None] * normals
        if base != "incumbent":
            total = total + SOT_BPS * (npg + sub["PK"].to_numpy()[:, None])
        if base in ("C_noopp", "C"):
            mu = sub["rate_ng"].to_numpy() * minutes / 90 * (sub["mult"].to_numpy() if base == "C" else 1)
            total = total + SOT_BPS * poisson(u_ng, mu)
        bonus = simulate_bonus.assign_bonus(total)
        exp_bonus[idx], p_any[idx] = bonus.mean(axis=1), (bonus > 0).mean(axis=1)
    return exp_bonus, p_any


def summarize(frame):
    out = {"n_player_matches": int(len(frame)), "n_matches": int(frame["Match_Key"].nunique()),
           "support": frame["Pos"].value_counts().reindex(POSITIONS).fillna(0).astype(int).to_dict()}
    for arm in ARMS:
        err = frame[f"pred_{arm}"] - frame["Bonus"]
        out[arm] = {"mse": float((err ** 2).mean()), "mae": float(err.abs().mean()),
                    "brier_any_bonus": float(((frame[f"pany_{arm}"] - (frame["Bonus"] > 0)) ** 2).mean()),
                    "position_bias": err.groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict()}
    # Calibration of the non-goal shot rates on Sofascore-matched rows.
    m = frame[frame["ng_matched"]]
    out["nongoal_sot_mean_by_pos"] = {
        "actual": m.groupby("Pos")["ng_actual"].mean().reindex(POSITIONS).round(4).to_dict(),
        "expected_noopp": m.groupby("Pos")["E_ng_noopp"].mean().reindex(POSITIONS).round(4).to_dict(),
        "expected_C": m.groupby("Pos")["E_ng_C"].mean().reindex(POSITIONS).round(4).to_dict()}
    out["npg_mean_by_pos"] = {"actual": frame.groupby("Pos")["NPG"].mean().reindex(POSITIONS).round(4).to_dict(),
                              "expected_lagged": frame.groupby("Pos")["E_g"].mean().reindex(POSITIONS).round(4).to_dict()}
    return out


PAIRS = [("C", "incumbent"), ("C_goal", "incumbent"), ("C_noopp", "C_goal"), ("C", "C_noopp"),
         ("C_drawn", "incumbent_drawn"), ("C_goal_drawn", "incumbent_drawn")]


def bootstrap(frame):
    keys = np.sort(frame["Match_Key"].unique())
    sq = {arm: ((frame[f"pred_{arm}"] - frame["Bonus"]) ** 2).groupby(frame["Match_Key"]).sum().reindex(keys).to_numpy()
          for arm in ARMS}
    n = frame.groupby("Match_Key").size().reindex(keys).to_numpy()
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(p, minlength=len(keys))
                  for p in rng.integers(0, len(keys), size=(acc.BOOTSTRAP_REPLICATES, len(keys)))])
    mse = {arm: (w @ sq[arm]) / (w @ n) for arm in ARMS}
    return {f"{a}_minus_{b}_mse_ci95": np.percentile(mse[a] - mse[b], [2.5, 97.5]).round(5).tolist() for a, b in PAIRS}


def definition_check(fit):
    """Controlled within-player coefficients (BPS per event) of non-goal SoT and NP goals not won."""
    d = fit[~fit["Canonical_Player_ID"].str.startswith("unjoined:")].copy()
    won = d["GF"] > d["GA"]
    d["npg_nowin"], d["npg_win"] = d["NPG"] * ~won, d["NPG"] * won
    d["pk_miss"] = (d["PKatt"] - d["PK"]).clip(lower=0)
    xs = ["ng_actual", "npg_nowin", "npg_win", "PK", "pk_miss", "PB_actual"] + [
        c for c in d if c.startswith("c_") and c != "c_onTargetScoringAttempt"]
    g = d.groupby("Player")
    y = (d["r_full"] - g["r_full"].transform("mean")).to_numpy(float)
    x = np.column_stack([d[c] - g[c].transform("mean") for c in xs]).astype(float)
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return {"nongoal_sot": float(coef[0]), "npg_not_won": float(coef[1]), "npg_won": float(coef[2]),
            "pk_goal": float(coef[3])}


def main():
    frozen_before = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    rows, excluded = acc.load_rows()
    so = load_sofascore()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    join = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    controls = ["c_" + k for k in vpb.CONTROLS]
    rows = rows.merge(join[key + ["Canonical_Player_ID", "Kickoff", "band", "ng"] + controls], on=key, how="left")
    joined = float(rows["Canonical_Player_ID"].notna().mean())
    rows["ng_matched"] = rows["ng"].notna()
    rows["ng_actual"] = rows["ng"].fillna(0)
    rows["PB_actual"] = rows["band"].fillna(0)
    rows["Canonical_Player_ID"] = rows["Canonical_Player_ID"].fillna("unjoined:" + rows["Player"])
    res = pd.read_csv(calibration.RESULTS_FILE, encoding="utf-8-sig", low_memory=False)
    sides = pd.concat([
        res[["Season", "Home", "Away", "Home_Goals", "Away_Goals"]].set_axis(["Season", "Team", "Opponent", "GF", "GA"], axis=1).assign(Venue="Home"),
        res[["Season", "Away", "Home", "Away_Goals", "Home_Goals"]].set_axis(["Season", "Team", "Opponent", "GF", "GA"], axis=1).assign(Venue="Away")])
    rows = rows.merge(sides, on=["Season", "Team", "Opponent", "Venue"], how="left", validate="many_to_one")
    known = residual.add_known_bps(rows, legacy=False, save_scale=1.0)
    rows["r_full"] = (rows["BPS"] - known["known_bps"]).astype(float)
    rows["r_rem"] = rows["r_full"] - rows["PB_actual"]
    rows["r_goal"] = rows["r_rem"] - SOT_BPS * (rows["NPG"] + rows["PK"])
    rows["r_sot"] = rows["r_goal"] - SOT_BPS * rows["ng_actual"]
    rows["Explicit_BPS"] = known["known_bps"].to_numpy(float)
    rows["goal_rate_bps"] = rows["Pos"].map(residual.GOAL_BPS).astype(float)
    const_pb = vpb.constants(so)
    const = constants(so)

    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    k_rem = vpb.split_half_k(wins["A_2025"][0], "r_rem")
    assert all(abs(k_rem[p] - pass_bands.K_REM[p]) < 1e-3 for p in POSITIONS), (k_rem, pass_bands.K_REM)
    results = {"join_rate": joined, "excluded_played_rows_without_position": excluded,
               "constants": {k: v for k, v in const.items()}, "k_rem_frozen_68": pass_bands.K_REM,
               "definition_check": {}, "windows": {}}

    for name, (fit, _, _) in wins.items():
        results["definition_check"][name] = definition_check(fit)
    coefs = [v for c in results["definition_check"].values() for k, v in c.items() if k in ("nongoal_sot", "npg_not_won")]
    print(json.dumps({"constants": results["constants"], "definition_check": results["definition_check"]},
                     indent=2, default=float))
    if not all(1.4 <= c <= 2.6 for c in coefs):
        results["stopped"] = "definition check failed"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        return

    tested = []
    for name, (fit, test, _) in wins.items():
        test = test.copy()
        for prefix, col in [("inc", "r_rem"), ("goal", "r_goal"), ("sot", "r_sot")]:
            table, pos = vpb.player_residuals(fit, col, k_rem)
            vpb.attach(test, table, pos, prefix)
        season = int(test["Season"].iloc[0])
        cutoff = test["Kickoff"].min()
        _, tilted, _ = vpb.band_distributions(test, so, const_pb, season, cutoff)
        test["p_tilt"] = list(tilted)
        test["rate_ng"], test["mult"], test["rate_g"], opp = shot_rates(test, so, const, season, cutoff)
        test["E_ng_noopp"] = test["rate_ng"] * test["Minutes"] / 90
        test["E_ng_C"] = test["E_ng_noopp"] * test["mult"]
        test["E_g"] = test["rate_g"] * test["Minutes"] / 90
        # §68's stream (normals, pass-band uniforms) first, so the incumbent reproduces §68 arm A.
        rng, rng2 = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
        draws = {}
        for k_, idx in sorted(test.groupby("Match_Key").indices.items()):
            normals, u_pb = rng.standard_normal((len(idx), N_DRAWS)), rng.random((len(idx), N_DRAWS))
            draws[k_] = (normals, u_pb, rng2.random((len(idx), N_DRAWS)), rng2.random((len(idx), N_DRAWS)))
        for arm in ARMS:
            test[f"pred_{arm}"], test[f"pany_{arm}"] = simulate(test, draws, arm)
            print(f"{name} {arm}: done")
        results["windows"][name] = summarize(test)
        results["windows"][name]["cutoff"] = str(cutoff)
        results["windows"][name]["opponent_multipliers"] = {k_: round(v, 3) for k_, v in opp.items()}
        tested.append(test.assign(Window=name))

    pooled = pd.concat(tested, ignore_index=True)
    ref = pd.read_csv(PASS_BANDS_RESULTS, encoding="utf-8-sig")
    chk = pooled.merge(ref[["Season", "MatchID", "Player", "pred_A"]], on=["Season", "MatchID", "Player"], how="left")
    results["incumbent_reproduces_68_arm_A_max_abs_diff"] = float((chk["pred_incumbent"] - chk["pred_A"]).abs().max())
    results["pooled"] = summarize(pooled)
    results["pooled"]["bootstrap"] = bootstrap(pooled)

    def rule(cand):
        p, b = results["pooled"], results["pooled"]["bootstrap"]
        inc, c = p["incumbent"], p[cand]
        r = {"i_mse_improves_0.5pct": c["mse"] <= 0.995 * inc["mse"],
             "i_interval_below_zero": b[f"{cand}_minus_incumbent_mse_ci95"][1] < 0,
             "ii_brier_not_worse_1pct": c["brier_any_bonus"] <= 1.01 * inc["brier_any_bonus"],
             "iii_no_position_abs_bias_up_003": all(
                 abs(c["position_bias"][q]) - abs(inc["position_bias"][q]) <= 0.03 for q in POSITIONS),
             "iv_guard_drawn_mse_not_worse": p[cand + "_drawn"]["mse"] <= p["incumbent_drawn"]["mse"]}
        r["pass"] = all(r.values())
        return r

    decision = {"C": rule("C"), "C_goal": rule("C_goal")}
    decision["adopt"] = "C" if decision["C"]["pass"] else ("C_goal" if decision["C_goal"]["pass"] else "incumbent")
    results["decision_rule"] = decision
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = ["Window", "Season", "Round", "MatchID", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "BPS",
            "Bonus", "NPG", "PK", "ng_actual", "E_ng_noopp", "E_ng_C", "E_g"] + [
        c for c in pooled if c.startswith(("pred_", "pany_"))]
    pooled[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k not in ("windows", "constants")}, indent=2, default=float))
    for name, w in results["windows"].items():
        print(name, {arm: round(w[arm]["mse"], 5) for arm in ARMS})


if __name__ == "__main__":
    main()
