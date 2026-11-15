"""
Experiment D (pre-registered in CHANGELOG §72): a pre-match bonus hindcast, and
team-goal coherence tested on it.

Runs the production simulate_bonus.simulate_match on every test match with a
features frame built from pre-match information only, except realized minutes:
pre-match goal/assist ratings logged by build_player_ratings.py
(GOAL_MODEL_PREMATCH_LOG), one-match-ahead team predictions from
team_ratings_history.csv, fit-window penalty shares and defcon rates, and the
§68/§70 residual and tilted pass bands. Arms: incumbent (production) and
coherent (the §71 variant, shadow_team_goal_coherence.coherent_match). Both arms
reset the simulator's generator to the same per-match seed.

Validity is checked before any comparison exists (§89): rating coverage first, then
the incumbent alone and its bonus per match, and only then the coherent arm.

Run locally (after writing the pre-match rating log to RATING_LOG, see §72/§89):
    python validate_prematch_coherence.py
"""
import hashlib
import json

import numpy as np
import pandas as pd

import build_residual_bps_baseline as residual
import shadow_team_goal_coherence as coherence
import simulate_bonus as sb
import validate_bonus_accounting as acc
import validate_bonus_calibration as calibration
import validate_pass_bands as vpb

# §89 rerun; the §72 run's evidence stays in prematch_bonus_20261028/
OUT_DIR = calibration.ROOT / "data" / "validation" / "prematch_bonus_rerun_20261028"
RATING_LOG = OUT_DIR / "player_ratings_prematch.csv"
POSITIONS = acc.POSITIONS
N_DRAWS = acc.N_DRAWS
SEED = acc.SEED
ARMS = ["incumbent", "coherent"]
PEN_ATTEMPTS_PER_NPG = 792 / 7734
PEN_FACTOR = 1 + PEN_ATTEMPTS_PER_NPG * 0.79
VENUE_ATT = {"Home": 1.11, "Away": 0.89}
SHRINKAGE_PRIOR = 30
DEFCON_K = 10


def conversion_lookup(season):
    """Career (NPG + 30) / (npxG + 30) from player_match_stats rows before `season`."""
    pms = pd.read_csv(calibration.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Season", "Player", "Goals", "PK", "npxG"])
    pms = pms[pms["Season"] < season]
    t = pms.groupby("Player")[["Goals", "PK", "npxG"]].sum()
    return ((t["Goals"] - t["PK"] + SHRINKAGE_PRIOR) / (t["npxG"] + SHRINKAGE_PRIOR)).to_dict()


def prematch_ratings(test):
    log = pd.read_csv(RATING_LOG, encoding="utf-8-sig")
    log = log[log["Data_Source"].eq("FPL_API")].rename(columns={"MatchID": "PMS_MatchID"})
    key = ["Player", "Team", "Opponent", "Venue", "PMS_MatchID"]
    dups = int(log.duplicated(key).sum())
    log = log.drop_duplicates(key, keep=False)
    out = test[key].merge(log[key + ["NPG_Rating_Pre", "Assist_Rating_Pre"]], on=key, how="left")
    return out[["NPG_Rating_Pre", "Assist_Rating_Pre"]].to_numpy(), dups


def team_context(test, fit, history):
    """Team/opponent one-match-ahead predictions and the opponent's pre-match defence rating."""
    h = history.set_index(["Season", "MatchID"])
    idx = pd.MultiIndex.from_arrays([test["Season"], test["MatchID"]])
    m = h.reindex(idx)
    home = test["Venue"].eq("Home").to_numpy()
    team_pred = np.where(home, m["Home_Prediction"], m["Away_Prediction"])
    opp_pred = np.where(home, m["Away_Prediction"], m["Home_Prediction"])
    opp_def = np.where(home, m["Away_Defense_Rating"], m["Home_Defense_Rating"])
    fit_matches = h.reindex(pd.MultiIndex.from_arrays([fit["Season"], fit["MatchID"]]).unique())
    avg_def = float(pd.concat([fit_matches["Home_Defense_Rating"], fit_matches["Away_Defense_Rating"]]).mean())
    return team_pred, opp_pred, opp_def, avg_def


def penalty_shares(fit, fallback):
    def shares(f):
        p = f.groupby(["Team", "Player"])["PKatt"].sum()
        t = p.groupby(level=0).transform("sum")
        return (p / t.where(t > 0)).dropna()
    s = shares(fit)
    if fallback is not None:
        fb = shares(fallback)
        missing = sorted(set(fb.index.get_level_values(0)) - set(s.index.get_level_values(0)))
        s = pd.concat([s, fb[fb.index.get_level_values(0).isin(missing)]])
    return s


def defcon_rates(fit):
    f = fit[fit["Minutes"] >= 60].copy()
    f["count"] = np.where(f["Pos"].eq("D"), f["CBI"] + f["Tkl"],
                          np.where(f["Pos"].isin(["M", "F"]), f["CBI"] + f["Tkl"] + f["Recov"], 0))
    g = f.groupby(["Player", "Pos"]).agg(c=("count", "sum"), mins=("Minutes", "sum"), n=("count", "size"))
    g["rate"] = g["c"] / g["mins"] * 90
    pos_rate = (f.groupby("Pos")["count"].sum() / f.groupby("Pos")["Minutes"].sum() * 90).to_dict()
    pr = g.index.get_level_values("Pos").map(pos_rate)
    g["mu90"] = (g["n"] * g["rate"] + DEFCON_K * pr) / (g["n"] + DEFCON_K)
    return g["mu90"], pos_rate


def build_features(name, fit, test, rows, so, const_pb, k_rem, history):
    test = test.copy()
    season = int(test["Season"].iloc[0])
    ratings, dups = prematch_ratings(test)
    test["NPG_Rating_Pre"], test["Assist_Rating_Pre"] = ratings[:, 0], ratings[:, 1]
    coverage = float(test["NPG_Rating_Pre"].notna().mean())
    for col in ["NPG_Rating_Pre", "Assist_Rating_Pre"]:
        test[col] = test[col].fillna(test.groupby("Pos")[col].transform("median"))
    conv = conversion_lookup(season)
    test["conversion"] = test["Player"].map(conv).fillna(1.0)
    team_pred, opp_pred, opp_def, avg_def = team_context(test, fit, history)
    test["Team_Pred_NPG"], test["Opp_Pred_NPG"] = team_pred, opp_pred
    scale = opp_def * test["Venue"].map(VENUE_ATT).to_numpy() / avg_def
    test["NPG90_scaled"] = test["NPG_Rating_Pre"] * test["conversion"] * scale
    test["Ast90_scaled"] = test["Assist_Rating_Pre"] * scale

    fallback = rows[rows["Season"] == 2025] if season == 2026 else None
    shares = penalty_shares(fit, fallback)
    share = pd.Series(shares.reindex(pd.MultiIndex.from_arrays([test["Team"], test["Player"]])).to_numpy()).fillna(0)
    test["PenAttempts"] = test["Team_Pred_NPG"] * PEN_ATTEMPTS_PER_NPG * share.to_numpy()

    mu, pos_rate = defcon_rates(fit)
    test["Defcon_Mu90"] = mu.reindex(pd.MultiIndex.from_arrays([test["Player"], test["Pos"]])).to_numpy()
    test["Defcon_Mu90"] = test["Defcon_Mu90"].fillna(test["Pos"].map(pos_rate)).fillna(0)
    test.loc[test["Pos"].eq("G"), "Defcon_Mu90"] = 0.0

    table, pos = vpb.player_residuals(fit, "r_rem", k_rem)
    vpb.attach(test, table, pos, "rem")
    test["ResidualBPS_Mean"], test["ResidualBPS_Std"] = test["rem_m"], test["rem_s"]
    _, tilted, _ = vpb.band_distributions(test, so, const_pb, season, test["Kickoff"].min())
    test[["PB_P0", "PB_P2", "PB_P4", "PB_P6"]] = tilted
    test["P1"] = 1.0
    test["P60"] = (test["Minutes"] >= 60).astype(float)
    return test, {"rating_coverage": coverage, "rating_log_duplicates_dropped": dups,
                  "avg_defense_rating": avg_def, "players_with_penalty_share": int((share > 0).sum())}


def simulate(test, history, arm):
    """Expected bonus and any-bonus probability per row. assign_bonus is wrapped to keep
    each match's bonus matrix, since simulate_match returns only its mean."""
    fn = sb.simulate_match if arm == "incumbent" else coherence.coherent_match
    h = history.set_index(["Season", "MatchID"])
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    original, captured = sb.assign_bonus, []
    sb.assign_bonus = lambda m: captured.append(original(m)) or captured[-1]
    try:
        for i, (key, idx) in enumerate(sorted(test.groupby("Match_Key").indices.items())):
            sub = test.iloc[idx]
            m = h.loc[(int(sub["Season"].iloc[0]), int(sub["MatchID"].iloc[0]))]
            sb.rng = np.random.default_rng(SEED + i)
            exp_bonus[idx] = fn(sub, m["Away_Prediction"] * PEN_FACTOR, m["Home_Prediction"] * PEN_FACTOR,
                                m["Home"], N_DRAWS)[0]
            p_any[idx] = (captured.pop() > 0).mean(axis=1)
    finally:
        sb.assign_bonus = original
    return exp_bonus, p_any


def input_calibration(test):
    frac = np.where(test["P60"] == 1, 1.0, sb.AVG_MINUTES_SHORT / 90)
    pred = pd.DataFrame({"Pos": test["Pos"], "NPG": test["NPG90_scaled"] * frac, "Assists": test["Ast90_scaled"] * frac,
                         "PKatt": test["PenAttempts"], "Defcon": test["Defcon_Mu90"] * frac})
    count = np.where(test["Pos"].eq("D"), test["CBI"] + test["Tkl"],
                     np.where(test["Pos"].isin(["M", "F"]), test["CBI"] + test["Tkl"] + test["Recov"], 0))
    act = pd.DataFrame({"Pos": test["Pos"], "NPG": test["NPG"], "Assists": test["Assists"],
                        "PKatt": test["PKatt"], "Defcon": count})
    out = {c: {"predicted": pred.groupby("Pos")[c].mean().reindex(POSITIONS).round(4).to_dict(),
               "actual": act.groupby("Pos")[c].mean().reindex(POSITIONS).round(4).to_dict()}
           for c in ["NPG", "Assists", "PKatt", "Defcon"]}
    teams = test.drop_duplicates(["Match_Key", "Team"])
    out["team_goals_conceded"] = {"expected": float((teams["Opp_Pred_NPG"] * PEN_FACTOR).mean()),
                                  "actual": float(teams["Goals_Conceded_Team"].mean())}
    return out


def summarize(frame):
    per_match = frame.groupby("Match_Key")
    out = {"n_player_matches": int(len(frame)), "n_matches": int(per_match.ngroups),
           "actual_bonus_per_match": float(per_match["Bonus"].sum().mean())}
    for arm in ARMS:
        err = frame[f"pred_{arm}"] - frame["Bonus"]
        out[arm] = {"mse": float((err ** 2).mean()), "mae": float(err.abs().mean()),
                    "brier_any_bonus": float(((frame[f"pany_{arm}"] - (frame["Bonus"] > 0)) ** 2).mean()),
                    "overall_bias": float(err.mean()),
                    "bonus_per_match": float(per_match[f"pred_{arm}"].sum().mean()),
                    "position_bias": err.groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(frame["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_share": (frame.groupby("Pos")[f"pred_{arm}"].sum() / frame[f"pred_{arm}"].sum())
                    .reindex(POSITIONS).round(4).to_dict()}
    out["actual_position_share"] = (frame.groupby("Pos")["Bonus"].sum() / frame["Bonus"].sum()).reindex(POSITIONS).round(4).to_dict()
    return out


def bootstrap(frame):
    keys = np.sort(frame["Match_Key"].unique())
    n = frame.groupby("Match_Key").size().reindex(keys).to_numpy()
    sq = {a: ((frame[f"pred_{a}"] - frame["Bonus"]) ** 2).groupby(frame["Match_Key"]).sum().reindex(keys).to_numpy() for a in ARMS}
    e = {a: (frame[f"pred_{a}"] - frame["Bonus"]).groupby([frame["Match_Key"], frame["Pos"]]).sum()
         .unstack(fill_value=0).reindex(index=keys, columns=POSITIONS, fill_value=0).to_numpy() for a in ARMS}
    npos = frame.groupby(["Match_Key", "Pos"]).size().unstack(fill_value=0).reindex(
        index=keys, columns=POSITIONS, fill_value=0).to_numpy()
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(p, minlength=len(keys))
                  for p in rng.integers(0, len(keys), size=(acc.BOOTSTRAP_REPLICATES, len(keys)))])
    mse = {a: (w @ sq[a]) / (w @ n) for a in ARMS}
    ci = lambda x: np.percentile(x, [2.5, 97.5]).round(5).tolist()
    bias = {a: (w @ e[a]) / (w @ npos) for a in ARMS}
    return {"coherent_minus_incumbent_mse_ci95": ci(mse["coherent"] - mse["incumbent"]),
            "coherent_minus_incumbent_position_bias_ci95": {
                p: ci(bias["coherent"][:, i] - bias["incumbent"][:, i]) for i, p in enumerate(POSITIONS)}}


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
    gc = rows.groupby(["Season", "MatchID", "Team"])["Goals_Conceded"].max().rename("Goals_Conceded_Team")
    rows = rows.merge(gc.reset_index(), on=["Season", "MatchID", "Team"], how="left")
    const_pb = vpb.constants(so)
    history = pd.read_csv(calibration.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)

    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    k_rem = vpb.split_half_k(wins["A_2025"][0], "r_rem")
    results = {"excluded_played_rows_without_position": excluded, "windows": {}, "validity": {}}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def stop(reason):
        results["stopped"] = f"validity check failed ({reason}); arms not compared"
        results["frozen_gw8_sha256_unchanged"] = (
            frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(json.dumps({k: v for k, v in results.items() if k != "windows"}, indent=2, default=float))

    # 1. inputs and rating coverage, before anything is simulated
    built = {}
    for name, (fit, test, _) in wins.items():
        test, info = build_features(name, fit, test, rows, so, const_pb, k_rem, history)
        results["windows"][name] = {"inputs": info, "input_calibration": input_calibration(test)}
        print(name, json.dumps(info))
        built[name] = test
    results["validity"]["rating_coverage_ge_95"] = all(
        w["inputs"]["rating_coverage"] >= 0.95 for w in results["windows"].values())
    if not results["validity"]["rating_coverage_ge_95"]:
        return stop("rating coverage")

    # 2. the incumbent alone; its bonus per match is the second validity check
    for name, test in built.items():
        test["pred_incumbent"], test["pany_incumbent"] = simulate(test, history, "incumbent")
        print(f"{name} incumbent: done")
    inc_pooled = pd.concat(built.values(), ignore_index=True)
    bpm = float(inc_pooled.groupby("Match_Key")["pred_incumbent"].sum().mean())
    results["validity"]["incumbent_bonus_per_match"] = bpm
    results["validity"]["incumbent_bonus_per_match_6.1_to_6.6"] = 6.1 <= bpm <= 6.6
    if not results["validity"]["incumbent_bonus_per_match_6.1_to_6.6"]:
        return stop("incumbent bonus per match")

    # 3. only now the coherent arm, then any comparison
    tested = []
    for name, test in built.items():
        test["pred_coherent"], test["pany_coherent"] = simulate(test, history, "coherent")
        print(f"{name} coherent: done")
        results["windows"][name].update(summarize(test))
        tested.append(test.assign(Window=name))
    pooled = pd.concat(tested, ignore_index=True)
    results["pooled"] = summarize(pooled)
    results["pooled"]["bootstrap"] = bootstrap(pooled)
    inc, coh = results["pooled"]["incumbent"], results["pooled"]["coherent"]
    rule = {"i_mse_not_worse": coh["mse"] <= inc["mse"],
            "ii_brier_not_worse_1pct": coh["brier_any_bonus"] <= 1.01 * inc["brier_any_bonus"],
            "iii_no_position_abs_bias_up_003": all(
                abs(coh["position_bias"][p]) - abs(inc["position_bias"][p]) <= 0.03 for p in POSITIONS)}
    rule["adopt"] = all(rule.values())
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    keep = ["Window", "Season", "Round", "MatchID", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "BPS",
            "Bonus", "NPG", "Assists", "NPG90_scaled", "Ast90_scaled", "PenAttempts", "Defcon_Mu90",
            "Team_Pred_NPG", "Opp_Pred_NPG"] + [c for c in pooled if c.startswith(("pred_", "pany_"))]
    pooled[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "windows"}, indent=2, default=float))
    for name, w in results["windows"].items():
        print(name, json.dumps(w["input_calibration"]))
        print(name, {a: (round(w[a]["mse"], 5), round(w[a]["bonus_per_match"], 3)) for a in ARMS})


if __name__ == "__main__":
    main()
