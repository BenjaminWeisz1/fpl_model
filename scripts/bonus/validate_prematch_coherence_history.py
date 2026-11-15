"""
Team-goal coherence and keeper bonus on 2018-2024 history (pre-registered in CHANGELOG §144).

The §89 pre-match hindcast rebuilt on vaastav player-fixtures: six-gameweek test blocks, each fitted
on the 380 fixtures before it, with each season's own BPS rules (goals conceded cost nothing before
2024; penalty saves were 15 before 2024 and 9 in 2024). Minutes are realized, as in §89.

Run locally (after writing the pre-match rating log to RATING_LOG, see §144):
    python validate_prematch_coherence_history.py
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import build_residual_bps_baseline as residual
import pass_bands
import shadow_team_goal_coherence as coherence
import simulate_bonus as sb
import validate_bonus_calibration as calibration
import validate_pass_bands as vpb
import validate_prematch_coherence as vpc

OUT_DIR = calibration.ROOT / "data" / "validation" / "prematch_bonus_history_20261030"
RATING_LOG = OUT_DIR / "player_ratings_prematch.csv"
SEASONS = range(2018, 2025)
BLOCK_GWS = 6
FIT_FIXTURES = 380
OPP_BETA = {"npg": 0.7, "ast": 0.8}
SUB_UPLIFT = {"npg": 1.2003, "ast": 1.1964}
WORKERS = 6
POSITIONS = vpc.POSITIONS
ARMS = vpc.ARMS
ELEMENT_TYPES = {1: "G", 2: "D", 3: "M", 4: "F"}


def era_rules(season):
    return {"gc": -4 if season >= 2024 else 0, "pen_save": 9 if season == 2024 else 15 if season < 2024 else 7}


def set_rules(season):
    rules = era_rules(season)
    sb.GC_BPS = residual.GC_BPS = rules["gc"]
    sb.PEN_SAVE_BPS = residual.PEN_SAVE_BPS = rules["pen_save"]


def load_rows():
    v = pd.read_csv(calibration.DATA / "vaastav_player_match_history.csv", encoding="utf-8-sig", low_memory=False)
    positions = {}
    for path in sorted((calibration.ROOT / "data" / "raw" / "vaastav").glob("20*/players_raw.csv")):
        raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
        positions[int(path.parent.name[:4])] = dict(zip(raw["id"], raw["element_type"].map(ELEMENT_TYPES)))
    v["Pos"] = [positions[s].get(e) for s, e in zip(v["Season"], v["Element_FPL"])]
    v = v[v["minutes"] > 0].copy()
    v["Player"] = v["Local_Player"].fillna(v["Name_FPL"])
    v["Canonical_Player_ID"] = v["Canonical_Player_ID"].fillna("unjoined:" + v["Player"])
    v = v.rename(columns={"minutes": "Minutes", "assists": "Assists", "clean_sheets": "Clean_Sheet", "saves": "Saves",
                          "goals_conceded": "Goals_Conceded", "penalties_saved": "Penalties_Saved", "bps": "BPS",
                          "bonus": "Bonus", "GW": "Round"})
    pms = pd.read_csv(calibration.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Season", "Player", "Team", "Opponent", "Venue", "PK", "PKatt", "Tkl", "CBI", "Recov"])
    key = ["Season", "Player", "Team", "Opponent", "Venue"]
    v = v.merge(pms.drop_duplicates(key), on=key, how="left")
    joined = float(v["CBI"].notna().mean())
    v[["PK", "Tkl", "CBI", "Recov"]] = v[["PK", "Tkl", "CBI", "Recov"]].fillna(0)
    v["PKatt"] = v["PKatt"].fillna(v["penalties_missed"])
    v["NPG"] = (v["goals_scored"] - v["PK"]).clip(lower=0)
    v["Kickoff"] = pd.to_datetime(v["Kickoff"], utc=True)
    v["Match_Key"] = v["Season"].astype(str) + "_" + v["Fixture_FPL"].astype(str)
    gc = v.groupby(["Match_Key", "Team"])["Goals_Conceded"].max().rename("Goals_Conceded_Team")
    return v.merge(gc.reset_index(), on=["Match_Key", "Team"], how="left").reset_index(drop=True), joined


def attach_bands_and_residuals(rows, so):
    key = ["Season", "Team", "Opponent", "Venue", "Canonical_Player_ID"]
    rows = rows.merge(so.dropna(subset=["Canonical_Player_ID"]).drop_duplicates(key)[key + ["band"]], on=key, how="left")
    rows["PB_actual"] = rows["band"].fillna(0)
    parts = []
    for season, g in rows.groupby("Season"):
        set_rules(int(season))
        known = residual.add_known_bps(g, legacy=False, save_scale=1.0)
        parts.append((g["BPS"] - known["known_bps"]).astype(float) - g["PB_actual"])
    rows["r_rem"] = pd.concat(parts).reindex(rows.index)
    return rows


def blocks(rows):
    """(season, block, fit, test): six-gameweek test blocks, each fitted on the 380 fixtures before it."""
    order = rows.groupby("Match_Key")["Kickoff"].min().sort_values()
    out = []
    for season in SEASONS:
        s = rows[rows["Season"] == season]
        for b in sorted(((s["Round"] - 1) // BLOCK_GWS).unique()):
            test = s[(s["Round"] - 1) // BLOCK_GWS == b]
            start = test["Kickoff"].min()
            fit_keys = order[order < start].index[-FIT_FIXTURES:]
            out.append((season, int(b), rows[rows["Match_Key"].isin(fit_keys)], test))
    return out


def team_context(test, fit, history):
    h = history.set_index(["Season", "Home", "Away"])
    side = test["Venue"].eq("Home").to_numpy()
    home = np.where(side, test["Team"], test["Opponent"])
    away = np.where(side, test["Opponent"], test["Team"])
    m = h.reindex(pd.MultiIndex.from_arrays([test["Season"], home, away]))
    team = np.where(side, m["Home_Prediction"], m["Away_Prediction"])
    opp = np.where(side, m["Away_Prediction"], m["Home_Prediction"])
    opp_def = np.where(side, m["Away_Defense_Rating"], m["Home_Defense_Rating"])
    fh = fit.drop_duplicates("Match_Key")
    fs = fh["Venue"].eq("Home").to_numpy()
    fm = h.reindex(pd.MultiIndex.from_arrays([fh["Season"], np.where(fs, fh["Team"], fh["Opponent"]),
                                              np.where(fs, fh["Opponent"], fh["Team"])]))
    avg = float(pd.concat([fm["Home_Defense_Rating"], fm["Away_Defense_Rating"]]).mean())
    return team, opp, opp_def, avg, home, away


def build_block(fit, test, so, const, ratings, history):
    test = test.copy()
    season = int(test["Season"].iloc[0])
    key = ["Season", "Player", "Team", "Opponent", "Venue"]
    r = test[key].merge(ratings, on=key, how="left")
    test["NPG_Rating_Pre"], test["Assist_Rating_Pre"] = r["NPG_Rating_Pre"].to_numpy(), r["Assist_Rating_Pre"].to_numpy()
    covered = int(test["NPG_Rating_Pre"].notna().sum())
    for col in ["NPG_Rating_Pre", "Assist_Rating_Pre"]:
        test[col] = test[col].fillna(test.groupby("Pos")[col].transform("median"))
    test["conversion"] = test["Player"].map(vpc.conversion_lookup(season)).fillna(1.0)
    team, opp, opp_def, avg, home, away = team_context(test, fit, history)
    test["Team_Pred_NPG"], test["Opp_Pred_NPG"], test["Home"], test["Away"] = team, opp, home, away
    venue = test["Venue"].map(vpc.VENUE_ATT).to_numpy()
    short = (test["Minutes"] < 60).to_numpy()
    ratio = opp_def / avg
    test["NPG90_scaled"] = (test["NPG_Rating_Pre"] * test["conversion"] * ratio ** OPP_BETA["npg"] * venue
                            * np.where(short, SUB_UPLIFT["npg"], 1.0))
    test["Ast90_scaled"] = (test["Assist_Rating_Pre"] * ratio ** OPP_BETA["ast"] * venue
                            * np.where(short, SUB_UPLIFT["ast"], 1.0))
    shares = vpc.penalty_shares(fit, None)
    share = pd.Series(shares.reindex(pd.MultiIndex.from_arrays([test["Team"], test["Player"]])).to_numpy()).fillna(0)
    test["PenAttempts"] = test["Team_Pred_NPG"] * vpc.PEN_ATTEMPTS_PER_NPG * share.to_numpy()
    mu, pos_rate = vpc.defcon_rates(fit)
    test["Defcon_Mu90"] = mu.reindex(pd.MultiIndex.from_arrays([test["Player"], test["Pos"]])).to_numpy()
    test["Defcon_Mu90"] = test["Defcon_Mu90"].fillna(test["Pos"].map(pos_rate)).fillna(0)
    test.loc[test["Pos"].eq("G"), "Defcon_Mu90"] = 0.0
    table, pos = vpb.player_residuals(fit, "r_rem", pass_bands.K_REM)
    vpb.attach(test, table, pos, "rem")
    test["ResidualBPS_Mean"], test["ResidualBPS_Std"] = test["rem_m"], test["rem_s"]
    _, tilted, _ = vpb.band_distributions(test, so, const, season, test["Kickoff"].min())
    test[["PB_P0", "PB_P2", "PB_P4", "PB_P6"]] = tilted
    test["P1"] = 1.0
    test["P60"] = (test["Minutes"] >= 60).astype(float)
    return test, covered


def simulate_block(args):
    """Expected bonus and any-bonus probability of one block's rows for one arm."""
    test, arm, seed_base = args
    set_rules(int(test["Season"].iloc[0]))
    fn = sb.simulate_match if arm == "incumbent" else coherence.coherent_match
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    original, captured = sb.assign_bonus, []
    sb.assign_bonus = lambda m: captured.append(original(m)) or captured[-1]
    try:
        for i, (_, idx) in enumerate(sorted(test.groupby("Match_Key").indices.items())):
            sub = test.iloc[idx]
            first = sub.iloc[0]
            sb.rng = np.random.default_rng(seed_base + i)
            exp_gc_home = (first["Opp_Pred_NPG"] if first["Venue"] == "Home" else first["Team_Pred_NPG"]) * vpc.PEN_FACTOR
            exp_gc_away = (first["Team_Pred_NPG"] if first["Venue"] == "Home" else first["Opp_Pred_NPG"]) * vpc.PEN_FACTOR
            exp_bonus[idx] = fn(sub, exp_gc_home, exp_gc_away, first["Home"], vpc.N_DRAWS)[0]
            p_any[idx] = (captured.pop() > 0).mean(axis=1)
    finally:
        sb.assign_bonus = original
    return test.index.to_numpy(), exp_bonus, p_any


def run_arm(tests, arm):
    jobs = [(t, arm, vpc.SEED + 1000 * n) for n, t in enumerate(tests)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        return list(pool.map(simulate_block, jobs))


def main():
    frozen_before = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    rows, joined = load_rows()
    assert rows["Pos"].notna().all(), "position missing"
    so = vpb.load_sofascore()
    rows = attach_bands_and_residuals(rows, so)
    const = vpb.constants(so)
    history = pd.read_csv(calibration.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    log = pd.read_csv(RATING_LOG, encoding="utf-8-sig", low_memory=False)
    key = ["Season", "Player", "Team", "Opponent", "Venue"]
    ratings = log.drop_duplicates(key, keep="first")[key + ["NPG_Rating_Pre", "Assist_Rating_Pre"]]
    results = {"pms_join_rate": joined, "validity": {}, "era_rules": {s: era_rules(s) for s in SEASONS}}

    def stop(reason):
        results["stopped"] = f"validity check failed ({reason}); arms not compared"
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
        print(json.dumps(results, indent=2, default=float))

    # 1. inputs and rating coverage, before anything is simulated
    tests, covered = [], 0
    for season, b, fit, test in blocks(rows):
        t, c = build_block(fit, test, so, const, ratings, history)
        tests.append(t)
        covered += c
    n_rows = sum(len(t) for t in tests)
    results["validity"]["rating_coverage"] = covered / n_rows
    results["n_blocks"] = len(tests)
    print("blocks", len(tests), "rows", n_rows, "coverage", covered / n_rows, flush=True)
    if covered / n_rows < 0.95:
        return stop("rating coverage")

    # 2. the incumbent alone
    frame = pd.concat(tests)
    for idx, b, p in run_arm(tests, "incumbent"):
        frame.loc[idx, "pred_incumbent"], frame.loc[idx, "pany_incumbent"] = b, p
    bpm = float(frame.groupby("Match_Key")["pred_incumbent"].sum().mean())
    results["validity"]["incumbent_bonus_per_match"] = bpm
    print("incumbent bonus per match", bpm, flush=True)
    if not 6.1 <= bpm <= 6.6:
        return stop("incumbent bonus per match")

    # 3. only now the coherent arm, then any comparison
    for idx, b, p in run_arm([frame.loc[t.index] for t in tests], "coherent"):
        frame.loc[idx, "pred_coherent"], frame.loc[idx, "pany_coherent"] = b, p
    results["pooled"] = vpc.summarize(frame)
    results["pooled"]["bootstrap"] = vpc.bootstrap(frame)
    results["by_season"] = {int(s): vpc.summarize(g) for s, g in frame.groupby("Season")}
    results["season_2024_bootstrap"] = vpc.bootstrap(frame[frame["Season"] == 2024])
    results["before_2024_bootstrap"] = vpc.bootstrap(frame[frame["Season"] < 2024])
    inc, coh = results["pooled"]["incumbent"], results["pooled"]["coherent"]
    rule = {"i_mse_not_worse": coh["mse"] <= inc["mse"],
            "ii_brier_not_worse_1pct": coh["brier_any_bonus"] <= 1.01 * inc["brier_any_bonus"],
            "iii_no_position_abs_bias_up_003": all(
                abs(coh["position_bias"][p]) - abs(inc["position_bias"][p]) <= 0.03 for p in POSITIONS)}
    rule["adopt"] = all(rule.values())
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    keep = ["Season", "Round", "Match_Key", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "BPS", "Bonus",
            "NPG90_scaled", "Ast90_scaled", "PenAttempts", "Defcon_Mu90", "Team_Pred_NPG", "Opp_Pred_NPG",
            "pred_incumbent", "pany_incumbent", "pred_coherent", "pany_coherent"]
    frame[keep].to_csv(OUT_DIR / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "by_season"}, indent=2, default=float))
    for s, w in results["by_season"].items():
        print(s, {a: (round(w[a]["mse"], 5), round(w[a]["bonus_per_match"], 3), w[a]["position_bias"]) for a in ARMS},
              "actual share", w["actual_position_share"])


if __name__ == "__main__":
    main()
