"""CHANGELOG §165: the §162 fixture save rate inside the bonus simulation, against the pooled rate.

Rows, blocks, era rules and features as validate_prematch_coherence_history.py; coherence on. Test only.
Run: python -u -X utf8 scripts/bonus/validate_keeper_save_bonus.py
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "assemble")]
from fpl_model.bonus import build_residual_bps_baseline as residual  # noqa: E402
from fpl_model.bonus import pass_bands  # noqa: E402,F401
from fpl_model.bonus import simulate_bonus as sb  # noqa: E402
from research.bonus import validate_bonus_accounting as acc  # noqa: E402
from research.bonus import validate_bonus_calibration as calibration  # noqa: E402
from research.bonus import validate_pass_bands as vpb  # noqa: E402
from research.bonus import validate_prematch_coherence as vpc  # noqa: E402
from research.bonus import validate_prematch_coherence_history as hist  # noqa: E402
from research.assemble.validate_team_coherence import NAMES  # noqa: E402

OUT = calibration.ROOT / "data" / "validation" / "keeper_save_bonus_20261101"
FITS = calibration.ROOT / "data" / "validation" / "keeper_saves_20261101" / "results.json"
SEASONS = range(2019, 2025)
ARMS = ["control", "fixture"]
POSITIONS = vpc.POSITIONS
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]


def repair_names(rows):
    """§160's unrated names: the stats file's spelling, then the penalties and actions lost at the join."""
    mapped = rows["Player"].isin(NAMES)
    rows.loc[mapped, "Player"] = rows.loc[mapped, "Player"].map(NAMES)
    cols = ["PK", "PKatt", "Tkl", "CBI", "Recov"]
    pms = pd.read_csv(calibration.DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False, usecols=KEY + cols).drop_duplicates(KEY)
    m = rows.loc[mapped, KEY].merge(pms, on=KEY, how="left")
    for c in cols:
        rows.loc[mapped, c] = m[c].fillna(rows.loc[mapped, c].reset_index(drop=True)).to_numpy()
    rows["NPG"] = (rows["goals_scored"] - rows["PK"]).clip(lower=0)
    known = set(pms["Player"])
    print("names repaired on rows", int(mapped.sum()), "still unrated rows", int((~rows["Player"].isin(known)).sum()), flush=True)
    return rows, known


def fixture_multiplier(test, history, betas):
    season = int(test["Season"].iloc[0])
    beta = betas[season]
    earlier = history[history["Season"] < season]
    ref = float((pd.concat([earlier["Home_Prediction"], earlier["Away_Prediction"]]) ** beta).mean())
    return test["Opp_Pred_NPG"].to_numpy() ** beta / ref


def simulate_block(args):
    test, seed_base, rules_season = args
    sb.COHERENCE = True
    hist.set_rules(rules_season)
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    original, captured = sb.assign_bonus, []
    sb.assign_bonus = lambda m: captured.append(original(m)) or captured[-1]
    try:
        for i, (_, idx) in enumerate(sorted(test.groupby("Match_Key").indices.items())):
            sub = test.iloc[idx]
            first = sub.iloc[0]
            sb.rng = np.random.default_rng(seed_base + i)
            sb.SAVE_RNG = np.random.default_rng(seed_base + i + 500_000)
            home = first["Venue"] == "Home"
            gc_home = (first["Opp_Pred_NPG"] if home else first["Team_Pred_NPG"]) * vpc.PEN_FACTOR
            gc_away = (first["Team_Pred_NPG"] if home else first["Opp_Pred_NPG"]) * vpc.PEN_FACTOR
            home_team = first["Team"] if home else first["Opponent"]
            exp_bonus[idx] = sb.simulate_match(sub, gc_home, gc_away, home_team, vpc.N_DRAWS)[0]
            p_any[idx] = (captured.pop() > 0).mean(axis=1)
    finally:
        sb.assign_bonus = original
    return test.index.to_numpy(), exp_bonus, p_any


def run_arm(frame, tests, arm):
    jobs = []
    for n, idx in enumerate(tests):
        sub = frame.loc[idx]
        if arm == "control":
            sub = sub.drop(columns="GK_Save_Mult")
        jobs.append((sub, vpc.SEED + 1000 * n, int(sub["Season"].iloc[0])))
    with ProcessPoolExecutor(max_workers=hist.WORKERS) as pool:
        for idx, b, p in pool.map(simulate_block, jobs):
            frame.loc[idx, f"pred_{arm}"], frame.loc[idx, f"pany_{arm}"] = b, p


def boot(f):
    """Paired match bootstrap of the MSE difference, all rows and keepers."""
    keys = np.sort(f["Match_Key"].unique())
    w = np.random.default_rng(vpc.SEED).multinomial(len(keys), np.ones(len(keys)) / len(keys), size=2000)
    out = {}
    for label, g in (("all", f), ("G", f[f["Pos"].eq("G")])):
        n = g.groupby("Match_Key").size().reindex(keys, fill_value=0).to_numpy()
        s = {a: ((g[f"pred_{a}"] - g["Bonus"]) ** 2).groupby(g["Match_Key"]).sum().reindex(keys, fill_value=0).to_numpy() for a in ARMS}
        m = {a: (w @ s[a]) / (w @ n) for a in ARMS}
        out[label] = {"pct_ci95": np.percentile(100 * (m["fixture"] / m["control"] - 1), [2.5, 97.5]).round(3).tolist()}
    return out


def summarize(f):
    per = f.groupby("Match_Key")
    g = f[f["Pos"].eq("G")].copy()
    s = np.log(g["Team_Pred_NPG"] / g["Opp_Pred_NPG"])
    g["Q"] = pd.qcut(s, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out = {"rows": int(len(f)), "matches": int(per.ngroups), "actual_bonus_per_match": float(per["Bonus"].sum().mean()),
           "actual_share": (f.groupby("Pos")["Bonus"].sum() / f["Bonus"].sum()).reindex(POSITIONS).round(4).to_dict()}
    for arm in ARMS:
        err = f[f"pred_{arm}"] - f["Bonus"]
        b = (g["Bonus"] - g[f"pred_{arm}"]).groupby(g["Q"], observed=True).mean()
        out[arm] = {"mse": float((err ** 2).mean()), "bonus_per_match": float(per[f"pred_{arm}"].sum().mean()),
                    "brier": float(((f[f"pany_{arm}"] - (f["Bonus"] > 0)) ** 2).mean()),
                    "position_bias": err.groupby(f["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(f["Pos"]).mean().reindex(POSITIONS).round(5).to_dict(),
                    "share": (f.groupby("Pos")[f"pred_{arm}"].sum() / f[f"pred_{arm}"].sum()).reindex(POSITIONS).round(4).to_dict(),
                    "keeper_actual_minus_pred_by_strength": b.round(4).to_dict(), "keeper_gap": float(b["Q4"] - b["Q1"])}
    out["bootstrap"] = boot(f)
    return out


def later_windows(history, betas):
    hist.set_rules(2026)
    rows, _ = acc.load_rows()
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
    const = vpb.constants(so)
    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    k_rem = vpb.split_half_k(wins["A_2025"][0], "r_rem")
    parts = []
    for name, (fit, test, _) in wins.items():
        t, _ = vpc.build_features(name, fit, test, rows, so, const, k_rem, history)
        t["GK_Save_Mult"] = fixture_multiplier(t, history, betas)
        parts.append(t.assign(Window=name))
    f = pd.concat(parts, ignore_index=True)
    idx = [g.index for _, g in f.groupby("Window")]
    for arm in ARMS:
        for n, ix in enumerate(idx):
            sub = f.loc[ix].drop(columns="GK_Save_Mult") if arm == "control" else f.loc[ix]
            i, b, p = simulate_block((sub, vpc.SEED + 7000 + 1000 * n, 2026))
            f.loc[i, f"pred_{arm}"], f.loc[i, f"pany_{arm}"] = b, p
    return f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    betas = {int(r["Season"]): r["beta"] for r in json.loads(FITS.read_text(encoding="utf-8"))["fits"]}
    history = pd.read_csv(calibration.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    rows, _ = hist.load_rows()
    rows, known = repair_names(rows)
    so = vpb.load_sofascore()
    rows = hist.attach_bands_and_residuals(rows, so)
    const = vpb.constants(so)
    log = pd.read_csv(hist.RATING_LOG, encoding="utf-8-sig", low_memory=False)
    ratings = log.drop_duplicates(KEY, keep="first")[KEY + ["NPG_Rating_Pre", "Assist_Rating_Pre"]]
    tests = []
    for season, b, fit, test in hist.blocks(rows):
        if season in SEASONS:
            t, _ = hist.build_block(fit, test, so, const, ratings, history)
            t["GK_Save_Mult"] = fixture_multiplier(t, history, betas)
            tests.append(t)
    frame = pd.concat(tests)
    print("blocks", len(tests), "rows", len(frame), "mult mean", round(float(frame.loc[frame.Pos.eq("G"), "GK_Save_Mult"].mean()), 3), flush=True)
    idx = [t.index for t in tests]
    run_arm(frame, idx, "control")
    bpm = float(frame.groupby("Match_Key")["pred_control"].sum().mean())
    print("control bonus per match", bpm, flush=True)
    results = {"validity": {"control_bonus_per_match": bpm}}
    if not 6.1 <= bpm <= 6.6:
        results["stopped"] = "control bonus per match outside [6.1, 6.6]; arms not compared"
        (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(results)
        return
    run_arm(frame, idx, "fixture")
    scored = frame[frame["Player"].isin(known)]
    print("scored rows", len(scored), "of", len(frame), flush=True)
    results["pooled"] = p = summarize(scored)
    results["by_season"] = {int(s): summarize(g) for s, g in scored.groupby("Season")}
    later = later_windows(history, betas)
    results["later"] = l = summarize(later)
    c, x = p["control"], p["fixture"]
    s24 = results["by_season"][2024]
    results["decision_rule"] = rule = {
        "i_keeper_mse": x["position_mse"]["G"] < c["position_mse"]["G"] and p["bootstrap"]["G"]["pct_ci95"][1] < 0,
        "ii_all_mse": x["mse"] <= c["mse"],
        "iii_brier": x["brier"] <= 1.01 * c["brier"],
        "iv_gap": abs(x["keeper_gap"]) < abs(c["keeper_gap"]),
        "v_bias": all(abs(x["position_bias"][q]) - abs(c["position_bias"][q]) <= (.01 if q == "G" else .03) for q in POSITIONS),
        "vi_2024_and_later": s24["fixture"]["position_mse"]["G"] <= s24["control"]["position_mse"]["G"]
        and l["fixture"]["position_mse"]["G"] <= l["control"]["position_mse"]["G"]}
    rule["adopt"] = all(rule.values())
    results["frozen_gw8_sha256_unchanged"] = frozen == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    frame.reset_index(drop=True)[["Season", "Round", "Match_Key", "Player", "Pos", "Team", "Opponent", "Venue", "Minutes", "Bonus", "Team_Pred_NPG", "Opp_Pred_NPG", "GK_Save_Mult",
                                  "pred_control", "pred_fixture", "pany_control", "pany_fixture"]].to_csv(OUT / "hindcast_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
