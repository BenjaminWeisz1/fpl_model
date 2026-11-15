"""
Residual BPS training window (pre-registered in CHANGELOG §146).

Arms differ only in the rows behind the residual: W0 current season (production), W1 plus the previous
season, W2 plus the two previous seasons, earlier seasons centred on the current season-position mean.
Rows, era rules, blocks and features as validate_prematch_coherence_history.py; coherence on.

Run locally (uses the §144 rating log):
    python -m research.bonus.validate_residual_window
"""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from fpl_model.bonus import pass_bands
from fpl_model.bonus import simulate_bonus as sb
from research.bonus import validate_bonus_calibration as calibration
from research.bonus import validate_pass_bands as vpb
from research.bonus import validate_prematch_coherence as vpc
from research.bonus import validate_prematch_coherence_history as hist

OUT_DIR = calibration.ROOT / "data" / "validation" / "residual_window_20261030"
SEASONS = range(2019, 2025)
ARMS = ["W0", "W1", "W2"]
EXTRA = {"W0": 0, "W1": 1, "W2": 2}
MIN_POS_ROWS = 50
POSITIONS = vpc.POSITIONS


def fit_rows(rows, test, extra):
    """Residual rows for one block: the current season to date plus `extra` centred earlier seasons."""
    season, start = int(test["Season"].iloc[0]), test["Kickoff"].min()
    cur = rows[(rows["Season"] == season) & (rows["Kickoff"] < start)]
    pos_mean = cur.groupby("Pos")["r_rem"].mean()
    pos_n = cur.groupby("Pos").size()
    parts = [cur[["Player", "Pos", "r_rem", "MatchID_"]]]
    base = cur.empty
    seasons = [season - k for k in range(1, 1 + max(extra, 1 if base else 0))]
    for s in seasons:
        old = rows[rows["Season"] == s][["Player", "Pos", "r_rem", "MatchID_"]].copy()
        shift = (old["Pos"].map(pos_mean) - old["Pos"].map(old.groupby("Pos")["r_rem"].mean())).where(
            old["Pos"].map(pos_n).fillna(0) >= MIN_POS_ROWS, 0).fillna(0)
        old["r_rem"] = old["r_rem"] + shift
        parts.append(old)
    return pd.concat(parts, ignore_index=True)


def simulate_block(args):
    test, arm, seed_base = args
    sb.COHERENCE = True
    hist.set_rules(int(test["Season"].iloc[0]))
    exp_bonus, p_any = np.zeros(len(test)), np.zeros(len(test))
    original, captured = sb.assign_bonus, []
    sb.assign_bonus = lambda m: captured.append(original(m)) or captured[-1]
    try:
        for i, (_, idx) in enumerate(sorted(test.groupby("Match_Key").indices.items())):
            sub = test.iloc[idx]
            first = sub.iloc[0]
            sb.rng = np.random.default_rng(seed_base + i)
            home = first["Venue"] == "Home"
            gc_home = (first["Opp_Pred_NPG"] if home else first["Team_Pred_NPG"]) * vpc.PEN_FACTOR
            gc_away = (first["Team_Pred_NPG"] if home else first["Opp_Pred_NPG"]) * vpc.PEN_FACTOR
            exp_bonus[idx] = sb.simulate_match(sub, gc_home, gc_away, first["Home"], vpc.N_DRAWS)[0]
            p_any[idx] = (captured.pop() > 0).mean(axis=1)
    finally:
        sb.assign_bonus = original
    return test.index.to_numpy(), exp_bonus, p_any


def run_arm(frame, tests, rows, arm):
    jobs = []
    for n, t in enumerate(tests):
        sub = frame.loc[t.index].copy()
        table, pos = vpb.player_residuals(fit_rows(rows, sub, EXTRA[arm]), "r_rem", pass_bands.K_REM)
        vpb.attach(sub, table, pos, "rem")
        sub["ResidualBPS_Mean"], sub["ResidualBPS_Std"] = sub["rem_m"], sub["rem_s"]
        jobs.append((sub, arm, vpc.SEED + 1000 * n))
    with ProcessPoolExecutor(max_workers=hist.WORKERS) as pool:
        for idx, b, p in pool.map(simulate_block, jobs):
            frame.loc[idx, f"pred_{arm}"], frame.loc[idx, f"pany_{arm}"] = b, p


def summarize(f):
    per = f.groupby("Match_Key")
    out = {"n_player_matches": int(len(f)), "n_matches": int(per.ngroups)}
    for arm in ARMS:
        if f"pred_{arm}" not in f:
            continue
        err = f[f"pred_{arm}"] - f["Bonus"]
        out[arm] = {"mse": float((err ** 2).mean()), "bonus_per_match": float(per[f"pred_{arm}"].sum().mean()),
                    "brier_any_bonus": float(((f[f"pany_{arm}"] - (f["Bonus"] > 0)) ** 2).mean()),
                    "position_bias": err.groupby(f["Pos"]).mean().reindex(POSITIONS).round(4).to_dict(),
                    "position_mse": (err ** 2).groupby(f["Pos"]).mean().reindex(POSITIONS).round(4).to_dict()}
    return out


def bootstrap(f):
    keys = np.sort(f["Match_Key"].unique())
    n = f.groupby("Match_Key").size().reindex(keys).to_numpy()
    sq = {a: ((f[f"pred_{a}"] - f["Bonus"]) ** 2).groupby(f["Match_Key"]).sum().reindex(keys).to_numpy() for a in ARMS}
    rng = np.random.default_rng(vpc.SEED)
    w = np.stack([np.bincount(p, minlength=len(keys)) for p in rng.integers(0, len(keys), size=(2000, len(keys)))])
    mse = {a: (w @ sq[a]) / (w @ n) for a in ARMS}
    return {f"{a}_minus_W0_ci95": np.percentile(mse[a] - mse["W0"], [2.5, 97.5]).round(5).tolist() for a in ("W1", "W2")}


def main():
    frozen_before = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, _ = hist.load_rows()
    so = vpb.load_sofascore()
    rows = hist.attach_bands_and_residuals(rows, so)
    rows["MatchID_"] = rows["Match_Key"]
    const = vpb.constants(so)
    history = pd.read_csv(calibration.DATA / "team_ratings_history.csv", encoding="utf-8-sig", low_memory=False)
    log = pd.read_csv(hist.RATING_LOG, encoding="utf-8-sig", low_memory=False)
    key = ["Season", "Player", "Team", "Opponent", "Venue"]
    ratings = log.drop_duplicates(key, keep="first")[key + ["NPG_Rating_Pre", "Assist_Rating_Pre"]]
    tests = [hist.build_block(fit, test, so, const, ratings, history)[0]
             for season, b, fit, test in hist.blocks(rows) if season in SEASONS]
    frame = pd.concat(tests)
    print("blocks", len(tests), "rows", len(frame), flush=True)

    run_arm(frame, tests, rows, "W0")
    bpm = float(frame.groupby("Match_Key")["pred_W0"].sum().mean())
    results = {"validity": {"W0_bonus_per_match": bpm}, "n_blocks": len(tests)}
    print("W0 bonus per match", bpm, flush=True)
    if not 6.1 <= bpm <= 6.6:
        results["stopped"] = "W0 bonus per match outside [6.1, 6.6]; arms not compared"
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(results)
        return
    for arm in ("W1", "W2"):
        run_arm(frame, tests, rows, arm)
        print(arm, "done", flush=True)

    results["pooled"] = summarize(frame)
    results["pooled"]["bootstrap"] = bootstrap(frame)
    early = frame[frame["Round"] <= 12]
    results["GW1_12"], results["GW1_12"]["bootstrap"] = summarize(early), bootstrap(early)
    s24 = frame[frame["Season"] == 2024]
    results["season_2024"], results["season_2024"]["bootstrap"] = summarize(s24), bootstrap(s24)
    results["by_season"] = {int(s): {a: round(v[a]["mse"], 5) for a in ARMS} for s, g in frame.groupby("Season")
                            for v in [summarize(g)]}
    p = results["pooled"]
    best = min(("W1", "W2"), key=lambda a: p[a]["mse"])
    rule = {"best_arm": best, "i_mse_better_interval_below_zero": p[best]["mse"] < p["W0"]["mse"]
            and p["bootstrap"][f"{best}_minus_W0_ci95"][1] < 0,
            "ii_brier_not_worse_1pct": p[best]["brier_any_bonus"] <= 1.01 * p["W0"]["brier_any_bonus"],
            "iii_no_position_abs_bias_up_003": all(
                abs(p[best]["position_bias"][q]) - abs(p["W0"]["position_bias"][q]) <= 0.03 for q in POSITIONS)}
    rule["adopt"] = bool(rule["i_mse_better_interval_below_zero"] and rule["ii_brier_not_worse_1pct"]
                         and rule["iii_no_position_abs_bias_up_003"])
    results["decision_rule"] = rule
    results["frozen_gw8_sha256_unchanged"] = (
        frozen_before == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest())
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
