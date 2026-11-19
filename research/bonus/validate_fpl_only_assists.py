"""
FPL-only assists: level correction (A1) and volume component (A2) against production (A0),
pre-registered in CHANGELOG §91.

Target: match FPL assists. Conventional part = pre-match assist rating x venue/opponent factor x
realized minutes / 90 (the current code's pre-match log). A1 scales it by a per-position factor
fitted on the other fold; A2 adds a Poisson model of FPL-only assists on position and the
player's pre-match per-90 rates of rebound-capable shots and penalties won (previous 38 Sofascore
appearances, all seasons, shrunk toward the fit fold's position mean with K = 10 x 90 minutes).
Folds: fit 2025 -> test 2026, and fit 2026 -> test 2025.
§93 adds A3: penalties won are credited only for the share of penalties a teammate takes
(pens-won rate x (1 - own share) x c), plus a GLM on position and rebound rate fitted on the
FPL-only assists that remain after removing identified penalty assists.

Run locally:
    python -m research.bonus.validate_fpl_only_assists
"""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fpl_model import paths
from research.bonus import validate_bonus_accounting as acc
from research.bonus import validate_bonus_calibration as calibration

OUT_DIR = calibration.ROOT / "data" / "validation" / "fpl_only_assists_20261028"
RATING_LOG = calibration.ROOT / "data" / "validation" / "prematch_bonus_rerun_20261028" / "player_ratings_prematch.csv"
LOCAL_RENAMES = {"James Mcatee": "James McAtee"}
WINDOW = 38
K = 10.0
N_BOOT, SEED = 5000, 42
POSITIONS = acc.POSITIONS
VOLUMES = ["rebound_rate", "pens_rate"]


def sofascore_history():
    so = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=calibration.DATA), encoding="utf-8-sig", low_memory=False)
    so = so[so["Appearance_Status"].eq("played")].copy()
    so["Kickoff"] = pd.to_datetime(so["Kickoff"], utc=True)
    so["Local_Player"] = so["Local_Player"].replace(LOCAL_RENAMES)
    num = lambda c: pd.to_numeric(so[f"Sofascore_{c}"], errors="coerce").fillna(0)
    so["mins"] = num("minutesPlayed")
    so["conv_assists"] = num("goalAssist")
    so["rebound_shots"] = ((num("onTargetScoringAttempt") - num("goals")).clip(lower=0)
                           + num("blockedScoringAttempt") + num("hitWoodwork"))
    so["pens_won"] = num("penaltyWon")
    so = so.sort_values(["Canonical_Player_ID", "Kickoff"])
    g = so.groupby("Canonical_Player_ID")
    for c in ["mins", "rebound_shots", "pens_won"]:
        # previous WINDOW appearances only: shift first so a match never sees itself
        so[f"prior_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=1).sum()).fillna(0)
    return so


def load():
    rows, _ = acc.load_rows()
    rows = rows[rows["Season"] >= 2025].copy()
    so = sofascore_history()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    s = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    rows = rows.merge(s[key + ["conv_assists", "pens_won", "prior_mins", "prior_rebound_shots", "prior_pens_won"]],
                      on=key, how="inner")
    log = pd.read_csv(RATING_LOG, encoding="utf-8-sig")
    log = log[log["Data_Source"].eq("FPL_API")].rename(columns={"MatchID": "PMS_MatchID"})
    lkey = ["Player", "Team", "Opponent", "Venue", "PMS_MatchID"]
    log = log.drop_duplicates(lkey, keep=False)
    rows = rows.merge(log[lkey + ["pred_ast_90", "weight"]], on=lkey, how="left")
    rows["conv_part"] = rows["pred_ast_90"] * rows["Minutes"] / 90
    coverage = float(rows["conv_part"].notna().mean())
    rows["conv_part"] = rows["conv_part"].fillna(rows.groupby("Pos")["conv_part"].transform("median"))
    rows["fpl_only"] = (rows["Assists"] - rows["conv_assists"]).clip(lower=0)
    return rows.reset_index(drop=True), coverage


def add_rates(frame, fit):
    out = frame.copy()
    for vol, col in (("rebound_rate", "rebound_shots"), ("pens_rate", "pens_won")):
        mu = (fit.groupby("Pos")[f"prior_{col}"].sum() / fit.groupby("Pos")["prior_mins"].sum() * 90)
        prior = out["Pos"].map(mu).fillna(0)
        out[vol] = (out[f"prior_{col}"] + prior * K) / (out["prior_mins"] / 90 + K)
    return out


def predictions(fit, test):
    fit, test = add_rates(fit, fit), add_rates(test, fit)
    m_pos = (fit.groupby("Pos")["Assists"].sum() / fit.groupby("Pos")["conv_part"].sum()).to_dict()
    m_pos["G"] = 1.0
    test["A0"] = test["conv_part"]
    test["A1"] = test["conv_part"] * test["Pos"].map(m_pos)
    outfield = fit[fit["Pos"] != "G"]
    design = lambda f: sm.add_constant(pd.concat([pd.get_dummies(f["Pos"], dtype=float).reindex(columns=["M", "F"], fill_value=0.0),
                                                  f[VOLUMES]], axis=1), has_constant="add")
    glm = sm.GLM(outfield["fpl_only"], design(outfield), family=sm.families.Poisson(),
                 offset=np.log(outfield["Minutes"].clip(lower=1) / 90)).fit()
    rate = np.exp(design(test) @ glm.params.to_numpy())
    test["A2"] = test["conv_part"] + np.where(test["Pos"].eq("G"), 0.0, rate * test["Minutes"] / 90)

    # A3 (§93): penalty part only for the share of penalties teammates take
    def identified(f):
        teammate_pk = f.groupby(["Match_Key", "Team"])["PK"].transform("sum") - f["PK"]
        return np.minimum(f["pens_won"], teammate_pk).clip(lower=0)
    att = fit.groupby(["Team", "Player"])["PKatt"].sum()
    share = (att / att.groupby(level=0).transform("sum").where(lambda t: t > 0)).fillna(0)
    lookup = lambda f: pd.Series(share.reindex(pd.MultiIndex.from_arrays([f["Team"], f["Player"]])).to_numpy(),
                                 index=f.index).fillna(0)
    fit["share"], test["share"] = lookup(fit), lookup(test)
    fit["pen_assist"], test["pen_assist"] = identified(fit), identified(test)
    c = float(fit["pen_assist"].sum() / (fit["pens_won"] * (1 - fit["share"])).sum())
    outfield = fit[fit["Pos"] != "G"]
    design3 = lambda f: sm.add_constant(pd.concat([pd.get_dummies(f["Pos"], dtype=float).reindex(columns=["M", "F"], fill_value=0.0),
                                                   f[["rebound_rate"]]], axis=1), has_constant="add")
    glm3 = sm.GLM((outfield["fpl_only"] - outfield["pen_assist"]).clip(lower=0), design3(outfield),
                  family=sm.families.Poisson(), offset=np.log(outfield["Minutes"].clip(lower=1) / 90)).fit()
    rate3 = np.exp(design3(test) @ glm3.params.to_numpy()) + test["pens_rate"] * (1 - test["share"]) * c
    test["A3"] = test["conv_part"] + np.where(test["Pos"].eq("G"), 0.0, rate3 * test["Minutes"] / 90)
    a3 = {"c": round(c, 4), "glm_coef": glm3.params.round(4).to_dict(), "glm_se": glm3.bse.round(4).to_dict()}
    return test, {k: round(float(v), 4) for k, v in m_pos.items()}, glm, a3


def compare(test, a, b):
    eb = (test[a] - test["Assists"]) ** 2
    ec = (test[b] - test["Assists"]) ** 2
    g = pd.DataFrame({"p": test["Player"], "eb": eb, "ec": ec}).groupby("p")[["eb", "ec"]].sum().to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(g), size=(N_BOOT, len(g)))
    s = g[idx].sum(axis=1)
    rel = 100 * (s[:, 1] / s[:, 0] - 1)
    return {"mse_" + a: float(eb.mean()), "mse_" + b: float(ec.mean()), "change_pct": float(100 * (ec.mean() / eb.mean() - 1)),
            "ci95_pct": [float(np.percentile(rel, 2.5)), float(np.percentile(rel, 97.5))], "p_worse": float((rel > 0).mean())}


def deviance(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return float(2 * np.mean(np.where(y > 0, y * np.log(y / mu), 0) - (y - mu)))


def calibration_table(test, by):
    return {str(k): {"rows": int(len(g)), "actual": int(g["Assists"].sum()),
                     **{a: round(float(g[a].sum() / g["Assists"].sum()), 3) for a in ("A0", "A1", "A2")}}
            for k, g in test.groupby(by, observed=True)}


def main():
    rows, coverage = load()
    out = {"rows": int(len(rows)), "conventional_part_coverage": round(coverage, 4), "folds": {}}
    tested = []
    for fit_s, test_s in ((2025, 2026), (2026, 2025)):
        test, m_pos, glm, a3 = predictions(rows[rows["Season"] == fit_s].copy(), rows[rows["Season"] == test_s].copy())
        test["taker"] = np.where(test["share"] >= 0.5, "taker", "non_taker")
        takers = {k: {"rows": int(len(g)), "fpl_only_actual": int(g["fpl_only"].sum()),
                      "A2": float((g["A2"] - g["A0"]).sum()), "A3": float((g["A3"] - g["A0"]).sum())}
                  for k, g in test.groupby("taker")}
        tested.append(takers)
        test["rebound_tercile"] = pd.qcut(test["rebound_rate"], 3, labels=["low", "mid", "high"])
        out["folds"][f"fit{fit_s}_test{test_s}"] = {
            "test_rows": int(len(test)), "test_fpl_assists": int(test["Assists"].sum()),
            "m_pos": m_pos, "glm_coef": glm.params.round(4).to_dict(), "glm_se": glm.bse.round(4).to_dict(),
            "A1_vs_A0": compare(test, "A0", "A1"), "A2_vs_A1": compare(test, "A1", "A2"), "A2_vs_A0": compare(test, "A0", "A2"),
            "A3": a3, "A3_vs_A2": compare(test, "A2", "A3"), "fpl_only_by_taker": takers,
            "poisson_deviance": {a: deviance(test["Assists"].to_numpy(), test[a].to_numpy()) for a in ("A0", "A1", "A2", "A3")},
            "calibration_by_position": calibration_table(test, "Pos"),
            "calibration_by_rebound_tercile": calibration_table(test, "rebound_tercile")}
    f = out["folds"]
    dec = {"A1_beats_A0_both_folds": all(v["A1_vs_A0"]["change_pct"] < 0 for v in f.values()),
           "A2_beats_A1_both_folds": all(v["A2_vs_A1"]["change_pct"] < 0 for v in f.values())}
    dec["adopt"] = ("A2" if dec["A1_beats_A0_both_folds"] and dec["A2_beats_A1_both_folds"]
                    else "A1" if dec["A1_beats_A0_both_folds"] else "A0")
    out["decision"] = dec
    pooled = {k: {a: sum(t[k][a] for t in tested if k in t) for a in ("fpl_only_actual", "A2", "A3")} for k in ("taker", "non_taker")}
    err = lambda k, a: abs(pooled[k][a] / pooled[k]["fpl_only_actual"] - 1)
    out["decision_93"] = {
        "A3_mse_within_0.2pct_of_A2_both_folds": all(v["A3_vs_A2"]["change_pct"] <= 0.2 for v in f.values()),
        "takers_calibration_error_shrinks": err("taker", "A3") < err("taker", "A2"),
        "pooled_fpl_only_by_taker": pooled}
    out["decision_93"]["adopt_A3"] = (out["decision_93"]["A3_mse_within_0.2pct_of_A2_both_folds"]
                                      and out["decision_93"]["takers_calibration_error_shrinks"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
