"""
Shadow sizing of team-goal coherence in the bonus simulator (diagnostic, CHANGELOG §71).
No production writes; outputs go to data/validation/team_goal_coherence_shadow_20261028/

Coherent variant: a team's goals conceded = the opposing players' simulated NP goals
+ their simulated scored penalties + Poisson(remainder), remainder mean =
max(Exp_GC - expected player NP goals - expected penalty goals, 0). Player marginals are
unchanged; only the dependence between scorers and clean sheets / goals conceded changes.
Noise floor: production with seed 43 vs seed 42. Uses the current part-2 inputs.

Run locally:
    python -m research.bonus.shadow_team_goal_coherence
"""
import hashlib

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.bonus import simulate_bonus as sb

S = paths.VALIDATION / "team_goal_coherence_shadow_20261028"
sb.COHERENCE = False  # §145: the harnesses importing this keep the independent draws as their incumbent
ORIGINAL = sb.simulate_match
DIAG = []


def coherent_match(match_players, exp_gc_home, exp_gc_away, home_team, n_draws):
    rng = sb.rng
    n = len(match_players)
    pos, team = match_players["Pos"].values, match_players["Team"].values
    p1, p60 = match_players["P1"].values, match_players["P60"].values
    npg90, ast90 = match_players["NPG90_scaled"].values, match_players["Ast90_scaled"].values
    pen_att, mu90 = match_players["PenAttempts"].values, match_players["Defcon_Mu90"].values
    resid_mean, resid_std = match_players["ResidualBPS_Mean"].values, match_players["ResidualBPS_Std"].values
    goal_bps_rate = np.array([sb.GOAL_BPS.get(p, 0) for p in pos])
    cs_bps_rate = np.array([sb.CS_BPS.get(p, 0) for p in pos])

    u = rng.random((n, n_draws))
    played_any, played_full = u < p1[:, None], u < p60[:, None]
    minutes_drawn = np.where(played_full, sb.AVG_MINUTES_FULL, np.where(played_any, sb.AVG_MINUTES_SHORT, 0))
    minutes_bps = np.where(played_full, 6, np.where(played_any, 3, 0))
    minute_frac = minutes_drawn / 90.0
    goals = rng.poisson(np.clip(npg90[:, None] * minute_frac, 0, None))
    assists = rng.poisson(np.clip(ast90[:, None] * minute_frac, 0, None))
    goal_bps, assist_bps = goals * goal_bps_rate[:, None], assists * sb.ASSIST_BPS
    pen_taken = rng.random((n, n_draws)) < pen_att[:, None]
    pen_scored = pen_taken & (rng.random((n, n_draws)) < 0.79)
    pen_missed = pen_taken & ~pen_scored
    pen_goal_bps, pen_miss_bps = pen_scored * sb.PEN_GOAL_BPS, pen_missed * sb.MISS_PEN_BPS
    r = np.clip(3 * mu90[:, None] * minute_frac, 1e-6, None)
    defcon_bps = rng.negative_binomial(r, 0.75) // 3

    # --- coherent team goals (the only change) ---
    home_mask = team == home_team
    e_frac = p60 + (p1 - p60) * sb.AVG_MINUTES_SHORT / 90
    e_player = npg90 * e_frac + pen_att * 0.79
    scored = goals + pen_scored
    away_exp, home_exp = e_player[~home_mask].sum(), e_player[home_mask].sum()
    rem_home = max(exp_gc_home - away_exp, 0)  # home concedes = away scores
    rem_away = max(exp_gc_away - home_exp, 0)
    DIAG.append({"home_ratio": away_exp / exp_gc_home, "away_ratio": home_exp / exp_gc_away})
    goals_conceded_home = scored[~home_mask].sum(axis=0) + rng.poisson(rem_home, n_draws)
    goals_conceded_away = scored[home_mask].sum(axis=0) + rng.poisson(rem_away, n_draws)
    team_goals_conceded = np.where(home_mask[:, None], goals_conceded_home[None, :], goals_conceded_away[None, :])

    gk_mask = pos == "G"
    gk_save_bps = np.zeros_like(minutes_bps, dtype=float)
    if np.any(gk_mask):
        gk_active = gk_mask[:, None] & played_full
        gk_saves = np.zeros_like(minutes_bps, dtype=float)
        gk_saves[gk_mask] = rng.poisson(np.clip(sb.AVG_GK_SAVES_PER_90 * minute_frac[gk_mask], 0, None))
        pen_save_draw = rng.random((n, n_draws)) < np.where(gk_active, sb.PEN_SAVE_RATE, 0.0)
        gk_save_bps = (np.where(gk_active, gk_saves * sb.SAVE_BPS, 0)
                       + np.where(gk_active & pen_save_draw, sb.PEN_SAVE_BPS, 0)) * sb.GK_SAVE_BPS_SCALE
    cs_bps = (played_full & (team_goals_conceded == 0)) * cs_bps_rate[:, None]
    gc_bps_rate = np.where(np.isin(pos, ["D", "G"]), sb.GC_BPS, 0)
    on_pitch_gc = np.where(played_full, team_goals_conceded, rng.binomial(team_goals_conceded, minute_frac))
    gc_bps = played_any * on_pitch_gc * gc_bps_rate[:, None]
    residual_bps = np.where(played_any, rng.normal(resid_mean[:, None], resid_std[:, None], size=(n, n_draws)), 0)
    band_probs = match_players[["PB_P0", "PB_P2", "PB_P4", "PB_P6"]].to_numpy(float)
    pass_band_bps = 0
    if np.any(band_probs[:, 0] < 1):
        cum = np.cumsum(band_probs, axis=1)
        idx = (rng.random((n, n_draws))[:, :, None] > cum[:, None, :]).sum(axis=2).clip(max=3)
        pass_band_bps = np.where(played_full, sb.PASS_BANDS[idx], 0)
    total = (minutes_bps + goal_bps + assist_bps + pen_goal_bps + pen_miss_bps + defcon_bps + cs_bps + gc_bps
             + residual_bps + gk_save_bps + pass_band_bps)
    if sb.INTEGER_BPS:
        total = np.rint(total)
    bonus = sb.assign_bonus(total)
    return bonus.mean(axis=1), goals.mean(axis=1), assists.mean(axis=1)


def main():
    S.mkdir(parents=True, exist_ok=True)
    prod_hash = hashlib.sha256(sb.OUT_FILE.read_bytes()).hexdigest()
    runs = {}
    for name, seed, fn in [("prod42", 42, ORIGINAL), ("prod43", 43, ORIGINAL), ("coherent42", 42, coherent_match)]:
        sb.rng = np.random.default_rng(seed)
        sb.simulate_match = fn
        sb.OUT_FILE = S / f"bonus_{name}.csv"
        sb.main()
        runs[name] = pd.read_csv(sb.OUT_FILE, encoding="utf-8-sig")
    sb.simulate_match = ORIGINAL
    print("\nproduction file unchanged:", prod_hash == hashlib.sha256(
        (paths.current_file("bonus_predictions.csv", root=sb._DATA)).read_bytes()).hexdigest())

    d = pd.DataFrame(DIAG)
    print("expected player goals / team expected goals (coherent run): median %.3f, IQR %.3f-%.3f, share >1 %.1f%%" % (
        np.median(d.values), *np.percentile(d.values, [25, 75]), 100 * (d.values > 1).mean()))

    base = runs["prod42"]
    for other in ["prod43", "coherent42"]:
        x = base[["Player", "Pos", "Team", "Round"]].copy()
        x["delta"] = runs[other]["Expected_Bonus"] - base["Expected_Bonus"]
        five = x.groupby(["Player", "Pos", "Team"])["delta"].sum()
        rel = x.loc[base["Expected_Bonus"] > 0.05]
        print(f"\n{other} vs prod42: five-GW per player mean |change| {five.abs().mean():.4f}, "
              f"p95 {five.abs().quantile(.95):.3f}, max {five.abs().max():.3f}, >0.1: {(five.abs() > .1).sum()}, "
              f">0.25: {(five.abs() > .25).sum()}; per player-GW (pred>0.05) mean |change| {rel['delta'].abs().mean():.4f}")
        print("  position totals change:", x.groupby("Pos")["delta"].sum().round(2).to_dict(),
              "total", round(x["delta"].sum(), 2))
        if other == "coherent42":
            print("  largest rises:\n" + five.sort_values().tail(8).round(3).to_string())
            print("  largest falls:\n" + five.sort_values().head(8).round(3).to_string())


if __name__ == "__main__":
    main()
