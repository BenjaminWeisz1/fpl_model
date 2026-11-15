"""AGENDA §2 feasibility (no outcomes used): GW1 odds -> implied goals per side,
compared with the model's own preseason prediction for the same fixture.
Implied goals: independent-Poisson (lam_h, lam_a) that best reproduce de-vigged
market-average 1X2 + over/under 2.5 (Pinnacle is absent from the 2026 file). Converted to NPG units by dividing
by CS_PEN_GOAL_MULT (odds price all goals; the ratings model predicts NPG)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "team_ratings"))
sys.path.insert(0, str(ROOT / "scripts" / "fetch"))
import evaluate_team_ratings_oot as H
from fetch_football_data import load_season, FIRST_SEASON, CURRENT_SEASON

G = np.arange(11)

def probs(lh, la):
    m = np.outer(poisson.pmf(G, lh), poisson.pmf(G, la))
    return np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), 1 - m[np.add.outer(G, G) <= 2].sum()

def implied(ph, pd_, pa, pov):
    f = lambda x: np.array(probs(*x)) - [ph, pd_, pa, pov]
    r = least_squares(f, [1.4, 1.1], bounds=([0.05, 0.05], [6, 6]))
    return r.x[0], r.x[1], float(np.abs(r.fun).max())

def devig(*odds):
    inv = 1 / np.asarray(odds, dtype=float)
    return inv / inv.sum()

rows = []
for s in range(FIRST_SEASON, CURRENT_SEASON + 1):
    fd = load_season(s).reset_index(drop=True)
    seen = set()
    pre = "BbAv" if "BbAvH" in fd.columns else "Avg"
    ov, un = f"{pre}>2.5", f"{pre}<2.5"
    for _, r in fd.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        if h in seen or a in seen:   # keep only fixtures that are BOTH teams' first match
            seen |= {h, a}; continue
        seen |= {h, a}
        ph, pd_, pa = devig(r[f"{pre}H"], r[f"{pre}D"], r[f"{pre}A"])
        pov = devig(r[ov], r[un])[0]
        lh, la, err = implied(ph, pd_, pa, pov)
        rows.append({"Season": s, "Home": h, "Away": a, "Odds_Lam_H": lh, "Odds_Lam_A": la, "fit_err": err})
od = pd.DataFrame(rows)

df, prior_map, _ = H.load_inputs()
params = {"alpha": H.alpha, "beta": H.beta, "home_advantage": H.home_advantage, "money_weight": H.money_weight}
ave = float(pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean())
res, _ = H.run_sequential(df, ave, params, prior_map)
m = od.merge(res[["Season", "Home", "Away", "Home_Prediction", "Away_Prediction"]], on=["Season", "Home", "Away"], how="left")
assert m["Home_Prediction"].notna().all()
m["Odds_NPG_H"] = m["Odds_Lam_H"] / H.CS_PEN_GOAL_MULT
m["Odds_NPG_A"] = m["Odds_Lam_A"] / H.CS_PEN_GOAL_MULT

teams = {s: set(g["Home"]) | set(g["Away"]) for s, g in df.groupby("Season")}
promoted = lambda s, t: s > FIRST_SEASON and t not in teams[s - 1]
m["Home_Promoted"] = [promoted(s, t) for s, t in zip(m.Season, m.Home)]
m["Away_Promoted"] = [promoted(s, t) for s, t in zip(m.Season, m.Away)]

print(f"GW1 fixtures with both teams' first match: {len(m)}  (max fit error {m.fit_err.max():.4f})")
print(f"promoted team-seasons covered: {int(m.Home_Promoted.sum() + m.Away_Promoted.sum())}")
print(f"\nmean NPG all GW1 sides: odds {np.r_[m.Odds_NPG_H, m.Odds_NPG_A].mean():.3f} | model {np.r_[m.Home_Prediction, m.Away_Prediction].mean():.3f}")
print(f"corr(odds, model) all GW1 sides: {np.corrcoef(np.r_[m.Odds_NPG_H, m.Odds_NPG_A], np.r_[m.Home_Prediction, m.Away_Prediction])[0,1]:.3f}")

# Per promoted team: goals FOR (attack side) and AGAINST (defence side), odds vs model
pt = []
for _, r in m.iterrows():
    for side, opp in (("H", "A"), ("A", "H")):
        if r[f"{'Home' if side == 'H' else 'Away'}_Promoted"]:
            pt.append({"Season": r.Season, "Team": r["Home" if side == "H" else "Away"],
                       "Opp": r["Away" if side == "H" else "Home"], "Venue": side,
                       "For_odds": r[f"Odds_NPG_{side}"], "For_model": r[f"{'Home' if side == 'H' else 'Away'}_Prediction"],
                       "Agst_odds": r[f"Odds_NPG_{opp}"], "Agst_model": r[f"{'Home' if opp == 'H' else 'Away'}_Prediction"]})
pt = pd.DataFrame(pt)
pt["For_diff"] = pt.For_odds - pt.For_model
pt["Agst_diff"] = pt.Agst_odds - pt.Agst_model
print(f"\nPromoted teams, GW1 (NPG units; diff = odds - model):")
print(pt.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
print(f"\nmean diff: goals for {pt.For_diff.mean():+.3f} (sd {pt.For_diff.std():.3f}) | "
      f"goals against {pt.Agst_diff.mean():+.3f} (sd {pt.Agst_diff.std():.3f})")
od.to_csv(ROOT / "_odds_gw1_implied.csv", index=False)
