"""Step 1 of AGENDA §1: measure clean-sheet calibration OOT, under both definitions.
A) harness definition:   P = exp(-pred_NPG),          event = opp NPG == 0
B) production definition: P = exp(-1.081*pred_NPG),   event = opp Goals == 0
Reuses the harness's own run_sequential / frozen_predict with production params."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\bdwei\OneDrive\Desktop\Programz\fpl_model")
sys.path.insert(0, str(ROOT / "scripts" / "team_ratings"))
import evaluate_team_ratings_oot as H

PEN_MULT = 1 + (792 / 7734) * 0.79  # build_simulation_features.py:147-149

params = {"alpha": H.alpha, "beta": H.beta, "home_advantage": H.home_advantage,
          "money_weight": H.money_weight}
df, prior_map, _ = H.load_inputs()
for c in ("Home_Goals", "Away_Goals"):
    assert df[c].notna().all(), c

rows = {"frozen": [], "walk_forward": []}
for c in H.DEFAULT_CUTOFFS:
    train = df[df["Season"] < c]
    ave = float(pd.concat([train["Home_NPG"], train["Away_NPG"]]).mean())
    res, snaps = H.run_sequential(df[df["Season"] <= c], ave, params, prior_map, snapshot_seasons={c})
    val = res[res["Season"] == c]
    f_ph, f_pa = H.frozen_predict(val, snaps[c], prior_map, ave, params)
    for mode, ph, pa in (("frozen", f_ph, f_pa),
                         ("walk_forward", val["Home_Prediction"].values, val["Away_Prediction"].values)):
        # one row per team-match: lam = opponent's predicted NPG
        rows[mode].append(pd.DataFrame({
            "season": c, "lam": np.r_[pa, ph],
            "opp_npg": np.r_[val["Away_NPG"].values, val["Home_NPG"].values],
            "opp_goals": np.r_[val["Away_Goals"].values, val["Home_Goals"].values]}))

def ll(p, e):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-(e * np.log(p) + (1 - e) * np.log(1 - p))))

def bins(p, e, n=5):
    q = pd.qcut(p, n, labels=False, duplicates="drop")
    t = pd.DataFrame({"b": q, "p": p, "e": e}).groupby("b").agg(n=("p", "size"), pred=("p", "mean"), obs=("e", "mean"))
    t["gap_pp"] = (t["obs"] - t["pred"]) * 100
    # binomial SE of obs, in pp
    t["se_pp"] = np.sqrt(t["obs"] * (1 - t["obs"]) / t["n"]) * 100
    return t

for mode in ("frozen", "walk_forward"):
    d = pd.concat(rows[mode], ignore_index=True)
    print(f"\n================ {mode}  ({len(d)} team-matches, seasons {sorted(d.season.unique())}) ================")
    for name, p, e in (("A harness  exp(-NPG) vs NPG==0", np.exp(-d.lam), (d.opp_npg == 0).astype(float)),
                       ("B prod     exp(-1.081*NPG) vs Goals==0", np.exp(-PEN_MULT * d.lam), (d.opp_goals == 0).astype(float))):
        p, e = p.values, e.values
        print(f"\n  {name}: pred {p.mean()*100:.2f}%  obs {e.mean()*100:.2f}%  gap {(e.mean()-p.mean())*100:+.2f}pp  CS LL {ll(p,e):.4f}")
        print(bins(p, e).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n  per-season (B, production definition):")
    for s, g in d.groupby("season"):
        p = np.exp(-PEN_MULT * g.lam); e = (g.opp_goals == 0)
        print(f"    {s}: n={len(g)} pred {p.mean()*100:.1f}% obs {e.mean()*100:.1f}% gap {(e.mean()-p.mean())*100:+.1f}pp")
    # diagnostic only (not the fix): variance/mean of goals vs Poisson, and in-pool NB r profile
    lam_t = PEN_MULT * d.lam.values; y = d.opp_goals.values
    print(f"\n  dispersion diag: mean(y)={y.mean():.3f} mean(lam)={lam_t.mean():.3f} "
          f"mean((y-lam)^2)/mean(lam)={np.mean((y-lam_t)**2)/lam_t.mean():.3f} (Poisson ~1)")
