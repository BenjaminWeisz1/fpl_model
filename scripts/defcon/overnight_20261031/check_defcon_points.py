"""§152 follow-up: production-path defcon points against FPL's real Defensive_Contribution, 2025-26, all rows.
p = held-out probability, re-scaled to realized minutes as production does."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from defcon_scaling import nb_sf, prob_at_minutes

HERE = Path(__file__).resolve().parent
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]
p = pd.read_csv(ROOT / "data/current/player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
p = p[p.Data_Source == "FPL_API"][KEY + ["Minutes", "Defensive_Contribution", "CBI", "Tkl", "Recov"]].drop_duplicates(KEY)
print("Defensive_Contribution null share", p.Defensive_Contribution.isna().mean())
pc = pd.read_csv(ROOT / "data/current/players_current.csv", encoding="utf-8-sig")
rows = []
for h, col, thr in (("D", "p_control", 10), ("M", "p_count", 12), ("M", "p_logistic", 12), ("M", "p_control", 12), ("F", "p_control", 12)):
    d = pd.read_csv(HERE / f"item2_predictions_{h}.csv", encoding="utf-8-sig")
    d = d[d.Season >= 2025].drop_duplicates(KEY).merge(p.drop(columns="Minutes"), on=KEY, how="inner", validate="one_to_one")
    grid = np.linspace(.05, 40, 800)
    mu = np.interp(d[col].values, nb_sf(grid, thr - 1), grid)
    d["scaled"] = np.asarray(prob_at_minutes(mu, d.Minutes.values, thr), dtype=float)
    d["hit"] = (d.Defensive_Contribution >= thr).astype(float)
    # The count FPL actually uses, and a check that the column equals its parts at this level.
    d["band"] = pd.cut(d.Minutes, [0, 30, 59, 75, 89, 200], labels=["1-30", "31-59", "60-75", "76-89", "90+"])
    g = d.groupby("band", observed=True).agg(n=("hit", "size"), raw_p=(col, "mean"), scaled_p=("scaled", "mean"), target=("Target", "mean"), fpl_hit=("hit", "mean"))
    print(f"\n{h} {col}: all rows n={len(d)} raw_p {d[col].mean():.4f} scaled {d.scaled.mean():.4f} model target {d.Target.mean():.4f} FPL hit {d.hit.mean():.4f}")
    print(g.round(4).to_string())
    b = pd.qcut(d[d.Minutes >= 60].scaled, 6, duplicates="drop")
    print(d[d.Minutes >= 60].groupby(b, observed=True).agg(n=("hit", "size"), scaled=("scaled", "mean"), hit=("hit", "mean")).round(3).T.to_string())
