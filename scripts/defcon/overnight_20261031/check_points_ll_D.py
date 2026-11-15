"""§149 guard for defenders and forwards: OppFactor v control on FPL's real hits at realized minutes."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from defcon_scaling import nb_sf, prob_at_minutes
HERE = Path(__file__).resolve().parent
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]
p = pd.read_csv(ROOT / "data/current/player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
p = p[p.Data_Source == "FPL_API"][KEY + ["Defensive_Contribution"]].drop_duplicates(KEY)
d = pd.read_csv(HERE / "item1_predictions_D.csv", encoding="utf-8-sig")
d = d[d.Season >= 2025].drop_duplicates(KEY).merge(p, on=KEY)
d["hit"] = (d.Defensive_Contribution >= 10).astype(float)
grid = np.linspace(.05, 40, 800)
for s, g in list(d.groupby("Season")) + [("all", d)]:
    for c in ("p_control", "p_opp_factor"):
        mu = np.interp(g[c].values, nb_sf(grid, 9), grid)
        q = np.clip(np.asarray(prob_at_minutes(mu, g.Minutes.values, 10), float), 1e-6, 1 - 1e-6)
        print(s, c, "LL", round(log_loss(g.hit, q), 5), "mean", round(q.mean(), 4), "hit", round(g.hit.mean(), 4))
