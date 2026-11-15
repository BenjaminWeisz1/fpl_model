"""§150 guard: do the midfield arms also hold on what pays, FPL's real hits at realized minutes? Uses held-out probabilities."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from defcon_scaling import nb_sf, prob_at_minutes
HERE = Path(__file__).resolve().parent
KEY = ["Season", "Player", "Team", "Opponent", "Venue"]
p = pd.read_csv(ROOT / "data/current/player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
p = p[p.Data_Source == "FPL_API"][KEY + ["Defensive_Contribution"]].drop_duplicates(KEY)
d = pd.read_csv(HERE / "item2_predictions_M.csv", encoding="utf-8-sig")
o = pd.read_csv(HERE / "item1_predictions_M.csv", encoding="utf-8-sig")[KEY + ["p_control"]].rename(columns={"p_control": "p_rf_raw"})
d = d[d.Season >= 2025].drop_duplicates(KEY).merge(p, on=KEY).merge(o.drop_duplicates(KEY), on=KEY)
d["hit"] = (d.Defensive_Contribution >= 12).astype(float)
grid = np.linspace(.05, 40, 800)
for s, g in list(d.groupby("Season")) + [("all", d)]:
    for c in ("p_rf_raw", "p_control", "p_count", "p_logistic"):
        mu = np.interp(g[c].values, nb_sf(grid, 11), grid)
        q = np.clip(np.asarray(prob_at_minutes(mu, g.Minutes.values, 12), float), 1e-6, 1 - 1e-6)
        print(s, c, "LL", round(log_loss(g.hit, q), 5), "AUC", round(roc_auc_score(g.hit, q), 4), "mean", round(q.mean(), 4), "hit", round(g.hit.mean(), 4))
