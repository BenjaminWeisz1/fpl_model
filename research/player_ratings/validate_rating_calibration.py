"""Goal/assist calibration after the rating (CHANGELOG §156). Offline on a pre-match log; no production writes.
python -u -X utf8 scripts/player_ratings/validate_rating_calibration.py
"""
from fpl_model import paths
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

HERE = Path(__file__).resolve().parent
ROOT = paths.ROOT
LOG = paths.VALIDATION / "prematch_bonus_history_20261030" / "player_ratings_prematch.csv"
sys.argv = [sys.argv[0], str(LOG)]
from research.player_ratings import validate_rating_shrink as S

OUT = paths.VALIDATION / "rating_calibration_20261031"
OUT.mkdir(exist_ok=True)


class WeightedCIR:
    """Centred isotonic (defender_model.CenteredIsotonic) with sample weights."""

    def fit(self, s, y, w):
        f = IsotonicRegression(out_of_bounds="clip").fit(s, y, sample_weight=w).predict(s)
        lv = np.unique(f)
        self.x_ = np.array([np.average(s[f == v], weights=w[f == v]) for v in lv])
        self.y_ = lv
        return self

    def predict(self, s):
        return np.interp(s, self.x_, self.y_)


def fit(tr, t_):
    a, p, _ = S.TARGETS[t_]
    maps = {}
    for g, x in tr.groupby("group"):
        # The MSE's weights: squared minutes weight, collapsed on rounded scores to keep isotonic fast.
        rep = pd.DataFrame({"s": x[p].round(5), "y": x[a] * x["weight"] ** 2, "w": x["weight"] ** 2}).groupby("s").sum().reset_index()
        rep = rep[rep.w > 0]
        maps[g] = WeightedCIR().fit(rep.s.values, (rep.y / rep.w).values, rep.w.values)
    return maps


def apply(maps, d, t_):
    p = S.TARGETS[t_][1]
    out = d[p].copy().astype(float)
    for g, m in maps.items():
        k = d.group == g
        out[k] = m.predict(d.loc[k, p].values)
    return out


def main():
    d = S.frame()
    tr, te, te25 = d[d.Season <= 2022], d[d.Season >= 2023], d[d.Season >= 2025]
    res = {"rows": {"train": len(tr), "test": len(te), "test_2025plus": len(te25)}}
    for t_ in S.TARGETS:
        p = S.TARGETS[t_][1]
        maps = fit(tr, t_)
        r = {"knots": {g: len(m.x_) for g, m in maps.items()}}
        for name, s in (("train", tr), ("test", te), ("test_2025plus", te25)):
            c = apply(maps, s, t_)
            ch, ci = S.boot(s, S.sse(s, t_, s[p]), S.sse(s, t_, c))
            r[name] = {"change": ch, "ci95": ci, "deciles_base": S.deciles(s, t_, s[p]), "deciles_cand": S.deciles(s, t_, c)}
            print(t_, name, round(ch, 5), [round(x, 5) for x in ci], r[name]["deciles_base"], r[name]["deciles_cand"], flush=True)
        res[t_] = r
    ok = all(res[t]["test"]["change"] < 0 and res[t]["test_2025plus"]["change"] <= 0 and all(
        abs(res[t]["test"]["deciles_cand"][k] - 1) < abs(res[t]["test"]["deciles_base"][k] - 1) for k in (1, 10)) for t in S.TARGETS)
    res["adopt"] = bool(ok)
    print("adopt", ok, res["rows"])
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
