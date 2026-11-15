"""Item 3, AGENDA 2026-10-31 (CHANGELOG §151): how much would knowing the player's slot in advance be worth?

Oracle = the held-out row's actual Pos_* dummies. Realistic = the player's previous match's dummies, as a forecast sees.
Same fitted head for both. No production writes. Run: python -u -X utf8 -m research.defcon.evaluate_position_oracle
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from research.defcon import overnight_frames as H

YEARS = [2022, 2023, 2024, 2025, 2026]


def one(head, f, cols, year):
    tr, va = f[f.Season < year], f[f.Season == year]
    fit = H.fit_head(head, tr[cols].values, tr.Target.values, cols)
    pos = [c for c in cols if c.startswith("Pos_")]
    lag = va[cols].copy()
    lag[pos] = va[[f"lag_{c}" for c in pos]].values
    mode = va[cols].copy()
    mode[pos] = va[[f"mode_{c}" for c in pos]].values
    return head, va.index.values, H.predict(fit, va[cols].values), H.predict(fit, lag.values), H.predict(fit, mode.values)


def main():
    z = pd.read_pickle(H.OUT / "item1_frames.pkl")
    fr = H.frames()
    heads = {"D": fr["DEF_FEATURE_COLS"], "M": fr["MID_FEATURE_COLS"]}
    jobs, data = [], {}
    for h, cols in heads.items():
        cols = [c for c in cols if c != "OpponentAvgStat"] + ["OppFactor"]
        heads[h] = cols
        f = z["data"][h].sort_values(["Player", "Kickoff"]).reset_index(drop=True)
        pos = [c for c in cols if c.startswith("Pos_")]
        for c in pos:
            f[f"lag_{c}"] = f.groupby("Player")[c].shift(1).fillna(f[c])
        keys = [tuple(r) for r in f[pos].values.astype(int)]
        modal = np.zeros((len(f), len(pos)))
        for _, ix in f.groupby("Player", sort=False).indices.items():
            for n, j in enumerate(ix):
                seen = [keys[i] for i in ix[max(0, n - 5):n]] or [keys[j]]
                modal[j] = max(reversed(seen), key=seen.count)  # ties go to the most recent
        for i, c in enumerate(pos):
            f[f"mode_{c}"] = modal[:, i]
        f["changed"] = (f[pos].values != f[[f"lag_{c}" for c in pos]].values).any(axis=1)
        data[h] = f
        jobs += [(h, y) for y in YEARS]
    res = Parallel(n_jobs=4)(delayed(one)(h, data[h], heads[h], y) for h, y in jobs)
    out = {}
    for h, f in data.items():
        for hh, idx, a, b, c in res:
            if hh == h:
                f.loc[idx, "oracle"], f.loc[idx, "realistic"], f.loc[idx, "modal"] = a, b, c
        v = f[f.Season.isin(YEARS) & (f.IsForward.values == 0 if h == "M" else True)]
        out[h] = {"oracle": H.score(v.Target.values, v.oracle.values), "realistic": H.score(v.Target.values, v.realistic.values),
                  "gain": H.paired_bootstrap(v.Player.values, v.Target.values, v.realistic.values, v.oracle.values),
                  "share_rows_slot_changed": float(v.changed.mean())}
        out[h]["modal"] = H.score(v.Target.values, v.modal.values)
        out[h]["modal_vs_last"] = H.paired_bootstrap(v.Player.values, v.Target.values, v.realistic.values, v.modal.values)
        c = v[v.changed]
        out[h]["changed_rows"] = {"n": len(c), "oracle": H.score(c.Target.values, c.oracle.values)["ll"], "realistic": H.score(c.Target.values, c.realistic.values)["ll"]}
    (H.OUT / "item3_results.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
