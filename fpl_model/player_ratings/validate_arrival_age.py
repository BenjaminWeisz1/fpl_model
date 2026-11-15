"""
Arrivals' starting rating by age at arrival (CHANGELOG §240).
Protocol, frozen before scoring: data/validation/arrival_age_20261113/protocol.json.

    python validate_arrival_age.py    # -> data/validation/arrival_age_20261113/results.json

Same exact replay as §238 (validate_arrival_position.py): scaling an arrival's start by m moves his rating before
row t by s_t x start x (m - 1). Arm A: m = k for every arrival. Arm B: m = exp(a + b x (age - 25)), age clipped 18-34.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model.player_ratings import validate_arrival_position as v  # noqa: E402
from fpl_model.player_ratings import validate_position_alpha as vpa  # noqa: E402

OUT = v.ROOT / "data" / "validation" / "arrival_age_20261113"
K_GRID = np.round(np.arange(0.80, 1.3001, 0.01), 2)
A_GRID = np.round(np.arange(-0.20, 0.3001, 0.02), 2)
B_GRID = np.round(np.arange(-0.06, 0.0201, 0.005), 3)
CORRECTED_239 = {("Lucas Moura", 2017), ("Matt Ritchie", 2017), ("Adama Traoré", 2018), ("Alireza Jahanbakhsh", 2018),
                 ("Bernard", 2018), ("Hélder Costa", 2018), ("Max Meyer", 2018), ("Dennis Praet", 2019),
                 ("Mario Vrančić", 2019), ("Carlos Forbs", 2024), ("Chuba Akpom", 2026)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = v.load()
    a_by, rs = {}, {}
    for t in v.PAIRS:
        a_by[t], rs[t] = v.step_weights(log, t)
    resets = rs["npxG"] | rs["xAG"]
    segs, counts = v.arrival_segments(log, resets)
    ages = pd.read_csv(v.DATA / "new_player_priors.csv", encoding="utf-8-sig").set_index(["Player", "Season"])["Age"]
    ages = ages[~ages.index.duplicated()]
    res = {"arrivals": counts, "targets": {}}
    for t, (ac, pc, rc, _) in v.PAIRS.items():
        F = v.segment_features(log, segs, t, a_by[t])
        r = F["row"]
        w = log["w"].to_numpy()[r]
        f = log[f"f_{t}"].to_numpy()[r]
        rating = log[rc].to_numpy()[r]
        act = log[ac].to_numpy()[r] * w
        base_pred = rating * f * w
        base = (base_pred - act) ** 2
        # each segment's start = the rating on its first row
        key = list(zip(F["player"], F["season"]))
        start = {}
        for i, k in enumerate(key):
            start.setdefault(k, rating[i])
        st = np.array([start[k] for k in key])
        age = np.array([ages.get(k, np.nan) for k in key], dtype=float)
        has_age = ~np.isnan(age)
        agec = np.clip(np.nan_to_num(age, nan=25.0), 18, 34) - 25
        live = rating > 0
        tr = F["season"] <= 2022
        te = F["season"] >= 2023
        late = F["season"] >= 2025
        young = has_age & (age < 23)
        old = has_age & (age >= 23)
        not_fixed = np.array([k not in CORRECTED_239 for k in key])

        def pred(m):
            adj = np.where(live & has_age, rating + F["s"] * st * (m - 1.0), rating)
            return np.maximum(adj, 0.0) * f * w

        arms = {}
        # arm A
        best_k = min(K_GRID, key=lambda k: float(((pred(np.full(len(r), k)) - act) ** 2)[tr].sum()))
        arms["A_level"] = {"params": {"k": float(best_k)}, "pred": pred(np.full(len(r), best_k))}
        # arm B
        best, bl = None, None
        for a_ in A_GRID:
            for b_ in B_GRID:
                l = float(((pred(np.exp(a_ + b_ * agec)) - act) ** 2)[tr].sum())
                if bl is None or l < bl:
                    best, bl = (float(a_), float(b_)), l
        mB = np.exp(best[0] + best[1] * agec)
        arms["B_age"] = {"params": {"a": best[0], "b": best[1], "multiplier_at_21": float(np.exp(best[0] - 4 * best[1])),
                                    "multiplier_at_29": float(np.exp(best[0] + 4 * best[1]))}, "pred": pred(mB)}
        out = {"rows": {"train": int(tr.sum()), "test": int(te.sum()), "with_age": int(has_age.sum())},
               "calibration_test_production": float(base_pred[te].sum() / act[te].sum()), "arms": {}}
        for name, arm in arms.items():
            p = arm.pop("pred")
            c = (p - act) ** 2
            o = dict(arm)
            o["train_pct"] = vpa.rel(c, base, tr)
            o["test_pct"], o["test_ci"] = vpa.boot(base[te], c[te], F["player"][te])
            o["test_2025_26_pct"] = vpa.rel(c, base, te & late)
            o["test_under23_pct"] = vpa.rel(c, base, te & young)
            o["test_23plus_pct"] = vpa.rel(c, base, te & old)
            o["test_without_239_corrections_pct"] = vpa.rel(c, base, te & not_fixed)
            o["calibration_test"] = float(p[te].sum() / act[te].sum())
            o["calibration_test_under23"] = {"production": float(base_pred[te & young].sum() / act[te & young].sum()),
                                             "arm": float(p[te & young].sum() / act[te & young].sum())}
            o["calibration_test_23plus"] = {"production": float(base_pred[te & old].sum() / act[te & old].sum()),
                                            "arm": float(p[te & old].sum() / act[te & old].sum())}
            all_base = ((log[pc].to_numpy() - log[ac].to_numpy()) * log["w"].to_numpy()) ** 2
            all_c = all_base.copy()
            all_c[r] = c
            o["all_rows_test_pct"] = vpa.rel(all_c, all_base, log["Season"].to_numpy() >= 2023)
            cal_ok = abs(o["calibration_test"] - 1) <= abs(out["calibration_test_production"] - 1)
            o["passes_first_three"] = bool(o["test_pct"] < 0 and o["test_ci"][1] < 0 and o["test_2025_26_pct"] <= 0.1 and cal_ok)
            out["arms"][name] = o
        A, B = out["arms"]["A_level"], out["arms"]["B_age"]
        out["recommend_B"] = bool(B["passes_first_three"] and B["test_pct"] < A["test_pct"])
        out["report_A_as_level_gap"] = bool(not out["recommend_B"] and A["passes_first_three"])
        # Manzambi's start under arm B
        mk = ("Johan Manzambi", 2026)
        if mk in start and not np.isnan(ages.get(mk, np.nan)):
            ag = float(np.clip(ages[mk], 18, 34) - 25)
            out["manzambi"] = {"age": float(ages[mk]), "multiplier_B": float(np.exp(best[0] + best[1] * ag)),
                               "multiplier_A": float(best_k), "start": float(start[mk])}
        res["targets"][t] = out
        print(t, json.dumps(out, indent=1), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
