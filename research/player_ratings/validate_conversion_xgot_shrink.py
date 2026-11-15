"""CHANGELOG §164, after the first run: is xGOT's small gain just more shrinkage?

Compares α arms with the same formula at a larger c on the same rows. A disclosed follow-up, not part of the rule.
Run: python -u -X utf8 scripts/player_ratings/validate_conversion_xgot_shrink.py
"""
import json

from research.player_ratings import validate_conversion_xgot as V


def main():
    p = V.frame()
    log = V.pd.read_csv(V.OUT / "log.csv", encoding="utf-8-sig", low_memory=False).drop_duplicates(V.KEY + ["Player"])
    p = p.merge(log[V.KEY + ["Player", "pred_goals_90"]], on=V.KEY + ["Player"], how="left", validate="one_to_one")
    base = (p.Gc + 30) / (p.xGc + 30)
    p["f_ctrl"], p["r_ctrl"] = p.npxG * base, p.pred_goals_90 * p.Minutes / 90 * base
    for c in (45, 60, 100, 200):
        k = ((p.Gc + c) / (p.xGc + c))
        p[f"f_c{c}"], p[f"r_c{c}"] = p.npxG * k, p.pred_goals_90 * p.Minutes / 90 * k
    a = (p.Gc + .5 * (p.Xc - p.G2c) + 30) / (p.xGc + 30)
    p["f_a50"], p["r_a50"] = p.npxG * a, p.pred_goals_90 * p.Minutes / 90 * a
    for c in (60, 100):
        k = (p.Gc + .5 * (p.Xc - p.G2c) + c) / (p.xGc + c)
        p[f"f_a50c{c}"], p[f"r_a50c{c}"] = p.npxG * k, p.pred_goals_90 * p.Minutes / 90 * k
    t = p[p.Season.isin(V.TEST)].reset_index(drop=True)
    res = {}
    for arm in ("a50", "c45", "c60", "c100", "c200", "a50c60", "a50c100"):
        res[arm] = {"conversion": V.paired(t, "f_ctrl", "f_" + arm), "production_form": V.paired(t, "r_ctrl", "r_" + arm), "by_season": {int(s): V.paired(g, "f_ctrl", "f_" + arm)["mse_pct"] for s, g in t.groupby("Season")}}
        print(arm, {k: round(v, 3) for k, v in res[arm]["conversion"].items() if k in ("mse_pct", "lo", "hi")}, "production", {k: round(v, 3) for k, v in res[arm]["production_form"].items() if k in ("mse_pct", "lo", "hi")}, {k: round(v, 2) for k, v in res[arm]["by_season"].items()})
    res["a50_v_c100"] = V.paired(t, "f_c100", "f_a50")
    res["a50c100_v_c100"] = V.paired(t, "f_c100", "f_a50c100")
    print("a50 on c100", res["a50_v_c100"]); print("a50 at c100 on c100", res["a50c100_v_c100"])
    (V.OUT / "results_shrink.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
