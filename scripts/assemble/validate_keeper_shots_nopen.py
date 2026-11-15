"""CHANGELOG §163, added at the user's request: arm K on penalty-free goals prevented.

Same rule and fixed shrinkage as arm K. No per-kick xGOT is held, so penalties are removed by estimate: a kick has
xGOT about 0.86 when scored (58 single-shot takers) and an assumed 0.80 when saved (7 misses and saves averaged 0.70).
NP GP = GP - 0.80 x saved + 0.14 x penalties not saved. Constants fixed before scoring. No production writes.
Run: python -u -X utf8 scripts/assemble/validate_keeper_shots_nopen.py
"""
import json

import numpy as np
import pandas as pd

import validate_keeper_saves as K
import validate_keeper_shots as S

SAVED_KICK_XGOT, UNSAVED_KICK_SHORTFALL = .80, .14


def table():
    e = S.keeper_table()
    p = pd.read_csv(K.DATA / "sofascore_player_extra.csv.gz", low_memory=False, usecols=["Event_ID_Sofascore", "Player_ID_Sofascore", "penaltyFaced", "penaltySave"])
    e = e.merge(p, on=["Event_ID_Sofascore", "Player_ID_Sofascore"], how="left", validate="one_to_one")
    faced, saved = e.penaltyFaced.fillna(0), e.penaltySave.fillna(0)
    e["goalsPrevented"] = e.goalsPrevented - SAVED_KICK_XGOT * saved + UNSAVED_KICK_SHORTFALL * (faced - saved)
    return e


def main():
    x = K.frame()
    e = table()
    kp, k = S.keeper_features(e, x)
    x = x.merge(kp, on=K.KEY, how="left", validate="many_to_one")
    x = x[x.Season >= 2018].reset_index(drop=True)
    x["lO"] = np.log(x.O)
    x["KeeperGP"] = x.KeeperGP.fillna(0.0)
    t, c = S.run(x[x.Season >= 2022], [2023, 2024, 2025, 2026], ["ctrl", "k"], first=2022)
    p = K.paired(t, "ctrl", "k")
    by = {int(s): float(((g.Pts - g.pts_k) ** 2).mean() - ((g.Pts - g.pts_ctrl) ** 2).mean()) for s, g in t.groupby("Season")}
    d = S.keeper_diagnostics(x, e, k)
    res = {"rows": len(t), "pooled": p, "by_season": by, "passes": bool(p["mse_hi"] < 0 and p["ll_lo"] > 0 and sum(by[s] < 0 for s in (2023, 2024, 2025)) >= 2),
           "coef": c[c.Arm == "k"][["Season", "KeeperGP", "KeeperGP_se"]].to_dict("records"), "goals_conceded_in_sample": d["goals_conceded_in_sample"], "gp_persistence": d["gp_persistence"]}
    (S.OUT / "results_nopen.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
