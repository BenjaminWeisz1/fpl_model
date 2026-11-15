"""Is FPL Review's xMins on the recorded-minutes scale? (CHANGELOG §177)

Protocol: data/validation/xmins_scale_20261102/protocol.json. Reads archived forecasts and FPL event/N/live JSON
saved in the folder given as argv[1]. Changes no data file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "validation" / "xmins_scale_20261102"
SNAPS = {8: "GW08/20261022T170038_665091Z_partall", 9: "GW09/20261030T112240_265815Z_partall"}
# The arm as registered, kept so this test reproduces. 990 = 11 x 90; 1043 = mean team xMins total, GW9-13 of
# minutes.csv on 2026-11-02. Production computes the scale weekly: recorded_clock_scale in build_simulation_features.py.
SCALE = 990 / 1043
N_BOOT, SEED = 5000, 177


def load(api):
    boot = json.loads((api / "boot.json").read_text(encoding="utf-8"))
    short = {t["id"]: t["short_name"] for t in boot["teams"]}
    id_of = {(e["web_name"], short[e["team"]]): e["id"] for e in boot["elements"]}
    pc = pd.read_csv(ROOT / "data" / "current" / "players_current.csv", encoding="utf-8-sig")
    raw = {(w, t): (r, s) for w, t, r, s in zip(pc.Web_Name, pc.Team, pc.Raw_Web_Name, pc.Team_Short)}
    dl = pd.read_csv(ROOT / "data" / "current" / "gameweek_deadlines.csv", encoding="utf-8-sig")
    print(dl[dl.iloc[:, 0].isin([8, 9])].to_string(index=False))
    rows = []
    for gw, snap in SNAPS.items():
        s = pd.read_csv(ROOT / "data" / "snapshots" / "2026" / snap / "gameweek" / "simulation_features.csv", encoding="utf-8-sig")
        s = s[s.Round == gw].copy()
        live = {e["id"]: e["stats"] for e in json.loads((api / f"live{gw}.json").read_text(encoding="utf-8"))["elements"]}
        fix = json.loads((api / f"fix{gw}.json").read_text(encoding="utf-8"))
        played = {short[t] for f in fix if f["started"] for t in (f["team_h"], f["team_a"])}
        s["short"] = [raw.get((w, t), (w, None))[1] for w, t in zip(s.Web_Name, s.Team)]
        ids = [id_of.get(raw.get((w, t), (None, None))) for w, t in zip(s.Web_Name, s.Team)]
        s["element"] = ids
        print(f"GW{gw}: {len(s)} rows, {s.element.isna().sum()} without an FPL id; teams played {len(played)}")
        s = s[s.short.isin(played)]
        for c, k in [("mins", "minutes"), ("xg", "expected_goals"), ("xa", "expected_assists")]:
            s[c] = [float(live.get(int(i), {}).get(k, 0)) if pd.notna(i) else 0.0 for i in s.element]
        s["gw"] = gw
        rows.append(s[["gw", "Web_Name", "short", "xMins", "mins", "xg", "xa", "NPG90_scaled", "Ast90_scaled"]])
    return pd.concat(rows, ignore_index=True)


def main():
    d = load(Path(sys.argv[1]))
    arms = {"raw": d.xMins, "cap": d.xMins.clip(upper=90), "flat": d.xMins * SCALE}
    res = {"rows": len(d), "by_round": {}}
    for gw, g in d.groupby("gw"):
        r = {"rows": len(g), "actual": float(g.mins.sum()), "teams": int(g.short.nunique()),
             "actual_per_team": round(float(g.groupby("short").mins.sum().mean()), 1)}
        for a, v in arms.items():
            x = v[g.index]
            r[a] = dict(total=round(float(x.sum())), ratio=round(float(g.mins.sum() / x.sum()), 4),
                        mse=round(float(((g.mins - x) ** 2).mean()), 2))
        res["by_round"][int(gw)] = r
        print(gw, json.dumps(r))

    pl = (d.Web_Name + "|" + d.short).values
    u, inv = np.unique(pl, return_inverse=True)
    rng = np.random.default_rng(SEED)
    ix = rng.integers(0, len(u), size=(N_BOOT, len(u)))
    se = {a: (d.mins.values - v.values) ** 2 for a, v in arms.items()}

    def compare(a, b):
        ta, tb = np.bincount(inv, se[a]), np.bincount(inv, se[b])
        v = tb[ix].sum(1) / ta[ix].sum(1) - 1
        return round(float(se[b].sum() / se[a].sum() - 1) * 100, 3), [round(float(x) * 100, 3) for x in np.percentile(v, [2.5, 97.5])]

    res["flat_v_raw"], res["flat_v_cap"], res["cap_v_raw"] = compare("raw", "flat"), compare("cap", "flat"), compare("raw", "cap")
    print("flat v raw", res["flat_v_raw"], "flat v cap", res["flat_v_cap"], "cap v raw", res["cap_v_raw"])

    d["band"] = pd.cut(d.xMins, [-1, 0, 20, 45, 70, 85, 100], labels=["0", "1-20", "21-45", "46-70", "71-85", "86+"])
    b = d.groupby("band", observed=True).agg(rows=("mins", "size"), xMins=("xMins", "mean"), actual=("mins", "mean")).round(1)
    b["flat"] = (b.xMins * SCALE).round(1)
    print(b.to_string())
    res["bands"] = b.reset_index().to_dict("records")

    lvl = {}
    for gw, g in d.groupby("gw"):
        lvl[int(gw)] = {}
        for name, rate, act in [("xG", "NPG90_scaled", "xg"), ("xA", "Ast90_scaled", "xa")]:
            f = float((g[rate] * g.xMins / 90).sum())
            lvl[int(gw)][name] = dict(forecast_raw=round(f, 1), forecast_flat=round(f * SCALE, 1), actual=round(float(g[act].sum()), 1))
    res["level_not_gated"] = lvl
    print(json.dumps(lvl))

    r = res["by_round"]
    g1 = all(.98 <= r[k]["flat"]["ratio"] <= 1.02 for k in r)
    g2 = res["flat_v_raw"][1][1] < 0
    g3 = all(r[k]["flat"]["mse"] < r[k]["raw"]["mse"] for k in r)
    g4 = res["flat_v_cap"][0] <= 0
    res["gate"] = dict(ratio=bool(g1), interval=bool(g2), each_round=bool(g3), not_worse_than_cap=bool(g4),
                       verdict="PASS" if g1 and g2 and g3 and g4 else "NO CHANGE")
    print("GATE", res["gate"])
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
