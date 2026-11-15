"""
Youth prior: fitted multiplier on the position average (CHANGELOG §113).

    python validate_youth_prior.py      # static rolling out-of-time test, position-group arm, loop check

Reads the §112 K = 16 pre-match log (data/validation/prior_k_20261029/log_k16.csv); youth rows there do not
depend on the new-player priors file. Writes data/validation/youth_prior_20261029/results.json.
"""
from fpl_model import paths
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings.validate_prior_k import entrants, boot_diff, OOT_PRIORS  # noqa: E402

ROOT = paths.ROOT
DATA = paths.CURRENT
LOG = paths.VALIDATION / "prior_k_20261029" / "log_k16.csv"
OUT = paths.VALIDATION / "youth_prior_20261029"
TARGETS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"),
           "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}
GROUP = {"CB": "DEF", "FB": "DEF", "WB": "DEF", "DM": "MID", "CM": "MID", "WM": "MID",
         "AM": "ATT", "W": "ATT", "FW": "ATT"}


def youth_rows(log):
    ent = entrants(log, min_season=2017)
    t = pd.read_csv(DATA / "transfers.csv", encoding="utf-8-sig")
    ent = ent[ent["Type"] == "youth"].copy()
    pos = {}
    for p, s0 in zip(ent["Player"], ent["Entry"]):
        r = t[(t["Player"] == p) & (t["Season"] <= s0)].sort_values("Season").iloc[-1]
        pos[p] = r["Pos"]
    ent["Pos"] = ent["Player"].map(pos)
    ent["Group"] = ent["Pos"].str.split(",").str[0].map(GROUP)
    d = log.reset_index().merge(ent, on="Player")
    d = d[d["Season"] == d["Entry"]].sort_values("index")       # first-season rows, log order
    out = {}
    for t_, (a, p, r) in TARGETS.items():
        first = d.groupby("Player")[r].transform("first")         # the starting (prior) rating
        coef = np.where((t_ == "npxG") & (d["Pos"] == "FW"), 0.7, 0.75)
        base = first / coef                                         # position prior x team factor
        factor = d[p] / d[r]                                        # venue x opponent x sub uplift
        f = base * factor * d["weight"]
        y = d[a] * d["weight"]
        ok = (d[r] > 0).to_numpy()
        out[t_] = pd.DataFrame({"Player": d["Player"], "Entry": d["Entry"], "Group": d["Group"],
                                "f": f, "inc": coef * f, "y": y})[ok]
    return out, ent


def fit_c(df):
    return float((df["y"] * df["f"]).sum() / (df["f"] ** 2).sum())


def rolling(df, by_group):
    pred = pd.Series(np.nan, index=df.index)
    cs = {}
    for s in range(2019, 2027):
        tr, te = df[df["Entry"] < s], df["Entry"] == s
        if by_group:
            for g in df["Group"].unique():
                m = te & (df["Group"] == g)
                pred[m] = fit_c(tr[tr["Group"] == g]) * df.loc[m, "f"]
        else:
            cs[s] = fit_c(tr)
            pred[te] = cs[s] * df.loc[te, "f"]
    return pred, cs


def loop_check(c_npg, c_ast, ent):
    log_path = OUT / "loop_youth.csv"
    env = dict(os.environ, GOAL_MODEL_PREMATCH_LOG=str(log_path), GOAL_MODEL_NEW_PRIORS_FILE=str(OOT_PRIORS),
               GOAL_MODEL_YOUTH_NPG=str(c_npg), GOAL_MODEL_YOUTH_AST=str(c_ast), PYTHONHASHSEED="0")
    with open(OUT / "loop_youth.run.log", "w", encoding="utf-8") as fh:
        subprocess.run([sys.executable, "-u", "-X", "utf8", "-m", "fpl_model.player_ratings.build_player_ratings"], env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)
    base, cand = pd.read_csv(LOG, encoding="utf-8-sig"), pd.read_csv(log_path, encoding="utf-8-sig")
    assert (base["Player"].to_numpy() == cand["Player"].to_numpy()).all()
    m = base[["Player", "Season"]].merge(ent[["Player", "Entry"]], on="Player", how="left")
    mask = (m["Entry"] >= 2023) & (m["Season"] <= m["Entry"] + 1)
    nonyouth_same = bool(np.allclose(base.loc[m["Entry"].isna(), "pred_goals_90"],
                                     cand.loc[m["Entry"].isna(), "pred_goals_90"]))
    res = {"rows": int(mask.sum()), "non_youth_unchanged": nonyouth_same}
    for t_, (a, p, _) in TARGETS.items():
        w = base["weight"]
        eb = ((base[p] - base[a]) * w)[mask] ** 2
        ec = ((cand[p] - cand[a]) * w)[mask] ** 2
        d, ci = boot_diff(eb.to_numpy(), ec.to_numpy(), base.loc[mask, "Player"].to_numpy())
        res[t_] = {"change": d, "ci95": ci}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(LOG, encoding="utf-8-sig")
    rows, ent = youth_rows(log)
    res = {"youth_entrants": int(len(ent)), "by_entry": ent["Entry"].value_counts().sort_index().to_dict(),
           "targets": {}}
    for t_, df in rows.items():
        df = df.reset_index(drop=True)
        test = df["Entry"] >= 2019
        single, cs = rolling(df, False)
        group, _ = rolling(df, True)
        r = {"rows_test": int(test.sum()),
             "incumbent_calibration_all": float(df["inc"].sum() / df["y"].sum()),
             "c_by_fold": cs, "c_2017_2022": fit_c(df[df["Entry"] <= 2022]),
             "c_production_2017_2025": fit_c(df[df["Entry"] <= 2025]),
             "c_by_group_2017_2025": {g: fit_c(s) for g, s in df[df["Entry"] <= 2025].groupby("Group")}}
        y = df.loc[test, "y"]
        e_inc = ((df.loc[test, "inc"] - y) ** 2).to_numpy()
        for name, pred in (("single", single), ("group", group)):
            e = ((pred[test] - y) ** 2).to_numpy()
            d, ci = boot_diff(e_inc, e, df.loc[test, "Player"].to_numpy())
            r[name] = {"mse_change": d, "ci95": ci,
                       "calibration": float(pred[test].sum() / y.sum())}
        r["incumbent_calibration_test"] = float(df.loc[test, "inc"].sum() / y.sum())
        for g, s in df[test].groupby("Group"):
            r.setdefault("incumbent_calibration_by_group", {})[g] = float(s["inc"].sum() / s["y"].sum())
        res["targets"][t_] = r
    c_npg, c_ast = res["targets"]["npxG"]["c_2017_2022"], res["targets"]["xAG"]["c_2017_2022"]
    res["loop_check"] = loop_check(c_npg, c_ast, ent)
    T = res["targets"]
    res["adopt"] = bool(all(T[t]["single"]["mse_change"] < 0 and
                            abs(T[t]["single"]["calibration"] - 1) < abs(T[t]["incumbent_calibration_test"] - 1) and
                            res["loop_check"][t]["change"] <= 0 for t in TARGETS))
    res["group_beats_single"] = bool(all(T[t]["group"]["mse_change"] < T[t]["single"]["mse_change"] for t in TARGETS))
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
