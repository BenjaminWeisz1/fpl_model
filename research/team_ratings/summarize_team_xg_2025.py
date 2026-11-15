"""
Summary of the 2025 team-xG rebuild test (CHANGELOG §81): reads the harness's
--dump-predictions files for the baseline and candidate runs under
data/validation/team_xg_2025_20261028/ and reports, per held-out season and mode,
mean predicted - actual NPG per team-match and the Poisson log-loss kernel; also
checks that the 2023 and 2024 splits are identical between the two runs.

Run locally (after both harness runs):
    python summarize_team_xg_2025.py
"""
from fpl_model import paths
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIR = paths.VALIDATION / "team_xg_2025_20261028"


def long(frame):
    parts = []
    for side in ("Home", "Away"):
        for mode, prefix in (("frozen", "Frozen"), ("walk_forward", "WF")):
            parts.append(pd.DataFrame({"Cutoff": frame["Cutoff"], "mode": mode,
                                       "pred": frame[f"{prefix}_{side}"], "npg": frame[f"{side}_NPG"]}))
    return pd.concat(parts, ignore_index=True)


def main():
    b, c = (pd.read_csv(DIR / f"pred_{x}_all.csv") for x in ("baseline", "candidate"))
    early = b["Cutoff"] < 2025
    cols = ["Frozen_Home", "Frozen_Away", "WF_Home", "WF_Away"]
    identical_2023_2024 = bool((b.loc[early, cols].to_numpy() == c.loc[early, cols].to_numpy()).all())
    out = {"identical_2023_2024": identical_2023_2024, "by_season": {}}
    for label, frame in (("baseline", b), ("candidate", c)):
        d = long(frame)
        d["ll"] = d["pred"] - d["npg"] * np.log(d["pred"])
        d["err"] = d["pred"] - d["npg"]
        g = d.groupby(["Cutoff", "mode"]).agg(bias=("err", "mean"), ll_per_tm=("ll", "mean"), n=("ll", "size"))
        out["by_season"][label] = {f"{k[0]}_{k[1]}": {kk: round(float(vv), 5) for kk, vv in v.items()}
                                   for k, v in g.to_dict("index").items()}
    (DIR / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
