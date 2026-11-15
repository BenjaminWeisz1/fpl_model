"""
Score the §128 returners' prior candidate: baseline vs candidate pre-match logs (CHANGELOG §128).

    python validate_returner_priors.py <dir with log_base.csv, log_cand.csv, returner_priors.csv>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.player_ratings.validate_prior_k import TARGETS, boot_diff, sq_errors  # noqa: E402

D = Path(sys.argv[1])
base = pd.read_csv(D / "log_base.csv", encoding="utf-8-sig", low_memory=False)
cand = pd.read_csv(D / "log_cand.csv", encoding="utf-8-sig", low_memory=False)
assert len(base) == len(cand) and (base["Player"].to_numpy() == cand["Player"].to_numpy()).all()
keys = set(map(tuple, pd.read_csv(D / "returner_priors.csv", encoding="utf-8-sig")[["Player", "Season"]].to_numpy()))
p, s = base["Player"].to_numpy(), base["Season"].to_numpy()
ret = np.array([(a, b) in keys or (a, b - 1) in keys for a, b in zip(p, s)]) & (s >= 2019)
all19 = s >= 2019
eb, ec = sq_errors(base), sq_errors(cand)
res = {"returner_rows": int(ret.sum()), "returners": len({(a, b) for a, b in zip(p[ret], s[ret])})}
for t in TARGETS:
    ch, ci = boot_diff(eb[t][ret], ec[t][ret], p[ret])
    a, pr = TARGETS[t]
    w = base["weight"].to_numpy()[ret]
    res[t] = {"change": ch, "ci95": ci, "all_2019": float(ec[t][all19].sum() / eb[t][all19].sum() - 1),
              "calibration_base": float((base[pr].to_numpy()[ret] * w).sum() / (base[a].to_numpy()[ret] * w).sum()),
              "calibration_cand": float((cand[pr].to_numpy()[ret] * w).sum() / (cand[a].to_numpy()[ret] * w).sum())}
res["adopt"] = bool(all(res[t]["change"] < 0 and res[t]["all_2019"] <= 0.0005 for t in TARGETS))
(D / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
