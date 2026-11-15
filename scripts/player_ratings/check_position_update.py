"""
Check that build_player_ratings.py's entrant position update (CHANGELOG §249) reproduces §247's exact replay.

    python check_position_update.py <log_off> <log_on> <priors_file> <out_json>

log_off / log_on: GOAL_MODEL_PREMATCH_LOG runs with GOAL_MODEL_POS_UPDATE=0 / 1, same inputs, PYTHONHASHSEED=0.
The replay (validate_position_forms.py, form F0, file rates, Kpos 1, c 0.4 npxG / 0.35 xAG) applied to log_off must
give log_on's pre-match ratings on every entrant row.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_arrival_position as vap  # noqa: E402
import validate_position_rates as vpr  # noqa: E402
import validate_position_forms as vpf  # noqa: E402

C = {"npxG": 0.4, "xAG": 0.35}
KPOS = 1.0


def main(log_off, log_on, priors_file, out_json):
    vap.LOG = Path(log_off)
    log = vap.load()
    on = pd.read_csv(log_on, encoding="utf-8-sig", low_memory=False).sort_values("Player", kind="stable") \
        .reset_index(drop=True)
    key = ["Player", "Team", "Opponent", "Venue", "Season", "MatchID"]
    assert len(on) == len(log) and (on[key].to_numpy() == log[key].to_numpy()).all(), "logs not aligned"
    seg = vpr.segments(log, priors_file)
    sid, ent = seg["seg"], seg["type"] != ""
    rows_by_seg = {s_: np.where((sid == s_) & ent)[0] for s_ in sorted(set(sid[ent]))}
    tbl = vpf.rate_table("file")(2026)
    w = log["w"].to_numpy()
    out = {"entrant_rows": int(ent.sum()), "segments": len(rows_by_seg)}
    for t, (_, _, rat_col, col) in vap.PAIRS.items():
        a, _ = vap.step_weights(log, t)
        slot_rate = np.array([tbl.loc[s, col] if isinstance(s, str) and s != "GK" else np.nan for s in log["Slot"]])
        SW, SWR = vpf.evidence(log, seg, rows_by_seg, slot_rate, None, None, None, None, "F0", None, w)
        expect = log[rat_col].to_numpy().copy()
        got = on[rat_col].to_numpy()
        carried = 0
        for s_, rows in rows_by_seg.items():
            pl = vpf.combined(tbl, seg["pos"][rows[0]], col)
            s = 1.0
            # A returner's segment starts from a blend that holds his earlier stint's (shifted) rating: that carried
            # shift decays with the same share s. The replay scored segments on their own, so it had none.
            carry = got[rows[0]] - expect[rows[0]]
            carried += abs(carry) > 1e-12
            for i in rows:
                p_t = (KPOS * pl + SWR[i]) / (KPOS + SW[i])
                expect[i] = expect[i] + carry * s + C[t] * s * (p_t - pl)
                s *= 1.0 - a[i]
        diff = np.abs(got - expect)[ent]
        off_diff = np.abs(got - log[rat_col].to_numpy())
        out[t] = {"segments_with_carried_shift": int(carried), "entrant_max_abs_diff": float(diff.max()), "entrant_rows_over_1e-9": int((diff > 1e-9).sum()),
                  "rows_changed_by_update": int((off_diff > 1e-12).sum()),
                  "mean_abs_shift_entrant_rows": float(off_diff[ent].mean())}
    Path(out_json).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
