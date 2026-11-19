"""
Frozen constants and the rate model for FPL-only assists (CHANGELOG §90–§95).

FPL credits assists conventional data misses: rebounds from saved, blocked or woodwork shots,
deflected passes, penalties and free kicks won and scored, forced own goals. The goal/assist
loop rates conventional assists only (FPL xA), so they are added here as a per-90 rate:
    FPL-only per 90 = exp(CONST + POS[pos] + B_REBOUND * rebound_rate)
                      + pens_rate * (1 - penalty share) * C_PEN
where rebound_rate and pens_rate are the player's per-90 rebound-capable shots (saved +
blocked + woodwork) and penalties won over his last HISTORY Sofascore appearances, shrunk
toward PRIOR with K full matches. A penalty won earns an assist only when a teammate takes and
scores it, hence (1 - share) x C_PEN (§93–§95; C_PEN is close to the conversion rate). Keepers
use their fallback rate. When Sofascore is not usable, every player gets FALLBACK[pos], the
observed 2025+ position rate.

Fitted on all 2025+ FPL-era rows (12,377 outfield player-matches; production_fit_A3.json in
data/validation/fpl_only_assists_20261028/). Do not re-estimate without a new pre-registration.
"""
import json

import numpy as np

from fpl_model import paths

_DATA = paths.CURRENT

HISTORY = 38
K = 10.0
CONST = -4.2059
POS = {"D": 0.0, "M": 0.2752, "F": -0.1453}
B_REBOUND = 0.5862
C_PEN = 0.7722
PRIOR = {"rebound_rate": {"D": 0.31626, "M": 0.91039, "F": 1.28597, "G": 0.00039},
         "pens_rate": {"D": 0.00442, "M": 0.01135, "F": 0.02605, "G": 0.0}}
FALLBACK = {"D": 0.02144, "M": 0.04100, "F": 0.04006, "G": 0.00109}


def mode(data=_DATA):
    """(mode, reason): 'model', or 'fallback' (position rates) when Sofascore is not usable."""
    path = paths.current_file("sofascore_status.json", root=data)
    if not path.exists():
        return "fallback", "sofascore_status.json missing"
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("usable_for_model") is not True:
        return "fallback", "Sofascore not usable"
    return "model", "Sofascore usable"


def shrunk_rate(total, minutes, pos, kind):
    """Per-90 rate over the history window, shrunk toward the position prior with K matches."""
    return (total + PRIOR[kind][pos] * K) / (minutes / 90 + K)


def rate(pos, rebound_rate, pens_rate, share=0.0):
    """share: the player's expected share of his team's penalties (penalty_share.csv)."""
    if pos == "G":
        return FALLBACK["G"]
    share = min(max(float(share), 0.0), 1.0)
    return float(np.exp(CONST + POS[pos] + B_REBOUND * rebound_rate) + pens_rate * (1 - share) * C_PEN)
