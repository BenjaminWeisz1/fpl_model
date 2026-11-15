"""
Shared pass-band logic for the bonus model (adopted in CHANGELOG §68).

Pass-band BPS (>= 30 passes attempted: 2/4/6 for 70/80/90%+ completion) is a
discrete simulated term on 60+ minute draws. Each player's distribution over
{0, 2, 4, 6} comes from his last HISTORY Sofascore 60+ appearances, shrunk toward
his position's distribution, then exponentially tilted by opponent and venue
effects. The residual excludes actual pass-band BPS and uses K_REM shrinkage.

The constants were fixed by the §68 experiment (2018-2024 Sofascore and the
2025/26 fit split) and must not be re-estimated from later outcomes without a new
pre-registration. Pass bands are active only when Sofascore is usable
(sofascore_status.json); otherwise every stage falls back to the §64 model.
"""
import json

import numpy as np
import pandas as pd
from scipy import optimize

from fpl_model import paths

_DATA = paths.CURRENT

BANDS = np.array([0, 2, 4, 6])
HISTORY = 38
PRIOR = {"D": [0.2728, 0.1849, 0.3537, 0.1886], "M": [0.3604, 0.1460, 0.3339, 0.1597],
         "F": [0.8333, 0.0627, 0.0810, 0.0229], "G": [0.8116, 0.0792, 0.0707, 0.0384]}
K_PB = {"D": 1.9639, "M": 2.0001, "F": 2.9731, "G": 3.5142}
VENUE = {"Home": 0.0734, "Away": -0.0738}
K_OPP = 3.5993
K_REM = {"D": 17.666, "M": 15.100, "F": 21.311, "G": 100.0}


def active(data=_DATA):
    """(bool, reason): whether pass bands replace the §64 residual-only model this run."""
    path = data / "sofascore_status.json"
    if not path.exists():
        return False, "sofascore_status.json missing"
    status = json.loads(path.read_text(encoding="utf-8"))
    if status.get("usable_for_model") is not True:
        return False, "Sofascore not usable: " + "; ".join(status.get("errors", []) or
                                                         [f"lag {status.get('lag_rounds')} rounds"])
    return True, "Sofascore usable"


def pass_band(total, accurate):
    total, accurate = np.asarray(total, float), np.asarray(accurate, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total > 0, accurate / total, 0)
    return np.where(total >= 30, np.select([share >= .9, share >= .8, share >= .7], [6, 4, 2], 0), 0)


def load_sofascore(data=_DATA):
    """Played Sofascore rows with pass-band BPS, 60+ flag and kickoff time."""
    so = pd.read_csv(data / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Canonical_Player_ID", "Local_Player", "Team", "Opponent", "Venue",
                              "Kickoff", "Event_ID_Sofascore", "Appearance_Status", "Sofascore_minutesPlayed",
                              "Sofascore_totalPass", "Sofascore_accuratePass"])
    so = so[so["Appearance_Status"].eq("played")].copy()
    so["Kickoff"] = pd.to_datetime(so["Kickoff"], utc=True)
    so["band"] = pass_band(so["Sofascore_totalPass"].fillna(0), so["Sofascore_accuratePass"].fillna(0))
    so["full"] = so["Sofascore_minutesPlayed"] >= 60
    return so


def shrunk(counts, pos):
    prior = np.array(PRIOR[pos])
    return (counts + K_PB[pos] * prior) / (counts.sum() + K_PB[pos])


def tilt(p, shift):
    """Exponential tilt of a band distribution so its mean moves by `shift` BPS."""
    target = float(np.clip(p @ BANDS + shift, 0.02, 5.98))
    def f(lam):
        return (p * np.exp(lam * BANDS)) @ BANDS / (p * np.exp(lam * BANDS)).sum() - target
    lam = optimize.brentq(f, -10, 10)
    q = p * np.exp(lam * BANDS)
    return q / q.sum()


def history(so):
    """Per canonical player: band counts and mean over his last HISTORY 60+ appearances."""
    recent = so[so["full"]].sort_values("Kickoff").groupby("Canonical_Player_ID").tail(HISTORY)
    counts = recent.groupby("Canonical_Player_ID")["band"].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=BANDS, fill_value=0)
    return counts, recent.groupby("Canonical_Player_ID")["band"].mean()


def opponent_effects(so, season, pmean):
    """Shrunken effect of facing each opponent this season, in pass-band BPS."""
    cur = so[so["full"] & (so["Season"] == season)].copy()
    cur["dev"] = cur["band"] - cur["Canonical_Player_ID"].map(pmean)
    per_match = cur.groupby(["Opponent", "Event_ID_Sofascore"])["dev"].mean().groupby("Opponent").agg(["sum", "size"])
    return (per_match["sum"] / (per_match["size"] + K_OPP)).to_dict()
