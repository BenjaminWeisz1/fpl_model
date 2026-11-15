"""Defcon model: position classification and the per-match feature helpers shared by training and serving.

Split out of predict.py (2026-11-16 cleanup) without changes. Every per-90 quantity excludes sub-60-minute appearances
(QUALIFYING_MINUTES_P90, CHANGELOG §36); raw averages exclude cameos under RAW_CAMEO_MIN minutes (§15).
"""
import numpy as np
import pandas as pd


# ── Position classification ────────────────────────────────────────────
# Accepts both detailed fbref positions (CB, FB, WB, etc.) and coarse FPL-API
# positions (D, M, F) -- the latter come from 2025/26 rows where detailed
# tactical position data isn't available.
DEFENDER_POSITIONS = {"CB", "FB", "WB", "LB", "RB", "D"}
MIDFIELDER_POSITIONS = {"DM", "CM", "AM", "WM", "W", "LM", "RM", "LW", "RW", "M"}
FORWARD_POSITIONS = {"FW", "F"}

def is_defender_pos(pos_str):
    if pd.isna(pos_str): return False
    return any(p.strip() in DEFENDER_POSITIONS for p in str(pos_str).split(","))

def is_midfielder_or_forward_pos(pos_str):
    """Midfielders and forwards share the same defcon rules."""
    if pd.isna(pos_str): return False
    positions = [p.strip() for p in str(pos_str).split(",")]
    return any(p in MIDFIELDER_POSITIONS or p in FORWARD_POSITIONS for p in positions)

def is_forward_pos(pos_str):
    """Forward rows (any FW/F token) -- the forward-specific model's row
    selector. Mirrors the any-token convention of the other classifiers:
    a fbref row 'FW,AM' is a forward row; the transfers fallback writes a
    single first token, so first-token and any-token coincide there."""
    if pd.isna(pos_str): return False
    return any(p.strip() in FORWARD_POSITIONS for p in str(pos_str).split(","))

# Detailed position lists for the new Pos_* multi-hot features -- lets the
# model learn position-specific defcon tendencies directly (e.g. CB > FB > WB)
# rather than relying entirely on indirect signal through a player's own
# individual stats.
DEFENDER_DETAILED_POS = ["CB", "FB", "WB", "LB", "RB"]
MID_FWD_DETAILED_POS = ["DM", "CM", "AM", "WM", "W", "LM", "RM", "LW", "RW", "FW"]

def get_pos_dummies(pos_str, detailed_positions):
    """Multi-hot encoding: a row like 'CB,FB' sets BOTH Pos_CB and Pos_FB to 1,
    reflecting that the player featured in both positions that match."""
    if pd.isna(pos_str):
        codes = set()
    else:
        codes = set(p.strip() for p in str(pos_str).split(","))
    return {f"Pos_{c}": (1 if c in codes else 0) for c in detailed_positions}


# The >= 60 minute bar every per-90 feature in this model uses.
QUALIFYING_MINUTES_P90 = 60


def pooled_prior_std(est_std, k, prior_per90, pl_values):
    """StdPer90 for a prior-blended player: the sample std of the COMBINED
    evidence -- k pseudo-matches at the prior-league rate (spread est_std,
    borrowed from rate-similar PL players) plus the player's qualifying PL
    per-90s -- via two-sample pooled variance:

        [(k-1) est^2 + (n-1) s^2 + (k n / (k+n)) (xbar - prior)^2] / (k+n-1)

    The PL-only std (old behaviour) lets a 2-match arrival's variance be
    dominated by noise: matches of 7 and 19 -> std 8.5 when the rate-implied
    std is ~6, and the model reads that phantom volatility as risk,
    depressing a proven arrival's probability. Edge cases, all derived from
    the same formula:
      n = 0     -> est_std exactly (the combined sample IS the k
                   pseudo-matches) -- covers debut rows and tier-2 players
      n = 1     -> [(k-1) est^2 + (k/(k+1)) (x1 - prior)^2] / k: one match
                   carries no sample variance, but its distance from the
                   prior rate is real spread evidence (old behaviour: flat
                   est_std, discarding that information)
      k + n < 2 -> est_std fallback (fewer than two observations to span,
                   e.g. Min_Old < 90 with no PL matches yet)
    """
    vals = [float(v) for v in pl_values if pd.notna(v)]
    n = len(vals)
    if n == 0 or k + n < 2:
        return float(est_std)
    num = (k - 1.0) * est_std * est_std
    if n > 1:
        num += (n - 1.0) * float(np.std(vals, ddof=1)) ** 2
    num += (k * n / (k + n)) * (float(np.mean(vals)) - prior_per90) ** 2
    return float(np.sqrt(max(num, 0.0) / (k + n - 1.0)))


# ── Cameo filtering for raw features (AvgRaw / Recent5AvgRaw) ─────────
# Per-90 features already exclude sub-60-min matches (a cameo's per-90 is
# extrapolation noise: 3 actions in 1 minute -> 270 per-90). Raw per-match
# volume has the OPPOSITE failure: a cameo's raw total is mechanically
# SMALL (fewer minutes -> fewer actions), so cameo matches bias per-match
# raw averages DOWNWARD -- a bench player with 7 raw per full match shows
# AvgRaw ~4 once 15-minute appearances are averaged in. AvgRaw and
# Recent5AvgRaw therefore count only non-cameo matches (Minutes >=
# RAW_CAMEO_MIN). RAW_CAMEO_MIN = 15 was selected from candidates 15/30/45:
# 30 was uniformly worse out-of-time and the degradation grew monotonically
# with the threshold (45 was cancelled mid-sweep), so 15 -- the gentlest
# filter -- shipped even though the no-filter incumbent edged it on OOT log
# loss (+0.0026 sum); the trade buys unbiased per-match raw volume for bench
# players (CHANGELOG §15). Re-test as data accumulates (0 reproduces the pre-§15 behaviour exactly).
# Edge cases (same discipline as pooled_prior_std):
#   - no non-cameo match + prior    -> AvgRaw = prior rate (the blend
#     degenerates to the k pseudo-matches; also covers debut rows)
#   - no non-cameo match, no prior  -> 0.0 (same convention as AvgPer90
#     when no qualifying matches exist)
#   - last-5 window mostly cameos (>= RECENT5_CAMEO_LIMIT cameos among
#     the last 5) -> Recent5AvgRaw falls back to the cameo-filtered
#     AvgRaw instead of collapsing to a zero-/one-match "recent" average
#   - short window with too little full-match signal (fewer than
#     RECENT5_MIN_QUALIFYING non-cameo matches -- e.g. a new arrival whose
#     2-3 career matches mix one full match with cameos) -> same fallback;
#     a full 5-match window with <= 2 cameos always keeps >= 3 qualifying
#     matches, so this only bites short windows
RAW_CAMEO_MIN = 15
RECENT5_CAMEO_LIMIT = 3
RECENT5_MIN_QUALIFYING = 2
# Recent5AvgPer90 was ONE OF TWO per-90 features with no cameo filter (the
# opponent/team/venue aggregates were the other -- see compute_season_aggregates), and it
# suffered exactly the extrapolation failure the comment above describes:
# 40% of defender last-5 windows contain a sub-60' match, and ~84% of
# windows above 14 per 90 were bench cameos rather than form. The model
# learned that correctly and inverted -- raising Recent5AvgPer90 from 10 to
# 14 LOWERED the prediction for 85.5% of served defender rows, while the
# cameo-filtered relationship keeps rising (0.459 -> 0.468 observed, and
# 0.507 -> 0.611 among starters). The feature now uses the same >= 60'
# qualifying bar as every other per-90 feature (CHANGELOG §36).


def avg_raw_qualifying(sum_raw_q, n_raw_q, k=0.0, prior_per90=0.0):
    """AvgRaw over non-cameo matches only, blended with k prior-league
    pseudo-matches at the prior rate. 0.0 when there is no evidence at
    all (no non-cameo match and no prior)."""
    denom = n_raw_q + k
    if denom <= 0:
        return 0.0
    return (sum_raw_q + k * prior_per90) / denom


def recent5_qualifying_raw(df, stat_col):
    """Recent5AvgRaw candidate from a chronologically ordered history frame:
    the mean raw stat over the non-cameo matches among the player's last
    5. Returns None when the window is mostly cameos (>= RECENT5_CAMEO_LIMIT
    cameos -- no recent full-match signal exists) or holds fewer than
    RECENT5_MIN_QUALIFYING non-cameo matches (a short window whose "recent
    form" would otherwise rest on a single match); the caller then falls
    back to the cameo-filtered AvgRaw, so the feature can never collapse
    to a zero- or one-match average."""
    last5 = df.tail(5)
    if int((last5["Minutes"] < RAW_CAMEO_MIN).sum()) >= RECENT5_CAMEO_LIMIT:
        return None
    qual = last5[last5["Minutes"] >= RAW_CAMEO_MIN][stat_col]
    if len(qual) < RECENT5_MIN_QUALIFYING:
        return None
    return float(qual.mean())


def recent5_qualifying_per90(df, per90_col):
    """Recent5AvgPer90 candidate from a chronologically ordered history frame:
    the mean per-90 over the QUALIFYING (>= 60 minute) matches among the
    player's last 5 appearances. Returns None when fewer than
    RECENT5_MIN_QUALIFYING of them qualify -- the caller then falls back to
    the player's career AvgPer90, mirroring how Recent5AvgRaw falls back to
    AvgRaw. Cameo per-90s are extrapolation noise (3 actions in 1 minute ->
    270 per-90) and belong in no per-90 feature."""
    last5 = df.tail(5)
    qual = last5[last5["Minutes"] >= QUALIFYING_MINUTES_P90][per90_col]
    if len(qual) < RECENT5_MIN_QUALIFYING:
        return None
    return float(qual.mean())


def estimate_std_and_pct(pos_feats):
    """Nearest-neighbour estimator for two career-level statistics of
    a player with a given in-dataset AvgPer90: StdPer90 (spread) and
    PctAboveThreshold (target rate). Median of the 10 nearest PL
    players (>=5 qualifying matches) sorted by career AvgPer90."""
    stats = [(f["AvgPer90"], f["StdPer90"], f["PctAboveThreshold"])
             for f in pos_feats.values() if f["MatchesPlayed"] >= 5]
    if not stats: return lambda p: (4.0, 0.3)
    stats = sorted(stats, key=lambda x: x[0])
    avg_per90s = np.array([s[0] for s in stats])
    std_per90s = np.array([s[1] for s in stats])
    pct_aboves = np.array([s[2] for s in stats])
    def estimate(per90):
        idx = np.argsort(np.abs(avg_per90s - per90))[:10]
        return (np.median(std_per90s[idx]),
                np.median(pct_aboves[idx]))
    return estimate
