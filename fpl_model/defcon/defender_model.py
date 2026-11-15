"""Locked monotonic defender candidate; validated before production selection.

Constraints apply to evidence of defensive activity, with every other feature
held fixed. Experience, position and match context remain unconstrained; the
default feature schema excludes spread.
"""

POSITIVE_EVIDENCE = (
    "AvgPer90", "AvgRaw", "Recent5AvgPer90", "Recent5AvgRaw",
    "PctAboveThreshold", "RecentEWPer90",
)

# §46: remove spread and replace both recent-five windows (the production schema; the others are kept for
# research and the frozen validation copies).
DEFAULT_FEATURE_MODE = "no_std_ew"


def monotonic_params(feature_cols):
    """One pre-registered configuration, not a search over held-out seasons."""
    return {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_leaf_nodes": 7,
        "min_samples_leaf": 50,
        "l2_regularization": 10.0,
        "early_stopping": False,
        "monotonic_cst": [int(c in POSITIVE_EVIDENCE) for c in feature_cols],
    }


# §106: activity features at the current league level. The defender per-90 level drifts by season
# (9.1 in 2017, 7.3 in 2020-21, 7.9 in 2024), and career averages lag it most for long careers.
# Each earlier >= 60-minute match is assigned its season's level (a match in the row's own season gets
# the previous season's, so no future level is used); AvgPer90/AvgRaw are scaled by
# L(s-1) / their plain mean, RecentEWPer90 by L(s-1) / their EW mean. Rows with no earlier qualifying
# match, or no previous-season level, keep factor 1. One helper for training rows and forecasts.
LEVEL_PLAIN_COLS = ("AvgPer90", "AvgRaw")
LEVEL_EW_COLS = ("RecentEWPer90",)


def season_levels(frame):
    """Mean >= 60-minute per-90 of the defender dataset, by season."""
    q = frame[frame["Minutes"] >= 60]
    return q.groupby("Season")["Per90Stat"].mean().to_dict()


def _factors(seen, season, levels, decay):
    if not seen or (season - 1) not in levels:
        return 1.0, 1.0
    cur = levels[season - 1]
    lv = [levels[x] if x < season else cur for x in seen]
    n = len(lv)
    w = [decay ** (n - 1 - i) for i in range(n)]
    return cur * n / sum(lv), cur * sum(w) / sum(a * b for a, b in zip(w, lv))


def level_factors(frame, levels, forecast_season, half_life):
    """Per-row (plain, EW) factors for a frame in production order, plus each player's factors
    for a forecast row in forecast_season (all his rows count as earlier matches)."""
    import numpy as np
    decay = 2.0 ** (-1.0 / half_life)
    plain, ew = np.ones(len(frame)), np.ones(len(frame))
    latest = {}
    for player, pos in frame.groupby("Player", sort=False).indices.items():
        seen = []
        for j, s, mins in zip(pos, frame["Season"].values[pos], frame["Minutes"].values[pos]):
            plain[j], ew[j] = _factors(seen, int(s), levels, decay)
            if mins >= 60:
                seen.append(int(s))
        latest[player] = _factors(seen, int(forecast_season), levels, decay)
    return plain, ew, latest


def apply_level_factors(frame, plain, ew):
    out = frame.copy()
    for c in LEVEL_PLAIN_COLS:
        if c in out:
            out[c] = out[c] * plain
    for c in LEVEL_EW_COLS:
        if c in out:
            out[c] = out[c] * ew
    return out


def level_adjust_features(feat, factors):
    """A forecast feature dict at the current level; factors = (plain, EW) from level_factors' latest."""
    plain, ew = factors
    out = dict(feat)
    for c in LEVEL_PLAIN_COLS:
        if c in out:
            out[c] = out[c] * plain
    for c in LEVEL_EW_COLS:
        if c in out:
            out[c] = out[c] * ew
    return out


class CenteredIsotonic:
    """Centred isotonic calibration (§107; Oron & Flournoy 2017). Fits sklearn's isotonic regression, then
    interpolates linearly between each block's centre (mean score in the block) at the block's fitted value,
    clipped to the end values. Monotone like isotonic, without the step plateaus that gave many defenders
    identical probabilities."""

    def fit(self, scores, y):
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        scores, y = np.asarray(scores, dtype=float), np.asarray(y, dtype=float)
        fitted = IsotonicRegression(out_of_bounds="clip").fit(scores, y).predict(scores)
        levels = np.unique(fitted)
        self.x_ = np.array([scores[fitted == v].mean() for v in levels])
        self.y_ = levels
        return self

    def predict(self, scores):
        import numpy as np
        return np.interp(np.asarray(scores, dtype=float), self.x_, self.y_)
