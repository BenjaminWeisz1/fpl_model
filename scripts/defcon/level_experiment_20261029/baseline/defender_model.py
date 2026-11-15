"""Locked monotonic defender candidate; validated before production selection.

Constraints apply to evidence of defensive activity, with every other feature
held fixed. Experience, position and match context remain unconstrained; the
default feature schema excludes spread.
"""

POSITIVE_EVIDENCE = (
    "AvgPer90", "AvgRaw", "Recent5AvgPer90", "Recent5AvgRaw",
    "PctAboveThreshold", "RecentEWPer90",
)

# §46: remove spread and replace both recent-five windows. The baseline
# monotonic schema remains available through DEFCON_DEFENDER_FEATURES=baseline.
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
