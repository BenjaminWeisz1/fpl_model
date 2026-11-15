"""Locked monotonic defender candidate; validated before production selection.

Constraints apply to evidence of defensive activity, with every other feature
held fixed. Spread, experience, position and match context remain unconstrained.
"""

POSITIVE_EVIDENCE = (
    "AvgPer90", "AvgRaw", "Recent5AvgPer90", "Recent5AvgRaw",
    "PctAboveThreshold",
)


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
