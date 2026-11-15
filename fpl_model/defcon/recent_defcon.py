"""Causal exposure-weighted recent defensive activity, without a hard window.

Decay is per >=60-minute appearance. Short appearances do not alter or age this
rate. A newly applicable transfer prior starts a new recent segment. Use the
same helper for historical pre-match rows and production player profiles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EW_HALF_LIFE = 4.0


@dataclass
class RecentRate:
    numerator: float = 0.0
    exposure: float = 0.0
    prior_id: int | None = None
    half_life: float = EW_HALF_LIFE

    def apply_prior(self, arrival, prior):
        if prior is not None and arrival != self.prior_id:
            self.numerator = float(prior["matches"])*float(prior["per90"])
            self.exposure = float(prior["matches"])
            self.prior_id = arrival

    def value(self, fallback=0.0):
        return self.numerator/self.exposure if self.exposure > 0 else float(fallback)

    def observe(self, actions, minutes):
        if minutes < 60:
            return
        if not math.isfinite(actions) or not math.isfinite(minutes) or actions < 0:
            raise ValueError("Qualifying EW observations require finite nonnegative actions")
        decay = 2.0**(-1.0/self.half_life)
        self.numerator = decay*self.numerator+float(actions)
        self.exposure = decay*self.exposure+float(minutes)/90.0


def eligible_prior(records, season):
    eligible = [(int(s), p) for s, p in records if s <= season]
    return max(eligible, key=lambda x:x[0]) if eligible else (None, None)


def attach_recent_rates(frame, priors_by_player, forecast_season):
    """Return pre-row rates and post-history lookup; preserve existing row order.

    frame must have the production build_dataset order (each player's rows in
    chronological order, by its Chrono column where it has one, else MatchID).
    RawStat/Minutes are outcomes: each is consumed only after emitting its rate.
    """
    result = np.empty(len(frame), dtype=float)
    latest = {}
    order_col = "Chrono" if "Chrono" in frame.columns else "MatchID"
    for player, positions in frame.groupby("Player", sort=False).indices.items():
        state = RecentRate()
        records = priors_by_player.get(player, [])
        group = frame.iloc[positions]
        if not group[order_col].is_monotonic_increasing:
            raise ValueError(f"EW history is out of order for {player}")
        for j, row in zip(positions, group.itertuples(index=False)):
            arrival, prior = eligible_prior(records, int(row.Season))
            state.apply_prior(arrival, prior)
            result[j] = state.value(row.AvgPer90)
            state.observe(float(row.RawStat), float(row.Minutes))
        arrival, prior = eligible_prior(records, forecast_season)
        state.apply_prior(arrival, prior)
        latest[player] = state.value(float(group.iloc[-1].AvgPer90))
    return result, latest


def feature_columns(baseline_cols, mode):
    if mode not in {"baseline", "no_std", "ew", "no_std_ew", "lean"}:
        raise ValueError(f"Unknown defender feature mode: {mode}")
    dropped = set()
    if mode in {"no_std", "no_std_ew", "lean"}:
        dropped.add("StdPer90")
    if mode in {"ew", "no_std_ew", "lean"}:
        dropped.update(["Recent5AvgPer90", "Recent5AvgRaw"])
    if mode == "lean":
        dropped.add("AvgRaw")
    cols = [c for c in baseline_cols if c not in dropped]
    if mode in {"ew", "no_std_ew", "lean"}:
        cols.append("RecentEWPer90")
    return cols
