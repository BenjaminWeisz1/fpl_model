"""
Minutes-scaling of defcon threshold probabilities -- the one copy.

The defcon model predicts P(defensive contribution >= threshold | 90 minutes).
To use that for a player expected to play fewer than 90 minutes, invert the
probability back to an implied mean defcon rate through the NB(r=3*mu, p=0.75)
relationship used throughout this project, scale the mean by xMins/90, then
recompute the probability at the scaled mean.

``fpl_model/defcon/predict.py`` does the inversion itself and writes both the
unscaled probability (``Probability_at_90min``) and its own scaled column; the
inverted mean is then carried through ``simulation_features.csv`` as
``Defcon_Mu90``. Anything in part 2 of the pipeline that needs a point-estimate
probability should re-scale that mu with the CURRENT expected minutes through
``prob_at_minutes`` here, because predict.py runs in part 1, a day or more
before the minutes forecast is refreshed.
"""
import numpy as np
from scipy import stats

from fpl_model import config

__all__ = ["DEFENDER_POINTS_FACTOR", "nb_sf", "points_mu90", "points_prob_at_90", "prob_at_minutes"]

# CHANGELOG §153/§159: defender forecasts run under FPL's real Defensive_Contribution hits. Fitted on all FPL-era
# held-out rows; refit when a season ends. Defcon points, charts and player_ratings.csv: bonus keeps the raw Defcon_Mu90.
DEFENDER_POINTS_FACTOR = 1.083
_POINTS_CALIBRATION = config.get("DEFCON_POINTS_CALIBRATION")


def nb_sf(mu, k, p: float = 0.75):
    """P(X > k) for X ~ NB(r=3*mu, p); vectorised over *mu* and *k*."""
    r = np.maximum(3 * np.asarray(mu, dtype=float), 1e-9)
    return 1.0 - stats.nbinom.cdf(k, r, p)


def points_mu90(mu90, pos):
    """*mu90* on FPL's real defcon scale, for points: defenders (Pos "D") times the §153 factor, others unchanged."""
    mu = np.asarray(mu90, dtype=float)
    if not _POINTS_CALIBRATION:
        return mu
    return np.where(np.asarray(pos) == "D", mu * DEFENDER_POINTS_FACTOR, mu)


def points_prob_at_90(prob90, pos, threshold=10):
    """A 90-minute defcon probability on FPL's real scale: through the implied mean for defenders, others unchanged."""
    prob = np.asarray(prob90, dtype=float)
    grid = np.linspace(0.01, 40, 8000)
    mu = np.interp(prob, nb_sf(grid, threshold - 1), grid)
    return np.where(np.isnan(prob), np.nan, np.where(np.asarray(pos) == "D", nb_sf(points_mu90(mu, "D"), threshold - 1), prob))


def prob_at_minutes(mu90, xmins, threshold):
    """P(defcon stat >= threshold) for a player expected to play *xmins*.

    *mu90* is the 90-minute mean defcon rate (``Defcon_Mu90`` in
    simulation_features.csv), *threshold* the position's defcon threshold.
    Vectorised: all three arguments may be arrays or Series.
    """
    mu_scaled = np.asarray(mu90, dtype=float) * np.asarray(xmins, dtype=float) / 90.0
    return nb_sf(mu_scaled, np.asarray(threshold) - 1)
