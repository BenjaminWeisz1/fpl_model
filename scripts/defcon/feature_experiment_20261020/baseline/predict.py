import os
import sys
import gzip
import json
import pickle
import re
import time
import hashlib
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import sklearn
from sklearn.model_selection import (
    cross_val_score, cross_val_predict, StratifiedKFold, RandomizedSearchCV,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from defender_model import monotonic_params
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from scipy.stats import chi2

_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SCRIPTS_DIR / "team_ratings"))
from get_next_round import get_next_round

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
CURRENT_SEASON = 2026

# ── Load data ──────────────────────────────────────────────────────────
# Read the consolidated player_match_stats.csv (replaces the old player_matchlogs.csv).
# CBI is now a native column. FPL_API-era Position is a mix: many rows carry a
# native position (coarse D/M/F/G from the builder, or detailed fbref-style
# codes like FW/CB/CM/W), and the rest are NULL and filled first from
# players_current.csv (coarse D/M/F/G) and then from transfers.csv below --
# see the transfers fallback block for the departed-player rule.
matchlogs = pd.read_csv(_DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
players_current = pd.read_csv(_DATA / "players_current.csv", encoding="utf-8-sig")

# Fill NULL Position (FPL_API rows) from players_current.csv
_pos_lookup = dict(zip(players_current["Player"], players_current["Pos"]))
matchlogs["Position"] = matchlogs["Position"].fillna(matchlogs["Player"].map(_pos_lookup))
fixtures = pd.read_csv(_DATA / "fixtures.csv")
# Restrict to the next 5 rounds -- minutes_probabilities.csv only covers GW1-5,
# so anything beyond that would silently fall back to unscaled 90-minute
# probabilities in the minutes-scaling step below. Filtering here also avoids
# wastefully scoring ~33 rounds' worth of fixtures we don't use.
_next_round = get_next_round(_DATA, CURRENT_SEASON)
fixtures = fixtures[(fixtures["Round"] >= _next_round) & (fixtures["Round"] < _next_round + 5)].reset_index(drop=True)
print(f"Next round from match_results.csv: {_next_round}")
print(f"Filtered fixtures to rounds {_next_round}-{_next_round+4}: {len(fixtures)} fixtures")
transfers = pd.read_csv(_DATA / "transfers.csv")

# ── Transfers-based position fallback (departed players) ──────────────
# players_current.csv only covers current squads, so FPL_API rows by players
# who have since left the league (post-window departures, releases, loans)
# keep NULL Position after the fill above. transfers.csv now covers every
# such player (user-completed 2026-09-03): their position is the FIRST
# token before the comma of their most recent arrival record at or before
# the row's season (user rule: "the first position before a comma -- if
# it's FW then it's forward"). GK first tokens match no position set, so
# departed keepers stay excluded from all position datasets. fbref-era
# rows always carry detailed positions and are never touched (verified
# 2026-09-03: 0 fbref rows have NULL Position), so pre-2025 training data
# is bit-for-bit unchanged -- only FPL-era validation rows and the
# production training frame expand. Rows whose player has no transfers
# record remain NULL and drop out of the position datasets exactly as
# before.
_pos_records = {}
for _, _trow in transfers.dropna(subset=["Pos"]).iterrows():
    _pos_records.setdefault(_trow["Player"], []).append(
        (int(_trow["Season"]), str(_trow["Pos"]).split(",")[0].strip()))
for _p in _pos_records:
    _pos_records[_p].sort(key=lambda x: x[0])

def _transfer_position(player, season):
    """First position token of the player's most recent arrival record at
    or before `season` (earliest record if only later arrivals exist)."""
    recs = _pos_records.get(player)
    if not recs:
        return None
    best = recs[0]
    for s_arr, tok in recs:
        if s_arr <= season:
            best = (s_arr, tok)
    return best[1]

_null_pos_mask = matchlogs["Position"].isna()
if _null_pos_mask.any():
    _filled = 0
    for _i in matchlogs.index[_null_pos_mask]:
        _tok = _transfer_position(matchlogs.at[_i, "Player"], int(matchlogs.at[_i, "Season"]))
        if _tok is not None:
            matchlogs.at[_i, "Position"] = _tok
            _filled += 1
    print(f"Transfers position fallback: filled {_filled} of {int(_null_pos_mask.sum())} "
          f"NULL-position rows ({int(matchlogs['Position'].isna().sum())} remain unclassifiable)")

# ── Transfer priors: prior-league defensive evidence ──────────────────
# transfers.csv's Defcon column (populated for new-to-league arrivals,
# currently only the 2026 intake) seeds a debutant's history features with
# their prior-league rate instead of an all-zero "rookie" vector -- so a
# 28-year-old proven Bundesliga CB and a 17-year-old academy debutant no
# longer present identical feature vectors. Keyed on (Player, arrival
# season): a prior only ever applies to matches from its arrival season
# onward, which keeps it leak-safe (a 2026 prior can never touch a 2025
# or earlier match).

# Credibility cap on the transfer prior (CHANGELOG §36). The prior weight
# was k = Min_Old / 90 with no shrinkage, so a full Championship season
# bought ~32 pseudo-matches -- more weight than two thirds of a Premier
# League season, and a promoted-club player's PL form could not move his
# career features (a 22-defcon match moved Bobby Thomas's AvgPer90 by 0.33).
# Measured credibility on the 2026 arrival cohort (31 players with >= 3
# qualifying PL matches): within-player per-match variance 12.36 divided by
# prior-error variance 1.41 => k* = 8.8 pseudo-matches, i.e. the prior earns
# about a third of the weight it was given. Capped at 10 -- the nearest round
# number above the estimate, erring toward keeping prior weight, and still
# leaving the prior dominant on a debut (its whole purpose, §10).
# DEFCON_PRIOR_MAX_K=0 restores the uncapped behaviour.
PRIOR_MAX_K = float(os.environ.get("DEFCON_PRIOR_MAX_K", "10"))


def build_prior_lookup(transfers_df):
    """(Player, arrival_season) -> {'per90': prior-league rate, 'matches':
    prior weight in 90-minute-equivalent matches}."""
    priors = {}
    for _, row in transfers_df.iterrows():
        if pd.isna(row["Min_Old"]) or pd.isna(row["Defcon"]) or row["Min_Old"] <= 0:
            continue
        k = row["Min_Old"] / 90.0
        priors[(row["Player"], int(row["Season"]))] = {
            "per90": row["Defcon"] * 90.0 / row["Min_Old"],
            "matches": min(k, PRIOR_MAX_K) if PRIOR_MAX_K > 0 else k,
        }
    return priors

def prior_for_season(player, season):
    """Most recent transfer prior whose arrival season <= the given season."""
    best = None
    for s_arr, v in _PRIORS_BY_PLAYER.get(player, []):
        if s_arr <= season and (best is None or s_arr > best[0]):
            best = (s_arr, v)
    return best[1] if best else None

# Clean team names (fixtures have trailing newlines)
fixtures["Home"] = fixtures["Home"].str.strip()
fixtures["Away"] = fixtures["Away"].str.strip()

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

matchlogs["IsDefender"] = matchlogs["Position"].apply(is_defender_pos)
matchlogs["IsMidfielderOrForward"] = matchlogs["Position"].apply(is_midfielder_or_forward_pos)
matchlogs["IsForward"] = matchlogs["Position"].apply(is_forward_pos)

# Transfer priors (built after the position helpers above are defined).
_TRANSFER_PRIORS = build_prior_lookup(transfers)
_PRIORS_BY_PLAYER = {}
for (_p, _s), _v in _TRANSFER_PRIORS.items():
    _PRIORS_BY_PLAYER.setdefault(_p, []).append((_s, _v))
for _p in _PRIORS_BY_PLAYER:
    _PRIORS_BY_PLAYER[_p].sort(key=lambda x: x[0])

# ── FPL_API -> fbref stat-scale reconciliation ──────────────────────
# FPL's native defensive stats are NOT on fbref's scale for the same
# players. Paired same-player era comparison (players with >=5 qualifying
# matches in both eras; ratio-of-sums of player mean per-90 rates; n=130
# mid/fwd, n=85 defenders):
#     mid/fwd CBI 0.736 | defender CBI 0.975 (continuous across the switch)
#     Tkl 0.86-0.92 paired but ~0.97 unpaired (the paired set is aging
#     crossover veterans; composition confound) | Recov: defenders 0.759,
#     mid/fwd 0.864
# This is the root cause of the mid/fwd 0-9 newcomer over-prediction: the
# model trained mostly on fbref-scale stats kept predicting fbref-scale
# rates for players whose current-season evidence sits on the lower FPL
# scale -- the exact regime shift the falsifiable prior-mechanism test
# ruled out. Tkl and defender CBI stay untouched (within noise); the two
# clearly definitional gaps are rescaled onto the fbref scale that 8 of 10
# training seasons -- and transfers.csv's prior-league Defcon rates -- are
# on. (Mid/fwd CBI+Tkl+Recov reconstructs at x1.19, vs the x1.198 the
# paired full-stat ratio requires.) Raw values in player_match_stats.csv
# are NOT modified: build_residual_bps_baseline.py models FPL's actual BPS
# scoring from FPL-native counts, so the reconciliation lives here, at the
# defcon model's only ingestion point. Rows whose position is unknown
# (departed players absent from players_current.csv) are left raw -- they
# are already excluded from both position datasets below.
_FPL_STAT_SCALE = {
    "CBI":   {"def": 1.00, "midfwd": 1.36},  # inverses of 0.975 / 0.736
    "Recov": {"def": 1.32, "midfwd": 1.16},  # inverses of 0.759 / 0.864
}

_fpl_src = matchlogs["Data_Source"] == "FPL_API"
_def_fpl = _fpl_src & matchlogs["IsDefender"]
_mid_fpl = _fpl_src & matchlogs["IsMidfielderOrForward"] & ~matchlogs["IsDefender"]
for _stat, _factors in _FPL_STAT_SCALE.items():
    matchlogs.loc[_def_fpl, _stat] = matchlogs.loc[_def_fpl, _stat] * _factors["def"]
    matchlogs.loc[_mid_fpl, _stat] = matchlogs.loc[_mid_fpl, _stat] * _factors["midfwd"]
print(f"FPL_API stat rescale (onto fbref scale): {int(_def_fpl.sum())} defender rows, "
      f"{int(_mid_fpl.sum())} mid/fwd rows; "
      f"{int((_fpl_src & ~_def_fpl & ~_mid_fpl).sum())} FPL rows without a position left raw")

# Compute raw defensive stats -- CBI is a native column in player_match_stats.csv
# for both sources (summed from fbref components, native for FPL_API), reconciled
# onto one scale above, so no need to sum three separate columns here.
matchlogs["DefStats_Def"] = matchlogs["CBI"] + matchlogs["Tkl"]
matchlogs["DefStats_Mid"] = matchlogs["DefStats_Def"] + matchlogs["Recov"]

# Per-90 normalization
matchlogs["Per90_Def"] = np.where(matchlogs["Minutes"] > 0, matchlogs["DefStats_Def"] * 90 / matchlogs["Minutes"], 0)
matchlogs["Per90_Mid"] = np.where(matchlogs["Minutes"] > 0, matchlogs["DefStats_Mid"] * 90 / matchlogs["Minutes"], 0)

# Season comes from player_match_stats.csv's native Season column (fbref
# 2017-2024, FPL_API 2025 and 2026). The old MatchID//380 derivation is
# stale: it labelled ALL FPL_API rows 2025, which silently folded the
# 2026/27 rounds 1-2 rows (natively Season 2026) into the 2025 validation
# season, and FPL_API MatchIDs aren't one-per-match anyway (the 2025/26
# season's ~11.5k rows share just 38 IDs -- one per gameweek).
matchlogs["Season"] = matchlogs["Season"].astype(int)
MAX_SEASON = int(matchlogs["Season"].max())
RECENCY_DECAY = 0.8  # exponential decay factor per season
# The >= 60 minute bar every per-90 feature in this model uses.
QUALIFYING_MINUTES_P90 = 60
# Cameo filter for the opponent/team/venue per-90 aggregates (CHANGELOG §36).
# DEFCON_AGG_CAMEO=0 restores the unfiltered behaviour.
AGG_CAMEO_FILTERED = os.environ.get("DEFCON_AGG_CAMEO", "1") == "1"

# ── Detailed per-match position from Understat (CHANGELOG §38, AGENDA §4) ──
# The Pos_* multi-hot dummies below are built from this column. Historically
# they came from `Position`, which is detailed for fbref rows (2017-2024) but
# only coarse D/M/F for FPL-era rows (2025+) -- so every FPL-era row, including
# all Split A/B validation rows and every inference vector, had all-zero Pos_*
# dummies while fbref training rows carried real ones (a train/serve split).
# player_match_positions.csv (scripts/fetch/build_position_crosswalk.py) maps
# Understat's per-match formation slot onto the same row identity, giving FPL-era
# rows the detailed slot too. It is used ONLY for the dummies: IsDefender /
# IsMidfielderOrForward / IsForward, the target and the dataset a row joins still
# come from `Position`, so the ON/OFF comparison holds rows and targets fixed and
# changes nothing but the position features. SHIPPED 2026-10-15 (§38): the default
# is now ON -- the OOT test improved 5 of 6 head×split cells on both LL and AUC
# (defender + mid/fwd on both splits, forward on the 29-hit Split A). The lone
# breach was the forward head on the ~2-hit Split B (AUC noise on a micro-sample),
# which the user judged uninformative and overrode the pre-registered guardrail to
# ship. DEFCON_UNDERSTAT_POS=0 restores the pre-§38 behaviour bit-for-bit.
UNDERSTAT_POS = os.environ.get("DEFCON_UNDERSTAT_POS", "1") == "1"
_POSITIONS_FILE = _DATA / "player_match_positions.csv"
if UNDERSTAT_POS and _POSITIONS_FILE.exists():
    _pos_art = pd.read_csv(_POSITIONS_FILE, encoding="utf-8-sig", low_memory=False)
    _pos_art = _pos_art.dropna(subset=["Position_Resolved"])
    _resolved_lookup = {
        (int(s), t, o, v, p): r
        for s, t, o, v, p, r in zip(_pos_art["Season"], _pos_art["Team"],
                                    _pos_art["Opponent"], _pos_art["Venue"],
                                    _pos_art["Player"], _pos_art["Position_Resolved"])
    }
    _keys = list(zip(matchlogs["Season"], matchlogs["Team"], matchlogs["Opponent"],
                     matchlogs["Venue"], matchlogs["Player"]))
    _detailed = pd.Series([_resolved_lookup.get(k) for k in _keys], index=matchlogs.index)
    matchlogs["Position_Detailed"] = _detailed.fillna(matchlogs["Position"])
    _fpl_era = matchlogs["Season"] >= 2025
    _got = int((_fpl_era & matchlogs["Position_Detailed"].notna()).sum())
    print(f"Understat detailed positions ON: {_got}/{int(_fpl_era.sum())} FPL-era rows "
          f"({_got / max(int(_fpl_era.sum()), 1):.1%}) carry a resolved slot for the Pos_* dummies")
elif UNDERSTAT_POS:
    print(f"WARNING: DEFCON_UNDERSTAT_POS=1 but {_POSITIONS_FILE.name} is missing -- "
          f"falling back to coarse Position (run build_position_crosswalk.py).")
    matchlogs["Position_Detailed"] = matchlogs["Position"]
else:
    matchlogs["Position_Detailed"] = matchlogs["Position"]

# ── Pre-compute recency-weighted seasonal aggregates ──────────────────
def compute_season_aggregates(pos_mask, stat_col, per90_col):
    """Pre-compute per-season aggregates for opponent, team, and venue.
    All three now use per90_col -- previously opponent/team used raw stat_col,
    which confounded them with average minutes played (a team whose players
    got subbed off early would show an artificially low TeamAvgStat purely
    from minutes, not true defensive intensity).

    That per-90 switch also inherited per-90's cameo exposure, so the rows are
    now filtered to full appearances (>= 60') like every other per-90 feature:
    a 1-minute substitute with 2 defensive actions extrapolates to 180 per 90,
    and exactly one such row (Latibeaudiere vs Nott'ham Forest) moved Coventry's
    TeamAvgStat from 8.50 to 14.79 -- against a league mean near 8.8 -- which put
    six Coventry defenders in the top six of the defcon chart, one of whom had
    cleared the threshold in 0 of 4 appearances (CHANGELOG §36)."""
    subset = matchlogs[pos_mask]
    if AGG_CAMEO_FILTERED:
        subset = subset[subset["Minutes"] >= QUALIFYING_MINUTES_P90]

    # Per-season aggregates (all per-90, for consistency)
    opp_by_season = subset.groupby(["Season", "Opponent"])[per90_col].mean().reset_index()
    team_by_season = subset.groupby(["Season", "Team"])[per90_col].mean().reset_index()
    venue_by_season = subset.groupby(["Season", "Venue"])[per90_col].mean().reset_index()

    # Build cumulative recency-weighted averages for each season boundary
    # For prediction at season S, use data from seasons 0..S-1
    all_seasons = sorted(subset["Season"].unique())
    prior_opp = {}   # {season: {team: weighted_avg}}
    prior_team = {}
    prior_venue = {}

    for idx, s in enumerate(all_seasons):
        if idx == 0:
            prior_opp[s] = {}
            prior_team[s] = {}
            prior_venue[s] = {}
            continue

        prev_seasons = all_seasons[:idx]

        # Opponent aggregates (recency-weighted, normalized)
        opp_agg = {}
        opp_weights = {}
        for ps in prev_seasons:
            w = RECENCY_DECAY ** (s - 1 - ps)
            for _, row in opp_by_season[opp_by_season["Season"] == ps].iterrows():
                key = row["Opponent"]
                opp_agg[key] = opp_agg.get(key, 0.0) + w * row[per90_col]
                opp_weights[key] = opp_weights.get(key, 0.0) + w
        for key in opp_agg:
            opp_agg[key] /= opp_weights[key]
        prior_opp[s] = opp_agg

        # Team aggregates
        team_agg = {}
        team_weights = {}
        for ps in prev_seasons:
            w = RECENCY_DECAY ** (s - 1 - ps)
            for _, row in team_by_season[team_by_season["Season"] == ps].iterrows():
                key = row["Team"]
                team_agg[key] = team_agg.get(key, 0.0) + w * row[per90_col]
                team_weights[key] = team_weights.get(key, 0.0) + w
        for key in team_agg:
            team_agg[key] /= team_weights[key]
        prior_team[s] = team_agg

        # Venue aggregates
        venue_agg = {}
        venue_weights = {}
        for ps in prev_seasons:
            w = RECENCY_DECAY ** (s - 1 - ps)
            for _, row in venue_by_season[venue_by_season["Season"] == ps].iterrows():
                key = row["Venue"]
                venue_agg[key] = venue_agg.get(key, 0.0) + w * row[per90_col]
                venue_weights[key] = venue_weights.get(key, 0.0) + w
        for key in venue_agg:
            venue_agg[key] /= venue_weights[key]
        prior_venue[s] = venue_agg

    # Final aggregates (all seasons) for prediction-time use
    final_opp = {}
    final_opp_weights = {}
    final_team = {}
    final_team_weights = {}
    final_venue = {}
    final_venue_weights = {}
    for ps in all_seasons:
        w = RECENCY_DECAY ** (MAX_SEASON - ps)
        for _, row in opp_by_season[opp_by_season["Season"] == ps].iterrows():
            key = row["Opponent"]
            final_opp[key] = final_opp.get(key, 0.0) + w * row[per90_col]
            final_opp_weights[key] = final_opp_weights.get(key, 0.0) + w
        for _, row in team_by_season[team_by_season["Season"] == ps].iterrows():
            key = row["Team"]
            final_team[key] = final_team.get(key, 0.0) + w * row[per90_col]
            final_team_weights[key] = final_team_weights.get(key, 0.0) + w
        for _, row in venue_by_season[venue_by_season["Season"] == ps].iterrows():
            key = row["Venue"]
            final_venue[key] = final_venue.get(key, 0.0) + w * row[per90_col]
            final_venue_weights[key] = final_venue_weights.get(key, 0.0) + w
    for key in final_opp:
        final_opp[key] /= final_opp_weights[key]
    for key in final_team:
        final_team[key] /= final_team_weights[key]
    for key in final_venue:
        final_venue[key] /= final_venue_weights[key]

    return prior_opp, prior_team, prior_venue, final_opp, final_team, final_venue

print("Computing temporal opponent/team/venue aggregates...")
def_agg = compute_season_aggregates(matchlogs["IsDefender"], "DefStats_Def", "Per90_Def")
mid_agg = compute_season_aggregates(matchlogs["IsMidfielderOrForward"], "DefStats_Mid", "Per90_Mid")
def_prior_opp, def_prior_team, def_prior_venue, def_final_opp, def_final_team, def_final_venue = def_agg
mid_prior_opp, mid_prior_team, mid_prior_venue, mid_final_opp, mid_final_team, mid_final_venue = mid_agg
# Forward-only aggregates for the forward-specific model: a forward's
# opponent/team/venue context drawn from how forwards perform, not the
# mid-dominated pooled population.
fwd_agg = compute_season_aggregates(matchlogs["IsForward"], "DefStats_Mid", "Per90_Mid")
fwd_prior_opp, fwd_prior_team, fwd_prior_venue, fwd_final_opp, fwd_final_team, fwd_final_venue = fwd_agg

# ── Player career aggregates (pure) + prior std/pct estimators ────────
# Computed BEFORE build_dataset because prior-seeded rows borrow StdPer90 /
# PctAboveThreshold estimates from players with similar in-dataset rates.
# "Pure" = in-dataset only (no prior blending), so the estimator reflects
# actual PL behaviour.

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
# players (CHANGELOG §15). Override via the DEFCON_RAW_CAMEO_MIN env var to
# re-test as data accumulates (0 reproduces the pre-§15 behaviour exactly).
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
RAW_CAMEO_MIN = int(os.environ.get("DEFCON_RAW_CAMEO_MIN", "15"))
RECENT5_CAMEO_LIMIT = 3
RECENT5_MIN_QUALIFYING = 2
# Recent5AvgPer90 was ONE OF TWO per-90 features with no cameo filter (the
# opponent/team/venue aggregates were the other -- see AGG_CAMEO_FILTERED), and it
# suffered exactly the extrapolation failure the comment above describes:
# 40% of defender last-5 windows contain a sub-60' match, and ~84% of
# windows above 14 per 90 were bench cameos rather than form. The model
# learned that correctly and inverted -- raising Recent5AvgPer90 from 10 to
# 14 LOWERED the prediction for 85.5% of served defender rows, while the
# cameo-filtered relationship keeps rising (0.459 -> 0.468 observed, and
# 0.507 -> 0.611 among starters). The feature now uses the same >= 60'
# qualifying bar as every other per-90 feature (CHANGELOG §36).
# DEFCON_RECENT5_CAMEO=0 restores the unfiltered behaviour.
RECENT5_PER90_FILTERED = os.environ.get("DEFCON_RECENT5_CAMEO", "1") == "1"


def avg_raw_qualifying(sum_raw_q, n_raw_q, k=0.0, prior_per90=0.0):
    """AvgRaw over non-cameo matches only, blended with k prior-league
    pseudo-matches at the prior rate. 0.0 when there is no evidence at
    all (no non-cameo match and no prior)."""
    denom = n_raw_q + k
    if denom <= 0:
        return 0.0
    return (sum_raw_q + k * prior_per90) / denom


def recent5_qualifying_raw(df, stat_col):
    """Recent5AvgRaw candidate from an ordered (by MatchID) history frame:
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
    """Recent5AvgPer90 candidate from an ordered (by MatchID) history frame:
    the mean per-90 over the QUALIFYING (>= 60 minute) matches among the
    player's last 5 appearances. Returns None when fewer than
    RECENT5_MIN_QUALIFYING of them qualify -- the caller then falls back to
    the player's career AvgPer90, mirroring how Recent5AvgRaw falls back to
    AvgRaw. Cameo per-90s are extrapolation noise (3 actions in 1 minute ->
    270 per-90) and belong in no per-90 feature."""
    if not RECENT5_PER90_FILTERED:
        return float(df.tail(5)[per90_col].mean())
    last5 = df.tail(5)
    qual = last5[last5["Minutes"] >= QUALIFYING_MINUTES_P90][per90_col]
    if len(qual) < RECENT5_MIN_QUALIFYING:
        return None
    return float(qual.mean())


def precompute_player_features(pos_mask, stat_col, per90_col, threshold, detailed_positions,
                               priors=None, prior_season=None, std_pct_estimator=None):
    """Career-wide player features. With priors=None this is the pure
    in-dataset version (feeds the nearest-neighbour std/pct estimator).
    With priors set, aggregates are blended with each player's transfer
    prior exactly as build_dataset blends per-row history, keeping
    inference vectors consistent with training vectors."""
    subset = matchlogs[pos_mask].sort_values(["Player", "MatchID"])
    grouped = subset.groupby("Player")
    QUALIFYING_MINUTES = 60
    qualifying_grouped = subset[subset["Minutes"] >= QUALIFYING_MINUTES].groupby("Player")

    all_players = grouped.size().index
    raw_qualifying_grouped = subset[subset["Minutes"] >= RAW_CAMEO_MIN].groupby("Player")
    n_rq = raw_qualifying_grouped.size().reindex(all_players).fillna(0).astype(float)
    sum_rq = raw_qualifying_grouped[stat_col].sum().reindex(all_players).fillna(0.0)
    n_q = qualifying_grouped.size().reindex(all_players).fillna(0).astype(float)
    sum_q = qualifying_grouped[per90_col].sum().reindex(all_players).fillna(0.0)
    hits_q = qualifying_grouped[per90_col].apply(lambda x: (x >= threshold).sum()).reindex(all_players).fillna(0).astype(float)
    std_q = qualifying_grouped[per90_col].std().reindex(all_players)
    qual_vals = {p: g.tolist() for p, g in qualifying_grouped[per90_col]}
    recent5_per90_direct = {p: recent5_qualifying_per90(g, per90_col)
                            for p, g in grouped}
    recent5_raw_direct = {p: recent5_qualifying_raw(g, stat_col) for p, g in grouped}
    most_recent_pos = grouped["Position"].last()
    # §38: the Pos_* dummies use the detailed Understat slot when the switch is
    # on (identical to Position when off), so a player who has shifted role is
    # served on their most recent actual formation slot, not coarse D/M/F.
    most_recent_detailed_pos = grouped["Position_Detailed"].last()

    features = {}
    for player in all_players:
        prior = prior_for_season(player, prior_season) if priors is not None else None
        if prior is not None:
            k = prior["matches"]
            prior_per90 = prior["per90"]
            est_std, est_pct = std_pct_estimator(prior_per90)
        else:
            k = prior_per90 = est_std = est_pct = 0.0
        nq, sq, hq = float(n_q[player]), float(sum_q[player]), float(hits_q[player])
        nrq, srq = float(n_rq[player]), float(sum_rq[player])
        sd = std_q[player]
        sd = 0.0 if pd.isna(sd) else float(sd)
        # Prior-blended players pool the prior's k pseudo-matches with their
        # qualifying PL per-90s instead of using the PL-only std (same
        # formula as build_dataset -- see pooled_prior_std).
        if prior is not None:
            std_p90 = pooled_prior_std(est_std, k, prior_per90, qual_vals.get(player, []))
        else:
            std_p90 = sd if nq > 1 else 0.0
        tot = nq + k
        avg_per90_val = (sq + k * prior_per90) / tot if tot > 0 else 0.0
        # Too few non-cameo matches in the last 5 -> fall back to the career
        # rate rather than to a cameo-driven number (same rule as AvgRaw).
        r5p_direct = recent5_per90_direct[player]
        r5p = (avg_per90_val if r5p_direct is None or pd.isna(r5p_direct)
               else float(r5p_direct))
        # AvgRaw counts non-cameo matches only; Recent5AvgRaw falls back to
        # it when the last-5 window is mostly cameos or holds too few
        # non-cameo matches (see the helpers).
        avg_raw_val = avg_raw_qualifying(srq, nrq, k, prior_per90)
        r5r_direct = recent5_raw_direct[player]
        r5r = r5r_direct if r5r_direct is not None else avg_raw_val
        features[player] = {
            "AvgPer90": avg_per90_val,
            "AvgRaw": avg_raw_val,
            "StdPer90": std_p90,
            "MatchesPlayed": tot,
            "Recent5AvgPer90": r5p, "Recent5AvgRaw": r5r,
            "PctAboveThreshold": (hq + k * est_pct) / tot if tot > 0 else 0.0,
            "Is_FPL_API": 1,  # inference: all future fixtures are FPL_API data
            **get_pos_dummies(most_recent_detailed_pos[player], detailed_positions),
        }
    return features


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


def_player_feats_pure = precompute_player_features(matchlogs["IsDefender"], "DefStats_Def", "Per90_Def", 10, DEFENDER_DETAILED_POS)
mid_player_feats_pure = precompute_player_features(matchlogs["IsMidfielderOrForward"], "DefStats_Mid", "Per90_Mid", 12, MID_FWD_DETAILED_POS)
def_std_estimator = estimate_std_and_pct(def_player_feats_pure)
mid_std_estimator = estimate_std_and_pct(mid_player_feats_pure)
fwd_player_feats_pure = precompute_player_features(matchlogs["IsForward"], "DefStats_Mid", "Per90_Mid", 12, MID_FWD_DETAILED_POS)
fwd_std_estimator = estimate_std_and_pct(fwd_player_feats_pure)

# ── Build player-match feature dataset ────────────────────────────────
def build_dataset(df, pos_cols, stat_col, per90_col, threshold, prior_opp, prior_team, prior_venue, detailed_positions,
                  use_priors=True, std_pct_estimator=None):
    """Build labeled dataset with temporal opponent/team/venue features.

    With use_priors=True, rows from a player's prior-league arrival season
    onward blend their (possibly empty) league history with the transfer
    prior as k pseudo-matches at the prior-league rate -- the same
    empirical-Bayes shrinkage the inference tiers apply, keeping training
    vectors and serving vectors consistent. StdPer90 for prior-carrying
    rows is the pooled std of the combined sample (see pooled_prior_std),
    not the PL-only std. AvgRaw/Recent5AvgRaw count only non-cameo matches
    (Minutes >= RAW_CAMEO_MIN; see avg_raw_qualifying /
    recent5_qualifying_raw)."""
    mask = df[pos_cols[0]].copy()
    for col in pos_cols[1:]:
        mask = mask | df[col]
    subset = df[mask & (df["Minutes"] >= 1)].copy()
    subset = subset.sort_values(["Player", "MatchID"]).reset_index(drop=True)

    records = []
    for player, grp in subset.groupby("Player"):
        grp = grp.sort_values("MatchID").reset_index(drop=True)

        for i in range(len(grp)):
            row = grp.iloc[i]
            per90_stat = row[per90_col]
            raw_stat = row[stat_col]
            # A match only counts as a genuine "hit" if the player played at
            # least 60 minutes -- otherwise a couple of actions in a brief
            # cameo (e.g. 2 actions in 13 minutes) gets extrapolated via
            # per-90 math into a false threshold-crossing that wouldn't hold
            # up over a full match. This applies both to the training target
            # and to the PctAboveThreshold feature below.
            QUALIFYING_MINUTES = 60
            target = 1 if (per90_stat >= threshold and row["Minutes"] >= QUALIFYING_MINUTES) else 0
            row_season = int(row["Season"])

            # Player-level historical features (strict temporal windowing).
            # Transfer priors: from a player's prior-league arrival season
            # onward, league history is blended with the prior as k
            # pseudo-matches at the prior-league rate (k = Min_Old / 90), so
            # a proven arrival's debut row carries real evidence instead of
            # an all-zero vector indistinguishable from an academy unknown.
            prev = grp.iloc[:i]
            prior = prior_for_season(player, row_season) if use_priors else None
            if prior is not None:
                k = prior["matches"]
                prior_per90 = prior["per90"]
                est_std, est_pct = std_pct_estimator(prior_per90)
            if len(prev) == 0:
                if prior is not None:
                    avg_per90 = avg_raw = recent_avg_per90 = recent_avg_raw = prior_per90
                    # n_q = 0: the pooled std IS est_std -- the combined
                    # sample is just the k pseudo-matches.
                    std_per90 = pooled_prior_std(est_std, k, prior_per90, [])
                    matches_played = k
                    pct_above = est_pct
                else:
                    avg_per90 = avg_raw = std_per90 = 0
                    matches_played = 0
                    recent_avg_per90 = recent_avg_raw = pct_above = 0
            else:
                # AvgPer90/StdPer90 use only qualifying (>=60 min) matches --
                # same fix as PctAboveThreshold, extended here. A simple mean of
                # per-match per-90 rates is vulnerable to short-cameo extrapolation
                # noise (e.g. 1 minute played, 3 actions -> extrapolated to 270
                # per-90), which can dominate the average even though it
                # represents a tiny fraction of true playing time. AvgRaw and
                # Recent5AvgRaw have the OPPOSITE cameo failure: raw per-match
                # volume is proportional to minutes, so a cameo's near-zero raw
                # total drags a per-match raw mean DOWNWARD. They therefore use
                # only non-cameo matches (Minutes >= RAW_CAMEO_MIN), and a
                # last-5 window that is mostly cameos (or too short to hold
                # >= 2 non-cameo matches) falls back to the cameo-filtered
                # AvgRaw instead of collapsing to a zero- or
                # one-match "recent" average.
                prev_qualifying = prev[prev["Minutes"] >= QUALIFYING_MINUTES]
                prev_raw_qualifying = prev[prev["Minutes"] >= RAW_CAMEO_MIN]
                recent5_per90_direct = recent5_qualifying_per90(prev, per90_col)
                recent5_raw_direct = recent5_qualifying_raw(prev, stat_col)
                if prior is not None:
                    # Empirical-Bayes shrinkage toward the prior-league rate,
                    # in exactly the form the inference tiers use.
                    n_q = len(prev_qualifying)
                    sum_q = prev_qualifying[per90_col].sum() if n_q > 0 else 0.0
                    hits_q = (prev_qualifying[per90_col] >= threshold).sum() if n_q > 0 else 0
                    matches_played = n_q + k
                    avg_per90 = (sum_q + k * prior_per90) / matches_played
                    avg_raw = avg_raw_qualifying(prev_raw_qualifying[stat_col].sum(),
                                                 len(prev_raw_qualifying), k, prior_per90)
                    pct_above = (hits_q + k * est_pct) / matches_played
                    # Pooled std of the prior's k pseudo-matches plus the
                    # qualifying PL per-90s; n_q = 0/1 and degenerate
                    # small-k cases are handled inside pooled_prior_std.
                    std_per90 = pooled_prior_std(est_std, k, prior_per90,
                                                 prev_qualifying[per90_col].tolist())
                else:
                    avg_per90 = prev_qualifying[per90_col].mean() if len(prev_qualifying) > 0 else 0
                    avg_per90 = 0 if pd.isna(avg_per90) else avg_per90
                    avg_raw = avg_raw_qualifying(prev_raw_qualifying[stat_col].sum(),
                                                 len(prev_raw_qualifying))
                    std_per90 = prev_qualifying[per90_col].std() if len(prev_qualifying) > 1 else 0
                    std_per90 = 0 if pd.isna(std_per90) else std_per90
                    matches_played = len(prev_qualifying)
                    pct_above = (prev_qualifying[per90_col] >= threshold).mean() if len(prev_qualifying) > 0 else 0
                # Recent5AvgRaw: mean over the non-cameo matches among the
                # last 5; a mostly-cameo window or one with too few
                # non-cameo matches falls back to the cameo-filtered AvgRaw.
                # Recent5AvgPer90 follows the same rule against the >= 60'
                # qualifying bar, falling back to AvgPer90.
                recent_avg_raw = recent5_raw_direct if recent5_raw_direct is not None else avg_raw
                recent_avg_per90 = (recent5_per90_direct
                                    if recent5_per90_direct is not None
                                    else avg_per90)

            # Temporal opponent/team/venue features (no data leak)
            lookup_season = row_season  # use prior_opp[season] = data before this season
            opp_lookup = prior_opp.get(lookup_season, {})
            team_lookup = prior_team.get(lookup_season, {})
            venue_lookup = prior_venue.get(lookup_season, {})

            records.append({
                "MatchID": row["MatchID"], "Player": player,
                "Team": row["Team"], "Opponent": row["Opponent"],
                "Venue": row["Venue"], "Minutes": row["Minutes"],
                "Season": row_season,
                "RawStat": raw_stat, "Per90Stat": per90_stat, "Target": target,
                "AvgPer90": avg_per90, "AvgRaw": avg_raw, "StdPer90": std_per90,
                "MatchesPlayed": matches_played,
                "Recent5AvgPer90": recent_avg_per90, "Recent5AvgRaw": recent_avg_raw,
                "PctAboveThreshold": pct_above,
                "OpponentAvgStat": opp_lookup.get(row["Opponent"], avg_per90),
                "TeamAvgStat": team_lookup.get(row["Team"], avg_per90),
                "VenueAvg": venue_lookup.get(row["Venue"], avg_per90),
                "IsHome": 1 if row["Venue"] == "Home" else 0,
                "Is_FPL_API": 1 if row["Data_Source"] == "FPL_API" else 0,
                # Row-level forward flag (NOT a feature -- absent from every
                # FEATURE_COLS list): lets the shared mid/fwd dataset be
                # sliced into its forward rows for the forward-model
                # head-to-head evaluation.
                "IsForward": 1 if is_forward_pos(row["Position"]) else 0,
                **get_pos_dummies(row["Position_Detailed"], detailed_positions),
            })

    return pd.DataFrame(records)

print("Building defender dataset...")
df_def = build_dataset(matchlogs, ["IsDefender"], "DefStats_Def", "Per90_Def", 10,
                       def_prior_opp, def_prior_team, def_prior_venue, DEFENDER_DETAILED_POS,
                       std_pct_estimator=def_std_estimator)
print(f"  Defender records: {len(df_def)}")

print("Building midfielder+forward dataset...")
df_mid = build_dataset(matchlogs, ["IsMidfielderOrForward"], "DefStats_Mid", "Per90_Mid", 12,
                       mid_prior_opp, mid_prior_team, mid_prior_venue, MID_FWD_DETAILED_POS,
                       std_pct_estimator=mid_std_estimator)
print(f"  Midfielder+Forward records: {len(df_mid)}")

# Forward-specific dataset: same target definition (CBI+Tkl+Recov per-90
# >= 12 with >= 60 min) but forward rows only -- the shared mid/fwd model
# can separate forwards from mids only through their stats and Is_FPL_API
# (coarse F/M positions match no detailed position dummy), which is the
# suspected mechanism behind the forward over-prediction (artifact mean
# forward probability 0.0602 vs 0.0282 observed on the model-label scale,
# 2026-09-02 probe).
print("Building forward dataset...")
df_fwd = build_dataset(matchlogs, ["IsForward"], "DefStats_Mid", "Per90_Mid", 12,
                       fwd_prior_opp, fwd_prior_team, fwd_prior_venue, MID_FWD_DETAILED_POS,
                       std_pct_estimator=fwd_std_estimator)
print(f"  Forward records: {len(df_fwd)}")

print(f"\nDefender target rate: {df_def['Target'].mean():.3f}")
print(f"Midfielder+Forward target rate: {df_mid['Target'].mean():.3f}")
print(f"Forward target rate: {df_fwd['Target'].mean():.3f}")

BASE_FEATURE_COLS = [
    "AvgPer90", "AvgRaw", "StdPer90", "MatchesPlayed",
    "Recent5AvgPer90", "Recent5AvgRaw", "PctAboveThreshold",
    "OpponentAvgStat", "TeamAvgStat", "VenueAvg", "IsHome", "Is_FPL_API"
]
DEF_FEATURE_COLS = BASE_FEATURE_COLS + [f"Pos_{c}" for c in DEFENDER_DETAILED_POS]
MID_FEATURE_COLS = BASE_FEATURE_COLS + [f"Pos_{c}" for c in MID_FWD_DETAILED_POS]
# The forward model shares the mid/fwd schema (same 10 detailed-position
# dummies -- within the forward dataset they separate pure-FW rows from
# hybrid W/FW or AM/FW rows).
FWD_FEATURE_COLS = MID_FEATURE_COLS

# ── Out-of-time validation ────────────────────────────────────────────
# Train on all seasons except the most recent; evaluate on the most recent.
# This mirrors real-world deployment (train on history, predict the future)
# and gives an honest estimate of generalisation performance.

def stratified_reliability(proba, y, matches_played, label, n_bins=5, min_cell=25):
    """Reliability check stratified by MatchesPlayed (individual-evidence strength).

    Aggregate ECE can hide evidence-dependent miscalibration: a low-evidence
    player pushed up toward the mean and a high-evidence player pushed down
    toward the mean land in the same predicted-probability bin and cancel
    each other out. This checks calibration, prediction spread, and
    discrimination WITHIN evidence buckets, cross-tabulates observed rates
    by (probability bin x evidence bucket), and formally tests
    (likelihood-ratio) whether P(target | prediction) depends on
    MatchesPlayed. The compression signature is observed rates RISING with
    MatchesPlayed in the top probability bins and FALLING in the bottom
    bins -- i.e. the same prediction meaning different things depending on
    how much individual evidence backs it."""
    p = np.asarray(proba, dtype=float)
    yv = np.asarray(y)
    mp = np.asarray(matches_played, dtype=float)

    bucket_labels = ["0-9", "10-29", "30-79", "80-149", "150+"]
    buckets = pd.cut(mp, bins=[-0.5, 9.5, 29.5, 79.5, 149.5, np.inf], labels=bucket_labels)
    bucket_masks = {lab: np.asarray(buckets == lab) for lab in bucket_labels}
    active = [lab for lab in bucket_labels if bucket_masks[lab].sum() > 0]

    # Per-bucket calibration, prediction spread, within-bucket discrimination.
    # A model that properly trusts individual evidence should show BOTH:
    # prediction spread (p_std) growing with MatchesPlayed, and within-bucket
    # AUC growing with MatchesPlayed (more evidence -> more discrimination).
    summary = {"buckets": {}, "lrt": None}
    print(f"\n  Reliability by MatchesPlayed bucket:")
    print(f"  {'bucket':<10s} {'n':>6s} {'obs':>6s} {'pred':>6s} {'gap':>6s} {'p_std':>6s} {'AUC':>6s} {'Brier':>7s}")
    for lab in active:
        m = bucket_masks[lab]
        obs, pred = yv[m].mean(), p[m].mean()
        summary["buckets"][lab] = {"n": int(m.sum()), "obs": float(obs), "pred": float(pred)}
        try:
            auc = roc_auc_score(yv[m], p[m])
        except ValueError:
            auc = float("nan")
        print(f"  {lab:<10s} {m.sum():>6d} {obs:>6.3f} {pred:>6.3f} {pred - obs:>+6.3f} "
              f"{p[m].std():>6.3f} {auc:>6.3f} {brier_score_loss(yv[m], p[m]):>7.3f}")

    # Observed rate by (probability bin x evidence bucket): the compression
    # signature lives HERE, not in the aggregate numbers above. If the model
    # under-trusts strong evidence, high-evidence players inside a given
    # probability bin will hit MORE often than low-evidence players in the
    # same bin (and vice versa in the bottom bins).
    interior = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if len(interior) > 0:
        pbin = np.digitize(p, interior)
        nb = len(interior) + 1
        print(f"\n  Observed target rate by (predicted-probability bin x MatchesPlayed bucket):")
        hdr = f"  {'prob bin':<18s} {'n':>5s} {'pred':>6s} {'all':>6s}"
        hdr += "".join(f" {lab:>9s}" for lab in active)
        print(hdr)
        bin_ranges = {}
        for b in range(nb):
            m = pbin == b
            if m.sum() == 0:
                continue
            lo = interior[b - 1] if b > 0 else p.min()
            hi = interior[b] if b < len(interior) else p.max()
            bin_ranges[b] = (lo, hi)
            row = f"  [{lo:.3f}, {hi:.3f})".ljust(19) + f"{m.sum():>5d} {p[m].mean():>6.3f} {yv[m].mean():>6.3f}"
            for lab in active:
                mm = m & bucket_masks[lab]
                row += f" {yv[mm].mean():>9.3f}" if mm.sum() >= min_cell else f" {'--':>9s}"
            print(row)

        # Direct compression test: within each probability bin, compare
        # observed rates for lowest- vs highest-evidence players. Same bin,
        # different evidence -- a consistent nonzero gap means predictions
        # carry different meaning depending on sample size.
        lo_lab, hi_lab = active[0], active[-1]
        print(f"\n  Within-bin evidence gaps (observed rate, {hi_lab} minus {lo_lab} MatchesPlayed):")
        for b, (lo, hi) in bin_ranges.items():
            m = pbin == b
            lo_m = m & bucket_masks[lo_lab]
            hi_m = m & bucket_masks[hi_lab]
            if lo_m.sum() < min_cell or hi_m.sum() < min_cell:
                continue
            gap = yv[hi_m].mean() - yv[lo_m].mean()
            print(f"    [{lo:.3f}, {hi:.3f}): {yv[hi_m].mean():.3f} (n={hi_m.sum():>4d}) vs "
                  f"{yv[lo_m].mean():.3f} (n={lo_m.sum():>4d}) -> gap {gap:+.3f}")

    # Formal omnibus test: LRT of y ~ logit(p) against y ~ logit(p) * bucket.
    # (Caveat: rows cluster within players and matches, so the p-value is
    # anti-conservative -- treat a small p as a flag, not proof.)
    # Skipped when y has only one class (e.g. forward 2026 with 0 positives)
    # — logistic regression requires at least 2 classes.
    if len(active) > 1 and len(np.unique(yv)) > 1:
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        lp = np.log(pc / (1 - pc))
        lp = (lp - lp.mean()) / (lp.std() + 1e-12)
        dummies = np.column_stack([bucket_masks[lab].astype(float) for lab in active])
        X_red = np.column_stack([np.ones(len(p)), lp])
        extra = [dummies[:, j] for j in range(1, len(active))]
        extra += [lp * dummies[:, j] for j in range(1, len(active))]
        X_full = np.column_stack([X_red] + extra)

        def _loglik(X):
            clf = LogisticRegression(C=1e8, max_iter=5000)
            clf.fit(X, yv)
            ph = np.clip(clf.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            return float(np.sum(yv * np.log(ph) + (1 - yv) * np.log(1 - ph)))

        lrt = 2.0 * (_loglik(X_full) - _loglik(X_red))
        dof = X_full.shape[1] - X_red.shape[1]
        p_val = float(1.0 - chi2.cdf(lrt, dof))
        print(f"\n  LRT (calibration independent of MatchesPlayed): chi2={lrt:.2f}, df={dof}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  -> P(target | pred) differs by evidence level: evidence-dependent bias present.")
        else:
            print("  -> No detectable evidence-dependent miscalibration at the 5% level.")
        summary["lrt"] = {"chi2": lrt, "dof": dof, "p": p_val}
    return summary


def _report_metrics(proba, y_val, X_val, feature_cols, label):
    """Metrics + MatchesPlayed-stratified reliability for a scored validation
    frame -- shared by the freshly-trained and already-trained (ablation)
    validation paths. Returns the stratified-reliability summary, if any."""
    ll = log_loss(y_val, proba, labels=[0, 1])
    # AUC is undefined when y_true has only one class (e.g. forward 2026
    # with 0 positives among a tiny validation set); guard with NaN.
    try:
        auc = roc_auc_score(y_val, proba)
    except ValueError:
        auc = float("nan")
    brier = brier_score_loss(y_val, proba)

    # Expected calibration error (10 equal-width bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (proba >= bin_edges[b]) & (proba < bin_edges[b + 1])
        if b == n_bins - 1:
            mask = mask | (proba == bin_edges[b + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_val[mask].mean() - proba[mask].mean())
    ece /= len(y_val)

    print(f"\n{'=' * 60}")
    print(f"OUT-OF-TIME VALIDATION: {label}")
    print(f"{'=' * 60}")
    print(f"  Val: {len(y_val)} records | target rate: {y_val.mean():.3f}")
    print(f"  Log loss:  {ll:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  ECE:       {ece:.4f}")

    # Stratified reliability check -- aggregate ECE can hide compression
    # (a low-evidence player pushed up toward the mean and a high-evidence
    # player pushed down toward the mean land in the same probability bin
    # and cancel out), so check calibration within MatchesPlayed buckets.
    summary = None
    if "MatchesPlayed" in feature_cols:
        mp_idx = feature_cols.index("MatchesPlayed")
        summary = stratified_reliability(proba, y_val, X_val[:, mp_idx], label)
    return summary


# ── Hyperparameter grids and model configs ──────────────────────────
# Tuning grids for RandomizedSearchCV. Defaults below are the locked-in
# §19 production picks (2026-09-03 tuned run). DEFCON_TUNE=0 is the
# weekly-run default since 2026-09-04; set DEFCON_TUNE=1 to re-run the
# full ~5h search (9 tunings per run, 30 candidates × 5 folds each).
GBM_PARAM_GRID = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.1, 0.15],
    "subsample": [0.7, 0.8, 0.9, 1.0],
}
RF_PARAM_GRID = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [6, 8, 10, None],
    "min_samples_leaf": [1, 3, 5],
    "max_features": ["sqrt", "log2", None],
}
_MODEL_CONFIGS = {
    # Defender: production-frame tuned pick (2026-09-03, §19).
    "GBM": (GradientBoostingClassifier, GBM_PARAM_GRID,
            {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
             "subsample": 0.7}),
    # Shared mid/forward: identical pick in all three §19 tuning calls
    # (Split A, Split B, production). Serves mids always; forwards when
    # FORWARD_MODEL_SHIPPED is off.
    "RF_MF": (RandomForestClassifier, RF_PARAM_GRID,
              {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 3,
               "max_features": "sqrt", "n_jobs": 1}),
    # Forward-only: production-frame tuned pick. Different optimum from
    # RF_MF because the forward dataset is ~4x smaller with a ~4x lower
    # target rate. Serves forwards when FORWARD_MODEL_SHIPPED is on.
    "RF_FWD": (RandomForestClassifier, RF_PARAM_GRID,
               {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5,
                "max_features": "log2", "n_jobs": 1}),
}
_DO_TUNE = os.environ.get("DEFCON_TUNE", "0") == "1"

# Validated replacement for the defender head (CHANGELOG §41). Increasing
# activity evidence cannot reduce P(hit), holding all other features fixed.
# The candidate's configuration is locked; DEFCON_TUNE does not search it.
_MODEL_CONFIGS["HGB_MONOTONIC"] = (
    HistGradientBoostingClassifier, {}, monotonic_params(DEF_FEATURE_COLS)
)
_DEFENDER_MODEL_NAME = (
    "HGB_MONOTONIC" if os.environ.get("DEFCON_MONOTONIC_DEF", "1") == "1" else "GBM"
)


# ── Calibration wrapper ──────────────────────────────────────────────
class _CalibratedModel:
    """Base model + standalone isotonic calibrator.
    predict_proba applies scaler (if set) then calibrates the raw output.
    Replaces CalibratedClassifierCV with an explicit cross_val_predict
    pipeline so calibration is a single isotonic fit on OOF predictions
    from the *tuned* model, not an ensemble of 5 internally-refitted
    copies."""
    def __init__(self, base_model, calibrator, scaler=None):
        self.base_model = base_model
        self.calibrator = calibrator
        self.scaler = scaler

    def predict_proba(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1 - cal, cal])


# ── Trained-model cache + parallel RF fits ──────────────────────────
# Weekly runs retrained all 9 models even when nothing that determines
# them had changed. tune_and_calibrate now consults a disk cache keyed
# on a content hash of EVERYTHING that determines the artifact:
#   * md5 of predict.py itself -- any code edit invalidates the cache
#   * sklearn/numpy versions
#   * model class, label, tuning path (do_tune), default params, and
#     the random_state/cv/n_iter call arguments
#   * the training frame itself (X and y bytes) -- NOT the input-file
#     md5s: fixtures.csv, minutes_probabilities.csv and match_results.csv
#     change weekly but feed only the prediction loops, never training
#     rows, so keying on them would force pointless retrains; conversely
#     any edit to player_match_stats.csv / players_current.csv /
#     transfers.csv that changes a frame's contents changes X and
#     invalidates naturally. Input-file md5s are still written to
#     input_fingerprints.json each run for diagnostics ("why did the
#     cache miss?").
# The payload stores base model + calibrator + scaler + best_params as
# plain sklearn objects (not the _CalibratedModel wrapper, which lives
# in __main__ and would be unpicklable elsewhere). Stale entries from
# earlier weeks are pruned once all nine trainings are done.
# DEFCON_MODEL_CACHE=0 disables the cache entirely (forced retrain);
# DEFCON_RF_N_JOBS overrides the RF fit parallelism (default -1 = all
# cores; with random_state fixed, n_jobs is result-invariant).
_CACHE_ENABLED = os.environ.get("DEFCON_MODEL_CACHE", "1") == "1"
_CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "defcon"
_PREDICT_SOURCE_MD5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()
try:
    _RF_N_JOBS = int(os.environ.get("DEFCON_RF_N_JOBS", "-1"))
except ValueError:
    _RF_N_JOBS = -1
_CACHE_USED_KEYS = set()
_CACHE_INITIALISED = False


def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _initialise_model_cache():
    """Create the cache dir and record input-file md5s once per run
    (diagnostics only -- cache keys are content-based, see above)."""
    global _CACHE_INITIALISED
    if _CACHE_INITIALISED or not _CACHE_ENABLED:
        return
    _CACHE_INITIALISED = True
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprints = {name: _md5_file(_DATA / name)
                    for name in ("player_match_stats.csv", "players_current.csv",
                                 "transfers.csv", "fixtures.csv",
                                 "minutes_probabilities.csv", "match_results.csv")
                    if (_DATA / name).exists()}
    payload = {"written": time.strftime("%Y-%m-%d %H:%M:%S"),
               "predict_source_md5": _PREDICT_SOURCE_MD5,
               "files": fingerprints}
    tmp = _CACHE_DIR / "input_fingerprints.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, _CACHE_DIR / "input_fingerprints.json")


def _model_cache_key(X, y, model_cls, label, do_tune, default_params,
                     random_state, cv, n_iter):
    """md5 over everything that determines the trained artifact."""
    h = hashlib.md5()
    h.update(_PREDICT_SOURCE_MD5.encode())
    h.update(f"sklearn={sklearn.__version__};numpy={np.__version__}".encode())
    h.update(f"{model_cls.__name__}|{label}|tune={int(bool(do_tune))}".encode())
    h.update(repr(sorted((default_params or {}).items())).encode())
    h.update(f"rs={random_state}|cv={cv}|n_iter={n_iter}".encode())
    h.update(f"X{X.shape}|{X.dtype}|".encode())
    h.update(X.tobytes())
    h.update(f"y{y.shape}|{y.dtype}|".encode())
    h.update(y.tobytes())
    return h.hexdigest()


def _cache_path_for(label, key):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
    return _CACHE_DIR / f"{slug}__{key}.pkl.gz"


def _load_cached_model(path, label):
    """Return (calibrated, scaler, best_params) from the cache, or None
    on any failure (corrupt entry, unpicklable payload) -- cache
    problems must never break a run, only cost a retrain."""
    try:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        return (_CalibratedModel(payload["base_model"], payload["calibrator"],
                                 payload["scaler"]),
                payload["scaler"], payload["best_params"])
    except Exception as exc:
        print(f"  [{label}] cache entry unreadable ({exc}) -- retraining")
        try:
            path.unlink()
        except OSError:
            pass
        return None


def _store_cached_model(path, label, production, calibrator, scaler,
                        best_params):
    """Persist a freshly trained artifact atomically (tmp + rename) so a
    crash mid-write can never leave a corrupt cache entry behind."""
    payload = {"base_model": production, "calibrator": calibrator,
               "scaler": scaler, "best_params": best_params}
    tmp = path.with_name(path.name + ".tmp")
    try:
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)
    except Exception as exc:
        print(f"  [{label}] cache write failed ({exc}) -- continuing uncached")
        try:
            tmp.unlink()
        except OSError:
            pass


def _prune_model_cache():
    """Delete entries whose keys the current run did not produce: last
    week's training frames are strictly superseded, so their cached
    models can never be hit again (and the cache sits in OneDrive-synced
    storage, so dead weight is real sync churn). Runs after the last of
    the nine trainings."""
    if not (_CACHE_ENABLED and _CACHE_DIR.exists()):
        return
    kept = removed = 0
    freed = 0
    for f in sorted(_CACHE_DIR.glob("*.pkl.gz")):
        if f.name in _CACHE_USED_KEYS:
            kept += 1
        else:
            freed += f.stat().st_size
            f.unlink()
            removed += 1
    for f in sorted(_CACHE_DIR.glob("*.tmp")):
        f.unlink()
    print(f"Model cache: {kept} entr{'y' if kept == 1 else 'ies'} kept, "
          f"{removed} stale pruned ({freed / 1e6:.1f} MB freed)")


def tune_and_calibrate(X, y, model_cls, param_grid, label, cv=5,
                       random_state=42, n_iter=30, do_tune=True,
                       default_params=None):
    """Tune hyperparameters via RandomizedSearchCV, then calibrate the
    tuned model with cross_val_predict + isotonic regression.

    Steps:
      1. Tune: RandomizedSearchCV(n_iter, cv, scoring=neg_log_loss) to
         find best hyperparameters on the training data.
      2. Calibrate: cross_val_predict with the same fold structure to
         generate out-of-fold probabilities; fit IsotonicRegression on
         OOF p vs y.
      3. Final model: train the tuned model on ALL training data; attach
         the calibrator for inference.

    Returns (_CalibratedModel, scaler, best_params).
    The _CalibratedModel wraps scaler + base_model + calibrator so
    predict_proba accepts raw (unscaled) feature arrays.

    Results are served from the model cache when the content key
    matches a stored artifact (see the model-cache section above); RF
    fits run with n_jobs=_RF_N_JOBS, which with random_state fixed is
    result-invariant and purely faster."""
    _initialise_model_cache()
    cache_path = None
    if _CACHE_ENABLED:
        key = _model_cache_key(X, y, model_cls, label, do_tune,
                               default_params, random_state, cv, n_iter)
        cache_path = _cache_path_for(label, key)
        if cache_path.exists():
            cached = _load_cached_model(cache_path, label)
            if cached is not None:
                _CACHE_USED_KEYS.add(cache_path.name)
                print(f"  [{label}] cache hit -- skipping training "
                      f"(key {key[:12]})")
                return cached
    t_start = time.time()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Step 1: Tune hyperparameters
    if do_tune and param_grid:
        search = RandomizedSearchCV(
            model_cls(random_state=random_state),
            param_grid, n_iter=n_iter, cv=skf,
            scoring="neg_log_loss", random_state=random_state, n_jobs=4,
        )
        search.fit(X_scaled, y)
        best_params = search.best_params_
        print(f"  [{label}] tuning best: {best_params}")
    else:
        best_params = default_params or {}
        print(f"  [{label}] skipping tuning (using defaults)")

    # RF fits parallelise across trees (result-invariant with the fixed
    # random_state). The search estimator above stays single-threaded so
    # its n_jobs=4 candidate parallelism is not oversubscribed.
    def _make(params):
        p = dict(params)
        p["random_state"] = random_state
        if model_cls is RandomForestClassifier:
            p["n_jobs"] = _RF_N_JOBS
        return model_cls(**p)

    # Step 2: OOF calibration on the tuned model
    base_for_cv = _make(best_params)
    oof_proba = cross_val_predict(
        base_for_cv, X_scaled, y, cv=skf, method="predict_proba",
    )
    oof_p = np.clip(oof_proba[:, 1], 1e-6, 1 - 1e-6)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof_p, y)

    # Step 3: Train final production model on ALL training data
    production = _make(best_params)
    production.fit(X_scaled, y)

    print(f"  [{label}] trained on {len(y)} records")
    if cache_path is not None:
        _store_cached_model(cache_path, label, production, calibrator,
                            scaler, best_params)
        _CACHE_USED_KEYS.add(cache_path.name)
        print(f"  [{label}] cached for next run (key {key[:12]})")
    print(f"  [{label}] training wall time {time.time() - t_start:.1f}s")
    return _CalibratedModel(production, calibrator, scaler), scaler, best_params


def evaluate_oot(X_train, y_train, X_val, y_val, feature_cols, label, model_name):
    """Train on earlier seasons, evaluate on the most recent season.
    Reports log loss, AUC, Brier score, expected calibration error, and a
    MatchesPlayed-stratified reliability check (evidence-dependent
    calibration -- i.e. whether compression toward the mean is really gone,
    not just invisible in the aggregate ECE). Returns (calibrated model,
    scaler, reliability summary) so ablation arms can reuse the identically
    trained model."""
    model_cls, param_grid, default_params = _MODEL_CONFIGS[model_name]
    calibrated, scaler, _best_params = tune_and_calibrate(
        X_train, y_train, model_cls, param_grid, label,
        do_tune=_DO_TUNE, default_params=default_params,
    )

    # Wrapper handles scaling internally; pass raw validation features.
    proba = calibrated.predict_proba(X_val)[:, 1]
    summary = _report_metrics(proba, y_val, X_val, feature_cols, label)

    # Feature importances from the production base model
    base = calibrated.base_model
    if hasattr(base, 'feature_importances_'):
        imp = base.feature_importances_
        fi = sorted(zip(feature_cols, imp), key=lambda x: -x[1])
        print(f"\n  Top 10 features:")
        for name, w in fi[:10]:
            print(f"    {name:<24s} {w:.4f} ({w / sum(imp) * 100:.1f}%)")
    print()
    return calibrated, scaler, summary


def score_validation(calibrated, X_val, y_val, feature_cols, label):
    """Score a validation frame with an already-trained model -- used for
    the priors-ON vs priors-OFF ablation, where training rows are identical
    under both configs and only the validation features differ.
    The _CalibratedModel wrapper handles scaling internally."""
    proba = calibrated.predict_proba(X_val)[:, 1]
    return _report_metrics(proba, y_val, X_val, feature_cols, label)


print("\n" + "=" * 60)
print("OUT-OF-TIME VALIDATION")
print("=" * 60)

# Two splits, two questions:
#
# A) Last COMPLETE season (Season 2025): full-powered calibration reference.
#    No transfer priors exist for pre-2026 arrivals, so prior blending cannot
#    move anything here -- this doubles as the no-prior control: the newcomer
#    (0-9 MatchesPlayed) over-prediction should persist in this split.
#
# B) Current season (Season 2026, rounds 1-2 so far): the only season where
#    Defcon priors exist, hence the only place the prior mechanism can act.
#    Scored with priors ON (production config) and priors OFF (ablation)
#    using the SAME trained model -- training rows are identical under both
#    configs (no priors exist for seasons <= 2025), so any difference is
#    purely the prior-seeded validation features. Falsifiable prediction if
#    missing prior evidence really is the mechanism: the 0-9 bucket gap and
#    the LRT shrink from OFF to ON. Caveat: only ~2 rounds of data, so this
#    split is low-powered by construction.
_SEASON_CURRENT = int(df_def["Season"].max())   # 2026 -- partial season
_SEASON_LAST_FULL = _SEASON_CURRENT - 1         # 2025 -- last complete season

print(f"\n--- Split A: train <{_SEASON_LAST_FULL}, validate Season {_SEASON_LAST_FULL} "
      f"(full-season reference / no-prior control) ---")
evaluate_oot(
    df_def[df_def["Season"] < _SEASON_LAST_FULL][DEF_FEATURE_COLS].values,
    df_def[df_def["Season"] < _SEASON_LAST_FULL]["Target"].values,
    df_def[df_def["Season"] == _SEASON_LAST_FULL][DEF_FEATURE_COLS].values,
    df_def[df_def["Season"] == _SEASON_LAST_FULL]["Target"].values,
    DEF_FEATURE_COLS, f"Defender (Season {_SEASON_LAST_FULL})", _DEFENDER_MODEL_NAME,
)
mid_model_a, mid_scaler_a, _sum_mid_a = evaluate_oot(
    df_mid[df_mid["Season"] < _SEASON_LAST_FULL][MID_FEATURE_COLS].values,
    df_mid[df_mid["Season"] < _SEASON_LAST_FULL]["Target"].values,
    df_mid[df_mid["Season"] == _SEASON_LAST_FULL][MID_FEATURE_COLS].values,
    df_mid[df_mid["Season"] == _SEASON_LAST_FULL]["Target"].values,
    MID_FEATURE_COLS, f"Midfielder/Forward (Season {_SEASON_LAST_FULL})", "RF_MF",
)
_val_mid_only_a = df_mid[
    (df_mid["Season"] == _SEASON_LAST_FULL) & (df_mid["IsForward"] == 0)
]
score_validation(
    mid_model_a,
    _val_mid_only_a[MID_FEATURE_COLS].values, _val_mid_only_a["Target"].values,
    MID_FEATURE_COLS, f"Midfielder-only (Season {_SEASON_LAST_FULL})",
)

# Forward-only model under the identical split protocol.
fwd_model_a, fwd_scaler_a, _sum_fwd_a = evaluate_oot(
    df_fwd[df_fwd["Season"] < _SEASON_LAST_FULL][FWD_FEATURE_COLS].values,
    df_fwd[df_fwd["Season"] < _SEASON_LAST_FULL]["Target"].values,
    df_fwd[df_fwd["Season"] == _SEASON_LAST_FULL][FWD_FEATURE_COLS].values,
    df_fwd[df_fwd["Season"] == _SEASON_LAST_FULL]["Target"].values,
    FWD_FEATURE_COLS, f"Forward-only (Season {_SEASON_LAST_FULL})", "RF_FWD",
)

# Head-to-head: the SHARED mid/fwd Split A model scored on the same 2025
# forward matches, using its own pipeline's features (mid/fwd-blended
# history + mid/fwd opponent/team/venue aggregates). Rows and targets are
# identical to the forward model's validation frame; only the feature
# pipeline and the model differ. This comparison decides which model
# serves forwards (see FORWARD_MODEL_SHIPPED).
_val_fwd_shared = df_mid[(df_mid["Season"] == _SEASON_LAST_FULL) & (df_mid["IsForward"] == 1)]
_sum_fwd_shared = score_validation(
    mid_model_a,
    _val_fwd_shared[MID_FEATURE_COLS].values, _val_fwd_shared["Target"].values,
    MID_FEATURE_COLS, f"SHARED mid/fwd model on forward rows (Season {_SEASON_LAST_FULL})")

_df_fwd_val_a = df_fwd[df_fwd["Season"] == _SEASON_LAST_FULL]
_y_fwd_a = _df_fwd_val_a["Target"].values
_p_fwd_own = fwd_model_a.predict_proba(_df_fwd_val_a[FWD_FEATURE_COLS].values)[:, 1]
_p_fwd_shr = mid_model_a.predict_proba(_val_fwd_shared[MID_FEATURE_COLS].values)[:, 1]
_ll_fwd_own = log_loss(_y_fwd_a, _p_fwd_own)
_ll_fwd_shr = log_loss(_val_fwd_shared["Target"].values, _p_fwd_shr)
_auc_fwd_own = roc_auc_score(_y_fwd_a, _p_fwd_own)
_auc_fwd_shr = roc_auc_score(_val_fwd_shared["Target"].values, _p_fwd_shr)
print("\n" + "=" * 60)
print(f"FORWARD MODEL HEAD-TO-HEAD (Split A, Season {_SEASON_LAST_FULL} forward rows)")
print("=" * 60)
print(f"  validation rows: {len(_y_fwd_a)} (forward frame) / {len(_val_fwd_shared)} (shared frame), "
      f"hits: {int(_y_fwd_a.sum())} / {int(_val_fwd_shared['Target'].sum())}")
print(f"  forward-only model: LL {_ll_fwd_own:.4f} | AUC {_auc_fwd_own:.4f} | "
      f"mean pred {_p_fwd_own.mean():.4f} | observed {_y_fwd_a.mean():.4f}")
print(f"  shared mid/fwd    : LL {_ll_fwd_shr:.4f} | AUC {_auc_fwd_shr:.4f} | "
      f"mean pred {_p_fwd_shr.mean():.4f}")
# Pre-registered decision rule (2026-09-03, before the first run): ship the
# forward model iff its forward-only Split A log loss beats the shared
# model's by > 0.002 AND its AUC is not more than 0.02 lower. At ~30
# validation positives, margins under 0.002 are indistinguishable from
# noise, so a near-tie keeps the simpler incumbent.
_fwd_ships = (_ll_fwd_own < _ll_fwd_shr - 0.002) and (_auc_fwd_own > _auc_fwd_shr - 0.02)
print(f"  decision rule: LL better by > 0.002 and AUC not > 0.02 lower -> "
      f"{'SHIP forward model' if _fwd_ships else 'KEEP shared model'}")

print(f"\n--- Split B: train <{_SEASON_CURRENT}, validate Season {_SEASON_CURRENT} "
      f"(prior-mechanism test: priors ON vs OFF) ---")
_train_b_def = df_def[df_def["Season"] < _SEASON_CURRENT]
_val_b_on_def = df_def[df_def["Season"] >= _SEASON_CURRENT].copy()
_train_b_mid = df_mid[df_mid["Season"] < _SEASON_CURRENT]
_val_b_on_mid = df_mid[df_mid["Season"] >= _SEASON_CURRENT].copy()

def_model_b, def_scaler_b, _sum_def_on = evaluate_oot(
    _train_b_def[DEF_FEATURE_COLS].values, _train_b_def["Target"].values,
    _val_b_on_def[DEF_FEATURE_COLS].values, _val_b_on_def["Target"].values,
    DEF_FEATURE_COLS, f"Defender (Season {_SEASON_CURRENT}, priors ON)", _DEFENDER_MODEL_NAME,
)
mid_model_b, mid_scaler_b, _sum_mid_on = evaluate_oot(
    _train_b_mid[MID_FEATURE_COLS].values, _train_b_mid["Target"].values,
    _val_b_on_mid[MID_FEATURE_COLS].values, _val_b_on_mid["Target"].values,
    MID_FEATURE_COLS, f"Midfielder/Forward (Season {_SEASON_CURRENT}, priors ON)", "RF_MF",
)
fwd_model_b, fwd_scaler_b, _sum_fwd_on = evaluate_oot(
    df_fwd[df_fwd["Season"] < _SEASON_CURRENT][FWD_FEATURE_COLS].values,
    df_fwd[df_fwd["Season"] < _SEASON_CURRENT]["Target"].values,
    df_fwd[df_fwd["Season"] >= _SEASON_CURRENT][FWD_FEATURE_COLS].values,
    df_fwd[df_fwd["Season"] >= _SEASON_CURRENT]["Target"].values,
    FWD_FEATURE_COLS, f"Forward-only (Season {_SEASON_CURRENT}, priors ON)", "RF_FWD",
)
# Shared model on the same 2026 forward rows -- a few rounds of data only,
# so this is directional context, not a decision input.
_val_fwd_b_shared = df_mid[(df_mid["Season"] >= _SEASON_CURRENT) & (df_mid["IsForward"] == 1)]
score_validation(
    mid_model_b,
    _val_fwd_b_shared[MID_FEATURE_COLS].values, _val_fwd_b_shared["Target"].values,
    MID_FEATURE_COLS, f"SHARED mid/fwd model on forward rows (Season {_SEASON_CURRENT})")

# Ablation arm: rebuild the current-season rows of prior-carrying players
# with use_priors=False. Rebuilding only these players is exact -- per-row
# history features depend solely on the player's own rows, and the
# opponent/team/venue lookups are the same precomputed globals.
_prior_players = {p for p in matchlogs["Player"].unique()
                  if prior_for_season(p, _SEASON_CURRENT) is not None}
print(f"\nPlayers carrying a transfer prior: {len(_prior_players)}")

def _no_prior_val_rows(pos_cols, stat_col, per90_col, threshold,
                       prior_opp, prior_team, prior_venue, detailed_positions):
    """Current-season validation rows for prior-carrying players, rebuilt
    without prior blending (the ablation arm)."""
    subset_logs = matchlogs[matchlogs["Player"].isin(_prior_players)]
    if not _prior_players or len(subset_logs) == 0:
        return pd.DataFrame()
    built = build_dataset(subset_logs, pos_cols, stat_col, per90_col, threshold,
                          prior_opp, prior_team, prior_venue, detailed_positions,
                          use_priors=False)
    return built[built["Season"] >= _SEASON_CURRENT]

_off_def = _no_prior_val_rows(["IsDefender"], "DefStats_Def", "Per90_Def", 10,
                              def_prior_opp, def_prior_team, def_prior_venue, DEFENDER_DETAILED_POS)
_off_mid = _no_prior_val_rows(["IsMidfielderOrForward"], "DefStats_Mid", "Per90_Mid", 12,
                              mid_prior_opp, mid_prior_team, mid_prior_venue, MID_FWD_DETAILED_POS)

_val_b_off_def = pd.concat(
    [_val_b_on_def[~_val_b_on_def["Player"].isin(_prior_players)], _off_def], ignore_index=True)
_val_b_off_mid = pd.concat(
    [_val_b_on_mid[~_val_b_on_mid["Player"].isin(_prior_players)], _off_mid], ignore_index=True)

_sum_def_off = score_validation(def_model_b,
                                _val_b_off_def[DEF_FEATURE_COLS].values, _val_b_off_def["Target"].values,
                                DEF_FEATURE_COLS, f"Defender (Season {_SEASON_CURRENT}, priors OFF -- ablation)")
_sum_mid_off = score_validation(mid_model_b,
                                _val_b_off_mid[MID_FEATURE_COLS].values, _val_b_off_mid["Target"].values,
                                MID_FEATURE_COLS, f"Midfielder/Forward (Season {_SEASON_CURRENT}, priors OFF -- ablation)")

# Compact falsifiable-test comparison: the two numbers the mechanism claim
# makes a directional prediction about.
def _mech_gap(s):
    b = (s or {}).get("buckets", {}).get("0-9")
    return f"gap {b['pred'] - b['obs']:+.3f} (n={b['n']})" if b else "no 0-9 rows"

def _mech_lrt(s):
    lrt = (s or {}).get("lrt")
    return f"LRT chi2={lrt['chi2']:.2f}, p={lrt['p']:.4f}" if lrt else "LRT n/a"

print("\n" + "=" * 60)
print("PRIOR-MECHANISM CHECK (falsifiable test)")
print("=" * 60)
print("Prediction if missing prior-league evidence is the real mechanism:")
print("0-9 gap and LRT shrink from priors OFF to priors ON in the Season "
      f"{_SEASON_CURRENT} split, while Split A keeps its gap (no priors exist there).")
for _label, _s_on, _s_off in [("Defender", _sum_def_on, _sum_def_off),
                              ("Midfielder/Forward", _sum_mid_on, _sum_mid_off)]:
    print(f"\n  {_label}, Season {_SEASON_CURRENT} validation:")
    print(f"    priors OFF: {_mech_gap(_s_off)} | {_mech_lrt(_s_off)}")
    print(f"    priors ON:  {_mech_gap(_s_on)} | {_mech_lrt(_s_on)}")

# Micro-report: the prior-carrying rows themselves -- did prior seeding move
# their predictions toward their observed outcomes, and did high-weight
# priors migrate them out of the 0-9 bucket?
print("\nPrior-carrying rows only (did prior seeding move predictions toward outcomes?):")
for _label, _val_on, _off, _model, _cols in [
    ("Defender", _val_b_on_def, _off_def, def_model_b, DEF_FEATURE_COLS),
    ("Midfielder/Forward", _val_b_on_mid, _off_mid, mid_model_b, MID_FEATURE_COLS),
]:
    if len(_off) == 0:
        print(f"\n  {_label}: no prior-carrying rows in the Season {_SEASON_CURRENT} validation set.")
        continue
    on_k = _val_on[_val_on["Player"].isin(_prior_players)].set_index(["Player", "MatchID"]).sort_index()
    off_k = _off.set_index(["Player", "MatchID"]).sort_index()
    common = on_k.index.intersection(off_k.index)
    on_k, off_k = on_k.loc[common], off_k.loc[common]
    p_on = _model.predict_proba(on_k[_cols].values)[:, 1]
    p_off = _model.predict_proba(off_k[_cols].values)[:, 1]
    yv = on_k["Target"].values
    print(f"\n  {_label}: {len(common)} prior-carrying validation rows")
    print(f"    mean prediction, priors OFF: {p_off.mean():.4f}")
    print(f"    mean prediction, priors ON:  {p_on.mean():.4f}")
    print(f"    observed target rate:        {yv.mean():.4f}")
    print(f"    rows moved DOWN by the prior: {int((p_on < p_off).sum())} / {len(common)}")
    _bins = [-0.5, 9.5, 29.5, 79.5, 149.5, np.inf]
    _labs = ["0-9", "10-29", "30-79", "80-149", "150+"]
    b_on = pd.cut(on_k["MatchesPlayed"], _bins, labels=_labs)
    b_off = pd.cut(off_k["MatchesPlayed"], _bins, labels=_labs)
    parts = [f"{lab}: {int((b_off == lab).sum())} -> {int((b_on == lab).sum())}"
             for lab in _labs if (b_off == lab).sum() or (b_on == lab).sum()]
    print(f"    MatchesPlayed buckets OFF -> ON: {' | '.join(parts)}")

# ── Promoted team adjustments ─────────────────────────────────────────
PROMOTED_TEAMS = {"Hull City", "Coventry City", "Ipswich Town"}
PROMOTED_DEF_RATIO = 1.067

def_team_global_base = np.mean(list(def_final_team.values()))
mid_team_global_base = np.mean(list(mid_final_team.values()))
fwd_team_global_base = np.mean(list(fwd_final_team.values()))

for team in PROMOTED_TEAMS:
    if team not in def_final_team:
        def_final_team[team] = def_team_global_base * PROMOTED_DEF_RATIO
    if team not in mid_final_team:
        mid_final_team[team] = mid_team_global_base
    if team not in fwd_final_team:
        fwd_final_team[team] = fwd_team_global_base

for team in PROMOTED_TEAMS:
    if team not in def_final_opp:
        def_final_opp[team] = np.mean(list(def_final_opp.values())) / PROMOTED_DEF_RATIO
    if team not in mid_final_opp:
        mid_final_opp[team] = np.mean(list(mid_final_opp.values()))
    if team not in fwd_final_opp:
        fwd_final_opp[team] = np.mean(list(fwd_final_opp.values()))

# ── Train models ──────────────────────────────────────────────────────

def train_and_evaluate(df, label, model_name, feature_cols):
    """Train the specified model with optional hyperparameter tuning, then
    calibrate with cross_val_predict + isotonic regression. Returns
    (_CalibratedModel, scaler, model_name)."""
    X = df[feature_cols].values
    y = df["Target"].values
    model_cls, param_grid, default_params = _MODEL_CONFIGS[model_name]
    calibrated, scaler, _best_params = tune_and_calibrate(
        X, y, model_cls, param_grid, label,
        do_tune=_DO_TUNE, default_params=default_params,
    )
    return calibrated, scaler, model_name

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

def_model, def_scaler, def_model_name = train_and_evaluate(df_def, "Defender", _DEFENDER_MODEL_NAME, DEF_FEATURE_COLS)
mid_model, mid_scaler, mid_model_name = train_and_evaluate(df_mid, "Midfielder/Forward", "RF_MF", MID_FEATURE_COLS)
# Forward production model -- trained unconditionally so the weekly
# head-to-head stays honest and the serving switch can be flipped without
# code changes (see FORWARD_MODEL_SHIPPED).
fwd_model, fwd_scaler, fwd_model_name = train_and_evaluate(df_fwd, "Forward", "RF_FWD", FWD_FEATURE_COLS)

# All nine trainings (six validation-battery + three production) are
# done; entries left over from earlier runs used superseded training
# frames and can never be hit again -- prune them now.
_prune_model_cache()

# ── Build player profiles for prediction ──────────────────────────────
print("\n" + "=" * 60)
print("GENERATING PREDICTIONS FOR FUTURE FIXTURES")
print("=" * 60)

# Career aggregates blended with each player's transfer prior as of the
# current season -- the same formulas build_dataset applies to training
# rows, so an arrival who has already played presents prior + those
# matches and the inference vector stays consistent with training.
# AvgPer90/StdPer90/PctAboveThreshold/MatchesPlayed rest on qualifying
# (>=60 min) matches only (a mean of per-match per-90 rates is skewed by
# short-cameo extrapolation noise: 1 minute played, 3 actions -> 270
# per-90); AvgRaw/Recent5AvgRaw count only non-cameo matches
# (Minutes >= RAW_CAMEO_MIN -- raw per-match volume is proportional to
# minutes, so cameo raws bias a per-match mean downward), with the
# Recent5 fallback to AvgRaw when the window is mostly cameos or holds
# too few non-cameo matches.
def_player_feats = precompute_player_features(
    matchlogs["IsDefender"], "DefStats_Def", "Per90_Def", 10, DEFENDER_DETAILED_POS,
    priors=True, prior_season=MAX_SEASON, std_pct_estimator=def_std_estimator)
mid_player_feats = precompute_player_features(
    matchlogs["IsMidfielderOrForward"], "DefStats_Mid", "Per90_Mid", 12, MID_FWD_DETAILED_POS,
    priors=True, prior_season=MAX_SEASON, std_pct_estimator=mid_std_estimator)
fwd_player_feats = precompute_player_features(
    matchlogs["IsForward"], "DefStats_Mid", "Per90_Mid", 12, MID_FWD_DETAILED_POS,
    priors=True, prior_season=MAX_SEASON, std_pct_estimator=fwd_std_estimator)
print(f"Pre-computed defender features: {len(def_player_feats)} players")
print(f"Pre-computed mid/fwd features: {len(mid_player_feats)} players")
print(f"Pre-computed forward features: {len(fwd_player_feats)} players")

# Match players (exact match only)
matchlog_players = set(matchlogs["Player"].unique())
current_players = set(players_current["Player"].unique())
matched = current_players & matchlog_players
print(f"Players with matchlog history: {len(matched)} / {len(current_players)}")
player_map = {p: p for p in matched}

# ── Transfer-based features ──────────────────────────────────────────
# Season-aware fallback for squad players with NO matchlog rows at all:
# seed features from their transfer prior for the current season (the same
# prior-seeding build_dataset applies to debut rows). Arrivals who have
# already played are covered by the prior-blended matchlog tier above --
# this fires only for players yet to appear in the data.
def build_transfer_lookup(season):
    lookup = {}
    for player in set(transfers["Player"].unique()):
        prior = prior_for_season(player, season)
        if prior is not None:
            lookup[player] = prior
    return lookup

def_transfer_lookup = build_transfer_lookup(CURRENT_SEASON)
mid_transfer_lookup = def_transfer_lookup  # prior rate/weight are position-agnostic

# Std/pct estimators were already computed from the pure (unblended) feature
# frames before build_dataset -- no duplicate definitions here.

print(f"Transfer players (defender model): {len(def_transfer_lookup)}")
print(f"Transfer players (mid/fwd model): {len(mid_transfer_lookup)}")

# ── Generate predictions (batch mode) ────────────────────────────────
# Position dummies default to all-zero for youth/no-history and transfer-tier
# players -- we have no reliable detailed-position evidence for either group
# (transfers.csv only has coarse D/M/F, not detailed tactical positions), so
# rather than guess, these players simply get no position-specific signal from
# this feature -- same as their situation before this feature existed.
DEF_DEFAULTS = {"AvgPer90": 8.0, "AvgRaw": 7.0, "StdPer90": 4.0, "MatchesPlayed": 0,
                "Recent5AvgPer90": 8.0, "Recent5AvgRaw": 7.0, "PctAboveThreshold": 0.3,
                "Is_FPL_API": 1,
                **{f"Pos_{c}": 0 for c in DEFENDER_DETAILED_POS}}
MID_DEFAULTS = {"AvgPer90": 9.0, "AvgRaw": 8.0, "StdPer90": 4.0, "MatchesPlayed": 0,
                "Recent5AvgPer90": 9.0, "Recent5AvgRaw": 8.0, "PctAboveThreshold": 0.25,
                "Is_FPL_API": 1,
                **{f"Pos_{c}": 0 for c in MID_FWD_DETAILED_POS}}
# Forward-calibrated youth defaults (only used when the forward model
# serves): the mid/fwd defaults above describe the POOLED mid+forward
# population (target rate ~0.25); a no-history forward presented with
# those values would look like an extreme high-rate outlier to a model
# trained on forwards only. Rounded from the fbref-era forward population
# (2026-09-03 probe): qualifying per-90 mean 5.38 (player-level AvgPer90
# mean 5.69), std 3.20 (player-level StdPer90 mean 2.83), non-cameo raw
# mean 4.14, hit rate 0.0424.
FWD_DEFAULTS = {"AvgPer90": 5.5, "AvgRaw": 4.0, "StdPer90": 3.0, "MatchesPlayed": 0,
                "Recent5AvgPer90": 5.5, "Recent5AvgRaw": 4.0, "PctAboveThreshold": 0.04,
                "Is_FPL_API": 1,
                **{f"Pos_{c}": 0 for c in MID_FWD_DETAILED_POS}}

# Serving switch for the forward-specific model: when on, forwards resolve
# through the forward pipeline (forward-only history features, forward
# aggregates, forward prior estimator, forward youth defaults); mids always
# keep the shared mid/fwd model. DECIDED 2026-09-03 by the pre-registered
# Split A head-to-head rule (see the FORWARD MODEL HEAD-TO-HEAD section and
# CHANGELOG §16): forward-only LL 0.0901 vs shared-on-forward 0.1244 (rule
# needs > 0.002) and AUC 0.7095 vs 0.5762 (floor was -0.02) -> SHIPPED, with
# mean pred 0.0164 vs observed 0.0195 (the shared model over-predicted
# forwards ~2x: 0.0379). Split B (Season 2026) agrees in direction. The
# comparison re-runs every execution so the decision can be revisited as
# data accumulates. Override via DEFCON_FWD_MODEL=1/0.
FORWARD_MODEL_SHIPPED = os.environ.get("DEFCON_FWD_MODEL", "1") == "1"

def make_transfer_features(per90, approx_matches, estimator, detailed_positions):
    est_std, est_pct = estimator(per90)
    # Tier 2 (no PL matches yet): the pooled std degenerates to est_std.
    return {"AvgPer90": per90, "AvgRaw": per90,
            "StdPer90": pooled_prior_std(est_std, approx_matches, per90, []),
            "MatchesPlayed": approx_matches, "Recent5AvgPer90": per90,
            "Recent5AvgRaw": per90, "PctAboveThreshold": est_pct,
            "Is_FPL_API": 1,
            **{f"Pos_{c}": 0 for c in detailed_positions}}

def_opp_global = np.mean(list(def_final_opp.values()))
def_team_global = np.mean(list(def_final_team.values()))
mid_opp_global = np.mean(list(mid_final_opp.values()))
mid_team_global = np.mean(list(mid_final_team.values()))
fwd_opp_global = np.mean(list(fwd_final_opp.values()))
fwd_team_global = np.mean(list(fwd_final_team.values()))
def_venue_global = np.mean(list(def_final_venue.values()))
mid_venue_global = np.mean(list(mid_final_venue.values()))
fwd_venue_global = np.mean(list(fwd_final_venue.values()))

pred_rows = []
fixture_teams = []
for _, fixture in fixtures.iterrows():
    fixture_teams.append((fixture["MatchID"], fixture["Home"], fixture["Away"], "Home"))
    fixture_teams.append((fixture["MatchID"], fixture["Away"], fixture["Home"], "Away"))

# Include D, M, and F players
dmf_players = players_current[players_current["Pos"].isin(["D", "M", "F"])].copy()

for match_id, team, opponent, venue in fixture_teams:
    team_plys = dmf_players[dmf_players["Team"] == team]
    for _, prow in team_plys.iterrows():
        player_name = prow["Player"]
        pos = prow["Pos"]
        ml_name = player_map.get(player_name)

        if pos == "D":
            if ml_name and ml_name in def_player_feats:
                feat = def_player_feats[ml_name]; has_history = True
            elif player_name in def_transfer_lookup:
                t = def_transfer_lookup[player_name]
                feat = make_transfer_features(t["per90"], t["matches"], def_std_estimator, DEFENDER_DETAILED_POS)
                has_history = True
            else:
                feat = DEF_DEFAULTS; has_history = False
            opp_val = def_final_opp.get(opponent, def_opp_global)
            team_val = def_final_team.get(team, def_team_global)
            venue_val = def_final_venue.get(venue, def_venue_global)
            pred_rows.append({
                "MatchID": match_id, "Player": player_name, "Team": team,
                "Opponent": opponent, "Venue": venue, "Pos": pos,
                "Threshold": 10, "StatDescription": "CBI+Tackles", "HasHistory": has_history,
                **feat, "OpponentAvgStat": opp_val, "TeamAvgStat": team_val,
                "VenueAvg": venue_val, "IsHome": 1 if venue == "Home" else 0, "_model": "def",
            })
        elif pos == "F" and FORWARD_MODEL_SHIPPED:
            # Forward-specific serving path (forward-only history features,
            # forward aggregates, forward prior estimator and youth
            # defaults) -- see FORWARD_MODEL_SHIPPED above.
            if ml_name and ml_name in fwd_player_feats:
                feat = fwd_player_feats[ml_name]; has_history = True
            elif player_name in mid_transfer_lookup:
                t = mid_transfer_lookup[player_name]
                feat = make_transfer_features(t["per90"], t["matches"], fwd_std_estimator, MID_FWD_DETAILED_POS)
                has_history = True
            else:
                feat = FWD_DEFAULTS; has_history = False
            opp_val = fwd_final_opp.get(opponent, fwd_opp_global)
            team_val = fwd_final_team.get(team, fwd_team_global)
            venue_val = fwd_final_venue.get(venue, fwd_venue_global)
            pred_rows.append({
                "MatchID": match_id, "Player": player_name, "Team": team,
                "Opponent": opponent, "Venue": venue, "Pos": pos,
                "Threshold": 12, "StatDescription": "CBI+Tackles+Recoveries", "HasHistory": has_history,
                **feat, "OpponentAvgStat": opp_val, "TeamAvgStat": team_val,
                "VenueAvg": venue_val, "IsHome": 1 if venue == "Home" else 0, "_model": "fwd",
            })
        elif pos in ("M", "F"):
            if ml_name and ml_name in mid_player_feats:
                feat = mid_player_feats[ml_name]; has_history = True
            elif player_name in mid_transfer_lookup:
                t = mid_transfer_lookup[player_name]
                feat = make_transfer_features(t["per90"], t["matches"], mid_std_estimator, MID_FWD_DETAILED_POS)
                has_history = True
            else:
                feat = MID_DEFAULTS; has_history = False
            opp_val = mid_final_opp.get(opponent, mid_opp_global)
            team_val = mid_final_team.get(team, mid_team_global)
            venue_val = mid_final_venue.get(venue, mid_venue_global)
            pred_rows.append({
                "MatchID": match_id, "Player": player_name, "Team": team,
                "Opponent": opponent, "Venue": venue, "Pos": pos,
                "Threshold": 12, "StatDescription": "CBI+Tackles+Recoveries", "HasHistory": has_history,
                **feat, "OpponentAvgStat": opp_val, "TeamAvgStat": team_val,
                "VenueAvg": venue_val, "IsHome": 1 if venue == "Home" else 0, "_model": "mid",
            })

pred_df = pd.DataFrame(pred_rows)
print(f"Total prediction rows: {len(pred_df)}")

# Batch predict -- _CalibratedModel handles scaling internally.
def_mask = pred_df["_model"] == "def"
if def_mask.any():
    X_def = pred_df.loc[def_mask, DEF_FEATURE_COLS].values
    pred_df.loc[def_mask, "Probability"] = def_model.predict_proba(X_def)[:, 1]

mid_mask = pred_df["_model"] == "mid"
if mid_mask.any():
    X_mid = pred_df.loc[mid_mask, MID_FEATURE_COLS].values
    pred_df.loc[mid_mask, "Probability"] = mid_model.predict_proba(X_mid)[:, 1]

fwd_mask = pred_df["_model"] == "fwd"
if fwd_mask.any():
    X_fwd = pred_df.loc[fwd_mask, FWD_FEATURE_COLS].values
    pred_df.loc[fwd_mask, "Probability"] = fwd_model.predict_proba(X_fwd)[:, 1]

results = pred_df.drop(columns=["_model"]).copy()
results["Probability_at_90min"] = results["Probability"].round(4)

# ── Scale each player's probability to their ACTUAL expected minutes ──
# The model above predicts P(reach threshold | 90 minutes). To use this for
# a player with partial expected minutes, we invert the probability back to
# an implied mean defcon rate via the same NB(r=3*mu, p=0.75) relationship
# used for the original DC/90 threshold-probability formula, scale that
# mean by xMins/90, then recompute the probability at the scaled mean.
from scipy import stats as _stats
from scipy import optimize as _optimize

def _nb_sf(mu, k, p=0.75):
    r = 3 * mu
    r = max(r, 1e-9)
    return 1 - _stats.nbinom.cdf(k, r, p)

def _invert_mu(prob, k, lo=1e-6, hi=60):
    """Solve for mu such that _nb_sf(mu, k) == prob."""
    if prob <= 1e-9:
        return 0.0
    if prob >= 1 - 1e-9:
        return hi  # degenerate case, effectively "very high rate"
    f = lambda mu: _nb_sf(mu, k) - prob
    try:
        return _optimize.brentq(f, lo, hi)
    except ValueError:
        # prob is outside what's achievable in [lo, hi] -- clip to nearest end
        return lo if f(lo) < 0 else hi


GW_COL = "Round" 

minutes_probs = pd.read_csv(_DATA / "minutes_probabilities.csv", encoding="utf-8-sig")
minutes_lookup = {}
for _, mrow in minutes_probs.iterrows():
    for gw in [1, 2, 3, 4, 5]:
        minutes_lookup[(mrow["Player"], gw)] = mrow[f"xMins_GW{gw}"]

# minutes_probabilities.csv's "Player" column holds Web_Name values
player_to_webname = dict(zip(players_current["Player"], players_current["Web_Name"]))

# `results` never actually carries a "Round" column -- derive
# it from MatchID via `fixtures`.
# GW = Round offset to 1-5
matchid_to_gw = dict(zip(fixtures["MatchID"], fixtures["Round"] - fixtures["Round"].min() + 1))

def _scale_row(row):
    threshold = row["Threshold"]
    prob_90 = row["Probability_at_90min"]
    gw = matchid_to_gw.get(row["MatchID"])
    web_name = player_to_webname.get(row["Player"])
    xmins = minutes_lookup.get((web_name, gw)) if web_name else None
    if xmins is None:
        return prob_90  # no minutes forecast available -- fall back to unscaled
    mu90 = _invert_mu(prob_90, threshold - 1)
    mu_scaled = mu90 * (xmins / 90.0)
    return _nb_sf(mu_scaled, threshold - 1)

results["Probability"] = results.apply(_scale_row, axis=1).round(4)
results = results.sort_values(["MatchID", "Team", "Player"]).reset_index(drop=True)

# ── Save results ──────────────────────────────────────────────────────
results.to_csv(_DATA / "predictions_defensive.csv", index=False)
print(f"\nPredictions saved to predictions_defensive.csv")
print(f"Total predictions: {len(results)}")
print(f"\nSample predictions:")
print(results.head(20).to_string(index=False))

print(f"\n--- Summary ---")
for p in ["D", "M", "F"]:
    sub = results[results["Pos"] == p]
    print(f"  {p}: {len(sub)} predictions, mean probability {sub['Probability'].mean():.4f}")
print(f"Players with history: {results['HasHistory'].sum()} / {len(results)}")

print(f"\n--- Data Source Breakdown (unique D/M/F players) ---")
ml_set = set(matchlogs["Player"].unique())
tr_set = set(transfers["Player"].unique())
dmf_current = players_current[players_current["Pos"].isin(["D", "M", "F"])]
n_ml, n_tr, n_youth = 0, 0, 0
for p in dmf_current["Player"].unique():
    if p in ml_set: n_ml += 1
    elif p in tr_set: n_tr += 1
    else: n_youth += 1
print(f"  Matchlog history: {n_ml}")
print(f"  Transfer data only: {n_tr}")
print(f"  Youth/defaults (no data): {n_youth}")
print(f"  Total unique D/M/F players: {n_ml + n_tr + n_youth}")

# ── Average-opponent 90-minute defcon (player_ratings.csv artifact) ────
# A fixture-neutral defcon probability for the weekly player_ratings.csv
# export (assembled by scripts/player_ratings/export_player_ratings.py):
# P(defcon >= threshold | full 90-minute match vs an average opponent).
# The model's raw probability IS the 90-minute probability -- training
# targets are defined at 90 minutes, and the NB minutes machinery above
# only scales DOWN to expected minutes -- so no minutes scaling happens
# here. Fixture context is neutralised as follows:
#   - OpponentAvgStat -> population mean across the 20 current teams
#   - TeamAvgStat     -> the player's own team's value (unchanged)
#   - VenueAvg/IsHome -> the model runs twice per player (Home and Away)
#     and the two probabilities are averaged, so no venue bias leaks in
# Per-player evidence features resolve through the same three tiers as the
# fixture loop above (matchlog history / transfer prior / youth defaults);
# keep the two loops in sync if the tiers ever change.
print("\n" + "=" * 60)
print("AVERAGE-OPPONENT 90-MIN DEFCON (player_ratings artifact)")
print("=" * 60)

_current_teams = sorted(players_current["Team"].unique())
def_opp_pop = float(np.mean([def_final_opp.get(t, def_opp_global) for t in _current_teams]))
mid_opp_pop = float(np.mean([mid_final_opp.get(t, mid_opp_global) for t in _current_teams]))
fwd_opp_pop = float(np.mean([fwd_final_opp.get(t, fwd_opp_global) for t in _current_teams]))
print(f"Population-mean OpponentAvgStat across {len(_current_teams)} current teams: "
      f"defender {def_opp_pop:.3f}, mid/fwd {mid_opp_pop:.3f}, forward {fwd_opp_pop:.3f}")

avg_rows = []
for _, prow in dmf_players.drop_duplicates(subset=["Player"]).iterrows():
    player_name = prow["Player"]
    pos = prow["Pos"]
    team = prow["Team"]
    ml_name = player_map.get(player_name)
    if pos == "D":
        if ml_name and ml_name in def_player_feats:
            feat = def_player_feats[ml_name]
        elif player_name in def_transfer_lookup:
            t = def_transfer_lookup[player_name]
            feat = make_transfer_features(t["per90"], t["matches"], def_std_estimator, DEFENDER_DETAILED_POS)
        else:
            feat = DEF_DEFAULTS
        team_val = def_final_team.get(team, def_team_global)
        for venue in ("Home", "Away"):
            avg_rows.append({**feat, "Player": player_name, "Pos": pos,
                             "OpponentAvgStat": def_opp_pop, "TeamAvgStat": team_val,
                             "VenueAvg": def_final_venue.get(venue, def_venue_global),
                             "IsHome": 1 if venue == "Home" else 0, "_model": "def"})
    elif pos == "F" and FORWARD_MODEL_SHIPPED:
        # Forward-specific artifact path -- mirrors the fixture loop's
        # forward tiers (keep the two loops in sync).
        if ml_name and ml_name in fwd_player_feats:
            feat = fwd_player_feats[ml_name]
        elif player_name in mid_transfer_lookup:
            t = mid_transfer_lookup[player_name]
            feat = make_transfer_features(t["per90"], t["matches"], fwd_std_estimator, MID_FWD_DETAILED_POS)
        else:
            feat = FWD_DEFAULTS
        team_val = fwd_final_team.get(team, fwd_team_global)
        for venue in ("Home", "Away"):
            avg_rows.append({**feat, "Player": player_name, "Pos": pos,
                             "OpponentAvgStat": fwd_opp_pop, "TeamAvgStat": team_val,
                             "VenueAvg": fwd_final_venue.get(venue, fwd_venue_global),
                             "IsHome": 1 if venue == "Home" else 0, "_model": "fwd"})
    elif pos in ("M", "F"):
        if ml_name and ml_name in mid_player_feats:
            feat = mid_player_feats[ml_name]
        elif player_name in mid_transfer_lookup:
            t = mid_transfer_lookup[player_name]
            feat = make_transfer_features(t["per90"], t["matches"], mid_std_estimator, MID_FWD_DETAILED_POS)
        else:
            feat = MID_DEFAULTS
        team_val = mid_final_team.get(team, mid_team_global)
        for venue in ("Home", "Away"):
            avg_rows.append({**feat, "Player": player_name, "Pos": pos,
                             "OpponentAvgStat": mid_opp_pop, "TeamAvgStat": team_val,
                             "VenueAvg": mid_final_venue.get(venue, mid_venue_global),
                             "IsHome": 1 if venue == "Home" else 0, "_model": "mid"})

avg_df = pd.DataFrame(avg_rows)
def_mask = avg_df["_model"] == "def"
if def_mask.any():
    X = avg_df.loc[def_mask, DEF_FEATURE_COLS].values
    avg_df.loc[def_mask, "Probability"] = def_model.predict_proba(X)[:, 1]
fwd_mask = avg_df["_model"] == "fwd"
if fwd_mask.any():
    X = avg_df.loc[fwd_mask, FWD_FEATURE_COLS].values
    avg_df.loc[fwd_mask, "Probability"] = fwd_model.predict_proba(X)[:, 1]
mid_mask = avg_df["_model"] == "mid"
if mid_mask.any():
    X = avg_df.loc[mid_mask, MID_FEATURE_COLS].values
    avg_df.loc[mid_mask, "Probability"] = mid_model.predict_proba(X)[:, 1]

avg_out = (avg_df.groupby(["Player", "Pos"], as_index=False)["Probability"].mean()
           .rename(columns={"Probability": "Defcon"}))
avg_out["Defcon"] = avg_out["Defcon"].round(4)
avg_out = avg_out.sort_values(["Pos", "Player"]).reset_index(drop=True)
avg_out.to_csv(_DATA / "defcon_average_probabilities.csv", index=False)
print(f"\nWrote {len(avg_out)} players to defcon_average_probabilities.csv")
print("Mean Defcon by position:")
print(avg_out.groupby("Pos")["Defcon"].agg(["count", "mean"]).round(4).to_string())
