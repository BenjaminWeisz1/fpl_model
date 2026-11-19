"""
Out-of-time (OOT) evaluation harness for the team ratings model.

Built BEFORE any of the planned improvements (hyperparameter tuning,
team-specific home/away factors, season-boundary rebalancing) so each
change can be evaluated independently, one at a time, against a fixed
baseline -- the same discipline as the defcon model's OOT harness
(predict.py's evaluate_oot(), CHANGELOG sections 8+).

Design
------
The ratings model is sequential and path-dependent: each match's
ratings depend on every prior match, so (unlike defcon's per-row
classifier) the split must be chronological. For each cutoff season C:

  * TRAIN: the full unchanged sequential process runs over every match
    with Season < C, exactly as update_team_ratings.py does.
  * HELD-OUT: Season == C matches are scored, never used for updates.

Two evaluation modes are reported per split:

  frozen       (PRIMARY) -- ratings are frozen at the C season boundary
      (after pruning + the money-prior blend that starts every season)
      and every held-out match is predicted from that static table.
      Held-out results influence nothing: the strict out-of-time test,
      mirroring defcon's train-on-past / validate-on-future design.
  walk-forward (SECONDARY) -- the unchanged predict-then-update loop
      continues through the held-out season; each prediction is still
      made BEFORE that match's update, so no match uses its own result,
      but later held-out matches benefit from earlier held-out results.
      This matches how the pipeline actually runs week to week; it is
      reported for interpretation, not as the decision metric.

One deliberate deviation from production: ave_goals (the goal-rate
normalisation constant) is computed from TRAINING matches only.
Production computes it from the full match_results.csv -- correct at
run time because only past matches exist then -- but a harness run over
the full frame would leak the held-out season's mean goal rate into
both predictions and (via the update denominators) the rating dynamics.

Metrics (held-out matches only)
-------------------------------
  Poisson log loss (PRIMARY): pred - actual * ln(pred) per team-match,
      summed over held-out home and away predictions (the instruction's
      formula: the Poisson negative log-likelihood kernel; the
      ln(actual!) term is dropped as it is constant across models).
      Lower = better. The per-team-match mean is the canonical headline
      (held-out sets grow as 2026 accumulates); the sum is also shown.
  MAE (secondary): mean |pred - actual| per team-match -- the same
      quantity update_team_ratings.py prints, but held-out only.
  Clean-sheet log loss (downstream calibration): P(concede 0) =
      exp(-1.081 x pred_opponent_NPG) (penalty goals added, exactly as
      build_simulation_features.py does) vs actual goals conceded == 0
      (CHANGELOG §31; before that it was exp(-pred NPG) vs NPG == 0), mean
      binary log loss over held-out team-matches -- a directly
      checkable proxy for whether ratings stay calibrated once they
      flow into the clean-sheet / goals-conceded / player-scaling
      consumers downstream. A binned reliability table is printed
      alongside so aggregate numbers cannot hide miscalibration.

Fidelity guarantee
------------------
run_sequential() below is a verbatim copy of update_team_ratings.py's
loop (its module constants are imported, so they can never drift; CLI
overrides are for experiments only). On every run the harness
re-executes the loop over the full history with production parameters
and compares match-level predictions against team_ratings_history.csv
-- bit-for-bit expected. If that equivalence check fails, the
harness's numbers are not trustworthy.

run_sequential() also serves as the SEMI-PERMANENT VERBATIM REFERENCE
IMPLEMENTATION for the fast tuning engine (ratings_fast.py): the fast
engine is a dict-based port that must stay bit-exact against this
loop, and every tuning/experiment run verifies itself against it via
ratings_fast.verify_against_reference(). If update_team_ratings.py's
rating logic ever changes, update BOTH copies together and re-run the
guards.

Boundary-logic experiments each get their own VERBATIM TWIN below:
run_sequential_rebalanced() (season-boundary rebalancing,
TEAM_RATINGS_REBALANCING.md) and run_sequential_discounted()
(promoted-team entry discount, discount_team_ratings.py) -- each is
the identical production loop plus exactly one experimental change,
and the slow reference implementation for the matching
ratings_fast.py mode. Keeping them separate twins (rather than
switches inside run_sequential) keeps the production reference above
pristine, so the fidelity check never depends on experiment code. The
experimental modes are exclusive: one change at a time.

Usage
-----
  python -m research.team_ratings.evaluate_team_ratings_oot --label baseline --save
      First run: establishes the stored baseline.

  python -m research.team_ratings.evaluate_team_ratings_oot --alpha 0.06 --label exp1_v1 --save
      Experiment: overrides hyperparameters; automatically prints a
      before/after comparison against the stored baseline.

  python -m research.team_ratings.evaluate_team_ratings_oot --compare exp1_v1 ...
      Compare against any stored run.

Results are stored in data/validation/team_ratings_oot/<label>.json with
a matching <label>_report.txt. match_results.csv grows weekly: re-run
--label baseline (overwrite) on the current data snapshot before each
experiment comparison so both runs see identical data. Experiments
that change the SEASON-BOUNDARY or ENTRY logic (rebalancing,
promoted-team discount) get their own verbatim twin --
run_sequential_rebalanced() / run_sequential_discounted() below -- so
run_sequential() stays the pristine production reference; see
rebalance_team_ratings.py / discount_team_ratings.py.
Parameter-only experiments use the CLI flags.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.minutes_penalties_cs.build_penalty_predictions import PEN_ATTEMPTS_PER_NPG
from fpl_model.team_ratings.update_team_ratings import alpha, beta, home_advantage, money_weight

# Production clean-sheet definition (build_simulation_features.py:147-150):
# expected goals conceded = opponent's predicted NPG plus expected penalty
# goals (attempts-per-NPG x 0.79 conversion), scored against ACTUAL goals
# conceded (penalties included). The 0.79 is inline in production; keep the
# two in step if it ever changes.
CS_PEN_GOAL_MULT = 1 + PEN_ATTEMPTS_PER_NPG * 0.79

_DATA = paths.CURRENT
MATCH_RESULTS_FILE = paths.current_file("match_results.csv", root=paths.CURRENT)
MONEY_PRIORS_FILE = paths.current_file("money_priors_by_season.csv", root=paths.CURRENT)
HISTORY_FILE = paths.current_file("team_ratings_history.csv", root=paths.CURRENT)
OUT_DIR = paths.VALIDATION / "team_ratings_oot"

# Default held-out seasons: each is scored once, under its own cutoff,
# with 6+ seasons of training behind it. Overridable via --cutoffs.
DEFAULT_CUTOFFS = (2023, 2024, 2025, 2026)

EPS = 1e-9  # metric guards (the update loop keeps production's own eps)

_REPORT_LINES = []


def say(line=""):
    print(line)
    _REPORT_LINES.append(line)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Input loading ────────────────────────────────────────────────────

def load_inputs():
    df = pd.read_csv(MATCH_RESULTS_FILE)
    money_priors = pd.read_csv(MONEY_PRIORS_FILE)
    for col in ("Home_NPG", "Away_NPG", "Home_NPxG_Adj", "Away_NPxG_Adj"):
        if df[col].isna().any():
            raise ValueError(f"{MATCH_RESULTS_FILE.name}: NaN in required column {col}")
    prior_map = {
        (row["Season"], row["Team"]): (row["Money_GF"], row["Money_GA"])
        for _, row in money_priors.iterrows()
    }
    data_meta = {
        "match_results": {
            "rows": int(len(df)),
            "md5": md5sum(MATCH_RESULTS_FILE),
            "mtime": datetime.fromtimestamp(MATCH_RESULTS_FILE.stat().st_mtime).isoformat(timespec="seconds"),
        },
        "money_priors": {
            "rows": int(len(money_priors)),
            "md5": md5sum(MONEY_PRIORS_FILE),
            "mtime": datetime.fromtimestamp(MONEY_PRIORS_FILE.stat().st_mtime).isoformat(timespec="seconds"),
        },
    }
    return df, prior_map, data_meta


def check_chronology(df):
    """File order must be chronological within each season -- the whole
    sequential design depends on it. fbref links carry the match date;
    FPL_API-sourced seasons (2025+) have no links, so those seasons are
    trusted in file order exactly as update_team_ratings.py does."""
    def parse_date(link):
        ms = re.findall(r"([A-Za-z]+)-(\d{1,2})-(\d{4})", str(link))
        if not ms:
            return None
        mon, day, yr = ms[-1]
        try:
            return pd.Timestamp(f"{mon.title()} {int(day)} {int(yr)}")
        except ValueError:
            return None

    dates = df["Link"].map(parse_date)
    verified, violations = [], []
    for season, g in df.assign(_d=dates).groupby("Season"):
        if g["_d"].isna().any():
            continue  # linkless season -- cannot verify
        if not (g["_d"].diff().dropna() >= pd.Timedelta(0)).all():
            violations.append(int(season))
        verified.append(int(season))
    return {
        "verified_seasons": verified,
        "violations": violations,
        "unparseable_links": int(dates.isna().sum()),
    }


# ── Verbatim replication of the production loop ─────────────────────

def run_sequential(df, ave_goals, params, prior_map, snapshot_seasons=()):
    """Exact copy of update_team_ratings.py's sequential loop, with the
    module constants replaced by `params` and per-row predictions/
    ratings collected. `ave_goals` is passed in (train-only for OOT
    runs, full-frame for the equivalence check). When a season in
    `snapshot_seasons` begins, captures the ratings table AFTER the
    season transition (prune + money-prior blend) and extended with
    every season team still missing (newcomers at their pure prior
    values, exactly what the lazy team-add would give them) -- the
    frozen "preseason snapshot" for that season.

    Returns (df with prediction/rating columns, {season: snapshot df}).
    """
    alpha_ = params["alpha"]
    beta_ = params["beta"]
    home_adv_ = params["home_advantage"]
    money_w_ = params["money_weight"]

    ratings = pd.DataFrame(columns=["Team", "Attack_Rating", "Defense_Rating"])
    current_season = None
    home_preds, away_preds = [], []
    home_attack_rating, home_defense_rating = [], []
    away_attack_rating, away_defense_rating = [], []
    snapshots = {}

    def _prior_or_fail(season, team):
        if (season, team) not in prior_map:
            raise KeyError(
                f"No money prior for ({season}, {team!r}) -- "
                f"update_team_ratings.py would fail identically")
        return prior_map[(season, team)]

    for _, row in df.iterrows():
        season = row["Season"]
        if season != current_season:
            current_season = season
            season_teams = set(df[df["Season"] == season]["Home"]) | set(df[df["Season"] == season]["Away"])
            ratings = ratings[ratings["Team"].isin(season_teams)].reset_index(drop=True)
            for i in ratings.index:
                team = ratings.at[i, "Team"]
                if (season, team) in prior_map:
                    money_atk, money_def = prior_map[(season, team)]
                    ratings.at[i, "Attack_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Attack_Rating"] + money_w_ * money_atk
                    )
                    ratings.at[i, "Defense_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Defense_Rating"] + money_w_ * money_def
                    )
            if season in snapshot_seasons and season not in snapshots:
                snap = ratings.copy(deep=True)
                for team in sorted(season_teams):
                    if team not in snap["Team"].values:
                        money_atk, money_def = _prior_or_fail(season, team)
                        snap.loc[len(snap)] = [team, money_atk, money_def]
                snapshots[season] = snap.reset_index(drop=True)
        for team in (row["Home"], row["Away"]):
            if team not in ratings["Team"].values:
                money_atk, money_def = _prior_or_fail(season, team)
                ratings.loc[len(ratings)] = [team, money_atk, money_def]

        home = row["Home"]
        away = row["Away"]
        h_atk = ratings.loc[ratings["Team"] == home, "Attack_Rating"].iloc[0]
        a_atk = ratings.loc[ratings["Team"] == away, "Attack_Rating"].iloc[0]
        h_def = ratings.loc[ratings["Team"] == home, "Defense_Rating"].iloc[0]
        a_def = ratings.loc[ratings["Team"] == away, "Defense_Rating"].iloc[0]

        pred_home_goals = h_atk * (a_def / ave_goals) * home_adv_
        pred_away_goals = a_atk * (h_def / ave_goals) * (2 - home_adv_)

        home_preds.append(pred_home_goals)
        away_preds.append(pred_away_goals)
        home_attack_rating.append(h_atk)
        home_defense_rating.append(h_def)
        away_attack_rating.append(a_atk)
        away_defense_rating.append(a_def)

        h_npxg, a_npxg = row["Home_NPxG_Adj"], row["Away_NPxG_Adj"]
        h_npg, a_npg = row["Home_NPG"], row["Away_NPG"]
        eps = 1e-6

        h_atk *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))
        a_atk *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        h_def *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        a_def *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))

        ratings.loc[ratings["Team"] == home, "Attack_Rating"] = h_atk
        ratings.loc[ratings["Team"] == home, "Defense_Rating"] = h_def
        ratings.loc[ratings["Team"] == away, "Attack_Rating"] = a_atk
        ratings.loc[ratings["Team"] == away, "Defense_Rating"] = a_def

    out = df.copy()
    out["Home_Prediction"] = home_preds
    out["Away_Prediction"] = away_preds
    out["Home_Attack_Rating"] = home_attack_rating
    out["Home_Defense_Rating"] = home_defense_rating
    out["Away_Attack_Rating"] = away_attack_rating
    out["Away_Defense_Rating"] = away_defense_rating
    return out, snapshots


def run_sequential_rebalanced(df, ave_goals, params, prior_map,
                              snapshot_seasons=(), anchor="midpoint"):
    """Verbatim twin of run_sequential() plus the season-boundary
    rebalancing step from TEAM_RATINGS_REBALANCING.md -- the slow
    reference implementation for ratings_fast.py's rebalanced mode.
    update_team_ratings.py itself is never touched; this twin exists
    so experiments have a bit-exact slow reference that includes the
    new step.

    The rebalancing step runs at the start of EVERY season (2017
    included: there the carried table is empty, so the step operates
    on the pure money priors themselves), AFTER the existing
    prune + money-prior blend and BEFORE the season's first match:

      1. Extend the table with every season team still missing, at its
         pure money-prior values (the post-blend state for promoted
         teams: 100% money-implied).
      2. avg_Attack / avg_Defense across ALL season teams. Both
         engines accumulate these in SORTED team order with plain
         float addition -- ratings_fast.fast_run mirrors this
         operation-for-operation so the two stay bit-exact.
      3. Rescale every team: Attack *= k_a, Defense *= k_d.
         anchor="midpoint" (the doc's primary choice, a real DESIGN
         DECISION, not an obvious default): M = (avg_Attack +
         avg_Defense)/2, k_a = M/avg_Attack, k_d = M/avg_Defense.
         anchor="defense" (the sensitivity alternative): hold the
         Defense average fixed, k_d = 1, k_a = avg_Defense/avg_Attack.

    Everything else is line-for-line run_sequential(). The boundary
    snapshot is captured AFTER rebalancing (all teams already present,
    so the snapshot's own extension loop is a no-op). Returns the same
    (df, {season: snapshot df}) shape as run_sequential().
    """
    alpha_ = params["alpha"]
    beta_ = params["beta"]
    home_adv_ = params["home_advantage"]
    money_w_ = params["money_weight"]

    ratings = pd.DataFrame(columns=["Team", "Attack_Rating", "Defense_Rating"])
    current_season = None
    home_preds, away_preds = [], []
    home_attack_rating, home_defense_rating = [], []
    away_attack_rating, away_defense_rating = [], []
    snapshots = {}

    def _prior_or_fail(season, team):
        if (season, team) not in prior_map:
            raise KeyError(
                f"No money prior for ({season}, {team!r}) -- "
                f"update_team_ratings.py would fail identically")
        return prior_map[(season, team)]

    for _, row in df.iterrows():
        season = row["Season"]
        if season != current_season:
            current_season = season
            season_teams = set(df[df["Season"] == season]["Home"]) | set(df[df["Season"] == season]["Away"])
            ratings = ratings[ratings["Team"].isin(season_teams)].reset_index(drop=True)
            for i in ratings.index:
                team = ratings.at[i, "Team"]
                if (season, team) in prior_map:
                    money_atk, money_def = prior_map[(season, team)]
                    ratings.at[i, "Attack_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Attack_Rating"] + money_w_ * money_atk
                    )
                    ratings.at[i, "Defense_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Defense_Rating"] + money_w_ * money_def
                    )
            # ── Rebalancing step (TEAM_RATINGS_REBALANCING.md) ──
            # Newcomers first (post-blend state = pure priors), then
            # league averages over ALL season teams in sorted order,
            # then the rescale. Mirrored bit-for-bit by
            # ratings_fast.fast_run(rebalance=...).
            for team in sorted(season_teams):
                if team not in ratings["Team"].values:
                    money_atk, money_def = _prior_or_fail(season, team)
                    ratings.loc[len(ratings)] = [team, money_atk, money_def]
            sum_atk = 0.0
            sum_def = 0.0
            for team in sorted(season_teams):
                sum_atk += ratings.loc[ratings["Team"] == team, "Attack_Rating"].iloc[0]
                sum_def += ratings.loc[ratings["Team"] == team, "Defense_Rating"].iloc[0]
            n_teams = len(season_teams)
            avg_atk = sum_atk / n_teams
            avg_def = sum_def / n_teams
            if anchor == "midpoint":
                m_ = (avg_atk + avg_def) / 2
                k_a = m_ / avg_atk
                k_d = m_ / avg_def
            elif anchor == "defense":
                k_a = avg_def / avg_atk
                k_d = 1.0
            else:
                raise ValueError(f"unknown anchor {anchor!r} (expected 'midpoint' or 'defense')")
            for i in ratings.index:
                ratings.at[i, "Attack_Rating"] = ratings.at[i, "Attack_Rating"] * k_a
                ratings.at[i, "Defense_Rating"] = ratings.at[i, "Defense_Rating"] * k_d
            # ── end rebalancing step ──
            if season in snapshot_seasons and season not in snapshots:
                snap = ratings.copy(deep=True)
                for team in sorted(season_teams):
                    if team not in snap["Team"].values:
                        money_atk, money_def = _prior_or_fail(season, team)
                        snap.loc[len(snap)] = [team, money_atk, money_def]
                snapshots[season] = snap.reset_index(drop=True)
        for team in (row["Home"], row["Away"]):
            if team not in ratings["Team"].values:
                money_atk, money_def = _prior_or_fail(season, team)
                ratings.loc[len(ratings)] = [team, money_atk, money_def]

        home = row["Home"]
        away = row["Away"]
        h_atk = ratings.loc[ratings["Team"] == home, "Attack_Rating"].iloc[0]
        a_atk = ratings.loc[ratings["Team"] == away, "Attack_Rating"].iloc[0]
        h_def = ratings.loc[ratings["Team"] == home, "Defense_Rating"].iloc[0]
        a_def = ratings.loc[ratings["Team"] == away, "Defense_Rating"].iloc[0]

        pred_home_goals = h_atk * (a_def / ave_goals) * home_adv_
        pred_away_goals = a_atk * (h_def / ave_goals) * (2 - home_adv_)

        home_preds.append(pred_home_goals)
        away_preds.append(pred_away_goals)
        home_attack_rating.append(h_atk)
        home_defense_rating.append(h_def)
        away_attack_rating.append(a_atk)
        away_defense_rating.append(a_def)

        h_npxg, a_npxg = row["Home_NPxG_Adj"], row["Away_NPxG_Adj"]
        h_npg, a_npg = row["Home_NPG"], row["Away_NPG"]
        eps = 1e-6

        h_atk *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))
        a_atk *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        h_def *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        a_def *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))

        ratings.loc[ratings["Team"] == home, "Attack_Rating"] = h_atk
        ratings.loc[ratings["Team"] == home, "Defense_Rating"] = h_def
        ratings.loc[ratings["Team"] == away, "Attack_Rating"] = a_atk
        ratings.loc[ratings["Team"] == away, "Defense_Rating"] = a_def

    out = df.copy()
    out["Home_Prediction"] = home_preds
    out["Away_Prediction"] = away_preds
    out["Home_Attack_Rating"] = home_attack_rating
    out["Home_Defense_Rating"] = home_defense_rating
    out["Away_Attack_Rating"] = away_attack_rating
    out["Away_Defense_Rating"] = away_defense_rating
    return out, snapshots


def run_sequential_discounted(df, ave_goals, params, prior_map,
                              snapshot_seasons=(), discount=None):
    """Verbatim twin of run_sequential() plus the promoted-team entry
    discount (see discount_team_ratings.py) -- the slow reference
    implementation for ratings_fast.py's discounted mode.
    update_team_ratings.py itself is never touched.

    discount: None (behaves identically to run_sequential) or a
    (d_attack, d_defense) tuple. When set, every team entering the
    ratings pool via the 100%-money-implied branch -- the lazy add at
    its first match of a season, and the matching snapshot extension
    -- enters at (d_attack * Money_GF, d_defense * Money_GA) instead
    of the pure prior. Applied ONLY after the frame's first season
    (the first season's 20 teams are pool initialization, not
    promotion) and ONLY to pool entrants: continuing teams'
    money_weight blend is untouched. The multiplication order
    (discount factor * prior) mirrors ratings_fast.fast_run exactly,
    for bit-exactness.

    Returns the same (df, {season: snapshot df}) shape as
    run_sequential().
    """
    alpha_ = params["alpha"]
    beta_ = params["beta"]
    home_adv_ = params["home_advantage"]
    money_w_ = params["money_weight"]
    first_season_ = df["Season"].iloc[0]
    d_atk = d_def = None
    if discount is not None:
        d_atk, d_def = discount

    ratings = pd.DataFrame(columns=["Team", "Attack_Rating", "Defense_Rating"])
    current_season = None
    home_preds, away_preds = [], []
    home_attack_rating, home_defense_rating = [], []
    away_attack_rating, away_defense_rating = [], []
    snapshots = {}

    def _prior_or_fail(season, team):
        if (season, team) not in prior_map:
            raise KeyError(
                f"No money prior for ({season}, {team!r}) -- "
                f"update_team_ratings.py would fail identically")
        return prior_map[(season, team)]

    for _, row in df.iterrows():
        season = row["Season"]
        if season != current_season:
            current_season = season
            season_teams = set(df[df["Season"] == season]["Home"]) | set(df[df["Season"] == season]["Away"])
            ratings = ratings[ratings["Team"].isin(season_teams)].reset_index(drop=True)
            for i in ratings.index:
                team = ratings.at[i, "Team"]
                if (season, team) in prior_map:
                    money_atk, money_def = prior_map[(season, team)]
                    ratings.at[i, "Attack_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Attack_Rating"] + money_w_ * money_atk
                    )
                    ratings.at[i, "Defense_Rating"] = (
                        (1 - money_w_) * ratings.at[i, "Defense_Rating"] + money_w_ * money_def
                    )
            if season in snapshot_seasons and season not in snapshots:
                snap = ratings.copy(deep=True)
                for team in sorted(season_teams):
                    if team not in snap["Team"].values:
                        money_atk, money_def = _prior_or_fail(season, team)
                        if d_atk is not None and season > first_season_:
                            snap.loc[len(snap)] = [team, d_atk * money_atk, d_def * money_def]
                        else:
                            snap.loc[len(snap)] = [team, money_atk, money_def]
                snapshots[season] = snap.reset_index(drop=True)
        for team in (row["Home"], row["Away"]):
            if team not in ratings["Team"].values:
                money_atk, money_def = _prior_or_fail(season, team)
                if d_atk is not None and season > first_season_:
                    ratings.loc[len(ratings)] = [team, d_atk * money_atk, d_def * money_def]
                else:
                    ratings.loc[len(ratings)] = [team, money_atk, money_def]

        home = row["Home"]
        away = row["Away"]
        h_atk = ratings.loc[ratings["Team"] == home, "Attack_Rating"].iloc[0]
        a_atk = ratings.loc[ratings["Team"] == away, "Attack_Rating"].iloc[0]
        h_def = ratings.loc[ratings["Team"] == home, "Defense_Rating"].iloc[0]
        a_def = ratings.loc[ratings["Team"] == away, "Defense_Rating"].iloc[0]

        pred_home_goals = h_atk * (a_def / ave_goals) * home_adv_
        pred_away_goals = a_atk * (h_def / ave_goals) * (2 - home_adv_)

        home_preds.append(pred_home_goals)
        away_preds.append(pred_away_goals)
        home_attack_rating.append(h_atk)
        home_defense_rating.append(h_def)
        away_attack_rating.append(a_atk)
        away_defense_rating.append(a_def)

        h_npxg, a_npxg = row["Home_NPxG_Adj"], row["Away_NPxG_Adj"]
        h_npg, a_npg = row["Home_NPG"], row["Away_NPG"]
        eps = 1e-6

        h_atk *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))
        a_atk *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        h_def *= (1 - alpha_ + alpha_ * ((a_npxg * beta_ + a_npg * (1 - beta_)) / max(pred_away_goals, eps)))
        a_def *= (1 - alpha_ + alpha_ * ((h_npxg * beta_ + h_npg * (1 - beta_)) / max(pred_home_goals, eps)))

        ratings.loc[ratings["Team"] == home, "Attack_Rating"] = h_atk
        ratings.loc[ratings["Team"] == home, "Defense_Rating"] = h_def
        ratings.loc[ratings["Team"] == away, "Attack_Rating"] = a_atk
        ratings.loc[ratings["Team"] == away, "Defense_Rating"] = a_def

    out = df.copy()
    out["Home_Prediction"] = home_preds
    out["Away_Prediction"] = away_preds
    out["Home_Attack_Rating"] = home_attack_rating
    out["Home_Defense_Rating"] = home_defense_rating
    out["Away_Attack_Rating"] = away_attack_rating
    out["Away_Defense_Rating"] = away_defense_rating
    return out, snapshots


def frozen_predict(val_df, snap, prior_map, ave_goals, params):
    """Predict held-out matches from a frozen preseason snapshot table.
    The snapshot already contains every season team, so the prior
    fallback is belt-and-braces only."""
    lookup = snap.set_index("Team")[["Attack_Rating", "Defense_Rating"]].to_dict("index")
    pred_home, pred_away = [], []
    for _, row in val_df.iterrows():
        home, away, season = row["Home"], row["Away"], row["Season"]
        h = lookup.get(home)
        if h is None:
            h_atk, h_def = prior_map[(season, home)]
            h = {"Attack_Rating": h_atk, "Defense_Rating": h_def}
        a = lookup.get(away)
        if a is None:
            a_atk, a_def = prior_map[(season, away)]
            a = {"Attack_Rating": a_atk, "Defense_Rating": a_def}
        pred_home.append(
            h["Attack_Rating"] * (a["Defense_Rating"] / ave_goals) * params["home_advantage"])
        pred_away.append(
            a["Attack_Rating"] * (h["Defense_Rating"] / ave_goals) * (2 - params["home_advantage"]))
    return np.asarray(pred_home, dtype=float), np.asarray(pred_away, dtype=float)


# ── Metrics ──────────────────────────────────────────────────────────

def summarize_predictions(pred_home, pred_away, home_npg, away_npg,
                          home_goals=None, away_goals=None):
    """Poisson LL kernel (pred - actual*ln(pred)), MAE, and clean-sheet
    log loss for a set of matches. Returns (summary dict, CS prob array,
    CS event array) -- the arrays feed the pooled reliability bins.

    Clean-sheet definition (CHANGELOG §31):
      production -- home_goals/away_goals given: P = exp(-CS_PEN_GOAL_MULT
          * pred_opponent_NPG) vs actual goals conceded == 0, exactly what
          build_simulation_features.py feeds the points model. main() uses it.
      legacy_npg -- goals omitted: P = exp(-pred_opponent_NPG) vs opponent
          NPG == 0. Kept only so the closed §21-§25 experiment scripts, which
          pass NPG positionally, still reproduce their recorded numbers."""
    ph = np.maximum(np.asarray(pred_home, dtype=float), EPS)
    pa = np.maximum(np.asarray(pred_away, dtype=float), EPS)
    yh = np.asarray(home_npg, dtype=float)
    ya = np.asarray(away_npg, dtype=float)

    ll = np.concatenate([ph - yh * np.log(ph), pa - ya * np.log(pa)])
    mae_arr = np.concatenate([np.abs(ph - yh), np.abs(pa - ya)])

    # Home CS: away team held to 0 -> P = exp(-lambda_away_goals).
    # Away CS: home team held to 0 -> P = exp(-lambda_home_goals).
    if (home_goals is None) != (away_goals is None):
        raise ValueError("pass both home_goals and away_goals, or neither")
    if home_goals is not None:
        cs_definition, mult = "production", CS_PEN_GOAL_MULT
        gh = np.asarray(home_goals, dtype=float)
        ga = np.asarray(away_goals, dtype=float)
    else:
        cs_definition, mult = "legacy_npg", 1.0
        gh, ga = yh, ya
    p_home_cs = np.exp(-mult * pa)
    e_home_cs = (ga == 0).astype(float)
    p_away_cs = np.exp(-mult * ph)
    e_away_cs = (gh == 0).astype(float)
    probs = np.concatenate([p_home_cs, p_away_cs])
    events = np.concatenate([e_home_cs, e_away_cs])
    pc = np.clip(probs, EPS, 1.0 - EPS)
    cs_ll = float(np.mean(-(events * np.log(pc) + (1.0 - events) * np.log(1.0 - pc))))

    summary = {
        "n_matches": int(len(ph)),
        "n_team_matches": int(2 * len(ph)),
        "poisson_ll_sum": float(ll.sum()),
        "poisson_ll_per_tm": float(ll.mean()),
        "mae": float(mae_arr.mean()),
        "cs_definition": cs_definition,
        "cs_ll": cs_ll,
        "obs_cs_rate": float(events.mean()),
        "pred_cs_rate": float(probs.mean()),
    }
    return summary, probs, events


def cs_bin_table(probs, events, n_bins=5):
    """Equal-count bins on predicted P(clean sheet): mean predicted vs
    observed rate per bin -- the stratified-reliability check that keeps
    the aggregate CS log loss honest."""
    edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    for i in range(1, len(edges)):  # guarantee strictly increasing edges
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    idx = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        lo = 0.0 if not np.isfinite(edges[b]) else edges[b]
        hi = 1.0 if not np.isfinite(edges[b + 1]) else edges[b + 1]
        rows.append({
            "bin": f"[{lo:.3f}, {hi:.3f})",
            "n": n,
            "pred": float(probs[m].mean()),
            "obs": float(events[m].mean()),
        })
    return rows


# ── Equivalence check ────────────────────────────────────────────────

def equivalence_check(df, prior_map, params, params_are_defaults, run_it=True):
    """Re-run the verbatim loop over the FULL history with the FULL-frame
    ave_goals (production's own definition) and compare against
    team_ratings_history.csv. Also yields the in-sample reference MAE
    (one-step-ahead over all history -- what update_team_ratings.py
    prints; NOT an OOT number)."""
    ave_full = pd.concat([df["Home_NPG"], df["Away_NPG"]]).mean()
    if not run_it:
        return None, {"performed": False, "reason": "disabled via --no-equivalence-check"}
    out, _ = run_sequential(df, ave_full, params, prior_map)
    ref_mae = float(np.mean(np.concatenate([
        np.abs(out["Home_Prediction"].values - out["Home_NPG"].values),
        np.abs(out["Away_Prediction"].values - out["Away_NPG"].values),
    ])))
    if not params_are_defaults:
        return out, {
            "performed": False,
            "reason": "params overridden: replication ran with non-production params; file comparison skipped",
            "reference_insample_mae": ref_mae,
        }
    if not HISTORY_FILE.exists():
        return out, {
            "performed": False,
            "reason": f"{HISTORY_FILE.name} missing; run update_team_ratings.py first",
            "reference_insample_mae": ref_mae,
        }
    hist = pd.read_csv(HISTORY_FILE)
    if len(hist) != len(df) or not (hist["MatchID"].values == df["MatchID"].values).all():
        return out, {
            "performed": False,
            "reason": f"{HISTORY_FILE.name} is stale (rows/MatchIDs differ from match_results.csv); "
                      f"run update_team_ratings.py to refresh; comparison skipped",
            "reference_insample_mae": ref_mae,
        }
    diffs = {}
    for col in ("Home_Prediction", "Away_Prediction", "Home_Attack_Rating",
                "Home_Defense_Rating", "Away_Attack_Rating", "Away_Defense_Rating"):
        diffs[col] = float(np.max(np.abs(out[col].values - hist[col].values)))
    max_diff = max(diffs.values())
    return out, {
        "performed": True,
        "rows_compared": int(len(df)),
        "max_abs_diff_by_col": diffs,
        "max_abs_diff": max_diff,
        "passed": bool(max_diff < 1e-9),
        "history_mtime": datetime.fromtimestamp(HISTORY_FILE.stat().st_mtime).isoformat(timespec="seconds"),
        "reference_insample_mae": ref_mae,
    }


# ── Stored-run comparison ────────────────────────────────────────────

def pooled_metric(run, mode, metric, split_keys):
    """Weighted (by n_team_matches) pooled metric over the given splits,
    recomputed from per-split values so runs with different cutoff sets
    compare over their common splits only."""
    num = den = 0.0
    for k in split_keys:
        s = run["splits"][k][mode]
        num += s[metric] * s["n_team_matches"]
        den += s["n_team_matches"]
    return num / den if den else float("nan")


def do_compare(base, cur, base_label):
    say()
    say("=" * 60)
    say(f"COMPARISON vs stored run '{base_label}'")
    say("=" * 60)
    say(f"  stored: saved {base.get('saved_at', '?')} | params alpha={base['params']['alpha']}, "
        f"beta={base['params']['beta']}, home_adv={base['params']['home_advantage']:.6f}, "
        f"money_weight={base['params']['money_weight']}")
    md5_same = (base["data"]["match_results"]["md5"] == cur["data"]["match_results"]["md5"])
    if not md5_same:
        say("  WARNING: match_results.csv differs between runs (weekly data growth?) --")
        say("           re-run the baseline on the current snapshot before judging deltas.")
    else:
        say("  data snapshot: identical match_results.csv")

    common = [k for k in map(str, cur["cutoffs"]) if k in base["splits"]]
    if not common:
        say("  No common cutoff seasons between runs -- nothing to compare.")
        return
    say(f"  common splits: {', '.join(common)}")
    say()
    for mode, mode_label in (("frozen", "frozen (PRIMARY)"), ("walk_forward", "walk-forward (secondary)")):
        say(f"  --- {mode_label} ---")
        say(f"  {'metric':<26s}{'baseline':>12s}{'current':>12s}{'delta':>12s}")
        # Runs saved before §31 carry no cs_definition and used legacy_npg;
        # their clean-sheet LL is a different quantity, so don't diff it.
        k0 = common[0]
        cs_same = (base["splits"][k0][mode].get("cs_definition", "legacy_npg")
                   == cur["splits"][k0][mode].get("cs_definition", "legacy_npg"))
        metrics = [("poisson_ll_per_tm", "Poisson LL / team-match"), ("mae", "MAE")]
        if cs_same:
            metrics.append(("cs_ll", "Clean-sheet LL"))
        for metric, label in metrics:
            b = pooled_metric(base, mode, metric, common)
            c = pooled_metric(cur, mode, metric, common)
            say(f"  {label:<26s}{b:>12.4f}{c:>12.4f}{c - b:>+12.4f}")
        if not cs_same:
            say(f"  {'Clean-sheet LL':<26s}  not compared: stored run uses a different CS")
            say(f"  {'':<26s}  definition (pre-§31 legacy_npg) -- re-save that run")
        parts, improved = [], 0
        for k in common:
            b = base["splits"][k][mode]["poisson_ll_per_tm"]
            c = cur["splits"][k][mode]["poisson_ll_per_tm"]
            parts.append(f"{k} {c - b:+.4f}")
            improved += int(c < b)
        say(f"  per-split Poisson LL deltas: {' | '.join(parts)}")
        say(f"  splits improved (Poisson LL): {improved}/{len(common)}")
        say()
    say("  NOTE: pass/fail comes from the experiment's pre-registered rule --")
    say("  the harness reports deltas only.")


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Team ratings out-of-time evaluation harness")
    p.add_argument("--cutoffs", default=",".join(str(c) for c in DEFAULT_CUTOFFS),
                   help=f"comma-separated cutoff seasons (default: {','.join(str(c) for c in DEFAULT_CUTOFFS)}); "
                        f"for each C: train Season < C, evaluate Season == C")
    p.add_argument("--alpha", type=float, default=None,
                   help="override update_team_ratings.py alpha (default: production value)")
    p.add_argument("--beta", type=float, default=None,
                   help="override beta (default: production value)")
    p.add_argument("--home-advantage", type=float, default=None,
                   help="override home_advantage (default: production value)")
    p.add_argument("--money-weight", type=float, default=None,
                   help="override money_weight (default: production value)")
    p.add_argument("--label", default="adhoc", help="run label used for stored artifacts")
    p.add_argument("--save", action="store_true",
                   help="store results as <label>.json + <label>_report.txt in data/validation/team_ratings_oot/")
    p.add_argument("--compare", default=None,
                   help="label of stored run to compare against (default: 'baseline' when present)")
    p.add_argument("--no-equivalence-check", action="store_true",
                   help="skip the full-history replication vs team_ratings_history.csv")
    p.add_argument("--match-results", type=Path, default=None,
                   help="experiment input: read this file instead of data/current/sourced/match_results.csv "
                        "(the equivalence check then needs --no-equivalence-check; CHANGELOG §81)")
    p.add_argument("--dump-predictions", type=Path, default=None,
                   help="write every held-out team-match's frozen and walk-forward predictions to this CSV")
    return p.parse_args()


def main():
    global MATCH_RESULTS_FILE
    args = parse_args()
    if args.match_results is not None:
        if not args.no_equivalence_check:
            raise SystemExit("--match-results needs --no-equivalence-check: team_ratings_history.csv "
                             "was built from the production file")
        MATCH_RESULTS_FILE = args.match_results
    overrides = {k: v for k, v in (("alpha", args.alpha), ("beta", args.beta),
                                   ("home_advantage", args.home_advantage),
                                   ("money_weight", args.money_weight)) if v is not None}
    params = {"alpha": alpha, "beta": beta,
              "home_advantage": home_advantage, "money_weight": money_weight}
    params.update(overrides)
    params_are_defaults = not overrides
    cutoffs = [int(c.strip()) for c in args.cutoffs.split(",") if c.strip()]

    df, prior_map, data_meta = load_inputs()
    seasons_avail = sorted(int(s) for s in df["Season"].unique())
    for c in cutoffs:
        if c not in seasons_avail:
            raise SystemExit(f"cutoff season {c} not present in match_results.csv (seasons: {seasons_avail})")
        if not (df["Season"] < c).any():
            raise SystemExit(f"cutoff season {c} has no training matches before it")

    say("=" * 60)
    say("TEAM RATINGS OUT-OF-TIME EVALUATION")
    say("=" * 60)
    say(f"  label: {args.label}")
    src = "update_team_ratings.py defaults" if params_are_defaults else f"OVERRIDES {overrides}"
    say(f"  params ({src}): alpha={params['alpha']}, beta={params['beta']}, "
        f"home_advantage={params['home_advantage']:.9f}, money_weight={params['money_weight']}")
    say(f"  data: match_results.csv {data_meta['match_results']['rows']} rows "
        f"(md5 {data_meta['match_results']['md5'][:10]}..., {data_meta['match_results']['mtime']}), "
        f"money_priors {data_meta['money_priors']['rows']} rows")
    say(f"  cutoffs: {cutoffs}  (train Season < C as normal, evaluate Season == C only)")

    chrono = check_chronology(df)
    say(f"  chronology: file order verified for seasons "
        f"{chrono['verified_seasons'][0]}-{chrono['verified_seasons'][-1]} via fbref links "
        f"({len(chrono['violations'])} violations); {chrono['unparseable_links']} linkless rows "
        f"(FPL_API seasons) trusted in file order, as production does")
    if chrono["violations"]:
        say(f"  WARNING: seasons {chrono['violations']} violate chronological file order!")

    # Fidelity: replicate the production run and compare to its artifact.
    _, equiv = equivalence_check(df, prior_map, params, params_are_defaults,
                                 run_it=not args.no_equivalence_check)
    say()
    if equiv.get("performed"):
        status = "PASS" if equiv["passed"] else "FAIL"
        say(f"  equivalence check vs team_ratings_history.csv: {status} "
            f"({equiv['rows_compared']} rows, max |diff| = {equiv['max_abs_diff']:.3e} across "
            f"prediction + rating columns)")
        if not equiv["passed"]:
            say("  FAIL means the harness's replication has drifted from production --")
            say("  do not trust these numbers until fixed.")
    else:
        say(f"  equivalence check: SKIPPED ({equiv['reason']})")
    if "reference_insample_mae" in equiv:
        say(f"  in-sample reference: one-step-ahead MAE over the FULL history = "
            f"{equiv['reference_insample_mae']:.4f} (what update_team_ratings.py prints; NOT an OOT number)")
    say()
    say("  modes: frozen = ratings frozen at the cutoff boundary, held-out never updates")
    say("         walk-forward = unchanged predict-then-update loop continues through the")
    say("         held-out season (production-realistic; secondary). ave_goals is")
    say("         train-only in both, so no held-out mean leaks into anything.")
    say("  metrics: Poisson LL kernel = pred - actual*ln(pred) per team-match (primary; lower")
    say("           = better; ln(actual!) dropped as constant across models). MAE secondary.")
    say(f"           Clean-sheet LL = exp(-{CS_PEN_GOAL_MULT:.4f} x pred_opponent_NPG) vs actual goals")
    say("           conceded == 0 -- the production definition (build_simulation_features.py).")

    # ── Per-split evaluation ──
    splits = {}
    dumped = []
    pooled = {mode: {"ph": [], "pa": [], "yh": [], "ya": [], "gh": [], "ga": []} for mode in ("frozen", "walk_forward")}
    season_min = seasons_avail[0]

    for col in ("Home_Goals", "Away_Goals"):
        if df[col].isna().any():
            raise SystemExit(f"{MATCH_RESULTS_FILE.name}: NaN in {col} (needed for the clean-sheet metric)")

    for c in cutoffs:
        train = df[df["Season"] < c]
        ave = float(pd.concat([train["Home_NPG"], train["Away_NPG"]]).mean())
        run_df = df[df["Season"] <= c]
        res, snaps = run_sequential(run_df, ave, params, prior_map, snapshot_seasons={c})
        if c not in snaps:
            raise SystemExit(f"internal error: no snapshot captured for cutoff {c}")
        val = res[res["Season"] == c]
        yh = val["Home_NPG"].values.astype(float)
        ya = val["Away_NPG"].values.astype(float)
        gh = val["Home_Goals"].values.astype(float)
        ga = val["Away_Goals"].values.astype(float)

        f_ph, f_pa = frozen_predict(val, snaps[c], prior_map, ave, params)
        f_sum, f_probs, f_events = summarize_predictions(f_ph, f_pa, yh, ya, gh, ga)
        wf_ph = val["Home_Prediction"].values.astype(float)
        wf_pa = val["Away_Prediction"].values.astype(float)
        wf_sum, wf_probs, wf_events = summarize_predictions(wf_ph, wf_pa, yh, ya, gh, ga)

        if args.dump_predictions is not None:
            dumped.append(pd.DataFrame({
                "Cutoff": c, "Home": val["Home"].values, "Away": val["Away"].values,
                "Frozen_Home": f_ph, "Frozen_Away": f_pa, "WF_Home": wf_ph, "WF_Away": wf_pa,
                "Home_NPG": yh, "Away_NPG": ya, "Home_Goals": gh, "Away_Goals": ga}))
        splits[str(c)] = {
            "train_matches": int(len(train)),
            "ave_goals_train": ave,
            "frozen": f_sum,
            "walk_forward": wf_sum,
        }
        pooled["frozen"]["ph"].append(f_ph); pooled["frozen"]["pa"].append(f_pa)
        pooled["frozen"]["yh"].append(yh); pooled["frozen"]["ya"].append(ya)
        pooled["frozen"]["gh"].append(gh); pooled["frozen"]["ga"].append(ga)
        pooled["walk_forward"]["ph"].append(wf_ph); pooled["walk_forward"]["pa"].append(wf_pa)
        pooled["walk_forward"]["yh"].append(yh); pooled["walk_forward"]["ya"].append(ya)
        pooled["walk_forward"]["gh"].append(gh); pooled["walk_forward"]["ga"].append(ga)

        partial = " -- partial season, low power" if len(val) < 300 else ""
        say()
        say(f"--- Split: cutoff {c} | train {season_min}-{c - 1} ({len(train)} matches) | "
            f"held-out season {c} ({len(val)} matches / {f_sum['n_team_matches']} team-matches){partial} ---")
        say(f"  ave_goals(train-only) = {ave:.5f}")
        for mode, s in (("frozen -- PRIMARY", f_sum), ("walk-forward -- secondary", wf_sum)):
            say(f"  [{mode:<27s}] Poisson LL {s['poisson_ll_sum']:>9.2f} sum | "
                f"{s['poisson_ll_per_tm']:.4f} /tm | MAE {s['mae']:.4f} | CS LL {s['cs_ll']:.4f} | "
                f"CS obs {s['obs_cs_rate'] * 100:.1f}% vs pred {s['pred_cs_rate'] * 100:.1f}%")

    if dumped:
        pd.concat(dumped, ignore_index=True).to_csv(args.dump_predictions, index=False)
        say(f"  wrote held-out predictions to {args.dump_predictions}")

    # ── Pooled + reliability bins ──
    pooled_sums, pooled_bins = {}, {}
    for mode in ("frozen", "walk_forward"):
        p = pooled[mode]
        s, probs, events = summarize_predictions(
            np.concatenate(p["ph"]), np.concatenate(p["pa"]),
            np.concatenate(p["yh"]), np.concatenate(p["ya"]),
            np.concatenate(p["gh"]), np.concatenate(p["ga"]))
        pooled_sums[mode] = s
        pooled_bins[mode] = cs_bin_table(probs, events)

    say()
    say(f"--- Pooled held-out ({cutoffs[0]}-{cutoffs[-1]}: "
        f"{pooled_sums['frozen']['n_matches']} matches / "
        f"{pooled_sums['frozen']['n_team_matches']} team-matches) ---")
    for mode, s in (("frozen -- PRIMARY", pooled_sums["frozen"]),
                    ("walk-forward -- secondary", pooled_sums["walk_forward"])):
        say(f"  [{mode:<27s}] Poisson LL {s['poisson_ll_sum']:>9.2f} sum | "
            f"{s['poisson_ll_per_tm']:.4f} /tm | MAE {s['mae']:.4f} | CS LL {s['cs_ll']:.4f} | "
            f"CS obs {s['obs_cs_rate'] * 100:.1f}% vs pred {s['pred_cs_rate'] * 100:.1f}%")

    for mode in ("frozen", "walk_forward"):
        say()
        say(f"  Clean-sheet reliability, pooled held-out ({mode}):")
        say(f"    {'P(CS) bin':<18s}{'n':>7s}{'pred':>9s}{'obs':>9s}{'gap':>9s}")
        for r in pooled_bins[mode]:
            say(f"    {r['bin']:<18s}{r['n']:>7d}{r['pred']:>9.3f}{r['obs']:>9.3f}"
                f"{r['obs'] - r['pred']:>+9.3f}")

    # ── Assemble run record ──
    current = {
        "label": args.label,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "params": params,
        "param_overrides": overrides,
        "params_are_production_defaults": params_are_defaults,
        "data": data_meta,
        "cutoffs": cutoffs,
        "chronology_check": chrono,
        "equivalence_check": equiv,
        "splits": splits,
        "pooled": pooled_sums,
    }

    # ── Comparison against a stored run ──
    compare_label = args.compare
    if compare_label is None and args.label != "baseline" and (OUT_DIR / "baseline.json").exists():
        compare_label = "baseline"
    if compare_label:
        cpath = OUT_DIR / f"{compare_label}.json"
        if cpath.exists():
            with open(cpath, encoding="utf-8") as f:
                do_compare(json.load(f), current, compare_label)
        else:
            say()
            say(f"  (no stored run '{compare_label}' at {cpath} -- nothing to compare)")

    # ── Save artifacts ──
    if args.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        jpath = OUT_DIR / f"{args.label}.json"
        rpath = OUT_DIR / f"{args.label}_report.txt"
        if jpath.exists():
            say(f"  (overwriting existing {jpath.name})")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
        with open(rpath, "w", encoding="utf-8") as f:
            f.write("\n".join(_REPORT_LINES) + "\n")
        say()
        say(f"  Saved {jpath}")
        say(f"  Saved {rpath}")


if __name__ == "__main__":
    main()
