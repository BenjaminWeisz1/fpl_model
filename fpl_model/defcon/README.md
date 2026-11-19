# Defcon: defensive contribution model

Predicts the probability that a player's defensive contribution in an
upcoming fixture clears a position-specific threshold — the "Defcon" facet
of the points pipeline.

## What it predicts

A player's defensive contribution is a position-specific stat combination:

- **Defenders**: clearances + blocks + interceptions + tackles, threshold 10
- **Midfielders and forwards**: the same actions + recoveries, threshold 12

The model estimates the probability of reaching the threshold **per 90
minutes** (matches with under 60 played minutes count as automatic
negatives), then rescales to the player's expected minutes for each fixture.

## How it works

The stage is `predict.py` (`python -m fpl_model.defcon.predict`), which runs the steps below and imports its parts:

| Module | Job |
|---|---|
| `predict.py` | Loads the data, builds the training frames, runs the validation battery, trains the three heads and writes the forecasts |
| `features.py` | Position classification and the per-match feature helpers (cameo filters, prior-blended spread, the nearest-neighbour estimator) |
| `recent_defcon.py` | The causal EW recent rate, shared by training and serving, and the feature schemas |
| `defender_model.py` | The locked monotonic defender configuration, league-level adjustment and centred isotonic calibrator |
| `opponent_model.py` | The §149 opponent factor |
| `model_cache.py` | The calibrated-model wrapper, `tune_and_calibrate` and the trained-model cache |
| `evaluation.py` | Out-of-time metrics: log loss, AUC, Brier and reliability by evidence bucket |

On every run it:

1. Loads the full match history from `data/current/sourced/player_match_stats.csv`
   — fbref rows 2017–2024 plus FPL-API rows from 2025 on. FPL-API defensive
   stats live on a different scale than fbref's, so they are reconciled to
   the fbref scale at load time (position-specific multipliers).
2. Builds per-player evidence features (per-90 and raw averages, recency form,
   threshold hit-rate) blended with empirical-Bayes
   **transfer priors** for new-to-league arrivals from `transfers.csv`, plus
   recency-weighted opponent/team/venue aggregates. The serving copies of those
   averages (the values used for the forecast) also weight each season by its
   number of rows (§195); the training copies
   keep equal season votes, which tested better.
3. Trains three classifier heads, all with `random_state=42`:
   - **Defenders**: monotonic histogram gradient boosting (sklearn
     `HistGradientBoostingClassifier`), with depth 3, minimum 50 rows per leaf,
     L2 regularization, and increasing constraints on career/recent activity
     averages and career threshold hit-rate. Holding other features fixed,
     weaker evidence cannot increase the probability. Experience,
     position and match context remain unconstrained.
     The defender head excludes `StdPer90` and both recent-five features. It
     instead uses `RecentEWPer90`: exponentially weighted actions divided by
     exponentially weighted minutes/90, with a four-qualifying-appearance
     half-life. Sub-60-minute cameos do not update or age this rate. A new
     transfer prior seeds the recent state at arrival; youth/transfer fallbacks
     use their baseline rate. The raw career average is retained after testing.
     Since §106, `AvgPer90`, `AvgRaw` and `RecentEWPer90` are expressed at the
     current league level: each is scaled by last season's mean defender per-90
     over the mean level of the seasons the player's history comes from
     (`defender_model.season_levels` / `level_factors`, shared by training and
     forecasts). The level fell from 9.1 (2017) to 7.3 (2020–21) and recovered;
     without this, long careers lag it and experienced defenders were under-predicted.
     The defender calibrator is centred isotonic (§107, `defender_model.CenteredIsotonic`):
     it interpolates between isotonic block centres, so defenders no longer share
     identical step values.
   - **Midfielders**: random forest
   - **Forwards**: a separate, forward-only random forest (forwards clear
     the threshold far more rarely than midfielders and deserve their own fit)
   Defender and midfielder heads use `OppFactor` instead of `OpponentAvgStat` (§149): the actions an opponent has conceded ÷ the pre-match expected actions of the
   players who faced it, by position group (CB, wide defender, central mid, other mid), half-life 19 opponent matches, shrunk 20 pseudo rows, relative to the league.
   It updates through the season; the old feature used prior seasons only. Shared code: `opponent_model.py`. Forwards keep `OpponentAvgStat`.
4. Calibrates each head with isotonic regression fitted on out-of-fold
   predictions (5-fold stratified), then refits the base model on all data.
5. Runs an out-of-time validation battery (validate-on-2025 and
   validate-on-2026 splits, including a forward head-to-head against the
   shared mid/forward model) and prints the results — this runs every time
   so the serving decisions stay honest as data accumulates.

Players with no league history resolve through three inference tiers:
matchlog history → transfer prior → youth defaults.

Minutes scaling: the raw model output is the 90-minute probability; per
fixture it is rescaled through a negative-binomial survival function
(r = 3μ, p = 0.75) applied to the expected minutes from the minutes
forecast.

## Outputs

| File | Contents |
|---|---|
| `data/current/forecasts/predictions_defensive.csv` | Per-fixture defcon probabilities (minutes-scaled) |
| `data/current/forecasts/defcon_average_probabilities.csv` | Fixture-neutral 90-minute probability (home/away averaged) — feeds `player_ratings.csv` |
| `cache/defcon/` | Trained-model cache (pickles; regenerable, not in the repo) |

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `DEFCON_TUNE` | `0` | `1` runs hyperparameter search (~5 h); `0` uses the locked defaults |
| `DEFCON_MODEL_CACHE` | `1` | `0` disables the trained-model cache |
| `DEFCON_RF_N_JOBS` | `-1` | Cores for random-forest fits (result-invariant) |

These are declared in `fpl_model/config.py`. The model's settled revert switches (monotonic defender head, feature
schema, forward head, centred isotonic, opponent factor, count head, level adjustment, cameo filters, row-weighted
serving, detailed positions) were retired in the 2026-11-16 cleanup; their behaviour is now fixed. Minutes below 15
count as cameos for the raw averages (`RAW_CAMEO_MIN`) and the transfer prior is capped at 10 matches (`PRIOR_MAX_K`).

## Runtime

About 20 minutes on fresh data (nine model trainings: three production heads
plus six validation splits); around 6 minutes when the trained-model cache
hits unchanged training data. Any edit to a model module (`MODEL_SOURCES` in `model_cache.py`, plus
`defender_model.py`) invalidates the cache keys and forces a full retrain once.

## Other scripts

- `merge_defcon_into_transfers.py` — merges prior-season defcon data for
  new arrivals into `transfers.csv` (maintains the transfer-prior table).
- `evaluate_defender_monotonic.py` — the §41 pre-registered defender comparison;
  preserves the control before edits, checks exact training-frame cache keys,
  reports held-out metrics and player-cluster bootstrap intervals, and audits
  isolated evidence updates. Evidence is in `experiment_20261020/`.
- `evaluate_defender_recency.py` (§105, not adopted) and `evaluate_defender_level.py`
  (§106, adopted) — out-of-time tests on the saved §46 frame, evidence in
  `recency_experiment_20261029/` and `level_experiment_20261029/` (the latter also holds
  the pre-edit baseline of `predict.py`, its outputs and model cache).
- `evaluate_feature_changes.py` — §46 fixed-schema ablations and locked chronological
  selection/confirmation, with evidence under `feature_experiment_20261020/`.

**Scale warning (§152–§153).** The target is per-90 at the fbref level, not FPL's raw count. Against FPL's real hits at realized minutes, 2025–26, forecasts run: defenders 0.175 v 0.219 actual, midfielders about 0.11 v 0.103, forwards 0.007 v 0.003. A defender correction (factor 1.083) narrowly failed its pre-registered rule; see CHANGELOG §153. Score any new head on FPL hits as well as on the target.

## Defender validation, 2026-10-20

The replacement passed the pre-registered rule on held-out 2025 and partial 2026
(GW1–7). Log loss moved 0.482282 → 0.477232 and 0.455699 → 0.453604 respectively.
All five increasing-evidence audits had zero reversals, and joint falling-evidence
reversals fell from 130 to 0 (2025) and 20 to 0 (2026). The 2025 95th-percentile
absolute isolated evidence update fell from 6.86 to 5.26 percentage points.

This is not an improvement on every metric: 2025 Brier worsened 0.154606 →
0.156055 and ECE worsened 0.026237 → 0.040572; 2026 Brier improved. Both player-
cluster log-loss confidence intervals include zero. The evidence supports fixing
the backwards response with acceptable held-out accuracy, not a proven universal
accuracy gain. Isotonic calibration still has flat regions and jumps; constraints
do not guarantee small weekly changes when nuisance/context features or the model
itself change. See CHANGELOG §41 for the complete decision and limitations.

## Feature simplification, 2026-10-20 (§46)

The default defender schema is now `no_std_ew` (15 inputs): no spread input,
one EW recent rate instead of both recent-five windows, with AvgRaw retained.
The existing history columns are still exported for audit; their presence in
the fixture CSV does not mean the defender model uses them.

Four fixed schemas were compared on held-out 2022–24. The selected combination
improved pooled LL .469948 → .467864 and Brier .151390 → .150661, with paired
player-cluster intervals favoring it. Removing AvgRaw weakened the result, so
it stayed. The selected schema then passed the previously fixed confirmation
limits: 2025 LL .477232 → .475428 and Brier .156055 → .155203; partial 2026 LL
.453604 → .455295 and Brier .146695 → .147400 (both worse, within the .002
limits). Current-season AUC .762681 → .754048 and ECE .031873 → .043665 also
worsen. These tradeoffs are retained, not described as improvements.

Direct sensitivity to StdPer90 is exactly zero, and old games no longer expire
from the recent-rate input. However, the expected broad reduction in weekly
prediction movement was not confirmed: p95 personal-feature updates increased
in four of five splits. The change fixes two feature mechanisms and has
acceptable comparative accuracy; it does not establish universally smoother
probabilities. Role changes, match count, tree bins, calibration and refits
can still cause jumps. No new position or calibration changes were bundled.

## Prospective check: GW8–12 of 2026

The update is already adopted for next gameweek; completing GW8–12 is not an
adoption requirement. The forward check is ongoing monitoring. A separate immutable comparison
preserves both GW7-fitted heads and their identical-input forecasts for GW8–12
under `data/validation/defcon_forward_20261020/`. This tests frozen forecasts
on unseen outcomes; normal weekly production refits continue independently.
`protocol.json` fixes the target, metrics, expected effect and assessment rule.
The forecast, input and model files are protected by SHA256 hashes.

`run_post_gameweek.py` runs `fpl_model/defcon/forward_check.py score` after fetching
player stats and match results. It never trains, changes forecasts or switches
the production model. `reports/latest.json` contains completed-match coverage,
paired log loss/Brier, calibration, per-round scores and separate player/fixture
cluster bootstrap intervals. No-appearance rows are excluded, not labelled as
threshold failures. The primary target exactly matches training (>=60 minutes
and per90 actions >=10); the 60+-minute subset is reported separately. The
90-minute probabilities are used, avoiding the stale minutes-scaled column.

The main assessment waits for all 50 fixtures plus complete stats, >=300 scored
appearances and >=20 hits. Interim scores cannot trigger a winner or stopping.
An inconclusive result retains the update; convincing deterioration is flagged
for review without automatic rollback. Run the scorer manually with:

```
python -m fpl_model.defcon.forward_check score
```

The initial snapshot was created with `forward_check.py freeze`; running that
command again verifies the existing snapshot and cannot replace it. The scorer
does not depend on the regenerable production model cache. No background or
scheduled job was created; the normal post-gameweek command performs the check.

For a decision using currently available evidence, `evaluate_defender_history.py`
adds fixed-configuration chronological tests on 2022, 2023 and 2024 to the
existing 2025/26 results. Each fit and calibration uses earlier seasons only.
The locked protocol is `experiment_20261020/historical_preregistration.json`.
These are retrospective checks, not pristine independent test sets: historical
data informed earlier pipeline development, and the earlier seasons use fbref
rather than today's FPL source. They do not replace future monitoring.

The additional tests completed on 2026-10-20 and support **keeping the update
now**. Both LL and Brier improve in each of the three seasons. Across 11,899
held-out appearances, LL moves .479337 → .469948 and Brier .152891 → .151390;
paired player-cluster 95% intervals for both differences favor the update.
ECE worsens in the pooled results, so improved calibration is not claimed.
Full results and limitations are in CHANGELOG §43 and
`experiment_20261020/historical_results.json`.

## Points calibration

Defender forecasts ran under FPL's real `Defensive_Contribution` hits, so `fpl_model/defcon_scaling.py` multiplies a defender's 90-minute mean by 1.083 for defcon points, the defcon charts and `player_ratings.csv`. Bonus keeps the raw mean. It's applied downstream, so this model's own outputs are unchanged. `DEFCON_POINTS_CALIBRATION=0` reverts it; refit it when a season ends (CHANGELOG §153, §159).
