# Defcon fixes — pre-registration (2026-10-13)

Written **before** any run of the modified code. Baseline captured first from the
unmodified `predict.py` (`_defcon_fix/baseline.log`), so the control is fixed.

## Baseline (unmodified code, current data)

| Split | Arm | n | Log loss | AUC | Brier | ECE |
|---|---|---|---|---|---|---|
| A (train <2025, val 2025) | Defender | 3953 | 0.4819 | 0.7164 | 0.1572 | 0.0248 |
| A | Midfielder/Forward | — | — | 0.8056 | 0.1060 | 0.0200 |
| B (train <2026, val 2026) | Defender, priors ON | 640 | 0.4476 | 0.7692 | 0.1440 | 0.0363 |
| B | Defender, priors OFF | 640 | — | 0.7512 | 0.1460 | 0.0318 |

Prior-carrying defender rows (Split B, n=119): mean pred OFF 0.2123, ON 0.2532,
observed 0.2689.

Bobby Thomas, fixture-neutral: **0.4313** after a 22-defcon match, down from
**0.5049** before it (same model, verified by counterfactual scoring).

## The three changes

**F1 — cameo filter on `Recent5AvgPer90`.** It is the only per-90 feature without
the ≥60-minute qualifying bar. 40.3% of defender last-5 windows contain a sub-60'
match and ~84% of windows above 14 per 90 are bench cameos, so the model learned a
*negative* response to recent form: raising the feature from 10 to 14 lowers the
prediction for 85.5% of served defender rows, while the cameo-filtered empirical
relationship keeps rising (0.459 → 0.468 overall, 0.507 → 0.611 among starters).
The feature now uses the same ≥60' bar as every other per-90 feature, falling back
to career `AvgPer90` when fewer than 2 of the last 5 qualify (mirroring
`Recent5AvgRaw` → `AvgRaw`). Applied identically in training and serving.

**F2 — credibility cap on the transfer prior.** `k = Min_Old / 90`, uncapped, gave
a full Championship season ~32 pseudo-matches. Measured credibility on the 2026
arrival cohort (31 players with ≥3 qualifying PL matches): within-player per-match
variance 12.36 ÷ prior-error variance 1.41 → **k\* = 8.8**. Capped at 10.

**F3 — data: 341 current-season rows carrying `Season = 2025`.** The FPL-era block
keys completed seasons by gameweek id (38 ids, ~300 rows each) but writes one id
per row for the live season; `backfill_season_column.py` stamped the 341 per-row
entries then in the file (MatchID 3079–3419) with the previous season. They became
invisible to the fetcher's `(Season, Player, Opponent, Venue)` dedup key, so the
same matches were appended again under 2026: **337 of the 341 are duplicates of a
2026 row, 336 stat-for-stat identical**. They double-count matches in every
affected player's history, pollute the Split A validation set with current-season
rows, and sort after the entire real previous season.

## Predictions (stated now, before running)

**F1**, measured on Split A defender — no transfer priors exist before 2026, so
Split A isolates F1 from F2 cleanly:
- Log loss **improves** (falls) by **0.002–0.015**. It is one of ~17 features and
  the GBM can partly route around it via `AvgRaw`/`AvgPer90`, so a large move would
  be surprising.
- AUC **up** by 0.002–0.015.
- The partial dependence on `Recent5AvgPer90` becomes **monotone non-decreasing
  over 8 → 20 per 90** (baseline: peaks at 8 and falls from 0.238 to 0.186).

**F2**, measured on Split B prior-carrying defender rows (n=119):
- Their mean prediction moves **up**, toward the observed 0.2689 (baseline ON:
  0.2532), because the cohort's realised PL rate (8.82) exceeds its prior (8.45)
  and capping lets their own evidence carry more weight.
- The `MatchesPlayed` bucket migration weakens (baseline ON put 68 rows in 10–29
  and 48 in 30–79; with k ≤ 10 the 30–79 bucket should largely empty).
- Low power at n=119, so F2 is justified by the measured k\*, not by this split.

**F3**: Split A validation n falls (it currently contains current-season rows).
|Δ log loss| < 0.01. Justified by correctness regardless of the metric.

**Bobby Thomas** (the "makes sense" test): his duplicated Arsenal match disappears
(7 → 6 qualifying matches) and his prior weight falls 32.5 → 10, so `AvgPer90`
rises 9.21 → ≈10.07 while `Recent5AvgPer90` stays 13.0 (his last five are all
≥60'). Predicted fixture-neutral defcon **≥ 0.50**, and — the actual requirement —
**his post-22 value must exceed his pre-22 value under the fixed model**.

## Ship rules (fixed now)

- **F1 ships** iff Split A defender log loss does not worsen by more than 0.002
  (the same noise margin the forward-model decision used) **and** the
  `Recent5AvgPer90` partial dependence is monotone non-decreasing over 8 → 20.
- **F2 ships** unless Split B defender log loss worsens by more than 0.01.
- **F3 ships** unconditionally — it is a data-correctness fix, not a model change.
- If F1 fails its rule, it is reverted and reported as rejected, F2/F3 kept.

## Arms to run

1. **Attribution arm** — F3 applied, F1/F2 disabled via `DEFCON_RECENT5_CAMEO=0`
   and `DEFCON_PRIOR_MAX_K=0`. Isolates the data fix: (arm 1 − baseline) = F3.
2. **Production arm** — all three at their defaults. (arm 2 − arm 1) on Split A is
   pure F1; on Split B it is F1+F2.

---

# F4 — cameo filter on the opponent/team/venue aggregates (added 2026-10-13, after arm 2)

Written **before** the F4 run. Arm 2 is the control; F4 is the only change.

## What arm 2 exposed

With F3 applied, `TeamAvgStat` for Coventry is **14.789** — the next highest team is
Everton at 10.73 and the league mean is ~8.8. Coventry's defenders then take the top
six places in the defcon chart, including Aurèle Amenda at 0.504 who has cleared the
threshold in **0 of 4** qualifying appearances (mean 6.0 per 90).

Cause: `compute_season_aggregates` averages `per90_col` over **all** rows, cameos
included. Joel Latibeaudiere played 1 minute against Nott'ham Forest and made 2
defensive actions — 180.0 per 90. That single row lifts Coventry's team aggregate
from 8.50 (qualifying appearances only) to 14.79. It is the same cameo-extrapolation
defect as F1, in a third place: the function's own docstring says it moved
opponent/team aggregates onto per-90 "for consistency", which is what exposed them.

F3 did not cause this — the duplicated 2025 rows were diluting the outlier, so removing
them concentrated it. Baseline Coventry was 12.39, itself already an outlier.

## The change

`subset = matchlogs[pos_mask]` gains the `Minutes >= 60` qualifying bar, so all six
aggregates (`prior_*` and `final_*`, feeding `OpponentAvgStat`, `TeamAvgStat`,
`VenueAvg`) are built from full appearances only. `DEFCON_AGG_CAMEO=0` restores the
old behaviour.

## Predictions (stated now)

- Coventry's `TeamAvgStat` falls to ≈8.5 and is no longer an outlier; the spread of
  `TeamAvgStat` across the 20 teams narrows materially.
- Coventry no longer occupies the top six defender slots. **Amenda falls below 0.40**
  (0 of 4, 6.0 per 90). Thomas and Pinnock, who have earned their rates, stay above 0.45.
- Split A defender log loss: improvement of **0.000–0.010**. Established teams average
  hundreds of rows so their aggregates barely move; the effect concentrates on teams
  and seasons with few rows.
- This is the fix I expect to matter most for accuracy of the three feature fixes,
  because it is the only one whose corruption is this extreme in magnitude.

## Ship rule

F4 ships iff Split A defender log loss does not worsen by more than 0.002 **and**
Coventry's `TeamAvgStat` is no longer more than 1.5× the second-highest team's.
