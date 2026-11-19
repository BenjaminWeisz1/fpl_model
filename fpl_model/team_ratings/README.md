# Team ratings

Exponential-smoothing attack and defence ratings for every Premier League
team, updated match by match from non-penalty goals, and used to forecast
goals in upcoming fixtures.

## Production scripts

### red_card_adjust.py

Part-1 stage (`--apply`) before the ratings update (CHANGELOG §124, §133). Using the cached Understat shots, it writes
`NPxG_Adj` for every red-card match: base npxG × a shot-weighted mix of 0.8 for the side with more players and 1.2 for
the side with fewer. It recomputes from the base every run and leaves matches without a red card untouched; factors go
to `red_card_factors.csv`. Related tools:
- `estimate_red_card_factors.py` measures the factors (§134: 0.66 / 1.95).
- `validate_red_card_variants.py` is the OOT comparison (§135: no gain from changing 0.8 / 1.2).
- `diagnose_red_card_info.py` tests whether a red card predicts the next match (§137: no).

### update_team_ratings.py

The production ratings engine. Replays every match in
`data/current/sourced/match_results.csv` (2017–present) in chronological order,
updating each team's attack and defence rating with exponential smoothing
after every match:

```
alpha = 0.045          # smoothing weight per match
beta = 0.8             # attack/defence split of the update
home_advantage ≈ 1.084 # multiplicative home-edge on expected goals
money_weight = 0.3     # weight of the money prior in the blended rating
```

Money priors (`data/current/lookups/money_priors_by_season.csv`) are team-strength anchors blended
into the ratings at each season start (promoted teams start at 100% money).
They come from `data/current/sourced/historical_money.csv`: each club's Transfermarkt squad value at
the 15 August cut-off (the club market-value page with `?stichtag=`, which
leaves out later signings) and its fbref wage bill. Each season's prior is a
regression of season goals for and against on the clubs' wage and value
shares, fitted on earlier seasons only. Outputs:

- `data/current/state/team_ratings_history.csv` — ratings snapshot after every match
- `data/current/state/team_ratings_current.csv` — the latest per-team ratings

### compute_preseason_ratings.py

Season-start only (the `--preseason` flag of `run_post_gameweek.py`). Blends
end-of-season ratings with next-season money priors for continuing teams and
uses money-only priors for newly promoted teams. Output: `data/current/state/team_ratings_preseason_2026.csv`.

### predict_fixture_goals.py

Forecasts non-penalty goals for and against each team for the next ten
rounds (the forecast window (`fpl_model/forecast_horizon.py`, 10 gameweeks since GW11 of 2026/27)) of `fixtures.csv` from the current ratings and home advantage.
Output: `data/current/forecasts/fixture_goal_predictions.csv` — the fixture-strength input every
downstream facet (penalties, clean sheets, bonus simulation) consumes.

### get_next_round.py

Small helper (imported by the defcon model and others) that determines the
next round to predict from `match_results.csv` and `fixtures.csv`.

## Periodic utilities

- `build_money_priors_lookup.py` — the original one-time extraction of money
  priors. Superseded: never rerun it (it fits in-sample).
- `rebuild_money_priors_past_only.py` — rewrites the completed seasons'
  priors (2017–2025; extend its `REBUILD` range each season) from earlier
  seasons only; dry run by default,
  `--apply` writes. Rerun `update_team_ratings.py` afterwards.
- `update_money_priors.py` — refreshes the current season's priors when
  `historical_money.csv` changes (ridge regression, via a companion script
  kept in `data/archive/`, outside this repo). It rewrites the whole priors
  file through pandas, which reformats the other seasons' numbers.

## Experiment and diagnostic scripts

These are measuring tools, not part of the weekly pipeline; none of them
modifies the production ratings:

- `ratings_fast.py` — a dict-based port of the production sequential loop
  (~800x faster), with `verify_against_reference()` fidelity checks.
  Imported as a module by the scripts below.
- `evaluate_team_ratings_oot.py` — out-of-time evaluation harness with
  rolling-origin cutoffs (2023–2026), frozen and walk-forward modes.
- `tune_team_ratings.py` — grid search over (alpha, beta, money_weight)
  scored by pooled Poisson log-loss on the held-out seasons; report-only,
  shipping a new parameter set is a separate manual decision.
- `kalman_team_ratings.py` — state-space (Kalman filter) reformulation
  experiment.
- `symmetric_update_team_ratings.py` — symmetric log-space update
  experiment.
- `discount_team_ratings.py` — promoted-team entry-discount experiment.
- `rebalance_team_ratings.py` — season-boundary rebalancing experiment.
