# Predicted points assembly

The final stage of the pipeline: combines every facet into per-player,
per-gameweek predicted FPL points for the forecast window (`scripts/forecast_horizon.py`, 10 gameweeks since GW11 of 2026/27).

## Scripts

### build_simulation_features.py

Assembles the consolidated per-player-per-gameweek feature table
`data/current/simulation_features.csv`, joining:

- appearance and 60+-minute probabilities (`minutes_probabilities.csv`),
- goal/assist ratings (`player_goal_assist_ratings.csv`),
- fixture-level defcon probabilities (`predictions_defensive.csv`),
- penalty predictions (`penalty_predictions.csv`),
- clean-sheet probabilities (`fixture_goal_predictions.csv` via team
  ratings), and
- the residual BPS baseline and position defaults
  (`residual_bps_baseline.csv`, `residual_bps_position_defaults.csv`).

Each row also carries its fixture opponent and venue for the next-gameweek
leaderboards.

The bonus simulation (`scripts/bonus/simulate_bonus.py`) consumes this table
— that is why this stage runs before it in the orchestrator.

### build_predicted_points.py

The assembly step. Reads `simulation_features.csv` plus the bonus
predictions and sums every facet into FPL points: appearance points,
non-penalty goals, assists, clean sheets (where position-eligible), penalty
outcomes, defcon threshold hits, bonus points, and goalkeeper saves.

Goalkeeper save rates are pooled from played FPL API goalkeeper history,
identified through `player_match_positions.csv`, scaled by expected minutes and
by the fixture: (expected goals conceded)^0.66, normalised to keep the pooled
level (CHANGELOG §162; `KEEPER_SAVE_FIXTURE=0` reverts). The bonus simulation uses
the same multiplier (§165; `BONUS_SAVE_FIXTURE=0` reverts). Save points use the expected Poisson value of one point per three saves. The
separate five-point penalty-save award is not forecast because its observed rate
does not improve held-out point error over predicting zero.
Goals-conceded deductions use the Poisson expectation of one point per two goals,
with keeper exposure scaled by expected minutes. Goalkeepers do not receive
outfield defensive-contribution points. Penalty-taker points remain in `PenPts`;
that component does not represent penalty saves.

Output: `data/current/predicted_points.csv` — one row per player with one
column per upcoming gameweek (headers are the actual gameweek numbers) and
a `Total` column, sorted by `Total` descending.

The part-2 heatmap export includes a goalkeeper-specific leaderboard at
`images/predicted_points/predicted_points_gk.png`.

## Pipeline position

Both stages run automatically inside `run_pre_deadline.py` ("Simulation
features" then "Assemble predicted points", with "Bonus simulation" in
between). They depend on artifacts from the team-ratings, player-ratings,
defcon, minutes and penalty stages.

## Card points

`build_predicted_points.py` adds `CardPts` per gameweek: -(yellows/90 + 3 x reds/90) x xMins/90, from `scripts/card_points.py` (CHANGELOG 175). A yellow rate is the position rate x the player's shrunk record; a new player gets the position rate. Reds are pooled by position. `CARD_POINTS=0` reverts. Since CHANGELOG 195: starts and cameos have their own yellow rates, mixed per fixture by (P1 - P60) x mean cameo length / raw xMins (`CARD_ROLE_SPLIT=0` reverts; `CARD_ROLE_MIX=trailing` uses the last five club matches instead), and a player one yellow short of a ban before its cut-off gets x 0.851 on the yellow rate (`CARD_SUSPENSION=0` reverts).

## xMins clock

FPL Review's xMins sum to about 1,043 per team; recorded minutes sum to 990 (CHANGELOG 177). `build_simulation_features.py` writes `xMins` on the recorded clock and keeps `xMins_Raw`. The scale is computed each run: 11 x 90 / the median team-gameweek total of FPL Review's xMins (`recorded_clock_scale`). `XMINS_RECORDED=0` reverts.
