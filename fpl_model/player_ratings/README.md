# Player goal/assist ratings

Per-player non-penalty goals per 90 minutes (NPG) and assists per 90
(Assists) — the attacking facet of the points pipeline.

## Scripts

### build_player_match_stats.py

One-off, already run: it refuses to overwrite the file, because a rebuild would delete the current season.
It built the unified match-history table `data/current/player_match_stats.csv`
by concatenating:

- fbref player match stats 2017–2024 (`data/raw/all_player_stats.csv`), and
- FPL-API player match stats from 2025 on
  (`data/raw/player_stats_2025_with_npg.csv`),

with a `Data_Source` column marking each row's origin. Every downstream
player model (goal/assist ratings, defcon, residual BPS) reads this file.

### build_player_ratings.py

The rating engine. Computes NPG/90 and Assists/90 for every player from the
match history, with:

- team-ratings context from `team_ratings_history.csv` /
  `team_ratings_preseason_2026.csv` (how strong the player's team is at
  scoring),
- league coefficients (`league_coefficients.csv`) and position combinations
  (`position_combinations.csv`, single positions only; a multi-position entry such
  as `W,AM,FW` is computed as their weighted average with each listed position
  counting twice the next, 4:2:1) for cross-season comparability,
- shrinkage-corrected conversion rates so low-minute players don't produce
  extreme per-90 values,
- transfer handling via `transfers.csv`,
- new-player priors from `new_player_priors.csv` for arrivals from Ligue 1 and the
  Championship (CHANGELOG §103–§104).

### new_player_priors.py, build_new_player_priors.py

Goal/assist priors for players new to the PL. Arrivals from La Liga, Bundesliga,
Serie A, Ligue 1 and the RFPL use their last three seasons from the frozen
Understat league files (`fpl_model/fetch/fetch_understat_leagues.py`), recency-weighted;
all others use their manual `transfers.csv` values. A fitted league factor (partially
pooled), position-prior shrinkage and new-team factor turn them into a per-90 prior;
FPL-era xAG priors are scaled by 0.921 (FPL expected assists run below fbref xAG).
The parameters are frozen in `new_player_prior_params.json` (`--fit` refits them; that
needs a new pre-registration). The builder runs as a part-1 stage before the ratings
and writes `data/current/new_player_priors.csv`; only rows marked `Adopted` (league
groups that passed the out-of-time test) are used. Championship arrivals found in the frozen FotMob league
lists (`data/raw/fotmob/`, `fpl_model/fetch/fetch_fotmob_leagues.py`) instead use the FotMob-path model in
`fotmob_priors.py` (up to three FotMob seasons; parameters in `new_player_prior_params_fotmob.json`,
CHANGELOG §116). Since §127 every blank-Notes Championship arrival (and,
since §139 by user decision, every La Liga / Bundesliga / Serie A arrival) uses that structure refit on the β-dampened loop (`new_player_prior_params_refit.json`, `--fit-refit`), and tagged Championship arrivals (mid-season, two leagues) get the fitted prior too
(§130). Hand-found FotMob IDs live in `data/current/fotmob_id_overrides.csv`.
Loop settings: starting rating worth K = 16 matches (§112, `GOAL_MODEL_PRIOR_K`); opponent factor dampened to
(concede / mean)^β with β = 0.7 npxG / 0.8 xAG, in the loop and in `build_simulation_features.py` (§123,
`GOAL_MODEL_OPP_BETA_NPG` / `_AST`); learning-rate floor 0.035 for npxG (§233, `GOAL_MODEL_ALPHA_NPG`;
`=0.025` reverts), 0.018 for centre-backs' npxG by match position (§235, `GOAL_MODEL_ALPHA_NPG_CB`; `=0.035` reverts) and 0.025 for xAG (`GOAL_MODEL_ALPHA`), the same for every position (a per-position floor failed in
§196, `validate_position_alpha.py`; the npxG floor was checked in `validate_npxg_floor.py`). Set pieces are about a fifth of
npxG and xA (§196, `diagnose_set_pieces.py`); no taker model yet. Validation:
`validate_new_player_priors.py`, `validate_new_player_priors_era.py`,
`check_new_player_priors_loop.py`; tests in `test_new_player_priors.py`. Experiments of 2026-11-12, none adopted
unless stated: `validate_prior_refit.py arm C` (league factors with a fixed scoring level, §231),
`diagnose_arrival_adaptation.py` (first-season dip, §231), `validate_evidence_k.py` (K by the prior's evidence; also an
exact replay of the loop from a pre-match log, §232), `validate_npxg_floor.py` (npxG floor, adopted 0.035, §233),
`validate_kalman_smoothing.py` (per-position Kalman smoothing, §234), `validate_defender_npxg.py` (defender floors,
centre-backs 0.018 adopted, §235).

Output: `data/current/player_goal_assist_ratings.csv`.

### fill_transfer_team_ratings.py

Fills the team attack rating for new rows in `transfers.csv` (new-to-league
arrivals) from the preseason or current team ratings.

### export_player_ratings.py

Assembles the weekly ratings card `data/current/player_ratings.csv`
(Player, Pos, Team, Price, NPG, Assists, Defcon) by joining
`player_goal_assist_ratings.csv` with the defcon model's
`defcon_average_probabilities.csv` through `players_current.csv`. `Price` is
taken from `players_current.csv` (current FPL price), not from the ratings
file, whose copy is frozen at `build_player_ratings.py`'s last run. Players who
have left the league (FPL status `u` and no expected minutes) are left out
(CHANGELOG §168). Goalkeepers keep a blank Defcon (the defcon model covers outfield players only). The
export fails loudly if either input artifact is missing.

## Pipeline position

`build_player_ratings.py` runs as a stage of `run_post_gameweek.py`
and runs manually; `export_player_ratings.py` runs automatically as the
"Player ratings export" stage, consuming whatever artifacts the manual runs
produced.
