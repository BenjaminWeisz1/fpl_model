# Research

The experiments behind the model, kept apart from the production package (`fpl_model/`).
Nothing in the weekly pipeline imports this folder.

Every change to a production model was tested here first. Each test was pre-registered with an
expected direction and size of effect, scored out of time against the incumbent model, and
reported whether it passed or failed. Most candidates failed and were not adopted. Their scripts
stay here as the record of what was tried.

| Folder | What it tests |
|---|---|
| `team_ratings/` | Alternative team-rating models (Kalman filter, Elo and odds priors, promoted-team learning, Dixon-Coles), money priors, red-card factors, manager tenures, and the out-of-time harness `evaluate_team_ratings_oot.py` that the others share |
| `player_ratings/` | The goal/assist rating loop: learning rates, priors for new arrivals, youth and returning players, position effects, opponent adjustments, calibration of FPL-era expected assists |
| `defcon/` | The defensive-contribution classifiers: monotonic constraints, feature sets, recency, opponent effects, calibration |
| `bonus/` | The bonus-points simulation: accounting checks, pass bands, FPL-only assists, team-goal coherence, keeper bonus |
| `assemble/` | Points assembly: keeper saves and shots, team coherence, error breakdowns by player and component |
| `absence/` | Effects of fixture congestion and key-player absence |
| `optimizer/` | Solver speed benchmarks |
| `general/` | Card-point models (exposure, suspensions) |

## Running a script

From the repository root, as a module:

```
python -m research.player_ratings.validate_npxg_floor
```

Scripts read the local `data/` folder, which is not published. Results go to
`data/validation/<experiment>_<date>/`.

## Re-checks scheduled for the end of the 2026/27 season

These scripts stay runnable because a decision depends on rerunning them on a full new season:
`team_ratings/promoted_learning_team_ratings.py`, `team_ratings/validate_promoted_learning_other_leagues.py`,
`team_ratings/validate_kalman_other_leagues.py`, `team_ratings/evaluate_team_ratings_oot.py`,
`player_ratings/validate_npxg_floor.py`, `player_ratings/validate_defender_npxg.py`,
`player_ratings/validate_position_forms.py`, `player_ratings/diagnose_source_club.py`,
`assemble/validate_keeper_shots_later.py`, `assemble/validate_keeper_saves_later.py`.
