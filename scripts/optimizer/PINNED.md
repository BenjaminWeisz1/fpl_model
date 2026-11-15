# Pinned open-fpl-solver

Source: https://github.com/solioanalytics/open-fpl-solver (Apache 2.0, free for personal use)
Commit: ec65f5e2b2be34441cb4b2a9efafea1ec0fe3b79 (2026-09-15)
highspy 1.15.1 and fuzzywuzzy 0.18.0, as in upstream's `uv.lock` at that commit.

`open_fpl_solver/` holds the files we use, unchanged, except `dev/solver.py`, which differs by `gw14_ft.patch`:

- `fts` and `fts_state` extend one week past the horizon with the same clamping rule, and the gain into that week is
  credited at that week's decay weight, so a transfer saved in the last GW has value (`optimizer_brief.md`).
- Each solution also returns the HiGHS status, the MIP gap and the end-of-horizon FT count. Upstream discards them, so a
  timeout would otherwise look like an optimum.
- An infeasible solve ends the plan loop and returns the plans found so far, instead of reading a non-solution.

Upstream's `score` carries a constant: its FT state values start at 1.5 for 0 FTs and there is no previous week at the
first GW, so the first week always adds 1.5 x (opening FTs + 1), which is +3.0 with 1 FT. `replay_plans.py` removes it
before comparing.

## Second local patch: `ft_tight_m.patch` (§227, 2026-11-11)

Applied on top of `gw14_ft.patch`. Adds the option `ft_tight_big_m` (default off: with it off the model is exactly
upstream's plus `gw14_ft.patch`). With it on, each of the ten FT-clamp big-M rows uses the smallest M the variable
bounds allow (fts in [0, 5], transfer_count in [0, 15], so the unclamped count is in [-14, 6]) instead of 20. The
integer solutions are the same; only the LP relaxation is tighter. Benchmark variant `ftm`; adoption into a mode
requires `replay_plans.py` agreement and the gap-0 optimum on all six benchmark instances.
