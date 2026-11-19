"""
Part 2 of the weekly pipeline -- run this later in the week, once expected
minutes and the penalty share are reasonably settled.

Everything here turns the ratings produced by part 1 (``run_post_gameweek.py``)
into per-player, per-gameweek predicted points:

    Sofascore import/build/status -> minutes probabilities -> fixture goal predictions -> penalty predictions
      -> residual BPS baseline -> pass-band rates -> simulation features -> bonus simulation
      -> predicted points -> part-2 heatmaps

Manual inputs it expects to be up to date: data/current/manual/minutes.csv (the FPL
Review expected-minutes forecast), data/current/manual/review_pk_share.xlsx (cleaned
by clean_penalty_share_paste.py) and data/current/manual/fixtures.csv.

Run locally (from anywhere -- paths are resolved relative to this file):
    python run_pre_deadline.py
    python run_pre_deadline.py --gameweek 10   # tag the snapshot for this gameweek
"""
import sys

from pipeline_stages import SCRIPTS, run_pipeline

# ── Pipeline stages (in execution order) ──────────────────────────────────────
# ORDER MATTERS: build_penalty_predictions.py reads fixture_goal_predictions.csv
# (the output of predict_fixture_goals.py), so "Fixture goal predictions" MUST
# run first -- otherwise penalty predictions silently consumes the previous
# run's fixture file, which covers the WRONG rounds as soon as the season
# advances (and, this week, was missing all promoted-team fixtures).
STAGES = [
    # Never fetches and always exits 0: imports Sofascore pages collected in Chrome
    # from Downloads, rebuilds only when needed, and writes sofascore_status.json,
    # which later bonus stages read to decide whether Sofascore may be used (§66).
    {
        "label": "Sofascore import, build and status (non-halting)",
        "script": SCRIPTS / "fetch" / "sofascore_weekly.py",
        "args": ["pipeline"],
        "summary_files": [],
    },
    {
        "label": "Minutes probabilities",
        "script": SCRIPTS / "minutes_penalties_cs" / "build_minutes_probabilities.py",
        "summary_files": ["minutes_probabilities.csv"],
    },
    {
        "label": "Fixture goal predictions",
        "script": SCRIPTS / "team_ratings" / "predict_fixture_goals.py",
        "summary_files": ["fixture_goal_predictions.csv"],
    },
    {
        "label": "Penalty predictions",
        "script": SCRIPTS / "minutes_penalties_cs" / "build_penalty_predictions.py",
        "summary_files": ["penalty_predictions.csv"],
    },
    {
        "label": "Residual BPS baseline",
        "script": SCRIPTS / "bonus" / "build_residual_bps_baseline.py",
        "summary_files": ["residual_bps_baseline.csv", "residual_bps_position_defaults.csv"],
    },
    # Pass-band probabilities per upcoming fixture (§68); header-only when Sofascore is
    # not usable, in which case the bonus simulation runs the §64 model.
    {
        "label": "Pass-band rates",
        "script": SCRIPTS / "bonus" / "build_pass_band_rates.py",
        "summary_files": ["pass_band_rates.csv"],
    },
    # FPL-only assist rates (§92): rebounds, deflections, penalties won, forced own goals.
    # Position fallback when Sofascore is not usable.
    {
        "label": "FPL-only assist rates",
        "script": SCRIPTS / "bonus" / "build_fpl_only_assist_rates.py",
        "summary_files": ["fpl_only_assist_rates.csv"],
    },
    {
        "label": "Simulation features",
        "script": SCRIPTS / "assemble" / "build_simulation_features.py",
        "summary_files": ["simulation_features.csv"],
    },
    {
        "label": "Bonus simulation",
        "script": SCRIPTS / "bonus" / "simulate_bonus.py",
        "summary_files": ["bonus_predictions.csv"],
    },
    {
        "label": "Assemble predicted points",
        "script": SCRIPTS / "assemble" / "build_predicted_points.py",
        "summary_files": ["predicted_points.csv"],
    },
    {
        "label": "Heatmap exports (predicted points, next gameweek)",
        "script": SCRIPTS / "export_weekly_heatmaps.py",
        "args": ["--part", "2"],
        "summary_files": [],
    },
    # Benchmark only, never read by a model: cleans the Solio Baselines capture in
    # data/raw/solio/<season>/GW<nn>/ (scripts/solio/SOLIO.md). Replaced the FPL Review
    # pastes on 2026-11-13 by user decision (§236); clean_fplreview_paste.py still runs
    # standalone. Always exits 0; the snapshot copies the result and the capture.
    {
        "label": "Solio baselines (benchmark, non-halting)",
        "script": SCRIPTS / "solio" / "clean_solio_baselines.py",
        "summary_files": [],
    },
    {
        "label": "Save gameweek model outputs (full pre-deadline snapshot)",
        "script": SCRIPTS.parent / "save_gameweek_outputs.py",
        "accepts_gameweek": True,
        "summary_files": [],
    },
]


if __name__ == "__main__":
    sys.exit(run_pipeline("FPL Pipeline -- Part 2: pre-deadline update", STAGES))
