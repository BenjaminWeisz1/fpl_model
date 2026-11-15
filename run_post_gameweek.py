"""
Part 1 of the weekly pipeline -- run this as soon as a gameweek has finished.

Everything here depends only on what has already happened on the pitch, so it
can run the moment the last match of the round is over:

    fetches -> team ratings -> player goal/assist ratings -> defcon
            -> player ratings export -> part-1 heatmaps

Part 2 (``run_pre_deadline.py``) is run later in the week, once the expected
minutes and penalty-share inputs have settled.

This is the slow half: the defcon stage trains its models (~20 min cold, ~6 min
with a warm cache), so budget 25-35 minutes for a cold run.

Run locally (from anywhere -- paths are resolved relative to this file):
    python run_post_gameweek.py
    python run_post_gameweek.py --preseason   # include preseason ratings blend
    python run_post_gameweek.py --gameweek 10 # tag the snapshot for this gameweek
"""
import sys

from pipeline_stages import SCRIPTS, run_pipeline

# ── Pipeline stages (in execution order) ──────────────────────────────────────
# ORDER MATTERS:
#   * fetch_fixtures_results.py derives each new result's team xG from the
#     player stats, so fetch_gameweek_stats.py MUST run before it -- otherwise
#     it warns and skips every fixture whose stats are not in yet.
#   * update_team_ratings.py reads match_results.csv, so both fetches precede it.
#   * export_player_ratings.py consumes player_goal_assist_ratings.csv AND
#     defcon_average_probabilities.csv, so it runs after both producers.
STAGES = [
    {
        "label": "Fetch squads and prices",
        "script": SCRIPTS / "fetch" / "fetch_bootstrap.py",
        "summary_files": ["players_current.csv"],
    },
    {
        "label": "Fetch gameweek player stats",
        "script": SCRIPTS / "fetch" / "fetch_gameweek_stats.py",
        "summary_files": ["player_match_stats.csv"],
    },
    {
        "label": "Fetch fixtures and results",
        "script": SCRIPTS / "fetch" / "fetch_fixtures_results.py",
        "summary_files": ["match_results.csv"],
    },
    {
        # Scores only the immutable GW8–12 forecasts registered in §42. This
        # runs after both outcome fetches and never trains or changes ratings.
        "label": "DEFCON prospective validation (frozen forecasts)",
        "script": SCRIPTS / "defcon" / "forward_check.py",
        "args": ["score"],
        "summary_files": [],
    },
    {
        # Per-match positions, shots and red cards from FotMob match pages, 2026 on
        # (CHANGELOG §197; replaced the Understat weekly fetch by user decision after
        # Understat went down during GW9). Runs after the results fetch: it fetches a
        # page for every played match_results.csv row without one (~1 min at 7 s per
        # page) and checks each page is that Premier League match. It HALTS when a
        # played match has no page (FOTMOB_ALLOW_MISSING=1 continues), so an outage
        # can't silently drop a round. 2017-2025 positions stay in the frozen
        # Understat cache (understat_match_positions.csv, no longer refreshed).
        "label": "Fetch per-match positions and shots (FotMob)",
        "script": SCRIPTS / "fetch" / "fetch_fotmob_matches.py",
        "summary_files": ["fotmob_match_positions.csv", "fotmob_shots.csv"],
    },
    {
        # Crosswalk the Understat roster slots (to 2025) and FotMob's (2026 on, §197)
        # onto player_match_stats.csv's own row identity (AGENDA §4 shared
        # prerequisites). Runs after BOTH the gameweek-stats fetch (writes the
        # FPL-era rows it keys on) and the FotMob fetch (the slots it maps). Pure CSV transform, no network.
        # Produces player_match_positions.csv (per-row detailed slot, Sub resolved
        # to the modal starting slot) + understat_fpl_crosswalk.csv, consumed by
        # the defcon model. Also prints the §4.5 FBref<->Understat agreement rates.
        "label": "Build position crosswalk (Understat/FotMob -> match stats)",
        "script": SCRIPTS / "fetch" / "build_position_crosswalk.py",
        "summary_files": ["player_match_positions.csv", "understat_fpl_crosswalk.csv"],
    },
    {
        # Penalties from Understat's shot data (CHANGELOG §75): appends any penalty
        # missing from penalty_log.csv (never edits or removes a manual row), then
        # reconciles PK/PKatt/NPG/NPxG_Adj in both data files (§73). Needs the cached
        # Understat matches and the crosswalk above, and must precede the team and
        # player ratings, which read NPG. No network.
        "label": "Penalty log from Understat/FotMob + reconcile",
        "script": SCRIPTS / "fetch" / "penalty_log_understat.py",
        "summary_files": ["penalty_log.csv"],
    },
    {
        # Red-card adjustment of team npxG from cached Understat shots, FotMob from 2026 (CHANGELOG §133, §197):
        # rewrites NPxG_Adj of every red-card match from its base, replacing the manual step.
        # After penalty reconcile (which moves the base), before the team ratings. No network.
        "label": "Red-card npxG adjustment",
        "script": SCRIPTS / "team_ratings" / "red_card_adjust.py",
        "args": ["--apply"],
        "summary_files": ["red_card_factors.csv"],
    },
    {
        "label": "Team ratings update",
        "script": SCRIPTS / "team_ratings" / "update_team_ratings.py",
        "summary_files": ["team_ratings_history.csv", "team_ratings_current.csv"],
    },
    {
        "label": "Preseason ratings blend (season-start only)",
        "script": SCRIPTS / "team_ratings" / "compute_preseason_ratings.py",
        "season_start_only": True,
        "summary_files": ["team_ratings_preseason_2026.csv"],
    },
    {
        # Priors for players new to the PL (CHANGELOG §103, §104): frozen parameters applied to
        # transfers.csv + the frozen Understat league files. Must run before the ratings stage.
        "label": "New-player goal/assist priors",
        "script": SCRIPTS / "player_ratings" / "build_new_player_priors.py",
        "summary_files": ["new_player_priors.csv"],
    },
    {
        # Player-level red-card factors (CHANGELOG §136; FotMob from 2026, §197): his own post-red shots / assisted
        # shots, read by the ratings stage (GOAL_MODEL_RED_CARD=0 reverts). No network.
        "label": "Red-card player factors",
        "script": SCRIPTS / "player_ratings" / "red_card_player_factors.py",
        "summary_files": ["red_card_player_factors.csv"],
    },
    {
        "label": "Player goal/assist ratings",
        "script": SCRIPTS / "player_ratings" / "build_player_ratings.py",
        "summary_files": ["player_goal_assist_ratings.csv"],
    },
    {
        # Slow stage (~20 min cold, ~6 min with a warm model cache).
        # NOTE: predict.py's minutes-scaled "Probability" column is scaled by
        # whatever minutes_probabilities.csv happens to hold when it runs --
        # i.e. last week's file in this split. Nothing downstream reads it any
        # more: build_simulation_features.py and build_predicted_points.py both
        # work from the unscaled "Probability_at_90min" and re-scale with the
        # fresh minutes in part 2.
        "label": "Defcon predictions",
        "script": SCRIPTS / "defcon" / "predict.py",
        "summary_files": ["predictions_defensive.csv",
                          "defcon_average_probabilities.csv"],
    },
    {
        # Weekly ratings card (player_ratings.csv): NPG/Assists from
        # player_goal_assist_ratings.csv + fixture-neutral 90-minute defcon
        # from defcon_average_probabilities.csv. Fails loudly if either of
        # the two preceding stages did not produce its artifact.
        "label": "Player ratings export",
        "script": SCRIPTS / "player_ratings" / "export_player_ratings.py",
        "summary_files": ["player_ratings.csv"],
    },
    {
        "label": "Heatmap exports (team ratings, player ratings)",
        "script": SCRIPTS / "export_weekly_heatmaps.py",
        "args": ["--part", "1"],
        "summary_files": [],
    },
    {
        # Fixture tickers for the next 8 gameweeks -> images/fdr/fixture_ticker_GW<nn>[_attack|_defense].png (§148). Reads only
        # part-1 inputs (team_ratings_current.csv, match_results.csv, players_current.csv) plus live FPL fixtures,
        # so it runs here; rerun it by hand if a match moves later in the week.
        "label": "Fixture ticker (images/fdr)",
        "script": SCRIPTS / "optimizer" / "fixture_ticker.py",
        "summary_files": [],
    },
    {
        # Predicted final table: current points + expected points (3 x P(win) + P(draw), Poisson goals from current
        # ratings, league penalty rate) for every remaining FPL fixture -> images/team_ratings/predicted_table.png
        # and data/current/predicted_table.csv (user request, 2026-11-04).
        "label": "Predicted final table",
        "script": SCRIPTS / "team_ratings" / "predicted_table.py",
        "summary_files": ["predicted_table.csv"],
    },
    {
        # Next gameweek's predicted scores from current ratings and the league penalty rate ->
        # images/next_gameweek/next_gw_predicted_scores.png (moved from part 2's heatmaps, 2026-11-04).
        "label": "Next gameweek predicted scores",
        "script": SCRIPTS / "team_ratings" / "predicted_scores.py",
        "summary_files": [],
    },
    {
        "label": "Save gameweek model outputs (ratings snapshot)",
        "script": SCRIPTS.parent / "save_gameweek_outputs.py",
        "args": ["--part", "1"],
        "accepts_gameweek": True,
        "summary_files": [],
    },
]


if __name__ == "__main__":
    sys.exit(run_pipeline("FPL Pipeline -- Part 1: post-gameweek update", STAGES))
