"""
Every folder the pipeline reads or writes, declared once.

Modules import these constants instead of rebuilding paths from ``__file__``, so moving a module
never changes where it finds its data. ``data/`` is local only and never committed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fpl_model"

DATA = ROOT / "data"
CURRENT = DATA / "current"          # live pipeline inputs and outputs, refreshed weekly
FORECASTS = CURRENT / "forecasts"
LOOKUPS = CURRENT / "lookups"
MANUAL = CURRENT / "manual"
SOURCED = CURRENT / "sourced"
STATE = CURRENT / "state"
RAW = DATA / "raw"                  # frozen downloads, never rewritten
SNAPSHOTS = DATA / "snapshots"      # immutable per-gameweek output archives
VALIDATION = DATA / "validation"    # experiment evidence and frozen protocols
ARCHIVE = DATA / "archive"          # superseded data kept for traceability
CURRENT_UNUSED = ARCHIVE / "current_unused_20261118"
OPTIMIZER = DATA / "optimizer"      # transfer optimizer runs and plans
SOLIO = DATA / "solio"
FPLREVIEW = DATA / "fplreview"

CACHE = ROOT / "cache"              # regenerable model cache
IMAGES = ROOT / "images"            # weekly charts, published in the README
SITE = ROOT / "site"                # the local website; site/data is written by part 3

_CURRENT_FILE_GROUPS = {
    "forecasts": {
        "bonus_predictions.csv",
        "defcon_average_probabilities.csv",
        "fixture_goal_predictions.csv",
        "fpl_only_assist_rates.csv",
        "minutes_probabilities.csv",
        "pass_band_rates.csv",
        "penalty_predictions.csv",
        "predicted_components.csv",
        "predicted_points.csv",
        "predicted_table.csv",
        "predictions_defensive.csv",
        "residual_bps_baseline.csv",
        "residual_bps_mode.json",
        "residual_bps_position_defaults.csv",
        "simulation_features.csv",
    },
    "lookups": {
        "fotmob_id_overrides.csv",
        "league_coefficients.csv",
        "mins_lookup_table.csv",
        "money_priors_by_season.csv",
        "money_ratings_2026.csv",
        "position_combinations.csv",
        "vaastav_name_overrides.csv",
        "web_name_overrides.csv",
    },
    "manual": {
        "fixtures.csv",
        "minutes.csv",
        "penalty_share.csv",
        "review_pk_share.xlsx",
    },
    "sourced": {
        "fotmob_match_positions.csv",
        "fotmob_shots.csv",
        "fplreview_projections.csv",
        "fplreview_status.json",
        "fplreview_team_strengths.csv",
        "fpl_player_codes.csv",
        "gameweek_deadlines.csv",
        "historical_money.csv",
        "match_results.csv",
        "penalty_log.csv",
        "penalty_log_fotmob_additions.csv",
        "penalty_log_understat_additions.csv",
        "players_current.csv",
        "player_match_positions.csv",
        "player_match_stats.csv",
        "set_piece_orders.csv",
        "sofascore_missing_players.csv",
        "sofascore_keeper_seasons.csv",
        "sofascore_player_extra.csv.gz",
        "sofascore_player_finishing_seasons.csv",
        "sofascore_player_match_stats.csv",
        "sofascore_player_on_pitch.csv",
        "sofascore_set_piece_events.csv",
        "sofascore_status.json",
        "solio_baselines.csv",
        "solio_status.json",
        "transfers.csv",
        "understat_fpl_crosswalk.csv",
        "understat_match_positions.csv",
        "vaastav_player_match_history.csv",
    },
    "state": {
        "new_player_priors.csv",
        "player_goal_assist_ratings.csv",
        "player_ratings.csv",
        "red_card_factors.csv",
        "red_card_player_factors.csv",
        "team_ratings_current.csv",
        "team_ratings_history.csv",
        "team_ratings_preseason_2026.csv",
    },
}

_ARCHIVED_CURRENT_FILES = {
    "fixture_identity_registry.csv",
    "historical_money_pre2013.csv",
    "identity_overrides.csv",
    "league_coefficients_20261029.csv",
    "manager_tenures.csv",
    "new_players_2026.csv",
    "player_identity_aliases.csv",
    "player_identity_registry.csv",
    "sofascore_field_coverage.csv",
    "team_ratings_current_midpoint_normalized.csv",
}


def current_file(filename, *, root=CURRENT):
    """Resolve a current-data filename to its category folder.

    Alternate roots are left flat so callers can keep using temporary fixture
    directories; only the project's data/current root uses the categorized layout.
    """
    root = Path(root)
    if root.resolve() != CURRENT.resolve():
        return root / filename

    name = Path(filename).name
    if name in _ARCHIVED_CURRENT_FILES:
        return CURRENT_UNUSED / name
    for folder, filenames in _CURRENT_FILE_GROUPS.items():
        if name in filenames or (
            folder == "state" and name.startswith("team_ratings_preseason_")
        ):
            return CURRENT / folder / name
    raise ValueError(f"Uncategorized current-data file: {filename}")
