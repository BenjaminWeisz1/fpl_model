"""Tests that need the local data/ folder skip when it is absent (CI on GitHub, where no data is published).

Two kinds: test modules whose code under test reads data/current when imported (listed below, skipped whole), and
single tests that read live data, marked @pytest.mark.needs_data. With the data present, everything runs.
"""
import pytest

from fpl_model import paths

HAVE_DATA = (paths.current_file("player_match_stats.csv", root=paths.CURRENT)).exists()

# The modules these tests import read data/current at import time (league_rates.current_season(), keeper_saves, ...).
NEEDS_DATA_AT_IMPORT = {
    "test_goalkeeper_points.py", "test_xmins_clock.py", "test_coherence.py", "test_integer_bps.py",
    "test_pass_bands.py", "test_fetch_fotmob_matches.py", "test_fixture_rounds.py", "test_penalty_log_understat.py",
    "test_sofascore.py", "test_sofascore_backfill.py", "test_sofascore_browser.py", "test_sofascore_weekly.py",
    "test_new_player_priors.py", "test_export_site_data.py", "test_save_gameweek_outputs.py",
}
_skipped_modules = []


def pytest_ignore_collect(collection_path, config):
    if not HAVE_DATA and collection_path.name in NEEDS_DATA_AT_IMPORT:
        _skipped_modules.append(collection_path.name)
        return True
    return None


def pytest_collection_modifyitems(config, items):
    if HAVE_DATA:
        return
    skip = pytest.mark.skip(reason="needs the local data/ folder")
    for item in items:
        if item.get_closest_marker("needs_data"):
            item.add_marker(skip)


def pytest_report_header(config):
    return None if HAVE_DATA else "data/current not found: tests that need the local data are skipped"


def pytest_terminal_summary(terminalreporter):
    if _skipped_modules:
        terminalreporter.write_line(f"{len(_skipped_modules)} test modules not collected (they need the local data): "
                                    + ", ".join(sorted(_skipped_modules)))
