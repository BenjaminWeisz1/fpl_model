import pytest

from fpl_model import paths


@pytest.mark.parametrize(
    ("filename", "directory"),
    [
        ("predicted_points.csv", paths.FORECASTS),
        ("league_coefficients.csv", paths.LOOKUPS),
        ("minutes.csv", paths.MANUAL),
        ("player_match_stats.csv", paths.SOURCED),
        ("team_ratings_current.csv", paths.STATE),
    ],
)
def test_current_file_uses_its_category_folder(filename, directory):
    assert paths.current_file(filename) == directory / filename


def test_current_file_routes_archived_files_to_the_archive():
    assert paths.current_file("manager_tenures.csv") == paths.CURRENT_UNUSED / "manager_tenures.csv"


def test_current_file_keeps_alternate_roots_flat(tmp_path):
    assert paths.current_file("player_match_stats.csv", root=tmp_path) == tmp_path / "player_match_stats.csv"


def test_current_file_rejects_uncategorized_names():
    with pytest.raises(ValueError, match="Uncategorized current-data file"):
        paths.current_file("unknown.csv")
