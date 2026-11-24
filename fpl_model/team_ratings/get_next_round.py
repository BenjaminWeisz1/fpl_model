"""
Determines the next round to generate predictions for, based on what's
actually been played in match_results.csv -- NOT fixtures.csv's own minimum,
since fixtures.csv is a static full-season file that never changes.

Importable by any script in the pipeline:
    from get_next_round import get_next_round
    next_round = get_next_round(data_dir, season=2026)
"""
from pathlib import Path

import pandas as pd

from fpl_model import paths


def get_next_round(data_dir: Path, season: int) -> int:
    """Returns the next round number to predict: the first round with an
    unplayed fixture (§186). A rearranged match leaves an earlier round open,
    so one past the highest played round would skip rounds. A season with no
    played match starts at its first round; a finished season returns one past
    its last round.

    Joins match_results.csv (which has no Round column) against fixtures.csv
    on (Season, Home, Away) to discover which round each played match
    corresponds to. This assumes fixtures.csv Round is current;
    fetch_fixtures_results.py checks it against FPL's event each part 1.
    """
    match_results = pd.read_csv(paths.current_file("match_results.csv", root=data_dir))
    fixtures = pd.read_csv(paths.current_file("fixtures.csv", root=data_dir))

    played = match_results[match_results["Season"] == season]
    fixtures_this_season = fixtures[fixtures["Season"] == season].copy()
    if fixtures_this_season.empty:
        raise ValueError(f"fixtures.csv has no rows for season {season}")
    for side in ("Home", "Away"):
        fixtures_this_season[side] = fixtures_this_season[side].str.strip()

    if played.empty:
        return int(fixtures_this_season["Round"].min())

    merged = played.merge(
        fixtures_this_season[["Season", "Home", "Away", "Round"]],
        on=["Season", "Home", "Away"], how="left", validate="many_to_one"
    )

    unmatched = merged[merged["Round"].isna()]
    if len(unmatched) > 0:
        print(f"WARNING: {len(unmatched)} played match(es) in match_results.csv have no "
              f"corresponding row in fixtures.csv -- check for a team-name mismatch between "
              f"the two files:")
        print(unmatched[["Home", "Away"]].drop_duplicates().to_string(index=False))

    if merged["Round"].notna().sum() == 0:
        raise ValueError(
            f"No played matches for season {season} could be matched to fixtures.csv at all -- "
            f"cannot determine next round. Check team-name consistency between "
            f"match_results.csv and fixtures.csv before proceeding."
        )

    played_keys = set(zip(played["Home"], played["Away"]))
    # A blank Round is a postponed match with no date yet: it belongs to no round, so it can't hold the window.
    open_rounds = [r for h, a, r in zip(fixtures_this_season["Home"], fixtures_this_season["Away"],
                                        fixtures_this_season["Round"]) if (h, a) not in played_keys and pd.notna(r)]
    if not open_rounds:
        return int(fixtures_this_season["Round"].max()) + 1
    return int(min(open_rounds))
