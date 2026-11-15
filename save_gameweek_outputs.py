"""Save immutable weekly model outputs, automatically or before a deadline.

    python save_gameweek_outputs.py                      # full, next gameweek
    python save_gameweek_outputs.py --gameweek 8          # explicit gameweek
    python save_gameweek_outputs.py --part 1              # ratings only
    python save_gameweek_outputs.py --verify <directory>  # verify saved hashes

Copies original output CSVs plus useful identity/fixture context, and writes
gameweek-specific views and a SHA256 manifest under data/snapshots/<season>/GWxx/.
Multiple runs create separate UTC-stamped revisions; no saved run is replaced.

Folders are named <UTC stamp>_GWxx_<timing>_part<part>. The gameweek is the one the
forecast is for: --gameweek when given (run_pre_deadline.py and run_post_gameweek.py
pass theirs through), otherwise the round after the last with local results. Timing
is pre_deadline or after_deadline against gameweek_deadlines.csv, written by
fetch_bootstrap.py, or deadline_unknown without it. Only a pre_deadline snapshot is a
genuine forecast for prospective scoring. Saved before 2026-11-01, names lack the tag.

Full snapshots require the pre-deadline outputs to cover the current forecast
origin. Part 1 intentionally omits stale part-2 outputs. No fetches or model fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
SNAPSHOTS = paths.SNAPSHOTS
from fpl_model.league_rates import current_season  # noqa: E402

CURRENT_SEASON = current_season()

PART1 = (
    "team_ratings_current.csv", "player_goal_assist_ratings.csv",
    "predictions_defensive.csv", "defcon_average_probabilities.csv", "player_ratings.csv",
)
PART2 = (
    "minutes_probabilities.csv", "fixture_goal_predictions.csv", "penalty_predictions.csv",
    "residual_bps_baseline.csv", "residual_bps_position_defaults.csv",
    "sofascore_status.json", "residual_bps_mode.json", "pass_band_rates.csv", "fpl_only_assist_rates.csv",
    "simulation_features.csv", "bonus_predictions.csv", "predicted_components.csv", "predicted_points.csv",
)
# Benchmarks: copied when present and built for the snapshot's gameweek, never required, never read by a model.
# Folder in the snapshot -> (status file, cleaned files, raw file patterns). Solio replaced FPL Review on
# 2026-11-13 (§236); FPL Review is still copied when its status matches, for re-running old weeks.
BENCHMARKS = {
    "solio": ("solio_status.json", ("solio_baselines.csv",), ("*.csv", "*.json")),
    "fplreview": ("fplreview_status.json", ("fplreview_projections.csv", "fplreview_team_strengths.csv"), ("*.xlsx",)),
}
CONTEXT = ("players_current.csv", "fixtures.csv", "match_results.csv")
PROVENANCE = (
    "player_match_stats.csv", "player_match_positions.csv", "transfers.csv",
    "minutes.csv", "review_pk_share.xlsx", "penalty_share.csv", "league_coefficients.csv",
    "new_player_priors.csv", "gameweek_deadlines.csv",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def clean(frame):
    frame = frame.copy()
    for c in ("Home", "Away", "Team", "Opponent"):
        if c in frame:
            frame[c] = frame[c].str.strip()
    return frame


def snapshot_plan(data, season, gameweek=None, part="all"):
    """Validate every required artifact before creating any archive directory."""
    names = [*PART1, *(PART2 if part == "all" else ()), *CONTEXT]
    missing = [name for name in names if not (data/name).is_file()]
    if missing:
        raise ValueError(f"Missing required files: {', '.join(missing)}")
    fixtures = clean(read_csv(data/"fixtures.csv"))
    fixtures = fixtures[fixtures.Season.eq(season)].copy()
    if fixtures.empty or fixtures.MatchID.duplicated().any():
        raise ValueError("Missing season schedule or duplicate fixture MatchIDs")
    results = clean(read_csv(data/"match_results.csv"))
    results = results[results.Season.eq(season)]
    completed = results[results.Home_Goals.notna() & results.Away_Goals.notna()]
    # Use only played matches to locate origin: a duplicate elsewhere in the
    # future schedule must not multiply already-played match identities.
    played = completed.merge(fixtures[["Season", "Home", "Away", "Round"]],
                             on=["Season", "Home", "Away"], how="left")
    if played.Round.isna().any():
        raise ValueError("A completed match has no schedule mapping")
    origin = int(played.Round.max())+1 if len(played) else 1
    gameweek = origin if gameweek is None else gameweek
    if gameweek < origin:
        raise ValueError(f"GW{gameweek} has started or passed according to local results; next origin is GW{origin}")
    target = fixtures[fixtures.Round.eq(gameweek)]
    if target.empty or target.duplicated(["Season", "Home", "Away"]).any():
        raise ValueError(f"No unambiguous fixtures for season {season}, GW{gameweek}")
    if (data/"player_match_stats.csv").exists():
        stats = clean(read_csv(data/"player_match_stats.csv"))
        stats = stats[stats.Season.eq(season)]
        legs = pd.concat([target.rename(columns={"Home":"Team", "Away":"Opponent"}).assign(Venue="Home"),
                          target.rename(columns={"Away":"Team", "Home":"Opponent"}).assign(Venue="Away")])
        if len(stats.merge(legs[["Season", "Team", "Opponent", "Venue"]],
                           on=["Season", "Team", "Opponent", "Venue"])):
            raise ValueError(f"GW{gameweek} player outcomes already exist; cannot save a pre-gameweek forecast")
    views = {"fixtures.csv": target}
    defensive = read_csv(data/"predictions_defensive.csv")
    d = defensive.merge(fixtures[["MatchID", "Season", "Round"]], on="MatchID", how="left", validate="many_to_one")
    if d.Round.isna().any() or int(d.Round.min()) != origin:
        raise ValueError(f"DEFCON outputs are stale or unmapped; run the post-gameweek update for GW{origin}")
    if not d.Round.eq(gameweek).any():
        raise ValueError(f"DEFCON outputs do not cover GW{gameweek}")
    views["predictions_defensive.csv"] = d[d.Round.eq(gameweek)].copy()
    if part == "all":
        dependencies = [*PART2[:-1], "predictions_defensive.csv", "player_goal_assist_ratings.csv", "team_ratings_current.csv"]
        dependencies += [n for n in ("minutes.csv", "penalty_share.csv") if (data/n).exists()]
        newer = [n for n in dependencies if (data/n).stat().st_mtime_ns > (data/"predicted_points.csv").stat().st_mtime_ns]
        if newer:
            raise ValueError(f"Predicted points predate updated inputs/outputs ({', '.join(newer)}); run python run_pre_deadline.py first")
        for name in ("fixture_goal_predictions.csv", "simulation_features.csv", "bonus_predictions.csv"):
            f = read_csv(data/name)
            if "Round" not in f or f.empty or int(f.Round.min()) != origin or not f.Round.eq(gameweek).any():
                raise ValueError(f"{name} is stale or lacks GW{gameweek}; run python run_pre_deadline.py first")
            views[name] = f[f.Round.eq(gameweek)].copy()
        points = read_csv(data/"predicted_points.csv")
        rounds = sorted(int(c) for c in points.columns if str(c).isdigit())
        if not rounds or rounds[0] != origin or gameweek not in rounds:
            raise ValueError(f"predicted_points.csv is stale or lacks GW{gameweek}; run python run_pre_deadline.py first")
        pts = points[[c for c in ("Player", "Pos", "Team", "Price", str(gameweek)) if c in points]].copy()
        views["predicted_points.csv"] = pts.rename(columns={str(gameweek):"PredictedPoints"}).assign(Season=season, Round=gameweek)
        relative = gameweek-origin+1
        for name in ("minutes_probabilities.csv", "penalty_predictions.csv"):
            f = read_csv(data/name)
            suffix = f"_GW{relative}"
            selected = [c for c in f if c.endswith(suffix)]
            if not selected:
                raise ValueError(f"{name} lacks relative GW{relative}")
            identity = [c for c in ("Player", "Web_Name", "Pos", "Team", "Price") if c in f]
            views[name] = f[identity+selected].rename(columns={c:c[:-len(suffix)] for c in selected}).assign(Season=season, Round=gameweek)
    return {"files": names, "views": views, "origin": origin, "gameweek": gameweek}


def deadline_timing(data, gameweek, now):
    """(deadline, tag): whether `now` is before the gameweek's FPL deadline."""
    path = data/"gameweek_deadlines.csv"
    if not path.is_file():
        return None, "deadline_unknown"
    deadlines = read_csv(path)
    row = deadlines[deadlines.Gameweek.eq(gameweek)]
    if row.empty:
        return None, "deadline_unknown"
    deadline = pd.Timestamp(row.Deadline_UTC.iloc[0]).to_pydatetime()
    return deadline.isoformat(), "pre_deadline" if now < deadline else "after_deadline"


def verify_snapshot(path):
    manifest = json.loads((path/"manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Snapshot is incomplete")
    base = path.resolve()
    for relative, details in manifest["saved_files"].items():
        target = (path/relative).resolve()
        if not target.is_relative_to(base) or digest(target) != details["sha256"]:
            raise ValueError(f"Saved file changed: {relative}")
    return manifest


def benchmark_files(data, gameweek):
    """Snapshot key -> source path for each benchmark whose status was built for this gameweek.

    A benchmark left over from another week (e.g. FPL Review after the switch to Solio), or a failed build
    that never learned its gameweek, is skipped so it can't be filed under the wrong week. A failed Solio
    build records its gameweek, so its status is still copied and the snapshot shows why the table is missing.
    """
    files = {}
    for group, (status_name, cleaned, patterns) in BENCHMARKS.items():
        status_path = data/status_name
        if not status_path.is_file():
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("gameweek") != gameweek:
            continue
        files[f"{group}/{status_name}"] = status_path
        files.update({f"{group}/{name}": data/name for name in cleaned if (data/name).is_file()})
        folder = status.get("source_folder")
        raw = data.parent.parent/folder if folder else None
        if raw is not None and raw.is_dir():
            files.update({f"{group}/raw/{p.name}": p for pattern in patterns for p in sorted(raw.glob(pattern))})
    return files


def save_snapshot(data=DATA, output=SNAPSHOTS, season=CURRENT_SEASON, gameweek=None, part="all", include_images=False, now=None):
    names = [*PART1, *(PART2 if part == "all" else ()), *CONTEXT, *PROVENANCE]
    observed = {name: digest(data/name) for name in names if (data/name).is_file()}
    plan = snapshot_plan(data, season, gameweek, part)
    benchmark = benchmark_files(data, plan["gameweek"]) if part == "all" else {}
    benchmark_hashes = {key: digest(path) for key, path in benchmark.items()}
    for name, expected in observed.items():
        if digest(data/name) != expected:
            raise ValueError(f"Sources changed while checking snapshot: {name}; retry after the pipeline finishes")
    now = now or datetime.now(UTC)
    deadline, timing = deadline_timing(data, plan["gameweek"], now)
    label = f"GW{plan['gameweek']:02d}"
    destination = output/str(season)/label/f"{now.strftime('%Y%m%dT%H%M%S_%fZ')}_{label}_{timing}_part{part}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in plan["files"]:
        folder = "context" if name in CONTEXT else "outputs"
        target = destination/folder/name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(data/name, target)
        if digest(target) != observed[name]:
            raise ValueError(f"Source changed during snapshot: {name}; incomplete snapshot at {destination}")
    for key, source in benchmark.items():
        target = destination/key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != benchmark_hashes[key]:
            raise ValueError(f"Benchmark file changed during snapshot: {key}")
    for name, frame in plan["views"].items():
        target = destination/"gameweek"/name
        target.parent.mkdir(exist_ok=True)
        frame.to_csv(target, index=False, encoding="utf-8-sig")
    if include_images:
        families = ["team_ratings", "player_ratings"] + (["predicted_points", "next_gameweek"] if part == "all" else [])
        for family in families:
            for source in (ROOT/"images"/family).glob("*.png"):
                image_hash = digest(source)
                target = destination/"images"/family/source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                if digest(target) != image_hash or digest(source) != image_hash:
                    raise ValueError(f"Image changed during snapshot: {source.name}")
    for name, expected in observed.items():
        if digest(data/name) != expected:
            raise ValueError(f"Inputs/outputs changed during snapshot: {name}; retry after the pipeline finishes")
    saved = {str(p.relative_to(destination)).replace("\\", "/"):
             {"sha256": digest(p), "bytes": p.stat().st_size,
              "modified_utc": datetime.fromtimestamp(p.stat().st_mtime, UTC).isoformat()}
             for p in sorted(destination.rglob("*")) if p.is_file()}
    sources = [ROOT/"run_post_gameweek.py", ROOT/"run_pre_deadline.py", Path(__file__),
               ROOT/"fpl_model"/"defcon"/"predict.py", ROOT/"fpl_model"/"defcon"/"defender_model.py",
               ROOT/"fpl_model"/"defcon"/"recent_defcon.py",ROOT/"fpl_model"/"defcon"/"opponent_model.py",
               ROOT/"fpl_model"/"defcon"/"features.py",ROOT/"fpl_model"/"defcon"/"model_cache.py",
               ROOT/"fpl_model"/"defcon"/"evaluation.py"]
    sources += [ROOT/"fpl_model"/relative for relative in (
        "team_ratings/update_team_ratings.py", "team_ratings/predict_fixture_goals.py",
        "player_ratings/build_player_ratings.py", "player_ratings/export_player_ratings.py",
        "player_ratings/new_player_priors.py", "player_ratings/build_new_player_priors.py",
        "player_ratings/new_player_prior_params.json", "player_ratings/fotmob_priors.py",
        "player_ratings/new_player_prior_params_fotmob.json", "fetch/fetch_fotmob_leagues.py",
        "player_ratings/new_player_prior_params_refit.json", "team_ratings/red_card_adjust.py",
        "player_ratings/red_card_player_factors.py",
        "minutes_penalties_cs/build_minutes_probabilities.py", "minutes_penalties_cs/build_penalty_predictions.py",
        "bonus/build_residual_bps_baseline.py", "bonus/simulate_bonus.py",
        "bonus/pass_bands.py", "bonus/build_pass_band_rates.py", "fetch/sofascore_weekly.py",
        "bonus/fpl_only_assists.py", "bonus/build_fpl_only_assist_rates.py",
        "assemble/build_simulation_features.py", "assemble/build_predicted_points.py", "defcon_scaling.py",
        "fplreview/clean_fplreview_paste.py", "solio/clean_solio_baselines.py",
        "keeper_saves.py", "card_points.py")]
    manifest = {"status":"complete", "season":season, "gameweek":plan["gameweek"],
        "forecast_origin_gameweek":plan["origin"], "part":part, "saved_utc":now.isoformat(),
        "deadline_utc":deadline, "timing":timing,
        "saved_files":saved, "source_data_sha256":observed,
        "source_code_sha256":{str(p.relative_to(ROOT)).replace("\\", "/"):digest(p) for p in sources if p.exists()},
        "versions":{name:version(name) for name in ("pandas", "numpy", "scikit-learn")},
        "notes":["Original CSVs are unchanged copies; gameweek/ contains labelled filtered views",
                 "Relative GW1 in minutes/penalty files is anchored to forecast_origin_gameweek",
                 "Ratings-only snapshots do not contain pre-deadline predicted points",
                 "Part-1 DEFCON Probability can use stale minutes; use Probability_at_90min for comparisons",
                 "Local results and stats were checked for target outcomes; timing compares the save time with the FPL deadline",
                 "Optional images are copied as present; CSV round checks are authoritative"]}
    with (destination/"manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    verify_snapshot(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gameweek", type=int)
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--part", choices=["1", "all"], default="all")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--verify", type=Path, metavar="SNAPSHOT_DIRECTORY")
    args = parser.parse_args()
    try:
        if args.verify is not None:
            manifest = verify_snapshot(args.verify)
            print(f"Verified season {manifest['season']}, GW{manifest['gameweek']} ({len(manifest['saved_files'])} files).")
        else:
            result = save_snapshot(season=args.season, gameweek=args.gameweek, part=args.part, include_images=args.include_images)
            print(f"Saved immutable model outputs: {result}")
            print("Earlier snapshots were retained.")
            if "_pre_deadline_" not in result.name:
                print("WARNING: not a pre-deadline snapshot; don't use it for prospective scoring of this gameweek.")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Snapshot not saved/verified: {error}\n")


if __name__ == "__main__":
    main()
