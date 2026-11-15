"""
Shared stage runner for the two weekly orchestrators.

The pipeline is split across two scripts, because its two halves want to run
at different times in the week:

  * ``run_post_gameweek.py``  -- immediately after a gameweek finishes: refresh
    the data, then re-fit everything that depends only on what happened on the
    pitch (team ratings, player goal/assist ratings, defcon).
  * ``run_pre_deadline.py``   -- later in the week, once expected minutes and
    the penalty-share table have settled: everything that turns those ratings
    into per-gameweek predicted points.

Both import ``run_pipeline`` from here so the stage mechanics (UTF-8-safe
subprocess capture, per-stage timing, halt-on-first-failure, output-file
summaries) live in exactly one place.

Stage dicts take:
    label              -- printed name
    script             -- Path to the script to run
    args               -- optional list of extra CLI args
    summary_files      -- optional list of data/current filenames to summarise
    season_start_only  -- optional bool; skipped unless --preseason was passed
    accepts_gameweek   -- optional bool; receives --gameweek N when the run was given it
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "fpl_model"
DATA = ROOT / "data" / "current"

# Child Pythons write UTF-8 to their captured stdout/stderr pipes
# (PYTHONIOENCODING affects only the child's stdio, not its file I/O), and
# we decode the same way below -- otherwise any stage printing a name like
# "Słonina" through the pipe crashes with UnicodeEncodeError under the
# Windows cp1252 default.
_CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}


def stage_command(script: Path) -> list:
    """The command that runs a stage: package modules as `python -m`, anything else by path."""
    if script.is_relative_to(SCRIPTS):
        return [sys.executable, "-m", ".".join(script.relative_to(ROOT).with_suffix("").parts)]
    return [sys.executable, str(script)]


def _file_info(path: Path) -> str:
    """Return a one-line summary of a CSV file (row count + last-modified)."""
    if not path.exists():
        return f"  {path.name}: NOT FOUND"
    import csv
    with open(path, encoding="utf-8-sig") as f:
        n_rows = sum(1 for _ in csv.reader(f)) - 1  # subtract header
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))
    return f"  {path.name}: {n_rows} rows (updated {mtime})"


def parse_gameweek(argv: list) -> int | None:
    """The value of --gameweek N, or None when it wasn't given."""
    if "--gameweek" not in argv:
        return None
    position = argv.index("--gameweek") + 1
    value = argv[position] if position < len(argv) else ""
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"--gameweek needs a positive whole number, got '{value}'")
    return int(value)


def run_stage(stage: dict, preseason: bool = False, gameweek: int = None) -> bool:
    """Run one pipeline stage. Returns True on success, False on failure."""
    if stage.get("season_start_only") and not preseason:
        print(f"  SKIP  {stage['label']} (--preseason not set)")
        return True

    script = stage["script"]
    if not script.exists():
        print(f"  ✗ FAILED {stage['label']}: {script} not found")
        return False

    print(f"\n{'─' * 60}")
    print(f"▶ {stage['label']}")
    print(f"  Running: {script.relative_to(ROOT)}")
    t0 = time.time()
    extra = ["--gameweek", str(gameweek)] if gameweek is not None and stage.get("accepts_gameweek") else []

    result = subprocess.run(
        [*stage_command(script), *stage.get("args", []), *extra],
        cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env=_CHILD_ENV,
    )
    elapsed = time.time() - t0

    # Print captured stdout (trimmed)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  │ {line}")
    if result.returncode != 0:
        print(f"  ✗ FAILED (exit code {result.returncode}, {elapsed:.1f}s)")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"  │ ERR: {line}")
        return False

    print(f"  ✓ Done ({elapsed:.1f}s)")

    # Summarize output files
    for fname in stage.get("summary_files", []):
        print(_file_info(DATA / fname))

    return True


def run_pipeline(title: str, stages: list, argv: list = None) -> int:
    """Run *stages* in order, halting at the first failure.

    Returns a process exit code (0 on success, 1 on failure) so callers can
    ``sys.exit(run_pipeline(...))``.
    """
    argv = sys.argv if argv is None else argv
    preseason = "--preseason" in argv
    try:
        gameweek = parse_gameweek(list(argv))
    except ValueError as error:
        print(error)
        return 1

    print("=" * 60)
    print(title)
    print("=" * 60)
    if preseason:
        print("  Mode: season-start (preseason ratings blend included)")
    else:
        print("  Mode: weekly update")
    if gameweek is not None:
        print(f"  Snapshot gameweek: GW{gameweek} (--gameweek)")

    t_start = time.time()
    failed = []

    for stage in stages:
        ok = run_stage(stage, preseason=preseason, gameweek=gameweek)
        if not ok:
            failed.append(stage["label"])
            print(f"\n⚠ Pipeline halted at '{stage['label']}' -- fix the error above and re-run.")
            break

    total = time.time() - t_start
    print(f"\n{'═' * 60}")
    if failed:
        print(f"Pipeline FAILED at: {', '.join(failed)} ({total:.1f}s total)")
    else:
        print(f"Pipeline completed successfully ({total:.1f}s total)")
    print(f"{'═' * 60}")
    return 1 if failed else 0
