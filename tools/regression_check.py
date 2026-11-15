"""
Regression check for refactoring: proves a code change leaves every pipeline output unchanged.

It runs the offline stages of the weekly pipeline (parts 1-3, minus anything that fetches from the
network, reads the Downloads folder or writes a snapshot) in a throwaway sandbox on local disk,
against a frozen copy of the inputs, and records a content signature of every output. Two runs
are then compared file by file; CSVs are compared with row order ignored, because some stages'
row order depends on Python's hash seed (CLAUDE.md §5.2).

Typical use during a refactor:

    python tools/regression_check.py freeze                  # once: freeze data/current + cache
    python tools/regression_check.py run --ref golden-base   # baseline, built from a git tag/commit
    python tools/regression_check.py run                     # the working tree
    python tools/regression_check.py compare golden-base worktree

Both runs read the same frozen inputs on the same day, so stages that look at the calendar or
the clock affect both alike. Sandboxes live in %LOCALAPPDATA%/fpl_regression (outside OneDrive);
manifests and logs are kept in data/validation/regression_check/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "validation" / "regression_check"
FROZEN = STORE / "frozen_inputs"
SANDBOXES = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "fpl_regression"

# Stages left out of the check, by script name, with the reason.
SKIPPED_STAGES = {
    "fetch_bootstrap": "network",
    "fetch_gameweek_stats": "network",
    "fetch_fixtures_results": "network",
    "fetch_fotmob_matches": "network",
    "forward_check": "scores frozen forecasts into data/validation",
    "fixture_ticker": "network (live FPL fixtures)",
    "predicted_table": "network (live FPL fixtures)",
    "predicted_scores": "network (live FPL fixtures)",
    "sofascore_weekly": "reads the Downloads folder",
    "save_gameweek_outputs": "writes a snapshot",
}
# Data folders the checked stages only read: linked into the sandbox, then checked for writes.
LINKED_DATA = ["raw", "fplreview", "optimizer", "solio", "snapshots", "validation", "archive"]
WRITE_CHECKED = ["raw", "fplreview", "optimizer", "solio"]  # the big folders are not rescanned
OUTPUT_DIRS = ["data/current", "images", "site/data"]
# Environment variables that switch model behaviour; cleared so both runs use the defaults.
SWITCH_PREFIXES = ("GOAL_", "DEFCON_", "BONUS_", "CARD_", "KEEPER_", "XMINS_", "FPL_ONLY_",
                   "FOTMOB_", "DEFENDER_", "SAVE_", "SUB_")
# JSON keys holding the time of the run itself; two runs of the same code differ only in these
# (measured on 2026-11-15: two baseline runs, 119 of 123 files identical, the rest only these keys).
VOLATILE_JSON_KEYS = {"built_utc", "generated_utc"}


def _drop_volatile(obj):
    if isinstance(obj, dict):
        return {k: _drop_volatile(v) for k, v in obj.items() if k not in VOLATILE_JSON_KEYS}
    if isinstance(obj, list):
        return [_drop_volatile(v) for v in obj]
    return obj


def freeze() -> None:
    """Copy data/current and the model cache once; every later run reads this copy."""
    if FROZEN.exists():
        sys.exit(f"{FROZEN} already exists; delete it first to re-freeze.")
    FROZEN.mkdir(parents=True)
    shutil.copytree(ROOT / "data" / "current", FROZEN / "current")
    if (ROOT / "cache").exists():
        shutil.copytree(ROOT / "cache", FROZEN / "cache")
    (FROZEN / "FROZEN_AT.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print(f"Frozen inputs in {FROZEN}")


def _code_files_worktree() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def _build_sandbox(label: str, ref: str | None) -> Path:
    box = SANDBOXES / label
    if box.exists():
        shutil.rmtree(box)
    box.mkdir(parents=True)
    if ref:
        archive = subprocess.run(["git", "archive", "--format=tar", ref], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(box, filter="data")
    else:
        for rel in _code_files_worktree():
            src = ROOT / rel
            if src.is_file():
                (box / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, box / rel)
    (box / "data").mkdir(exist_ok=True)
    shutil.copytree(FROZEN / "current", box / "data" / "current")
    if (FROZEN / "cache").exists():
        shutil.copytree(FROZEN / "cache", box / "cache")
    import _winapi  # Windows junctions need no admin rights, unlike symlinks
    for name in LINKED_DATA:
        if (ROOT / "data" / name).exists():
            _winapi.CreateJunction(str(ROOT / "data" / name), str(box / "data" / name))
    return box


def _stage_list(box: Path) -> list[dict]:
    """Read the stage lists from the sandbox's own orchestrators, so moved scripts are followed."""
    code = (
        "import json, sys\n"
        "import run_post_gameweek as a, run_pre_deadline as b, run_site_export as c\n"
        "out = []\n"
        "for s in a.STAGES + b.STAGES + c.STAGES:\n"
        "    if s.get('season_start_only'): continue\n"
        "    out.append({'label': s['label'], 'script': str(s['script']), 'args': s.get('args', [])})\n"
        "print(json.dumps(out))\n"
    )
    res = subprocess.run([sys.executable, "-X", "utf8", "-c", code], cwd=box,
                         capture_output=True, text=True, encoding="utf-8")
    if res.returncode:
        sys.exit(f"Could not read the stage lists:\n{res.stderr}")
    stages = json.loads(res.stdout.strip().splitlines()[-1])
    return [s for s in stages if Path(s["script"]).stem not in SKIPPED_STAGES]


def _clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(SWITCH_PREFIXES)}
    env.update(PYTHONIOENCODING="utf-8", PYTHONHASHSEED="0", MPLBACKEND="Agg")
    return env


def _newer_files(folder: Path, since: float) -> list[str]:
    hits = []
    for dirpath, _, files in os.walk(folder):
        for f in files:
            p = Path(dirpath) / f
            if p.stat().st_mtime > since:
                hits.append(str(p))
    return hits


def _signature(path: Path) -> dict:
    if path.suffix == ".csv":
        text = path.read_text(encoding="utf-8-sig")
        lines = [line.rstrip() for line in text.splitlines()]
        header, rows = (lines[0] if lines else ""), sorted(lines[1:])
        digest = hashlib.sha256("\n".join([header, *rows]).encode()).hexdigest()
        return {"kind": "csv", "rows": len(rows), "sha256": digest}
    if path.suffix == ".json":
        try:
            obj = _drop_volatile(json.loads(path.read_text(encoding="utf-8-sig")))
            digest = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
            return {"kind": "json", "sha256": digest}
        except ValueError:
            pass
    return {"kind": "bytes", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(label: str | None, ref: str | None) -> None:
    if not FROZEN.exists():
        sys.exit("No frozen inputs; run `freeze` first.")
    label = label or (ref.replace("/", "_") if ref else "worktree")
    box = _build_sandbox(label, ref)
    logs = STORE / "logs" / label
    if logs.exists():
        shutil.rmtree(logs)
    logs.mkdir(parents=True)
    started = time.time()
    stages = _stage_list(box)
    print(f"[{label}] {len(stages)} stages in {box}")
    for i, stage in enumerate(stages, 1):
        script = Path(stage["script"])
        if not script.is_relative_to(box):
            sys.exit(f"Stage {stage['label']} points outside the sandbox: {script}")
        t0 = time.time()
        with open(logs / f"{i:02d}_{script.stem}.log", "w", encoding="utf-8") as log:
            # package modules run as `python -m` from the sandbox root, as pipeline_stages.py runs them
            target = (["-m", ".".join(script.relative_to(box).with_suffix("").parts)]
                      if script.is_relative_to(box / "fpl_model") else [str(script)])
            code = subprocess.call([sys.executable, "-u", "-X", "utf8", *target, *stage["args"]],
                                   cwd=box, env=_clean_env(), stdout=log, stderr=subprocess.STDOUT)
        print(f"  {i:02d} {stage['label']}: exit {code}, {time.time() - t0:.0f} s")
        if code:
            sys.exit(f"Stage failed; see {logs}")
    written = [f for name in WRITE_CHECKED if (ROOT / "data" / name).exists()
               for f in _newer_files(ROOT / "data" / name, started)]
    manifest = {
        "label": label, "ref": ref, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "frozen_at": (FROZEN / "FROZEN_AT.txt").read_text(encoding="utf-8"),
        "stages": [s["label"] for s in stages], "writes_to_linked_data": written, "files": {},
    }
    manifest["files"] = _sign_outputs(box)
    (STORE / f"manifest_{label}.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[{label}] {len(manifest['files'])} files signed, {time.time() - started:.0f} s total")
    if written:
        print(f"WARNING: {len(written)} files written into linked data folders, e.g. {written[:3]}")


def _sign_outputs(box: Path) -> dict:
    files = {}
    for out_dir in OUTPUT_DIRS:
        base = box / out_dir
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file():
                    files[p.relative_to(box).as_posix()] = _signature(p)
    return files


def resign(label: str) -> None:
    """Recompute a finished run's signatures (after a change to the signing rules), without rerunning it."""
    path = STORE / f"manifest_{label}.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"] = _sign_outputs(SANDBOXES / label)
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[{label}] re-signed {len(manifest['files'])} files")


def _csv_detail(a: Path, b: Path) -> str:
    try:
        da = pd.read_csv(a, encoding="utf-8-sig", low_memory=False)
        db = pd.read_csv(b, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:  # unreadable CSVs are reported, not compared
        return f"unreadable ({exc})"
    if list(da.columns) != list(db.columns):
        return f"columns differ: -{sorted(set(da.columns) - set(db.columns))} +{sorted(set(db.columns) - set(da.columns))}"
    if len(da) != len(db):
        return f"rows {len(da)} -> {len(db)}"
    key = [c for c in da.columns if da[c].dtype == object] or list(da.columns)
    da = da.sort_values(key, kind="mergesort").reset_index(drop=True)
    db = db.sort_values(key, kind="mergesort").reset_index(drop=True)
    num = da.select_dtypes("number").columns
    worst = max(((float((da[c] - db[c]).abs().max()), c) for c in num), default=(0.0, None))
    text_diff = [c for c in da.columns if c not in num and not da[c].equals(db[c])]
    return f"max |diff| {worst[0]:.3g} in {worst[1]}; text columns differing: {text_diff[:5]}"


def compare(a: str, b: str) -> int:
    ma = json.loads((STORE / f"manifest_{a}.json").read_text(encoding="utf-8"))
    mb = json.loads((STORE / f"manifest_{b}.json").read_text(encoding="utf-8"))
    fa, fb = ma["files"], mb["files"]
    changed = sorted(k for k in fa.keys() & fb.keys() if fa[k]["sha256"] != fb[k]["sha256"])
    only_a, only_b = sorted(fa.keys() - fb.keys()), sorted(fb.keys() - fa.keys())
    print(f"{a} v {b}: {len(fa.keys() & fb.keys())} common files, {len(changed)} differ, "
          f"{len(only_a)} only in {a}, {len(only_b)} only in {b}")
    for k in only_a:
        print(f"  only in {a}: {k}")
    for k in only_b:
        print(f"  only in {b}: {k}")
    for k in changed:
        detail = ""
        if k.endswith(".csv"):
            detail = _csv_detail(SANDBOXES / a / k, SANDBOXES / b / k)
        print(f"  DIFF {k} {detail}")
    if ma["stages"] != mb["stages"]:
        print("  stage lists differ:", ma["stages"], mb["stages"], sep="\n    ")
    ok = not (changed or only_a or only_b)
    print("IDENTICAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p_run = sub.add_parser("run")
    p_run.add_argument("--ref", help="git ref to build the sandbox from (default: the working tree)")
    p_run.add_argument("--label", help="name for this run (default: the ref, or 'worktree')")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    p_sign = sub.add_parser("resign")
    p_sign.add_argument("label")
    args = parser.parse_args()
    if args.cmd == "freeze":
        freeze()
    elif args.cmd == "run":
        run(args.label, args.ref)
    elif args.cmd == "resign":
        resign(args.label)
    else:
        return compare(args.a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
