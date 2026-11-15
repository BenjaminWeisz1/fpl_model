"""CHANGELOG §167: reruns §144, §146, §152 and §160 on the repaired identities, into new folders.

The rating log stays the 2026-10-30 one, so differences come from the repair.
Run: python -u -X utf8 scripts/bonus/rerun_after_name_repair.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "assemble")]
from fpl_model.bonus import validate_bonus_calibration as calibration  # noqa: E402
from fpl_model.bonus import validate_prematch_coherence_history as hist  # noqa: E402
from fpl_model.bonus import validate_residual_window as window  # noqa: E402

VAL = calibration.ROOT / "data" / "validation"


def main():
    hist.OUT_DIR = VAL / "prematch_bonus_history_20261101"
    hist.OUT_DIR.mkdir(exist_ok=True)
    print("=== §144 ===", flush=True)
    hist.main()
    window.OUT_DIR = VAL / "residual_window_20261101"
    print("=== §146 ===", flush=True)
    window.main()
    from fpl_model.assemble import player_error_breakdown as E
    from fpl_model.assemble import points_error_breakdown as P
    P.OUT = VAL / "points_error_20261101"
    P.OUT.mkdir(exist_ok=True)
    print("=== §152 ===", flush=True)
    P.main()
    E.OUT = VAL / "player_error_20261101_repaired"
    E.OUT.mkdir(exist_ok=True)
    print("=== §160 ===", flush=True)
    E.main()


if __name__ == "__main__":
    main()
