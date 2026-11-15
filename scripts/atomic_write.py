"""Write a CSV through a temp file, so a crash or a locked file can't leave a half-written one (§189).

Same pattern as player_identity.write_csv: write beside the target, then replace it in one step.
"""
from pathlib import Path


def to_csv_atomic(frame, path, **to_csv_args):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(tmp, **to_csv_args)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
