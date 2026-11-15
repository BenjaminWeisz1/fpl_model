"""
Rebuilds the 2017-2024 rows of money_priors_by_season.csv from past seasons only (CHANGELOG §200-§202).

The stored 2017-2025 priors came from one OLS fit of season GF and GA on Wages_pct and Squad_Value_pct over
2013-2024, applied to every season, so each 2017-2024 prior had its own season (and later ones) in its training
data. This refits each season s on seasons < s only, with the same estimator (OLS) and the same features as
historical_money.csv gives them, and writes predicted season totals / 38 as before.

Since §211 (2026-11-09: 2025 squad values re-taken at Transfermarkt's 15 Aug cut-off) 2025 is rebuilt too, the
2017-2024 rows must come out unchanged unless their inputs changed, and 2026 comes from
update_money_priors.py (ridge on past seasons) and is left alone. Only the Money_GF / Money_GA cells of 2017-2025
rows change: the file is edited as text, line endings and number format (10 significant digits) kept, and a
backup written first. Rerun update_team_ratings.py afterwards.

Run locally:
    python rebuild_money_priors_past_only.py            # dry run: prints the changes
    python rebuild_money_priors_past_only.py --apply
"""
import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"
MONEY_FILE = _DATA / "money_priors_by_season.csv"
HISTORY_FILE = _DATA / "historical_money.csv"
FEATURES = ["Wages_pct", "Squad_Value_pct"]
REBUILD = range(2017, 2026)
CHECK = None  # was 2025 until §211; set a season here to verify instead of rewriting it


def past_only_priors(seasons):
    df = pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    for col in ("Wages", "Squad_Value"):
        df[f"{col}_pct"] = df.groupby("Season")[col].transform(lambda x: x / x.sum())
    known = df[df["GF"].notna()]
    out = {}
    for s in seasons:
        train = known[known["Season"] < s]
        current = df[df["Season"] == s]
        gf = LinearRegression().fit(train[FEATURES], train["GF"]).predict(current[FEATURES]) / 38
        ga = LinearRegression().fit(train[FEATURES], train["GA"]).predict(current[FEATURES]) / 38
        out.update({(s, t): (f, a) for t, f, a in zip(current["Team"], gf, ga)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    raw = MONEY_FILE.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(newline)
    header = lines[0].lstrip("﻿").split(",")
    if header != ["Season", "Team", "Money_GF", "Money_GA"]:
        raise ValueError(f"unexpected header {header}")
    priors = past_only_priors([*REBUILD, *([CHECK] if CHECK else [])])

    changed, checked, biggest = 0, 0, 0.0
    for i, line in enumerate(lines[1:], start=1):
        if not line:
            continue
        season, team, gf, ga = line.split(",")
        key = (int(season), team)
        if key[0] == CHECK:
            if max(abs(float(gf) - priors[key][0]), abs(float(ga) - priors[key][1])) > 1e-8:
                raise ValueError(f"{key} isn't past-only OLS as expected; stop and investigate")
            checked += 1
        elif key[0] in REBUILD:
            if key not in priors:
                raise ValueError(f"no historical_money.csv row for {key}")
            new_gf, new_ga = priors[key]
            biggest = max(biggest, abs(new_gf - float(gf)), abs(new_ga - float(ga)))
            lines[i] = f"{season},{team},{new_gf:.10g},{new_ga:.10g}"
            changed += 1
    expected = 20 * len(REBUILD)
    if changed != expected or checked != (20 if CHECK else 0):
        raise ValueError(f"rewrote {changed} rows (expected {expected}) and checked {checked} {CHECK} rows")
    print(f"{changed} rows rebuilt past-only (largest change {biggest:.4f} goals per match)"
          + (f"; {CHECK} confirmed past-only" if CHECK else ""))
    if not args.apply:
        print("Dry run; pass --apply to write")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup = _DATA.parent / "archive" / f"money_priors_by_season_before_past_only_{stamp}.csv"
    if not backup.exists():
        shutil.copy2(MONEY_FILE, backup)
    tmp = MONEY_FILE.with_suffix(".csv.tmp")
    tmp.write_bytes(newline.join(lines).encode("utf-8"))
    tmp.replace(MONEY_FILE)
    print(f"Wrote {MONEY_FILE.name}; backup {backup.name}. Now rerun update_team_ratings.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
