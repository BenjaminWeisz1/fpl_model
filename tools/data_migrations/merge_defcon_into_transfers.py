"""
Merges Defcon (prior-season defensive contribution data for new-to-league
players) into the main transfers.csv, matching on Player + Season.

Defcon will be null for the vast majority of rows -- it's only populated for
players new to the league this season who have prior-league defensive data.
That's expected, not a data quality issue: existing-league players get their
defcon history from player_match_stats.csv directly, not from transfers.csv.

Run locally:s
    python merge_defcon_into_transfers.py
"""

import pandas as pd

from fpl_model import paths

_DATA = paths.CURRENT

MAIN_TRANSFERS_FILE = paths.current_file("transfers.csv", root=paths.CURRENT)
DEFCON_TRANSFERS_FILE = paths.SOURCED / "defcon_transfers.csv"  # the separate Player,Season,Pos,Minutes,Defcon file

main = pd.read_csv(MAIN_TRANSFERS_FILE)
defcon = pd.read_csv(DEFCON_TRANSFERS_FILE)

defcon_lookup = defcon.set_index(["Player", "Season"])["Defcon"].to_dict()
main["Defcon"] = main.apply(lambda r: defcon_lookup.get((r["Player"], r["Season"])), axis=1)

n_populated = main["Defcon"].notna().sum()
print(f"Defcon populated for {n_populated} of {len(main)} rows in {MAIN_TRANSFERS_FILE}")
print("(expected to be small -- only new-to-league players get this)")

unmatched = set(defcon["Player"]) - set(main["Player"])
if unmatched:
    print(f"\n{len(unmatched)} players in {DEFCON_TRANSFERS_FILE} not found in {MAIN_TRANSFERS_FILE} at all:")
    for p in sorted(unmatched):
        print(f"  {p}")

main.to_csv(MAIN_TRANSFERS_FILE, index=False, encoding="utf-8-sig")
print(f"\nUpdated {MAIN_TRANSFERS_FILE} in place.")
