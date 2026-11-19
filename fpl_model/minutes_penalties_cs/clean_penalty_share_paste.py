"""
Converts the raw Excel paste from FPL Review's penalty-share page into a
clean penalty_share.csv.

The source paste has THREE rows per player (as of this version): a stats row
(team code, price, GW values, avg, elite own%), a name row
("{Name}{Team} • {Pos}"), and a third row containing either a bare number
(observed: 86, 93, 84, 11, 0...) or a dash "-" for fringe/bench players
(backup goalkeepers, academy players, squad depth generally) -- likely some
kind of starts-confidence or minutes-projection metric that's undefined for
players unlikely to feature. That third row is skipped and discarded either
way, since nothing in this pipeline currently needs it -- but its presence
and shape is validated on every run specifically so that if the website's
layout changes again, this script fails loudly with a clear diagnostic
rather than silently mis-parsing the way the 2-row version did when this
3-row structure was first introduced.

An earlier version of this script assumed exactly 2 rows per player -- that
assumption broke when FPL Review added the mystery third row, causing
progressive misalignment that cascaded into most rows failing to parse.

Output values are plain decimals (0.94, not "94%") -- build_penalty_predictions.py
expects this directly.

Usage: copy-paste from FPL Review into Excel, save as review_pk_share.xlsx
in this folder, then run this script.

Run locally:
    python -m fpl_model.minutes_penalties_cs.clean_penalty_share_paste
"""
import re

import pandas as pd

from fpl_model import paths
from fpl_model.forecast_horizon import GAMEWEEKS, HORIZON  # noqa: E402  (the paste must cover the pipeline's window)

_DATA = paths.CURRENT
RAW_FILE = paths.current_file("review_pk_share.xlsx", root=paths.CURRENT)
OUT_FILE = paths.current_file("penalty_share.csv", root=paths.CURRENT)

MYSTERY_ROW_PATTERN = re.compile(r"^(\d+(\.\d+)?|-)$")


def main():
    df = pd.read_excel(RAW_FILE, header=None)

    header_candidates = df[df[1].astype(str).str.contains("Price", na=False)].index
    if len(header_candidates) == 0:
        raise ValueError("Could not find the header row (looking for 'Price' in column B) -- "
                          "check that the paste includes the actual data table, not just the "
                          "page settings panel above it.")
    header_row = header_candidates[0]
    # The gameweek columns follow Price: HORIZON of them (forecast_horizon.py), then "Avg".
    labels = [str(v).strip() for v in df.iloc[header_row, 2:].tolist()]
    if not all(re.fullmatch(r"GW\d+", lab) for lab in labels[:HORIZON]) or labels[HORIZON] != "Avg":
        raise ValueError(f"Expected {HORIZON} gameweek columns then 'Avg' after Price "
                         f"(forecast_horizon.HORIZON), found {labels}")
    print(f"Gameweek columns: {labels[0]}-{labels[HORIZON - 1]} -> GW1-GW{HORIZON}")
    data = df.iloc[header_row + 1:].reset_index(drop=True)

    rows = []
    skipped = []
    mystery_row_mismatches = 0
    i = 0
    while i < len(data) - 2:
        stats_row = data.iloc[i]
        name_row = data.iloc[i + 1]
        mystery_row = data.iloc[i + 2]

        if pd.isna(stats_row[0]) or pd.isna(name_row[0]):
            i += 1
            continue

        team = str(stats_row[0]).strip()
        price_raw = str(stats_row[1])
        price = price_raw.split("PP:")[0].replace("£", "").replace("m", "").strip()

        name_pos = str(name_row[0])
        if " • " not in name_pos:
            skipped.append(name_pos)
            i += 3
            continue
        name_and_team, pos = name_pos.split(" • ")

        if name_and_team.endswith(team):
            player = name_and_team[:-len(team)]
        else:
            skipped.append(name_pos)
            i += 3
            continue

        # validate the mystery third row matches the expected bare-number
        # shape -- if it doesn't, the layout has likely shifted again
        mystery_val = mystery_row[0]
        if pd.isna(mystery_val) or not MYSTERY_ROW_PATTERN.match(str(mystery_val).strip()):
            mystery_row_mismatches += 1

        gw_values = stats_row[2:2 + HORIZON].tolist()  # relabelled GW1..GW<HORIZON>, GW1 = next round
        rows.append({"Player": player, "Team": team, "Price": price,
                     **{f"GW{gw}": v for gw, v in zip(GAMEWEEKS, gw_values)}})
        i += 3

    if skipped:
        print(f"WARNING: {len(skipped)} rows couldn't be parsed cleanly, skipped:")
        for s in skipped:
            print(f"  {s}")

    if mystery_row_mismatches > 0:
        print(f"\nWARNING: {mystery_row_mismatches} of {len(rows)} parsed players had a "
              f"third row that didn't match the expected bare-number pattern. This may mean "
              f"the site's layout has changed AGAIN (e.g. a 4th row added, or the mystery "
              f"row's content changed) -- results were still written, but verify them "
              f"carefully rather than trusting this run blindly.")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(out)} players to {OUT_FILE}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
