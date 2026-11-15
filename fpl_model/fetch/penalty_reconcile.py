"""
Reconciles FPL-era penalty columns with the manual penalty_log.csv (CHANGELOG §73).

The fetchers used to read the log only when a match row was first appended, so a
penalty logged (or corrected) after its match was fetched never reached the data.
This recomputes, for every FPL-era row (Season >= FIRST_FPL_SEASON):
  player_match_stats.csv  PK, PKatt from the log; NPG = Goals - PK;
                          npxG = xG - 0.79 * PKatt, where xG is FPL's penalty-inclusive
                          expected_goals (CHANGELOG §80)
  match_results.csv       Home_/Away_PK, _PKatt from the log; _NPG = _Goals - _PK;
                          _NPxG_Adj shifted by -0.79 per change in PKatt, so any
                          hand edit to NPxG_Adj survives.
Only those columns change; every other value and the row order are preserved. Both
fetchers call reconcile_* on every run, and this script applies it once directly.

Run locally (dry run prints the changes; --apply writes them):
    python penalty_reconcile.py [--apply]
One-off (§80): add the xG column, filled from the then penalty-inclusive FPL-era npxG:
    python penalty_reconcile.py --migrate-xg --apply
"""
from fpl_model import paths
import argparse
import json

import sys
import pandas as pd
from pathlib import Path

from fpl_model.atomic_write import to_csv_atomic  # noqa: E402

_DATA = paths.CURRENT
PLAYER_STATS_FILE = paths.CURRENT / "player_match_stats.csv"
MATCH_RESULTS_FILE = paths.CURRENT / "match_results.csv"
PENALTY_LOG_FILE = paths.CURRENT / "penalty_log.csv"

FIRST_FPL_SEASON = 2025
PEN_XG_VALUE = 0.79

# penalty_log.csv name -> player_match_stats.csv name, for takers whose FPL name differs
ALIASES = {
    "Raúl Jiménez Rodríguez": "Raúl Jiménez",
    "Hwang Hee-Chan": "Hwang Hee-chan",
    "Eli Junior Kroupi": "Junior Kroupi",
}


def load_log(path=PENALTY_LOG_FILE):
    log = pd.read_csv(path, encoding="utf-8-sig")
    log["Player"] = log["Player"].replace(ALIASES)
    log["is_scored"] = log["Scored"].astype(str).str.strip().str.upper() == "TRUE"
    return log.rename(columns={"HomeAway": "Venue"})


def reconcile_player_stats(stats, log):
    """Returns (updated stats, report). Rows outside the FPL era are untouched."""
    out = stats.copy()
    # Team is part of the key: a mid-season transfer can face the same opponent at the
    # same venue twice in one season (Semenyo, 2025/26).
    key = ["Season", "Player", "Team", "Opponent", "Venue"]
    per = log.groupby(key).agg(PKatt=("is_scored", "size"), PK=("is_scored", "sum"))
    fpl = out["Data_Source"].eq("FPL_API") & (out["Season"] >= FIRST_FPL_SEASON)
    idx = pd.MultiIndex.from_frame(out.loc[fpl, key])
    new_pk = per["PK"].reindex(idx).fillna(0).astype(int).to_numpy()
    new_att = per["PKatt"].reindex(idx).fillna(0).astype(int).to_numpy()
    cols = ["PK", "PKatt", "NPG", "npxG"]
    if fpl.any() and ("xG" not in out.columns or out.loc[fpl, "xG"].isna().any()):
        raise ValueError("FPL-era rows need the penalty-inclusive xG column "
                         "(run penalty_reconcile.py --migrate-xg --apply once; CHANGELOG §80)")
    old = out.loc[fpl, cols].copy()
    out.loc[fpl, "PK"] = new_pk
    out.loc[fpl, "PKatt"] = new_att
    out.loc[fpl, "NPG"] = out.loc[fpl, "Goals"] - new_pk
    if (out.loc[fpl, "NPG"] < 0).any():
        raise ValueError("penalty_log gives more scored penalties than goals for some rows")
    # FPL's expected_goals includes 0.79 per penalty attempt; fbref-era npxG never did.
    npxg = out.loc[fpl, "xG"].astype(float) - PEN_XG_VALUE * new_att
    clipped = out.loc[fpl].loc[npxg < -0.005]
    out.loc[fpl, "npxG"] = npxg.clip(lower=0).round(2)
    changed = (old.fillna(-1) != out.loc[fpl, cols].fillna(-1)).any(axis=1)

    matched = per.index.isin(pd.MultiIndex.from_frame(out.loc[fpl, key]))
    unmatched = per[~matched & (per.index.get_level_values("Season") >= FIRST_FPL_SEASON)].reset_index()
    rows = out.loc[fpl]
    misses = (rows["PKatt"] - rows["PK"]) != rows["Penalties_Missed"].fillna(0)
    report = {"rows_changed": int(changed.sum()),
              "changed": out.loc[changed[changed].index, key + ["Goals", "PK", "PKatt", "NPG", "xG", "npxG"]]
              .assign(old_PK=old.loc[changed, "PK"], old_NPG=old.loc[changed, "NPG"],
                      old_npxG=old.loc[changed, "npxG"]).to_dict("records"),
              "xg_below_penalty_xg_clipped": clipped[key + ["PKatt", "xG"]].to_dict("records"),
              "unmatched_log_rows": unmatched.to_dict("records"),
              "log_misses_differ_from_fpl_penalties_missed": rows.loc[misses, key + [
                  "PK", "PKatt", "Penalties_Missed"]].to_dict("records")}
    return out, report


def reconcile_match_results(results, log):
    """Returns (updated results, report). Seasons before the FPL era are untouched."""
    out = results.copy()
    per = log.groupby(["Season", "Team", "Opponent", "Venue"]).agg(
        PKatt=("is_scored", "size"), PK=("is_scored", "sum"))
    fpl = out["Season"] >= FIRST_FPL_SEASON
    cols = [f"{s}_{c}" for s in ("Home", "Away") for c in ("PK", "PKatt", "NPG", "NPxG_Adj")]
    old = out.loc[fpl, cols].copy()
    for side, other in (("Home", "Away"), ("Away", "Home")):
        idx = pd.MultiIndex.from_arrays([out.loc[fpl, "Season"], out.loc[fpl, side], out.loc[fpl, other],
                                         pd.Series(side, index=out.index[fpl])])
        pk = per["PK"].reindex(idx).fillna(0).astype(int).to_numpy()
        att = per["PKatt"].reindex(idx).fillna(0).astype(int).to_numpy()
        delta_att = pd.Series(att - out.loc[fpl, f"{side}_PKatt"].fillna(0).to_numpy(), index=out.index[fpl])
        out.loc[fpl, f"{side}_PK"] = pk
        out.loc[fpl, f"{side}_PKatt"] = att
        out.loc[fpl, f"{side}_NPG"] = out.loc[fpl, f"{side}_Goals"] - pk
        # Shift NPxG_Adj only where the attempt count changed, so other values (and hand
        # edits) keep their exact text.
        moved = delta_att[delta_att != 0].index
        out.loc[moved, f"{side}_NPxG_Adj"] = (out.loc[moved, f"{side}_NPxG_Adj"]
                                              - PEN_XG_VALUE * delta_att[moved]).round(4)
    changed = (old.fillna(-1) != out.loc[fpl, cols].fillna(-1)).any(axis=1)
    sides = pd.concat([out.loc[fpl, ["Season", "Home", "Away"]].rename(columns={"Home": "Team", "Away": "Opponent"}).assign(Venue="Home"),
                       out.loc[fpl, ["Season", "Away", "Home"]].rename(columns={"Away": "Team", "Home": "Opponent"}).assign(Venue="Away")])
    unmatched = per[~per.index.isin(pd.MultiIndex.from_frame(sides[["Season", "Team", "Opponent", "Venue"]]))
                    & (per.index.get_level_values("Season") >= FIRST_FPL_SEASON)].reset_index()
    report = {"rows_changed": int(changed.sum()),
              "changed": out.loc[changed[changed].index, ["Season", "Home", "Away"] + cols]
              .assign(**{f"old_{c}": old.loc[changed, c] for c in cols}).to_dict("records"),
              "unmatched_log_team_matches": unmatched.to_dict("records")}
    return out, report


def warn(report, label):
    for k in ("unmatched_log_rows", "unmatched_log_team_matches"):
        if report.get(k):
            print(f"WARNING ({label}): {len(report[k])} penalty_log rows match no data row "
                  f"(name/team/venue spelling?): {report[k]}")
    if report.get("xg_below_penalty_xg_clipped"):
        print(f"WARNING ({label}): xG below 0.79 x PKatt, npxG set to 0: "
              f"{report['xg_below_penalty_xg_clipped']}")


def migrate_xg(apply):
    """One-off (§80): append the xG column, copying the exact text of FPL-era npxG (still
    penalty-inclusive at that point); fbref rows stay blank."""
    text = pd.read_csv(PLAYER_STATS_FILE, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "xG" in text.columns:
        raise ValueError("player_match_stats.csv already has an xG column")
    fpl = text["Data_Source"].eq("FPL_API") & (pd.to_numeric(text["Season"]) >= FIRST_FPL_SEASON)
    text["xG"] = text["npxG"].where(fpl, "")
    print(f"xG column: {int(fpl.sum())} FPL-era rows filled from npxG, {int((~fpl).sum())} blank")
    if apply:
        to_csv_atomic(text, PLAYER_STATS_FILE, index=False, encoding="utf-8-sig")


def reconcile_files(apply, report_path=None):
    """Reconcile both data files with the log; with `apply`, write only the changed cells."""
    log = load_log()
    stats = pd.read_csv(PLAYER_STATS_FILE, encoding="utf-8-sig", low_memory=False)
    results = pd.read_csv(MATCH_RESULTS_FILE, encoding="utf-8-sig", low_memory=False)
    new_stats, rs = reconcile_player_stats(stats, log)
    new_results, rr = reconcile_match_results(results, log)
    warn(rs, "player_match_stats")
    warn(rr, "match_results")
    print(f"player_match_stats rows changed: {rs['rows_changed']}; match_results rows changed: {rr['rows_changed']}")
    print(f"log misses differing from FPL penalties_missed: {rs['log_misses_differ_from_fpl_penalties_missed']}")
    if report_path:
        report_path.write_text(json.dumps({"player_match_stats": rs, "match_results": rr}, indent=2,
                                          default=str, ensure_ascii=False), encoding="utf-8")
    if apply:
        write_cells(PLAYER_STATS_FILE, stats, new_stats, ["PK", "PKatt", "NPG", "npxG"], "utf-8-sig")
        write_cells(MATCH_RESULTS_FILE, results, new_results,
                    [f"{s}_{c}" for s in ("Home", "Away") for c in ("PK", "PKatt", "NPG", "NPxG_Adj")], "utf-8")
        print("Applied.")
    return rs, rr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--migrate-xg", action="store_true")
    args = parser.parse_args()
    if args.migrate_xg:
        migrate_xg(args.apply)
        if not args.apply:
            return  # the dry-run reconcile needs the column on disk
    reconcile_files(args.apply, args.report)


def write_cells(path, old, new, cols, encoding):
    """Rewrite only the cells whose value changed, editing the file as text so every other
    cell keeps its exact original text (a typed round trip turns e.g. '0.00' into '0.0')."""
    text = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
    assert len(text) == len(old) == len(new)
    if all(((old[c] == new[c]) | (old[c].isna() & new[c].isna())).all() for c in cols):
        return  # nothing changed: leave the file untouched
    for c in cols:
        moved = ~((old[c] == new[c]) | (old[c].isna() & new[c].isna()))
        for i in moved[moved].index:
            v = new.at[i, c]
            if c.endswith("NPxG_Adj"):
                text.at[i, c] = repr(round(float(v), 4))
            elif c == "npxG":
                text.at[i, c] = repr(round(float(v), 2))
            else:
                text.at[i, c] = str(int(v))
    to_csv_atomic(text, path, index=False, encoding=encoding)


if __name__ == "__main__":
    main()
