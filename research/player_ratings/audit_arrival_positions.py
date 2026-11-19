"""
Audit transfers.csv positions of arrivals against Transfermarkt starting positions before arrival (CHANGELOG §239).

    python -m research.player_ratings.audit_arrival_positions     # -> data/validation/position_audit_20261113/audit.csv, flagged.csv, summary.json

Source: the frozen Transfermarkt dataset in data/raw/transfermarkt/ (game_lineups: one row per player and match with
his lineup position, 2013 - July 2026, the major European leagues and cups). Only starting-lineup rows are used; a
substitute's row carries his general position, not where he played. Window: the three seasons before the arrival
season (1 Aug of Season - 3 to 31 Jul of Season - 1), any club and competition.

A listing is plainly wrong (rule fixed before any player was looked at, and independent of Premier League outcomes):
    * at least MIN_STARTS starts in the window, and
    * the first listed position, with its label neighbours (COMPAT), has under FIRST_SHARE_MAX of those starts, and
    * Second Striker is under SS_MAX of the starts (it can be a #10 or a forward in a front two: ambiguous), and
    * the listing built from the starts (positions with at least SHARE_MIN of starts, most first, at most three) differs
      from the current one by at least RATE_GAP per 90 in the npxG or xAG position rate (§77 rule).
Flagged rows are proposals; transfers.csv is edited separately (text-level, backed up).

Amended after the first run (before any edit, and not by outcomes; CHANGELOG §239): Transfermarkt has no wing-back
label (wing-backs in a back three are "Left/Right Midfield") and its winger / wide-midfielder label follows the
formation, so neighbouring wide labels count as compatible; Second Striker is ambiguous; and the loosest name match
(one shared token) was dropped after it paired Nicolas Pépé with Pepe Reina.
"""
import json
import re
import unicodedata

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
TM = paths.RAW / "transfermarkt"
OUT = paths.VALIDATION / "position_audit_20261113"

MIN_STARTS = 15
SS_MAX = 0.20
# Labels that can describe the same role in the two sources: the first listed position counts these starts too.
COMPAT = {"W": {"W", "WM"}, "WM": {"WM", "W", "WB"}, "WB": {"WB", "WM", "FB"}, "FB": {"FB", "WB"},
          "AM": {"AM", "SS"}, "FW": {"FW", "SS"}}
FIRST_SHARE_MAX = 0.10
SHARE_MIN = 0.20
RATE_GAP = 0.05
# Transfermarkt lineup position -> position_combinations.csv single position. Second Striker has no slot of its own;
# it is mapped to AM (its rate sits between AM and FW) and its share is reported.
TM_POS = {"Centre-Back": "CB", "Sweeper": "CB", "Left-Back": "FB", "Right-Back": "FB", "Defensive Midfield": "DM",
          "Central Midfield": "CM", "Attacking Midfield": "AM", "Left Midfield": "WM", "Right Midfield": "WM",
          "Left Winger": "W", "Right Winger": "W", "Centre-Forward": "FW", "Second Striker": "SS", "Goalkeeper": "GK"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", s).split()


def rate_fn():
    single = pd.read_csv(paths.current_file("position_combinations.csv", root=DATA)).set_index("Pos")

    def rate(pos, col):
        parts = [p.strip() for p in str(pos).split(",")]
        w = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
        return float(np.average(single.loc[parts, col], weights=w))
    return rate


def tm_player_ids(arrivals):
    """Transfermarkt player_id per (Player, Season): a Premier League (GB1) appearance in the arrival season with the same
    full name, else a unique full-name match among all Transfermarkt players, else a GB1 appearance that season sharing
    the last name and at least one more name token."""
    players = pd.read_csv(TM / "players.csv.gz", usecols=["player_id", "name"])
    players["nm"] = players["name"].map(lambda x: " ".join(norm(x)))
    unique_name = players.drop_duplicates("nm", keep=False).set_index("nm")["player_id"]
    apps = pd.read_csv(TM / "appearances.csv.gz", usecols=["player_id", "player_name", "competition_id", "date"])
    apps["date"] = pd.to_datetime(apps["date"])
    apps["season"] = np.where(apps["date"].dt.month >= 7, apps["date"].dt.year, apps["date"].dt.year - 1)
    pl = apps[apps["competition_id"].eq("GB1")].drop_duplicates(["player_id", "season"])
    by_season = {s: list(zip(g["player_id"], g["player_name"].map(norm))) for s, g in pl.groupby("season")}
    out = {}
    for r in arrivals.itertuples():
        ours = norm(r.Player)
        cands = by_season.get(r.Season, [])
        exact = [pid for pid, nm in cands if nm == ours]
        if len(exact) == 1:
            out[(r.Player, r.Season)] = (exact[0], "full name")
            continue
        if " ".join(ours) in unique_name.index:
            out[(r.Player, r.Season)] = (int(unique_name[" ".join(ours)]), "full name, all players")
            continue
        last = ours[-1] if ours else ""
        two = [pid for pid, nm in cands if nm and nm[-1] == last and len(set(nm) & set(ours)) >= 2]
        if len(set(two)) == 1:
            out[(r.Player, r.Season)] = (two[0], "last name + token")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_csv(paths.current_file("transfers.csv", root=DATA), encoding="utf-8-sig", usecols=["Player", "Season", "Pos", "League", "Notes"])
    arr = t[t["Notes"].isna() & t["League"].ne("Premier League") & t["Pos"].notna() & t["Season"].ge(2017)
            & ~t["Pos"].astype(str).str.contains("GK")].copy()
    ids = tm_player_ids(arr)
    lu = pd.read_csv(TM / "game_lineups.csv.gz", usecols=["date", "player_id", "type", "position"])
    lu = lu[lu["type"].eq("starting_lineup")].copy()
    lu["date"] = pd.to_datetime(lu["date"])
    lu["slot"] = lu["position"].map(TM_POS)
    by_pid = dict(tuple(lu.groupby("player_id")))
    rate = rate_fn()
    rows = []
    for r in arr.itertuples():
        pid, how = ids.get((r.Player, r.Season), (None, None))
        row = {"Player": r.Player, "Season": r.Season, "League": r.League, "Listed": r.Pos, "TM_player_id": pid,
               "Match": how}
        if pid is not None and pid in by_pid:
            g = by_pid[pid]
            lo, hi = pd.Timestamp(r.Season - 3, 8, 1), pd.Timestamp(r.Season, 7, 31)
            g = g[(g["date"] >= lo) & (g["date"] <= hi) & g["slot"].notna()]
            n = len(g)
            share = g["slot"].value_counts(normalize=True)
            row.update({"TM_starts": n, "TM_shares": "; ".join(f"{k} {v:.0%}" for k, v in share.items()),
                        "SS_share": float((g["position"] == "Second Striker").mean()) if n else 0.0})
            if n:
                first = str(r.Pos).split(",")[0].strip()
                # derived listing: Second Striker read as AM (its rate sits between AM and FW)
                share_am = g["slot"].replace("SS", "AM").value_counts(normalize=True)
                derived = [k for k, v in share_am.items() if v >= SHARE_MIN and k != "GK"][:3] or [share_am.index[0]]
                row["TM_listing"] = ",".join(derived)
                row["First_listed_share"] = float(sum(share.get(k, 0.0) for k in COMPAT.get(first, {first})))
                for col, lab in (("NPG/90", "npxG"), ("Assists/90", "xAG")):
                    row[f"{lab}_rate_listed"] = rate(r.Pos, col)
                    row[f"{lab}_rate_TM"] = rate(row["TM_listing"], col)
                gap = max(abs(row["npxG_rate_TM"] - row["npxG_rate_listed"]), abs(row["xAG_rate_TM"] - row["xAG_rate_listed"]))
                row["Rate_gap"] = gap
                ss = float(share.get("SS", 0.0))
                plain = n >= MIN_STARTS and row["First_listed_share"] < FIRST_SHARE_MAX and gap >= RATE_GAP
                row["Flag"] = bool(plain and ss < SS_MAX)
                row["Ambiguous_SS"] = bool(plain and ss >= SS_MAX)
                if "GK" in share.index and share["GK"] > 0.5:
                    row["Flag"], row["Match"] = False, f"{how}; rejected: Transfermarkt player is a keeper"
        rows.append(row)
    a = pd.DataFrame(rows)
    a["Flag"] = a.get("Flag", pd.Series(False, index=a.index)).fillna(False).astype(bool)
    a.to_csv(OUT / "audit.csv", index=False, encoding="utf-8-sig")
    flagged = a[a["Flag"]].sort_values(["Season", "Player"])
    flagged.to_csv(OUT / "flagged.csv", index=False, encoding="utf-8-sig")
    amb = a[a.get("Ambiguous_SS", pd.Series(False, index=a.index)).fillna(False).astype(bool)]
    summary = {"arrivals": int(len(a)), "matched_to_transfermarkt": int(a["TM_player_id"].notna().sum()),
               "with_15_starts": int((a.get("TM_starts", 0) >= MIN_STARTS).sum()), "flagged": int(len(flagged)), "ambiguous_second_striker": amb[["Player", "Season", "Listed", "TM_shares"]].values.tolist(),
               "match_methods": a["Match"].fillna("unmatched").value_counts().to_dict(),
               "rule": {"min_starts": MIN_STARTS, "first_share_max": FIRST_SHARE_MAX, "share_min": SHARE_MIN, "rate_gap": RATE_GAP},
               "by_league_matched": a.groupby("League")["TM_player_id"].apply(lambda s: f"{s.notna().sum()}/{len(s)}").to_dict()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_league_matched"}, indent=1))
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60):
        print(flagged[["Player", "Season", "League", "Listed", "TM_listing", "TM_starts", "First_listed_share",
                       "Rate_gap", "TM_shares"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
