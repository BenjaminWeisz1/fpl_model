"""
Automated red-card adjustment factors from the cached Understat match files (FotMob from 2026) (AGENDA item 9; CHANGELOG §124, §133).

The user's manual rule for NPxG_Adj: after a red card, the 11-man side's non-penalty xG counts 0.8x and the 10-man
side's 1.2x. Here, per match and side, factor = sum(xG_i * f_i) / sum(xG_i) over the side's non-penalty Understat
shots, f_i = 0.8 when the shooting side has more players at that minute, 1.2 when fewer, 1 otherwise (two-red
matches handled through the running player counts). The factor multiplies the Opta (fbref / FPL) team npxG, so
Understat's xG level never enters, only its shot timing and relative sizes.

Dismissal minute: a starter's `time` (minutes played); a substitute (roster_out = the player he replaced) enters
at the replaced player's exit minute, recursively, plus his own `time`. A shot counts as after the red card if its
minute is strictly greater than the dismissal minute.

From 2026 (CHANGELOG §197) the shots and dismissal minutes come from FotMob instead (fotmob_shots.csv,
fotmob_match_positions.csv' Red_Minute, minutes as minute + added time / 100); Understat files of those seasons are
skipped. FotMob's xG is Opta's, the provider of the base it scales. Same factor rule; own goals carry no xG and are
left out.

Since §133 the factors are applied as a part-1 stage, replacing the user's manual step. NPxG_Adj of a red-card match
= base x factor, recomputed from the base every run (never compounded); matches without a red card are not touched.
Base: 2017-2024 fbref team xG - 0.79 x summed player PKatt (2022-2024 also x the season's no-red ratio
NPxG_Adj / base: the user's source reads a little higher there); 2025+ summed player npxG (kept current by
penalty_reconcile).

    python red_card_adjust.py            # writes data/current/red_card_factors.csv and prints a check
    python red_card_adjust.py --apply    # also writes NPxG_Adj of every red-card match into match_results.csv
"""
import argparse
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.fetch.fetch_fotmob_matches import FIRST_SEASON as FOTMOB_FIRST_SEASON  # noqa: E402
from fpl_model.fetch.fetch_fotmob_matches import POSITIONS_OUT as FOTMOB_POSITIONS
from fpl_model.fetch.fetch_fotmob_matches import SHOTS_OUT as FOTMOB_SHOTS

ROOT = paths.ROOT
MATCHES = paths.RAW / "understat" / "matches"
DATA = paths.CURRENT
OUT = paths.CURRENT / "red_card_factors.csv"
UP, DOWN = 0.8, 1.2   # the user's factors for the side with more / fewer players
UP2, DOWN2 = None, None   # two-player differential (None = same as one)


def exit_minute(roster, rid, seen=()):
    """Minute at which roster entry rid left the pitch (or the match ended)."""
    v = roster[rid]
    start = 0 if v["roster_out"] in ("0", "", None) or v["roster_out"] in seen else \
        exit_minute(roster, v["roster_out"], seen + (rid,))
    return start + int(v["time"])


def dismissals(d):
    out = []
    for side in "ha":
        R = {v["id"]: v for v in d["rosters"][side].values()}
        for v in R.values():
            if v["red_card"] != "0":
                out.append((exit_minute(R, v["id"]), side, v["player"]))
    return sorted(out)


def counts_at(reds, minute):
    """Players on the pitch per side at `minute` (a shot at the dismissal minute counts as before it)."""
    n = {"h": 11, "a": 11}
    for m, sd, _ in reds:
        if minute > m:
            n[sd] -= 1
    return n


def factor_for(diff, up=UP, down=DOWN, up2=UP2, down2=DOWN2):
    """Weight for a side with `diff` more players than its opponent."""
    if diff == 0:
        return 1.0
    if diff > 0:
        return up if diff == 1 or up2 is None else up2
    return down if diff == -1 or down2 is None else down2


def match_end(d, reds):
    shots = d["shots"]["h"] + d["shots"]["a"]
    return max([90] + [int(s["minute"]) for s in shots] + [m for m, _, _ in reds])


def side_factors(d, reds, up=UP, down=DOWN, up2=UP2, down2=DOWN2):
    """Per side: shot-weighted factor and the share of its non-penalty xG taken in unequal numbers."""
    fac, share = {}, {}
    for side in "ha":
        other = "a" if side == "h" else "h"
        num = den = post = 0.0
        for s in d["shots"][side]:
            if s.get("situation") == "Penalty":
                continue
            minute, xg = int(s["minute"]), float(s["xG"])
            n = counts_at(reds, minute)
            num += xg * factor_for(n[side] - n[other], up, down, up2, down2)
            den += xg
            post += xg if n[side] != n[other] else 0.0
        fac[side] = num / den if den > 0 else 1.0
        share[side] = post / den if den > 0 else 0.0
    return fac, share


def unequal_minutes(reds, end):
    return sum(1 for t in range(end) if (lambda n: n["h"] != n["a"])(counts_at(reds, t + 0.5)))


def team_map():
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig",
                    usecols=["Season", "MatchID_Understat", "Team", "Team_Understat"]).drop_duplicates()
    names = dict(zip(u["Team_Understat"], u["Team"]))
    season = dict(zip(u["MatchID_Understat"].astype(str), u["Season"]))
    return names, season


def fotmob_matches():
    """FotMob red-card matches: (Season, Home, Away, match id, reds [(minute, side, player)], shots by side)."""
    pos = pd.read_csv(FOTMOB_POSITIONS, encoding="utf-8-sig", low_memory=False)
    shots = pd.read_csv(FOTMOB_SHOTS, encoding="utf-8-sig", low_memory=False)
    shots = shots[~shots["Own_Goal"].astype(bool)]
    by_match = {k: g for k, g in shots.groupby("MatchID_FotMob")}
    for mid, g in pos[pos["Red_Minute"].notna()].groupby("MatchID_FotMob"):
        reds = sorted((float(m), "h" if v == "Home" else "a", p)
                      for m, v, p in zip(g["Red_Minute"], g["Venue"], g["Player_FotMob"]))
        sh = by_match.get(mid, shots.iloc[0:0])
        side = {"h": [], "a": []}
        for v, m, x, sit in zip(sh["Venue"], sh["Minute"], sh["xG"], sh["Situation"]):
            side["h" if v == "Home" else "a"].append({"minute": float(m), "xG": float(x) if pd.notna(x) else 0.0,
                                                     "situation": sit})
        r0 = g.iloc[0]
        yield int(r0["Season"]), r0["Home"], r0["Away"], int(mid), reds, side


def side_factors_shots(shots, reds, up=UP, down=DOWN, up2=UP2, down2=DOWN2):
    """side_factors over plain shot lists {'h': [{minute, xG, situation}], 'a': [...]} (float minutes)."""
    fac, share = {}, {}
    for side in "ha":
        other = "a" if side == "h" else "h"
        num = den = post = 0.0
        for s in shots[side]:
            if s.get("situation") == "Penalty":
                continue
            n = counts_at(reds, s["minute"])
            num += s["xG"] * factor_for(n[side] - n[other], up, down, up2, down2)
            den += s["xG"]
            post += s["xG"] if n[side] != n[other] else 0.0
        fac[side] = num / den if den > 0 else 1.0
        share[side] = post / den if den > 0 else 0.0
    return fac, share


def build(up=UP, down=DOWN, up2=UP2, down2=DOWN2, out_path=OUT):
    names, season = team_map()
    rows = []
    for f in sorted(MATCHES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        reds = dismissals(d)
        if not reds:
            continue
        shots = d["shots"]["h"] + d["shots"]["a"]
        if not shots:
            continue
        s0 = shots[0]
        if season.get(f.stem, int(s0["season"])) >= FOTMOB_FIRST_SEASON:
            continue   # FotMob serves these seasons (§197)
        fac, share = side_factors(d, reds, up, down, up2, down2)
        end = match_end(d, reds)
        rows.append({"Season": season.get(f.stem, int(s0["season"])), "Home": names.get(s0["h_team"], s0["h_team"]),
                     "Away": names.get(s0["a_team"], s0["a_team"]), "MatchID_Understat": int(f.stem),
                     "Reds": "; ".join(f"{p} ({'home' if sd == 'h' else 'away'}) {m}'" for m, sd, p in reds),
                     "First_Red_Minute": reds[0][0], "Home_Factor": round(fac["h"], 4), "Away_Factor": round(fac["a"], 4),
                     "Home_Post_Share": round(share["h"], 4), "Away_Post_Share": round(share["a"], 4),
                     "Unequal_Minutes": unequal_minutes(reds, end), "Match_Minutes": end, "Source": "understat"})
    for ssn, home, away, mid, reds, shots in fotmob_matches():
        fac, share = side_factors_shots(shots, reds, up, down, up2, down2)
        end = int(max([90.0] + [s["minute"] for s in shots["h"] + shots["a"]] + [m for m, _, _ in reds]))
        rows.append({"Season": ssn, "Home": home, "Away": away, "MatchID_Understat": None,
                     "Reds": "; ".join(f"{p} ({'home' if sd == 'h' else 'away'}) "
                                       f"{int(m)}{'+' + str(round((m - int(m)) * 100)) if m != int(m) else ''}'"
                                       for m, sd, p in reds),
                     "First_Red_Minute": int(reds[0][0]), "Home_Factor": round(fac["h"], 4), "Away_Factor": round(fac["a"], 4),
                     "Home_Post_Share": round(share["h"], 4), "Away_Post_Share": round(share["a"], 4),
                     "Unequal_Minutes": unequal_minutes(reds, end), "Match_Minutes": end, "Source": "fotmob",
                     "MatchID_FotMob": mid})
    out = pd.DataFrame(rows).sort_values(["Season", "MatchID_Understat", "MatchID_FotMob"])
    if out_path is not None:
        to_csv_atomic(out, out_path, index=False, encoding="utf-8-sig")
    return out


def bases(m, stats=None):
    """Unadjusted team npxG of each match_results row, per side (see module docstring)."""
    p = stats if stats is not None else pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig",
                                                    low_memory=False)
    g = p.groupby(["Season", "Team", "Opponent", "Venue"]).agg(npxG=("npxG", "sum"), PKatt=("PKatt", "sum"))
    out = {}
    for side, other in (("Home", "Away"), ("Away", "Home")):
        gg = g.reindex(pd.MultiIndex.from_arrays([m["Season"], m[side], m[other], pd.Series(side, index=m.index)]))
        fb = m[f"{side}_xG"].astype(float).to_numpy() - 0.79 * gg["PKatt"].fillna(0).to_numpy()
        out[side] = pd.Series(np.where(m["Season"] <= 2024, fb, gg["npxG"].to_numpy()), index=m.index)
    return out


def red_rows(m, fac):
    keys = set(zip(fac["Season"], fac["Home"], fac["Away"]))
    return pd.Series([k in keys for k in zip(m["Season"], m["Home"], m["Away"])], index=m.index)


def level_ratios(m, red, b):
    """2022-2024: the user's no-red values sit a little above fbref; red-card matches keep that level."""
    ratio = {}
    for season in (2022, 2023, 2024):
        rows = (m["Season"] == season) & ~red
        num = sum(m.loc[rows, f"{s}_NPxG_Adj"].astype(float).sum() for s in ("Home", "Away"))
        den = sum(b[s][rows].sum() for s in ("Home", "Away"))
        ratio[season] = num / den if den > 0 else 1.0
    return ratio


def adjusted_values(m, fac, stats=None, ratio=None):
    """New NPxG_Adj of the red-card rows of match_results frame m, per side, given a factors frame."""
    b = bases(m, stats)
    red = red_rows(m, fac)
    ratio = ratio if ratio is not None else level_ratios(m, red, b)
    f = fac.set_index(["Season", "Home", "Away"])
    level = m["Season"].map(ratio).fillna(1.0)
    out = {}
    for side in ("Home", "Away"):
        idx = m.index[red]
        k = pd.MultiIndex.from_arrays([m.loc[idx, "Season"], m.loc[idx, "Home"], m.loc[idx, "Away"]])
        out[side] = b[side][idx] * level[idx] * f[f"{side}_Factor"].reindex(k).to_numpy()
    return out, ratio


def apply(fac):
    """Write NPxG_Adj of every red-card match into match_results.csv, text-level (other cells untouched)."""
    path = DATA / "match_results.csv"
    enc = "utf-8-sig" if path.read_bytes().startswith(b"\xef\xbb\xbf") else "utf-8"
    text = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=enc)
    m = pd.read_csv(path, encoding="utf-8-sig")
    vals, ratio = adjusted_values(m, fac)
    assert all(v.notna().all() for v in vals.values()), "red-card match without a base"
    changed = 0
    for side, v in vals.items():
        for i, x in v.items():
            new = repr(round(float(x), 4))
            if text.at[i, f"{side}_NPxG_Adj"] != new:
                text.at[i, f"{side}_NPxG_Adj"] = new
                changed += 1
    to_csv_atomic(text, path, index=False, encoding=enc)
    print(f"NPxG_Adj written for {len(vals['Home'])} red-card matches ({changed} cells changed); "
          f"2022-2024 level ratios {({k: round(r, 4) for k, r in ratio.items()})}")


def check(out):
    """Agreement with the user's manual 2026 values: NPxG_Adj / summed player npxG."""
    m = pd.read_csv(DATA / "match_results.csv", encoding="utf-8-sig")
    p = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    s = p.groupby(["Season", "Team", "Opponent", "Venue"])["npxG"].sum()
    rows = []
    for r in out[out["Season"] == 2026].itertuples():
        mm = m[(m["Season"] == 2026) & (m["Home"] == r.Home) & (m["Away"] == r.Away)]
        if mm.empty:
            rows.append((r.Home, r.Away, None, None, r.Home_Factor, r.Away_Factor))
            continue
        hs, as_ = s.get((2026, r.Home, r.Away, "Home"), np.nan), s.get((2026, r.Away, r.Home, "Away"), np.nan)
        rows.append((r.Home, r.Away, round(mm.iloc[0]["Home_NPxG_Adj"] / hs, 3) if hs else None,
                     round(mm.iloc[0]["Away_NPxG_Adj"] / as_, 3) if as_ else None, r.Home_Factor, r.Away_Factor))
    return pd.DataFrame(rows, columns=["Home", "Away", "manual_home", "manual_away", "auto_home", "auto_away"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write NPxG_Adj of red-card matches into match_results.csv")
    a = ap.parse_args()
    out = build()
    print(f"{len(out)} matches with a red card -> {OUT.name}; by season:", out.groupby("Season").size().to_dict())
    if a.apply:
        apply(out)
    else:
        print(check(out).to_string(index=False))
