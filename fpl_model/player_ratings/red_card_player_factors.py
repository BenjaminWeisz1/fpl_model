"""
Player-level red-card factors for the goal/assist loop (CHANGELOG §136).

For every player-match in a match with a red card (cached Understat files; match-stats names and Understat IDs from
player_match_positions.csv), the factor that turns his npxG / xAG into its 11 v 11 equivalent, with the team
factors in force (red_card_adjust.UP / DOWN / UP2 / DOWN2):
  npxG: sum(xG_i f_i) / sum(xG_i) over his non-penalty shots, f_i from the man-advantage state at the shot's minute;
  xAG:  the same over the shots he assisted (`player_assisted`, matched by name within his team's roster);
  with no such shots, the mean f over his on-pitch minutes (1 for a player off before the red card).
build_player_ratings.py multiplies npxG / xAG by them at load; the data files are unchanged.

From 2026 (CHANGELOG §197) the matches come from FotMob (fotmob_match_positions.csv, fotmob_shots.csv): npxG as
above over his FotMob shots; FotMob does not record who assisted a shot, so xAG always takes the minutes factor.
His minutes run from kick-off (or his entry) to his substitution, his own dismissal or the match end.

    python red_card_player_factors.py     # -> data/current/red_card_player_factors.csv
"""
import json

import pandas as pd

from fpl_model.team_ratings import red_card_adjust as rca  # noqa: E402

OUT = rca.DATA / "red_card_player_factors.csv"


def minutes_factor(reds, side, start, end, up, down, up2, down2):
    other = "a" if side == "h" else "h"
    fs = [rca.factor_for((lambda n: n[side] - n[other])(rca.counts_at(reds, t + 0.5)), up, down, up2, down2)
          for t in range(start, end)]
    return sum(fs) / len(fs) if fs else 1.0


def shot_factor(shots, reds, side, up, down, up2, down2):
    other = "a" if side == "h" else "h"
    num = den = 0.0
    for s in shots:
        n = rca.counts_at(reds, s["minute"] if isinstance(s["minute"], float) else int(s["minute"]))
        xg = float(s["xG"])
        num += xg * rca.factor_for(n[side] - n[other], up, down, up2, down2)
        den += xg
    return num / den if den > 0 else None


def fotmob_rows(up, down, up2, down2):
    pos = pd.read_csv(rca.FOTMOB_POSITIONS, encoding="utf-8-sig", low_memory=False)
    shots = pd.read_csv(rca.FOTMOB_SHOTS, encoding="utf-8-sig", low_memory=False)
    shots = shots[(shots["Situation"] != "Penalty") & ~shots["Own_Goal"].astype(bool)]
    reds_by = {mid: reds for _, _, _, mid, reds, _ in rca.fotmob_matches()}
    rows = []
    for mid, g in pos[pos["MatchID_FotMob"].isin(reds_by)].groupby("MatchID_FotMob"):
        reds = reds_by[mid]
        end_match = int(max([90.0] + list(shots.loc[shots["MatchID_FotMob"] == mid, "Minute"]) + [m for m, _, _ in reds]))
        for r in g[g["Player"].notna()].itertuples():
            side = "h" if r.Venue == "Home" else "a"
            start = 0 if r.Is_Starter else int(r.Sub_On_Minute)
            ends = [x for x in (r.Sub_Off_Minute, r.Red_Minute) if pd.notna(x)]
            end = int(min(ends)) if ends else end_match
            # FotMob gives a stoppage-time entry as minute 90: keep at least his entry minute, so he gets its state
            mf = minutes_factor(reds, side, start, max(end, start + 1), up, down, up2, down2)
            mine = shots[(shots["MatchID_FotMob"] == mid) & (shots["Player_ID_FotMob"] == r.Player_ID_FotMob)]
            fg = shot_factor([{"minute": m, "xG": x if pd.notna(x) else 0.0} for m, x in zip(mine["Minute"], mine["xG"])],
                             reds, side, up, down, up2, down2)
            rows.append({"Season": int(r.Season), "Player": r.Player, "Team": r.Team, "Opponent": r.Opponent,
                         "Venue": r.Venue, "MatchID_Understat": None, "MatchID_FotMob": int(mid),
                         "Factor_npxG": round(fg if fg is not None else mf, 4), "Factor_xAG": round(mf, 4),
                         "Minutes_Factor": round(mf, 4)})
    return rows


def build(up=rca.UP, down=rca.DOWN, up2=rca.UP2, down2=rca.DOWN2, out_path=OUT):
    # match-stats names with their Understat IDs (player_match_positions.csv), keyed to the Understat match ID
    team = pd.read_csv(rca.DATA / "understat_match_positions.csv", encoding="utf-8-sig",
                       usecols=["Season", "MatchID_Understat", "Team", "Opponent", "Venue"]).drop_duplicates()
    pmp = pd.read_csv(rca.DATA / "player_match_positions.csv", encoding="utf-8-sig",
                      usecols=["Season", "Team", "Opponent", "Venue", "Player", "Player_ID_Understat"])
    pos = pmp.merge(team, on=["Season", "Team", "Opponent", "Venue"]).groupby("MatchID_Understat")
    rows = []
    for f in sorted(rca.MATCHES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        reds = rca.dismissals(d)
        if not reds or int(f.stem) not in pos.groups:
            continue
        players = pos.get_group(int(f.stem))
        for side in "ha":
            roster = {v["id"]: v for v in d["rosters"][side].values()}
            by_pid = {str(v["player_id"]): v for v in roster.values()}
            shots = [s for s in d["shots"][side] if s["situation"] != "Penalty" and s["result"] != "OwnGoal"]
            venue = "Home" if side == "h" else "Away"
            for r in players[players["Venue"] == venue].itertuples():
                v = by_pid.get(str(int(r.Player_ID_Understat))) if pd.notna(r.Player_ID_Understat) else None
                if v is None:
                    continue
                end = rca.exit_minute(roster, v["id"])
                start = end - int(v["time"])
                mf = minutes_factor(reds, side, start, end, up, down, up2, down2)
                fg = shot_factor([s for s in shots if str(s["player_id"]) == str(v["player_id"])],
                                 reds, side, up, down, up2, down2)
                fa = shot_factor([s for s in shots if s.get("player_assisted") == v["player"]],
                                 reds, side, up, down, up2, down2)
                rows.append({"Season": r.Season, "Player": r.Player, "Team": r.Team, "Opponent": r.Opponent,
                             "Venue": venue, "MatchID_Understat": int(f.stem),
                             "Factor_npxG": round(fg if fg is not None else mf, 4),
                             "Factor_xAG": round(fa if fa is not None else mf, 4),
                             "Minutes_Factor": round(mf, 4)})
    rows = [r for r in rows if r["Season"] < rca.FOTMOB_FIRST_SEASON] + fotmob_rows(up, down, up2, down2)
    out = pd.DataFrame(rows).drop_duplicates(["Season", "Player", "Team", "Opponent", "Venue"])
    if out_path is not None:
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    o = build()
    print(f"{len(o)} player-matches in {o['MatchID_Understat'].nunique() + o['MatchID_FotMob'].nunique()} red-card matches -> {OUT.name}; "
          f"factor != 1: npxG {int((o['Factor_npxG'] != 1).sum())}, xAG {int((o['Factor_xAG'] != 1).sum())}")
