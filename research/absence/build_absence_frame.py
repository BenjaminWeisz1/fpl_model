"""Key-player absence frame (AGENDA 2026-10-31 item 5, CHANGELOG §154). No production writes.

One row per team-match: the expected XI from trailing league minutes, who actually started, each absentee's reason from
Sofascore's own pre-match missing lists, and the quality gap between expected and actual XI on several pre-match measures.
Run: python -u -X utf8 -m research.absence.build_absence_frame
"""
import re
import unicodedata

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
RAW = paths.RAW / "transfermarkt"
OUT = paths.VALIDATION / "absence_20261031"
OUT.mkdir(exist_ok=True)
RATING_LOG = paths.VALIDATION / "prematch_bonus_history_20261030" / "player_ratings_prematch.csv"
WINDOW, MIN_WINDOW = 8, 4
SOFA_K, SOFA_HALF_LIFE = 7.0, 38.0  # about seven pseudo-games, one-year half-life, as the research doc summarises
SUSPENDED = {3, 11, 12, 13}
INJURED = {1, 2}
TM_TEAMS = {"AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Brentford FC": "Brentford", "Brighton & Hove Albion": "Brighton",
            "Burnley FC": "Burnley", "Chelsea FC": "Chelsea", "Everton FC": "Everton", "Fulham FC": "Fulham", "Huddersfield Town": "Huddersfield",
            "Liverpool FC": "Liverpool", "Manchester United": "Manchester Utd", "Newcastle United": "Newcastle Utd", "Nottingham Forest": "Nott'ham Forest",
            "Sheffield United": "Sheffield Utd", "Southampton FC": "Southampton", "Sunderland AFC": "Sunderland", "Tottenham Hotspur": "Tottenham",
            "Watford FC": "Watford", "West Bromwich Albion": "West Brom", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves"}


def norm(name):
    t = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", t).split()


def similar(a, b):
    a, b = set(norm(a)), set(norm(b))
    return len(a & b) / max(min(len(a), len(b)), 1)


def load_squads():
    s = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Event_ID_Sofascore", "Player_ID_Sofascore", "Player_Sofascore", "Local_Player", "Team", "Opponent", "Venue",
                             "Kickoff", "Is_Starter", "Position_Sofascore", "Sofascore_minutesPlayed"])
    s["Kickoff"] = pd.to_datetime(s.Kickoff, utc=True)
    s = s.rename(columns={"Player_ID_Sofascore": "PID", "Event_ID_Sofascore": "Event", "Sofascore_minutesPlayed": "Min", "Position_Sofascore": "Pos"})
    s["Min"] = s.Min.fillna(0)
    s["Is_Starter"] = s.Is_Starter.astype(str).eq("True")
    return s


def load_missing():
    m = pd.read_csv(paths.current_file("sofascore_missing_players.csv", root=DATA), encoding="utf-8-sig")
    m = m[m.Type == "missing"].rename(columns={"Player_ID_Sofascore": "PID", "Event_ID_Sofascore": "Event"})
    m["Why"] = np.where(m.Reason.isin(SUSPENDED), "suspended", np.where(m.Reason.isin(INJURED), "injured", "other"))
    m.loc[m.Description.fillna("").str.lower().str.contains("doping"), "Why"] = "other"
    return m.drop_duplicates(["Event", "PID"]).set_index(["Event", "PID"]).Why.to_dict()


def trailing_quality(s, ratings, value_map):
    """Each measure known before each kickoff, per (PID, kickoff), as asof lookups."""
    r = ratings.merge(s[["Season", "Local_Player", "Team", "Opponent", "Venue", "PID", "Kickoff"]].dropna(subset=["Local_Player"]),
                      left_on=["Season", "Player", "Team", "Opponent", "Venue"], right_on=["Season", "Local_Player", "Team", "Opponent", "Venue"], how="inner")
    r = r.sort_values("Kickoff")
    x = pd.read_csv(paths.current_file("sofascore_player_extra.csv.gz", root=DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=["Event_ID_Sofascore", "Player_ID_Sofascore", "rating", "minutesPlayed", "Position"])
    x = x.rename(columns={"Event_ID_Sofascore": "Event", "Player_ID_Sofascore": "PID"})
    x = x[x.rating.notna() & (x.minutesPlayed >= 30)].merge(s[["Event", "Kickoff"]].drop_duplicates(), on="Event").sort_values("Kickoff")
    # Mean within position of all prior ratings stands in for the shrinkage target; the first seasons run on few rows.
    d = 2.0 ** (-1.0 / SOFA_HALF_LIFE)
    state, pos_state, vals = {}, {}, []
    for pid, pos, rat in zip(x.PID, x.Position, x.rating):
        n, t = state.get(pid, (0.0, 0.0))
        pn, pt = pos_state.get(pos, (0.0, 0.0))
        target = pt / pn if pn > 0 else 6.8
        vals.append(((t + SOFA_K * target) / (n + SOFA_K)) - target)   # value before this match, relative to position mean
        state[pid] = (n * d + 1, t * d + rat)
        pos_state[pos] = (pn + 1, pt + rat)
    x["SofaQ"] = vals
    # Rows hold the value going INTO that match; asof with strict '<' takes a player's previous match-state, one step stale, as for ratings.
    return r[["PID", "Kickoff", "NPG_Rating_Pre", "Assist_Rating_Pre"]], x[["PID", "Kickoff", "SofaQ"]]


def attach_asof(frame, table, cols):
    t = table.dropna(subset=cols).sort_values("Kickoff")
    f = frame.sort_values("Kickoff")
    return pd.merge_asof(f, t, on="Kickoff", by="PID", allow_exact_matches=False)


def team_matches():
    h = pd.read_csv(paths.current_file("team_ratings_history.csv", root=DATA), encoding="utf-8-sig")
    a = pd.DataFrame({"Season": h.Season, "Team": h.Home, "Opponent": h.Away, "Venue": "Home", "NPxG": h.Home_NPxG_Adj, "NPxGA": h.Away_NPxG_Adj,
                      "NPG": h.Home_NPG, "NPGA": h.Away_NPG, "GD": h.Home_Goals - h.Away_Goals, "Pred": h.Home_Prediction, "PredA": h.Away_Prediction})
    b = pd.DataFrame({"Season": h.Season, "Team": h.Away, "Opponent": h.Home, "Venue": "Away", "NPxG": h.Away_NPxG_Adj, "NPxGA": h.Home_NPxG_Adj,
                      "NPG": h.Away_NPG, "NPGA": h.Home_NPG, "GD": h.Away_Goals - h.Home_Goals, "Pred": h.Away_Prediction, "PredA": h.Home_Prediction})
    return pd.concat([a, b], ignore_index=True)


def tm_values(s):
    """Value on 30 June of the season's start year per (Season, PID), joined by name within team-season through lineups."""
    games = pd.read_csv(RAW / "games.csv.gz", low_memory=False, usecols=["game_id", "competition_id", "season", "home_club_id", "home_club_name"])
    pl = games[games.competition_id == "GB1"]
    ids = set(pl.game_id)
    clubs = pl.drop_duplicates("home_club_id").set_index("home_club_id").home_club_name.map(lambda n: TM_TEAMS.get(n, n)).to_dict()
    seasons = pl.set_index("game_id").season.to_dict()
    lu = pd.read_csv(RAW / "game_lineups.csv.gz", low_memory=False, usecols=["game_id", "club_id", "player_id", "player_name"])
    lu = lu[lu.game_id.isin(ids)].assign(Season=lambda d: d.game_id.map(seasons), Team=lambda d: d.club_id.map(clubs))
    tm = lu.drop_duplicates(["Season", "Team", "player_id"])
    ours = s.drop_duplicates(["Season", "Team", "PID"])[["Season", "Team", "PID", "Player_Sofascore"]]
    rows = []
    for (season, team), g in ours.groupby(["Season", "Team"]):
        c = tm[(tm.Season == season) & (tm.Team == team)]
        names = list(zip(c.player_id, c.player_name))
        for pid, n in zip(g.PID, g.Player_Sofascore):
            best = max(names, key=lambda z: similar(n, z[1]), default=(None, ""))
            if best[0] is not None and similar(n, best[1]) >= .99:
                rows.append((season, team, pid, best[0]))
    cross = pd.DataFrame(rows, columns=["Season", "Team", "PID", "TM_ID"]).drop_duplicates(["Season", "Team", "PID"])
    v = pd.read_csv(RAW / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    v["date"] = pd.to_datetime(v.date)
    v = v.sort_values("date")
    out = []
    for season in sorted(cross.Season.unique()):
        g = v[v.date <= pd.Timestamp(int(season), 6, 30)].drop_duplicates("player_id", keep="last")
        out.append(cross[cross.Season == season].merge(g, left_on="TM_ID", right_on="player_id", how="left"))
    cross = pd.concat(out, ignore_index=True).rename(columns={"market_value_in_eur": "MV"})
    print("TM crosswalk: player-seasons", len(ours), "matched", len(cross), "with value", int(cross.MV.notna().sum()), flush=True)
    return cross[["Season", "Team", "PID", "MV"]]


def pair_gap(expected, actual, q):
    """Absentee-replacement pairs, same position group first, highest quality to highest quality. expected/actual: DataFrames."""
    absent = expected[~expected.PID.isin(actual.PID)]
    subs = actual[~actual.PID.isin(expected.PID)]
    pairs = []
    a_used, s_used = set(), set()
    for grp, ga in absent.groupby("Pos"):
        gs = subs[subs.Pos == grp].sort_values(q, ascending=False)
        for (ia, ra), (is_, rs) in zip(ga.sort_values(q, ascending=False).iterrows(), gs.iterrows()):
            pairs.append((ra, rs))
            a_used.add(ia)
            s_used.add(is_)
    rest_a = absent.drop(index=list(a_used)).sort_values(q, ascending=False)
    rest_s = subs.drop(index=list(s_used)).sort_values(q, ascending=False)
    pairs += [(ra, rs) for (_, ra), (_, rs) in zip(rest_a.iterrows(), rest_s.iterrows())]
    return pairs


def build():
    s = load_squads()
    why = load_missing()
    ratings = pd.read_csv(RATING_LOG, encoding="utf-8-sig", low_memory=False)
    mv = tm_values(s)
    rating_t, sofa_t = trailing_quality(s, ratings, None)

    order = s.groupby(["Season", "Team", "Opponent", "Venue", "Event"], as_index=False).Kickoff.first().sort_values("Kickoff")
    sched = {t: g.reset_index(drop=True) for t, g in order.groupby("Team")}
    by_event = {k: v for k, v in s.groupby(["Event", "Team"])}
    last_seen = s.groupby(["Team", "PID"]).Kickoff.max()   # used only to tell departures from absences
    pos_mode = s[s.Pos.notna()].groupby("PID").Pos.agg(lambda x: x.mode().iat[0]).to_dict()
    missing_team = pd.read_csv(paths.current_file("sofascore_missing_players.csv", root=DATA), encoding="utf-8-sig")
    missing_team = missing_team[missing_team.Type == "missing"].groupby(["Event_ID_Sofascore", "Side"]).Player_ID_Sofascore.apply(set).to_dict()

    rows, absences = [], []
    xi_rows = []
    for team, g in sched.items():
        for i in range(len(g)):
            event, season, opp, venue, ko = g.Event[i], g.Season[i], g.Opponent[i], g.Venue[i], g.Kickoff[i]
            past = g.iloc[max(0, i - WINDOW):i]
            if len(past) < MIN_WINDOW or (ko - past.Kickoff.iloc[-1]).days > 110 and len(past) < WINDOW:
                continue
            mins = pd.concat([by_event[(e, team)][["PID", "Min"]] for e in past.Event]).groupby("PID").Min.sum()
            mins = mins[[last_seen.get((team, p), ko) >= ko for p in mins.index]]  # still at the club
            exp = mins.sort_values(ascending=False).head(11)
            act = by_event[(event, team)]
            starters = act[act.Is_Starter]
            if len(exp) < 11 or len(starters) != 11:
                continue
            squad = set(act.PID)
            base = {"Event": event, "Season": season, "Team": team, "Opponent": opp, "Venue": venue, "Kickoff": ko, "Round_Team": i}
            for pid in exp.index:
                xi_rows.append({**base, "PID": pid, "Set": "E", "Share": exp[pid] / (90 * len(past)), "Pos": pos_mode.get(pid),
                                "Why": ("started" if pid in set(starters.PID) else why.get((event, pid), "rested" if pid in squad else "unlisted"))})
            for pid in starters.PID:
                xi_rows.append({**base, "PID": pid, "Set": "A", "Share": mins.get(pid, 0.0) / (90 * len(past)), "Pos": pos_mode.get(pid), "Why": "started"})
    xi = pd.DataFrame(xi_rows)
    print("XI rows", len(xi), xi[xi.Set == "E"].Why.value_counts().to_dict(), flush=True)
    xi = attach_asof(xi, rating_t, ["NPG_Rating_Pre"])
    xi = attach_asof(xi, sofa_t, ["SofaQ"])
    xi = xi.merge(mv, on=["Season", "Team", "PID"], how="left")
    xi["LogMV"] = np.log(xi.MV.clip(lower=1e5))
    for c in ["NPG_Rating_Pre", "Assist_Rating_Pre", "SofaQ", "LogMV"]:
        xi[c + "_filled"] = xi[c].fillna(xi.groupby(["Season", "Pos"])[c].transform("median")).fillna(xi[c].median())
    xi["Attack"] = xi.NPG_Rating_Pre_filled + xi.Assist_Rating_Pre_filled
    xi.to_pickle(OUT / "xi.pkl")
    return xi


if __name__ == "__main__":
    build()
