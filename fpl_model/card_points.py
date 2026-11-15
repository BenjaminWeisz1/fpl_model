"""Yellow and red card forecasts (CHANGELOG §175).

A player's yellow rate is their position's recent rate x a shrunk personal multiplier. A new player has
multiplier 1, so the position rate. Reds are pooled by position. FPL scoring: a red is -3 including any
yellow in that match, otherwise a yellow is -1. Settings are frozen by validate_card_points.py.

Since §195: starts and cameos get separate group rates, mixed by the player's recent cameo share of minutes
(§193), and a player one yellow short of a ban before its cut-off gets
x SUSPENSION_THETA on the yellow rate (§191). Reds are unchanged.
"""

import numpy as np
import pandas as pd

from fpl_model import config, paths

DATA = paths.CURRENT
GROUP_OF = {"GK": "GK", "CB": "D", "LB": "D", "RB": "D", "WB": "D", "DM": "M", "CM": "M", "AM": "M", "LM": "M",
            "RM": "M", "LW": "M", "RW": "M", "FW": "F"}
FPL_GROUP = {"G": "GK", "D": "D", "M": "M", "F": "F"}
KEYS = ["Season", "Team", "Opponent", "Venue", "Player"]


def build_frame():
    """Played rows in date order, with events, group and fouls."""
    d = pd.read_csv(DATA / "player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=KEYS + ["Minutes", "CrdY", "CrdR", "Fls"])
    d = d[d.Minutes > 0].copy()
    d["red"] = (d.CrdR.fillna(0) > 0).astype(float)
    d["yellow"] = ((d.CrdY.fillna(0) > 0) & (d.red == 0)).astype(float)
    d["pts"] = -(d.yellow + 3 * d.red)
    d["n90"] = d.Minutes / 90

    # match dates: Understat to 2025, FotMob from 2026 (CHANGELOG §197); without one a row would be dropped below
    u = pd.read_csv(DATA / "understat_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Team", "Opponent", "Venue", "Date"])
    fm = pd.read_csv(DATA / "fotmob_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                     usecols=["Season", "Team", "Opponent", "Venue", "Date"])
    u = pd.concat([u[u["Season"] < fm["Season"].min()], fm]).drop_duplicates(["Season", "Team", "Opponent", "Venue"])
    n = len(d)
    d = d.merge(u, on=["Season", "Team", "Opponent", "Venue"], how="left")
    assert len(d) == n
    d["Date"] = pd.to_datetime(d["Date"])
    missing = d.Date.isna()
    if missing.any():
        print(f"card_points: {int(missing.sum())} rows without a date dropped")
        d = d[~missing]

    p = pd.read_csv(DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=KEYS + ["Position_Resolved"]).drop_duplicates(KEYS)
    d = d.merge(p, on=KEYS, how="left")
    pc = pd.read_csv(DATA / "players_current.csv", encoding="utf-8-sig", usecols=["Player", "Pos"]).drop_duplicates("Player")
    d = d.merge(pc.rename(columns={"Pos": "_pos"}), on="Player", how="left")
    d["group"] = d.Position_Resolved.map(GROUP_OF).fillna(d._pos.map(FPL_GROUP))
    d["group"] = d.groupby("Player").group.transform(lambda s: s.ffill().bfill()).fillna("M")

    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Local_Player", "Team", "Opponent", "Venue", "Sofascore_fouls"])
    s = s.rename(columns={"Local_Player": "Player"}).dropna(subset=["Sofascore_fouls"]).drop_duplicates(KEYS)
    d = d.merge(s, on=KEYS, how="left")
    d["fouls"] = d.Sofascore_fouls.fillna(d.Fls)
    return d.sort_values(["Date", "Team", "Player"]).reset_index(drop=True)


def trailing_level(d, value, by=("group",)):
    """Rate per 90 by group: two previous seasons plus this season before the row's date."""
    by = list(by)
    mask = d[value].notna()
    day = d[mask].groupby(by + ["Season", "Date"])[[value, "n90"]].sum().reset_index().sort_values("Date")
    day["c_v"] = day.groupby(by + ["Season"])[value].cumsum() - day[value]
    day["c_n"] = day.groupby(by + ["Season"])["n90"].cumsum() - day["n90"]
    tot = d[mask].groupby(by + ["Season"])[[value, "n90"]].sum()
    v_map, n_map = tot[value].to_dict(), tot["n90"].to_dict()

    def key(row, s):
        return tuple(row[b] for b in by) + (s,) if len(by) else (s,)

    rows = day.to_dict("records")
    for r in rows:
        ks = [key(r, r["Season"] - i) for i in (1, 2)]
        r["_lvl"] = (r["c_v"] + sum(v_map.get(k, 0.0) for k in ks)) / max(r["c_n"] + sum(n_map.get(k, 0.0) for k in ks), 1e-9)
    day = pd.DataFrame(rows)[by + ["Season", "Date", "_lvl"]]
    return d[by + ["Season", "Date"]].merge(day, on=by + ["Season", "Date"], how="left")["_lvl"].values


# Pseudo-count in expected yellows: lowest card-point MSE on 2018-2023 of k = 2, 5, 10, 20, 40, 80 (§175).
# Reproduce with `python -m research.general.validate_card_points tune`; see tuned.json in data/validation/card_points_20261101/.
K = 5
RED_POINTS, YELLOW_POINTS = -3, -1      # FPL's scoring rules


def expected_card_points(yellows_90, reds_90, expected_minutes):
    """Expected FPL card points; a red includes any yellow in that match, so the rates don't overlap."""
    return (YELLOW_POINTS * np.asarray(yellows_90, dtype=float) + RED_POINTS * np.asarray(reds_90, dtype=float)) \
        * np.asarray(expected_minutes, dtype=float) / 90


# Starts and cameos at separate rates (§193, §195).
# The mix of a fixture's minutes. "forecast" (default): cameo minutes = (P1 - P60) x the mean cameo length,
# from that fixture's minutes forecast, so a returning starter isn't charged as a substitute. "trailing": the
# tested §193 arm F stand-in, the cameo share of the last five club matches since the player's first
# appearance for the club this season (at least three). CARD_ROLE_MIX=trailing selects it.
ROLE_MIX = config.get("CARD_ROLE_MIX")
MIX_WINDOW, MIX_MIN = 5, 3

# Suspension threat (§191, §195).
# Actual / forecast yellows on brink rows, 2018-2023 (§191). Reproduce with
# `python -m research.general.validate_card_suspension`: results.json "theta" in data/validation/card_suspension_20261102/.
SUSPENSION_THETA = 0.851
# Premier League bans: (season yellows before the match, last club match the threat applies to). 5 by match 19,
# 10 by match 32, 15 at any point.
BRINK = ((4, 19), (9, 32), (14, 10_000))


def with_roles(d):
    """Start or cameo per row: Sofascore Is_Starter, else 46+ minutes (§193)."""
    s = pd.read_csv(DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False,
                    usecols=["Season", "Local_Player", "Team", "Opponent", "Venue", "Is_Starter"]).rename(columns={"Local_Player": "Player"})
    s = s.dropna(subset=["Player"]).drop_duplicates(KEYS)
    n = len(d)
    d = d.merge(s, on=KEYS, how="left")
    assert len(d) == n
    d["start"] = d.Is_Starter.astype("boolean").fillna(d.Minutes >= 46).astype(bool)
    d["role"] = np.where(d.start, "S", "C")
    return d


def cameo_shares(d):
    """Cameo share of each player's minutes over their club's last MIX_WINDOW matches this season (§193 arm F)."""
    last = int(d.Season.max())
    cur = d[d.Season == last]
    team_dates = cur[["Team", "Date"]].drop_duplicates().sort_values("Date")
    latest_team = cur.sort_values("Date").groupby("Player").Team.last()
    out = {}
    for player, team in latest_team.items():
        rows = cur[(cur.Player == player) & (cur.Team == team)]
        dates = team_dates[(team_dates.Team == team) & (team_dates.Date >= rows.Date.min())].Date.values[-MIX_WINDOW:]
        if len(dates) < MIX_MIN:
            continue
        w = rows[rows.Date.isin(dates)]
        total = float(w.Minutes.sum())
        if total > 0:
            out[player] = float(w.Minutes[~w.start].sum()) / total
    return out


def club_matches_played(d):
    """League matches each club has played this season, keyed on the stats files' team names."""
    last = int(d.Season.max())
    return d[d.Season == last].groupby("Team").Date.nunique().to_dict()


def current_rates(players_current):
    """Yellows and reds per 90 for the next match, per row of players_current. New players get their group rate.

    Yellows_90 is the one-rate forecast (§175). With the role split, a player with a start / cameo mix gets
    Yellows_90_Start and Yellows_90_Cameo (and Cameo_Share with the trailing mix; attrs["cameo_minutes"] for
    the forecast mix); fixture_yellows_90 combines them. Season_Yellows
    is this season's league yellows so far (a red-card match counts 0), for the suspension threat.
    """
    d = build_frame()
    last = int(d.Season.max())
    recent = d[d.Season >= last - 2]
    tot = recent.groupby("group")[["yellow", "red", "n90"]].sum()
    level_y, level_r = (tot.yellow / tot.n90).to_dict(), (tot.red / tot.n90).to_dict()
    d["lvl_y"] = trailing_level(d, "yellow")
    d["e_y"] = d.lvl_y * d.n90
    _, multiplier = decayed_multiplier(d, "yellow", "e_y", K, final=True)
    latest_group = d.groupby("Player").group.last().to_dict()
    groups = [latest_group.get(p, FPL_GROUP.get(q, "M")) for p, q in zip(players_current["Player"], players_current["Pos"])]
    out = players_current[["Player"]].copy()
    out["Yellows_90"] = [level_y[g] * multiplier.get(p, 1.0) for g, p in zip(groups, players_current["Player"])]
    out["Reds_90"] = [level_r[g] for g in groups]
    season_y = d[d.Season == last].groupby("Player").yellow.sum().to_dict()
    out["Season_Yellows"] = [int(season_y.get(p, 0)) for p in players_current["Player"]]
    out["Yellows_90_Start"] = out["Yellows_90_Cameo"] = out["Cameo_Share"] = np.nan
    d = with_roles(d)
    rt = d[d.Season >= last - 2].groupby(["group", "role"])[["yellow", "n90"]].sum()
    level_role = (rt.yellow / rt.n90).to_dict()
    d["e_r"] = trailing_level(d, "yellow", by=("group", "role")) * d.n90
    _, mult_role = decayed_multiplier(d, "yellow", "e_r", K, final=True)
    m = [mult_role.get(p, 1.0) for p in players_current["Player"]]
    out["Yellows_90_Start"] = [level_role[(g, "S")] * x for g, x in zip(groups, m)]
    out["Yellows_90_Cameo"] = [level_role[(g, "C")] * x for g, x in zip(groups, m)]
    if ROLE_MIX == "trailing":
        shares = cameo_shares(d)
        out["Cameo_Share"] = [shares.get(p, np.nan) for p in players_current["Player"]]
    cameos = d[(d.Season >= last - 2) & ~d.start]
    out.attrs["cameo_minutes"] = float(cameos.Minutes.mean())
    out.attrs["club_played"] = club_matches_played(d)
    return out


def forecast_cameo_share(p1, p60, x_mins_raw, cameo_minutes):
    """Cameo share of a fixture's expected minutes: (P1 - P60) x mean cameo length / raw xMins, in [0, 1].

    P1 and P60 come from the raw minutes lookup, so the raw xMins is the matching total (§177)."""
    x = np.asarray(x_mins_raw, dtype=float)
    c = (np.asarray(p1, dtype=float) - np.asarray(p60, dtype=float)).clip(0) * cameo_minutes
    return np.where(x > 0, np.clip(c / np.where(x > 0, x, 1), 0, 1), np.nan)


def fixture_yellows_90(rates, club_match, cameo_share=None):
    """Yellows per 90 for one fixture per row of `rates` (from current_rates), given the club's match number.

    Role split: (1 - share) x start rate + share x cameo rate where a mix is known, else the one rate (§193).
    `cameo_share` is per row (forecast mix); without it, rates' trailing Cameo_Share is used.
    Suspension: x SUSPENSION_THETA while season yellows sit one short of a ban before that ban's cut-off (§191).
    """
    y = rates["Yellows_90"].to_numpy(dtype=float)
    share = rates["Cameo_Share"].to_numpy(dtype=float) if cameo_share is None else np.asarray(cameo_share, dtype=float)
    split = (1 - share) * rates["Yellows_90_Start"].to_numpy(dtype=float) + share * rates["Yellows_90_Cameo"].to_numpy(dtype=float)
    y = np.where(np.isfinite(split), split, y)
    n = rates["Season_Yellows"].to_numpy(dtype=float)
    c = np.asarray(club_match, dtype=float)
    brink = np.zeros(len(y), dtype=bool)
    for count, cut in BRINK:
        brink |= (n == count) & (c <= cut)
    y = np.where(brink, SUSPENSION_THETA * y, y)
    return y


def decayed_multiplier(d, event, expected, k, half_life=None, final=False):
    """(decayed events + k) / (decayed expected + k) before each row, per player, in date order."""
    keep = 1.0 if half_life is None else 0.5 ** (1 / half_life)
    ev, ex = d[event].values, d[expected].values
    out = np.ones(len(d))
    state = {}
    for player, idx in d.groupby("Player", sort=False).indices.items():
        a = e = 0.0
        for i in idx:
            out[i] = (a + k) / (e + k)
            if not np.isnan(ev[i]):
                a = a * keep + ev[i]
                e = e * keep + ex[i]
        state[player] = (a + k) / (e + k)
    return (out, state) if final else out
