"""EDA charts from the Sofascore set-piece tables (`fpl_model/fetch/sofascore_set_pieces.py`), in the weekly
heatmaps' look, both for the current season:

  images/set_pieces/set_piece_xa_top30.png       the 30 players with the most set-piece xA per 90
  images/set_pieces/set_piece_xg_conceded.png    every team, fewest set-piece xG conceded per match first

Set pieces here never include penalties.

Sofascore names the taker only for set pieces that ended in a shot, so every number here counts shots:
  SP xA/90     xG of the shots a player set up from corners and indirect free kicks (Sofascore's
               "set-piece" situation), per 90 Sofascore minutes. Direct free kicks have no assister.
  SP leading to shots  corners and free kicks he took that ended in a shot: corner and indirect free-kick shots he
               set up, plus his direct free-kick shots (Sofascore names no taker otherwise)
  Corner share the player's share of his team's corner shots that had a named assister, counting only
               shots while he was on the pitch. Ind. FK share is the same for indirect free kicks.
  Direct FK share  his share of his team's direct free-kick shots while he was on the pitch.
A share is 0% when his team had no shot of that kind while he was on the pitch.
Shown: players FPL Review predicts over MIN_NEXT_XMINS minutes in the next NEXT_GAMEWEEKS gameweeks
(minutes.csv) who have taken at least MIN_TAKEN set pieces that led to shots; departed players are dropped.

Team chart, per match (teams can have played different numbers of matches); headers break over two lines:
  Set pieces conceded/match       corners plus free kicks the opponent won in the team's half or on the
                                  wings (free kicks won in the opponent's own half are left out)
  Set-piece shots conceded/match  opponents' shots from corners, indirect and direct free kicks
  Set-piece xG conceded/match     those shots' xG (the sort column)
Throw-in set pieces are left out of both counts, since commentary records no throw-ins.

The charts are exploratory and feed no model. They skip (exit 0) when the tables are missing.

Run locally:
    python -m fpl_model.eda.set_piece_takers
"""
import pandas as pd

from fpl_model import export_weekly_heatmaps as hm
from fpl_model import league_rates, paths

NEXT_GAMEWEEKS = 5
MIN_NEXT_XMINS = 100
MIN_TAKEN = 5
OUT = "set_pieces/set_piece_xa_top30.png"
OUT_TEAMS = "set_pieces/set_piece_xg_conceded.png"
CONCEDED_SHOTS = ("corner", "set-piece", "free-kick")
DANGEROUS_FREE_KICK_ZONES = ("attacking half", "left wing", "right wing")
TEAM_COLUMNS = ["Set pieces\nconceded/match", "Set-piece shots\nconceded/match", "Set-piece xG\nconceded/match"]
CONCEDED, SHOTS, XGC = TEAM_COLUMNS
XA_SITUATIONS = ("corner", "set-piece")
SHARES = {"Corner\nshare": "corner", "Ind. FK\nshare": "set-piece", "Direct FK\nshare": "free-kick"}
TAKEN = "SP leading\nto shots"  # every data column takes the widest header line's width
CORNER, IND_FK, DIRECT_FK = SHARES
TAKEN_SITUATIONS = ("corner", "set-piece", "free-kick")


def shot_minute(df):
    return df["Minute"] + df["Added_Time"].fillna(0)


def table(events, pitch, players, season):
    """One row per player: minutes, SP xA/90, set pieces taken that ended in a shot, and the three shares."""
    ev = events[events["Season"] == season]
    on = pitch[pitch["Season"] == season].copy()
    # Per-90 rates use Sofascore's minutes played; the on/off clock (which runs into added time) only times shots.
    on["Minutes"] = on["Minutes_Played"].fillna((on["Off"] - on["On"]).clip(lower=0))
    shots = ev[ev["Event_Type"] == "shot"].copy()
    shots["T"] = shot_minute(shots)
    shots = shots[shots["Taker_ID_Sofascore"].notna()]
    shots["Taker_ID_Sofascore"] = shots["Taker_ID_Sofascore"].astype(int)

    out = on.groupby("Player_ID_Sofascore").agg(Minutes=("Minutes", "sum"), Local_Player=("Local_Player", "last"))
    xa = shots[shots["Situation"].isin(XA_SITUATIONS)].groupby("Taker_ID_Sofascore")["xG"].sum()
    out["SP xA/90"] = xa.reindex(out.index).fillna(0) * 90 / out["Minutes"].where(out["Minutes"] > 0)
    taken = shots[shots["Situation"].isin(TAKEN_SITUATIONS)].groupby("Taker_ID_Sofascore").size()
    out[TAKEN] = taken.reindex(out.index).fillna(0)

    # Team shots of each kind while the player was on the pitch (shots and spans share the commentary clock).
    joined = on.merge(shots[["Event_ID_Sofascore", "Team", "Situation", "T", "Taker_ID_Sofascore"]],
                      on=["Event_ID_Sofascore", "Team"])
    joined = joined[(joined["T"] >= joined["On"]) & (joined["T"] <= joined["Off"])]
    joined["Mine"] = joined["Taker_ID_Sofascore"] == joined["Player_ID_Sofascore"]
    for col, situation in SHARES.items():
        part = joined[joined["Situation"] == situation].groupby("Player_ID_Sofascore")["Mine"].agg(["sum", "count"])
        out[col] = (part["sum"] / part["count"]).reindex(out.index).fillna(0.0)  # 0%: no such team shots while on

    out = out.dropna(subset=["Local_Player"])
    pc = players[players["Status"] != "u"][["Player", "Web_Name", "Pos", "Team_Short", "Price"]]
    out = out.merge(pc, left_on="Local_Player", right_on="Player", how="inner")
    return out.drop(columns=["Player"]).rename(columns={"Web_Name": "Player", "Team_Short": "Team"})


def next_xmins(minutes, players):
    """(Web_Name, Team_Short) -> FPL Review xMins over the next NEXT_GAMEWEEKS gameweeks in minutes.csv."""
    cols = sorted((c for c in minutes.columns if c.endswith("_xMins")), key=lambda c: int(c.split("_")[0]))
    total = minutes[cols[:NEXT_GAMEWEEKS]].sum(axis=1)
    web = {(r.Raw_Web_Name, r.Team_Short): r.Web_Name for r in players.itertuples()}  # minutes.csv names are raw
    return {(web.get((n, t), n), t): x for n, t, x in zip(minutes["Name"], minutes["Team"], total, strict=True)}


def eligible(df, xmins):
    """Players predicted over MIN_NEXT_XMINS minutes who have taken at least MIN_TAKEN set pieces that led to shots."""
    predicted = df.apply(lambda r: xmins.get((r["Player"], r["Team"]), 0.0), axis=1)
    return df[(predicted > MIN_NEXT_XMINS) & (df[TAKEN] >= MIN_TAKEN)]


def team_table(events, season):
    """One row per team: set pieces, set-piece shots and their xG conceded, per match."""
    ev = events[events["Season"] == season]
    matches = pd.concat([ev[["Event_ID_Sofascore", "Team"]], ev[["Event_ID_Sofascore", "Opponent"]]
                         .rename(columns={"Opponent": "Team"})]).drop_duplicates().groupby("Team").size()
    won = ev[(ev["Event_Type"] == "corner_won")
             | ((ev["Event_Type"] == "free_kick_won") & ev["Situation"].isin(DANGEROUS_FREE_KICK_ZONES))]
    shots = ev[(ev["Event_Type"] == "shot") & ev["Situation"].isin(CONCEDED_SHOTS)]
    out = pd.DataFrame({"Matches": matches})
    out[CONCEDED] = won.groupby("Opponent").size().reindex(out.index).fillna(0) / out["Matches"]
    out[SHOTS] = shots.groupby("Opponent").size().reindex(out.index).fillna(0) / out["Matches"]
    out[XGC] = shots.groupby("Opponent")["xG"].sum().reindex(out.index).fillna(0) / out["Matches"]
    out["_Order"] = -out[XGC]  # fewest conceded first
    return out.reset_index().rename(columns={"index": "Team"})


def main():
    try:
        events = pd.read_csv(paths.current_file("sofascore_set_piece_events.csv"), encoding="utf-8-sig")
        pitch = pd.read_csv(paths.current_file("sofascore_player_on_pitch.csv"), encoding="utf-8-sig")
    except FileNotFoundError as exc:
        print(f"  [SKIP] set-piece chart: {exc.filename} missing (collect Sofascore commentary and shot maps)")
        return 0
    players = pd.read_csv(paths.current_file("players_current.csv"), encoding="utf-8-sig")
    season = league_rates.current_season()
    minutes = pd.read_csv(paths.current_file("minutes.csv"), encoding="utf-8-sig")
    df = eligible(table(events, pitch, players, season), next_xmins(minutes, players))
    hm._render_heatmap(
        df, sort_col="SP xA/90", display_cols=["SP xA/90", TAKEN, *SHARES],
        meta_cols=["Player", "Pos", "Team", "Price"],
        pct_cols=tuple(SHARES), int_cols=(TAKEN,),
        title=f"Set-Piece Creators {season}/{str(season + 1)[-2:]}",
        subtitle=(f"Players predicted to play over {MIN_NEXT_XMINS} minutes in the next {NEXT_GAMEWEEKS} gameweeks "
                  f"who have taken at least {MIN_TAKEN} set pieces leading to shots. Shares are of the team's "
                  "set-piece shots while the player was on the pitch."),
        filename=OUT, precision=2,
    )
    teams = team_table(events, season)
    hm._render_heatmap(
        teams, sort_col="_Order", display_cols=TEAM_COLUMNS, meta_cols=["Team"],
        invert_cols=tuple(TEAM_COLUMNS),  # fewer conceded is better, so green
        title=f"Set-Piece Defence {season}/{str(season + 1)[-2:]}",
        subtitle="Set pieces are corners and free kicks.",
        filename=OUT_TEAMS, precision=2, fig_height_per_row=0.48,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
