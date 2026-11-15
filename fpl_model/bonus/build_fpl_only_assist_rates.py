"""
Per-player FPL-only assist rates per 90 (CHANGELOG §92, §95), added to the conventional assist
rate by build_simulation_features.py.

For every player in players_current.csv: rebound-capable shots and penalties won over his last
38 Sofascore appearances (his identity is the canonical ID of his latest named appearance, as
build_pass_band_rates.py), shrunk toward the position prior, through the frozen model in
fpl_only_assists.py. Penalties won count for the share a teammate takes: the player's own share
is the mean of penalty_share.csv's GW columns (0 when he is not listed). Without Sofascore
history he gets the prior, so the model's position mean. When Sofascore is not usable every
player gets the position fallback.

Run locally (before build_simulation_features.py):
    python build_fpl_only_assist_rates.py
"""

import pandas as pd

from fpl_model.bonus import fpl_only_assists as foa
from fpl_model.minutes_penalties_cs.build_penalty_predictions import resolve_names  # noqa: E402

_DATA = foa._DATA
OUT_FILE = _DATA / "fpl_only_assist_rates.csv"
COLUMNS = ["Player", "Web_Name", "Team", "Pos", "Mode", "Sofascore_Appearances", "History_Minutes",
           "Rebound_Rate", "Pens_Rate", "Pen_Share", "FPLOnly_Ast90"]


def penalty_shares(players):
    """Mean penalty share over the forecast window per Web_Name, as build_penalty_predictions.py names them."""
    ps = pd.read_csv(_DATA / "penalty_share.csv")
    ps["Web_Name"] = resolve_names(ps, players, "Player")
    gw = [c for c in ps.columns if c.startswith("GW")]
    ps["share"] = ps[gw].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return ps.dropna(subset=["Web_Name"]).groupby("Web_Name")["share"].max().to_dict()


def history():
    so = pd.read_csv(_DATA / "sofascore_player_match_stats.csv", encoding="utf-8-sig", low_memory=False)
    so = so[so["Appearance_Status"].eq("played")].copy()
    so["Kickoff"] = pd.to_datetime(so["Kickoff"], utc=True)
    def num(c):
        return pd.to_numeric(so[f"Sofascore_{c}"], errors="coerce").fillna(0)
    so["mins"] = num("minutesPlayed")
    so["rebound_shots"] = ((num("onTargetScoringAttempt") - num("goals")).clip(lower=0)
                           + num("blockedScoringAttempt") + num("hitWoodwork"))
    so["pens_won"] = num("penaltyWon")
    so = so.sort_values("Kickoff")
    canonical = so.dropna(subset=["Local_Player"]).groupby("Local_Player")["Canonical_Player_ID"].last()
    last = so.groupby("Canonical_Player_ID").tail(foa.HISTORY)
    sums = last.groupby("Canonical_Player_ID").agg(apps=("mins", "size"), mins=("mins", "sum"),
                                                   rebound_shots=("rebound_shots", "sum"), pens_won=("pens_won", "sum"))
    return canonical, sums


def main():
    mode, reason = foa.mode()
    players = pd.read_csv(_DATA / "players_current.csv", encoding="utf-8-sig")
    share_of = penalty_shares(players) if mode == "model" else {}
    players = players[players["Pos"].isin(["D", "M", "F", "G"])]
    canonical, sums = history() if mode == "model" else (pd.Series(dtype=object), pd.DataFrame())
    rows = []
    for p in players.itertuples():
        cid = canonical.get(p.Player)
        h = sums.loc[cid] if cid is not None and cid in sums.index else None
        apps, mins = (int(h["apps"]), float(h["mins"])) if h is not None else (0, 0.0)
        reb = foa.shrunk_rate(h["rebound_shots"] if h is not None else 0.0, mins, p.Pos, "rebound_rate")
        pen = foa.shrunk_rate(h["pens_won"] if h is not None else 0.0, mins, p.Pos, "pens_rate")
        share = float(share_of.get(p.Web_Name, 0.0) or 0.0)
        value = foa.rate(p.Pos, reb, pen, share) if mode == "model" else foa.FALLBACK[p.Pos]
        rows.append({"Player": p.Player, "Web_Name": p.Web_Name, "Team": p.Team, "Pos": p.Pos, "Mode": mode,
                     "Sofascore_Appearances": apps, "History_Minutes": round(mins),
                     "Rebound_Rate": round(reb, 5), "Pens_Rate": round(pen, 5), "Pen_Share": round(share, 3),
                     "FPLOnly_Ast90": round(value, 6)})
    out = pd.DataFrame(rows, columns=COLUMNS)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"FPL-only assists: mode {mode} ({reason}); {len(out)} players -> {OUT_FILE.name}")
    print(out.groupby("Pos")["FPLOnly_Ast90"].describe().round(4).to_string())
    print(out.sort_values("FPLOnly_Ast90", ascending=False).head(10)[
        ["Web_Name", "Team", "Pos", "Sofascore_Appearances", "Rebound_Rate", "Pens_Rate", "Pen_Share", "FPLOnly_Ast90"]].to_string(index=False))


if __name__ == "__main__":
    main()
