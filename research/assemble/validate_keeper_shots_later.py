"""CHANGELOG §163, after the first run: is the 2025-26 gain real or level, and what is goalsPrevented?

Not part of the adoption rule; a disclosed follow-up.
Run: python -u -X utf8 -m research.assemble.validate_keeper_shots_later
"""
import json

import numpy as np
import pandas as pd

from fpl_model import paths
from research.assemble import validate_keeper_saves as K
from research.assemble import validate_keeper_shots as S


def level_match(t, arms):
    """Each arm scaled to each season's actual saves, using hindsight equally, so only spread is scored."""
    out = t.copy()
    for a in arms:
        lam = out["r_" + a] * out.Exposure
        out["r_" + a] *= out.groupby("Season").Saves.transform("sum") / lam.groupby(out.Season).transform("sum")
        lam = (out["r_" + a] * out.Exposure).values
        out["pts_" + a], out["ll_" + a] = K.expected_points_per_three_saves(lam), K.poisson_ll(out.Saves.values, lam)
    return out


def goals_prevented_definition(e):
    """xGOT faced = the opponents' summed player xGOT, 2022, where that field is nearly complete."""
    a = pd.read_csv(paths.current_file("sofascore_player_extra.csv.gz", root=K.DATA), low_memory=False,
                    usecols=["Season", "Event_ID_Sofascore", "Side", "Position", "expectedGoalsOnTarget", "minutesPlayed"])
    a = a[(a.Season == 2022)]
    faced = a[a.Position != "G"].groupby(["Event_ID_Sofascore", "Side"]).expectedGoalsOnTarget.sum().rename("For").reset_index()
    faced["Side"] = faced.Side.map({"home": "away", "away": "home"})
    H = pd.read_csv(paths.current_file("team_ratings_history.csv", root=K.DATA), encoding="utf-8-sig", low_memory=False)
    x = pd.concat([H.assign(Team=H.Home, Against=H.Away_Goals, Opponent=H.Away, Venue="Home"), H.assign(Team=H.Away, Against=H.Home_Goals, Opponent=H.Home, Venue="Away")])[K.KEY + ["Against"]]
    g = e[(e.Season == 2022) & e.goalsPrevented.notna() & (e.minutesPlayed >= 90)].merge(x, on=K.KEY, validate="many_to_one")
    side = pd.read_csv(paths.current_file("sofascore_player_extra.csv.gz", root=K.DATA), low_memory=False, usecols=["Event_ID_Sofascore", "Player_ID_Sofascore", "Side"])
    g = g.merge(side, on=["Event_ID_Sofascore", "Player_ID_Sofascore"], validate="one_to_one").merge(faced, on=["Event_ID_Sofascore", "Side"], validate="one_to_one")
    d = g.goalsPrevented - (g.For - g.Against)
    keeper_field = g.expectedGoalsOnTarget
    return {"rows": len(g), "corr": float(g.goalsPrevented.corr(g.For - g.Against)), "mean_gap": float(d.mean()), "keeper_row_xgot_mean": float(keeper_field.mean()), "sum_gp_per_keeper_match": float(g.goalsPrevented.mean())}


def main():
    x = K.frame()
    spells = x[["MatchID", "Team", "Spell"]].drop_duplicates()
    x = x.merge(S.team_features(S.shots_table(), spells), on=K.KEY, how="left", validate="many_to_one")
    x = x[x.Season >= 2018].reset_index(drop=True)
    x["lO"] = np.log(x.O)
    for c in ("OppSoT", "OwnSoTA", "OppXGShot"):
        x["l" + c] = np.log(x[c])
    t, _ = S.run(x, K.TEST + K.LATER, ["ctrl", "s", "q"])
    res = {}
    for label, seasons in (("2019-2024", K.TEST), ("2025-26", K.LATER), ("2025", [2025]), ("2026", [2026])):
        g = t[t.Season.isin(seasons)].reset_index(drop=True)
        print(label, "raw forecast saves per row by arm", {a: round(float((g["r_" + a] * g.Exposure).mean()), 3) for a in ("ctrl", "s", "q")}, "actual", round(float(g.Saves.mean()), 3))
        m = level_match(g, ["ctrl", "s", "q"])
        res[label] = {a: K.paired(m, "ctrl", a) for a in ("s", "q")}
        res[label]["q_v_s"] = K.paired(m, "s", "q")
        for a, p in res[label].items():
            print(" ", a, f"{p['mse_pct']:+.2f}% ({p['mse_lo']:+.2f}, {p['mse_hi']:+.2f}) LL {p['ll_gain']:+.4f} ({p['ll_lo']:+.4f}, {p['ll_hi']:+.4f})")
    res["goals_prevented"] = goals_prevented_definition(S.keeper_table())
    print(res["goals_prevented"])
    (S.OUT / "results_later.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
