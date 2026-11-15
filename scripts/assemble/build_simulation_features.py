"""
Assembles one consolidated per-player-per-gameweek table, combining every
facet built so far: minutes probabilities, goal/assist ratings (scaled per
fixture), defcon threshold probabilities (minutes-scaled), penalty
predictions, and clean sheet probabilities -- ready for the bonus
simulation to consume directly.

Run locally (after all the individual facet scripts):
    python build_simulation_features.py
"""
import os
import sys

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "team_ratings"))
from league_rates import RECORDED_TEAM_MINUTES, check_minutes_round, minute_bands, penalty_rates, recorded_clock_scale  # noqa: E402
from update_team_ratings import home_advantage  # noqa: E402

PEN_ATTEMPTS_PER_NPG, PEN_CONVERSION = penalty_rates()

# Opponent-multiplier dampening, shared with build_player_ratings.py (§123; 1 = undampened).
OPP_BETA_NPG = float(os.environ.get("GOAL_MODEL_OPP_BETA_NPG", "0.7"))
OPP_BETA_AST = float(os.environ.get("GOAL_MODEL_OPP_BETA_AST", "0.8"))

MINUTES_PROBS_FILE = _DATA / "minutes_probabilities.csv"
GOAL_ASSIST_FILE = _DATA / "player_goal_assist_ratings.csv"
DEFCON_FILE = _DATA / "predictions_defensive.csv"
PENALTY_FILE = _DATA / "penalty_predictions.csv"
FIXTURE_GOALS_FILE = _DATA / "fixture_goal_predictions.csv"
RESIDUAL_BPS_FILE = _DATA / "residual_bps_baseline.csv"
RESIDUAL_DEFAULTS_FILE = _DATA / "residual_bps_position_defaults.csv"
PLAYERS_CURRENT_FILE = _DATA / "players_current.csv"
TEAM_RATINGS_FILE = _DATA / "team_ratings_current.csv"  # latest ratings, as predict_fixture_goals.py (§180)
PASS_BAND_FILE = _DATA / "pass_band_rates.csv"
FPL_ONLY_ASSISTS_FILE = _DATA / "fpl_only_assist_rates.csv"

OUT_FILE = _DATA / "simulation_features.csv"

from forecast_horizon import GAMEWEEKS  # noqa: E402  (the forecast window, set in one place)
THRESHOLD = {"D": 10, "M": 12, "F": 12, "G": 0}
# Highest 90-minute defcon probability passed to invert_mu; observed legitimate values reach ~0.91
DEFCON_P90_MAX = 0.98
# Substitute uplift on expected short appearances (CHANGELOG §100–§102). The goal/assist loop learns
# starter-equivalent per-90 rates; cameos run hotter (fresh legs, open games), so expected 1-59 minute
# appearances, (P1 - P60) x SHORT_MINUTES (simulate_bonus.AVG_MINUTES_SHORT), get the minutes-weighted
# uplift of 2017-2024 short appearances (67% of their minutes from substitutes). GOAL_SUB_UPLIFT=0 turns it off.
SUB_SHORT_UPLIFT = {"npxG": 1.2003, "xAG": 1.1964}
SHORT_MINUTES = minute_bands()[1]
SUB_UPLIFT_ON = os.environ.get("GOAL_SUB_UPLIFT", "1") == "1"
# Every per-90 rate here is per recorded minute, and recorded minutes stop at 90 (CHANGELOG §177). FPL Review's
# xMins run on a longer clock, so xMins is rescaled to the recorded one and xMins_Raw keeps FPL Review's.
# P1 and P60 come from the raw lookup. XMINS_RECORDED=0 reverts.


def invert_mu(prob, k, lo=1e-6, hi=60):
    """Same NB inversion used throughout this project."""
    if prob <= 1e-9:
        return 0.0
    if prob >= 1 - 1e-9:
        return hi
    def nb_sf(mu):
        r = max(3 * mu, 1e-9)
        return 1 - stats.nbinom.cdf(k, r, 0.75)
    f = lambda mu: nb_sf(mu) - prob
    from scipy import optimize
    try:
        return optimize.brentq(f, lo, hi)
    except ValueError:
        return lo if f(lo) < 0 else hi


def main():
    players_current = pd.read_csv(PLAYERS_CURRENT_FILE, encoding="utf-8-sig")
    team_ratings = pd.read_csv(TEAM_RATINGS_FILE, encoding="utf-8-sig")
    defense_rating_lookup = dict(zip(team_ratings["Team"], team_ratings["Defense_Rating"]))
    # Neutral-opponent divisor for goal/assist scaling: was hardcoded to 1.5,
    # which no longer matched the actual average Defense_Rating -- that
    # mismatch was silently suppressing
    # every player's predicted goals/assists by ~9%, even against a perfectly
    # average defense. Computed dynamically here so it self-corrects whenever
    # the underlying ratings shift, rather than needing manual recalibration
    # each season.
    avg_defense_rating = team_ratings["Defense_Rating"].mean()
    print(f"Neutral-opponent divisor (mean Defense_Rating): {avg_defense_rating:.4f}")
    minutes_probs = pd.read_csv(MINUTES_PROBS_FILE, encoding="utf-8-sig")
    goal_assist = pd.read_csv(GOAL_ASSIST_FILE, encoding="utf-8-sig")
    defcon = pd.read_csv(DEFCON_FILE)
    penalties = pd.read_csv(PENALTY_FILE, encoding="utf-8-sig")
    fixture_goals = pd.read_csv(FIXTURE_GOALS_FILE, encoding="utf-8-sig")
    residual = pd.read_csv(RESIDUAL_BPS_FILE, encoding="utf-8-sig")
    residual_defaults = pd.read_csv(RESIDUAL_DEFAULTS_FILE, encoding="utf-8-sig")

    residual_default_lookup = residual_defaults.set_index("Pos_")[["pos_mean", "pos_std"]].to_dict("index")

    base = players_current[["Player", "Web_Name", "Pos", "Team", "Price"]].copy()
    base = base[base["Pos"].isin(["D", "M", "F", "G"])].copy()

    # residual bps: player-specific if available, else position default
    base = base.merge(residual[["Player", "ResidualBPS_Mean", "ResidualBPS_Std"]], on="Player", how="left")
    for pos, defaults in residual_default_lookup.items():
        mask = base["Pos"] == pos
        base.loc[mask, "ResidualBPS_Mean"] = base.loc[mask, "ResidualBPS_Mean"].fillna(defaults["pos_mean"])
        base.loc[mask, "ResidualBPS_Std"] = base.loc[mask, "ResidualBPS_Std"].fillna(defaults["pos_std"])

    # goal/assist ratings (season-long per-90 rates, to be fixture-scaled below)
    # NOTE: player_goal_assist_ratings.csv's "Player" column actually holds
    # Web_Name values (it was overwritten with out_df['Player']=out_df['Web_Name']
    # when that file was built) -- key on Web_Name here, not the full Player name,
    # or this silently zeroes out goals/assists for nearly everyone.
    ga_lookup = goal_assist.set_index("Player")[["NPG/90", "Assists/90"]].to_dict("index")
    # A player with no rating gets zero goals and assists below, with no error (§187).
    xmin_cols = [c for c in minutes_probs.columns if c.startswith("xMins_GW")]
    expected = minutes_probs.groupby("Player")[xmin_cols].sum().sum(axis=1)
    unrated = base[base["Web_Name"].map(expected).gt(0) & ~base["Web_Name"].isin(ga_lookup)]
    if len(unrated):
        print(f"WARNING: {len(unrated)} active players have xMins but no goal/assist rating, so they forecast 0 "
              f"goals and assists; add a transfers.csv row: {', '.join(unrated['Web_Name'] + ' (' + unrated['Team'] + ')')}")

    # FPL-only assists per 90 (rebounds, deflections, penalties won, forced own goals; CHANGELOG
    # §92), keyed on the full Player name. Added unscaled to the fixture-scaled conventional rate,
    # as tested in §91. All zeros when FPL_ONLY_ASSISTS=0 (build_fpl_only_assist_rates.py).
    if not FPL_ONLY_ASSISTS_FILE.exists():
        raise FileNotFoundError(f"{FPL_ONLY_ASSISTS_FILE.name} missing; run build_fpl_only_assist_rates.py first")
    fpl_only_lookup = pd.read_csv(FPL_ONLY_ASSISTS_FILE, encoding="utf-8-sig").set_index("Player")["FPLOnly_Ast90"].to_dict()
    missing_fpl_only = sorted(set(base["Player"]) - set(fpl_only_lookup))
    if missing_fpl_only:
        raise ValueError(f"{len(missing_fpl_only)} players lack an FPL-only assist rate (e.g. {missing_fpl_only[:3]}); "
                         f"rebuild {FPL_ONLY_ASSISTS_FILE.name}")

    # team predicted NPG per round (needed for opponent-defense scaling and CS)
    home_side = fixture_goals[["MatchID", "Round", "Home", "Away", "Home_Pred_NPG", "Away_Pred_NPG"]].rename(
        columns={"Home": "Team", "Away": "Opponent", "Home_Pred_NPG": "Team_Pred_NPG", "Away_Pred_NPG": "Opp_Pred_NPG"})
    home_side["Venue"] = "Home"
    away_side = fixture_goals[["MatchID", "Round", "Away", "Home", "Away_Pred_NPG", "Home_Pred_NPG"]].rename(
        columns={"Away": "Team", "Home": "Opponent", "Away_Pred_NPG": "Team_Pred_NPG", "Home_Pred_NPG": "Opp_Pred_NPG"})
    away_side["Venue"] = "Away"
    team_fixtures = pd.concat([home_side, away_side], ignore_index=True)

    min_round = fixture_goals["Round"].min()
    check_minutes_round(min_round)
    round_to_gw = {min_round + i: gw for i, gw in enumerate(GAMEWEEKS)}

    all_rows = []
    clipped_defcon = []
    for _, prow in base.iterrows():
        player = prow["Player"]
        pos = prow["Pos"]
        team = prow["Team"]
        threshold = THRESHOLD[pos]

        mp_row = minutes_probs[minutes_probs["Player"] == prow["Web_Name"]]
        pen_row = penalties[penalties["Player"] == player]
        defcon_row = defcon[(defcon["Player"] == player)] if "Player" in defcon.columns else pd.DataFrame()

        ga = ga_lookup.get(prow["Web_Name"], {"NPG/90": 0, "Assists/90": 0})

        for round_num, gw in round_to_gw.items():
            fixture_row = team_fixtures[(team_fixtures["Round"] == round_num) & (team_fixtures["Team"] == team)]
            if fixture_row.empty or mp_row.empty:
                continue
            fixture_row = fixture_row.iloc[0]
            venue_att = home_advantage if fixture_row["Venue"] == "Home" else 2 - home_advantage  # as the rating loop (§181)

            xmins = mp_row[f"xMins_GW{gw}"].values[0]
            p60 = mp_row[f"P60_GW{gw}"].values[0]
            apppts = mp_row[f"AppPts_GW{gw}"].values[0]
            p1 = apppts - p60  # AppPts = P1 + P60 by construction

            # FIX: goal/assist scaling needs the opponent's DEFENSE_RATING (a static
            # team-quality metric of how hard they are to score against) -- NOT
            # Opp_Pred_NPG, which reflects the opponent's own ATTACKING output in
            # this match and has nothing to do with scoring difficulty against them.
            # (Opp_Pred_NPG remains correct for the clean-sheet calculation below,
            # where we genuinely want "goals expected to be scored against us".)
            opp_defense = defense_rating_lookup.get(fixture_row["Opponent"], avg_defense_rating)
            short_frac = (min(max(p1 - p60, 0.0) * SHORT_MINUTES, xmins) / xmins) if (SUB_UPLIFT_ON and xmins > 0) else 0.0
            sub_uplift_npg = 1 + (SUB_SHORT_UPLIFT["npxG"] - 1) * short_frac
            sub_uplift_ast = 1 + (SUB_SHORT_UPLIFT["xAG"] - 1) * short_frac
            # (opponent defence / mean)^beta: the multiplier dampened as in the rating loop (§123)
            scaled_npg90 = ga["NPG/90"] * (opp_defense / avg_defense_rating) ** OPP_BETA_NPG * venue_att * sub_uplift_npg
            fpl_only_ast90 = fpl_only_lookup[player]
            scaled_ast90 = (ga["Assists/90"] * (opp_defense / avg_defense_rating) ** OPP_BETA_AST * venue_att * sub_uplift_ast
                            + fpl_only_ast90)

            pen_attempts = pen_row[f"PenAttempts_GW{gw}"].values[0] if not pen_row.empty else 0.0

            # Clean sheet probability: needs TOTAL goals conceded (non-penalty +
            # penalty), since our team-ratings model only predicts NPG. Opponent's
            # expected penalty goals against this team uses the same empirical
            # attempts-per-NPG ratio and 79% conversion rate used elsewhere.
            opp_pen_goals_expected = fixture_row["Opp_Pred_NPG"] * PEN_ATTEMPTS_PER_NPG * PEN_CONVERSION
            expected_goals_conceded = fixture_row["Opp_Pred_NPG"] + opp_pen_goals_expected
            cs_prob_team = np.exp(-expected_goals_conceded)

            # Defcon probability: join by (Player, MatchID) directly against
            # predictions_defensive.csv's real schema. IMPORTANT: use
            # Probability_at_90min (unscaled), not Probability (already
            # minutes-scaled internally by predict.py to the point-estimate
            # xMins) -- we need the unscaled rate to re-scale fresh for each
            # simulation draw's own randomly-sampled minutes.
            match_id = fixture_row["MatchID"]
            defcon_prob_90 = 0.0
            if not defcon_row.empty:
                this_match = defcon_row[defcon_row["MatchID"] == match_id]
                if not this_match.empty:
                    defcon_prob_90 = this_match["Probability_at_90min"].values[0]
            # A probability at or near 1 inverts to invert_mu's cap (60 actions), which once
            # sent a defender's bonus forecast from 0.16 to 1.19 (CHANGELOG §97, §98).
            if defcon_prob_90 > DEFCON_P90_MAX:
                clipped_defcon.append((prow["Web_Name"], round_num, round(float(defcon_prob_90), 4)))
                defcon_prob_90 = DEFCON_P90_MAX
            mu90 = invert_mu(defcon_prob_90, threshold - 1)  # unscaled -- simulation scales by drawn minutes

            all_rows.append({
                "Player": player, "Web_Name": prow["Web_Name"], "Pos": pos, "Team": team, "Price": prow["Price"],
                "GW": gw, "Round": round_num,
                "Opponent": fixture_row["Opponent"], "Venue": fixture_row["Venue"],
                "xMins": xmins, "xMins_Raw": xmins, "P1": p1, "P60": p60,
                "NPG90_scaled": scaled_npg90, "Ast90_scaled": scaled_ast90,
                "PenAttempts": pen_attempts,
                "FPLOnly_Ast90": fpl_only_ast90,
                "Sub_Uplift_NPG": sub_uplift_npg, "Sub_Uplift_Ast": sub_uplift_ast,
                "Threshold": threshold, "Defcon_Mu90": mu90,
                "CS_Prob_Team": cs_prob_team, "Expected_Goals_Conceded": expected_goals_conceded,
                "ResidualBPS_Mean": prow["ResidualBPS_Mean"], "ResidualBPS_Std": prow["ResidualBPS_Std"],
            })

    out = pd.DataFrame(all_rows)
    if clipped_defcon:
        print(f"WARNING: {len(clipped_defcon)} defcon probabilities above {DEFCON_P90_MAX} clipped before "
              f"inversion (check predict.py's calibration): {clipped_defcon[:10]}")

    # Pass-band probabilities (CHANGELOG §68). An empty rates file means pass bands are
    # inactive this run: P0 = 1 makes the simulator add nothing, the §64 model.
    rates = pd.read_csv(PASS_BAND_FILE, encoding="utf-8-sig") if PASS_BAND_FILE.exists() else pd.DataFrame()
    band_cols = ["PB_P0", "PB_P2", "PB_P4", "PB_P6"]
    if not rates.empty:
        out = out.merge(rates[["Player", "Round"] + band_cols], on=["Player", "Round"], how="left")
        missing = out["PB_P0"].isna()
        if missing.any():
            raise ValueError(f"{int(missing.sum())} rows lack pass-band rates; rebuild pass_band_rates.csv")
    else:
        out = out.assign(PB_P0=1.0, PB_P2=0.0, PB_P4=0.0, PB_P6=0.0)
    scale = recorded_clock_scale(out)
    out["xMins"] = out["xMins_Raw"] * scale
    print(f"xMins clock scale {scale:.4f} ({RECORDED_TEAM_MINUTES} recorded / typical team total on FPL Review's clock)")
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} player-gameweek rows to {OUT_FILE}")
    print(out.head())


if __name__ == "__main__":
    main()