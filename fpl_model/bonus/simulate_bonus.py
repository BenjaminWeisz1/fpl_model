"""
Monte Carlo simulation of bonus points: for each match, simulates N draws of
every player's BPS (minutes, goals, assists, penalties, defcon, clean sheet,
residual "everything else"), ranks players within the match, applies the
official tie-breaking rules, and averages bonus points across draws.

The explicit terms here must match what build_residual_bps_baseline.py
subtracts from actual BPS, so no component is counted twice (CHANGELOG §64).

Each draw's total BPS is rounded to an integer before ranking, as real BPS is, so
ties for a bonus place occur (CHANGELOG §70).

A team's goals conceded are built from the opposing players' simulated goals (CHANGELOG §145).

Keeper saves use the fixture multiplier of keeper_saves.py on the pooled rate, as points do
(CHANGELOG §165). BONUS_SAVE_FIXTURE=0 restores one rate for every keeper.

Run locally (after build_simulation_features.py):
    python -m fpl_model.bonus.simulate_bonus

Importing the module reads no data: the league constants are computed on first use (league_constants()).
"""
from fpl_model import config, paths
import functools
import json
import sys
from typing import NamedTuple

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

from fpl_model.bonus.bps_rules import ASSIST_BPS, CLEAN_SHEET_BPS, GC_BPS, GOAL_BPS, MISS_PEN_BPS, PEN_GOAL_BPS, PEN_SAVE_BPS, SAVE_BPS
from fpl_model.keeper_saves import goalkeeper_event_rates, goalkeeper_pen_save_rate, save_fixture_multiplier, save_fixture_reference
from fpl_model.league_rates import minute_bands, penalty_rates  # noqa: E402

_DATA = paths.CURRENT

FEATURES_FILE = paths.CURRENT / "simulation_features.csv"
TEAM_RATINGS_HISTORY_FILE = paths.CURRENT / "team_ratings_history.csv"
FIXTURE_GOALS_FILE = paths.CURRENT / "fixture_goal_predictions.csv"
OUT_FILE = paths.CURRENT / "bonus_predictions.csv"

SAVE_FIXTURE_ON = config.get("BONUS_SAVE_FIXTURE")  # fixture save rate (§165)

N_DRAWS = 3000
CS_BPS = {"D": CLEAN_SHEET_BPS, "M": 0, "F": 0, "G": CLEAN_SHEET_BPS}
PASS_BANDS = np.array([0, 2, 4, 6])
RESIDUAL_MODE_FILE = paths.CURRENT / "residual_bps_mode.json"

rng = np.random.default_rng(42)
SAVE_RNG = None  # harnesses set this so changing the save rate leaves every other draw alone (§165)


class LeagueConstants(NamedTuple):
    minutes_full: float           # recorded minutes of a 60+ appearance (§181)
    minutes_short: float          # recorded minutes of a 1-59 appearance
    gk_saves_per_90: float        # the keeper save rate points use
    pen_save_rate: float
    pen_attempts_per_npg: float
    pen_conversion: float


@functools.cache
def league_constants() -> LeagueConstants:
    """Computed from the data files once, on first use, so importing the module reads nothing."""
    players = pd.read_csv(_DATA / "players_current.csv", encoding="utf-8-sig")
    full, short = minute_bands()
    pen_attempts, conversion = penalty_rates()
    return LeagueConstants(full, short, goalkeeper_event_rates(players), goalkeeper_pen_save_rate(players),
                           pen_attempts, conversion)


# The module-level names these constants had before, still readable as simulate_bonus.<NAME> (research reads them).
_CONSTANT_NAMES = {"AVG_MINUTES_FULL": "minutes_full", "AVG_MINUTES_SHORT": "minutes_short",
                   "AVG_GK_SAVES_PER_90": "gk_saves_per_90", "PEN_SAVE_RATE": "pen_save_rate",
                   "PEN_ATTEMPTS_PER_NPG": "pen_attempts_per_npg", "PEN_CONVERSION": "pen_conversion"}


def __getattr__(name):
    if name in _CONSTANT_NAMES:
        return getattr(league_constants(), _CONSTANT_NAMES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def assign_bonus(bps_matrix):
    """bps_matrix: (n_players, n_draws). Returns bonus_matrix of the same
    shape, applying the official tie rules per draw."""
    n_players, n_draws = bps_matrix.shape
    bonus = np.zeros_like(bps_matrix)

    for d in range(n_draws):
        col = bps_matrix[:, d]
        order = np.argsort(-col)  # descending
        sorted_bps = col[order]

        # Standard competition ranking: rank 1, 2, 3 pays 3, 2, 1 and tied players share the
        # higher rank, so 3,3,1 and 3,2,2 occur and a three-way tie for first leaves nothing (§183).
        _, first_idx = np.unique(-sorted_bps, return_index=True)
        group_starts = np.append(np.sort(first_idx), n_players)
        for rank, stop in zip(group_starts[:-1], group_starts[1:]):
            if rank >= 3:
                break
            bonus[order[rank:stop], d] = 3 - rank
    return bonus


def simulate_match(match_players, exp_gc_home, exp_gc_away, home_team, n_draws):
    """match_players: DataFrame of all D/M/F/G players in this fixture (both teams)."""
    c = league_constants()
    n = len(match_players)
    pos = match_players["Pos"].values
    team = match_players["Team"].values
    p1 = match_players["P1"].values
    p60 = match_players["P60"].values
    npg90 = match_players["NPG90_scaled"].values
    ast90 = match_players["Ast90_scaled"].values
    pen_att = match_players["PenAttempts"].values
    mu90 = match_players["Defcon_Mu90"].values
    resid_mean = match_players["ResidualBPS_Mean"].values
    resid_std = match_players["ResidualBPS_Std"].values
    goal_bps_rate = np.array([GOAL_BPS.get(p, 0) for p in pos])
    cs_bps_rate = np.array([CS_BPS.get(p, 0) for p in pos])

    u = rng.random((n, n_draws))
    played_any = u < p1[:, None]
    played_full = u < p60[:, None]
    minutes_drawn = np.where(played_full, c.minutes_full, np.where(played_any, c.minutes_short, 0))
    minutes_bps = np.where(played_full, 6, np.where(played_any, 3, 0))

    minute_frac = minutes_drawn / 90.0

    goals = rng.poisson(np.clip(npg90[:, None] * minute_frac, 0, None))
    assists = rng.poisson(np.clip(ast90[:, None] * minute_frac, 0, None))
    goal_bps = goals * goal_bps_rate[:, None]
    assist_bps = assists * ASSIST_BPS

    # PenAttempts already carries minutes (§183), so this is the same expected number, on draws he plays.
    pen_given_played = np.divide(pen_att, p1, out=np.zeros(n), where=p1 > 0).clip(max=1)
    pen_taken = played_any & (rng.random((n, n_draws)) < pen_given_played[:, None])
    pen_scored = pen_taken & (rng.random((n, n_draws)) < c.pen_conversion)
    pen_missed = pen_taken & ~pen_scored
    pen_goal_bps = pen_scored * PEN_GOAL_BPS
    pen_miss_bps = pen_missed * MISS_PEN_BPS

    mu_scaled = mu90[:, None] * minute_frac
    r = np.clip(3 * mu_scaled, 1e-6, None)
    defcon_count = rng.negative_binomial(r, 0.75)
    defcon_bps = defcon_count // 3

    home_mask = team == home_team
    # Team goals conceded = the opponents' simulated goals + a remainder for own goals and
    # unlisted players, so scorers and clean sheets agree (CHANGELOG §71, §145).
    e_player = npg90 * (p60 * c.minutes_full + (p1 - p60) * c.minutes_short) / 90 + pen_att * c.pen_conversion
    scored = goals + pen_scored
    goals_conceded_home = scored[~home_mask].sum(axis=0) + rng.poisson(max(exp_gc_home - e_player[~home_mask].sum(), 0), n_draws)
    goals_conceded_away = scored[home_mask].sum(axis=0) + rng.poisson(max(exp_gc_away - e_player[home_mask].sum(), 0), n_draws)
    team_goals_conceded = np.where(home_mask[:, None], goals_conceded_home[None, :], goals_conceded_away[None, :])

    # Current 2026/27 keeper save model: 2 BPS per save; penalty saves are scored as 7 BPS
    # (not 8) because the big-chance bonus is already included in the new kit. We only award
    # goalkeeper save events for players who actually played, and we model penalty saves only on
    # those same active draws; otherwise the simulator massively overcounts future keepers who
    # never entered the match.
    gk_mask = pos == "G"
    gk_save_bps = np.zeros_like(minutes_bps, dtype=float)
    if np.any(gk_mask):
        # The residual excludes exactly these save terms (§64), so the official rate applies
        # unscaled.
        gk_active = gk_mask[:, None] & played_full
        save_mult = match_players["GK_Save_Mult"].values if "GK_Save_Mult" in match_players else np.ones(n)
        gk_save_rate = np.clip(c.gk_saves_per_90 * save_mult[gk_mask][:, None] * minute_frac[gk_mask], 0, None)
        gk_saves = np.zeros_like(minutes_bps, dtype=float)
        gk_saves[gk_mask] = (rng if SAVE_RNG is None else SAVE_RNG).poisson(gk_save_rate)

        pen_save_prob = np.where(gk_active, c.pen_save_rate, 0.0)
        pen_save_draw = rng.random((n, n_draws)) < pen_save_prob
        gk_pen_save_bps = np.where(gk_active & pen_save_draw, PEN_SAVE_BPS, 0)
        gk_save_bps = np.where(gk_active, gk_saves * SAVE_BPS, 0)
        gk_save_bps = gk_save_bps + gk_pen_save_bps

    cs_eligible = played_full & (team_goals_conceded == 0)
    cs_bps = cs_eligible * cs_bps_rate[:, None]

    # BPS deducts goals conceded while on the pitch at any minutes; a shorter
    # draw sees each team goal with probability minute_frac.
    gc_bps_rate = np.where(np.isin(pos, ["D", "G"]), GC_BPS, 0)
    on_pitch_gc = np.where(played_full, team_goals_conceded,
                           rng.binomial(team_goals_conceded, minute_frac))
    gc_bps = played_any * on_pitch_gc * gc_bps_rate[:, None]

    residual_bps = rng.normal(resid_mean[:, None], resid_std[:, None], size=(n, n_draws))
    residual_bps = np.where(played_any, residual_bps, 0)

    # Pass bands (§68) on 60+ draws. Skipped entirely when inactive (all P0 = 1), so the
    # §64 model's random stream and outputs are unchanged.
    band_probs = match_players[["PB_P0", "PB_P2", "PB_P4", "PB_P6"]].to_numpy(float)
    pass_band_bps = 0
    if np.any(band_probs[:, 0] < 1):
        cum = np.cumsum(band_probs, axis=1)
        u = rng.random((n, n_draws))
        index = (u[:, :, None] > cum[:, None, :]).sum(axis=2).clip(max=3)
        pass_band_bps = np.where(played_full, PASS_BANDS[index], 0)

    total_bps = (minutes_bps + goal_bps + assist_bps + pen_goal_bps + pen_miss_bps
                 + defcon_bps + cs_bps + gc_bps + residual_bps + gk_save_bps + pass_band_bps)
    total_bps = np.rint(total_bps)  # integer BPS, so ties occur (§70)

    bonus = assign_bonus(total_bps)
    return bonus.mean(axis=1), goals.mean(axis=1), assists.mean(axis=1)


def main():
    features = pd.read_csv(FEATURES_FILE, encoding="utf-8-sig")
    fixture_goals = pd.read_csv(FIXTURE_GOALS_FILE, encoding="utf-8-sig")

    # The residual must exclude pass bands exactly when the features carry them,
    # otherwise pass-band BPS would be counted twice or not at all.
    if "PB_P0" not in features:
        features = features.assign(PB_P0=1.0, PB_P2=0.0, PB_P4=0.0, PB_P6=0.0)
    mode = json.loads(RESIDUAL_MODE_FILE.read_text(encoding="utf-8")) if RESIDUAL_MODE_FILE.exists() else {}
    bands_on = bool((features["PB_P0"] < 1).any())
    if bands_on != bool(mode.get("pass_bands", False)):
        raise ValueError(f"Pass-band mismatch: features {'have' if bands_on else 'lack'} pass bands but the "
                         f"residual was built with pass_bands={mode.get('pass_bands')}; rerun part 2 in order")
    print(f"Pass bands: {'on' if bands_on else 'off'} ({mode.get('reason', 'no residual mode file')})")

    if SAVE_FIXTURE_ON:
        history = pd.read_csv(TEAM_RATINGS_HISTORY_FILE, encoding="utf-8-sig", low_memory=False)
        features = features.assign(GK_Save_Mult=save_fixture_multiplier(
            features["Expected_Goals_Conceded"], save_fixture_reference(history)))

    c = league_constants()
    fixture_goals = fixture_goals.copy()
    fixture_goals["Exp_GC_Home"] = fixture_goals["Away_Pred_NPG"] * (1 + c.pen_attempts_per_npg * c.pen_conversion)
    fixture_goals["Exp_GC_Away"] = fixture_goals["Home_Pred_NPG"] * (1 + c.pen_attempts_per_npg * c.pen_conversion)

    results = []
    for _, fx in fixture_goals.iterrows():
        round_num, home, away = fx["Round"], fx["Home"], fx["Away"]
        gw_row = features[features["Round"] == round_num]
        if gw_row.empty:
            continue
        gw = gw_row["GW"].iloc[0]

        match_players = features[(features["Round"] == round_num) &
                                  (features["Team"].isin([home, away]))]
        if match_players.empty:
            continue

        bonus_avg, goals_avg, assists_avg = simulate_match(
            match_players, fx["Exp_GC_Home"], fx["Exp_GC_Away"], home, N_DRAWS)

        for i, (_, prow) in enumerate(match_players.iterrows()):
            results.append({
                "Player": prow["Player"], "Web_Name": prow["Web_Name"], "Pos": prow["Pos"],
                "Team": prow["Team"], "GW": gw, "Round": round_num,
                "Expected_Bonus": round(bonus_avg[i], 4),
            })

        print(f"Round {round_num}: {home} vs {away} -- {len(match_players)} players simulated")

    out = pd.DataFrame(results)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nWrote {len(out)} player-gameweek bonus predictions to {OUT_FILE}")
    print(out.sort_values("Expected_Bonus", ascending=False).head(15))


if __name__ == "__main__":
    main()