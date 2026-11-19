"""
Computes NPG/90 and Assists/90 for every player in players_current.csv,
running the full rating pipeline fresh from the start of player_match_stats.csv
(the consolidated, already-harmonized merge of fbref 2017-2024 and FPL-API
2025/26+ data) rather than resuming from an uncertain player_ratings.csv
checkpoint.

This is a faithful reconstruction of the original pipeline from the plan
document, with one deliberate change: the old player_conversion_rates.csv
Goal_Conversion_Rate step is REPLACED by a shrinkage-corrected NPG/npxG
ratio computed directly from career history:
    conversion = (career_NPG + 30) / (career_npxG + 30)
    NPG/90 = npxG_rating * conversion

The stage runs in five steps, each a function: load_inputs() reads the data files, prepare_player_stats() builds the
match rows the loop learns from, build_initial_ratings() the starting ratings from transfers.csv,
smooth_ratings_and_mae() the smoothing loop, and final_ratings_frame() the output table, which main() writes.
Importing the module reads no file. Settings come from fpl_model.config (the GOAL_MODEL_* variables).

    python -m fpl_model.player_ratings.build_player_ratings
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from fpl_model import config, paths
from fpl_model.league_rates import current_season
from fpl_model.match_order import chrono_order
from fpl_model.player_ratings import position_rates as _pos_rates
from fpl_model.team_ratings.update_team_ratings import home_advantage

_DATA = paths.CURRENT

# ---- constants (from the original pipeline) ----
minutes_constant = 500
league_coefficient_multiplier = 0.7
youth_coefficient = 0.75
weight_team = 0.15
AVG_CONCEDED = 1.32023
SHRINKAGE_PRIOR = 30  # the new conversion-rate correction's pseudo-count

# ---- settings (fpl_model/config.py) ----
ALPHA = config.get("GOAL_MODEL_ALPHA")  # learning-rate floor; env for experiments (§196)
# npxG has its own floor since §233 (0.035; GOAL_MODEL_ALPHA_NPG=0.025 reverts). Without GOAL_MODEL_ALPHA_NPG, an
# experiment that sets GOAL_MODEL_ALPHA moves both floors, as it did before §233.
ALPHA_NPG = config.get("GOAL_MODEL_ALPHA_NPG")
# Centre-backs' npxG floor since §235 (0.018, user decision; GOAL_MODEL_ALPHA_NPG_CB=0.035 reverts). A centre-back row
# = the match's Position_Resolved is CB; a row without a position takes the player's previous match group, else his
# transfers.csv group (first listed position), as validate_kalman_smoothing.load(). When an experiment sets
# GOAL_MODEL_ALPHA or GOAL_MODEL_ALPHA_NPG without this, centre-backs follow ALPHA_NPG (pre-§235 behaviour).
ALPHA_NPG_CB = config.get("GOAL_MODEL_ALPHA_NPG_CB")
RETURN_H = config.get("GOAL_MODEL_RETURN_H")  # second-season blend discount (§114)
PRIOR_K = config.get("GOAL_MODEL_PRIOR_K")  # matches the starting rating counts as (§112; 5 = pre-§112)
# Players whose rating starts from a `youth` row (§126); defaults to PRIOR_K (no separate weight).
PRIOR_K_YOUTH = config.get("GOAL_MODEL_PRIOR_K_YOUTH")
# Dampening exponent on the opponent factor, per target (§123, adopted 0.7 / 0.8); 1 = the plain multiplier.
OPP_BETA_NPG = config.get("GOAL_MODEL_OPP_BETA_NPG")
OPP_BETA_AST = config.get("GOAL_MODEL_OPP_BETA_AST")
# §243-§244: expected assists calibrated to conventional assists by match slot. Adopted by user decision 2026-11-13:
# default fw_cb; GOAL_MODEL_XAG_CALIBRATION=off reverts. Season-end check frozen in
# data/validation/xag_calibration_seasonend_2027/protocol.json.
#   fw_cb: FPL-era (Opta) xA x assists/Opta xA for forwards and centre-backs only (factors from 2022-2024)
#   all:   the same for every slot (not recommended, §243)
#   both:  every row, fbref xAG x assists/fbref xAG (2017-2024) and Opta xA x assists/Opta xA (2022-2024) (not recommended)
XAG_CALIBRATION = config.get("GOAL_MODEL_XAG_CALIBRATION")

# ---- pre-match rating log (CHANGELOG §72, §74) ----
# When set to a path, every rated row's goal/assist rating *before* that match's
# update is written there with its one-step-ahead prediction and actual, and the
# would-be ratings go to <path>.ratings.csv instead of player_goal_assist_ratings.csv.
# Unset (production) leaves the output unchanged.
PREMATCH_LOG = config.get("GOAL_MODEL_PREMATCH_LOG")

# ---- role factor (position plan step 3, CHANGELOG §172) ----
# Experiment only: unset = no effect. Rating stays on the scale of the player's first logged line;
# the loop divides by (line rate / first-line rate)^k for the line played, and the log also records
# the honest prediction from role percentages. Refuses to run without PREMATCH_LOG, so never production.
_ROLE_K_ENV = ["GOAL_MODEL_ROLE_K_NPG_UP", "GOAL_MODEL_ROLE_K_NPG_DOWN",
               "GOAL_MODEL_ROLE_K_AST_UP", "GOAL_MODEL_ROLE_K_AST_DOWN"]
ROLE_K_ON = any(config.is_set(v) for v in _ROLE_K_ENV)

# ---- same-player move factors (user's form of the position plan, CHANGELOG §173) ----
# Experiment only: unset = no effect. One-off rating multiply when a move is detected, divided out on
# moving back. Refuses to run without PREMATCH_LOG, so never production.
MOVE_TABLE_FILE = config.get("GOAL_MODEL_MOVE_TABLE")

# ---- position update for entrants (CHANGELOG §247, §249; adopted for npxG and xAG by user decision 2026-11-14) ----
# An entrant's starting rating holds his listed position's rate p_listed (§77 rule). While he is an entrant (his
# segment's first two seasons: youth, second season, missed season and arrival rows), that part of the start moves
# towards the rate of the slots he has actually played: before each match the rating carries c x s x (p - p_listed),
# s = the share of the start still in the rating, p = (Kpos x p_listed + sum of minutes/90 x slot rate over his earlier
# matches in the segment) / (Kpos + sum of minutes/90). Every appearance counts (form F0). Rates are
# position_combinations.csv's. GOAL_MODEL_POS_UPDATE=0 reverts.
POS_UPDATE = config.get("GOAL_MODEL_POS_UPDATE")
POS_UPDATE_C = (config.get("GOAL_MODEL_POS_UPDATE_C_NPG"), config.get("GOAL_MODEL_POS_UPDATE_C_AST"))
POS_UPDATE_KPOS = 1.0
_PU_SLOT = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
            "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW"}   # GK and unknown slots give no evidence

# Substitute appearances are learned on a starter-equivalent scale (CHANGELOG §100, §101): their
# per-90 prediction carries the frozen uplift for their minutes band. Start flag: Understat Is_Starter.
SUB_UPLIFT = {"npxG": [(15, 1.663), (30, 1.277), (45, 1.170), (999, 0.868)],
              "xAG": [(15, 1.685), (30, 1.276), (45, 1.113), (999, 1.045)]}

# Position group of a match for the centre-back npxG floor (§235); groups as validate_kalman_smoothing.POS_GROUP.
_MATCH_GROUP = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "FB", "DM": "CM", "CM": "CM",
                "AM": "AW", "LM": "AW", "RM": "AW", "LW": "AW", "RW": "AW", "FW": "FW"}
_TRANSFER_GROUP = {"CB": "CB", "FB": "FB", "WB": "FB", "DM": "CM", "CM": "CM", "AM": "AW", "W": "AW", "WM": "AW",
                   "FW": "FW"}

_RATING_COLS = ['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']


# ---------------------------------------------------------------------------------------------------------------------
# Research hooks (never production: both refuse to run without the pre-match log)
# ---------------------------------------------------------------------------------------------------------------------
class RoleFactor(NamedTuple):
    k_npg_up: float
    k_npg_down: float
    k_ast_up: float
    k_ast_down: float
    rate_npg: np.ndarray
    rate_ast: np.ndarray
    table: dict


class MoveTable(NamedTuple):
    factors: dict
    keys: dict


def load_research_hooks() -> tuple[RoleFactor | None, MoveTable | None]:
    """The §172 role factor and the §173 move table, when their settings are on."""
    role = move = None
    if ROLE_K_ON:
        if not PREMATCH_LOG:
            raise SystemExit("GOAL_MODEL_ROLE_K_* needs GOAL_MODEL_PREMATCH_LOG: it must not reach production")
        from fpl_model.player_ratings.role_shares import LINE_RATES_FILE, role_table
        rates = json.loads(LINE_RATES_FILE.read_text(encoding="utf-8"))
        role = RoleFactor(*[config.get(v) or 0.0 for v in _ROLE_K_ENV],
                          np.array(rates["npxG"]), np.array(rates["xAG"]), role_table())
    if MOVE_TABLE_FILE:
        if not PREMATCH_LOG:
            raise SystemExit("GOAL_MODEL_MOVE_TABLE needs GOAL_MODEL_PREMATCH_LOG: it must not reach production")
        mt = json.loads(Path(MOVE_TABLE_FILE).read_text(encoding="utf-8"))
        move = MoveTable(mt["factors"], {tuple(k): v for k, v in mt["triggers"]})
    return role, move


def _role_mult(m1, up, down):
    return m1 ** (up if m1 > 1 else down)


# ---------------------------------------------------------------------------------------------------------------------
# Position priors (CHANGELOG §77)
# ---------------------------------------------------------------------------------------------------------------------
# position_combinations.csv holds the single-position priors only. A multi-position
# entry ("W,AM,FW") is their weighted average, each listed position counting twice the
# next: 2:1, 4:2:1, 8:4:2:1. That rule reproduces all 103 combinations the file used to
# list to within 5e-10, and covers every combination, so none can be missing.
def position_prior(pos, single):
    parts = [p.strip() for p in str(pos).split(',')]
    unknown = [p for p in parts if p not in single.index]
    if unknown:
        return None, unknown
    weights = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
    return {c: float(np.average(single.loc[parts, c], weights=weights)) for c in ['NPG/90', 'Assists/90']}, []


def build_position_priors(single_df, used):
    single = single_df.set_index('Pos')
    rows, bad = [], {}
    for pos in sorted(set(used) | set(single.index)):
        prior, unknown = position_prior(pos, single)
        if unknown:
            bad[pos] = unknown
        else:
            rows.append({'Pos': pos, **prior})
    if bad:
        # A blank prior gives a NaN initial rating that then sticks to the player for good.
        raise ValueError(f"transfers.csv uses positions not in position_combinations.csv: {bad}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------------------
# Step 1: inputs
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Inputs:
    transfers: pd.DataFrame
    single_positions: pd.DataFrame      # position_combinations.csv: one row per single position
    positions: pd.DataFrame             # single and every used combination (§77)
    leagues: pd.DataFrame
    matchlogs: pd.DataFrame             # team_ratings_history.csv
    team_ratings_now: pd.DataFrame      # this season's preseason team ratings
    players_current: pd.DataFrame
    player_stats: pd.DataFrame          # player_match_stats.csv, as read


def load_inputs(data: Path = _DATA) -> Inputs:
    transfers = pd.read_csv(paths.current_file('transfers.csv', root=data))
    for col in ['Prior_Minutes', 'Seasons_Missed', 'Team_Attack_Rating', 'Min_Old', 'npxG_Old', 'xAG_Old', 'NPG_Old',
                'Assists_Old']:
        if col in transfers.columns:
            transfers[col] = pd.to_numeric(transfers[col], errors='coerce')
    single = pd.read_csv(paths.current_file('position_combinations.csv', root=data))
    if single['Pos'].str.contains(',').any():
        raise ValueError("position_combinations.csv must hold single positions only; "
                         "combinations are computed (CHANGELOG §77)")
    positions = build_position_priors(single, transfers['Pos'].dropna())
    # §246 experiment: GOAL_MODEL_POSITION_RATES = slots | slots3 replaces the single-position rates, per transfers.csv
    # row, by PL starts' slot rates in the seasons before the row's season (position_rates.py). Default "file".
    if _pos_rates.WINDOW is not None:
        print(f"[§246] position rates from per-match slots, window {_pos_rates.WINDOW} "
              f"(min {_pos_rates.MIN_STARTS} starts)")
    return Inputs(
        transfers=transfers, single_positions=single, positions=positions,
        leagues=pd.read_csv(paths.current_file('league_coefficients.csv', root=data)),
        matchlogs=pd.read_csv(paths.current_file('team_ratings_history.csv', root=data)),
        team_ratings_now=pd.read_csv(paths.current_file(f'team_ratings_preseason_{current_season()}.csv', root=data)),
        players_current=pd.read_csv(paths.current_file('players_current.csv', root=data), encoding='utf-8-sig'),
        # player_match_stats.csv is already harmonized (fbref 2017-2024 + FPL-API 2025/26+): consistent MatchID
        # scheme, NPG computed, CBI computed for both sources. No manual schema reconciliation needed.
        player_stats=pd.read_csv(paths.current_file('player_match_stats.csv', root=data), encoding='utf-8-sig', low_memory=False),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Step 2: the match rows the loop learns from
# ---------------------------------------------------------------------------------------------------------------------
def prepare_player_stats(playerstats: pd.DataFrame, matchlogs: pd.DataFrame, data: Path = _DATA) -> pd.DataFrame:
    """Red-card adjustment, start flags and match slots, xAG calibration, the season key, team attack ratings and
    chronological order."""
    # ---- red-card adjustment of player npxG / xAG (§136): the team-level factors, per player from his own shots ----
    # Keyed on the native Season (before the overwrite below). The data files stay unadjusted.
    red_card_file = config.get("GOAL_MODEL_RED_CARD_FILE") or paths.current_file('red_card_player_factors.csv', root=data)
    if red_card_file.exists():
        rc = pd.read_csv(red_card_file, encoding='utf-8-sig')
        key = ['Season', 'Player', 'Team', 'Opponent', 'Venue']
        f = playerstats[key].merge(rc[key + ['Factor_npxG', 'Factor_xAG']], on=key, how='left')
        assert len(f) == len(playerstats)
        playerstats['npxG'] = playerstats['npxG'] * f['Factor_npxG'].fillna(1.0).to_numpy()
        playerstats['xAG'] = playerstats['xAG'] * f['Factor_xAG'].fillna(1.0).to_numpy()
        print(f"[§136] red-card factors applied to {int(f['Factor_npxG'].notna().sum())} player-match rows")

    # ---- start flag (§101) and the match slot, from player_match_positions.csv ----
    starts = pd.read_csv(paths.current_file('player_match_positions.csv', root=data), encoding='utf-8-sig', low_memory=False,
                         usecols=['Season', 'Team', 'Opponent', 'Venue', 'Player', 'Is_Starter', 'Position_Resolved'])
    starts = starts.drop_duplicates(['Season', 'Team', 'Opponent', 'Venue', 'Player'])
    n_before = len(playerstats)
    playerstats['_PosSeason'] = playerstats['Season']   # native season, the key of the position file
    playerstats = playerstats.merge(starts.rename(columns={'Season': '_StartSeason', 'Is_Starter': '_IsStarter',
                                                           'Position_Resolved': '_MatchPos'}),
                                    left_on=['Season', 'Team', 'Opponent', 'Venue', 'Player'],
                                    right_on=['_StartSeason', 'Team', 'Opponent', 'Venue', 'Player'], how='left')
    assert len(playerstats) == n_before
    playerstats = playerstats.drop(columns='_StartSeason')
    is_sub = playerstats['_IsStarter'].astype(str).str.lower().eq('false')
    print(f"[§101] start flag on {playerstats['_IsStarter'].notna().mean():.1%} of rows; "
          f"substitute rows {int(is_sub.sum())}")

    if XAG_CALIBRATION != "off":
        playerstats['xAG'] = _calibrated_xag(playerstats, data)

    if PREMATCH_LOG:
        playerstats['_LogSeason'] = playerstats['Season']

    # Derive Season: for fbref rows (MatchID 0..N, 380 matches/season starting 2017);
    # FPL_API rows keep their native Season (2025, 2026, ...). They used to be forced to
    # 2025, so a new arrival whose transfers.csv row is Season 2026 never found a valid
    # initial rating and his 2026/27 matches were skipped (CHANGELOG §74, §76; predict.py §9).
    playerstats['Season'] = np.where(
        playerstats['Data_Source'] == 'fbref',
        2017 + playerstats['MatchID'] // 380,
        playerstats['Season'],
    )

    # Team_Attack_Rating is not a column in player_match_stats.csv -- fill it
    # entirely from team_ratings_history.csv (per-match team attack ratings).
    # Season is part of the key: without it every row took the first season that
    # fixture occurred (often 2017), which the club-change adjustment below reads
    # (CHANGELOG §85).
    home_atk = matchlogs[['Season', 'Home', 'Away', 'Home_Attack_Rating']].rename(
        columns={'Home': 'Team', 'Away': 'Opponent', 'Home_Attack_Rating': 'Team_Attack_Rating'})
    home_atk['Venue'] = 'Home'
    away_atk = matchlogs[['Season', 'Away', 'Home', 'Away_Attack_Rating']].rename(
        columns={'Away': 'Team', 'Home': 'Opponent', 'Away_Attack_Rating': 'Team_Attack_Rating'})
    away_atk['Venue'] = 'Away'
    team_atk_lookup = pd.concat([home_atk, away_atk], ignore_index=True).drop_duplicates(
        subset=['Season', 'Team', 'Opponent', 'Venue'])
    atk_fill = playerstats[['Season', 'Team', 'Opponent', 'Venue']].merge(
        team_atk_lookup, on=['Season', 'Team', 'Opponent', 'Venue'], how='left')
    playerstats['Team_Attack_Rating'] = atk_fill['Team_Attack_Rating']
    missing_atk = playerstats['Team_Attack_Rating'].isna().sum()
    if missing_atk > 0:
        print(f"WARNING: {missing_atk} of {len(playerstats)} rows still have no "
              f"Team_Attack_Rating after the lookup -- check Team/Opponent/Venue spelling "
              f"consistency against team_ratings_history.csv")

    return playerstats.assign(_Chrono=chrono_order(playerstats, data)).sort_values('_Chrono').reset_index(drop=True)


def _calibrated_xag(playerstats: pd.DataFrame, data: Path) -> np.ndarray:
    """Expected assists calibrated to assists by match slot (§243); a row without a slot takes the player's most
    common slot, else the overall factor."""
    from fpl_model.player_ratings.xag_source_scale import CANON, assist_factors
    fpl = playerstats['Data_Source'].eq('FPL_API').to_numpy()
    slot = playerstats['_MatchPos'].map(CANON)
    modal = (playerstats.assign(_s=slot).dropna(subset=['_s'])
             .groupby('Player')['_s'].agg(lambda s: s.value_counts().index[0]))
    slot = slot.fillna(playerstats['Player'].map(modal))

    def factors_for(table, slots=None):
        f = slot.map(table).fillna(table['overall'])
        if slots is not None:
            f = f.where(slot.isin(slots), 1.0)
        return f.to_numpy()

    opta = assist_factors(data, "opta")
    keep = ["FW", "CB"] if XAG_CALIBRATION == "fw_cb" else None
    new = np.where(fpl, playerstats['xAG'] * factors_for(opta, keep), playerstats['xAG'])
    if XAG_CALIBRATION == "both":
        new = np.where(fpl, new, playerstats['xAG'] * factors_for(assist_factors(data, "fbref")))
    print(f"[§243] xAG calibrated to assists ({XAG_CALIBRATION}); Opta factors "
          + ", ".join(f"{k} {v:.3f}" for k, v in sorted(opta.items())))
    return new


def sub_uplift(kind, is_starter, minutes):
    if str(is_starter).lower() != 'false':
        return 1.0
    return next(u for upper, u in SUB_UPLIFT[kind] if minutes <= upper)


# Since §110 the loop's multiplier is the one the forecast applies (build_simulation_features.py):
# the opponent's pre-match Defense_Rating / that season's mean, keyed on (Season, Team, Opponent, Venue).
# It used to be the player's team's predicted goals keyed on MatchID: in 2017-2024 that also divided
# out his own team's attack (inflating weak-attack players), and FPL-era MatchIDs don't match, so 2025+
# rows got ~1 (CHANGELOG §78, §79, §110).
def build_concede_lookup(matchlogs):
    concede = {}
    season_mean = pd.concat([
        matchlogs[["Season", "Home_Defense_Rating"]].rename(columns={"Home_Defense_Rating": "Def"}),
        matchlogs[["Season", "Away_Defense_Rating"]].rename(columns={"Away_Defense_Rating": "Def"}),
    ]).groupby("Season")["Def"].mean()
    for _, row in matchlogs.iterrows():
        season, home_team, away_team = row["Season"], row["Home"], row["Away"]
        scale = AVG_CONCEDED / season_mean[season]
        concede[(season, home_team, away_team, "Home")] = row["Away_Defense_Rating"] * scale
        concede[(season, away_team, home_team, "Away")] = row["Home_Defense_Rating"] * scale
    return concede


# ---------------------------------------------------------------------------------------------------------------------
# Step 3: starting ratings from transfers.csv
# ---------------------------------------------------------------------------------------------------------------------
class InitialRatings(NamedTuple):
    initial: pd.DataFrame          # every branch: the loop's starting ratings
    first_season: pd.DataFrame     # every branch but 'second season': fallback for players never seen
    second_season: pd.DataFrame    # returners' new-league priors, blended in at their first match back
    youth: pd.DataFrame


def build_initial_ratings(transfers_df, positions_df, leagues_df, data: Path = _DATA) -> InitialRatings:
    """The six transfers.csv branches, the new-team factor, then the file-based priors (§103, §128)."""
    transfers_df = transfers_df.copy()
    rates = positions_df[['Pos', 'NPG/90', 'Assists/90']]
    coefficients = leagues_df[['League', 'Coefficient']]

    gk_df = transfers_df[(transfers_df['Pos'] == 'GK') & (transfers_df['Notes'] != 'second season')]
    gk_df = gk_df.merge(rates, on='Pos', how='left')
    gk_df['Initial_NPG_Rating'] = gk_df['NPG/90']
    gk_df['Initial_Assist_Rating'] = gk_df['Assists/90']

    transfers_2016_df = transfers_df[(transfers_df['Season'] == 2016) & (transfers_df['Pos'] != 'GK')]
    transfers_2016_df = transfers_2016_df.merge(rates, on='Pos', how='left')
    transfers_2016_df = _pos_rates.apply_season_rates(transfers_2016_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
    transfers_2016_df['Initial_NPG_Rating'] = (
        (transfers_2016_df['NPG_Old'] + transfers_2016_df['NPG/90'] * minutes_constant / 90) /
        ((transfers_2016_df['Min_Old'] / 90) + minutes_constant / 90)
    )
    transfers_2016_df['Initial_Assist_Rating'] = (
        (transfers_2016_df['Assists_Old'] + transfers_2016_df['Assists/90'] * minutes_constant / 90) /
        ((transfers_2016_df['Min_Old'] / 90) + minutes_constant / 90)
    )

    transfers_df['npxG_Old'] = transfers_df['npxG_Old'].fillna(transfers_df['NPG_Old'])
    transfers_df['xAG_Old'] = transfers_df['xAG_Old'].fillna(transfers_df['Assists_Old'])

    epl_coefficient = leagues_df.loc[leagues_df['League'] == 'Premier League', 'Coefficient'].values[0]

    since_2017 = transfers_df[
        (transfers_df['Season'] > 2016) & (transfers_df['Notes'] != 'youth') &
        (transfers_df['Notes'] != 'second season') & (transfers_df['Notes'] != 'missed season') &
        (transfers_df['Pos'] != 'GK')
    ]
    since_2017 = since_2017.merge(coefficients, on='League', how='left')
    since_2017 = since_2017.merge(rates, on='Pos', how='left')
    since_2017 = _pos_rates.apply_season_rates(since_2017)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
    since_2017['Initial_NPG_Rating'] = (
        (league_coefficient_multiplier * since_2017['Coefficient'] / epl_coefficient) *
        (since_2017['npxG_Old'] + since_2017['NPG/90'] * minutes_constant / 90) /
        ((since_2017['Min_Old'] / 90) + minutes_constant / 90)
    )
    since_2017['Initial_Assist_Rating'] = (
        (league_coefficient_multiplier * since_2017['Coefficient'] / epl_coefficient) *
        (since_2017['xAG_Old'] + since_2017['Assists/90'] * minutes_constant / 90) /
        ((since_2017['Min_Old'] / 90) + minutes_constant / 90)
    )

    missed = transfers_df[(transfers_df['Notes'] == 'missed season') & (transfers_df['Pos'] != 'GK')]
    missed = missed.merge(coefficients, on='League', how='left')
    missed = missed.merge(rates, on='Pos', how='left')
    missed = _pos_rates.apply_season_rates(missed)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
    half = minutes_constant / 2
    missed['Initial_NPG_Rating'] = np.where(
        missed['League'] == 'Premier League',
        ((missed['NPG_Old'] + missed['NPG/90'] * half / 90) / ((missed['Min_Old'] / 90) + half / 90)),
        (league_coefficient_multiplier * missed['Coefficient'] / epl_coefficient) *
        (missed['npxG_Old'] + missed['NPG/90'] * half / 90) / ((missed['Min_Old'] / 90) + half / 90)
    )
    missed['Initial_Assist_Rating'] = np.where(
        missed['League'] == 'Premier League',
        ((missed['Assists_Old'] + missed['Assists/90'] * half / 90) / ((missed['Min_Old'] / 90) + half / 90)),
        (league_coefficient_multiplier * missed['Coefficient'] / epl_coefficient) *
        (missed['xAG_Old'] + missed['Assists/90'] * half / 90) / ((missed['Min_Old'] / 90) + half / 90)
    )

    youth_df = transfers_df[(transfers_df['Notes'] == 'youth') & (transfers_df['Pos'] != 'GK')]
    youth_df = youth_df.merge(rates, on='Pos', how='left')
    youth_df = _pos_rates.apply_season_rates(youth_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
    youth_df['Initial_NPG_Rating'] = np.where(youth_df['Pos'] != 'FW', youth_coefficient * youth_df['NPG/90'],
                                              0.7 * youth_df['NPG/90'])
    youth_df['Initial_Assist_Rating'] = youth_coefficient * youth_df['Assists/90']
    # Fitted youth multipliers (§113): when set, one factor per target for every position replaces the 0.75 / 0.7 above.
    if config.is_set("GOAL_MODEL_YOUTH_NPG"):
        youth_df['Initial_NPG_Rating'] = config.get("GOAL_MODEL_YOUTH_NPG") * youth_df['NPG/90']
    if config.is_set("GOAL_MODEL_YOUTH_AST"):
        youth_df['Initial_Assist_Rating'] = config.get("GOAL_MODEL_YOUTH_AST") * youth_df['Assists/90']

    second = transfers_df[(transfers_df['Notes'] == 'second season') & (transfers_df['Pos'] != 'GK')]
    second = second.merge(coefficients, on='League', how='left')
    second = second.merge(rates, on='Pos', how='left')
    second = _pos_rates.apply_season_rates(second)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
    second['Initial_NPG_Rating'] = (
        (league_coefficient_multiplier * second['Coefficient'] / epl_coefficient) *
        (second['npxG_Old'] + second['NPG/90'] * minutes_constant / 90) /
        ((second['Min_Old'] / 90) + minutes_constant / 90)
    )
    second['Initial_Assist_Rating'] = (
        (league_coefficient_multiplier * second['Coefficient'] / epl_coefficient) *
        (second['xAG_Old'] + second['Assists/90'] * minutes_constant / 90) /
        ((second['Min_Old'] / 90) + minutes_constant / 90)
    )

    first_season_df = pd.concat([d[_RATING_COLS] for d in (gk_df, transfers_2016_df, since_2017, missed, youth_df)])
    initial_ratings_df = pd.concat([d[_RATING_COLS]
                                    for d in (gk_df, transfers_2016_df, since_2017, missed, youth_df, second)])
    for d in (initial_ratings_df, first_season_df):
        team_factor = (d['Team_Attack_Rating'] / AVG_CONCEDED) ** weight_team
        d['Initial_NPG_Rating'] = d['Initial_NPG_Rating'] * team_factor
        d['Initial_Assist_Rating'] = d['Initial_Assist_Rating'] * team_factor

    # ---- returners' new-league prior from a file (§128): replaces only the blend's new-league half ----
    return_priors_file = config.get("GOAL_MODEL_RETURN_PRIORS_FILE")
    if return_priors_file:
        rp = pd.read_csv(return_priors_file, encoding='utf-8-sig')
        rp_map = {(p, s): (g, a) for p, s, g, a in rp[['Player', 'Season', 'Initial_NPG_Rating',
                                                       'Initial_Assist_Rating']].itertuples(index=False)}
        keys = list(zip(second['Player'], second['Season']))
        hit = [k in rp_map for k in keys]
        second.loc[hit, 'Initial_NPG_Rating'] = [rp_map[k][0] for k, h in zip(keys, hit) if h]
        second.loc[hit, 'Initial_Assist_Rating'] = [rp_map[k][1] for k, h in zip(keys, hit) if h]
        print(f"[§128] returners' new-league prior from {Path(return_priors_file).name}: {sum(hit)} rows")

    # A blank blend input turns the player's rating NaN at his first match back, and NaN sticks for good (§185).
    blend_inputs = ['Initial_NPG_Rating', 'Initial_Assist_Rating', 'Prior_Minutes', 'Seasons_Missed']
    bad_blend = second[second[blend_inputs].isna().any(axis=1)]
    if len(bad_blend):
        raise SystemExit("transfers.csv 'second season' rows with a NaN blend input (fill Min_Old, NPG_Old, "
                         "Assists_Old, Prior_Minutes, Seasons_Missed; check League and Pos):\n"
                         f"{bad_blend[['Player', 'Season', 'League'] + blend_inputs].to_string(index=False)}")

    # ---- new-player priors (CHANGELOG §103, §104) ----
    # For arrivals in the league groups that passed the out-of-time test (Adopted), the prior comes from
    # new_player_priors.csv (build_new_player_priors.py). Those values already carry their own fitted
    # new-team factor, so they replace the rating after the weight_team step above.
    new_priors_file = config.get("GOAL_MODEL_NEW_PRIORS_FILE") or paths.current_file('new_player_priors.csv', root=data)
    if new_priors_file.exists():
        npr = pd.read_csv(new_priors_file, encoding='utf-8-sig')
        npr = npr[npr['Adopted'].astype(str).str.lower() == 'true']
        assert not npr.duplicated(['Player', 'Season']).any(), "new_player_priors.csv repeats a Player/Season"
        npr = npr.set_index(['Player', 'Season'])[['Initial_NPG_Rating', 'Initial_Assist_Rating']]
        n_hit = 0
        for d in (initial_ratings_df, first_season_df):
            key = pd.MultiIndex.from_arrays([d['Player'], d['Season']])
            hit = np.asarray(key.isin(npr.index))
            d.loc[hit, ['Initial_NPG_Rating', 'Initial_Assist_Rating']] = npr.loc[key[hit]].to_numpy()
            n_hit = max(n_hit, int(hit.sum()))
        print(f"[§103] new-player priors from {new_priors_file.name}: {n_hit} transfers.csv rows")
    else:
        print(f"WARNING: {new_priors_file.name} missing -- new arrivals use the old transfers.csv priors "
              f"(run build_new_player_priors.py)")
    return InitialRatings(initial_ratings_df, first_season_df, second, youth_df)


def _pu_branch(notes, season, league, pos):
    """Entry type of a transfers.csv row (None = not an entrant), as validate_position_rates._branch."""
    notes = notes if isinstance(notes, str) else ""
    if not isinstance(pos, str) or "GK" in pos:
        return None
    if notes in ("second season", "youth", "missed season"):
        return notes
    if season > 2016 and league != "Premier League":
        return "arrival"
    return None


def entrant_rows(transfers_df) -> dict:
    """(Player, Season) -> (entry type or None, listed Pos) for the §249 position update."""
    rows = {}
    for second in (False, True):   # a 'second season' row wins a (player, season) tie, as initial_ratings_df's order
        for r in transfers_df.itertuples(index=False):
            if (r.Notes == 'second season') == second:
                rows[(r.Player, r.Season)] = (_pu_branch(r.Notes, r.Season, r.League, r.Pos), r.Pos)
    return rows


def transfer_groups(transfers_df) -> dict:
    """Player -> (seasons, position group of each transfers.csv row's first listed position), for the §235 floor."""
    tg = transfers_df[['Player', 'Season', 'Pos']].dropna().copy()
    tg['g'] = tg['Pos'].str.split(',').str[0].str.strip().map(_TRANSFER_GROUP)
    tg = tg.sort_values('Season', kind='stable')
    return {p: (g['Season'].to_numpy(), g['g'].to_numpy()) for p, g in tg.groupby('Player')}


def _transfer_group(groups, player, season):
    v = groups.get(player)
    if v is None:
        return None
    i = np.searchsorted(v[0], season, side='right') - 1
    return v[1][i] if i >= 0 and isinstance(v[1][i], str) else None


# ---------------------------------------------------------------------------------------------------------------------
# Step 4: the smoothing loop
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class LoopContext:
    priors: InitialRatings
    single_positions: pd.DataFrame      # indexed by Pos
    entrants: dict                      # entrant_rows()
    groups: dict                        # transfer_groups()
    role: RoleFactor | None = None
    move: MoveTable | None = None


def smooth_ratings_and_mae(player_stats, concede_lu, ctx: LoopContext):
    second_season_df, initial_ratings_df = ctx.priors.second_season, ctx.priors.initial
    pu_single, role, move = ctx.single_positions, ctx.role, ctx.move

    def pu_listed(pos):
        prior, _ = position_prior(pos, pu_single)
        return (prior['NPG/90'], prior['Assists/90'])

    ratings = {}
    total_npg_error = 0.0
    total_ast_error = 0.0
    established_npg_error = 0.0
    established_ast_error = 0.0
    total_weight = 0.0
    established_weight = 0.0
    player_stats["PriorMatches"] = player_stats.groupby("Player").cumcount()

    youth_keys = set(zip(ctx.priors.youth["Player"], ctx.priors.youth["Season"]))
    youth_start = set()                 # players whose rating started from a youth row (§126)
    prematch_rows = []
    role_p, role_l0 = {}, {}            # role factor (§172); inert unless GOAL_MODEL_ROLE_K_*
    move_last = {}                      # move factors (§173); inert unless GOAL_MODEL_MOVE_TABLE
    last_group = {}                     # player -> group of his last match with a position (§235)
    pos_update = {}                     # player -> entrant position-update state (§249)
    n_pu_shifted = 0

    for _, row in player_stats.iterrows():
        minutes = row.get("Minutes", 0)
        if minutes <= 0:
            continue
        pu_start = None                 # (entry type, Pos) when an entrant segment starts at this row (§249)
        player = row["Player"]
        venue = row["Venue"]
        opp = row["Opponent"]
        m_id = row["MatchID"]
        npxg90 = (row.get("npxG", 0.0) or 0.0) * 90.0 / minutes
        xag90 = (row.get("xAG", 0.0) or 0.0) * 90.0 / minutes
        season = row["Season"]

        if ((second_season_df["Player"] == player) & (second_season_df["Season"] == season)).any():
            player_season_matches = player_stats[(player_stats["Player"] == player) & (player_stats["Season"] == season)]
            # A returner with no earlier logged rating (his first-stint transfers.csv row missing) would
            # crash here; his "second season" row then simply serves as his starting rating (overnight 10-30).
            if m_id == player_season_matches.iloc[0]["MatchID"] and player not in ratings:
                print(f"WARNING: {player} ({season}) is a 'second season' row without an earlier rating; "
                      f"its prior is used as his starting rating")
            elif m_id == player_season_matches.iloc[0]["MatchID"]:
                prev_rating_npg, prev_rating_ast = ratings[player]
                initial_row = second_season_df[(second_season_df["Player"] == player)
                                               & (second_season_df["Season"] == season)]
                if not initial_row.empty:
                    initial_npg = initial_row["Initial_NPG_Rating"].values[0]
                    initial_ast = initial_row["Initial_Assist_Rating"].values[0]
                    prior_min = initial_row["Prior_Minutes"].values[0]
                    seasons_out = initial_row["Seasons_Missed"].values[0]
                    # RETURN_H minutes of discount per season away (§114); inf = the new-league prior only
                    w = 0.0 if np.isinf(RETURN_H) else prior_min / (prior_min + RETURN_H * seasons_out)
                    new_rating_npg = w * prev_rating_npg + (1 - w) * initial_npg
                    new_rating_ast = w * prev_rating_ast + (1 - w) * initial_ast
                    ratings[player] = (new_rating_npg, new_rating_ast)
                    pu_start = ("second season", initial_row["Pos"].values[0])

        size = row["PriorMatches"]
        # The starting rating counts as PRIOR_K matches; once 1/(size+K) falls to ALPHA the fixed rate
        # takes over. K = 5 reproduces the original rule (1/(size+5) below 35 matches) exactly; 16 since §112.
        k_prior = PRIOR_K_YOUTH if player in youth_start else PRIOR_K
        alpha_eff = max(1 / (size + k_prior), ALPHA) * (minutes / 90.0)

        venue_coefficient = home_advantage if venue == "Home" else (2 - home_advantage)
        opp_concede = concede_lu.get((season, row["Team"], opp, venue), AVG_CONCEDED)

        if player in ratings:
            g_rating, a_rating = ratings[player]
        else:
            # find the most recent transfers.csv row valid as of this point in
            # time (Season <= this match's season) -- not an exact-season match,
            # since a player's transfers.csv baseline season and their actual
            # first tracked match season don't always coincide
            candidates = initial_ratings_df[
                (initial_ratings_df['Player'] == player) & (initial_ratings_df['Season'] <= season)
            ].sort_values('Season')
            if candidates.empty:
                # no transfers.csv row valid as of this point in time
                continue
            g_rating = candidates['Initial_NPG_Rating'].values[-1]
            pu_start = ctx.entrants.get((player, candidates['Season'].values[-1]), (None, None))
            a_rating = candidates['Initial_Assist_Rating'].values[-1]
            if (player, candidates['Season'].values[-1]) in youth_keys:
                youth_start.add(player)
                alpha_eff = max(1 / (size + PRIOR_K_YOUTH), ALPHA) * (minutes / 90.0)

        pu = None
        if POS_UPDATE:
            if pu_start is not None:
                if pu_start[0] is not None:
                    pos_update[player] = {"s0": int(row["_PosSeason"]), "pl": pu_listed(pu_start[1]), "s": [1.0, 1.0],
                                          "d": [0.0, 0.0], "sw": 0.0, "swr": [0.0, 0.0]}
                else:
                    pos_update.pop(player, None)
            pu = pos_update.get(player)
            if pu is not None and int(row["_PosSeason"]) > pu["s0"] + 1:
                del pos_update[player]   # past his first two seasons: the shift stays, it stops moving
                pu = None
            if pu is not None:
                d = [(POS_UPDATE_KPOS * pu["pl"][k] + pu["swr"][k]) / (POS_UPDATE_KPOS + pu["sw"]) - pu["pl"][k]
                     for k in (0, 1)]
                g_rating += POS_UPDATE_C[0] * pu["s"][0] * (d[0] - pu["d"][0])
                a_rating += POS_UPDATE_C[1] * pu["s"][1] * (d[1] - pu["d"][1])
                n_pu_shifted += d != pu["d"]
                pu["d"] = d

        # Per-90 predictions, compared with npxg90/xag90: the rating is a per-90 rate, as the
        # forecast uses it (x xMins/90). These used to carry an extra x 90/minutes, which made
        # the loop learn per-appearance rates and under-rate part-time players (CHANGELOG §86, §87).
        # (Kept as one left-to-right product: regrouping the factors changes the floating-point result.)
        up_g, up_a = sub_uplift("npxG", row["_IsStarter"], minutes), sub_uplift("xAG", row["_IsStarter"], minutes)
        pred_goals_90 = g_rating * venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_NPG * up_g
        pred_ast_90 = a_rating * venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_AST * up_a
        weight = minutes / 90.0
        role_extra = ()
        if role is not None:
            rec = role.table.get((int(row["_LogSeason"]), row["Team"], opp, venue, player))
            p_now, line_now = role_p.get(player), None
            if rec is not None:
                p_now, line_now, role_p[player] = rec[0], rec[1], rec[2]
                role_l0.setdefault(player, line_now)
            m1_g = m1_a = o1_g = o1_a = 1.0
            if player in role_l0 and rec is not None and rec[4] == 0:
                pass    # first start: no history yet, so the honest factor is 1
            elif player in role_l0 and p_now is not None:
                l0 = role_l0[player]
                m1_g = float(p_now @ role.rate_npg / role.rate_npg[l0])
                m1_a = float(p_now @ role.rate_ast / role.rate_ast[l0])
                o1_g = role.rate_npg[line_now] / role.rate_npg[l0] if line_now is not None else m1_g
                o1_a = role.rate_ast[line_now] / role.rate_ast[l0] if line_now is not None else m1_a
            h_g = _role_mult(m1_g, role.k_npg_up, role.k_npg_down)
            h_a = _role_mult(m1_a, role.k_ast_up, role.k_ast_down)
            pred_goals_90 *= _role_mult(o1_g, role.k_npg_up, role.k_npg_down)
            pred_ast_90 *= _role_mult(o1_a, role.k_ast_up, role.k_ast_down)
            role_extra = (pred_goals_90 * h_g / _role_mult(o1_g, role.k_npg_up, role.k_npg_down),
                          pred_ast_90 * h_a / _role_mult(o1_a, role.k_ast_up, role.k_ast_down), m1_g, m1_a)

        if size > 10:
            established_npg_error += abs(npxg90 - pred_goals_90) * weight
            established_ast_error += abs(xag90 - pred_ast_90) * weight
            established_weight += weight
        total_npg_error += abs(npxg90 - pred_goals_90) * weight
        total_ast_error += abs(xag90 - pred_ast_90) * weight
        total_weight += weight

        if PREMATCH_LOG:
            prematch_rows.append((player, row["Team"], opp, venue, int(m_id), row["Data_Source"],
                                  int(row["_LogSeason"]), float(g_rating), float(a_rating),
                                  float(npxg90), float(pred_goals_90), float(xag90),
                                  float(pred_ast_90), float(weight),
                                  float(opp_concede / AVG_CONCEDED), str(row["_IsStarter"])) + role_extra)

        grp = _MATCH_GROUP.get(row["_MatchPos"]) if isinstance(row["_MatchPos"], str) else None
        if grp is None:
            grp = last_group.get(player, _transfer_group(ctx.groups, player, row["_PosSeason"]))
        if grp is not None:
            last_group[player] = grp
        # same schedule as alpha_eff, with the npxG floor (§233; centre-backs §235)
        k_used = PRIOR_K_YOUTH if player in youth_start else PRIOR_K
        alpha_npg = max(1 / (size + k_used), ALPHA_NPG_CB if grp == "CB" else ALPHA_NPG) * (minutes / 90.0)
        upd_g, upd_a = pred_goals_90 > 0, pred_ast_90 > 0
        if pred_goals_90 > 0:
            ratio = npxg90 / pred_goals_90
            g_rating *= 1 + alpha_npg * (ratio - 1)
        if pred_ast_90 > 0:
            ratio = xag90 / pred_ast_90
            a_rating *= 1 + alpha_eff * (ratio - 1)
        if pu is not None:
            # A shift can take a rating to 0 or below (an entrant listed far above the slots he plays). The update
            # above is r + alpha x (actual / multiplier - r) whenever the prediction is positive; apply that same
            # form here too, so the rating keeps learning and the loop stays the linear rule §247's replay scored.
            m_g = venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_NPG * up_g
            m_a = venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_AST * up_a
            if not upd_g and g_rating <= 0 and m_g > 0:
                g_rating += alpha_npg * (npxg90 / m_g - g_rating)
                upd_g = True
            if not upd_a and a_rating <= 0 and m_a > 0:
                a_rating += alpha_eff * (xag90 / m_a - a_rating)
                upd_a = True
            pu["s"][0] *= 1 - (alpha_npg if upd_g else 0.0)
            pu["s"][1] *= 1 - (alpha_eff if upd_a else 0.0)
            slot = _PU_SLOT.get(row["_MatchPos"]) if isinstance(row["_MatchPos"], str) else None
            if slot is not None:
                pu["sw"] += weight
                pu["swr"][0] += weight * pu_single.at[slot, 'NPG/90']
                pu["swr"][1] += weight * pu_single.at[slot, 'Assists/90']
        if move is not None:
            mv = move.keys.get((int(row["_LogSeason"]), row["Team"], opp, venue, player))
            if mv is not None:
                last = move_last.get(player)
                if last is not None and last[0] == mv["new"]:   # back to the line before the last move
                    g_rating, a_rating = g_rating / last[1], a_rating / last[2]
                    del move_last[player]
                else:
                    d = "forward" if mv["fwd"] else "back"
                    fg, fa = move.factors["npxG"][d], move.factors["xAG"][d]
                    g_rating, a_rating = g_rating * fg, a_rating * fa
                    move_last[player] = (mv["ref"], fg, fa)
        if not (np.isfinite(g_rating) and np.isfinite(a_rating)):
            raise SystemExit(f"NaN rating for {player} ({season} v {opp}): NPG {g_rating}, assists {a_rating}; "
                             f"check his transfers.csv rows (§185)")
        ratings[player] = (g_rating, a_rating)

    mae_npg = total_npg_error / total_weight if total_weight else 0.0
    mae_ast = total_ast_error / total_weight if total_weight else 0.0
    print(f"MAE npxG: {mae_npg:.4f}  MAE xAG: {mae_ast:.4f}")
    if POS_UPDATE:
        print(f"[§249] entrant position update on (c npxG {POS_UPDATE_C[0]}, xAG {POS_UPDATE_C[1]}, Kpos "
              f"{POS_UPDATE_KPOS:g}): {n_pu_shifted} rows shifted")
    if PREMATCH_LOG:
        pd.DataFrame(prematch_rows, columns=[
            "Player", "Team", "Opponent", "Venue", "MatchID", "Data_Source", "Season",
            "NPG_Rating_Pre", "Assist_Rating_Pre", "npxg90", "pred_goals_90", "xag90",
            "pred_ast_90", "weight", "Opp_Factor", "Is_Starter"]
            + (["pred_goals_90_h", "pred_ast_90_h", "M1_npg", "M1_ast"] if role is not None else [])).to_csv(
            PREMATCH_LOG, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(prematch_rows)} pre-match rating rows to {PREMATCH_LOG}")
    return ratings, mae_npg, mae_ast


# ---------------------------------------------------------------------------------------------------------------------
# Step 5: the output table
# ---------------------------------------------------------------------------------------------------------------------
def final_ratings_frame(final_ratings, inp: Inputs, playerstats, priors: InitialRatings) -> pd.DataFrame:
    """Clamp, fall back to starting ratings for players never seen, adjust club changers, apply the conversion rate."""
    players_current = inp.players_current
    # A position-update shift can leave a rating below 0 mid-segment (§249); a forecast rate never is.
    neg = [p for p, (g, a) in final_ratings.items() if g < 0 or a < 0]
    if neg:
        print(f"[§249] {len(neg)} final ratings below 0 set to 0: {', '.join(sorted(neg)[:10])}")
        final_ratings = {p: (max(g, 0.0), max(a, 0.0)) for p, (g, a) in final_ratings.items()}

    # ---- new players not seen at all: fall back to their transfers.csv initial rating ----
    initial_ratings_dict = priors.first_season.set_index(["Player", "Season"])[
        ["Initial_NPG_Rating", "Initial_Assist_Rating"]].to_dict("index")
    current_player_names = set(players_current['Player'])
    for player in current_player_names:
        if player not in final_ratings:
            # a returning/new player may have multiple transfers.csv stints -- use
            # the most recent Season available for them
            matches = [k for k in initial_ratings_dict if k[0] == player]
            if matches:
                latest_key = max(matches, key=lambda k: k[1])
                init = initial_ratings_dict[latest_key]
                final_ratings[player] = (init["Initial_NPG_Rating"], init["Initial_Assist_Rating"])

    # ---- team-change adjustment for players who've since moved clubs ----
    team_ratings_now = inp.team_ratings_now
    for player, (g_rating, a_rating) in list(final_ratings.items()):
        if player not in current_player_names:
            continue
        data = playerstats[playerstats['Player'] == player]
        if len(data) == 0:
            continue
        prev_team = data.tail(1).Team.values[0]
        curr_team_row = players_current[players_current['Player'] == player]
        if curr_team_row.empty:
            continue
        curr_team = curr_team_row['Team'].values[0]
        is_2025_new = player in priors.initial.loc[priors.initial['Season'] == 2025, 'Player'].values
        if prev_team != curr_team and not is_2025_new:
            prev_team_atk_rating = data.tail(1).Team_Attack_Rating.values[0]
            curr_team_row_rating = team_ratings_now[team_ratings_now['Team'] == curr_team]
            if curr_team_row_rating.empty:
                continue
            curr_team_atk_rating = curr_team_row_rating['Attack_Rating'].values[0]
            new_g_rating = g_rating + 0.031 * (curr_team_atk_rating - prev_team_atk_rating) + 0.001
            new_a_rating = a_rating + 0.026 * (curr_team_atk_rating - prev_team_atk_rating) + 0.002
            if new_g_rating <= 0:
                new_g_rating = g_rating / 2
            if new_a_rating <= 0:
                new_a_rating = a_rating / 2
            final_ratings[player] = (new_g_rating, new_a_rating)

    filtered_final_ratings = {p: r for p, r in final_ratings.items() if p in current_player_names}

    # ---- shrinkage-corrected conversion rate, replacing Goal_Conversion_Rate ----
    career_totals = playerstats.groupby('Player').agg(
        career_NPG=('Goals', lambda x: x.sum()),
        career_PK=('PK', 'sum'),
        career_npxG=('npxG', 'sum'),
    ).reset_index()
    career_totals['career_NPG'] = career_totals['career_NPG'] - career_totals['career_PK']
    career_totals['Conversion'] = ((career_totals['career_NPG'] + SHRINKAGE_PRIOR)
                                   / (career_totals['career_npxG'] + SHRINKAGE_PRIOR))
    conversion_lookup = dict(zip(career_totals['Player'], career_totals['Conversion']))

    # ---- diagnostic: check for missing Team_Attack_Rating in transfers.csv ----
    missing_atk = inp.transfers[inp.transfers['Team_Attack_Rating'].isna()]
    if len(missing_atk) > 0:
        print(f"\nWARNING: {len(missing_atk)} rows in transfers.csv have a BLANK Team_Attack_Rating:")
        print(missing_atk[['Player', 'Season', 'Notes']].to_string(index=False))
        print()

    out_rows = []
    for p, (g, a) in filtered_final_ratings.items():
        conversion = conversion_lookup.get(p, 1.0)
        out_rows.append({'Player': p, 'NPG/90': g * conversion, 'Assists/90': a})

    out_df = pd.DataFrame(out_rows)
    out_df = out_df.merge(players_current[['Player', 'Web_Name', 'Pos', 'Team', 'Price']], on='Player', how='left')
    out_df['Player'] = out_df['Web_Name']
    out_df = out_df[['Player', 'Pos', 'Team', 'Price', 'NPG/90', 'Assists/90']]
    out_df = out_df.sort_values('NPG/90', ascending=False)
    still_null = out_df[out_df[['NPG/90', 'Assists/90']].isna().any(axis=1)]
    if len(still_null):
        raise SystemExit(f"NaN ratings, nothing written (§185):\n"
                         f"{still_null[['Player', 'Team']].to_string(index=False)}")
    return out_df


def write_ratings(out_df: pd.DataFrame, data: Path = _DATA) -> None:
    if PREMATCH_LOG:
        out_df.to_csv(PREMATCH_LOG + ".ratings.csv", index=False, encoding='utf-8-sig')
        print(f"\nGOAL_MODEL_PREMATCH_LOG set: ratings written to {PREMATCH_LOG}.ratings.csv, "
              f"not player_goal_assist_ratings.csv")
    else:
        out_df.to_csv(paths.current_file('player_goal_assist_ratings.csv', root=data), index=False, encoding='utf-8-sig')
        print(f"\nWrote {len(out_df)} players to player_goal_assist_ratings.csv")


def main() -> None:
    role, move = load_research_hooks()
    inp = load_inputs()
    playerstats = prepare_player_stats(inp.player_stats, inp.matchlogs)
    concede_lookup = build_concede_lookup(inp.matchlogs)
    priors = build_initial_ratings(inp.transfers, inp.positions, inp.leagues)
    ctx = LoopContext(priors=priors, single_positions=inp.single_positions.set_index('Pos'),
                      entrants=entrant_rows(inp.transfers), groups=transfer_groups(inp.transfers),
                      role=role, move=move)
    final_ratings, _, _ = smooth_ratings_and_mae(playerstats, concede_lookup, ctx)
    write_ratings(final_ratings_frame(final_ratings, inp, playerstats, priors))


if __name__ == "__main__":
    main()
