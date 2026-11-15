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

Run locally:
    python build_player_ratings.py
"""
import sys
import os
from collections import deque, defaultdict, Counter

import pandas as pd
import numpy as np
from pathlib import Path

_DATA = Path(__file__).resolve().parents[2] / "data" / "current"

# ---- constants (from the original pipeline) ----
minutes_constant = 500
league_coefficient_multiplier = 0.7
youth_coefficient = 0.75
ALPHA = float(os.environ.get("GOAL_MODEL_ALPHA", "0.025"))  # learning-rate floor; env for experiments (§196)
# npxG has its own floor since §233 (0.035; GOAL_MODEL_ALPHA_NPG=0.025 reverts). Without GOAL_MODEL_ALPHA_NPG, an
# experiment that sets GOAL_MODEL_ALPHA moves both floors, as it did before §233.
ALPHA_NPG = float(os.environ.get("GOAL_MODEL_ALPHA_NPG", os.environ.get("GOAL_MODEL_ALPHA", "0.035")))
# Centre-backs' npxG floor since §235 (0.018, user decision; GOAL_MODEL_ALPHA_NPG_CB=0.035 reverts). A centre-back row
# = the match's Position_Resolved is CB; a row without a position takes the player's previous match group, else his
# transfers.csv group (first listed position), as validate_kalman_smoothing.load(). When an experiment sets
# GOAL_MODEL_ALPHA or GOAL_MODEL_ALPHA_NPG without this, centre-backs follow ALPHA_NPG (pre-§235 behaviour).
_CB_DEFAULT = (str(ALPHA_NPG) if ("GOAL_MODEL_ALPHA_NPG" in os.environ or "GOAL_MODEL_ALPHA" in os.environ)
               else "0.018")
ALPHA_NPG_CB = float(os.environ.get("GOAL_MODEL_ALPHA_NPG_CB", _CB_DEFAULT))
AVG_CONCEDED = 1.32023
from fpl_model.league_rates import current_season  # noqa: E402
from fpl_model.match_order import chrono_order  # noqa: E402
from fpl_model.team_ratings.update_team_ratings import home_advantage  # noqa: E402
SHRINKAGE_PRIOR = 30  # the new conversion-rate correction's pseudo-count
RETURN_H = float(os.environ.get("GOAL_MODEL_RETURN_H", "5000"))  # second-season blend discount (§114)
PRIOR_K = float(os.environ.get("GOAL_MODEL_PRIOR_K", "16"))  # matches the starting rating counts as (§112; 5 = pre-§112)
# Players whose rating starts from a `youth` row (§126); defaults to PRIOR_K (no separate weight).
PRIOR_K_YOUTH = float(os.environ.get("GOAL_MODEL_PRIOR_K_YOUTH", str(PRIOR_K)))

# ---- role-change detection (CHANGELOG §39, AGENDA §4 (C)) ----
# Feeds the Understat per-match formation slots shipped into defcon (§38) into
# this goal/assist model: on a *sustained* slot change the smoothing loop opens a
# short high-learning-rate window so the rating re-converges to the new role
# instead of dragging the old role's rating forward (a recency reset, not a
# bounce -- the treatment cold-start data sources.md §4 recommends for structural
# breaks). Default OFF: GOAL_MODEL_UNDERSTAT_POS unset/0 reproduces the pre-§39
# output bit-for-bit. Mirrors defcon's DEFCON_UNDERSTAT_POS gate.
UNDERSTAT_POS_GOAL = os.environ.get("GOAL_MODEL_UNDERSTAT_POS", "0") == "1"
ROLE_APPLY = os.environ.get("GOAL_ROLE_APPLY", "1") == "1"   # 0 = detect+flag only
ROLE_DIAG = os.environ.get("GOAL_MODEL_DIAG", "0") == "1"    # dump per-row diagnostics
ROLE_W = int(os.environ.get("GOAL_ROLE_W", "5"))             # consecutive matches to call it sustained
ROLE_MIN_MATCHES = int(os.environ.get("GOAL_ROLE_MIN_MATCHES", "8"))  # matches to establish the reference slot
ROLE_COOLDOWN = int(os.environ.get("GOAL_ROLE_COOLDOWN", "10"))       # silence between fires (in PriorMatches)
ROLE_FACTOR = float(os.environ.get("GOAL_ROLE_FACTOR", "1.5"))       # attacking-prior ratio gate within a coarse group
ROLE_BOOST_ALPHA = float(os.environ.get("GOAL_ROLE_BOOST_ALPHA", "0.15"))  # learning rate during the boost window
ROLE_WINDOW = int(os.environ.get("GOAL_ROLE_WINDOW", "8"))   # boosted matches after a fire
ROLE_DIAG_OUT = os.environ.get("GOAL_DIAG_OUT", "_role_diag.csv")

# ---- pre-match rating log (CHANGELOG §72, §74) ----
# When set to a path, every rated row's goal/assist rating *before* that match's
# update is written there with its one-step-ahead prediction and actual, and the
# would-be ratings go to <path>.ratings.csv instead of player_goal_assist_ratings.csv.
# Unset (production) leaves the output unchanged.
PREMATCH_LOG = os.environ.get("GOAL_MODEL_PREMATCH_LOG", "")

# ---- role factor (position plan step 3, CHANGELOG §172) ----
# Experiment only: unset = no effect. Rating stays on the scale of the player's first logged line;
# the loop divides by (line rate / first-line rate)^k for the line played, and the log also records
# the honest prediction from role percentages. Refuses to run without PREMATCH_LOG, so never production.
_ROLE_K_ENV = ["GOAL_MODEL_ROLE_K_NPG_UP", "GOAL_MODEL_ROLE_K_NPG_DOWN",
               "GOAL_MODEL_ROLE_K_AST_UP", "GOAL_MODEL_ROLE_K_AST_DOWN"]
ROLE_K_ON = any(v in os.environ for v in _ROLE_K_ENV)
if ROLE_K_ON:
    if not PREMATCH_LOG:
        raise SystemExit("GOAL_MODEL_ROLE_K_* needs GOAL_MODEL_PREMATCH_LOG: it must not reach production")
    import json
    from fpl_model.player_ratings.role_shares import LINE_RATES_FILE, role_table
    K_NPG_UP, K_NPG_DOWN, K_AST_UP, K_AST_DOWN = [float(os.environ.get(v, "0")) for v in _ROLE_K_ENV]
    _rates = json.loads(LINE_RATES_FILE.read_text(encoding="utf-8"))
    RATE_NPG, RATE_AST = np.array(_rates["npxG"]), np.array(_rates["xAG"])
    ROLE_TABLE = role_table()


# ---- same-player move factors (user's form of the position plan, CHANGELOG §173) ----
# Experiment only: unset = no effect. One-off rating multiply when a move is detected, divided out on
# moving back. Refuses to run without PREMATCH_LOG, so never production.
MOVE_TABLE_FILE = os.environ.get("GOAL_MODEL_MOVE_TABLE", "")
if MOVE_TABLE_FILE:
    if not PREMATCH_LOG:
        raise SystemExit("GOAL_MODEL_MOVE_TABLE needs GOAL_MODEL_PREMATCH_LOG: it must not reach production")
    import json as _json
    _mt = _json.loads(Path(MOVE_TABLE_FILE).read_text(encoding="utf-8"))
    MOVE_F = _mt["factors"]
    MOVE_KEYS = {tuple(k): v for k, v in _mt["triggers"]}


def _role_mult(m1, up, down):
    return m1 ** (up if m1 > 1 else down)

# Canonicalise the fbref-style resolved slot onto position_combinations.csv's
# single-slot vocabulary, so lateral label churn (a full-back alternating LB/RB)
# is not mistaken for a role change.
SLOT_CANON = {
    "CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB",
    "DM": "DM", "CM": "CM", "AM": "AM",
    "LM": "WM", "RM": "WM", "LW": "W", "RW": "W",
    "FW": "FW", "GK": "GK",
}
SLOT_COARSE = {
    "CB": "DEF", "FB": "DEF", "WB": "DEF",
    "DM": "MID", "CM": "MID", "AM": "MID", "WM": "MID", "W": "MID",
    "FW": "FWD", "GK": "GK",
}
SLOT_PRIOR = {}  # canonical slot -> (NPG/90, Assists/90); filled from positions_df below

# ---- load inputs ----
transfers_df = pd.read_csv(_DATA / 'transfers.csv')
for col in ['Prior_Minutes', 'Seasons_Missed', 'Team_Attack_Rating', 'Min_Old', 'npxG_Old', 'xAG_Old', 'NPG_Old', 'Assists_Old']:
    if col in transfers_df.columns:
        transfers_df[col] = pd.to_numeric(transfers_df[col], errors='coerce')
# ---- position priors (CHANGELOG §77) ----
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


_single_positions = pd.read_csv(_DATA / 'position_combinations.csv')
if _single_positions['Pos'].str.contains(',').any():
    raise ValueError("position_combinations.csv must hold single positions only; "
                     "combinations are computed (CHANGELOG §77)")
positions_df = build_position_priors(_single_positions, transfers_df['Pos'].dropna())
# §246 experiment: GOAL_MODEL_POSITION_RATES = slots | slots3 replaces the single-position rates, per transfers.csv
# row, by PL starts' slot rates in the seasons before the row's season (position_rates.py). Default "file" = the file.
from fpl_model.player_ratings import position_rates as _pos_rates  # noqa: E402
if _pos_rates.WINDOW is not None:
    print(f"[§246] position rates from per-match slots, window {_pos_rates.WINDOW} (min {_pos_rates.MIN_STARTS} starts)")
leagues_df = pd.read_csv(_DATA / 'league_coefficients.csv')
matchlogs = pd.read_csv(_DATA / 'team_ratings_history.csv')
team_ratings_2026 = pd.read_csv(_DATA / f'team_ratings_preseason_{current_season()}.csv')
players_current = pd.read_csv(_DATA / 'players_current.csv', encoding='utf-8-sig')

# ---- load consolidated player match stats (fbref 2017-2024 + FPL-API 2025/26+) ----
# player_match_stats.csv is already harmonized: consistent MatchID scheme, NPG
# computed, CBI computed for both sources. No manual schema reconciliation needed.
playerstats = pd.read_csv(_DATA / 'player_match_stats.csv', encoding='utf-8-sig', low_memory=False)

# ---- red-card adjustment of player npxG / xAG (§136): the team-level factors, per player from his own shots ----
# Keyed on the native Season (before the overwrite below). The data files stay unadjusted.
RED_CARD_ON = os.environ.get("GOAL_MODEL_RED_CARD", "1") == "1"   # 0 reverts (§136)
RED_CARD_FILE = Path(os.environ.get("GOAL_MODEL_RED_CARD_FILE", _DATA / 'red_card_player_factors.csv'))
if RED_CARD_ON and RED_CARD_FILE.exists():
    _rc = pd.read_csv(RED_CARD_FILE, encoding='utf-8-sig')
    _key = ['Season', 'Player', 'Team', 'Opponent', 'Venue']
    _f = playerstats[_key].merge(_rc[_key + ['Factor_npxG', 'Factor_xAG']], on=_key, how='left')
    assert len(_f) == len(playerstats)
    playerstats['npxG'] = playerstats['npxG'] * _f['Factor_npxG'].fillna(1.0).to_numpy()
    playerstats['xAG'] = playerstats['xAG'] * _f['Factor_xAG'].fillna(1.0).to_numpy()
    print(f"[§136] red-card factors applied to {int(_f['Factor_npxG'].notna().sum())} player-match rows")

# ---- role-change (§39): merge the Understat per-match resolved slot ----
# Must run BEFORE the Season overwrite below: player_match_positions.csv is keyed
# on player_match_stats' NATIVE Season (2025 and 2026 distinct), while the overwrite
# collapses every FPL_API row to 2025 (MatchID//380 is a stale derivation, §9). The
# join key is (native Season, Team, Opponent, Venue, Player) -- never MatchID, which
# cannot join the FPL era (crosswalk docstring; CLAUDE.md §2.5).
if UNDERSTAT_POS_GOAL:
    _pc_single = positions_df[~positions_df['Pos'].astype(str).str.contains(',')]
    SLOT_PRIOR.update({r['Pos']: (float(r['NPG/90']), float(r['Assists/90']))
                       for _, r in _pc_single.iterrows()})
    _pos_file = _DATA / 'player_match_positions.csv'
    if _pos_file.exists():
        playerstats['_NativeSeason'] = playerstats['Season']
        _posdf = pd.read_csv(_pos_file, encoding='utf-8-sig', low_memory=False)
        _posdf = _posdf.dropna(subset=['Position_Resolved'])
        _n_before = len(playerstats)
        playerstats = playerstats.merge(
            _posdf[['Season', 'Team', 'Opponent', 'Venue', 'Player', 'Position_Resolved']],
            left_on=['_NativeSeason', 'Team', 'Opponent', 'Venue', 'Player'],
            right_on=['Season', 'Team', 'Opponent', 'Venue', 'Player'],
            how='left', suffixes=('', '_pos'))
        assert len(playerstats) == _n_before, \
            f"role-change merge fanned out rows ({_n_before} -> {len(playerstats)}); " \
            f"player_match_positions.csv is not unique on the 5-key"
        if 'Season_pos' in playerstats.columns:
            playerstats = playerstats.drop(columns=['Season_pos'])
        playerstats['_CanonSlot'] = playerstats['Position_Resolved'].map(SLOT_CANON)
        _got = int(playerstats['_CanonSlot'].notna().sum())
        print(f"[role-change S39] Position_Resolved merged onto {_got}/{len(playerstats)} "
              f"rows ({_got / len(playerstats):.1%}); ROLE_APPLY={int(ROLE_APPLY)}")
    else:
        print(f"WARNING: GOAL_MODEL_UNDERSTAT_POS=1 but {_pos_file.name} is missing -- "
              f"role-change detection disabled (run build_position_crosswalk.py).")
        UNDERSTAT_POS_GOAL = False

# Substitute appearances are learned on a starter-equivalent scale (CHANGELOG §100, §101): their
# per-90 prediction carries the frozen uplift for their minutes band. Start flag: Understat Is_Starter.
SUB_UPLIFT = {"npxG": [(15, 1.663), (30, 1.277), (45, 1.170), (999, 0.868)],
              "xAG": [(15, 1.685), (30, 1.276), (45, 1.113), (999, 1.045)]}
_starts = pd.read_csv(_DATA / 'player_match_positions.csv', encoding='utf-8-sig', low_memory=False,
                      usecols=['Season', 'Team', 'Opponent', 'Venue', 'Player', 'Is_Starter', 'Position_Resolved'])
_starts = _starts.drop_duplicates(['Season', 'Team', 'Opponent', 'Venue', 'Player'])
_n_before = len(playerstats)
playerstats['_PosSeason'] = playerstats['Season']   # native season, the key of the position file
playerstats = playerstats.merge(_starts.rename(columns={'Season': '_StartSeason', 'Is_Starter': '_IsStarter',
                                                        'Position_Resolved': '_MatchPos'}),
                                left_on=['Season', 'Team', 'Opponent', 'Venue', 'Player'],
                                right_on=['_StartSeason', 'Team', 'Opponent', 'Venue', 'Player'], how='left')
assert len(playerstats) == _n_before
playerstats = playerstats.drop(columns='_StartSeason')
_is_sub = playerstats['_IsStarter'].astype(str).str.lower().eq('false')
print(f"[§101] start flag on {playerstats['_IsStarter'].notna().mean():.1%} of rows; substitute rows {int(_is_sub.sum())}")

# §242: FPL-era xAG (Opta xA, 2025+) put on the fbref xAG scale by match slot, so the loop's assist target doesn't change
# model at 2025. Factors from the 2022-2024 overlap, computed each run (xag_source_scale.py). A row without a slot takes
# the player's most common FPL-era slot, else the overall factor. Default off until the user decides (CHANGELOG §242).
XAG_SLOT_SCALE = os.environ.get("GOAL_MODEL_XAG_SLOT_SCALE", "0") == "1"
# §243-§244: expected assists calibrated to conventional assists by match slot. Adopted by user decision 2026-11-13:
# default fw_cb; GOAL_MODEL_XAG_CALIBRATION=off reverts. Season-end check frozen in
# data/validation/xag_calibration_seasonend_2027/protocol.json.
#   fw_cb: FPL-era (Opta) xA x assists/Opta xA for forwards and centre-backs only (factors from 2022-2024)
#   all:   the same for every slot (not recommended, §243)
#   both:  every row, fbref xAG x assists/fbref xAG (2017-2024) and Opta xA x assists/Opta xA (2022-2024) (not recommended)
XAG_CALIBRATION = os.environ.get("GOAL_MODEL_XAG_CALIBRATION", "fw_cb")
if XAG_CALIBRATION not in ("off", "fw_cb", "all", "both"):
    raise SystemExit(f"GOAL_MODEL_XAG_CALIBRATION must be off, fw_cb, all or both, not {XAG_CALIBRATION!r}")
if XAG_SLOT_SCALE and XAG_CALIBRATION != "off":
    raise SystemExit("GOAL_MODEL_XAG_SLOT_SCALE and GOAL_MODEL_XAG_CALIBRATION are alternatives; set one")
if XAG_SLOT_SCALE or XAG_CALIBRATION != "off":
    from fpl_model.player_ratings.xag_source_scale import CANON as _XS_CANON, slot_factors as _xs_factors, assist_factors as _xs_assist
    _fpl = playerstats['Data_Source'].eq('FPL_API').to_numpy()
    _slot = playerstats['_MatchPos'].map(_XS_CANON)
    _modal = (playerstats.assign(_s=_slot).dropna(subset=['_s'])
              .groupby('Player')['_s'].agg(lambda s: s.value_counts().index[0]))
    _slot = _slot.fillna(playerstats['Player'].map(_modal))

    def _factors_for(table, slots=None):
        f = _slot.map(table).fillna(table['overall'])
        if slots is not None:
            f = f.where(_slot.isin(slots), 1.0)
        return f.to_numpy()

    if XAG_SLOT_SCALE:
        _xs = _xs_factors(_DATA)
        playerstats['xAG'] = np.where(_fpl, playerstats['xAG'] * _factors_for(_xs), playerstats['xAG'])
        print(f"[§242] FPL-era xAG put on the fbref scale on {int(_fpl.sum())} rows; factors "
              + ", ".join(f"{k} {v:.3f}" for k, v in sorted(_xs.items())))
    else:
        _opta = _xs_assist(_DATA, "opta")
        _keep = ["FW", "CB"] if XAG_CALIBRATION == "fw_cb" else None
        _new = np.where(_fpl, playerstats['xAG'] * _factors_for(_opta, _keep), playerstats['xAG'])
        if XAG_CALIBRATION == "both":
            _fb = _xs_assist(_DATA, "fbref")
            _new = np.where(_fpl, _new, playerstats['xAG'] * _factors_for(_fb))
        playerstats['xAG'] = _new
        print(f"[§243] xAG calibrated to assists ({XAG_CALIBRATION}); Opta factors "
              + ", ".join(f"{k} {v:.3f}" for k, v in sorted(_opta.items())))


SUB_UPLIFT_ON = os.environ.get("GOAL_MODEL_SUB_UPLIFT", "1") == "1"  # 0 restores the §87 loop


def sub_uplift(kind, is_starter, minutes):
    if not SUB_UPLIFT_ON or str(is_starter).lower() != 'false':
        return 1.0
    return next(u for upper, u in SUB_UPLIFT[kind] if minutes <= upper)


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
_home_atk = matchlogs[['Season', 'Home', 'Away', 'Home_Attack_Rating']].rename(
    columns={'Home': 'Team', 'Away': 'Opponent', 'Home_Attack_Rating': 'Team_Attack_Rating'})
_home_atk['Venue'] = 'Home'
_away_atk = matchlogs[['Season', 'Away', 'Home', 'Away_Attack_Rating']].rename(
    columns={'Away': 'Team', 'Home': 'Opponent', 'Away_Attack_Rating': 'Team_Attack_Rating'})
_away_atk['Venue'] = 'Away'
_team_atk_lookup = pd.concat([_home_atk, _away_atk], ignore_index=True).drop_duplicates(
    subset=['Season', 'Team', 'Opponent', 'Venue'])

_atk_fill = playerstats[['Season', 'Team', 'Opponent', 'Venue']].merge(
    _team_atk_lookup, on=['Season', 'Team', 'Opponent', 'Venue'], how='left')
playerstats['Team_Attack_Rating'] = _atk_fill['Team_Attack_Rating']

missing_atk = playerstats['Team_Attack_Rating'].isna().sum()
if missing_atk > 0:
    print(f"WARNING: {missing_atk} of {len(playerstats)} rows still have no "
          f"Team_Attack_Rating after the lookup -- check Team/Opponent/Venue spelling "
          f"consistency against team_ratings_history.csv")

playerstats = playerstats.assign(_Chrono=chrono_order(playerstats, _DATA)).sort_values('_Chrono').reset_index(drop=True)

# ---- build concede lookup ----
# Since §110 the loop's multiplier is the one the forecast applies (build_simulation_features.py):
# the opponent's pre-match Defense_Rating / that season's mean, keyed on (Season, Team, Opponent, Venue).
# It used to be the player's team's predicted goals keyed on MatchID: in 2017-2024 that also divided
# out his own team's attack (inflating weak-attack players), and FPL-era MatchIDs don't match, so 2025+
# rows got ~1 (CHANGELOG §78, §79, §110). GOAL_MODEL_OPP_DEFENCE=0 restores it.
OPP_DEFENCE = os.environ.get("GOAL_MODEL_OPP_DEFENCE", "1") == "1"
# Dampening exponent on the opponent factor, per target (§123, adopted 0.7 / 0.8); 1 = the plain multiplier.
OPP_BETA_NPG = float(os.environ.get("GOAL_MODEL_OPP_BETA_NPG", "0.7"))
OPP_BETA_AST = float(os.environ.get("GOAL_MODEL_OPP_BETA_AST", "0.8"))


def build_concede_lookup(matchlogs):
    concede = {}
    if OPP_DEFENCE:
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
    for _, row in matchlogs.iterrows():
        m_id = row["MatchID"]
        home_team = row["Home"]
        away_team = row["Away"]
        # The team prediction already contains update_team_ratings.py's home factor
        # (same value); strip it so the loop's venue_coefficient applies home/away
        # once, not twice (CHANGELOG §83, §84)
        concede[(m_id, home_team)] = row["Away_Prediction"] / (2 - home_advantage)
        concede[(m_id, away_team)] = row["Home_Prediction"] / home_advantage
    return concede

concede_lookup = build_concede_lookup(matchlogs)

# ---- build initial_ratings_df from all six transfers.csv branches ----
gk_df = transfers_df[(transfers_df['Pos'] == 'GK') & (transfers_df['Notes'] != 'second season')]
gk_df = gk_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
gk_df['Initial_NPG_Rating'] = gk_df['NPG/90']
gk_df['Initial_Assist_Rating'] = gk_df['Assists/90']

transfers_2016_df = transfers_df[(transfers_df['Season'] == 2016) & (transfers_df['Pos'] != 'GK')]
transfers_2016_df = transfers_2016_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
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

transfers_since_2017_df = transfers_df[
    (transfers_df['Season'] > 2016) & (transfers_df['Notes'] != 'youth') &
    (transfers_df['Notes'] != 'second season') & (transfers_df['Notes'] != 'missed season') &
    (transfers_df['Pos'] != 'GK')
]
transfers_since_2017_df = transfers_since_2017_df.merge(leagues_df[['League', 'Coefficient']], on='League', how='left')
transfers_since_2017_df = transfers_since_2017_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
transfers_since_2017_df = _pos_rates.apply_season_rates(transfers_since_2017_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
transfers_since_2017_df['Initial_NPG_Rating'] = (
    (league_coefficient_multiplier * transfers_since_2017_df['Coefficient'] / epl_coefficient) *
    (transfers_since_2017_df['npxG_Old'] + transfers_since_2017_df['NPG/90'] * minutes_constant / 90) /
    ((transfers_since_2017_df['Min_Old'] / 90) + minutes_constant / 90)
)
transfers_since_2017_df['Initial_Assist_Rating'] = (
    (league_coefficient_multiplier * transfers_since_2017_df['Coefficient'] / epl_coefficient) *
    (transfers_since_2017_df['xAG_Old'] + transfers_since_2017_df['Assists/90'] * minutes_constant / 90) /
    ((transfers_since_2017_df['Min_Old'] / 90) + minutes_constant / 90)
)

missed_season_df = transfers_df[(transfers_df['Notes'] == 'missed season') & (transfers_df['Pos'] != 'GK')]
missed_season_df = missed_season_df.merge(leagues_df[['League', 'Coefficient']], on='League', how='left')
missed_season_df = missed_season_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
missed_season_df = _pos_rates.apply_season_rates(missed_season_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
missed_season_df['Initial_NPG_Rating'] = np.where(
    missed_season_df['League'] == 'Premier League',
    ((missed_season_df['NPG_Old'] + missed_season_df['NPG/90'] * (minutes_constant / 2) / 90) /
     ((missed_season_df['Min_Old'] / 90) + (minutes_constant / 2) / 90)),
    (league_coefficient_multiplier * missed_season_df['Coefficient'] / epl_coefficient) *
    (missed_season_df['npxG_Old'] + missed_season_df['NPG/90'] * (minutes_constant / 2) / 90) /
    ((missed_season_df['Min_Old'] / 90) + (minutes_constant / 2) / 90)
)
missed_season_df['Initial_Assist_Rating'] = np.where(
    missed_season_df['League'] == 'Premier League',
    ((missed_season_df['Assists_Old'] + missed_season_df['Assists/90'] * (minutes_constant / 2) / 90) /
     ((missed_season_df['Min_Old'] / 90) + (minutes_constant / 2) / 90)),
    (league_coefficient_multiplier * missed_season_df['Coefficient'] / epl_coefficient) *
    (missed_season_df['xAG_Old'] + missed_season_df['Assists/90'] * (minutes_constant / 2) / 90) /
    ((missed_season_df['Min_Old'] / 90) + (minutes_constant / 2) / 90)
)

youth_df = transfers_df[(transfers_df['Notes'] == 'youth') & (transfers_df['Pos'] != 'GK')]
youth_df = youth_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
youth_df = _pos_rates.apply_season_rates(youth_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
youth_df['Initial_NPG_Rating'] = np.where(youth_df['Pos'] != 'FW', youth_coefficient * youth_df['NPG/90'], 0.7 * youth_df['NPG/90'])
youth_df['Initial_Assist_Rating'] = youth_coefficient * youth_df['Assists/90']
# Fitted youth multipliers (§113): when set, one factor per target for every position replaces the 0.75 / 0.7 above.
if os.environ.get("GOAL_MODEL_YOUTH_NPG"):
    youth_df['Initial_NPG_Rating'] = float(os.environ["GOAL_MODEL_YOUTH_NPG"]) * youth_df['NPG/90']
if os.environ.get("GOAL_MODEL_YOUTH_AST"):
    youth_df['Initial_Assist_Rating'] = float(os.environ["GOAL_MODEL_YOUTH_AST"]) * youth_df['Assists/90']

second_season_df = transfers_df[(transfers_df['Notes'] == 'second season') & (transfers_df['Pos'] != 'GK')]
second_season_df = second_season_df.merge(leagues_df[['League', 'Coefficient']], on='League', how='left')
second_season_df = second_season_df.merge(positions_df[['Pos', 'NPG/90', 'Assists/90']], on='Pos', how='left')
second_season_df = _pos_rates.apply_season_rates(second_season_df)  # §246; no-op unless GOAL_MODEL_POSITION_RATES
second_season_df['Initial_NPG_Rating'] = (
    (league_coefficient_multiplier * second_season_df['Coefficient'] / epl_coefficient) *
    (second_season_df['npxG_Old'] + second_season_df['NPG/90'] * minutes_constant / 90) /
    ((second_season_df['Min_Old'] / 90) + minutes_constant / 90)
)
second_season_df['Initial_Assist_Rating'] = (
    (league_coefficient_multiplier * second_season_df['Coefficient'] / epl_coefficient) *
    (second_season_df['xAG_Old'] + second_season_df['Assists/90'] * minutes_constant / 90) /
    ((second_season_df['Min_Old'] / 90) + minutes_constant / 90)
)

first_season_df = pd.concat([
    gk_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    transfers_2016_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    transfers_since_2017_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    missed_season_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    youth_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
])
initial_ratings_df = pd.concat([
    gk_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    transfers_2016_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    transfers_since_2017_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    missed_season_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    youth_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
    second_season_df[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating', 'Team_Attack_Rating']],
])

weight_team = 0.15
initial_ratings_df['Initial_NPG_Rating'] = initial_ratings_df['Initial_NPG_Rating'] * (initial_ratings_df['Team_Attack_Rating'] / AVG_CONCEDED) ** weight_team
initial_ratings_df['Initial_Assist_Rating'] = initial_ratings_df['Initial_Assist_Rating'] * (initial_ratings_df['Team_Attack_Rating'] / AVG_CONCEDED) ** weight_team
first_season_df['Initial_NPG_Rating'] = first_season_df['Initial_NPG_Rating'] * (first_season_df['Team_Attack_Rating'] / AVG_CONCEDED) ** weight_team
first_season_df['Initial_Assist_Rating'] = first_season_df['Initial_Assist_Rating'] * (first_season_df['Team_Attack_Rating'] / AVG_CONCEDED) ** weight_team

# ---- returners' new-league prior from a file (§128): replaces only the blend's new-league half ----
RETURN_PRIORS_FILE = os.environ.get("GOAL_MODEL_RETURN_PRIORS_FILE", "")
if RETURN_PRIORS_FILE:
    _rp = pd.read_csv(RETURN_PRIORS_FILE, encoding='utf-8-sig')
    _rp_map = {(p, s): (g, a) for p, s, g, a in _rp[['Player', 'Season', 'Initial_NPG_Rating', 'Initial_Assist_Rating']]
               .itertuples(index=False)}
    _hit = [(p, s) in _rp_map for p, s in zip(second_season_df['Player'], second_season_df['Season'])]
    second_season_df.loc[_hit, 'Initial_NPG_Rating'] = [_rp_map[(p, s)][0] for p, s, h in
        zip(second_season_df['Player'], second_season_df['Season'], _hit) if h]
    second_season_df.loc[_hit, 'Initial_Assist_Rating'] = [_rp_map[(p, s)][1] for p, s, h in
        zip(second_season_df['Player'], second_season_df['Season'], _hit) if h]
    print(f"[§128] returners' new-league prior from {Path(RETURN_PRIORS_FILE).name}: {sum(_hit)} rows")

# A blank blend input turns the player's rating NaN at his first match back, and NaN sticks for good (§185).
_blend_inputs = ['Initial_NPG_Rating', 'Initial_Assist_Rating', 'Prior_Minutes', 'Seasons_Missed']
_bad_blend = second_season_df[second_season_df[_blend_inputs].isna().any(axis=1)]
if len(_bad_blend):
    raise SystemExit("transfers.csv 'second season' rows with a NaN blend input (fill Min_Old, NPG_Old, Assists_Old, "
                     f"Prior_Minutes, Seasons_Missed; check League and Pos):\n"
                     f"{_bad_blend[['Player', 'Season', 'League'] + _blend_inputs].to_string(index=False)}")

# ---- new-player priors (CHANGELOG §103, §104) ----
# For arrivals in the league groups that passed the out-of-time test (Adopted), the prior comes from
# new_player_priors.csv (build_new_player_priors.py). Those values already carry their own fitted
# new-team factor, so they replace the rating after the weight_team step above. 0 restores the old method.
NEW_PRIORS_ON = os.environ.get("GOAL_MODEL_NEW_PRIORS", "1") == "1"
NEW_PRIORS_FILE = Path(os.environ.get("GOAL_MODEL_NEW_PRIORS_FILE", _DATA / 'new_player_priors.csv'))
if NEW_PRIORS_ON:
    if NEW_PRIORS_FILE.exists():
        _np = pd.read_csv(NEW_PRIORS_FILE, encoding='utf-8-sig')
        _np = _np[_np['Adopted'].astype(str).str.lower() == 'true']
        assert not _np.duplicated(['Player', 'Season']).any(), "new_player_priors.csv repeats a Player/Season"
        _np = _np.set_index(['Player', 'Season'])[['Initial_NPG_Rating', 'Initial_Assist_Rating']]
        _n_hit = 0
        for _df in (initial_ratings_df, first_season_df):
            _key = pd.MultiIndex.from_arrays([_df['Player'], _df['Season']])
            _hit = np.asarray(_key.isin(_np.index))
            _df.loc[_hit, ['Initial_NPG_Rating', 'Initial_Assist_Rating']] = _np.loc[_key[_hit]].to_numpy()
            _n_hit = max(_n_hit, int(_hit.sum()))
        print(f"[§103] new-player priors from {NEW_PRIORS_FILE.name}: {_n_hit} transfers.csv rows")
    else:
        print(f"WARNING: {NEW_PRIORS_FILE.name} missing -- new arrivals use the old transfers.csv priors "
              f"(run build_new_player_priors.py)")


# ---- position update for entrants (CHANGELOG §247, §249; adopted for npxG and xAG by user decision 2026-11-14) ----
# An entrant's starting rating holds his listed position's rate p_listed (§77 rule). While he is an entrant (his
# segment's first two seasons: youth, second season, missed season and arrival rows), that part of the start moves
# towards the rate of the slots he has actually played: before each match the rating carries c x s x (p - p_listed),
# s = the share of the start still in the rating, p = (Kpos x p_listed + sum of minutes/90 x slot rate over his earlier
# matches in the segment) / (Kpos + sum of minutes/90). Every appearance counts (form F0). Rates are
# position_combinations.csv's. GOAL_MODEL_POS_UPDATE=0 reverts.
POS_UPDATE = os.environ.get("GOAL_MODEL_POS_UPDATE", "1") == "1"
POS_UPDATE_C = (float(os.environ.get("GOAL_MODEL_POS_UPDATE_C_NPG", "0.4")),
                float(os.environ.get("GOAL_MODEL_POS_UPDATE_C_AST", "0.35")))
POS_UPDATE_KPOS = 1.0
_PU_SINGLE = _single_positions.set_index('Pos')
_PU_SLOT = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
            "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW"}   # GK and unknown slots give no evidence


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


_PU_ROWS = {}
for _second in (False, True):   # a 'second season' row wins a (player, season) tie, as initial_ratings_df's order
    for _r in transfers_df.itertuples(index=False):
        if (getattr(_r, 'Notes') == 'second season') == _second:
            _PU_ROWS[(_r.Player, _r.Season)] = (_pu_branch(_r.Notes, _r.Season, _r.League, _r.Pos), _r.Pos)


def _pu_listed(pos):
    prior, _ = position_prior(pos, _PU_SINGLE)
    return (prior['NPG/90'], prior['Assists/90'])


# ---- the main smoothing loop (npxG-only, no beta) ----
def _role_gate(ref, new):
    """True if a ref->new canonical slot change is material enough to fire (§39).

    Fires when the coarse group (DEF/MID/FWD/GK) changes, or -- within one coarse
    group -- when the total attacking prior (NPG/90 + Assists/90) differs by at
    least ROLE_FACTOR. The second clause is what catches the Calafiori CB->WB case
    (both DEF, but the wing-back assist prior is ~4.7x the centre-back's), while
    ignoring lateral churn like DM<->CM whose priors are near-identical.
    """
    if ref not in SLOT_PRIOR or new not in SLOT_PRIOR:
        return False
    if SLOT_COARSE.get(ref) != SLOT_COARSE.get(new):
        return True
    a = SLOT_PRIOR[ref][0] + SLOT_PRIOR[ref][1]
    b = SLOT_PRIOR[new][0] + SLOT_PRIOR[new][1]
    if a <= 0 or b <= 0:
        return False
    return max(a / b, b / a) >= ROLE_FACTOR


# Position group of a match for the centre-back npxG floor (§235); groups as validate_kalman_smoothing.POS_GROUP.
_MATCH_GROUP = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "FB", "DM": "CM", "CM": "CM",
                "AM": "AW", "LM": "AW", "RM": "AW", "LW": "AW", "RW": "AW", "FW": "FW"}
_TRANSFER_GROUP = {"CB": "CB", "FB": "FB", "WB": "FB", "DM": "CM", "CM": "CM", "AM": "AW", "W": "AW", "WM": "AW",
                   "FW": "FW"}
_tg = pd.read_csv(_DATA / 'transfers.csv', encoding='utf-8-sig', usecols=['Player', 'Season', 'Pos']).dropna()
_tg['g'] = _tg['Pos'].str.split(',').str[0].str.strip().map(_TRANSFER_GROUP)
_tg = _tg.sort_values('Season', kind='stable')
_TRANSFER_GROUPS = {p: (g['Season'].to_numpy(), g['g'].to_numpy()) for p, g in _tg.groupby('Player')}


def _transfer_group(player, season):
    v = _TRANSFER_GROUPS.get(player)
    if v is None:
        return None
    i = np.searchsorted(v[0], season, side='right') - 1
    return v[1][i] if i >= 0 and isinstance(v[1][i], str) else None


def smooth_ratings_and_mae(player_stats, concede_lu):
    ratings = {}
    total_npg_error = 0.0
    total_ast_error = 0.0
    established_npg_error = 0.0
    established_ast_error = 0.0
    total_weight = 0.0
    established_weight = 0.0
    player_stats["PriorMatches"] = player_stats.groupby("Player").cumcount()

    # ---- role-change detection state (§39); inert unless GOAL_MODEL_UNDERSTAT_POS ----
    role_recent = defaultdict(lambda: deque(maxlen=ROLE_W))
    role_ref = {}                       # player -> established canonical slot
    role_refcount = defaultdict(Counter)
    role_boost = defaultdict(int)       # player -> boosted matches remaining
    role_last_fire = {}                 # player -> PriorMatches at last fire (cooldown)
    youth_keys = set(zip(youth_df["Player"], youth_df["Season"]))
    youth_start = set()                 # players whose rating started from a youth row (§126)
    n_fires = 0
    diag_rows = []
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
                initial_row = second_season_df[(second_season_df["Player"] == player) & (second_season_df["Season"] == season)]
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

        # ---- role-change: detect a sustained slot change and open a boost window (§39).
        # Detection depends only on the slot sequence, never on the ratings, so the
        # GOAL_ROLE_APPLY=0 arm flags the identical rows while leaving alpha_eff (and
        # therefore the output) untouched -- the clean A/B the ship rule needs.
        boost_active = False
        canon_slot = row.get("_CanonSlot") if UNDERSTAT_POS_GOAL else None
        if UNDERSTAT_POS_GOAL and isinstance(canon_slot, str) and canon_slot:
            if player not in role_ref:
                # establish the reference slot over the first ROLE_MIN_MATCHES matches
                role_refcount[player][canon_slot] += 1
                if sum(role_refcount[player].values()) >= ROLE_MIN_MATCHES:
                    role_ref[player] = role_refcount[player].most_common(1)[0][0]
            else:
                role_recent[player].append(canon_slot)
                if (len(role_recent[player]) == ROLE_W
                        and len(set(role_recent[player])) == 1
                        and canon_slot != role_ref[player]
                        and _role_gate(role_ref[player], canon_slot)
                        and size - role_last_fire.get(player, -10**9) >= ROLE_COOLDOWN):
                    n_fires += 1
                    role_ref[player] = canon_slot          # re-anchor to the new role
                    role_boost[player] = ROLE_WINDOW
                    role_last_fire[player] = size
                    role_recent[player].clear()
            if role_boost[player] > 0:
                boost_active = True
                if ROLE_APPLY:
                    alpha_eff = ROLE_BOOST_ALPHA * (minutes / 90.0)
                role_boost[player] -= 1

        venue_coefficient = home_advantage if venue == "Home" else (2 - home_advantage)
        opp_key = (season, row["Team"], opp, venue) if OPP_DEFENCE else (m_id, opp)
        opp_concede = concede_lu.get(opp_key, AVG_CONCEDED)

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
            pu_start = _PU_ROWS.get((player, candidates['Season'].values[-1]), (None, None))
            a_rating = candidates['Initial_Assist_Rating'].values[-1]
            if (player, candidates['Season'].values[-1]) in youth_keys:
                youth_start.add(player)
                if not boost_active:
                    alpha_eff = max(1 / (size + PRIOR_K_YOUTH), ALPHA) * (minutes / 90.0)

        pu = None
        if POS_UPDATE:
            if pu_start is not None:
                if pu_start[0] is not None:
                    pos_update[player] = {"s0": int(row["_PosSeason"]), "pl": _pu_listed(pu_start[1]), "s": [1.0, 1.0],
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
        pred_goals_90 = g_rating * venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_NPG * sub_uplift("npxG", row["_IsStarter"], minutes)
        pred_ast_90 = a_rating * venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_AST * sub_uplift("xAG", row["_IsStarter"], minutes)
        weight = minutes / 90.0
        role_extra = ()
        if ROLE_K_ON:
            rec = ROLE_TABLE.get((int(row["_LogSeason"]), row["Team"], opp, venue, player))
            p_now, line_now = role_p.get(player), None
            if rec is not None:
                p_now, line_now, role_p[player] = rec[0], rec[1], rec[2]
                role_l0.setdefault(player, line_now)
            m1_g = m1_a = o1_g = o1_a = 1.0
            if player in role_l0 and rec is not None and rec[4] == 0:
                pass    # first start: no history yet, so the honest factor is 1
            elif player in role_l0 and p_now is not None:
                l0 = role_l0[player]
                m1_g = float(p_now @ RATE_NPG / RATE_NPG[l0])
                m1_a = float(p_now @ RATE_AST / RATE_AST[l0])
                o1_g = RATE_NPG[line_now] / RATE_NPG[l0] if line_now is not None else m1_g
                o1_a = RATE_AST[line_now] / RATE_AST[l0] if line_now is not None else m1_a
            h_g, h_a = _role_mult(m1_g, K_NPG_UP, K_NPG_DOWN), _role_mult(m1_a, K_AST_UP, K_AST_DOWN)
            pred_goals_90 *= _role_mult(o1_g, K_NPG_UP, K_NPG_DOWN)
            pred_ast_90 *= _role_mult(o1_a, K_AST_UP, K_AST_DOWN)
            role_extra = (pred_goals_90 * h_g / _role_mult(o1_g, K_NPG_UP, K_NPG_DOWN),
                          pred_ast_90 * h_a / _role_mult(o1_a, K_AST_UP, K_AST_DOWN), m1_g, m1_a)

        if size > 10:
            established_npg_error += abs(npxg90 - pred_goals_90) * weight
            established_ast_error += abs(xag90 - pred_ast_90) * weight
            established_weight += weight
        total_npg_error += abs(npxg90 - pred_goals_90) * weight
        total_ast_error += abs(xag90 - pred_ast_90) * weight
        total_weight += weight

        if UNDERSTAT_POS_GOAL and ROLE_DIAG:
            diag_rows.append((player, int(row["_NativeSeason"]), int(m_id), canon_slot,
                              int(boost_active), float(npxg90), float(pred_goals_90),
                              float(xag90), float(pred_ast_90), float(weight)))
        if PREMATCH_LOG:
            prematch_rows.append((player, row["Team"], opp, venue, int(m_id), row["Data_Source"],
                                  int(row["_LogSeason"]), float(g_rating), float(a_rating),
                                  float(npxg90), float(pred_goals_90), float(xag90),
                                  float(pred_ast_90), float(weight),
                                  float(opp_concede / AVG_CONCEDED), str(row["_IsStarter"])) + role_extra)

        grp = _MATCH_GROUP.get(row["_MatchPos"]) if isinstance(row["_MatchPos"], str) else None
        if grp is None:
            grp = last_group.get(player, _transfer_group(player, row["_PosSeason"]))
        if grp is not None:
            last_group[player] = grp
        if boost_active and ROLE_APPLY:
            alpha_npg = alpha_eff
        else:   # same schedule as alpha_eff, with the npxG floor (§233; centre-backs §235)
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
            m_g = venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_NPG * sub_uplift("npxG", row["_IsStarter"], minutes)
            m_a = venue_coefficient * (opp_concede / AVG_CONCEDED) ** OPP_BETA_AST * sub_uplift("xAG", row["_IsStarter"], minutes)
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
                pu["swr"][0] += weight * _PU_SINGLE.at[slot, 'NPG/90']
                pu["swr"][1] += weight * _PU_SINGLE.at[slot, 'Assists/90']
        if MOVE_TABLE_FILE:
            mv = MOVE_KEYS.get((int(row["_LogSeason"]), row["Team"], opp, venue, player))
            if mv is not None:
                last = move_last.get(player)
                if last is not None and last[0] == mv["new"]:   # back to the line before the last move
                    g_rating, a_rating = g_rating / last[1], a_rating / last[2]
                    del move_last[player]
                else:
                    d = "forward" if mv["fwd"] else "back"
                    fg, fa = MOVE_F["npxG"][d], MOVE_F["xAG"][d]
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
    if UNDERSTAT_POS_GOAL:
        print(f"[role-change S39] fires={n_fires}  apply={int(ROLE_APPLY)}  "
              f"boost_alpha={ROLE_BOOST_ALPHA}  window={ROLE_WINDOW}  W={ROLE_W}  "
              f"factor={ROLE_FACTOR}")
        if ROLE_DIAG and diag_rows:
            _diag = pd.DataFrame(diag_rows, columns=[
                "Player", "Season", "MatchID", "Slot", "BoostActive",
                "npxg90", "pred_goals_90", "xag90", "pred_ast_90", "weight"])
            _diag_path = Path(__file__).resolve().parents[2] / ROLE_DIAG_OUT
            _diag.to_csv(_diag_path, index=False, encoding="utf-8-sig")
            print(f"[role-change S39] wrote {len(_diag)} diagnostic rows "
                  f"({int(_diag['BoostActive'].sum())} boost-active) -> {_diag_path.name}")
    if PREMATCH_LOG:
        pd.DataFrame(prematch_rows, columns=[
            "Player", "Team", "Opponent", "Venue", "MatchID", "Data_Source", "Season",
            "NPG_Rating_Pre", "Assist_Rating_Pre", "npxg90", "pred_goals_90", "xag90",
            "pred_ast_90", "weight", "Opp_Factor", "Is_Starter"]
            + (["pred_goals_90_h", "pred_ast_90_h", "M1_npg", "M1_ast"] if ROLE_K_ON else [])).to_csv(
            PREMATCH_LOG, index=False, encoding="utf-8-sig")
        print(f"Wrote {len(prematch_rows)} pre-match rating rows to {PREMATCH_LOG}")
    return ratings, mae_npg, mae_ast


final_ratings, mae_npg, mae_ast = smooth_ratings_and_mae(playerstats, concede_lookup)
# A position-update shift can leave a rating below 0 mid-segment (§249); a forecast rate never is.
_neg = [p for p, (g, a) in final_ratings.items() if g < 0 or a < 0]
if _neg:
    print(f"[§249] {len(_neg)} final ratings below 0 set to 0: {', '.join(sorted(_neg)[:10])}")
    final_ratings = {p: (max(g, 0.0), max(a, 0.0)) for p, (g, a) in final_ratings.items()}

# ---- new players not seen at all: fall back to their transfers.csv initial rating ----
initial_ratings_dict = first_season_df.set_index(["Player", "Season"])[["Initial_NPG_Rating", "Initial_Assist_Rating"]].to_dict("index")
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
    is_2025_new = player in initial_ratings_df.loc[initial_ratings_df['Season'] == 2025, 'Player'].values
    if prev_team != curr_team and not is_2025_new:
        prev_team_atk_rating = data.tail(1).Team_Attack_Rating.values[0]
        curr_team_row_rating = team_ratings_2026[team_ratings_2026['Team'] == curr_team]
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
career_totals['Conversion'] = (career_totals['career_NPG'] + SHRINKAGE_PRIOR) / (career_totals['career_npxG'] + SHRINKAGE_PRIOR)
conversion_lookup = dict(zip(career_totals['Player'], career_totals['Conversion']))

# ---- diagnostic: check for missing Team_Attack_Rating in transfers.csv ----
missing_atk = transfers_df[transfers_df['Team_Attack_Rating'].isna()]
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
    raise SystemExit(f"NaN ratings, nothing written (§185):\n{still_null[['Player', 'Team']].to_string(index=False)}")
if PREMATCH_LOG:
    out_df.to_csv(PREMATCH_LOG + ".ratings.csv", index=False, encoding='utf-8-sig')
    print(f"\nGOAL_MODEL_PREMATCH_LOG set: ratings written to {PREMATCH_LOG}.ratings.csv, "
          f"not player_goal_assist_ratings.csv")
else:
    out_df.to_csv(_DATA / 'player_goal_assist_ratings.csv', index=False, encoding='utf-8-sig')
    print(f"\nWrote {len(out_df)} players to player_goal_assist_ratings.csv")
