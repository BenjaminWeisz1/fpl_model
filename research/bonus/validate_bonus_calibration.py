"""
Chronological, profile-level diagnostic for bonus awards.

Historical pre-match bonus forecasts and expected-minutes snapshots are not
retained, so this evaluates a causal position/profile historical-rate baseline
conditional on players who actually appeared. It does not validate the complete
pre-match bonus simulator.

The current-season split is fixed: GW1-4 fit the profile-rate benchmark and
GW5-7 are the untouched test set. The frozen GW8 forecast is hash-checked but
never read as an input or modified.

Run locally:
    python validate_bonus_calibration.py
"""
import hashlib
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from fpl_model import paths

ROOT = paths.ROOT
DATA = paths.CURRENT
STATS_FILE = paths.CURRENT / "player_match_stats.csv"
POSITIONS_FILE = paths.CURRENT / "player_match_positions.csv"
RESULTS_FILE = paths.CURRENT / "match_results.csv"
FIXTURES_FILE = paths.CURRENT / "fixtures.csv"
FROZEN_FORECAST = (
    paths.SNAPSHOTS / "2026" / "GW08" / "20261020T210325_940676Z_partall" / "gameweek" / "bonus_predictions.csv"
)
FROZEN_FORECAST_SHA256 = "cfe48d049c8a1562dc1f1ddad7236e2bef2bc9b8bb062342196f8728ac43bd67"
OUT_DIR = paths.VALIDATION / "bonus_profile_20261021"

GOAL_BPS = {"G": 12, "D": 12, "M": 18, "F": 24}
SLOT_POSITION = {
    "GK": "G",
    "CB": "D", "FB": "D", "WB": "D", "LB": "D", "RB": "D",
    "DM": "M", "CM": "M", "AM": "M", "WM": "M", "W": "M",
    "LM": "M", "RM": "M", "LW": "M", "RW": "M",
    "FW": "F",
}
ATTACK_MIN_PRIOR_MINUTES = 450
MIN_PROFILE_PLAYER_MATCHES = 200
MIN_PROFILE_MATCHES = 20
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_match_rows():
    stats = pd.read_csv(STATS_FILE, encoding="utf-8-sig", low_memory=False)
    stats = stats.loc[
        stats["Data_Source"].eq("FPL_API") & stats["Season"].isin([2025, 2026])
    ].copy()
    exact_duplicates_removed = int(stats.duplicated().sum())
    stats = stats.drop_duplicates().copy()
    stats = stats.rename(columns={"MatchID": "PMS_MatchID"})
    source_key = ["Season", "Team", "Opponent", "Venue", "Player"]
    if stats.duplicated(source_key).any():
        raise ValueError("Non-identical duplicate player-fixture rows in FPL API history")
    positions = pd.read_csv(POSITIONS_FILE, encoding="utf-8-sig", low_memory=False)
    results = pd.read_csv(RESULTS_FILE, encoding="utf-8-sig", low_memory=False)
    fixtures = pd.read_csv(FIXTURES_FILE, encoding="utf-8-sig", low_memory=False)

    result_sides = pd.concat([
        results[["MatchID", "Season", "Home", "Away"]].rename(
            columns={"Home": "Team", "Away": "Opponent"}
        ).assign(Venue="Home"),
        results[["MatchID", "Season", "Away", "Home"]].rename(
            columns={"Away": "Team", "Home": "Opponent"}
        ).assign(Venue="Away"),
    ], ignore_index=True)
    stats = stats.merge(
        result_sides, on=["Season", "Team", "Opponent", "Venue"],
        how="left", validate="many_to_one",
    )
    if stats["MatchID"].isna().any():
        raise ValueError("Some FPL API rows do not map to a completed match result")

    position_keys = ["Season", "Team", "Opponent", "Venue", "Player"]
    stats = stats.merge(
        positions[position_keys + [
            "Player_ID_Understat", "Position_Mapped", "Is_Starter",
        ]],
        on=position_keys, how="left", validate="many_to_one",
    )
    stats["Player_ID_Understat"] = pd.to_numeric(
        stats["Player_ID_Understat"], errors="coerce"
    )

    current_results = results.loc[results["Season"].eq(2026),
                                  ["MatchID", "Season", "Home", "Away"]]
    current_fixtures = fixtures.loc[fixtures["Season"].eq(2026),
                                    ["Season", "Home", "Away", "Round"]]
    completed_keys = set(zip(
        current_results["Season"], current_results["Home"], current_results["Away"]
    ))
    completed_fixture_mask = [
        (season, home, away) in completed_keys
        for season, home, away in zip(
            current_fixtures["Season"], current_fixtures["Home"],
            current_fixtures["Away"],
        )
    ]
    current_fixtures = current_fixtures.loc[completed_fixture_mask]
    round_lookup = current_results.merge(
        current_fixtures, on=["Season", "Home", "Away"],
        how="left", validate="one_to_one",
    )
    if round_lookup["Round"].isna().any():
        raise ValueError("Some completed 2026 fixtures have no gameweek mapping")
    stats = stats.merge(
        round_lookup[["MatchID", "Round"]], on="MatchID", how="left",
        validate="many_to_one",
    )
    if stats.loc[stats["Season"].eq(2026), "Round"].isna().any():
        raise ValueError("Some 2026 FPL API rows have no gameweek mapping")
    if stats.duplicated(["Season", "MatchID", "Player"]).any():
        raise ValueError("Duplicate player-match rows found in the FPL API history")
    stats = stats.sort_values(["Season", "MatchID", "Player"]).reset_index(drop=True)
    stats.attrs["exact_duplicates_removed"] = exact_duplicates_removed
    return stats


def add_causal_profiles(rows):
    """Add pre-match profile features, updating state only after each fixture."""
    player_state = defaultdict(lambda: {
        "minutes": 0.0, "attack_bps": 0.0, "defcon_bps": 0.0,
        "starting_slots": Counter(),
    })
    prior_minutes = np.zeros(len(rows), dtype=float)
    attack_share = np.full(len(rows), np.nan)
    role_slot = np.full(len(rows), None, dtype=object)
    broad_pos = np.full(len(rows), None, dtype=object)

    for _, match in rows.groupby(["Season", "MatchID"], sort=False):
        updates = []
        for idx, row in match.iterrows():
            pid = row["Player_ID_Understat"]
            if pd.isna(pid):
                continue
            state = player_state[pid]
            mapped = row["Position_Mapped"]
            is_starter = bool(row["Is_Starter"]) if pd.notna(row["Is_Starter"]) else False
            if is_starter and pd.notna(mapped) and mapped != "Sub":
                slot = str(mapped)
            elif state["starting_slots"]:
                slot = state["starting_slots"].most_common(1)[0][0]
            else:
                slot = None
            role_slot[idx] = slot
            pos = SLOT_POSITION.get(slot)
            broad_pos[idx] = pos
            prior_minutes[idx] = state["minutes"]
            if state["minutes"] >= ATTACK_MIN_PRIOR_MINUTES:
                total = state["attack_bps"] + state["defcon_bps"]
                if total > 0:
                    attack_share[idx] = state["attack_bps"] / total

            if pos:
                goals = row["NPG"] if pd.notna(row["NPG"]) else 0
                pens = row["PK"] if pd.notna(row["PK"]) else 0
                assists = row["Assists"] if pd.notna(row["Assists"]) else 0
                tackles = row["Tkl"] if pd.notna(row["Tkl"]) else 0
                cbi = row["CBI"] if pd.notna(row["CBI"]) else 0
                recoveries = row["Recov"] if pd.notna(row["Recov"]) else 0
                attack = (goals + pens) * GOAL_BPS[pos] + assists * 9
                defcon = tackles * 2 + np.floor(cbi / 3) + np.floor(recoveries / 3)
                updates.append((pid, float(row["Minutes"]), float(attack),
                                float(defcon), slot, is_starter))

        for pid, minutes, attack, defcon, slot, is_starter in updates:
            state = player_state[pid]
            state["minutes"] += minutes
            state["attack_bps"] += attack
            state["defcon_bps"] += defcon
            if is_starter and slot:
                state["starting_slots"][slot] += 1

    out = rows.copy()
    out["Prior_Minutes"] = prior_minutes
    out["Lagged_Attack_Share"] = attack_share
    out["Profile_Slot"] = role_slot
    out["Profile_Pos"] = broad_pos

    prior_2025 = out.loc[
        out["Season"].eq(2025)
        & out["Profile_Pos"].isin(["D", "M"])
        & out["Lagged_Attack_Share"].notna()
    ]
    cutoffs = prior_2025.groupby("Profile_Pos")["Lagged_Attack_Share"].median().to_dict()
    if not {"D", "M"}.issubset(cutoffs):
        raise ValueError(f"Insufficient 2025/26 lagged-profile history: {cutoffs}")

    def profile_groups(row):
        pos = row["Profile_Pos"]
        groups = []
        if pos:
            groups.append(("position", pos))
        slot = row["Profile_Slot"]
        if pos == "D":
            if slot == "CB":
                groups.append(("defender_role", "CB"))
            elif slot in {"FB", "WB", "LB", "RB"}:
                groups.append(("defender_role", "FB/WB"))
        if pos in {"D", "M"} and pd.notna(row["Lagged_Attack_Share"]):
            label = "attacking" if row["Lagged_Attack_Share"] > cutoffs[pos] else "defcon"
            groups.append(("orientation", f"{pos}-{label}"))
        return groups

    out["Profile_Groups"] = out.apply(profile_groups, axis=1)
    return out, cutoffs


def bootstrap_interval(group, match_col, prediction_col, target_col, seed):
    # Store additive sums explicitly to preserve equal-match cluster resampling.
    temp = group.assign(
        _error=group[prediction_col] - group[target_col],
        _abs_error=(group[prediction_col] - group[target_col]).abs(),
        _sq_error=(group[prediction_col] - group[target_col]) ** 2,
    )
    sums = temp.groupby(match_col).agg(
        n=("_error", "size"),
        error=("_error", "sum"),
        abs_error=("_abs_error", "sum"),
        sq_error=("_sq_error", "sum"),
    )
    rng = np.random.default_rng(seed)
    ids = sums.index.to_numpy()
    samples = rng.choice(ids, size=(BOOTSTRAP_REPLICATES, len(ids)), replace=True)
    errors = sums["error"].to_numpy()
    abs_errors = sums["abs_error"].to_numpy()
    sq_errors = sums["sq_error"].to_numpy()
    counts = sums["n"].to_numpy()
    lookup = {match_id: i for i, match_id in enumerate(ids)}
    sampled_idx = np.vectorize(lookup.__getitem__)(samples)
    den = counts[sampled_idx].sum(axis=1)
    bias = errors[sampled_idx].sum(axis=1) / den
    mae = abs_errors[sampled_idx].sum(axis=1) / den
    rmse = np.sqrt(sq_errors[sampled_idx].sum(axis=1) / den)
    return {
        "bias_ci_low": float(np.quantile(bias, 0.025)),
        "bias_ci_high": float(np.quantile(bias, 0.975)),
        "mae_ci_low": float(np.quantile(mae, 0.025)),
        "mae_ci_high": float(np.quantile(mae, 0.975)),
        "rmse_ci_low": float(np.quantile(rmse, 0.025)),
        "rmse_ci_high": float(np.quantile(rmse, 0.975)),
    }


def bootstrap_primary_score(test_long, supported_profiles, test_match_ids):
    if not supported_profiles:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    match_ids = np.asarray(test_match_ids)
    samples = rng.choice(
        match_ids, size=(BOOTSTRAP_REPLICATES, len(match_ids)), replace=True
    )
    lookup = {match_id: i for i, match_id in enumerate(match_ids)}
    sampled_idx = np.vectorize(lookup.__getitem__)(samples)
    group_errors = []
    group_counts = []
    for profile_type, profile in supported_profiles:
        group = test_long.loc[
            test_long["Profile_Type"].eq(profile_type)
            & test_long["Profile"].eq(profile)
        ].assign(_error=lambda d: d["Predicted_Bonus"] - d["Bonus"])
        by_match = group.groupby("MatchID").agg(
            error=("_error", "sum"), count=("_error", "size")
        )
        errors = np.zeros(len(match_ids), dtype=float)
        counts = np.zeros(len(match_ids), dtype=float)
        for match_id, row in by_match.iterrows():
            slot = lookup[match_id]
            errors[slot] = row["error"]
            counts[slot] = row["count"]
        group_errors.append(errors[sampled_idx].sum(axis=1))
        group_counts.append(counts[sampled_idx].sum(axis=1))

    abs_biases = []
    for errors, counts in zip(group_errors, group_counts):
        abs_biases.append(np.divide(
            np.abs(errors), counts,
            out=np.full_like(errors, np.nan, dtype=float),
            where=counts > 0,
        ))
    scores = np.nanmean(np.vstack(abs_biases), axis=0)
    return (
        float(np.quantile(scores, 0.025)),
        float(np.quantile(scores, 0.975)),
    )


def evaluate_profiles(rows):
    current = rows.loc[rows["Season"].eq(2026)].copy()
    fit = current.loc[current["Round"].between(1, 4)].copy()
    test = current.loc[current["Round"].between(5, 7)].copy()
    if fit.empty or test.empty:
        raise ValueError("Registered GW1-4 fit / GW5-7 test windows are incomplete")

    def expand_groups(frame):
        expanded = frame[["MatchID", "Bonus", "Profile_Groups"]].explode("Profile_Groups")
        pairs = [
            value if isinstance(value, tuple) else (None, None)
            for value in expanded["Profile_Groups"]
        ]
        expanded[["Profile_Type", "Profile"]] = pd.DataFrame(
            pairs, index=expanded.index, columns=["Profile_Type", "Profile"]
        )
        return expanded.dropna(subset=["Profile_Type", "Profile"])

    fit_long = expand_groups(fit)
    test_long = expand_groups(test)
    fit_long["Bonus"] = pd.to_numeric(fit_long["Bonus"], errors="raise")
    test_long["Bonus"] = pd.to_numeric(test_long["Bonus"], errors="raise")
    fit_long["Any_Bonus"] = fit_long["Bonus"].gt(0).astype(float)
    test_long["Any_Bonus"] = test_long["Bonus"].gt(0).astype(float)

    rates = fit_long.groupby(["Profile_Type", "Profile"]).agg(
        Fit_Player_Matches=("Bonus", "size"),
        Fit_Matches=("MatchID", "nunique"),
        Predicted_Bonus=("Bonus", "mean"),
        Predicted_Any_Bonus=("Any_Bonus", "mean"),
    ).reset_index()
    test_long = test_long.merge(
        rates, on=["Profile_Type", "Profile"], how="left", validate="many_to_one"
    )
    test_long = test_long.dropna(subset=["Predicted_Bonus"]).copy()

    results = []
    for (profile_type, profile), group in test_long.groupby(
            ["Profile_Type", "Profile"], sort=True):
        observed = group["Bonus"].to_numpy(dtype=float)
        predicted = group["Predicted_Bonus"].to_numpy(dtype=float)
        any_actual = group["Any_Bonus"].to_numpy(dtype=float)
        any_predicted = group["Predicted_Any_Bonus"].to_numpy(dtype=float)
        n_players = len(group)
        n_matches = group["MatchID"].nunique()
        supported = (n_players >= MIN_PROFILE_PLAYER_MATCHES
                     and n_matches >= MIN_PROFILE_MATCHES)
        metrics = {
            "profile_type": profile_type,
            "profile": profile,
            "fit_player_matches": int(group["Fit_Player_Matches"].iloc[0]),
            "fit_matches": int(group["Fit_Matches"].iloc[0]),
            "test_player_matches": n_players,
            "test_matches": n_matches,
            "supported_for_inference": bool(supported),
            "predicted_bonus_per_player_match": float(np.mean(predicted)),
            "observed_bonus_per_player_match": float(np.mean(observed)),
            "signed_bias_pred_minus_actual": float(np.mean(predicted - observed)),
            "mae_bonus_per_player_match": float(np.mean(np.abs(predicted - observed))),
            "rmse_bonus_per_player_match": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "predicted_any_bonus_rate": float(np.mean(any_predicted)),
            "observed_any_bonus_rate": float(np.mean(any_actual)),
            "any_bonus_brier": float(np.mean((any_predicted - any_actual) ** 2)),
            "predicted_bonus_total": float(np.sum(predicted)),
            "observed_bonus_total": float(np.sum(observed)),
            "observed_award_rate": float(np.mean(any_actual)),
        }
        metrics.update(bootstrap_interval(
            group, "MatchID", "Predicted_Bonus", "Bonus",
            BOOTSTRAP_SEED + len(results),
        ))
        results.append(metrics)

    report = pd.DataFrame(results)
    supported = report.loc[report["supported_for_inference"]]
    if supported.empty:
        primary_score = None
        primary_ci = (None, None)
    else:
        primary_score = float(
            supported["signed_bias_pred_minus_actual"].abs().mean()
        )
        supported_profiles = list(zip(
            supported["profile_type"], supported["profile"]
        ))
        primary_ci = bootstrap_primary_score(
            test_long, supported_profiles, test["MatchID"].unique()
        )
    return report, {
        "fit_rounds": [1, 4],
        "test_rounds": [5, 7],
        "fit_fixtures": int(fit["MatchID"].nunique()),
        "test_fixtures": int(test["MatchID"].nunique()),
        "fit_player_matches": int(len(fit)),
        "test_player_matches": int(len(test)),
        "primary_mean_absolute_group_bias": primary_score,
        "primary_bias_ci_low": primary_ci[0],
        "primary_bias_ci_high": primary_ci[1],
        "primary_supported_group_count": int(len(supported)),
    }


def main():
    if not FROZEN_FORECAST.exists():
        raise FileNotFoundError(f"Frozen GW8 forecast missing: {FROZEN_FORECAST}")
    frozen_hash_before = sha256(FROZEN_FORECAST)
    if frozen_hash_before != FROZEN_FORECAST_SHA256:
        raise ValueError("Frozen GW8 bonus forecast hash differs from preregistration")

    rows = build_match_rows()
    exact_duplicates_removed = rows.attrs.get("exact_duplicates_removed", 0)
    rows, cutoffs = add_causal_profiles(rows)
    metrics, split = evaluate_profiles(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "profile_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    fpl_api = rows.loc[rows["Season"].eq(2026)]
    gk_diagnostics = {}
    for season in (2025, 2026):
        keepers = rows.loc[
            rows["Season"].eq(season) & rows["Profile_Pos"].eq("G")
        ]
        under60 = keepers.loc[keepers["Minutes"].lt(60)]
        gk_diagnostics[str(season)] = {
            "keeper_player_matches": int(len(keepers)),
            "under60_player_matches": int(len(under60)),
            "under60_with_saves": int(under60["Saves"].gt(0).sum()),
            "under60_saves": int(under60["Saves"].sum()),
            "under60_bonus_awards": int(under60["Bonus"].gt(0).sum()),
            "under60_bonus_points": int(under60["Bonus"].sum()),
        }
    test_round_rows = fpl_api.loc[fpl_api["Round"].between(5, 7)]
    coverage = {
        "analysis_type": "conditional historical-rate benchmark; not incumbent forecast validation",
        "official_rules_reference": (
            "https://www.premierleague.com/en/news/4679946/whats-new-in-202627-fantasy-changes-to-bonus-points-system"
        ),
        "current_season_fpl_api_player_matches": int(len(fpl_api)),
        "exact_duplicate_source_rows_removed": int(exact_duplicates_removed),
        "current_season_completed_fixtures": int(fpl_api["MatchID"].nunique()),
        "current_season_rounds": sorted(
            int(value) for value in fpl_api["Round"].dropna().unique()
        ),
        "position_crosswalk_coverage": float(fpl_api["Profile_Pos"].notna().mean()),
        "profile_position_covered_fit_rows": int(
            fpl_api.loc[fpl_api["Round"].between(1, 4), "Profile_Pos"].notna().sum()
        ),
        "profile_position_covered_test_rows": int(
            test_round_rows["Profile_Pos"].notna().sum()
        ),
        "current_season_bonus_points": int(fpl_api["Bonus"].sum()),
        "goalkeeper_under60_save_diagnostic": gk_diagnostics,
        "fit_test_split": split,
        "attack_share_median_cutoffs_from_2025": {
            str(key): float(value) for key, value in cutoffs.items()
        },
        "profile_support_rule": {
            "minimum_player_matches": MIN_PROFILE_PLAYER_MATCHES,
            "minimum_fixtures": MIN_PROFILE_MATCHES,
            "thin_profile_rates_are_descriptive_only": True,
        },
        "bootstrap": {
            "cluster": "fixture",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "frozen_gw8_forecast_sha256_before": frozen_hash_before,
        "frozen_gw8_forecast_sha256_after": sha256(FROZEN_FORECAST),
        "frozen_gw8_forecast_unchanged": sha256(FROZEN_FORECAST) == frozen_hash_before,
        "interpretation": (
            "Expected bonus is a profile rate fitted on GW1-4 and evaluated "
            "conditionally among actual appearances in GW5-7. It cannot establish "
            "calibration of the production simulator's full pre-match forecasts."
        ),
    }
    if not coverage["frozen_gw8_forecast_unchanged"]:
        raise RuntimeError("Frozen GW8 forecast changed during validation")
    (OUT_DIR / "coverage_and_protocol.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(metrics)} profile rows to {metrics_path}")
    print(f"Fit fixtures: {split['fit_fixtures']}; test fixtures: {split['test_fixtures']}")
    print(metrics.to_string(index=False))
    print(f"Frozen GW8 forecast unchanged: {coverage['frozen_gw8_forecast_unchanged']}")


if __name__ == "__main__":
    main()
