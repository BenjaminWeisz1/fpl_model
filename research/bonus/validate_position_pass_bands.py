"""
Pass bands by detailed position (pre-registered in CHANGELOG §205).

Incumbent: §68 pass-band distribution (last 38 Sofascore 60+ appearances shrunk toward
the FPL-position prior). C1: prior and k mixed over the player's pre-match chance of
each of nine detailed roles. C2 (primary): C1 tilted so its mean is the role-mixed mean
plus the player's shrunk deviation from the means of the roles he actually played.

Part 1, one step ahead walk-forward on T1 = 2023-2024 and T2 = 2025-2026, constants
fitted on 2018-2022. Part 2, bonus safety check on §68's windows (constants 2018-2024).

Run locally:
    python validate_position_pass_bands.py
"""
import hashlib
import json
import sys
from collections import deque

import numpy as np
import pandas as pd

from fpl_model.bonus import build_residual_bps_baseline as residual
from fpl_model.bonus import pass_bands as pb
from research.bonus import validate_bonus_accounting as acc
from research.bonus import validate_bonus_calibration as calibration
from research.bonus import validate_pass_bands as vpb

OUT_DIR = calibration.ROOT / "data" / "validation" / "position_pass_bands_20261106"
BANDS = vpb.BANDS
HISTORY = vpb.HISTORY
ROLES = ["CB", "FB", "WB", "DM", "CM", "AM", "WM", "W", "FW"]
ROLE_OF = {"CB": "CB", "LB": "FB", "RB": "FB", "WB": "WB", "DM": "DM", "CM": "CM", "AM": "AM",
           "LM": "WM", "RM": "WM", "LW": "W", "RW": "W", "FW": "FW"}
HALF_LIFE, ROLE_PRIOR, RECENT_WEIGHT = 3, 0.1, 0.9  # §170 recipe (role_shares.py)
FIT = range(2018, 2023)
FIT_BONUS = range(2018, 2025)
PERIODS = {"T1": (2023, 2024), "T2": (2025, 2026)}
ARMS = ["incumbent", "C1", "C2"]
SEED, REPS = 42, 2000


def load():
    so = vpb.load_sofascore()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    pos = pd.read_csv(calibration.DATA / "player_match_positions.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=key + ["Position_Resolved", "Is_Starter"])
    n = len(so)
    so = so.merge(pos.rename(columns={"Player": "Local_Player"}), on=["Season", "Team", "Opponent", "Venue",
                                                                      "Local_Player"], how="left")
    assert len(so) == n, "position join fanned out"
    so["role"] = so["Position_Resolved"].map(ROLE_OF)
    so["starter"] = so["Is_Starter"].astype(str).eq("True")
    return so.sort_values(["Canonical_Player_ID", "Kickoff"]).reset_index(drop=True)


def mom_k(df, group, value, min_n=10):
    """§68 method of moments: within-player-season variance / consecutive-season covariance, per group."""
    st = df.groupby([group, "Canonical_Player_ID", "Season"])[value].agg(["mean", "var", "size"]).reset_index()
    out = {}
    for g, d in st[st["size"] >= min_n].groupby(group):
        nxt = d.assign(Season=d["Season"] - 1)[["Canonical_Player_ID", "Season", "mean"]]
        pair = d.merge(nxt, on=["Canonical_Player_ID", "Season"], suffixes=("", "_n"))
        tau2 = np.cov(pair["mean"], pair["mean_n"])[0, 1] if len(pair) > 2 else 0
        out[g] = float(np.clip(d["var"].mean() / tau2, 0.5, 100)) if tau2 > 0 else 100.0
    return out


def fit_constants(so, seasons):
    f = so[so["Season"].isin(seasons) & so["full"]].copy()
    dist = lambda g: np.array([(g["band"] == b).mean() for b in BANDS])
    c = {"prior": {p: dist(g) for p, g in f.groupby("Position_Sofascore")},
         "k": mom_k(f, "Position_Sofascore", "band")}
    r = f.dropna(subset=["role"])
    c["role_prior"] = {x: dist(g) for x, g in r.groupby("role")}
    c["role_mu"] = {x: float(g["band"].mean()) for x, g in r.groupby("role")}
    c["role_k"] = mom_k(r, "role", "band")
    r = r.assign(dev=r["band"] - r["role"].map(c["role_mu"]))
    c["k_e"] = mom_k(r, "Position_Sofascore", "dev")
    return c


class State:
    """Pre-match state of one player: recent 60+ bands with roles, role-share counts."""

    def __init__(self):
        self.hist = deque(maxlen=HISTORY)
        self.w = np.zeros(len(ROLES))
        self.career = np.zeros(len(ROLES))
        self.decay = 0.5 ** (1 / HALF_LIFE)

    def update(self, band, full, role, starter):
        if full:
            self.hist.append((band, role))
        if starter and isinstance(role, str):
            i = ROLES.index(role)
            self.w *= self.decay
            self.w[i] += 1
            self.career[i] += 1

    def pi(self):
        if self.career.sum() == 0:
            return None
        sh = lambda c: (c + ROLE_PRIOR / len(ROLES)) / (c.sum() + ROLE_PRIOR)
        return RECENT_WEIGHT * sh(self.w) + (1 - RECENT_WEIGHT) * sh(self.career)

    def forecast(self, pos, c):
        """(incumbent, C1, C2) band distributions, n, role-change flag."""
        counts = np.array([sum(1 for b, _ in self.hist if b == x) for x in BANDS], float)
        n = counts.sum()
        inc = (counts + c["k"][pos] * c["prior"][pos]) / (n + c["k"][pos])
        pi = self.pi()
        if pi is None:
            return inc, inc, inc, n, False
        prior = sum(pi[i] * c["role_prior"][r] for i, r in enumerate(ROLES))
        k = sum(pi[i] * c["role_k"][r] for i, r in enumerate(ROLES))
        c1 = (counts + k * prior) / (n + k)
        known = [(b, r) for b, r in self.hist if isinstance(r, str)]
        dev = sum(b - c["role_mu"][r] for b, r in known)
        mean = sum(pi[i] * c["role_mu"][r] for i, r in enumerate(ROLES)) + dev / (len(known) + c["k_e"][pos])
        c2 = pb.tilt(c1, mean - c1 @ BANDS)
        roles = [r for _, r in known]
        change = n >= 5 and len(roles) > 0 and ROLES[int(np.argmax(pi))] != max(set(roles), key=roles.count)
        return inc, c1, c2, n, change


def walk_forward(so, c):
    out = []
    for pid, g in so.groupby("Canonical_Player_ID", sort=False):
        s = State()
        for r in g.itertuples():
            score = (r.full and r.Position_Sofascore in ("D", "M", "F")
                     and any(lo <= r.Season <= hi for lo, hi in PERIODS.values()))
            if score:
                inc, c1, c2, n, change = s.forecast(r.Position_Sofascore, c)
                rec = {"pid": pid, "Season": r.Season, "match": r.match, "pos": r.Position_Sofascore,
                       "role": r.role, "band": r.band, "n": n, "change": change}
                for arm, p in zip(ARMS, (inc, c1, c2)):
                    rec["E_" + arm] = p @ BANDS
                    rec["LL_" + arm] = -np.log(max(p[list(BANDS).index(r.band)], 1e-9))
                rec["E_production"] = (lambda p: p @ BANDS)(
                    (np.array([sum(1 for b, _ in s.hist if b == x) for x in BANDS], float)
                     + pb.K_PB[r.Position_Sofascore] * np.array(pb.PRIOR[r.Position_Sofascore]))
                    / (len(s.hist) + pb.K_PB[r.Position_Sofascore]))
                out.append(rec)
            s.update(r.band, r.full, r.role, r.starter)
    return pd.DataFrame(out)


def score(d, arms=ARMS + ["production"]):
    res = {"n": int(len(d)), "matches": int(d["match"].nunique())}
    base = ((d["E_incumbent"] - d["band"]) ** 2).mean()
    for a in arms:
        mse = float(((d["E_" + a] - d["band"]) ** 2).mean())
        res[a] = {"mse": round(mse, 5), "rel_vs_incumbent_pct": round(100 * (mse / base - 1), 2)}
        if "LL_" + a in d:
            res[a]["logloss"] = round(float(d["LL_" + a].mean()), 5)
    return res


def boot(d, a, b="incumbent"):
    keys, idx = np.unique(d["match"], return_inverse=True)
    diff = np.bincount(idx, (d["E_" + a] - d["band"]) ** 2 - (d["E_" + b] - d["band"]) ** 2)
    n = np.bincount(idx)
    rng = np.random.default_rng(SEED)
    w = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys)) for _ in range(REPS)])
    return np.percentile((w @ diff) / (w @ n), [2.5, 97.5]).round(5).tolist()


def bonus_check(so, choice):
    """§68 windows: production arm A against A with the candidate's distribution (constants 2018-2024)."""
    c = fit_constants(so, FIT_BONUS)
    rows, _ = acc.load_rows()
    key = ["Season", "Team", "Opponent", "Venue", "Player"]
    join = so.rename(columns={"Local_Player": "Player"}).dropna(subset=["Player"]).drop_duplicates(key)
    rows = rows.merge(join[key + ["Canonical_Player_ID", "Kickoff", "band"]], on=key, how="left")
    rows["PB_actual"] = rows["band"].fillna(0)
    rows["Canonical_Player_ID"] = rows["Canonical_Player_ID"].fillna("unjoined:" + rows["Player"])
    known = residual.add_known_bps(rows, legacy=False, save_scale=1.0)
    rows["r_rem"] = (rows["BPS"] - known["known_bps"]).astype(float) - rows["PB_actual"]
    rows["Explicit_BPS"] = known["known_bps"].to_numpy(float)
    const = vpb.constants(so)
    wins = {k: v for k, v in acc.windows(rows).items() if v[2]}
    k_rem = vpb.split_half_k(wins["A_2025"][0], "r_rem")
    tested = []
    for name, (fit, test, _) in wins.items():
        test = test.copy()
        table, pos = vpb.player_residuals(fit, "r_rem", k_rem)
        vpb.attach(test, table, pos, "rem")
        season, cutoff = int(test["Season"].iloc[0]), test["Kickoff"].min()
        _, tilted, opp = vpb.band_distributions(test, so, const, season, cutoff)
        states = {}
        for pid, g in so[so["Kickoff"] < cutoff].groupby("Canonical_Player_ID", sort=False):
            s = State()
            for r in g.itertuples():
                s.update(r.band, r.full, r.role, r.starter)
            states[pid] = s
        cand = []
        for i, r in enumerate(test.itertuples()):
            if r.Pos == "G":
                cand.append(tilted[i])
                continue
            p = states.get(r.Canonical_Player_ID, State()).forecast(r.Pos, c)[ARMS.index(choice)]
            cand.append(pb.tilt(p, opp.get(r.Opponent, 0.0) + const["venue"].get(r.Venue, 0.0)))
        rng = np.random.default_rng(vpb.SEED)
        normals, uniforms = {}, {}
        for k_, idx in sorted(test.groupby("Match_Key").indices.items()):
            normals[k_] = rng.standard_normal((len(idx), vpb.N_DRAWS))
            uniforms[k_] = rng.random((len(idx), vpb.N_DRAWS))
        test["p_tilt"] = list(tilted)
        test["pred_A"], _ = vpb.simulate(test, normals, uniforms, "A")
        test["p_tilt"] = list(np.array(cand))
        test["pred_cand"], _ = vpb.simulate(test, normals, uniforms, "A")
        tested.append(test)
        print(f"bonus window {name}: done", flush=True)
    t = pd.concat(tested, ignore_index=True)
    out = {}
    for a in ["A", "cand"]:
        err = t["pred_" + a] - t["Bonus"]
        out[a] = {"mse": float((err ** 2).mean()),
                  "position_bias": err.groupby(t["Pos"]).mean().reindex(acc.POSITIONS).round(4).to_dict()}
    out["rel_pct"] = round(100 * (out["cand"]["mse"] / out["A"]["mse"] - 1), 2)
    out["pass"] = bool(out["cand"]["mse"] <= 1.005 * out["A"]["mse"] and all(
        abs(out["cand"]["position_bias"][p]) - abs(out["A"]["position_bias"][p]) <= 0.03 for p in acc.POSITIONS))
    return out


def main():
    frozen = hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    so = load()
    c = fit_constants(so, FIT)
    d = walk_forward(so, c)
    res = {"constants_fit_2018_2022": {
        "k": c["k"], "role_k": c["role_k"], "k_e": c["k_e"], "role_mu": {k: round(v, 3) for k, v in c["role_mu"].items()},
        "prior": {k: v.round(4).tolist() for k, v in c["prior"].items()}},
        "role_join_rate_60plus": float(so.loc[so["full"], "role"].notna().mean())}
    res["pooled"] = score(d)
    res["pooled"]["ci95_C2_minus_inc"] = boot(d, "C2")
    res["pooled"]["ci95_C1_minus_inc"] = boot(d, "C1")
    for p, (lo, hi) in PERIODS.items():
        res[p] = score(d[d["Season"].between(lo, hi)])
    groups = {"n_le_5": d["n"] <= 5, "n_6_15": d["n"].between(6, 15), "n_gt_15_no_change": (d["n"] > 15) & ~d["change"],
              "role_changers": d["change"]}
    res["subgroups"] = {k: score(d[m]) for k, m in groups.items()}
    res["by_fpl_position"] = {p: score(g) for p, g in d.groupby("pos")}
    res["bias_by_role"] = {r: {a: round(float((g["E_" + a] - g["band"]).mean()), 3) for a in ARMS} | {"n": int(len(g))}
                           for r, g in d.groupby("role")}
    rules = {}
    for arm in ["C2", "C1"]:
        ci = res["pooled"]["ci95_%s_minus_inc" % arm]
        rules[arm] = {"i_mse_1pct_ci_below_zero": res["pooled"][arm]["mse"] <= 0.99 * res["pooled"]["incumbent"]["mse"]
                      and ci[1] < 0,
                      "ii_both_periods": all(res[p][arm]["mse"] < res[p]["incumbent"]["mse"] for p in PERIODS),
                      "iii_logloss": res["pooled"][arm]["logloss"] <= res["pooled"]["incumbent"]["logloss"]}
    passing = [a for a in ["C2", "C1"] if all(rules[a].values())]
    choice = min(passing, key=lambda a: res["pooled"][a]["mse"]) if passing else "C2"
    res["bonus_safety_check_arm"] = choice
    res["bonus_safety_check"] = bonus_check(so, choice)
    for a in rules:
        rules[a]["iv_bonus"] = res["bonus_safety_check"]["pass"] if a == choice else None
    res["rules"] = rules
    res["adopt"] = choice if passing and res["bonus_safety_check"]["pass"] else None
    res["frozen_gw8_unchanged"] = frozen == hashlib.sha256(calibration.FROZEN_FORECAST.read_bytes()).hexdigest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT_DIR / "walk_forward_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
