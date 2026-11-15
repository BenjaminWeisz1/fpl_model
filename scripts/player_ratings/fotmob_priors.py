"""
FotMob path for new-player priors (CHANGELOG §116): player-season table from the frozen FotMob lists,
arrival matching, and the prior model with a third (FotMob) path.

The §103 model (new_player_priors.PriorModel) has two paths, Understat ("U:" leagues) and manual ("M:").
This adds "F:" slots: up to three FotMob seasons, their own pooled mean mu_F and club-strength coefficient
beta_F; d, b, M0 and gamma are shared with the other paths.
"""
import difflib
import json
import re
import unicodedata
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fetch"))
import new_player_priors as npp  # noqa: E402
from fetch_fotmob_leagues import LEAGUES, LEAGUE_DIR, FIRST_SEASON, CALENDAR_LEAGUES  # noqa: E402

FID_LEAGUE = {fid: name for name, (fid, _) in LEAGUES.items()}


def _norm(name):
    """Name tokens, accents dropped (same rule as build_new_player_priors._norm; kept here to avoid a circular import)."""
    t = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", " ", t).split()
OVERRIDES = npp.DATA / "fotmob_id_overrides.csv"
MIN_TEAMS = 8          # league-seasons listing fewer clubs are fragments and are dropped
MIN_LIST_SHARE = 0.5   # an xG (xA) list summing to less than this share of goals (assists) is partial


def _list(path):
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for tl in d.get("TopLists", []) for r in tl["StatList"]]
    return pd.DataFrame(rows) if rows else None


def load_fotmob():
    """One row per (league, season, FotMob player): minutes, npxG-or-NPG, xA-or-assists, club strength z."""
    out = []
    keys = sorted({tuple(p.stem.split("_")[:2]) for p in LEAGUE_DIR.glob("*_*_*.json")})
    for fid, season in keys:
        fid, season = int(fid), int(season)
        L = {s: _list(LEAGUE_DIR / f"{fid}_{season}_{s}.json") for s in
             ("mins_played", "expected_goals", "expected_assists", "goals", "goal_assist")}
        if L["mins_played"] is None:
            continue
        mp = L["mins_played"]
        df = pd.DataFrame({"FM_ID": mp["ParticiantId"], "Name": mp["ParticipantName"], "TeamId": mp["TeamId"],
                           "Minutes": mp["StatValue"].astype(float), "Matches": mp["MatchesPlayed"]})

        def col(name, field="StatValue"):
            s = L[name]
            return dict(zip(s["ParticiantId"], s[field].astype(float))) if s is not None else {}

        pen = col("goals", "SubStatValue")
        goals = col("goals")
        if df["TeamId"].nunique() < MIN_TEAMS:     # a fragment (e.g. Greece 2021 lists two clubs)
            continue
        # Some xG / xA lists are partial (e.g. Brazil 2020-22: summed xG ~0.1-0.2 of goals); a player missing
        # from one would get 0. Use such a list only if it sums to at least half the goals (assists) (§119).
        has_xg = L["expected_goals"] is not None and \
            L["expected_goals"]["StatValue"].sum() >= MIN_LIST_SHARE * sum(goals.values())
        has_xa = L["expected_assists"] is not None and \
            L["expected_assists"]["StatValue"].sum() >= MIN_LIST_SHARE * sum(col("goal_assist").values())
        df["PenGoals"] = df["FM_ID"].map(pen).fillna(0.0)
        df["NPG"] = (df["FM_ID"].map(goals).fillna(0.0) - df["PenGoals"]).clip(lower=0)
        if has_xg:
            df["npxG"] = (df["FM_ID"].map(col("expected_goals")).fillna(0.0) - df["PenGoals"]).clip(lower=0)
        else:
            df["npxG"] = df["NPG"]
        df["xA"] = df["FM_ID"].map(col("expected_assists") if has_xa else col("goal_assist")).fillna(0.0)
        df["Has_xG"], df["Has_xA"] = has_xg, has_xa
        team = df.groupby("TeamId").agg(x=("npxG", "sum"), n=("Matches", "max"))
        rate = team["x"] / team["n"].clip(lower=1)
        rate = rate[rate > 0]
        df["z"] = df["TeamId"].map(np.log(rate / rate.mean())).fillna(0.0)
        df["League"], df["Season"] = FID_LEAGUE[fid], season
        out.append(df)
    return pd.concat(out, ignore_index=True)


def match_id(player, pool):
    """Unique FotMob ID for a transfers.csv name in a league-season pool (rule of name_match_ids)."""
    tok = set(_norm(player))
    names = list(zip(pool["FM_ID"], pool["Name"]))
    hits = {int(i) for i, n in names
            if tok and (tok <= set(_norm(n)) or (len(set(_norm(n))) >= 2 and set(_norm(n)) <= tok))}
    if not hits:
        full = " ".join(_norm(player))
        hits = {int(i) for i, n in names if difflib.SequenceMatcher(None, full, " ".join(_norm(n))).ratio() >= 0.88}
    return hits.pop() if len(hits) == 1 else None


def last_season(league, season):
    """The FotMob season an arrival in PL season S last played: S-1 for European seasons (2025 = 2025/26),
    S itself for calendar-year leagues, which he leaves part-way through (transfers.csv's Min_Old convention)."""
    return season if league in CALENDAR_LEAGUES else season - 1


def attach_fotmob(arr, slots, fm):
    """Replace the slots of arrivals found in FotMob with their FotMob seasons (last season and the two before,
    >= FIRST_SEASON). Override IDs without a row in the last season fall back like an unmatched name."""
    arr = arr.copy()
    slots = {"m": slots["m"].copy(), "z": slots["z"].copy(), "league": slots["league"].copy(),
             "x": {k: v.copy() for k, v in slots["x"].items()}}
    by_id = {k: g for k, g in fm[fm["Minutes"] > 0].groupby("FM_ID")}
    overrides = {}
    if OVERRIDES.exists():   # hand-found FotMob IDs for names the matcher misses (user-maintained)
        o = pd.read_csv(OVERRIDES, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        overrides = {(r.Player, int(r.Season)): int(r.FotMob_ID) for r in o.itertuples() if r.FotMob_ID.strip()}
    pools = {k: g for k, g in fm[fm["Minutes"] > 0].groupby(["League", "Season"])}
    ids, n_seasons, no_xg = [], [], []
    K = slots["m"].shape[1]
    for i, r in enumerate(arr.itertuples()):
        fid = None
        last = last_season(r.League, r.Season)
        if r.League in LEAGUES and last >= FIRST_SEASON and (r.League, last) in pools:
            fid = overrides.get((r.Player, r.Season)) or match_id(r.Player, pools[(r.League, last)])
            if fid is not None and (fid not in by_id or not (by_id[fid]["Season"] == last).any()):
                if (r.Player, r.Season) in overrides:
                    print(f"[fotmob] override ID {fid} for {r.Player} ({r.Season}) has no {last} row; "
                          f"transfers.csv values kept")
                fid = None
        ids.append(fid)
        if fid is None:
            n_seasons.append(0); no_xg.append(False)
            continue
        g = by_id[fid]
        slots["m"][i, :] = 0; slots["z"][i, :] = 0; slots["league"][i, :] = ""
        for v in slots["x"].values():
            v[i, :] = 0
        found, lacks = 0, False
        for k in range(K):
            rows = g[g["Season"] == last - k]
            if not len(rows):
                continue
            found += 1
            lacks |= not rows["Has_xG"].all()
            slots["m"][i, k] = rows["Minutes"].sum()
            slots["x"]["npxG"][i, k] = rows["npxG"].sum()
            slots["x"]["xAG"][i, k] = rows["xA"].sum()
            slots["league"][i, k] = "F:" + rows.sort_values("Minutes").iloc[-1]["League"]
            slots["z"][i, k] = float(np.average(rows["z"], weights=rows["Minutes"]))
        n_seasons.append(found); no_xg.append(lacks)
    arr["FM_ID"] = ids
    arr["FM_Seasons"] = n_seasons
    arr["FM_NoXG"] = no_xg
    arr.loc[arr["FM_ID"].notna(), "Path"] = "fotmob"
    return arr, slots


class FPriorModel(npp.PriorModel):
    """§103 prior plus the FotMob path: theta = [d, beta_U, b, M0, gamma, mu_U, mu_M, alpha, mu_F, beta_F, u...]."""

    def __init__(self, target, use_strength=True, lam=1.0, use_age=False, use_strength_f=True):
        super().__init__(target, use_strength, lam, use_age)
        self.use_strength_f = use_strength_f

    def _unpack(self, theta, leagues):
        p = super()._unpack(np.r_[theta[:8], theta[10:]], leagues)
        p["mu_F"] = theta[8]
        p["beta_F"] = theta[9] if self.use_strength_f else 0.0
        return p

    @staticmethod
    def predict_with(p, arr, slots, target, pre=None):
        pre = npp.PriorModel.arrays(arr, target) if pre is None else pre
        lg = slots["league"]
        if "_codes" not in slots:
            slots["_codes"], inv = np.unique(lg.astype(str), return_inverse=True)
            slots["_inv"] = inv.reshape(lg.shape)

        def mu(L):
            return p["mu_U"] if L.startswith("U:") else p.get("mu_F", 0.0) if L.startswith("F:") else p["mu_M"]

        table = np.array([0.0 if not L else mu(L) + p["u"].get(L, 0.0) for L in slots["_codes"]])
        btab = np.array([p["beta"] if L.startswith("U:") else p.get("beta_F", 0.0) if L.startswith("F:") else 0.0
                         for L in slots["_codes"]])
        logf = table[slots["_inv"]]
        beta = btab[slots["_inv"]]
        w = p["d"] ** np.arange(lg.shape[1])
        V = (slots["m"] / 90.0 * w).sum(1)
        X = (w * np.exp(logf + beta * slots["z"]) * slots["x"][target]).sum(1)
        return ((X + p["b"] * pre["pos"] * p["M0"] / 90) / (V + p["M0"] / 90) * pre["tar"] ** p["gamma"]
                * np.exp(p.get("alpha", 0.0) * pre["age"]))

    def fit(self, arr, slots, tag="all"):
        t = self.target
        leagues = sorted({L for L in slots["league"].ravel() if L})
        A = arr[f"{tag}_{t}_A"].to_numpy(); B = arr[f"{tag}_{t}_B"].to_numpy()
        scale = max(B.sum(), 1.0) / 1000.0
        pre = self.arrays(arr, t)

        def loss(theta):
            p = self._unpack(theta, leagues)
            pr = self.predict_with(p, arr, slots, t, pre)
            return ((B * pr ** 2 - 2 * A * pr).sum() / scale
                    + self.lam * sum(v ** 2 for v in p["u"].values()))

        theta0 = np.r_[0.0, 0.0, 0.0, np.log(500.0), 0.15, np.log(0.7), np.log(0.7), 0.0, np.log(0.7), 0.0,
                       np.zeros(len(leagues))]
        bounds = [(-6, 6), (-2, 3), (np.log(0.2), np.log(5)), (np.log(10), np.log(20000)), (-1, 2),
                  (np.log(0.1), np.log(3)), (np.log(0.1), np.log(3)), (-0.3, 0.3),
                  (np.log(0.1), np.log(3)), (-2, 3)] + [(-2, 2)] * len(leagues)
        res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds)
        self.p = self._unpack(res.x, leagues)
        self.converged = bool(res.success)
        return self

    def summary(self):
        out = super().summary()
        p = self.p
        out["f"] = {L: float(np.exp((p["mu_U"] if L.startswith("U:") else p["mu_F"] if L.startswith("F:")
                                     else p["mu_M"]) + u)) for L, u in p["u"].items()}
        out["f_mean_fotmob"] = float(np.exp(p["mu_F"]))
        out["beta_F"] = float(p["beta_F"])
        return out


def fit_with_cv(arr, slots, target, use_strength_f=True):
    """Ridge strength by leave-one-season-out CV inside the training arrivals (as npp.fit_with_cv)."""
    seasons = sorted(arr["Season"].unique())
    best = None
    for lam in npp.LAMBDAS:
        tot = 0.0
        for s in seasons:
            tr, trs = npp.subset(arr, slots, arr["Season"] != s)
            te, tes = npp.subset(arr, slots, arr["Season"] == s)
            mdl = FPriorModel(target, lam=lam, use_strength_f=use_strength_f).fit(tr, trs)
            tot += npp.sse(mdl.predict(te, tes), te, target).sum()
        if best is None or tot < best[1]:
            best = (lam, tot)
    mdl = FPriorModel(target, lam=best[0], use_strength_f=use_strength_f).fit(arr, slots)
    mdl.lam_cv = best[0]
    return mdl
