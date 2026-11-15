"""
League factors anchored on the league coefficient and the league's scoring level (CHANGELOG §119, arm B).

The §116 model (fotmob_priors.FPriorModel) pools every league factor toward one mean per path. Here each slot's
log factor is
    mu_path + s * (coeff_L - coeff_PL) / 10 + c * log(PL level / level_L) + u_L,
with the ridge penalty on u_L only, so a league with few arrivals takes the value its coefficient and scoring
level imply instead of the common mean.
level = mean non-penalty xG per team-match over the league's seasons 2017-2025 with xG (Understat for the PL and
the five Understat leagues, FotMob lists for the 22 FotMob leagues); a league without data gets the mean log level.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fpl_model.player_ratings import fotmob_priors as fp  # noqa: E402
from fpl_model.player_ratings import new_player_priors as npp  # noqa: E402

COEFFS = npp.DATA / "league_coefficients_20261029.csv"
UNDERSTAT_NAME = {v: k for k, v in npp.UNDERSTAT_LEAGUE.items()}
UNDERSTAT_NAME["EPL"] = "Premier League"


def league_levels(fm=None):
    """League -> mean npxG per team-match (seasons 2017-2025 with xG)."""
    per = {}
    for path in sorted(npp.LEAGUE_CACHE.glob("*_*.json")):
        lg, season = path.stem.rsplit("_", 1)
        if not season.isdigit() or not 2017 <= int(season) <= 2025 or lg not in UNDERSTAT_NAME:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        x = sum(float(m["npxG"]) for t in d["teams"].values() for m in t["history"])
        n = sum(len(t["history"]) for t in d["teams"].values())
        if n:
            per.setdefault(UNDERSTAT_NAME[lg], []).append(x / n)
    fm = fp.load_fotmob() if fm is None else fm
    for (lg, s), g in fm[fm["Has_xG"]].groupby(["League", "Season"]):
        if s > 2025:
            continue
        n = g.groupby("TeamId")["Matches"].max().sum()
        if n:
            per.setdefault(lg, []).append(g["npxG"].sum() / n)
    return {lg: float(np.mean(v)) for lg, v in per.items()}


def slot_covariates(codes, levels, coeffs):
    """Per slot code ('U:La_liga', 'F:Championship', 'M:Eredivisie'): (coefficient term, level term)."""
    pl_c, pl_l = coeffs["Premier League"], levels["Premier League"]
    mean_log = float(np.mean([np.log(pl_l / v) for k, v in levels.items() if k != "Premier League"]))
    cz, lv = {}, {}
    for code in codes:
        if not code:
            continue
        name = code[2:]
        name = UNDERSTAT_NAME.get(name, name) if code.startswith("U:") else name
        cz[code] = (coeffs.get(name, np.nan) - pl_c) / 10.0
        lv[code] = np.log(pl_l / levels[name]) if name in levels else mean_log
        if np.isnan(cz[code]):
            raise ValueError(f"league {name!r} has no coefficient in {COEFFS.name}")
    return cz, lv


class BPriorModel(fp.FPriorModel):
    """theta = [d, beta_U, b, M0, gamma, mu_U, mu_M, alpha, mu_F, beta_F, s, c, u...]."""
    C_FIXED = None   # None: c fitted (§119 arm B); a number: c held at it (§231 arm C)

    def __init__(self, target, lam=1.0, cz=None, lv=None):
        super().__init__(target, lam=lam)
        self.cz, self.lv = cz, lv

    def _unpack(self, theta, leagues):
        p = super()._unpack(np.r_[theta[:10], theta[12:]], leagues)
        p["s"], p["c"] = theta[10], theta[11]
        p["cz"], p["lv"] = self.cz, self.lv
        return p

    @staticmethod
    def predict_with(p, arr, slots, target, pre=None):
        pre = npp.PriorModel.arrays(arr, target) if pre is None else pre
        lg = slots["league"]
        if "_codes" not in slots:
            slots["_codes"], inv = np.unique(lg.astype(str), return_inverse=True)
            slots["_inv"] = inv.reshape(lg.shape)

        def logf(L):
            mu = p["mu_U"] if L.startswith("U:") else p["mu_F"] if L.startswith("F:") else p["mu_M"]
            return mu + p["u"].get(L, 0.0) + p["s"] * p["cz"][L] + p["c"] * p["lv"][L]

        table = np.array([0.0 if not L else logf(L) for L in slots["_codes"]])
        btab = np.array([p["beta"] if L.startswith("U:") else p["beta_F"] if L.startswith("F:") else 0.0
                         for L in slots["_codes"]])
        w = p["d"] ** np.arange(lg.shape[1])
        V = (slots["m"] / 90.0 * w).sum(1)
        X = (w * np.exp(table[slots["_inv"]] + btab[slots["_inv"]] * slots["z"]) * slots["x"][target]).sum(1)
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
                       0.0, 0.0 if self.C_FIXED is None else self.C_FIXED, np.zeros(len(leagues))]
        bounds = [(-6, 6), (-2, 3), (np.log(0.2), np.log(5)), (np.log(10), np.log(20000)), (-1, 2),
                  (np.log(0.1), np.log(3)), (np.log(0.1), np.log(3)), (-0.3, 0.3),
                  (np.log(0.1), np.log(3)), (-2, 3), (-3, 3),
                  (-3, 3) if self.C_FIXED is None else (self.C_FIXED, self.C_FIXED)] + [(-2, 2)] * len(leagues)
        res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds)
        self.p = self._unpack(res.x, leagues)
        self.converged = bool(res.success)
        return self

    def summary(self):
        out = super().summary()
        p = self.p
        out["s"], out["c"] = float(p["s"]), float(p["c"])
        out["f"] = {L: float(np.exp((p["mu_U"] if L.startswith("U:") else p["mu_F"] if L.startswith("F:")
                                     else p["mu_M"]) + u + p["s"] * p["cz"][L] + p["c"] * p["lv"][L]))
                    for L, u in p["u"].items()}
        return out


class CPriorModel(BPriorModel):
    """§231 arm C: B with the scoring-level exponent fixed at 1 (output as a share of league scoring)."""
    C_FIXED = 1.0


def fit_with_cv(arr, slots, target, cz, lv, model=BPriorModel):
    """Ridge strength by leave-one-season-out CV inside the training arrivals (as npp.fit_with_cv)."""
    best = None
    for lam in npp.LAMBDAS:
        tot = 0.0
        for s in sorted(arr["Season"].unique()):
            tr, trs = npp.subset(arr, slots, arr["Season"] != s)
            te, tes = npp.subset(arr, slots, arr["Season"] == s)
            mdl = model(target, lam, cz, lv).fit(tr, trs)
            tot += npp.sse(mdl.predict(te, tes), te, target).sum()
        if best is None or tot < best[1]:
            best = (lam, tot)
    mdl = model(target, best[0], cz, lv).fit(arr, slots)
    mdl.lam_cv = best[0]
    return mdl


def load_coeffs():
    c = pd.read_csv(COEFFS, encoding="utf-8-sig")
    return dict(zip(c["League"], c["Coefficient"].astype(float)))
