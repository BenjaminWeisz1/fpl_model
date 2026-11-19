"""
New-to-the-PL players' goal/assist priors (AGENDA "OVERNIGHT, new-player priors"; CHANGELOG §103).

Shared by the validation (`validate_new_player_priors.py`) and the production writer
(`build_new_player_priors.py`). Three parts:

  * Understat season data for La Liga, Bundesliga, Serie A, Ligue 1 and the RFPL
    (`data/raw/understat/league/<LEAGUE>_<season>.json`, `fetch_understat_leagues.py`): player
    season totals plus each team-season's attacking strength,
        z = log(team npxG per match / league-season mean team npxG per match).
  * The arrivals dataset: `transfers.csv` rows with blank Notes, a non-PL League and a non-GK
    position. Each arrival gets up to three prior Understat seasons (S-3 .. S-1, by Understat
    player ID) when his League is an Understat league, else his manual `transfers.csv` values.
  * The candidate prior, per target (npxG or xAG), on the per-90 rating scale:
        V = sum_k d^(k-1) m_k / 90            (volume, recency-weighted, k = 1 is S-1)
        X = sum_k d^(k-1) f_L(k) e^(beta z_k) x_k
        prior = (X + b p M0/90) / (V + M0/90) x (TAR / 1.32023)^gamma
    p = position prior, TAR = the new PL team's attack rating (transfers.csv), f_L = league
    translation factor. log f_L = mu_path + u_L with a ridge penalty lam * sum u_L^2 (partial
    pooling toward one mean for Understat xG and one for the manual Opta/FotMob values). The
    manual path has one season and no team strength (no source team is recorded).
    Parameters are fitted by minimising the squared error of match output = prior x minutes/90
    over the arrivals' first-season PL matches, which reduces to per-player sufficient statistics.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from fpl_model import paths
from fpl_model.player_ratings import (
    position_rates as _pr,  # §246 (GOAL_MODEL_POSITION_RATES; default "file" changes nothing)
)

ROOT = paths.ROOT
DATA = paths.CURRENT
LEAGUE_CACHE = paths.RAW / "understat" / "league"

AVG_CONCEDED = 1.32023
UNDERSTAT_LEAGUE = {"La Liga": "La_liga", "Bundesliga": "Bundesliga", "Serie A": "Serie_A",
                    "Ligue 1": "Ligue_1", "Russian Premier League": "RFPL"}
REST_OF_WORLD = {"Brazilian Série A", "Brazilian Série B", "Liga Argentina", "MLS", "Paraguayan Primera Div",
                 "K League", "J. League", "Liga MX", "Saudi Pro League", "Chinese Super League",
                 "Ecuador Serie A"}
N_PRIOR_SEASONS = 3
TARGETS = {"npxG": ("npxG", "npxG_Old", "NPG_Old", "NPG/90"),
           "xAG": ("xA", "xAG_Old", "Assists_Old", "Assists/90")}


def league_group(league):
    if league in ("La Liga", "Bundesliga", "Serie A"):
        return "top3"
    if league == "Ligue 1":
        return "ligue1"
    if league == "Championship":
        return "championship"
    if league in REST_OF_WORLD:
        return "rest_of_world"
    return "other_europe"


# ── Understat league data ─────────────────────────────────────────────────────

def load_understat():
    """Player-season totals and team-season strength for the five Understat leagues."""
    players, teams = [], []
    for path in sorted(LEAGUE_CACHE.glob("*_*.json")):
        league, season = path.stem.rsplit("_", 1)
        if league == "EPL" or not season.isdigit():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        season = int(season)
        tstr = {}
        for t in d["teams"].values():
            h = t["history"]
            if h:
                tstr[t["title"]] = sum(float(m["npxG"]) for m in h) / len(h)
        mean = np.mean(list(tstr.values()))
        for title, v in tstr.items():
            teams.append({"ULeague": league, "Season": season, "Team": title, "z": float(np.log(v / mean))})
        for p in d["players"]:
            players.append({"ULeague": league, "Season": season, "Understat_ID": int(p["id"]),
                            "Name": p["player_name"], "Team": p["team_title"], "Minutes": float(p["time"]),
                            "npxG": float(p["npxG"]), "xA": float(p["xA"]), "Position": p["position"]})
    players, teams = pd.DataFrame(players), pd.DataFrame(teams)
    zmap = {(r.ULeague, r.Season, r.Team): r.z for r in teams.itertuples()}
    # a player who moved within the league lists "A,B": average the teams' strengths
    players["z"] = [np.mean([zmap.get((r.ULeague, r.Season, t.strip()), 0.0) for t in str(r.Team).split(",")])
                    for r in players.itertuples()]
    return players, teams


# ── position priors (identical rule to build_player_ratings.py, CHANGELOG §77) ──

def position_priors(positions, seasons=None):
    """{Pos: {NPG/90, Assists/90}} from position_combinations.csv. With `seasons` (one per position) and
    GOAL_MODEL_POSITION_RATES = slots | slots3 (§246), keyed {(Pos, Season): ...} on that season's slot rates."""
    if seasons is not None and _pr.WINDOW is not None:
        return {(pos, int(S)): _pr.combined_rate(pos, int(S)) for pos, S in set(zip(positions, seasons))}
    single = pd.read_csv(paths.current_file("position_combinations.csv", root=DATA)).set_index("Pos")
    out = {}
    for pos in set(positions):
        parts = [p.strip() for p in str(pos).split(",")]
        w = [2.0 ** (len(parts) - 1 - i) for i in range(len(parts))]
        out[pos] = {c: float(np.average(single.loc[parts, c], weights=w)) for c in ["NPG/90", "Assists/90"]}
    return out


# ── arrivals dataset ──────────────────────────────────────────────────────────

def in_scope_transfers():
    t = pd.read_csv(paths.current_file("transfers.csv", root=DATA), encoding="utf-8-sig")
    for c in ["Min_Old", "npxG_Old", "xAG_Old", "NPG_Old", "Assists_Old", "Team_Attack_Rating"]:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    return t[t["Notes"].isna() & (t["League"] != "Premier League") & (t["Pos"] != "GK") & (t["Season"] > 2016)].copy()


def understat_ids():
    """PMS player name -> Understat ID, from the crosswalk (both name conventions)."""
    cw = pd.read_csv(paths.current_file("understat_fpl_crosswalk.csv", root=DATA), encoding="utf-8-sig")
    ids = {}
    for col in ["PMS_Player_fbref", "PMS_Player_FPL"]:
        for name, uid in zip(cw[col], cw["Player_ID_Understat"]):
            if isinstance(name, str):
                ids.setdefault(name, int(uid))
    return ids


def attach_priors(arr, uplayers, ids=None):
    """Add the per-slot prior-season arrays (Understat path) or the manual inputs."""
    ids = understat_ids() if ids is None else ids
    arr = arr.copy()
    arr["Understat_ID"] = arr["Player"].map(ids)
    arr["Group"] = arr["League"].map(league_group)
    if _pr.WINDOW is None:
        pp = position_priors(arr["Pos"])
        arr["pos_npxG"] = arr["Pos"].map(lambda p: pp[p]["NPG/90"])
        arr["pos_xAG"] = arr["Pos"].map(lambda p: pp[p]["Assists/90"])
    else:   # §246: the arrival season's slot rates
        pp = position_priors(arr["Pos"], arr["Season"])
        arr["pos_npxG"] = [pp[(p, int(S))]["NPG/90"] for p, S in zip(arr["Pos"], arr["Season"])]
        arr["pos_xAG"] = [pp[(p, int(S))]["Assists/90"] for p, S in zip(arr["Pos"], arr["Season"])]
    by_id = {k: g for k, g in uplayers.groupby("Understat_ID")}
    K = N_PRIOR_SEASONS
    n = len(arr)
    m = np.zeros((n, K))
    z = np.zeros((n, K))
    lg = np.full((n, K), "", dtype=object)
    xs = {"npxG": np.zeros((n, K)), "xAG": np.zeros((n, K))}
    path = []
    for i, r in enumerate(arr.itertuples()):
        use_u = False
        if r.League in UNDERSTAT_LEAGUE and r.Understat_ID == r.Understat_ID and int(r.Understat_ID) in by_id:
            g = by_id[int(r.Understat_ID)]
            for k in range(K):
                rows = g[(g["Season"] == r.Season - 1 - k) & (g["Minutes"] > 0)]
                if len(rows):
                    use_u = True
                    m[i, k] = rows["Minutes"].sum()
                    xs["npxG"][i, k] = rows["npxG"].sum()
                    xs["xAG"][i, k] = rows["xA"].sum()
                    # league and strength of the season's main club (most minutes)
                    top = rows.sort_values("Minutes").iloc[-1]
                    lg[i, k] = "U:" + top["ULeague"]
                    z[i, k] = float(np.average(rows["z"], weights=rows["Minutes"]))
        if not use_u:
            m[i, :] = 0
            z[i, :] = 0
            lg[i, :] = ""
            xs["npxG"][i, :] = 0
            xs["xAG"][i, :] = 0
            m[i, 0] = r.Min_Old if r.Min_Old == r.Min_Old else 0.0
            lg[i, 0] = "M:" + r.League
            xs["npxG"][i, 0] = r.npxG_Old if r.npxG_Old == r.npxG_Old else (r.NPG_Old if r.NPG_Old == r.NPG_Old else 0.0)
            xs["xAG"][i, 0] = r.xAG_Old if r.xAG_Old == r.xAG_Old else (r.Assists_Old if r.Assists_Old == r.Assists_Old else 0.0)
        path.append("understat" if use_u else "manual")
    arr["Path"] = path
    arr = arr.reset_index(drop=True)
    return arr, {"m": m, "z": z, "league": lg, "x": xs}


LOG_COLS = {"npxG": ("npxg90", "pred_goals_90", "NPG_Rating_Pre"),
            "xAG": ("xag90", "pred_ast_90", "Assist_Rating_Pre")}


def attach_outcomes(arr, pms, log):
    """First-season PL outcome sufficient statistics, from the pre-match log's rows (minutes > 0).

    Match prediction = prior x the row's loop factor (venue x opponent x substitute uplift, the same
    for every arm: pred_90 / rating in the log) x minutes/90. Tags: all = every first-season match,
    f10 = the first 10, plain = all matches without the row factor (prior x minutes/90)."""
    pms = pms[pms["Minutes"] > 0].copy()
    pms["Season"] = np.where(pms["Data_Source"] == "fbref", 2017 + pms["MatchID"] // 380, pms["Season"])
    first_pl = pms.groupby("Player")["Season"].min()
    arr = arr.copy()
    arr["First_PL_Season"] = arr["Player"].map(first_pl)
    arr["New"] = arr["First_PL_Season"] == arr["Season"]
    key = set(zip(arr["Player"], arr["Season"]))
    sub = log[[k in key for k in zip(log["Player"], log["Season"])]].copy()
    sub["rank"] = sub.groupby(["Player", "Season"]).cumcount()
    stats = {}
    for tag, frame in [("all", sub), ("f10", sub[sub["rank"] < 10]), ("plain", sub)]:
        for t, (act, pred, rating) in LOG_COLS.items():
            y = frame[act] * frame["weight"]
            factor = 1.0 if tag == "plain" else (frame[pred] / frame[rating]).fillna(1.0)
            mm = frame["weight"] * factor
            g = pd.DataFrame({"Player": frame["Player"], "Season": frame["Season"], "A": y * mm,
                              "B": mm ** 2, "C": y ** 2, "Y": y, "M": mm, "N": 1.0})
            agg = g.groupby(["Player", "Season"]).sum()
            for c in ["A", "B", "C", "Y", "M", "N"]:
                stats[f"{tag}_{t}_{c}"] = agg[c]
    out = pd.DataFrame(stats).reset_index()
    return arr.merge(out, on=["Player", "Season"], how="left").fillna(
        {c: 0.0 for c in out.columns if c not in ("Player", "Season")})


def incumbent_from_log(arr, log):
    """The incumbent prior = each arrival's first logged pre-match rating in his arrival season."""
    first = log.groupby(["Player", "Season"], sort=False).first().reset_index()
    first = first.rename(columns={"NPG_Rating_Pre": "inc_npxG", "Assist_Rating_Pre": "inc_xAG"})
    return arr.merge(first[["Player", "Season", "inc_npxG", "inc_xAG"]], on=["Player", "Season"], how="left")


def birth_dates():
    """Local player name -> birth timestamp, from the cached Sofascore lineups (dateOfBirthTimestamp)."""
    dob = {}
    for f in sorted((ROOT / "data" / "raw" / "sofascore" / "matches").glob("*/*.json")):
        p = json.loads(f.read_text(encoding="utf-8")).get("payload") or {}
        for side in ("home", "away"):
            for pl in (p.get(side) or {}).get("players", []):
                q = pl.get("player", {})
                if q.get("dateOfBirthTimestamp") is not None:
                    dob[q["id"]] = q["dateOfBirthTimestamp"]
    s = pd.read_csv(paths.current_file("sofascore_player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False,
                    usecols=["Player_ID_Sofascore", "Local_Player"]).dropna().drop_duplicates()
    s["dob"] = s["Player_ID_Sofascore"].map(dob)
    return s.dropna(subset=["dob"]).groupby("Local_Player")["dob"].agg(lambda x: x.mode().iloc[0]).to_dict()


def attach_age(arr, dob):
    start = pd.to_datetime(arr["Season"].astype(int).astype(str) + "-08-01").astype("int64") // 10 ** 9
    ts = arr["Player"].map(dob)
    arr = arr.copy()
    arr["Age"] = ((start - ts) / (365.25 * 86400)).fillna(25.0)
    arr["Has_DOB"] = ts.notna()
    return arr


def build_dataset(log_path):
    uplayers, _ = load_understat()
    pms = pd.read_csv(paths.current_file("player_match_stats.csv", root=DATA), encoding="utf-8-sig", low_memory=False)
    log = pd.read_csv(log_path, encoding="utf-8-sig")
    arr = attach_outcomes(in_scope_transfers(), pms, log)
    arr, slots = attach_priors(arr, uplayers)
    arr = incumbent_from_log(arr, log)
    arr = attach_age(arr, birth_dates())
    return arr, slots


# ── candidate model ───────────────────────────────────────────────────────────

class PriorModel:
    """The candidate prior for one target. Fitted parameters live in self.p (a dict)."""

    def __init__(self, target, use_strength=True, lam=1.0, use_age=False):
        self.target, self.use_strength, self.lam, self.use_age = target, use_strength, lam, use_age
        self.p = None

    def _unpack(self, theta, leagues):
        p = {"d": 1 / (1 + np.exp(-theta[0])), "beta": theta[1] if self.use_strength else 0.0,
             "b": np.exp(theta[2]), "M0": np.exp(theta[3]), "gamma": theta[4],
             "mu_U": theta[5], "mu_M": theta[6], "alpha": theta[7] if self.use_age else 0.0}
        p["u"] = dict(zip(leagues, theta[8:]))
        return p

    @staticmethod
    def arrays(arr, target):
        """The per-arrival inputs the prior needs, as plain arrays (kept out of the optimiser's loss)."""
        return {"pos": arr[f"pos_{target}"].to_numpy(dtype=float),
                "tar": (arr["Team_Attack_Rating"].fillna(AVG_CONCEDED).to_numpy(dtype=float) / AVG_CONCEDED),
                "age": arr["Age"].to_numpy(dtype=float) - 25.0 if "Age" in arr else np.zeros(len(arr))}

    @staticmethod
    def predict_with(p, arr, slots, target, pre=None):
        pre = PriorModel.arrays(arr, target) if pre is None else pre
        lg = slots["league"]
        if "_codes" not in slots:
            slots["_codes"], inv = np.unique(lg.astype(str), return_inverse=True)
            slots["_inv"] = inv.reshape(lg.shape)
        table = np.array([0.0 if not L else (p["mu_U"] if L.startswith("U:") else p["mu_M"]) + p["u"].get(L, 0.0)
                          for L in slots["_codes"]])
        logf = table[slots["_inv"]]
        w = p["d"] ** np.arange(lg.shape[1])
        V = (slots["m"] / 90.0 * w).sum(1)
        X = (w * np.exp(logf + p["beta"] * slots["z"]) * slots["x"][target]).sum(1)
        return ((X + p["b"] * pre["pos"] * p["M0"] / 90) / (V + p["M0"] / 90) * pre["tar"] ** p["gamma"]
                * np.exp(p.get("alpha", 0.0) * pre["age"]))

    def fit(self, arr, slots, tag="all"):
        t = self.target
        leagues = sorted({L for L in slots["league"].ravel() if L})
        A = arr[f"{tag}_{t}_A"].to_numpy()
        B = arr[f"{tag}_{t}_B"].to_numpy()
        scale = max(B.sum(), 1.0) / 1000.0  # keeps the loss O(1) per thousand match-90s
        pre = self.arrays(arr, t)

        def loss(theta):
            p = self._unpack(theta, leagues)
            pr = self.predict_with(p, arr, slots, t, pre)
            return ((B * pr ** 2 - 2 * A * pr).sum() / scale
                    + self.lam * sum(v ** 2 for v in p["u"].values()))

        theta0 = np.r_[0.0, 0.0, 0.0, np.log(500.0), 0.15, np.log(0.7), np.log(0.7), 0.0, np.zeros(len(leagues))]
        bounds = [(-6, 6), (-2, 3), (np.log(0.2), np.log(5)), (np.log(10), np.log(20000)), (-1, 2),
                  (np.log(0.1), np.log(3)), (np.log(0.1), np.log(3)), (-0.3, 0.3)] + [(-2, 2)] * len(leagues)
        res = minimize(loss, theta0, method="L-BFGS-B", bounds=bounds)
        self.p = self._unpack(res.x, leagues)
        self.converged = bool(res.success)
        return self

    def predict(self, arr, slots):
        return self.predict_with(self.p, arr, slots, self.target)

    def summary(self):
        p = self.p
        out = {k: float(p[k]) for k in ["d", "beta", "b", "M0", "gamma", "alpha"]}
        out["f"] = {L: float(np.exp((p["mu_U"] if L.startswith("U:") else p["mu_M"]) + u)) for L, u in p["u"].items()}
        out["f_mean_understat"] = float(np.exp(p["mu_U"]))
        out["f_mean_manual"] = float(np.exp(p["mu_M"]))
        return out


def subset(arr, slots, mask):
    mask = np.asarray(mask)
    return arr[mask].reset_index(drop=True), {"m": slots["m"][mask], "z": slots["z"][mask],
                                              "league": slots["league"][mask],
                                              "x": {k: v[mask] for k, v in slots["x"].items()}}


def sse(pred, arr, t, tag="all"):
    return arr[f"{tag}_{t}_C"].to_numpy() - 2 * pred * arr[f"{tag}_{t}_A"].to_numpy() + pred ** 2 * arr[f"{tag}_{t}_B"].to_numpy()


LAMBDAS = [0.3, 1.0, 3.0, 10.0, 30.0]


def fit_with_cv(arr, slots, target, use_strength=True, use_age=False):
    """Ridge strength chosen by leave-one-season-out CV inside the training arrivals."""
    seasons = sorted(arr["Season"].unique())
    best = None
    if len(seasons) >= 2:
        for lam in LAMBDAS:
            tot = 0.0
            for s in seasons:
                tr, trs = subset(arr, slots, arr["Season"] != s)
                te, tes = subset(arr, slots, arr["Season"] == s)
                mdl = PriorModel(target, use_strength, lam, use_age).fit(tr, trs)
                tot += sse(mdl.predict(te, tes), te, target).sum()
            if best is None or tot < best[1]:
                best = (lam, tot)
    lam = best[0] if best else 3.0
    mdl = PriorModel(target, use_strength, lam, use_age).fit(arr, slots)
    mdl.lam_cv = lam
    return mdl
