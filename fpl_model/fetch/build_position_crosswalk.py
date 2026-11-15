"""
Understat -> player_match_stats position crosswalk (AGENDA §4 shared prerequisites).

Turns the raw per-match Understat roster slots in ``understat_match_positions.csv``
into a per-row artifact keyed on ``player_match_stats.csv``'s own identity, so the
defcon and goal models can give current-season (2025+) rows the detailed formation
slot they lack. The FPL API only supplies GK/DEF/MID/FWD and ``player_match_stats.csv``
stores NULL Position for every FPL-era row, so without this the defcon model's
``Pos_*`` position dummies are all-zero for exactly the rows it validates and serves
on. This is the "Understat<->FPL player-id crosswalk" of CLAUDE.md §2.5.

Three shared prerequisites are done here:

  * §2.5 crosswalk -- a within-match name matcher reconciles the three naming
    conventions (Understat full name, FPL full name in player_match_stats, fbref
    full name). MatchID cannot be used for the FPL era: player_match_stats keys
    FPL rows by ``offset + round`` (one id per gameweek) while Understat's MatchID
    is the true per-match id, so the join key is ``(Season, Team, Opponent, Venue)``
    plus a one-to-one optimal assignment of player names inside each match
    (``scipy.linear_sum_assignment``). The match is validated on the 2017-2024
    overlap, where player_match_stats already carries fbref's own ``Pos``.
  * §4.5 agreement check -- on the fbref overlap the mapped Understat slot is
    compared with fbref's ``Pos`` (exact-slot and coarse DEF/MID/FWD/GK group) and
    the rates are printed; a low rate would mean the matcher or the mapping cannot
    be trusted for 2025+.
  * §4.6 Sub resolution -- Understat labels every bench appearance ``Sub``. Each
    substitute is given the modal starting slot they occupy when they DO start
    (same-season first, then career), so a sub's row carries a usable position
    instead of ``Sub``.

Outputs (both in ``data/current/``):

  * ``player_match_positions.csv`` -- one row per matched player_match_stats row:
    its natural key (Season, Team, Opponent, Venue, Player), the Understat id/name,
    the raw slot, the FBref-style mapped slot, the Sub-resolved slot, and match
    diagnostics. This is the per-row artifact the models merge on.
  * ``understat_fpl_crosswalk.csv`` -- a player-level alias table (Understat id ->
    modal fbref-era name, modal FPL-era name, Web_Name when resolvable) for reuse
    by other work.

Understat stopped answering during GW9 of 2026 (CHANGELOG §197). Every match its frozen cache holds keeps its
Understat rows (2017 to 2026 GW8). A match it lacks takes its rows from fotmob_match_positions.csv instead (from
fetch_fotmob_matches.FIRST_SEASON): FotMob's slots rendered in Understat's vocabulary and keyed to project names
through Opta IDs, so no name matching. Understat is kept where it exists: defcon scored slightly worse with
FotMob's slots in its place (§197 test 1), while FotMob's beat no detailed positions clearly (§197 test 2). A FotMob
substitute gets his modal starting slot in that season's FotMob rows, then his career modal slot over every
earlier starter row (Understat included). `Source` says which provider a row came from; Understat id columns are
empty on FotMob rows.

Pure CSV transform (no network); deterministic. Run locally:
    python build_position_crosswalk.py
"""
import difflib
import re
import sys
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from fpl_model import paths
from fpl_model.atomic_write import to_csv_atomic  # noqa: E402
from fpl_model.fetch.fetch_fotmob_matches import POSITIONS_OUT as FOTMOB_FILE
from fpl_model.fetch.player_identity import Registry

_ROOT = paths.ROOT
_DATA = paths.CURRENT
PMS_FILE = paths.CURRENT / "player_match_stats.csv"
UP_FILE = paths.CURRENT / "understat_match_positions.csv"
PLAYERS_CURRENT = paths.CURRENT / "players_current.csv"
OUT_ROWS = paths.CURRENT / "player_match_positions.csv"
OUT_XWALK = paths.CURRENT / "understat_fpl_crosswalk.csv"

# Accept a within-match name pair only at or above this similarity. Optimal
# assignment runs first, then sub-threshold pairs are dropped, so a leftover
# player with no plausible counterpart stays unmatched rather than forced.
MATCH_THRESHOLD = 0.55

# Coarse position groups, for the §4.5 agreement check and for the resolved-slot
# fallback. Mirrors the defcon model's DEFENDER/MIDFIELDER/FORWARD position sets.
DEF_SLOTS = {"CB", "FB", "WB", "LB", "RB"}
MID_SLOTS = {"DM", "CM", "AM", "WM", "W", "LM", "RM", "LW", "RW"}
FWD_SLOTS = {"FW"}

_PUNC = re.compile(r"[^a-z0-9 ]+")


# ── name normalisation + similarity ────────────────────────────────────────────

def norm(s):
    """Accent-stripped, lowercase, punctuation-collapsed, whitespace-collapsed."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNC.sub(" ", s.lower())
    return " ".join(s.split())


def tokens(s):
    return norm(s).split(" ")


def surname(s):
    t = tokens(s)
    return t[-1] if t else ""


def name_similarity(a, b):
    """Similarity in [0, 1] between two player names.

    Weighted toward the signals that actually distinguish players inside one
    match's ~14-name pool: a shared surname, a shared distinctive single token
    (mononyms like 'Evanilson', or a full name that is a subset of the other),
    and a whole-string edit ratio for accent/spelling drift. Optimal assignment
    over the whole match resolves the ambiguous cases (e.g. two players sharing a
    first name), so this only needs to rank candidates sensibly.
    """
    na, nb = norm(a), norm(b)
    if na == nb and na:
        return 1.0
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    seq = difflib.SequenceMatcher(None, na, nb).ratio()
    jac = len(ta & tb) / len(ta | tb)
    contain = 1.0 if (ta <= tb or tb <= ta) else 0.0
    sur = 1.0 if (surname(a) and surname(a) == surname(b)) else 0.0
    return max(seq, jac, 0.5 * contain + 0.5 * sur,
               0.9 * sur + 0.1 * jac, 0.6 * contain + 0.4 * jac)


def assign_match(up_names, pms_names):
    """Optimal one-to-one assignment of Understat names to PMS names.

    Returns a list of (up_index, pms_name, score) for accepted pairs.
    """
    n, m = len(up_names), len(pms_names)
    if n == 0 or m == 0:
        return []
    sim = np.zeros((n, m))
    for i, un in enumerate(up_names):
        for j, pn in enumerate(pms_names):
            sim[i, j] = name_similarity(un, pn)
    rows, cols = linear_sum_assignment(sim, maximize=True)
    out = []
    for i, j in zip(rows, cols):
        s = sim[i, j]
        if s >= MATCH_THRESHOLD:
            out.append((i, pms_names[j], float(s)))
    return out


def coarse_group(labels):
    g = set()
    for lab in labels:
        if lab in DEF_SLOTS:
            g.add("DEF")
        elif lab in MID_SLOTS:
            g.add("MID")
        elif lab in FWD_SLOTS:
            g.add("FWD")
        elif lab == "GK":
            g.add("GK")
    return g


# ── Sub resolution (§4.6) ──────────────────────────────────────────────────────

def build_sub_resolver(up):
    """Per-Understat-id modal STARTING slot, same-season first then career.

    Returns a function (player_id, season) -> resolved mapped slot or None.
    """
    starters = up[up["Is_Starter"] & (up["Position"] != "Sub") & up["Position"].notna()]
    by_season = defaultdict(Counter)   # (pid, season) -> Counter(mapped slot)
    career = defaultdict(Counter)      # pid -> Counter(mapped slot)
    for pid, season, pos in zip(starters["Player_ID_Understat"],
                                starters["Season"], starters["Position"]):
        by_season[(pid, season)][pos] += 1
        career[pid][pos] += 1

    def resolve(pid, season):
        c = by_season.get((pid, season))
        if c:
            return c.most_common(1)[0][0]
        c = career.get(pid)
        if c:
            return c.most_common(1)[0][0]
        return None

    return resolve


def fotmob_rows(fm, prior, pms_keys):
    """FotMob positions (already labelled in Understat's vocabulary) -> crosswalk rows.

    A substitute takes his modal starting slot that season over both sources' rows, then his career modal slot.

    fm: fotmob_match_positions.csv frame; prior: crosswalk rows from earlier seasons (for a sub's career slot);
    pms_keys: (Season, Team, Opponent, Venue, Player) keys present in player_match_stats.csv.
    """
    fm = fm[fm["Player"].notna()]
    fm = fm[[k in pms_keys for k in zip(fm["Season"], fm["Team"], fm["Opponent"], fm["Venue"], fm["Player"])]]
    st = fm[fm["Is_Starter"].astype(bool)]
    pst = prior[prior["Is_Starter"].astype(bool) & (prior["Position_Mapped"] != "Sub") & prior["Position_Mapped"].notna()]
    by_season = defaultdict(Counter)    # his starts that season, from either source
    for player, season, pos in list(zip(pst["Player"], pst["Season"], pst["Position_Mapped"])) +             list(zip(st["Player"], st["Season"], st["Position"])):
        by_season[(player, season)][pos] += 1
    career = defaultdict(list)          # player -> [(season, slot)] over starter rows
    for player, season, pos in zip(pst["Player"], pst["Season"], pst["Position_Mapped"]):
        career[player].append((season, pos))
    for player, season, pos in zip(st["Player"], st["Season"], st["Position"]):
        career[player].append((season, pos))
    rows = []
    for r in fm.itertuples():
        starter = bool(r.Is_Starter)
        if starter:
            resolved, method = r.Position, "fotmob_starter"
        elif by_season.get((r.Player, r.Season)):
            resolved, method = by_season[(r.Player, r.Season)].most_common(1)[0][0], "fotmob_sub_modal_slot"
        else:
            c = Counter(p for s, p in career.get(r.Player, []) if s <= r.Season)
            resolved, method = (c.most_common(1)[0][0], "fotmob_sub_career_slot") if c else (None, "fotmob_sub_unresolved")
        rows.append({"Season": r.Season, "Team": r.Team, "Opponent": r.Opponent, "Venue": r.Venue,
                     "Player": r.Player, "Player_Understat": None, "Player_ID_Understat": None,
                     "Position_Understat": r.Position_Understat, "Position_Mapped": r.Position,
                     "Position_Resolved": resolved, "Is_Starter": starter, "Back_Three": bool(r.Back_Three),
                     "Match_Method": method, "Match_Score": 1.0, "Source": "fotmob",
                     "Player_ID_FotMob": r.Player_ID_FotMob})
    return pd.DataFrame(rows)


# ── main build ─────────────────────────────────────────────────────────────────

def main():
    if not UP_FILE.exists():
        sys.exit(f"missing {UP_FILE.name} -- run fetch_understat.py first.")
    if not PMS_FILE.exists():
        sys.exit(f"missing {PMS_FILE.name}.")

    pms = pd.read_csv(PMS_FILE, encoding="utf-8-sig", low_memory=False)
    up = pd.read_csv(UP_FILE, encoding="utf-8-sig")
    if not FOTMOB_FILE.exists():
        sys.exit(f"missing {FOTMOB_FILE.name} -- run fetch_fotmob_matches.py first.")
    registry = Registry(_DATA)
    registry.seed()

    # PMS names indexed by the natural match key. (Season, Team, Opponent, Venue)
    # is unique per match; a player appears once per match, so the 5-tuple is a
    # unique row identity we can merge back on.
    pms_names_by_key = defaultdict(list)
    for season, team, opp, venue, player in zip(
            pms["Season"], pms["Team"], pms["Opponent"], pms["Venue"], pms["Player"]):
        pms_names_by_key[(season, team, opp, venue)].append(player)

    resolve_sub = build_sub_resolver(up)

    up_keys = list(zip(up["Season"], up["Team"], up["Opponent"], up["Venue"]))
    up_by_key = defaultdict(list)      # key -> list of up positional indices
    for i, k in enumerate(up_keys):
        up_by_key[k].append(i)

    rows = []
    matched_up = 0
    total_up_with_pms = 0
    era_stat = {"fbref": [0, 0], "FPL": [0, 0]}
    for key, up_idxs in up_by_key.items():
        pms_names = pms_names_by_key.get(key)
        if not pms_names:
            continue
        total_up_with_pms += len(up_idxs)
        era = "fbref" if key[0] <= 2024 else "FPL"
        era_stat[era][1] += len(up_idxs)
        up_names = [up.at[i, "Player"] for i in up_idxs]
        provider_ids = [up.at[i, "Player_ID_Understat"] for i in up_idxs]
        for local_i, pms_name, sc in registry.understat_assignments(key, provider_ids, up_names, pms_names):
            gi = up_idxs[local_i]
            matched_up += 1
            era_stat[era][0] += 1
            raw = up.at[gi, "Position_Understat"]
            mapped = up.at[gi, "Position"]
            pid = up.at[gi, "Player_ID_Understat"]
            season = up.at[gi, "Season"]
            is_starter = bool(up.at[gi, "Is_Starter"])
            resolved = mapped if is_starter or mapped != "Sub" else resolve_sub(pid, season)
            rows.append({
                "Season": season,
                "Team": key[1],
                "Opponent": key[2],
                "Venue": key[3],
                "Player": pms_name,
                "Player_Understat": up.at[gi, "Player"],
                "Player_ID_Understat": pid,
                "Position_Understat": raw,
                "Position_Mapped": mapped,
                "Position_Resolved": resolved,
                "Is_Starter": is_starter,
                "Back_Three": bool(up.at[gi, "Back_Three"]),
                "Match_Method": "exact" if sc >= 0.999 else "fuzzy",
                "Match_Score": round(sc, 4),
            })

    out = pd.DataFrame(rows)
    out["Source"] = "understat"
    out["Player_ID_FotMob"] = None
    pms_keys = set(zip(pms["Season"], pms["Team"], pms["Opponent"], pms["Venue"], pms["Player"]))
    fm = pd.read_csv(FOTMOB_FILE, encoding="utf-8-sig", low_memory=False)
    have = set(zip(out["Season"], out["Team"], out["Opponent"], out["Venue"]))     # team-matches Understat covers
    fm = fm[[k not in have for k in zip(fm["Season"], fm["Team"], fm["Opponent"], fm["Venue"])]]
    fm_out = fotmob_rows(fm, out, pms_keys)
    print(f"  FotMob rows for {fm[['Season', 'Team', 'Opponent', 'Venue']].drop_duplicates().shape[0] // 2} "
          f"matches Understat lacks: {len(fm_out):,}; substitutes by method: "
          f"{fm_out[~fm_out['Is_Starter']]['Match_Method'].value_counts().to_dict()}")
    out = pd.concat([out, fm_out], ignore_index=True)
    # The merge key must be unique or the models would fan out rows.
    dup = out.duplicated(subset=["Season", "Team", "Opponent", "Venue", "Player"]).sum()
    if dup:
        sys.exit(f"internal error: {dup} duplicate merge keys in player_match_positions")

    out = out.sort_values(["Season", "Team", "Opponent", "Venue", "Player"]).reset_index(drop=True)
    for c in ("Player_ID_Understat", "Player_ID_FotMob"):     # integers, blank where the other source supplied the row
        out[c] = pd.to_numeric(out[c]).astype("Int64")
    registry.save()
    to_csv_atomic(out, OUT_ROWS, index=False, encoding="utf-8-sig")

    print("=== within-match name-match coverage (Understat rows -> PMS rows) ===")
    for era, (m, t) in era_stat.items():
        print(f"  {era}: {m}/{t} = {m / max(t, 1):.2%}")
    print(f"  wrote {len(out):,} rows -> {OUT_ROWS.name}")

    # PMS-side coverage: fraction of PMS rows that received a detailed position.
    pms_keys = set(zip(pms["Season"], pms["Team"], pms["Opponent"], pms["Venue"], pms["Player"]))
    got = set(zip(out["Season"], out["Team"], out["Opponent"], out["Venue"], out["Player"]))
    for era, lo, hi in [("fbref", 0, 2024), ("FPL", 2025, 9999)]:
        tot = sum(1 for k in pms_keys if lo <= k[0] <= hi)
        hit = sum(1 for k in got if lo <= k[0] <= hi)
        print(f"  PMS-side {era}: {hit}/{tot} = {hit / max(tot, 1):.2%} of rows get a detailed slot")

    sub_rows = out[out["Match_Method"] != "exact"]
    print(f"  fuzzy (non-exact) name matches: {len(sub_rows):,} "
          f"({len(sub_rows) / max(len(out), 1):.1%})")
    unresolved = int((out["Match_Method"] == "sub_unresolved").sum())
    n_sub = int((~out["Is_Starter"]).sum())
    print(f"  Sub rows: {n_sub:,}; resolved to a modal slot: {n_sub - unresolved:,}; "
          f"unresolved: {unresolved:,}")

    _agreement_report(out, pms)
    _build_player_crosswalk(out, up)


def _agreement_report(out, pms):
    """§4.5: mapped Understat slot vs fbref's own Pos on the 2017-2024 overlap."""
    fb = pms[pms["Season"] <= 2024]
    fb_pos = {}
    for season, team, opp, venue, player, pos in zip(
            fb["Season"], fb["Team"], fb["Opponent"], fb["Venue"], fb["Player"], fb["Position"]):
        fb_pos[(season, team, opp, venue, player)] = pos

    n = exact = coarse_ok = 0
    for season, team, opp, venue, player, mapped in zip(
            out["Season"], out["Team"], out["Opponent"], out["Venue"],
            out["Player"], out["Position_Mapped"]):
        if season > 2024 or mapped == "Sub" or pd.isna(mapped):
            continue
        fpos = fb_pos.get((season, team, opp, venue, player))
        if pd.isna(fpos):
            continue
        fb_tokens = set(str(fpos).split(","))
        n += 1
        if mapped in fb_tokens:
            exact += 1
        if coarse_group([mapped]) & coarse_group(fb_tokens):
            coarse_ok += 1

    print("\n=== §4.5 FBref<->Understat position agreement (2017-2024 overlap) ===")
    print(f"  comparable starter rows: {n:,}")
    print(f"  exact slot in fbref Pos token set: {exact / max(n, 1):.1%}")
    print(f"  coarse group (DEF/MID/FWD/GK) overlap: {coarse_ok / max(n, 1):.1%}")
    print("  (coarse agreement is the trust gate for using Understat alone in 2025+.)")


def _build_player_crosswalk(out, up):
    """Player-level alias table: Understat id -> modal PMS names per era + Web_Name."""
    pc = None
    if PLAYERS_CURRENT.exists():
        pc = pd.read_csv(PLAYERS_CURRENT, encoding="utf-8-sig")

    fb_name = defaultdict(Counter)
    fpl_name = defaultdict(Counter)
    up_name = {}
    team = defaultdict(Counter)
    for pid, season, pms_name, upn, tm in zip(
            out["Player_ID_Understat"], out["Season"], out["Player"],
            out["Player_Understat"], out["Team"]):
        if pd.isna(pid):
            continue
        up_name[pid] = upn
        team[pid][tm] += 1
        if season <= 2024:
            fb_name[pid][pms_name] += 1
        else:
            fpl_name[pid][pms_name] += 1

    # Web_Name via players_current: match on the modal FPL-era PMS name.
    web_by_player = {}
    if pc is not None:
        web_by_player = dict(zip(pc["Player"], pc["Web_Name"]))

    recs = []
    for pid in sorted(up_name):
        fpl_modal = fpl_name[pid].most_common(1)[0][0] if fpl_name[pid] else None
        fb_modal = fb_name[pid].most_common(1)[0][0] if fb_name[pid] else None
        recs.append({
            "Player_ID_Understat": pid,
            "Player_Understat": up_name[pid],
            "PMS_Player_fbref": fb_modal,
            "PMS_Player_FPL": fpl_modal,
            "Web_Name": web_by_player.get(fpl_modal) if fpl_modal else None,
            "Team_Modal": team[pid].most_common(1)[0][0] if team[pid] else None,
            "n_fbref_rows": int(sum(fb_name[pid].values())),
            "n_fpl_rows": int(sum(fpl_name[pid].values())),
        })
    xw = pd.DataFrame(recs).sort_values("Player_Understat").reset_index(drop=True)
    xw.to_csv(OUT_XWALK, index=False, encoding="utf-8-sig")
    print("\n=== player-level crosswalk ===")
    print(f"  {len(xw):,} Understat ids -> {OUT_XWALK.name}")
    print(f"  with a Web_Name resolved via players_current: "
          f"{int(xw['Web_Name'].notna().sum()):,}")


if __name__ == "__main__":
    main()
