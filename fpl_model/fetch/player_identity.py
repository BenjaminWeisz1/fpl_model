"""Persistent project identities shared by Understat and supplementary sources.

Provider IDs are global aliases; legacy names are scoped by season and match team.
Accepted Understat per-row assignments seed the registry without re-matching history.
No FPL IDs are invented from names. CSV schemas and access workflow: SOFASCORE.md.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from fpl_model import paths
from fpl_model.league_rates import current_season  # noqa: E402

DATA = paths.CURRENT


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def stable_id(prefix, *parts):
    return prefix + hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()[:24]


class Registry:
    SCHEMAS = {
        'player_identity_registry': ['Canonical_Player_ID', 'Display_Name'],
        'player_identity_aliases': ['Provider', 'Provider_Player_ID', 'Season', 'Team', 'Name', 'Canonical_Player_ID', 'Evidence'],
        'fixture_identity_registry': ['Provider', 'Event_ID', 'Canonical_Fixture_ID', 'Season', 'Home', 'Away', 'Kickoff'],
        'identity_overrides': ['Provider', 'Provider_Player_ID', 'Canonical_Player_ID', 'Reason'],
    }

    def __init__(self, data=DATA):
        self.data = Path(data)
        self.tables = {k: read_csv(self.data / (k + '.csv')) for k in self.SCHEMAS}
        self.alias_rows = {tuple(r.values()) for r in self.tables['player_identity_aliases']}
        self.alias = {}
        self.context = {}
        for r in self.tables['player_identity_aliases']:
            self._index(r)
        self.players = {r['Canonical_Player_ID']: r for r in self.tables['player_identity_registry']}
        for r in self.tables['identity_overrides']:
            if r['Canonical_Player_ID'] not in self.players or not r['Reason']:
                raise ValueError('Invalid reviewed identity override')
            key = (r['Provider'], r['Provider_Player_ID'])
            if key in self.alias and self.alias[key] != r['Canonical_Player_ID']:
                existing = [a for a in self.tables['player_identity_aliases']
                            if (a['Provider'], a['Provider_Player_ID']) == key]
                if any(a['Evidence'] != 'unresolved_source_only' for a in existing):
                    raise ValueError('Override conflicts with accepted identity; review migration explicitly')
                for a in existing:
                    a['Canonical_Player_ID'] = r['Canonical_Player_ID']
                    a['Evidence'] = 'reviewed_override:' + r['Reason']
            self.alias[key] = r['Canonical_Player_ID']
        if any(c not in self.players for c in self.alias.values()):
            raise ValueError('Orphan player alias')
        self.fixtures = {}
        for r in self.tables['fixture_identity_registry']:
            key = (r['Provider'], r['Event_ID'])
            if key in self.fixtures and self.fixtures[key] != r:
                raise ValueError('Conflicting fixture alias')
            self.fixtures[key] = r
        self.alias_rows = {tuple(r.values()) for r in self.tables['player_identity_aliases']}
        # Source-only IDs stay retryable; accepted provider IDs claim their identity.
        self.source_only = {(r['Provider'], r['Provider_Player_ID']) for r in self.tables['player_identity_aliases']
                            if r['Evidence'] == 'unresolved_source_only'}
        self.holders = {(p, c): pid for (p, pid), c in self.alias.items() if (p, pid) not in self.source_only}
        self.promoted = {}

    def _index(self, r):
        target = self.context if r['Provider'] == 'legacy_pms' else self.alias
        key = ((r['Season'], r['Team'], r['Name']) if target is self.context
               else (r['Provider'], r['Provider_Player_ID']))
        old = target.get(key)
        if old and old != r['Canonical_Player_ID']:
            raise ValueError(f'Identity collision: {key}')
        target[key] = r['Canonical_Player_ID']

    def add_alias(self, provider, pid, cid, name='', season='', team='', evidence=''):
        r = dict(zip(self.SCHEMAS['player_identity_aliases'],
                     map(str, (provider, pid, season, team, name, cid, evidence))))
        self._index(r)
        if provider != 'legacy_pms':
            if evidence == 'unresolved_source_only':
                self.source_only.add((provider, str(pid)))
            else:
                self.holders[(provider, str(cid))] = str(pid)
        marker = tuple(r.values())
        if marker not in self.alias_rows:
            self.tables['player_identity_aliases'].append(r)
            self.alias_rows.add(marker)

    def _promote(self, provider, pid, cid, evidence):
        """Move a source-only provider ID, with all its alias rows, to an accepted identity."""
        key = (provider, str(pid))
        for a in self.tables['player_identity_aliases']:
            if (a['Provider'], a['Provider_Player_ID']) == key:
                a['Canonical_Player_ID'] = cid
                a['Evidence'] = 'promoted_from_source_only:' + evidence
        self.alias_rows = {tuple(r.values()) for r in self.tables['player_identity_aliases']}
        self.alias[key] = cid
        self.source_only.discard(key)
        self.holders[(provider, cid)] = str(pid)
        self.promoted[key] = cid

    def ensure(self, provider, pid, name):
        key = (provider, str(pid))
        cid = self.alias.get(key, stable_id('P', provider, str(pid)))
        if cid not in self.players:
            r = {'Canonical_Player_ID': cid, 'Display_Name': name}
            self.players[cid] = r
            self.tables['player_identity_registry'].append(r)
        self.add_alias(provider, pid, cid, name, evidence='provider_id')
        return cid

    def seed(self):
        # Preserve accepted provider identities and ALL contextual historical names.
        for r in read_csv(self.data / 'understat_fpl_crosswalk.csv'):
            self.ensure('understat', r['Player_ID_Understat'], r['Player_Understat'])
        for r in read_csv(self.data / 'player_match_positions.csv'):
            if r.get('Source', 'understat') != 'understat':   # FotMob rows (§197) carry no Understat identity
                continue
            cid = self.ensure('understat', r['Player_ID_Understat'], r['Player_Understat'])
            self.add_alias('legacy_pms', '', cid, r['Player'], r['Season'], r['Team'], 'accepted_understat_row')
        for r in read_csv(self.data / 'understat_match_positions.csv'):
            self.ensure('understat', r['Player_ID_Understat'], r['Player'])
            self.fixture('understat', r['MatchID_Understat'], r['Season'], r['Home'], r['Away'], r['Date'])
        # Current names are useful only when season AND team confirm the identity.
        for r in read_csv(self.data / 'players_current.csv'):
            cid = self.context.get((str(current_season()), r['Team'], r['Player']))
            if cid:
                self.add_alias('current_web_name', r['Team'] + ':' + r['Web_Name'], cid,
                               r['Web_Name'], str(current_season()), r['Team'], 'current_context')

    def fixture(self, provider, event, season, home, away, kickoff):
        key = (provider, str(event))
        if key in self.fixtures:
            r = self.fixtures[key]
            if (r['Season'], r['Home'], r['Away']) != (str(season), home, away):
                raise ValueError('Provider fixture ID changed membership')
            return r['Canonical_Fixture_ID']
        # Reconcile only one season/home/away candidate, never a gameweek or PMS MatchID.
        candidates = {r['Canonical_Fixture_ID'] for r in self.fixtures.values()
                      if (r['Season'], r['Home'], r['Away']) == (str(season), home, away)
                      and abs((datetime.fromisoformat(r['Kickoff']).replace(tzinfo=None) -
                               datetime.fromisoformat(str(kickoff)).replace(tzinfo=None)).total_seconds()) <= 86400}
        if len(candidates) > 1:
            raise ValueError('Ambiguous repeated fixture; explicit mapping required')
        cid = next(iter(candidates), stable_id('F', provider, str(event)))
        r = dict(zip(self.SCHEMAS['fixture_identity_registry'],
                     map(str, (provider, event, cid, season, home, away, kickoff))))
        self.fixtures[key] = r
        self.tables['fixture_identity_registry'].append(r)
        return cid

    MINUTES_TOL = 10  # 99.85% of accepted Sofascore/local played pairs agree within 10

    @staticmethod
    def name_score(a, b):
        """Crosswalk similarity, raised to .95 when one name's tokens contain the other's
        (Alisson / Alisson Becker, Rodri / Rodrigo 'Rodri' Hernandez Cascante)."""
        import unicodedata

        from fpl_model.fetch.build_position_crosswalk import name_similarity

        def tokens(name):
            s = unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode().lower()
            return set(''.join(ch if ch.isalpha() else ' ' for ch in s).split())
        score = name_similarity(a, b)
        ta, tb = tokens(a), tokens(b)
        return max(score, .95) if ta and tb and (ta <= tb or tb <= ta) else score

    def roster(self, provider, entries, season, team, local_names, minutes=None, local_minutes=None):
        """Reuse accepted IDs, then conservative one-to-one assignment inside fixture/team only.

        Source-only IDs are retried in every fixture rather than frozen at their first
        failure (an unused substitute has no local row to match). When `minutes`
        (provider ID -> minutes or None) is given, only players who actually played can
        claim a local row, and with `local_minutes` (local name -> minutes) the minutes
        must agree within MINUTES_TOL. A pair is accepted only if the best assignment
        beats the best assignment without that pair by at least .10, and never takes
        an identity already held by another ID from the same provider.
        """
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        result, candidates, used = {}, [], set()
        for pid, name in entries:
            key = (provider, str(pid))
            cid = self.alias.get(key)
            played = minutes is None or (minutes.get(str(pid)) or 0) > 0
            if cid and key not in self.source_only:
                result[str(pid)] = (cid, 'provider_id')
                used.add(cid)
            elif played:
                candidates.append((str(pid), name))
            else:
                result[str(pid)] = (cid or self._source_only(provider, pid, name, season, team),
                                    'unresolved_cross_provider')
        local = [(n, self.context.get((str(season), team, n))) for n in sorted(set(local_names))]
        local = [(n, c) for n, c in local if c and c not in used and (provider, c) not in self.holders]
        if candidates and local:
            scores = np.array([[self.name_score(name, n) for n, _ in local] for _, name in candidates])
            if minutes is not None and local_minutes is not None:
                for i, (pid, _) in enumerate(candidates):
                    for j, (n, _) in enumerate(local):
                        a, b = minutes.get(pid), local_minutes.get(n)
                        if a is not None and b is not None and abs(float(a) - float(b)) > self.MINUTES_TOL:
                            scores[i, j] = 0
            ii, jj = linear_sum_assignment(scores, maximize=True)
            best = scores[ii, jj].sum()
            for i, j in zip(ii, jj):
                score = scores[i, j]
                if score < .90:
                    continue
                without = scores.copy()
                without[i, j] = -1e6
                ai, aj = linear_sum_assignment(without, maximize=True)
                alternative = np.clip(without[ai, aj], 0, None).sum()
                if best - alternative >= .10:
                    pid, name = candidates[i]
                    cid = local[j][1]
                    evidence = f'fixture_roster:{score:.4f}'
                    if (provider, pid) in self.source_only:
                        self._promote(provider, pid, cid, evidence)
                    self.add_alias(provider, pid, cid, name, season, team, evidence)
                    result[pid] = (cid, 'roster_accepted')
        # A valid independent project ID does not claim an unresolved cross-provider join.
        for pid, name in candidates:
            if pid not in result:
                result[pid] = (self._source_only(provider, pid, name, season, team), 'unresolved_cross_provider')
        if len({c for c, _ in result.values()}) != len(result):
            raise ValueError('Roster canonical identity collision')
        return result

    def _source_only(self, provider, pid, name, season, team):
        existing = self.alias.get((provider, str(pid)))
        if existing:
            return existing
        cid = stable_id('P', provider, str(pid))
        if cid not in self.players:
            r = {'Canonical_Player_ID': cid, 'Display_Name': name}
            self.players[cid] = r
            self.tables['player_identity_registry'].append(r)
        self.add_alias(provider, pid, cid, name, season, team, 'unresolved_source_only')
        return cid

    def save(self):
        for k, columns in self.SCHEMAS.items():
            write_csv(self.data / (k + '.csv'), self.tables[k], columns)

    def understat_assignments(self, key, provider_ids, names, local_names):
        """Accepted contextual IDs first; confidence-gated matching for new rows only."""
        from fpl_model.fetch.build_position_crosswalk import name_similarity
        season, team, _, _ = key
        local_names = sorted(set(local_names))
        assigned, used, unresolved = [], set(), []
        for i, (pid, name) in enumerate(zip(provider_ids, names)):
            cid = self.alias.get(('understat', str(pid)))
            candidates = [n for n in local_names if self.context.get((str(season), team, n)) == cid] if cid else []
            if len(candidates) == 1 and candidates[0] not in used:
                pn = candidates[0]
                assigned.append((i, pn, name_similarity(name, pn)))
                used.add(pn)
            else:
                unresolved.append(i)
        # Existing accepted history is reused; new names require a clear margin.
        for i in unresolved:
            scores = sorted([(name_similarity(names[i], n), n) for n in local_names if n not in used], reverse=True)
            if not scores:
                continue
            score, name = scores[0]
            runner_up = scores[1][0] if len(scores) > 1 else 0
            competing = max((name_similarity(names[j], name) for j in unresolved if j != i), default=0)
            if score >= .90 and score - max(runner_up, competing) >= .10:
                cid = self.ensure('understat', provider_ids[i], names[i])
                old = self.context.get((str(season), team, name))
                if old and old != cid:
                    continue
                self.add_alias('legacy_pms', '', cid, name, season, team, 'new_roster_confident')
                assigned.append((i, name, score))
                used.add(name)
        return sorted(assigned)


if __name__ == '__main__':
    registry = Registry()
    registry.seed()
    registry.save()
    print(f'Seeded {len(registry.players)} players, {len(registry.fixtures)} fixture aliases')
