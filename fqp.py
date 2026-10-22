"""
fqp.py -- federated query processing over per-client knowledge graphs
=====================================================================

Each client (hospital) holds its OWN knowledge graph: the UMLS graph around
the cohort is split by relation type, each relation held by `repl` clients
(default 2), so the clients' KGs differ and overlap only partly.

A patient's query (the CUIs of their codes) is answered by the client that
holds the patient, acting as the federated query engine:

  1. source selection : each source publishes only a Bloom-filter summary of
                        the entities it has triples about (no data, no
                        patients). A sub-query is sent to a source only if
                        its summary may contain the entity.
  2. decomposition    : one sub-query per selected source, holding only the
                        entities that source can answer (+ optional decoy
                        entities that hide which codes are asked about).
  3. local execution  : every source answers from its own KG only.
  4. merge            : union of the returned triples -> candidate concepts
                        with the query codes they are linked to.
  5. bound join (2nd hop, optional): the best first-hop candidates are sent
                        as a new bound sub-query (FedX-style bound join), so
                        paths code -> x (source A) -> y (source B) are found
                        across sources.

Batched execution: one request per source carries the sub-queries of many
patients (a standard FQP optimisation); per-query costs are still counted.

Everything measured: sources contacted, sub-queries, wasted contacts (Bloom
false positives / empty answers), decoys, request and response bytes,
summary bytes, wall time, and completeness against the central KG.
"""

from __future__ import annotations

import base64
import json
import math
import time
from collections import defaultdict

import numpy as np

_P = 2147483647
_HASH = np.random.default_rng(7).integers(1, 1 << 20, size=(16, 2))

class Bloom:
    def __init__(self, n=1, fp=0.01, m=None, k=None, bits=None):
        self.m = m or max(64, int(math.ceil(-max(n, 1) * math.log(fp) / (math.log(2) ** 2))))
        self.k = k or max(1, min(16, int(round(self.m / max(n, 1) * math.log(2)))))
        self.bits = bits if bits is not None else np.zeros(self.m, dtype=bool)

    def _idx(self, ids):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        return [((ids * _HASH[i, 0] + _HASH[i, 1]) % _P) % self.m for i in range(self.k)]

    def add(self, ids):
        for ix in self._idx(ids):
            self.bits[ix] = True

    def contains(self, ids):
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            return np.zeros(0, dtype=bool)
        out = np.ones(len(ids), dtype=bool)
        for ix in self._idx(ids):
            out &= self.bits[ix]
        return out

    def to_json(self):
        return {"m": self.m, "k": self.k, "bits": base64.b64encode(np.packbits(self.bits).tobytes()).decode()}

    @classmethod
    def from_json(cls, d):
        bits = np.unpackbits(np.frombuffer(base64.b64decode(d["bits"]), dtype=np.uint8))[:d["m"]].astype(bool)
        return cls(m=d["m"], k=d["k"], bits=bits)

    @property
    def nbytes(self):
        return (self.m + 7) // 8

def assign_relations(n_rel, n_sources, repl, seed):
    """relation id -> sources holding it (each relation at min(repl, n) sources)."""
    rng = np.random.default_rng(seed + 991)
    r = max(1, min(repl, n_sources))
    return {rel: sorted(rng.choice(n_sources, size=r, replace=False).tolist()) for rel in range(n_rel)}

class Source:
    """One client's local KG partition with an entity index (both directions)."""

    def __init__(self, triples):
        t = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
        h = np.concatenate([t[:, 0], t[:, 2]])
        o = np.concatenate([t[:, 2], t[:, 0]])
        r = np.concatenate([t[:, 1], t[:, 1]])
        order = np.argsort(h, kind="stable")
        self.h, self.o, self.r = h[order], o[order], r[order]
        self.ents = np.unique(self.h)
        self.n_triples = len(t)
        self.bloom = Bloom(len(self.ents))
        self.bloom.add(self.ents)

    def answer_one(self, ents):
        """entity id -> [[rel, other], ...] (only entities this source has)."""
        out = {}
        for e in ents:
            a, b = np.searchsorted(self.h, e), np.searchsorted(self.h, e, side="right")
            if b > a:
                out[int(e)] = np.stack([self.r[a:b], self.o[a:b]], 1).tolist()
        return out

    def answer_batch(self, queries):
        return [self.answer_one(q) for q in queries]

def build_sources(kgc, n_sources, repl, seed, only=None):
    """Split the KG by relation across n_sources. `only` = build just that
    source (a Docker client builds only its own KG)."""
    trip = kgc["triples"]
    owners = assign_relations(len(kgc["rels"]), n_sources, repl, seed)
    keep = range(n_sources) if only is None else [only]
    out = {}
    for s in keep:
        rels = np.array([rel for rel, ow in owners.items() if s in ow], dtype=np.int64)
        out[s] = Source(trip[np.isin(trip[:, 1], rels)])
    return out, owners

class FQPEngine:
    """Federated query engine running at the client that holds the patient.

    summaries : {source: Bloom}   (published summaries, no data)
    transport : f(source, list_of_entity_lists) -> (list_of_answers, req_bytes, resp_bytes)
    """

    def __init__(self, kg, summaries, transport, home, decoys=0, hops=2, join_top=20, seed=1,
                 sources_allowed=None):
        self.kg = kg
        self.ents = kg.kgc["ents"]
        self.e2i = {e: i for i, e in enumerate(self.ents)}
        self.summ = summaries if sources_allowed is None else {s: b for s, b in summaries.items() if s in sources_allowed}
        self.transport = transport
        self.home = home
        self.decoys = decoys
        self.hops = hops
        self.join_top = join_top
        self.rng = np.random.default_rng(seed)

    def _plan(self, queries):
        """source selection + decomposition: {source: [(query_idx, [entity ids])]}"""
        plan = defaultdict(list)
        for j, q in enumerate(queries):
            if not q:
                continue
            qa = np.array(q, dtype=np.int64)
            for s, bl in self.summ.items():
                sel = qa[bl.contains(qa)].tolist()
                if sel:
                    if self.decoys and s != self.home:
                        dec = self.rng.integers(0, len(self.ents), size=self.decoys * len(sel)).tolist()
                        sel = sel + dec
                    plan[s].append((j, sel))
        return plan

    def _execute(self, plan, st, nq):
        """run sub-queries (one batched request per source); returns per-query answers"""
        res = [defaultdict(list) for _ in range(nq)]
        for s, items in plan.items():
            t0 = time.time()
            answers, req_b, resp_b = self.transport(s, [q for _, q in items])
            st["secs"] += time.time() - t0
            remote = s != self.home
            st["req_bytes"] += req_b if remote else 0
            st["resp_bytes"] += resp_b if remote else 0
            st["requests"] += 1 if remote else 0
            for (j, q), ans in zip(items, answers):
                st["subqueries"] += 1
                st["remote_subqueries"] += int(remote)
                st["per_query_sources"][j].add(s)
                if not ans:
                    st["wasted"] += 1
                for e, pairs in ans.items():
                    res[j][int(e)] += pairs
        return res

    def coverage_batch(self, code_lists):
        """For each query (list of codes): cov {candidate CUI -> set(codes)} from
        the federation (1 hop) and two {candidate CUI -> set(codes)} from the
        bound join (2 hops)."""
        st = {"secs": 0.0, "req_bytes": 0, "resp_bytes": 0, "requests": 0, "subqueries": 0,
              "remote_subqueries": 0, "wasted": 0, "per_query_sources": defaultdict(set),
              "join_subqueries": 0}
        c2c = self.kg.code2cui
        queries, ent_codes = [], []
        for codes in code_lists:
            m = defaultdict(set)
            for c in codes:
                e = self.e2i.get(c2c.get(c))
                if e is not None:
                    m[e].add(c)
            ent_codes.append(m)
            queries.append(sorted(m))
        res = self._execute(self._plan(queries), st, len(queries))
        covs = []
        for j, m in enumerate(ent_codes):
            own = set(m)
            cov = defaultdict(set)
            for e, pairs in res[j].items():
                if e not in m:
                    continue
                for _, o in pairs:
                    if o not in own:
                        cov[o].update(m[e])
            covs.append(cov)
        twos = [defaultdict(set) for _ in queries]
        if self.hops >= 2:
            bound = [sorted(cov, key=lambda x: -len(cov[x]))[:self.join_top] for cov in covs]
            sub0 = st["subqueries"]
            res2 = self._execute(self._plan(bound), st, len(bound))
            st["join_subqueries"] = st["subqueries"] - sub0
            for j, cov in enumerate(covs):
                bset = set(bound[j])
                for x, pairs in res2[j].items():
                    if x not in bset:
                        continue
                    for _, y in pairs:
                        if y in cov:
                            twos[j][y].update(cov[x])
        out_cov = [{self.ents[k]: v for k, v in cov.items()} for cov in covs]
        out_two = [{self.ents[k]: v for k, v in tw.items()} for tw in twos]
        nq = max(1, len(queries))
        stats = {"queries": len(queries),
                 "sources_available": len(self.summ),
                 "sources_contacted_per_query": float(np.mean([len(st["per_query_sources"][j]) for j in range(len(queries))])) if queries else 0.0,
                 "remote_sources_per_query": float(np.mean([len(st["per_query_sources"][j] - {self.home}) for j in range(len(queries))])) if queries else 0.0,
                 "subqueries_per_query": st["subqueries"] / nq,
                 "remote_subqueries_per_query": st["remote_subqueries"] / nq,
                 "join_subqueries_per_query": st["join_subqueries"] / nq,
                 "wasted_contacts_per_query": st["wasted"] / nq,
                 "requests": st["requests"],
                 "req_bytes_per_query": st["req_bytes"] / nq,
                 "resp_bytes_per_query": st["resp_bytes"] / nq,
                 "total_mb": (st["req_bytes"] + st["resp_bytes"]) / 2**20,
                 "secs": round(st["secs"], 3),
                 "ms_per_query": 1000 * st["secs"] / nq,
                 "decoys_per_subquery": self.decoys,
                 "summary_bytes": int(sum(b.nbytes for b in self.summ.values()))}
        return out_cov, out_two, stats

def local_transport(sources):
    """in-process transport for the simulation; bytes measured as JSON."""
    def call(s, queries):
        ans = sources[s].answer_batch(queries)
        return ans, len(json.dumps(queries)), len(json.dumps(ans))
    return call

def rank_from_cov(cov, two, df, n_docs, budget, join_w=0.5, deg=None):
    def score(n):
        extra = len(two.get(n, set()) - cov[n]) if two else 0
        return (len(cov[n]) + join_w * math.log1p(extra)) * (math.log((n_docs + 1) / (df.get(n, 0) + 1)) + 1.0)
    if deg is None:
        return sorted(cov, key=lambda n: (-score(n), n))[:budget]
    return sorted(cov, key=lambda n: (-score(n), -deg.get(n, 0), n))[:budget]

def completeness(kg, codes, cov):
    """share of the central-KG 1-hop candidates that the federation found"""
    full = set()
    own = {kg.code2cui.get(c) for c in codes}
    for c in codes:
        for n in kg.neigh_of_code(c):
            if n not in own:
                full.add(n)
    if not full:
        return None
    return len(full & set(cov)) / len(full)
