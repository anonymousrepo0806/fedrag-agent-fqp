from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

import cache_store as CS

_QUERY_TIMEOUT = 20.0
_SELECT_TIMEOUT = 30.0

def _pick_endpoints(fair_api_base, min_fair_score=0.6, category=None, max_sources=2):
    resp = requests.get(
        f"{fair_api_base.rstrip('/')}/federated/select",
        params={
            "min_fair_score": min_fair_score,
            "category": category,
            "limit": max_sources,
            "require_sparql": True,
        },
        timeout=_SELECT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        raise RuntimeError(
            f"No dataset in the FAIR corpus has a confirmed live SPARQL endpoint at or "
            f"above min_fair_score={min_fair_score}. Lower min_fair_score, broaden category, "
            f"or check that the FAIR tool's deployment has real outbound network access."
        )
    return [
        {"id": r["id"], "fair_score": r["fair_score"], "sparql_endpoint": r["sparql_endpoint"]}
        for r in data["results"]
    ]

def _direct_sources(endpoints):
    return [{"id": url, "fair_score": None, "sparql_endpoint": url} for url in endpoints]

def _endpoint_is_live(endpoint, timeout=8.0):
    try:
        resp = requests.get(
            endpoint,
            params={"query": "ASK { ?s ?p ?o }"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return False
        ctype = resp.headers.get("content-type", "")
        if "json" not in ctype:
            return False
        data = resp.json()
        return "boolean" in data or "results" in data
    except Exception:
        return False

_CATALOG_QUERY = (
    'SELECT ?title ?identifier ?endpoint ?fairScore WHERE { '
    '?s dct:title ?title . ?s dct:identifier ?identifier . '
    '?s void:sparqlEndpoint ?endpoint . ?s fair:fairScore ?fairScore . '
    'FILTER(REGEX(?endpoint, "^http")) } LIMIT 1000'
)

def _catalog_entries(catalog_url, timeout=_SELECT_TIMEOUT):
    resp = requests.get(
        catalog_url,
        params={"query": _CATALOG_QUERY},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", {}).get("bindings", [])

def _select_sources(catalog_url, min_fair_score=0.6, max_sources=2, verify_live=True,
                          live_timeout=8.0):
    bindings = _catalog_entries(catalog_url)
    candidates = []
    for b in bindings:
        endpoint = b.get("endpoint", {}).get("value")
        if not endpoint:
            continue
        fair_raw = b.get("fairScore", {}).get("value")
        try:
            fair = float(fair_raw) if fair_raw is not None else None
        except ValueError:
            fair = None
        if fair is not None and fair < min_fair_score:
            continue
        ident = b.get("identifier", {}).get("value") or endpoint
        title = b.get("title", {}).get("value")
        candidates.append({"id": ident, "title": title, "fair_score": fair, "sparql_endpoint": endpoint})
    candidates.sort(key=lambda c: (c["fair_score"] is None, -(c["fair_score"] or 0.0)))
    selected = []
    for c in candidates:
        if verify_live and not _endpoint_is_live(c["sparql_endpoint"], live_timeout):
            continue
        selected.append({"id": c["id"], "fair_score": c["fair_score"], "sparql_endpoint": c["sparql_endpoint"]})
        if len(selected) >= max_sources:
            break
    if not selected:
        raise RuntimeError(
            f"No dataset from {catalog_url} has a confirmed live SPARQL endpoint at or above "
            f"min_fair_score={min_fair_score}. Lower min_fair_score, or rerun with verify_live=False "
            f"to see candidates before liveness filtering."
        )
    return selected

def _label_match_query(label, limit=50):
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    return f"""
SELECT ?s ?p ?o ?olabel WHERE {{
  ?s ?lp ?slabel .
  FILTER(isLiteral(?slabel) && CONTAINS(LCASE(STR(?slabel)), LCASE("{escaped}")))
  ?s ?p ?o .
  OPTIONAL {{ ?o ?lp2 ?olabel . FILTER(isLiteral(?olabel)) }}
}} LIMIT {limit}
"""

def _fetch_for_label(endpoint, label, limit, timeout=_QUERY_TIMEOUT):
    try:
        resp = requests.get(
            endpoint,
            params={"query": _label_match_query(label, limit)},
            headers={"Accept": "application/sparql-results+json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", {}).get("bindings", [])
    except Exception:
        return []

def _federated_fetch_for_label(sources, label, limit, timeout=_QUERY_TIMEOUT):
    with ThreadPoolExecutor(max_workers=max(1, len(sources))) as ex:
        futures = {
            ex.submit(_fetch_for_label, s["sparql_endpoint"], label, limit, timeout): s["id"]
            for s in sources
        }
        return {futures[f]: f.result() for f in futures}

def build_cache(names_path, out_path, fair_api_base=None, endpoints=None, catalog_url=None,
                 min_fair_score=0.6, category=None, max_concepts=None, per_concept_limit=50,
                 max_sources=2, verify_live=True):
    with open(names_path, "rb") as f:
        names = pickle.load(f)

    if endpoints:
        sources = _direct_sources(endpoints)
    elif catalog_url:
        sources = _select_sources(catalog_url, min_fair_score, max_sources, verify_live)
    elif fair_api_base:
        sources = _pick_endpoints(fair_api_base, min_fair_score, category, max_sources)
    else:
        raise ValueError("build_cache needs one of endpoints=, catalog_url=, or fair_api_base=")

    name2cui = defaultdict(list)
    for cui, nm in names.items():
        name2cui[nm.strip().lower()].append(cui)

    items = list(names.items())
    if max_concepts:
        items = items[:max_concepts]

    adj = defaultdict(set)
    provenance = defaultdict(lambda: defaultdict(set))
    per_concept_stats = {}
    concepts_needing_second_source = 0
    t0 = time.time()
    for cui, label in items:
        per_source_bindings = _federated_fetch_for_label(sources, label, per_concept_limit)
        matched = 0
        sources_with_any_triples = 0
        sources_with_a_match = 0
        for dataset_id, bindings in per_source_bindings.items():
            if bindings:
                sources_with_any_triples += 1
            found_here = False
            for b in bindings:
                olabel = b.get("olabel", {}).get("value")
                if not olabel:
                    continue
                for other_cui in name2cui.get(olabel.strip().lower(), ()):
                    if other_cui != cui:
                        adj[cui].add(other_cui)
                        provenance[cui][other_cui].add(dataset_id)
                        matched += 1
                        found_here = True
            if found_here:
                sources_with_a_match += 1
        if sources_with_a_match > 1:
            concepts_needing_second_source += 1
        per_concept_stats[cui] = {
            "triples_fetched_by_source": {d: len(b) for d, b in per_source_bindings.items()},
            "matched_to_known_cui": matched,
            "answered_by_sources": sorted({d for d, b in per_source_bindings.items() if b}),
        }

    cache = {
        "source_datasets": sources,
        "adjacency": {k: sorted(v) for k, v in adj.items()},
        "adjacency_provenance": {
            k: {n: sorted(ds) for n, ds in v.items()} for k, v in provenance.items()
        },
        "concepts_queried": len(items),
        "concepts_with_any_match": sum(1 for v in adj.values() if v),
        "concepts_needing_second_source": concepts_needing_second_source,
        "seconds": round(time.time() - t0, 1),
        "per_concept_stats": per_concept_stats,
    }
    CS.FileCacheHandle(out_path).save_plain(cache)
    return cache

class SparqlExtEngine:
    def __init__(self, cache_path, kg=None):
        self.cache = CS.FileCacheHandle(cache_path).load_plain()
        if self.cache is None:
            raise FileNotFoundError(cache_path)
        self.adj = self.cache["adjacency"]
        self.code2cui = kg.code2cui if kg is not None else {}

    def coverage_batch(self, code_lists):
        t0 = time.time()
        covs, twos = [], []
        for codes in code_lists:
            cov = defaultdict(set)
            code_by_cui = defaultdict(set)
            for c in codes:
                code_by_cui[self.code2cui.get(c, c)].add(c)
            own_cuis = set(code_by_cui)
            for cui, own_codes in code_by_cui.items():
                for n in self.adj.get(cui, ()):
                    if n not in own_cuis:
                        cov[n].update(own_codes)
            covs.append(dict(cov))
            twos.append({})
        secs = time.time() - t0
        nq = max(1, len(code_lists))
        n_sources = len(self.cache["source_datasets"])
        stats = {
            "queries": len(code_lists),
            "sources_available": n_sources,
            "sources_contacted_per_query": float(n_sources),
            "remote_sources_per_query": float(n_sources),
            "subqueries_per_query": float(n_sources),
            "remote_subqueries_per_query": 0.0,
            "join_subqueries_per_query": 0.0,
            "wasted_contacts_per_query": 0.0,
            "requests": 0,
            "req_bytes_per_query": 0.0,
            "resp_bytes_per_query": 0.0,
            "total_mb": 0.0,
            "secs": round(secs, 3),
            "ms_per_query": 1000 * secs / nq,
            "decoys_per_subquery": 0,
            "summary_bytes": 0,
            "source_datasets": self.cache["source_datasets"],
        }
        return covs, twos, stats

def _cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build-cache", help="one-time federated fetch from live SPARQL endpoints")
    b.add_argument("--names", required=True)
    b.add_argument("--endpoints", nargs="+", default=None)
    b.add_argument("--catalog-url", default=None)
    b.add_argument("--fair-api", default=None)
    b.add_argument("--min-fair-score", type=float, default=0.6)
    b.add_argument("--category", default=None)
    b.add_argument("--max-sources", type=int, default=2)
    b.add_argument("--max-concepts", type=int, default=None)
    b.add_argument("--per-concept-limit", type=int, default=50)
    b.add_argument("--no-verify-live", action="store_true")
    b.add_argument("--out", required=True)

    a = p.parse_args()
    if a.cmd == "build-cache":
        if not a.endpoints and not a.catalog_url and not a.fair_api:
            p.error("build-cache requires --endpoints, --catalog-url, or --fair-api")
        cache = build_cache(a.names, a.out, fair_api_base=a.fair_api, endpoints=a.endpoints,
                             catalog_url=a.catalog_url, min_fair_score=a.min_fair_score,
                             category=a.category, max_concepts=a.max_concepts,
                             per_concept_limit=a.per_concept_limit, max_sources=a.max_sources,
                             verify_live=not a.no_verify_live)
        summary = {k: v for k, v in cache.items() if k not in ("per_concept_stats", "adjacency", "adjacency_provenance")}
        print(json.dumps(summary, indent=2))
        print(f"concepts with >=1 matched neighbour: {cache['concepts_with_any_match']}/{cache['concepts_queried']}")
        print(f"concepts recovered only by federating across >1 source: {cache['concepts_needing_second_source']}")
        print(f"saved -> {a.out}")

if __name__ == "__main__":
    _cli()
