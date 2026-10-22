from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from collections import defaultdict

import numpy as np
import requests

import fedrag_sim as F
import fqp as Q

SYSTEM_PROMPT = (
    "You control knowledge-graph retrieval for ONE patient in a federated system of hospitals. "
    "You never see patient codes or concepts, only counts. Goal: obtain enough well-supported candidate "
    "concepts (wanted) while contacting as few sources as possible. Every turn you receive a JSON observation "
    "and must reply with exactly ONE JSON object and nothing else, one of: "
    '{"action":"finish"} ; '
    '{"action":"query","sources":[ids taken from unqueried_sources]} ; '
    '{"action":"expand"} (second-hop bound join over the best candidates; costs more requests, use it only when '
    'the candidate set is small or unstable) ; '
    '{"action":"endpoint","name":"external"} (external SPARQL endpoint, only if listed in external_available). '
    "Each queried source costs a request. Finish as soon as the candidates are sufficient: candidates at least "
    "three times wanted, high top_mean_support, or when more sources add little (top_set_stability close to 1)."
)

class Session:
    def __init__(self, engine, kg, codes, df, n_docs, budget, join_w, ext=None):
        self.eng, self.kg, self.budget = engine, kg, budget
        self.df, self.n_docs, self.join_w, self.ext = df, n_docs, join_w, ext
        self.home = engine.home
        self.st = {"secs": 0.0, "req_bytes": 0, "resp_bytes": 0, "requests": 0, "subqueries": 0,
                   "remote_subqueries": 0, "wasted": 0, "per_query_sources": defaultdict(set),
                   "join_subqueries": 0}
        c2c = kg.code2cui
        self.m = defaultdict(set)
        for c in codes:
            e = engine.e2i.get(c2c.get(c))
            if e is not None:
                self.m[e].add(c)
        self.codes = codes
        self.cov = defaultdict(set)
        self.two = defaultdict(set)
        self.ext_cov = {}
        self.queried = set()
        self.expanded = False
        self.ext_calls = 0
        self.prev_top = []

    def _merge(self, res):
        own = set(self.m)
        for e, pairs in res.items():
            if e not in self.m:
                continue
            for _, o in pairs:
                if o not in own:
                    self.cov[o].update(self.m[e])

    def query(self, srcs):
        srcs = [s for s in srcs if s in self.eng.summ and s not in self.queried]
        if not srcs or not self.m:
            return
        plan = self.eng._plan([sorted(self.m)])
        plan = {s: v for s, v in plan.items() if s in srcs}
        self.queried.update(srcs)
        if not plan:
            return
        res = self.eng._execute(plan, self.st, 1)
        self._merge(res[0])

    def expand(self, top=None, srcs=None):
        if self.expanded:
            return
        top = int(top or self.eng.join_top)
        bound = sorted(self.cov, key=lambda x: -len(self.cov[x]))[:top]
        self.expanded = True
        if not bound:
            return
        plan = self.eng._plan([bound])
        if srcs:
            plan = {s: v for s, v in plan.items() if s in srcs}
        sub0 = self.st["subqueries"]
        res = self.eng._execute(plan, self.st, 1)
        self.st["join_subqueries"] += self.st["subqueries"] - sub0
        bset = set(bound)
        for x, pairs in res[0].items():
            if x not in bset:
                continue
            for _, y in pairs:
                if y in self.cov:
                    self.two[y].update(self.cov[x])

    def endpoint(self):
        if self.ext is None or self.ext_calls:
            return
        t0 = time.time()
        covs, _, _ = self.ext.coverage_batch([self.codes])
        self.st["secs"] += time.time() - t0
        self.ext_calls += 1
        for n, cs in covs[0].items():
            self.ext_cov.setdefault(n, set()).update(cs)

    def result(self):
        ents = self.eng.ents
        cov = {ents[k]: set(v) for k, v in self.cov.items()}
        for n, cs in self.ext_cov.items():
            cov.setdefault(n, set()).update(cs)
        two = {ents[k]: set(v) for k, v in self.two.items()}
        return cov, two

    def observation(self, step):
        cov, two = self.result()
        top = Q.rank_from_cov(cov, two, self.df, self.n_docs, self.budget, self.join_w, self.kg.deg)
        cur = set(top)
        prev = set(self.prev_top)
        stable = round(len(prev & cur) / max(1, len(prev | cur)), 3) if self.prev_top else None
        self.prev_top = top
        resolved = set().union(*cov.values()) if cov else set()
        res_pct = sum(1 for cs in self.m.values() if cs & resolved) / max(1, len(self.m))
        supports = [len(cov[n]) for n in top]
        unq = {}
        if self.m:
            qa = np.array(sorted(self.m), dtype=np.int64)
            for s, bl in self.eng.summ.items():
                if s in self.queried:
                    continue
                k = int(bl.contains(qa).sum())
                if k:
                    unq[int(s)] = k
        return {"step": step, "patient_entities": len(self.m), "candidates": len(cov), "wanted": self.budget,
                "top_mean_support": round(float(np.mean(supports)), 2) if supports else 0.0,
                "entities_with_candidates": round(res_pct, 3), "top_set_stability": stable,
                "unqueried_sources": unq, "queried_sources": sorted(int(s) for s in self.queried),
                "expanded": self.expanded,
                "external_available": ["external"] if (self.ext is not None and not self.ext_calls) else [],
                "requests_so_far": self.st["subqueries"] + self.ext_calls}

class RulePolicy:
    name = "rule"

    def __init__(self, pool=3.0, k=3):
        self.pool, self.k = pool, k
        self.invalid = self.errors = self.calls = self.tokens = 0
        self.secs = 0.0

    def start(self):
        pass

    def act(self, obs):
        unq = obs["unqueried_sources"]
        if obs["step"] == 0 and unq and obs["candidates"] < self.pool * obs["wanted"]:
            top = sorted(unq, key=lambda s: -unq[s])[:self.k]
            return {"action": "query", "sources": top}
        return {"action": "finish"}

class LLMPolicy:
    name = "agent"

    def __init__(self, url, model, key=None, timeout=180, fallback=None):
        self.url, self.model, self.key, self.timeout = url, model, key, timeout
        self.fallback = fallback or RulePolicy()
        self.calls = self.tokens = self.invalid = self.errors = 0
        self.secs = 0.0
        self.msgs = []

    def start(self):
        self.msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _validate(self, act, obs):
        if not isinstance(act, dict):
            return None
        a = act.get("action")
        if a == "finish":
            return {"action": "finish"}
        if a == "query":
            unq = {str(s) for s in obs["unqueried_sources"]}
            srcs = [int(s) for s in act.get("sources", []) if str(s) in unq]
            return {"action": "query", "sources": srcs} if srcs else None
        if a == "expand":
            return None if obs["expanded"] or obs["candidates"] == 0 else {"action": "expand"}
        if a == "endpoint":
            return {"action": "endpoint", "name": "external"} if obs["external_available"] else None
        return None

    def act(self, obs):
        self.msgs.append({"role": "user", "content": json.dumps(obs)})
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        t0 = time.time()
        try:
            r = requests.post(self.url, headers=headers, timeout=self.timeout,
                              json={"model": self.model, "messages": self.msgs, "temperature": 0, "max_tokens": 120})
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            self.tokens += int((data.get("usage") or {}).get("total_tokens", 0))
        except Exception:
            self.secs += time.time() - t0
            self.calls += 1
            self.errors += 1
            self.msgs.pop()
            return self.fallback.act(obs)
        self.secs += time.time() - t0
        self.calls += 1
        self.msgs.append({"role": "assistant", "content": text})
        act = None
        try:
            act = json.loads(text[text.index("{"):text.rindex("}") + 1])
        except Exception:
            act = None
        act = self._validate(act, obs)
        if act is None:
            self.invalid += 1
            return self.fallback.act(obs)
        return act

def run_session(engine, kg, codes, df, n_docs, budget, join_w, policy, ext, max_steps):
    s = Session(engine, kg, codes, df, n_docs, budget, join_w, ext)
    s.query([engine.home])
    policy.start()
    trace = []
    for step in range(max_steps + 1):
        obs = s.observation(step)
        act = {"action": "finish"} if step == max_steps else policy.act(obs)
        trace.append({"obs": obs, "act": act})
        a = act["action"]
        if a == "finish":
            break
        if a == "query":
            s.query(act["sources"])
        elif a == "expand":
            s.expand()
        elif a == "endpoint":
            s.endpoint()
    return s, trace

def build_items(kg, patients, client, rng_mask, mask):
    items = []
    for i in client["test"]:
        codes = F.flat(patients[i])
        dx = [x for x in codes if x.startswith("D_") and x in kg.code2cui]
        if len(dx) < 3:
            continue
        hidden = rng_mask.sample(dx, max(1, int(round(mask * len(dx)))))
        query = [x for x in codes if x not in hidden]
        qcuis = {kg.code2cui.get(q) for q in query}
        gains, covers = {}, defaultdict(set)
        for h in hidden:
            hc = kg.code2cui[h]
            if hc in qcuis:
                continue
            gains[hc] = 2
            covers[hc].add(h)
            for nb in kg.adj.get(hc, ()):
                if nb not in qcuis:
                    gains.setdefault(nb, 1)
                    covers[nb].add(h)
        n_hidden = len({h for v in covers.values() for h in v})
        if not gains or n_hidden == 0:
            continue
        items.append((query, gains, covers, n_hidden))
    return items

ARM_NAME = {"local": "Client-side RAG, own KG only", "fqp": "FQP RAG (always federate)",
            "rule": "Rule-based adaptive FQP", "agent": "Agentic FQP RAG"}

def cmd_eval(a):
    with open(a.cohort, "rb") as f:
        patients = pickle.load(f)["patients"]
    kg = F.KG(pickle.load(open(a.kg, "rb")))
    ext = None
    if a.sparql_cache:
        import sparql_ext as SE
        ext = SE.SparqlExtEngine(a.sparql_cache, kg)
    rows, samples = [], []
    for seed in a.seeds:
        clients = F.make_clients(patients, a.clients, a.partition, a.alpha, seed)
        a.seed = seed
        sources, summaries, _ = F.fqp_setup(kg, a.clients, a)
        rng_mask = random.Random(seed)
        groups = []
        for ci, c in enumerate(clients):
            ldf, lN = F.doc_freq(kg, [patients[i] for i in c["train"]])
            groups.append([ci, ldf, lN, build_items(kg, patients, c, rng_mask, a.mask)])
        flat_idx = [(g, j) for g, grp in enumerate(groups) for j in range(len(grp[3]))]
        if a.max_queries and len(flat_idx) > a.max_queries:
            keep = set(random.Random(seed).sample(range(len(flat_idx)), a.max_queries))
            groups = [[ci, ldf, lN, [it for j, it in enumerate(items) if (flat_idx.index((g, j)) in keep)]]
                      for g, (ci, ldf, lN, items) in enumerate(groups)]
        for arm in a.arms:
            agg = defaultdict(list)
            cost = defaultdict(list)
            policy = None
            if arm == "rule":
                policy = RulePolicy(a.rule_pool, a.rule_k)
            elif arm == "agent":
                model = a.llm_model or os.environ.get("AGENT_LLM_MODEL")
                if not model:
                    raise SystemExit("set --llm_model or AGENT_LLM_MODEL for the agent arm")
                policy = LLMPolicy(a.llm_url or os.environ.get("AGENT_LLM_URL", "http://host.docker.internal:11434/v1/chat/completions"),
                                   model, os.environ.get("AGENT_LLM_KEY"), a.llm_timeout, RulePolicy(a.rule_pool, a.rule_k))
            stop_local = 0
            n_actions = []
            wall = 0.0
            for ci, ldf, lN, items in groups:
                if not items:
                    continue
                if arm in ("local", "fqp"):
                    eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci, decoys=0,
                                      hops=1 if arm == "local" else a.fqp_hops, join_top=a.fqp_join_top,
                                      seed=seed * 31 + ci, sources_allowed={ci} if arm == "local" else None)
                    t0 = time.perf_counter()
                    covs, twos, st = eng.coverage_batch([q for q, _, _, _ in items])
                    wall += time.perf_counter() - t0
                    rankings = [Q.rank_from_cov(cv, tw, ldf, lN, a.depth, join_w=a.fqp_join_w, deg=kg.deg)
                                for cv, tw in zip(covs, twos)]
                    for _ in items:
                        cost["sources"].append(st["sources_contacted_per_query"])
                        cost["subq"].append(st["subqueries_per_query"])
                        cost["remote_subq"].append(st["remote_subqueries_per_query"])
                        cost["engine_ms"].append(st["ms_per_query"])
                else:
                    eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci, decoys=0, hops=2,
                                      join_top=a.fqp_join_top, seed=seed * 31 + ci)
                    rankings = []
                    for query, _, _, _ in items:
                        t0 = time.perf_counter()
                        s, trace = run_session(eng, kg, query, ldf, lN, a.budget, a.fqp_join_w, policy, ext, a.max_steps)
                        wall += time.perf_counter() - t0
                        cov, two = s.result()
                        rankings.append(Q.rank_from_cov(cov, two, ldf, lN, a.depth, join_w=a.fqp_join_w, deg=kg.deg))
                        acts = [t["act"]["action"] for t in trace]
                        n_actions.append(len(acts) - 1)
                        stop_local += int(acts[0] == "finish")
                        cost["sources"].append(len(s.st["per_query_sources"][0]) + s.ext_calls)
                        cost["subq"].append(s.st["subqueries"] + s.ext_calls)
                        cost["remote_subq"].append(s.st["remote_subqueries"] + s.ext_calls)
                        cost["engine_ms"].append(1000 * s.st["secs"])
                        if len(samples) < a.save_traces and arm == "agent":
                            samples.append({"seed": seed, "client": ci, "actions": acts,
                                            "final": trace[-1]["obs"]})
                for (query, gains, covers, n_hidden), ranked in zip(items, rankings):
                    for key, v in F.ir_metrics(ranked, gains, covers, n_hidden, a.k).items():
                        agg[key].append(v)
            nq = len(agg["MRR"])
            row = {"clients": a.clients, "seed": seed, "arm": arm, "n_queries": nq,
                   "wall_ms_per_query": 1000 * wall / max(1, nq)}
            row.update({k: float(np.mean(v)) for k, v in agg.items()})
            row.update({"sources_per_query": float(np.mean(cost["sources"])) if cost["sources"] else None,
                        "subqueries_per_query": float(np.mean(cost["subq"])) if cost["subq"] else None,
                        "remote_subqueries_per_query": float(np.mean(cost["remote_subq"])) if cost["remote_subq"] else None,
                        "engine_ms_per_query": float(np.mean(cost["engine_ms"])) if cost["engine_ms"] else None})
            if arm in ("rule", "agent"):
                row.update({"stop_after_local_frac": stop_local / max(1, nq),
                            "actions_per_query": float(np.mean(n_actions)) if n_actions else 0.0,
                            "policy_calls": policy.calls, "policy_tokens": policy.tokens,
                            "policy_secs": round(policy.secs, 2), "invalid_actions": policy.invalid,
                            "llm_errors": policy.errors})
            rows.append(row)
            F.log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"rows": rows, "traces": samples, "args": {k: v for k, v in vars(a).items() if k != "cmd"}},
              open(a.out, "w"), indent=1)
    write_table(rows, os.path.splitext(a.out)[0] + ".tex", a.arms)
    F.log("saved", a.out)

def _ms(v, d=3):
    v = [x for x in v if x is not None]
    if not v:
        return "--"
    return f"{np.mean(v):.{d}f}" + (f"$\\pm${np.std(v):.{d}f}" if len(v) > 1 else "")

def write_table(rows, path, arms):
    lines = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
             "Retriever & P@10 & nDCG@10 & Hit@20 & Sources/q & Subq./q & ms/q \\\\", "\\midrule"]
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        if not rr:
            continue
        lines.append(f"{ARM_NAME[arm]} & {_ms([r['P@10'] for r in rr])} & {_ms([r['nDCG@10'] for r in rr])} & "
                     f"{_ms([r['Hit@20'] for r in rr])} & {_ms([r['sources_per_query'] for r in rr], 2)} & "
                     f"{_ms([r['subqueries_per_query'] for r in rr], 1)} & {_ms([r['wall_ms_per_query'] for r in rr], 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")

def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("eval")
    e.add_argument("--cohort", required=True)
    e.add_argument("--kg", required=True)
    e.add_argument("--partition", default="noniid")
    e.add_argument("--alpha", type=float, default=0.5)
    e.add_argument("--clients", type=int, default=10)
    e.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    e.add_argument("--arms", nargs="+", default=["local", "fqp", "rule", "agent"],
                   choices=["local", "fqp", "rule", "agent"])
    e.add_argument("--max_queries", type=int, default=300)
    e.add_argument("--mask", type=float, default=0.3)
    e.add_argument("--depth", type=int, default=100)
    e.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    e.add_argument("--budget", type=int, default=16)
    e.add_argument("--max_steps", type=int, default=4)
    e.add_argument("--fqp_repl", type=int, default=2)
    e.add_argument("--fqp_hops", type=int, default=2)
    e.add_argument("--fqp_join_top", type=int, default=20)
    e.add_argument("--fqp_join_w", type=float, default=0.2)
    e.add_argument("--rule_pool", type=float, default=3.0)
    e.add_argument("--rule_k", type=int, default=3)
    e.add_argument("--llm_url", default=None)
    e.add_argument("--llm_model", default=None)
    e.add_argument("--llm_timeout", type=float, default=180)
    e.add_argument("--sparql_cache", default=None)
    e.add_argument("--save_traces", type=int, default=30)
    e.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_eval(a)

if __name__ == "__main__":
    main()
