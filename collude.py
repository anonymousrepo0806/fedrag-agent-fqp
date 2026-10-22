import json, pickle, sys, types
import numpy as np
from collections import defaultdict
import fedrag_sim as F
import fqp as Q

cohort, kgp, out = sys.argv[1], sys.argv[2], sys.argv[3]
patients = pickle.load(open(cohort, "rb"))["patients"]
kg = F.KG(pickle.load(open(kgp, "rb")))
N = 10
res = defaultdict(lambda: defaultdict(list))
ask = defaultdict(list)
for seed in (1, 2, 3):
    clients = F.make_clients(patients, N, "noniid", 0.5, seed)
    a = types.SimpleNamespace(fqp_repl=2, seed=seed)
    sources, summaries, _ = F.fqp_setup(kg, N, a)
    rng = np.random.default_rng(seed)
    for ci, c in enumerate(clients):
        idx = list(c["test"])[:120]
        eng0 = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci)
        e2i = eng0.e2i
        queries = []
        for i in idx:
            ents = {e2i.get(kg.code2cui.get(x)) for x in F.flat(patients[i])}
            ents.discard(None)
            if ents:
                queries.append(sorted(ents))
        if not queries:
            continue
        remote = [s for s in range(N) if s != ci]
        for d in (0, 2, 5):
            eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci, decoys=d, seed=seed * 31 + ci)
            plan = eng._plan(queries)
            seen = [defaultdict(set) for _ in queries]
            for s, items in plan.items():
                if s == ci:
                    continue
                for j, sel in items:
                    seen[j][s] = set(sel)
            for j, q in enumerate(queries):
                real = set(q)
                for cs in (1, 2, 3, 5):
                    for _ in range(10):
                        co = rng.choice(remote, size=cs, replace=False)
                        cnt = defaultdict(int)
                        for s in co:
                            for e in seen[j].get(int(s), ()):
                                cnt[e] += 1
                        union = set(cnt)
                        rec = len(union & real) / len(real)
                        prec = len(union & real) / len(union) if union else 1.0
                        res[(d, cs)]["recall"].append(rec)
                        res[(d, cs)]["precision"].append(prec)
                        if cs >= 2:
                            flag = {e for e, v in cnt.items() if v >= 2}
                            res[(d, cs)]["h_recall"].append(len(flag & real) / len(real))
                            res[(d, cs)]["h_precision"].append(len(flag & real) / len(flag) if flag else 1.0)
                if d == 0:
                    per = [len(seen[j].get(s, ())) for s in remote]
                    res[(0, "per_src")]["ents_per_contacted"].append(float(np.mean([p for p in per if p > 0] or [0])))
                    nseed = len(q)
                    ask["seeds_per_query"].append(nseed)
                    ask["bloom_ents_per_remote_source"].append(float(np.mean(per)))
                    ask["probe_req_bytes"].append(len(remote) * len(json.dumps(q)))
                    ask["probe_resp_bytes"].append(len(remote) * len(json.dumps([True] * nseed)))
                    exact = 0
                    for s in remote:
                        has = np.isin(np.array(q), sources[s].ents).any()
                        contacted = s in seen[j]
                        if contacted and not has:
                            exact += 1
                    ask["bloom_wasted"].append(exact)
rows = {f"d{d}_c{c}": {k: float(np.mean(v)) for k, v in m.items()} for (d, c), m in res.items()}
asks = {k: float(np.mean(v)) for k, v in ask.items()}
json.dump({"collusion": rows, "ask": asks}, open(out, "w"), indent=1)
print(json.dumps(asks, indent=1))
for k in sorted(rows):
    print(k, {a: round(b, 3) for a, b in rows[k].items()})
