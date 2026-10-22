
"""
fl_net.py -- the same experiment as fedrag_sim.py, as separate services that
talk over HTTP (one Docker container each, all on one CPU machine).

  prep    builds cohort + KG + vocab once per task, writes one shard per client
  server  FL coordinator: aggregation, server-side RAG (per-code KG lookups),
          metrics, result files
  client  local retrieval (client-side RAG), local training with the chosen
          method's client-side term (FedProx / FedCurv / FedLwF / PARC),
          local evaluation, local drug-hypothesis generation, its own local
          cases (/user/client_<i>.csv) added to training

Federated query processing (arm fqp): every client container holds only
ITS OWN KG partition, publishes a Bloom summary via the server's directory,
and answers sub-queries from other clients on its own endpoint (:8090); the
patient's client is the query engine (source selection, decomposition,
bound join) and talks to the other clients directly. client_local = own KG
partition only.

Docker supports: tasks mortality_next | ade_next | ade | multitask;
arms none | random | server | client | client_local | fqp; FL fedavg | fedprox | fedcurv | fedlwf |
parc; models sage | bag | retain | transformer; --hypotheses; user data.
client_kgc, server_ctx, fedasync, local and central run in fedrag_sim.py.
"""

from __future__ import annotations

import argparse
import json
import socket
import os
import pickle
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import fedrag_sim as F
from edge_controller import EdgeController
from connection_module import HTTPConnectionModule

DOCKER_ARMS = ("none", "random", "server", "client", "client_local", "fqp")
DOCKER_FL = ("fedavg", "fedprox", "fedcurv", "fedlwf", "parc")

def env(name, default):
    return os.environ.get(name, default)

def task_dir(a):
    return os.path.join(a.work, a.task)

def split_tag(a):
    return f"c{a.clients}__{a.partition}__seed{a.seed}"

def run_tag(a):
    return F.run_tag(a)

def measurements_path(a):
    """lab / vital findings, computed once for all tasks (one streamed pass over
    LABEVENTS + CHARTEVENTS) and cached in /work/measurements.pkl"""
    p = os.path.join(a.work, "measurements.pkl")
    if os.path.exists(p):
        return p
    have = [t for t in ("LABEVENTS.csv", "CHARTEVENTS.csv", "LAB_FINDINGS.csv", "VITAL_FINDINGS.csv")
            if F.find_table(a.mimic, t)]
    if not have:
        F.log("note: LABEVENTS / CHARTEVENTS (or LAB_FINDINGS / VITAL_FINDINGS) not in the MIMIC folder -> no lab / vital findings "
              "(codes only). Add them and delete work/ to include them.")
        return None
    F.log("extracting lab / vital findings from " + " + ".join(have) + " (one pass; CHARTEVENTS takes a while)")
    F.MS.cmd_measurements(argparse.Namespace(mimic=a.mimic, out=p, kinds=a.meas_kinds, frac=a.meas_frac,
                                             chunksize=1_000_000))
    return p

def preflight(a):
    probs = []
    td = task_dir(a)
    if not os.path.exists(os.path.join(td, "cohort.pkl")):
        if not os.path.isdir(a.mimic):
            probs.append(f"MIMIC folder not mounted: {a.mimic}")
        else:
            for t in ("ADMISSIONS.csv", "DIAGNOSES_ICD.csv", "PROCEDURES_ICD.csv", "PRESCRIPTIONS.csv"):
                if F.find_table(a.mimic, t) is None:
                    probs.append(f"missing {t} (or .csv.gz) in the MIMIC folder")
            if probs:
                probs.append("MIMIC folder contains: " + ", ".join(sorted(os.listdir(a.mimic))[:20]))
    if not os.path.exists(os.path.join(td, "kg.pkl")):
        for rel in ("ICD9CM_to_UMLS.csv", "umls.csv", "concept_names.txt"):
            if F.kg_file(a.kgdir, rel) is None:
                probs.append(f"missing KG file {rel} in the KG folder")
        if os.path.isdir(a.kgdir) and any("KG file" in p for p in probs):
            probs.append("KG folder contains: " + ", ".join(sorted(os.listdir(a.kgdir))[:20]))
        if F.kg_file(a.kgdir, "ATC.csv") is None or F.kg_file(a.kgdir, "ATC_to_UMLS.csv") is None:
            F.log("note: ATC.csv / ATC_to_UMLS.csv not found -> drug hypotheses will find no drugs")
    try:
        os.makedirs(td, exist_ok=True)
        open(os.path.join(td, ".write_test"), "w").close()
        os.remove(os.path.join(td, ".write_test"))
    except OSError as e:
        probs.append(f"cannot write to {td}: {e}")
    if probs:
        F.log("PREP CANNOT START:")
        for p in probs:
            F.log("  - " + p)
        sys.exit(1)

def cmd_prep(a):
    preflight(a)
    m = F.Meter()
    td = task_dir(a)
    cohort, kgp, vp = (os.path.join(td, x) for x in ("cohort.pkl", "kg.pkl", "vocab.pkl"))
    user = a.user_data if os.path.isdir(a.user_data) else None
    if not os.path.exists(cohort):
        F.cmd_build(argparse.Namespace(mimic=a.mimic, task=a.task, no_drugs=False, max_codes_per_visit=128,
                                       exclude_file=os.path.join(a.user_data, "reserved_subjects.txt"),
                                       measurements=measurements_path(a), out=cohort))
    if not os.path.exists(kgp):
        F.cmd_kg(argparse.Namespace(cohort=cohort, kgdir=a.kgdir, user_data=user, out=kgp))
    patients = pickle.load(open(cohort, "rb"))["patients"]
    kg = F.KG(pickle.load(open(kgp, "rb")))
    if not os.path.exists(vp):
        extra = []
        if user:
            for p in sorted(os.listdir(user)):
                if p.startswith("client") and p.endswith(".csv"):
                    extra += F.load_user_csv(os.path.join(user, p), a.task)
        pickle.dump(F.build_vocab(patients + extra, kg), open(vp, "wb"), protocol=4)
    d = os.path.join(td, split_tag(a))
    if os.path.exists(os.path.join(d, "DONE")):
        F.log("shards exist:", d)
        return
    os.makedirs(d, exist_ok=True)
    for ci, c in enumerate(F.make_clients(patients, a.clients, a.partition, a.alpha, a.seed)):
        ids = sorted({i for s in c.values() for i in s})
        remap = {i: j for j, i in enumerate(ids)}
        shard = {"patients": [patients[i] for i in ids],
                 "splits": {s: [remap[i] for i in v] for s, v in c.items()}}
        pickle.dump(shard, open(os.path.join(d, f"client_{ci}.pkl"), "wb"), protocol=4)
    st = F.partition_stats(patients, F.make_clients(patients, a.clients, a.partition, a.alpha, a.seed))
    json.dump(st, open(os.path.join(d, "partition_stats.json"), "w"), indent=1)
    F.log(f"partition {a.partition}: sizes {[c['n'] for c in st['clients']]}, "
          f"mean JS divergence {st['mean_js_divergence']:.3f}")
    open(os.path.join(d, "DONE"), "w").write(json.dumps({"resources": m.read()}))
    F.log("prep done:", d, m.read())

class State:
    def __init__(self, a):
        import torch
        torch.manual_seed(a.seed)
        self.a = a
        self.heads = F.TASK_HEADS[a.task]
        td = task_dir(a)
        self.lock = threading.Condition()
        self.vocab_size = len(pickle.load(open(os.path.join(td, "vocab.pkl"), "rb")))
        model = F.make_model(a.model, self.vocab_size, a.dim, a.layers, a.dropout, len(self.heads))
        self.n_params = sum(p.numel() for p in model.parameters())
        self.g = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.blobs = {0: F.pack_state(self.g, a.comm_dtype)}
        self.best_blob, self.best, self.best_score, self.last = None, None, None, None
        self.registered = 0
        self.updates, self.evals, self.hyps = {}, {}, {}
        self.kg = F.KG(pickle.load(open(os.path.join(td, "kg.pkl"), "rb"))) if a.arm == "server" else None
        self.bytes_in = self.bytes_out = 0
        self.done = False
        self.t0 = time.time()
        self.meter = F.Meter()
        self.losses = []
        self.out = os.path.join(a.out, run_tag(a))
        os.makedirs(a.out, exist_ok=True)
        open(self.out + ".jsonl", "w").close()
        self.last_evs = None
        self.sources = {}

        import hypo_learn as HL
        self.hyp_names = HL.feature_names(sorted(F.HYP_REL_W))
        self.hyp_stats, self.hyp_norm, self.hyp_upd = {}, None, {}
        self.hyp_g = HL._state(HL.make_ranker(len(self.hyp_names), a.hyp_model, a.seed))
        self.hyp_blobs = {0: F.pack_state(self.hyp_g)}

    def aggregate(self, rnd):
        ups = self.updates.pop(rnd)
        W = float(sum(n for n, _, _ in ups.values()))
        for k in self.g:
            if self.g[k].is_floating_point():
                self.g[k] = self.g[k] + sum(d[k] * (n / W) for n, d, _ in ups.values())
        self.blobs[rnd] = F.pack_state(self.g, self.a.comm_dtype)
        self.blobs.pop(rnd - 2, None)
        self.losses = [l for _, _, l in ups.values() if l is not None]

    def finish_round(self, rnd):
        evs = self.evals.pop(rnd)
        cids = sorted(evs)
        H = len(self.heads)
        arr = lambda key, c: np.array(evs[c][key], dtype=float).reshape(-1, H)
        Yv = np.concatenate([arr("val_y", c) for c in cids])
        Pv = np.concatenate([arr("val_p", c) for c in cids])
        Yt = np.concatenate([arr("test_y", c) for c in cids])
        Pt = np.concatenate([arr("test_p", c) for c in cids])
        per = [(arr("test_y", c), arr("test_p", c)) for c in cids]
        Hd = F.evaluate_heads(Yv, Pv, Yt, Pt, per, self.heads)
        prim = Hd[self.heads[0]]
        rec = {"round": rnd, "train_loss": float(np.mean(self.losses)) if self.losses else None,
               "val": prim["val"], "test": prim["test"], "per_client_test": prim["per_client"],
               "worst_client_auroc": prim["worst_client_auroc"], "std_client_auroc": prim["std_client_auroc"],
               "heads": {h: {"val": Hd[h]["val"], "test": Hd[h]["test"],
                             "worst_client_auroc": Hd[h]["worst_client_auroc"],
                             "std_client_auroc": Hd[h]["std_client_auroc"]} for h in self.heads},
               "resources": {"server": self.meter.read(), "clients": {c: evs[c]["resources"] for c in cids},
                             "server_bytes_in": self.bytes_in, "server_bytes_out": self.bytes_out}}
        with open(self.out + ".jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        F.log(f"round {rnd}: " + " | ".join(
            f"{h} val_auprc={Hd[h]['val'].get('auprc', float('nan')):.3f} test_auroc={Hd[h]['test'].get('auroc', float('nan')):.3f}"
            for h in self.heads))
        sc = F.selection_score(Hd)
        if sc is not None and (self.best_score is None or sc > self.best_score):
            self.best, self.best_score, self.best_blob = rec, sc, self.blobs[rnd]
        self.last, self.last_evs = rec, evs
        if rnd == self.a.rounds:
            if self.best is None:
                self.best, self.best_blob = rec, self.blobs[rnd]
            if not self.a.hypotheses:
                self.write_summary()

    def write_summary(self, hyp=None):
        a, b, evs = self.a, self.best, self.last_evs
        cl = [e["resources"] for e in evs.values()]
        summ = {"tag": run_tag(a), "task": a.task, "heads": self.heads, "fl": a.fl, "model": a.model,
                "arm": a.arm, "meas": int(getattr(a, "meas", 1)), "clients": a.clients, "seed": a.seed, "partition": a.partition,
                "budget": a.budget, "rounds": a.rounds, "local_epochs": a.local_epochs,
                "lam_ewc": a.lam_ewc, "lam_kd": a.lam_kd, "mu": a.mu, "mode": "docker",
                "user_data": any(e.get("n_user", 0) for e in evs.values()),
                "best_round": b["round"], "val_at_best": b["val"], "test_at_best_val": b["test"],
                "heads_at_best": b["heads"], "per_client_test_at_best": b["per_client_test"],
                "worst_client_auroc_at_best": b["worst_client_auroc"],
                "std_client_auroc_at_best": b["std_client_auroc"],
                "final_test": self.last["test"], "final_heads": self.last["heads"],
                "graph_stats": {k: float(np.mean([e["graph_stats"][k] for e in evs.values()]))
                                for k in next(iter(evs.values()))["graph_stats"]},
                "client_sizes": [e["sizes"] for e in evs.values()],
                "hypotheses": hyp, "kgc": None,
                "resources": {
                    "retrieval_and_graphs": {"wall_s": max(c["retrieval_wall_s"] for c in cl)},
                    "total": {"wall_s": round(time.time() - self.t0, 2),
                              "peak_rss_mb": max([c["peak_rss_mb"] for c in cl] + [F.peak_rss_mb()]),
                              "server_peak_rss_mb": round(F.peak_rss_mb(), 1),
                              "client_peak_rss_mb": [c["peak_rss_mb"] for c in cl],
                              "client_cpu_s": [c["cpu_s"] for c in cl],
                              "server_cpu_s": round(F.cpu_secs(), 2)},
                    "n_params": int(self.n_params), "model_mb": round(self.n_params * 4 / 2**20, 2),
                    "msg_down_mb": round(len(self.blobs[max(self.blobs)]) / 2**20, 3),
                    "total_mb_up": round(self.bytes_in / 2**20, 2),
                    "total_mb_down": round(self.bytes_out / 2**20, 2),
                    "mean_round_wall_s": round((time.time() - self.t0) / a.rounds, 3),
                    "infer_ms_per_patient": float(np.mean([c["infer_ms_per_patient"] for c in cl])),
                    "comm_dtype": a.comm_dtype, "threads": a.threads}}
        fq = [e["fqp"] for e in evs.values() if e.get("fqp")]
        if fq:
            summ["fqp"] = {**F.merge_fqp_stats(fq), "hops": a.fqp_hops if a.arm == "fqp" else 1,
                           "partition": {"sources": a.clients, "replication": a.fqp_repl,
                                         "triples_per_source": [e["fqp"].get("own_triples") for e in evs.values()],
                                         "summary_bytes_per_source": [e["fqp"].get("own_summary_bytes") for e in evs.values()]},
                           "per_client": fq, "transport": "HTTP between client containers"}
        rows = [json.loads(l) for l in open(self.out + ".jsonl")]
        server_cum = [r["resources"]["server"]["wall_s"] for r in rows]
        round_wall_s = [server_cum[0]] + [server_cum[i] - server_cum[i - 1] for i in range(1, len(server_cum))]
        cids_sorted = sorted(rows[0]["resources"]["clients"], key=int) if rows else []
        per_round_client_secs = [[r["resources"]["clients"][c]["train_wall_s"] for c in cids_sorted] for r in rows]
        summ["efficiency"] = F.compute_training_efficiency(
            round_wall_s=round_wall_s,
            per_round_client_secs=per_round_client_secs,
            val_scores=[F.selection_score_from_row(r, self.heads) for r in rows],
            test_scores=[r["test"].get("auroc") for r in rows],
        )
        json.dump(summ, open(self.out + ".summary.json", "w"), indent=1)
        self.done = True
        F.log("summary written:", self.out + ".summary.json")

def make_handler(S):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n) if n else b""
            S.bytes_in += len(data)
            return data

        def _send(self, data, ctype="application/octet-stream", code=200):
            if isinstance(data, (dict, list)):
                data, ctype = json.dumps(data).encode(), "application/json"
            S.bytes_out += len(data)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/health":
                return self._send({"ok": True, "done": S.done})
            if u.path == "/model":
                rnd = int(q["after_round"])
                with S.lock:
                    ok = S.lock.wait_for(lambda: rnd in S.blobs, timeout=60)
                    blob = S.blobs.get(rnd) if ok else None
                return self._send(blob) if blob is not None else self._send({"wait": True}, code=202)
            if u.path == "/directory":
                with S.lock:
                    ok = S.lock.wait_for(lambda: len(S.sources) == S.a.clients, timeout=60)
                    d = dict(S.sources) if ok else None
                return self._send(d) if d is not None else self._send({"wait": True}, code=202)
            if u.path == "/hyp_norm":
                with S.lock:
                    ok = S.lock.wait_for(lambda: S.hyp_norm is not None, timeout=60)
                return self._send(S.hyp_norm) if ok else self._send({"wait": True}, code=202)
            if u.path == "/hyp_model":
                r = int(q["after"])
                with S.lock:
                    ok = S.lock.wait_for(lambda: r in S.hyp_blobs, timeout=60)
                return self._send(S.hyp_blobs[r]) if ok else self._send({"wait": True}, code=202)
            if u.path == "/best":
                with S.lock:
                    ok = S.lock.wait_for(lambda: S.best_blob is not None and S.last is not None
                                         and S.last["round"] == S.a.rounds, timeout=60)
                if not ok:
                    return self._send({"wait": True}, code=202)
                if q.get("blob"):
                    return self._send(S.best_blob)
                return self._send({"round": S.best["round"],
                                   "thr": {h: S.best["heads"][h]["val"].get("thr", 0.5) for h in S.heads}})
            self._send({"error": "not found"}, code=404)

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            body = self._body()
            if u.path == "/register":
                with S.lock:
                    if S.registered >= S.a.clients:
                        return self._send({"error": "all clients registered"}, code=409)
                    cid = S.registered
                    S.registered += 1
                F.log(f"client {cid} registered")
                return self._send({"client_id": cid})
            if u.path == "/source":
                d = json.loads(body)
                with S.lock:
                    S.sources[str(d["client"])] = {"url": d["url"], "bloom": d["bloom"]}
                    S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/kg_lookup":
                codes = json.loads(body)["codes"]
                return self._send({c: S.kg.popular(c)[:S.a.budget] for c in codes})
            if u.path == "/update":
                rnd, cid = int(q["round"]), int(q["client"])
                delta = F.unpack_state(body)
                loss = float(q["loss"]) if q.get("loss") not in (None, "None") else None
                with S.lock:
                    S.updates.setdefault(rnd, {})[cid] = (int(q["n"]), delta, loss)
                    if len(S.updates[rnd]) == S.a.clients:
                        S.aggregate(rnd)
                        S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/eval":
                rnd, cid = int(q["round"]), int(q["client"])
                with S.lock:
                    S.evals.setdefault(rnd, {})[cid] = json.loads(body)
                    if len(S.evals[rnd]) == S.a.clients:
                        S.finish_round(rnd)
                        S.lock.notify_all()
                return self._send({"ok": True, "done": S.done})
            if u.path == "/hyp_stats":
                import hypo_learn as HL
                d = json.loads(body)
                with S.lock:
                    S.hyp_stats[int(q["client"])] = (d["n"], np.array(d["sum"]), np.array(d["sumsq"]), d["pos"])
                    if len(S.hyp_stats) == S.a.clients:
                        st = [v[:3] for v in S.hyp_stats.values() if v[0]]
                        mean, std = HL.fed_norm(st) if st else (None, None)
                        npos = sum(v[3] for v in S.hyp_stats.values())
                        S.hyp_norm = ({"skip": True} if mean is None or npos == 0 else
                                      {"mean": mean.tolist(), "std": std.tolist()})
                        S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/hyp_update":
                import hypo_learn as HL
                r, cid, n = int(q["round"]), int(q["client"]), int(q["n"])
                st = F.unpack_state(body)
                with S.lock:
                    S.hyp_upd.setdefault(r, {})[cid] = (n, st)
                    if len(S.hyp_upd[r]) == S.a.clients:
                        ups = [v for v in S.hyp_upd.pop(r).values() if v[0] > 0]
                        if ups:
                            S.hyp_g = HL.fedavg([s for _, s in ups], [n_ for n_, _ in ups])
                        S.hyp_blobs[r] = F.pack_state(S.hyp_g)
                        S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/hyp":
                cid = int(q["client"])
                with S.lock:
                    S.hyps[cid] = json.loads(body)
                    if len(S.hyps) == S.a.clients:
                        parts = [S.hyps[c]["stats"] for c in sorted(S.hyps)]
                        out = F.merge_hyp_stats(parts)
                        out["head"] = S.hyps[0]["head"]
                        out["per_client"] = parts
                        if S.hyp_norm and not S.hyp_norm.get("skip"):
                            import hypo_learn as HL
                            out["learned"] = {"model": S.a.hyp_model, "rounds": S.a.hyp_rounds,
                                              "weights": HL.weights_report({"kind": S.a.hyp_model, "global": S.hyp_g},
                                                                           S.hyp_names)}
                        ex = [{"client": c, **e} for c in sorted(S.hyps) for e in S.hyps[c]["examples"]]
                        F.write_hyp_examples(os.path.join(task_dir(S.a), "kg"), ex,
                                             S.out + ".hypotheses_examples.json")
                        S.write_summary(out)
                        S.lock.notify_all()
                return self._send({"ok": True})
            self._send({"error": "not found"}, code=404)
    return H

def cmd_server(a):
    import torch
    torch.set_num_threads(a.threads)
    if a.arm not in DOCKER_ARMS or a.fl not in DOCKER_FL:
        sys.exit(f"Docker mode supports arms {DOCKER_ARMS} and FL {DOCKER_FL}; "
                 f"run '{a.arm}' / '{a.fl}' with fedrag_sim.py (scripts/run_sim.*)")
    S = State(a)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(S))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    F.log(f"server up on :{a.port} for {run_tag(a)} (vocab {S.vocab_size}, params {S.n_params})")
    while not S.done:
        time.sleep(1)
    time.sleep(3)
    srv.shutdown()
    F.log("server finished")

def fqp_client_retrieval(a, C, cid, kg, vocab, pts, splits):
    """Federated query processing between client containers.
    1. build ONLY this client's KG partition, 2. serve sub-queries on it,
    3. publish its Bloom summary to the directory, 4. run this client's
    patient queries over the federation (direct HTTP to the other clients)."""
    import fqp as Q
    own = Q.build_sources(kg.kgc, a.clients, a.fqp_repl, a.seed, only=cid)[0][cid]
    port = a.peer_port + (cid if a.peer_port_offset else 0)

    class PeerH(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n))["q"]
            body = json.dumps({"a": own.answer_batch(q)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("0.0.0.0", port), PeerH)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = a.peer_host or socket.gethostbyname(socket.gethostname())
    C.call("POST", "/source", {"client": cid, "url": f"http://{host}:{port}", "bloom": own.bloom.to_json()})
    directory = json.loads(C.call("GET", "/directory"))
    summaries = {int(k): Q.Bloom.from_json(v["bloom"]) for k, v in directory.items()}
    urls = {int(k): v["url"] for k, v in directory.items()}
    F.log(f"client {cid}: own KG {own.n_triples} triples, {len(own.ents)} entities; "
          f"directory of {len(urls)} sources")

    def transport(s, queries):
        if s == cid:
            return own.answer_batch(queries), 0, 0
        req = json.dumps({"q": queries}).encode()
        for i in range(30):
            try:
                r = urllib.request.Request(urls[s] + "/subq", data=req, method="POST",
                                           headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(r, timeout=600) as resp:
                    raw = resp.read()
                return json.loads(raw)["a"], len(req), len(raw)
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(1 + i)
        raise RuntimeError(f"source {s} unreachable")

    eng = Q.FQPEngine(kg, summaries, transport, home=cid, decoys=a.fqp_decoys,
                      hops=a.fqp_hops if a.arm == "fqp" else 1, join_top=a.fqp_join_top,
                      seed=a.seed * 31 + cid, sources_allowed={cid} if a.arm == "client_local" else None)
    data, gstats, fst = F.prepare_split_data_fqp(pts, splits, kg, vocab, a.budget, a.max_codes, eng,
                                                  join_w=a.fqp_join_w)
    fst["own_triples"] = int(own.n_triples)
    fst["own_summary_bytes"] = int(own.bloom.nbytes)
    return data, gstats, fst

Conn = HTTPConnectionModule

def cmd_client(a):
    import torch
    torch.set_num_threads(a.threads)
    C = Conn(a.server)
    cid = json.loads(C.call("POST", "/register", {}))["client_id"]
    random.seed(a.seed * 1000 + cid)
    torch.manual_seed(a.seed * 1000 + cid)
    td = task_dir(a)
    heads = F.TASK_HEADS[a.task]
    shard_path = os.path.join(td, split_tag(a), f"client_{cid}.pkl")
    for _ in range(900):
        if os.path.exists(shard_path):
            break
        time.sleep(1)
    shard = pickle.load(open(shard_path, "rb"))
    vocab = pickle.load(open(os.path.join(td, "vocab.pkl"), "rb"))
    kg = F.KG(pickle.load(open(os.path.join(td, "kg.pkl"), "rb")))
    pts, splits = shard["patients"], shard["splits"]
    if not int(getattr(a, "meas", 1)):
        pts = F.MS.strip(pts)
    users = F.user_patients_for(a.user_data if os.path.isdir(a.user_data) else None, cid, a.task)
    splits["train"] = splits["train"] + list(range(len(pts), len(pts) + len(users)))
    pts = pts + users

    m_ret = F.Meter()
    lookup, df, N = None, None, 1
    fstats = None
    if a.arm in F.FQP_ARMS:
        data, gstats, fstats = fqp_client_retrieval(a, C, cid, kg, vocab, pts, splits)
    elif a.arm == "server":
        codes = sorted({c for p in pts for c in F.flat(p)[:a.max_codes]})
        table = json.loads(C.call("POST", "/kg_lookup", {"codes": codes}))
        lookup = lambda c: table.get(c, [])
    elif a.arm == "client":
        df, N = F.doc_freq(kg, [pts[i] for i in splits["train"]])
    if a.arm not in F.FQP_ARMS:
        data, gstats = F.prepare_split_data(pts, splits, kg, vocab, a.arm, a.budget, a.max_codes,
                                            df, N, random.Random(a.seed * 1000 + cid), lookup)
    ret_res = m_ret.read()
    F.log(f"client {cid}: {len(data['train'])}/{len(data['val'])}/{len(data['test'])} patients "
          f"({len(users)} local user cases), retrieval {ret_res['wall_s']}s, "
          f"concepts/pt {gstats['mean_concepts_added']:.1f}")

    mk = lambda: F.make_model(a.model, len(vocab), a.dim, a.layers, a.dropout, len(heads))
    model, evalm = mk(), mk()
    teacher = mk() if a.fl in ("fedlwf", "parc") else None
    g = F.unpack_state(C.call("GET", "/model?after_round=0"))
    meter = F.Meter()
    edge = EdgeController(cid, tag=run_tag(a))
    yy = lambda split: [list(y) for _, y in data[split]]
    for rnd in range(1, a.rounds + 1):
        edge.start_round(rnd)
        st, loss, fs = F.local_train(model, g, data["train"], a, rnd, cid, a.fl, teacher)
        train_s = edge.end_round(rnd, loss=loss)
        C.call("POST", f"/update?round={rnd}&client={cid}&n={len(data['train'])}&loss={loss}",
               F.pack_state({k: st[k] - g[k] for k in g}, a.comm_dtype))
        g = F.unpack_state(C.call("GET", f"/model?after_round={rnd}"))
        evalm.load_state_dict(g)
        t1 = time.perf_counter()
        pv, pt = F.predict(evalm, data["val"]), F.predict(evalm, data["test"])
        infer_ms = 1000 * (time.perf_counter() - t1) / max(1, len(data["val"]) + len(data["test"]))
        r = meter.read()
        C.call("POST", f"/eval?round={rnd}&client={cid}", {
            "val_y": yy("val"), "val_p": pv.tolist(), "test_y": yy("test"), "test_p": pt.tolist(),
            "graph_stats": gstats, "sizes": {s: len(v) for s, v in data.items()}, "n_user": len(users),
            "fqp": fstats,
            "resources": {"train_wall_s": round(train_s, 3), "fisher_s": fs, "cpu_s": r["cpu_s"],
                          "rss_mb": r["rss_mb"], "peak_rss_mb": r["peak_rss_mb"],
                          "retrieval_wall_s": ret_res["wall_s"], "bytes_sent": C.sent,
                          "bytes_recv": C.recv, "infer_ms_per_patient": round(infer_ms, 4)}})
        F.log(f"client {cid} round {rnd}: loss={loss:.4f} train={train_s:.1f}s rss={r['rss_mb']}MB")

    if a.hypotheses:
        best = json.loads(C.call("GET", "/best"))
        evalm.load_state_dict(F.unpack_state(C.call("GET", "/best?blob=1")))
        head = heads.index("ade") if "ade" in heads else 0
        engine = F.HypothesisEngine(pickle.load(open(os.path.join(td, "kg_hyp.pkl"), "rb")), a.hyp_hops)
        rank_fn = None
        if a.hyp_learn:
            import hypo_learn as HL
            trp = [pts[i] for i in splits["train"]]
            ptr = F.predict(evalm, data["train"]).reshape(len(data["train"]), len(heads))
            X, y = F.hyp_training_data(engine, kg, trp, ptr, head, a.hyp_train_max, seed=a.seed + cid,
                                       n_pop=a.hyp_pop_cands)
            n, s1, s2 = HL.local_stats(X)
            dim = len(HL.feature_names(engine.h["rel_names"]))
            C.call("POST", f"/hyp_stats?client={cid}", {"n": int(n), "pos": float(y.sum()) if len(y) else 0.0,
                   "sum": (s1 if n else np.zeros(dim)).tolist(), "sumsq": (s2 if n else np.zeros(dim)).tolist()})
            norm = json.loads(C.call("GET", "/hyp_norm"))
            if not norm.get("skip"):
                mean, std = np.array(norm["mean"], dtype=np.float32), np.array(norm["std"], dtype=np.float32)
                ranker = HL.make_ranker(dim, a.hyp_model, a.seed)
                for r in range(1, a.hyp_rounds + 1):
                    g = F.unpack_state(C.call("GET", f"/hyp_model?after={r - 1}"))
                    st, _ = HL.local_fit(ranker, g, X, y, mean, std, seed=a.seed + 31 * cid + r, lr=a.hyp_lr)
                    C.call("POST", f"/hyp_update?round={r}&client={cid}&n={len(X)}", F.pack_state(st))
                g = F.unpack_state(C.call("GET", f"/hyp_model?after={a.hyp_rounds}"))
                rank_fn = HL.scorer({"global": g, "mean": mean, "std": std, "kind": a.hyp_model, "dim": dim})
                F.log(f"client {cid}: learned hypothesis ranker trained ({len(X)} local pairs)")
        probs = F.predict(evalm, data["test"]).reshape(len(data["test"]), len(heads))
        stats, examples = F.run_hypotheses(engine, kg, [pts[i] for i in splits["test"]], probs,
                                           best["thr"][heads[head]], [pts[i] for i in splits["train"]],
                                           a.hyp_k, a.hyp_max, head, rank_fn, n_pop=a.hyp_pop_cands)
        C.call("POST", f"/hyp?client={cid}", {"stats": stats, "examples": examples, "head": heads[head]})
        F.log(f"client {cid}: hypotheses for {stats['patients']} high-risk patients")
    F.log(f"client {cid} finished")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("role", choices=["prep", "server", "client"])
    ap.add_argument("--mimic", default=env("MIMIC_DIR", "/mimic"))
    ap.add_argument("--kgdir", default=env("KG_DIR", "/kg"))
    ap.add_argument("--user_data", default=env("USER_DIR", "/user"))
    ap.add_argument("--task", default=env("TASK", "mortality_next"),
                    choices=list(F.TASK_HEADS) + ["notes_diagnosis", "notes_course"])
    ap.add_argument("--work", default=env("WORK_DIR", "/work"))
    ap.add_argument("--out", default=env("OUT_DIR", "/results"))
    ap.add_argument("--server", default=env("SERVER_URL", "http://server:8080"))
    ap.add_argument("--port", type=int, default=int(env("PORT", "8080")))
    ap.add_argument("--arm", default=env("ARM", "client"))
    ap.add_argument("--fl", default=env("FL", "fedavg"))
    ap.add_argument("--model", default=env("MODEL", "sage"), choices=F.MODELS)
    ap.add_argument("--clients", type=int, default=int(env("CLIENTS", "10")))
    ap.add_argument("--seed", type=int, default=int(env("SEED", "1")))
    ap.add_argument("--partition", default=env("PARTITION", "noniid"))
    ap.add_argument("--alpha", type=float, default=float(env("ALPHA", "0.5")))
    ap.add_argument("--rounds", type=int, default=int(env("ROUNDS", "20")))
    ap.add_argument("--local_epochs", type=int, default=int(env("LOCAL_EPOCHS", "1")))
    ap.add_argument("--budget", type=int, default=int(env("BUDGET", "16")))
    ap.add_argument("--task_tag", default=env("TASK_TAG", env("TASK", "mortality_next")))
    ap.add_argument("--threads", type=int, default=int(env("THREADS", "1")))
    ap.add_argument("--comm_dtype", default=env("COMM_DTYPE", "fp32"))
    ap.add_argument("--mu", type=float, default=float(env("MU", "0.01")))
    ap.add_argument("--lam_ewc", type=float, default=float(env("LAM_EWC", "0.5")))
    ap.add_argument("--lam_kd", type=float, default=float(env("LAM_KD", "0.5")))
    ap.add_argument("--kd_temp", type=float, default=float(env("KD_TEMP", "2.0")))
    ap.add_argument("--fisher_samples", type=int, default=int(env("FISHER_SAMPLES", "200")))
    ap.add_argument("--ade_weight", type=float, default=float(env("ADE_WEIGHT", "1.0")))
    ap.add_argument("--batch", type=int, default=int(env("BATCH", "64")))
    ap.add_argument("--lr", type=float, default=float(env("LR", "1e-3")))
    ap.add_argument("--wd", type=float, default=float(env("WD", "1e-5")))
    ap.add_argument("--dim", type=int, default=int(env("DIM", "64")))
    ap.add_argument("--layers", type=int, default=int(env("LAYERS", "2")))
    ap.add_argument("--dropout", type=float, default=float(env("DROPOUT", "0.2")))
    ap.add_argument("--max_codes", type=int, default=int(env("MAX_CODES", "64")))
    ap.add_argument("--hypotheses", action="store_true", default=env("HYPOTHESES", "0") in ("1", "true", "yes"))
    ap.add_argument("--fqp_repl", type=int, default=int(env("FQP_REPL", "2")))
    ap.add_argument("--fqp_hops", type=int, default=int(env("FQP_HOPS", "2")))
    ap.add_argument("--fqp_decoys", type=int, default=int(env("FQP_DECOYS", "0")))
    ap.add_argument("--fqp_join_top", type=int, default=int(env("FQP_JOIN_TOP", "20")))
    ap.add_argument("--fqp_join_w", type=float, default=float(env("FQP_JOIN_W", "0.5")))
    ap.add_argument("--meas", type=int, default=int(env("MEAS", "1")), help="0 = ablation without lab / vital findings")
    ap.add_argument("--meas_frac", type=float, default=float(env("MEAS_FRAC", "0.3")))
    ap.add_argument("--meas_kinds", default=env("MEAS_KINDS", "labs,vitals"))
    ap.add_argument("--peer_port", type=int, default=int(env("PEER_PORT", "8090")))
    ap.add_argument("--peer_port_offset", action="store_true", default=env("PEER_PORT_OFFSET", "0") == "1",
                    help="local testing without Docker: port = PEER_PORT + client id")
    ap.add_argument("--peer_host", default=env("PEER_HOST", ""))

    ap.add_argument("--tg_model", default=env("TG_MODEL", "google/flan-t5-small"))
    ap.add_argument("--hide_dx", type=int, default=int(env("HIDE_DX", "0")))
    ap.add_argument("--lora_r", type=int, default=int(env("LORA_R", "8")))
    ap.add_argument("--tg_lr", type=float, default=float(env("TG_LR", "5e-4")))
    ap.add_argument("--tg_batch", type=int, default=int(env("TG_BATCH", "8")))
    ap.add_argument("--tg_gen_batch", type=int, default=int(env("TG_GEN_BATCH", "16")))
    ap.add_argument("--max_src", type=int, default=int(env("MAX_SRC", "384")))
    ap.add_argument("--max_tgt", type=int, default=int(env("MAX_TGT", "96")))
    ap.add_argument("--num_beams", type=int, default=int(env("NUM_BEAMS", "1")))
    ap.add_argument("--tg_max_train", type=int, default=int(env("TG_MAX_TRAIN", "400")))
    ap.add_argument("--tg_max_val", type=int, default=int(env("TG_MAX_VAL", "30")))
    ap.add_argument("--tg_max_test", type=int, default=int(env("TG_MAX_TEST", "100")))
    ap.add_argument("--max_items", type=int, default=int(env("MAX_ITEMS", "0")))
    ap.add_argument("--hyp_k", type=int, default=int(env("HYP_K", "5")))
    ap.add_argument("--hyp_max", type=int, default=int(env("HYP_MAX", "100")))
    ap.add_argument("--hyp_hops", type=int, default=int(env("HYP_HOPS", "3")))
    ap.add_argument("--hyp_learn", type=int, default=int(env("HYP_LEARN", "1")))
    ap.add_argument("--hyp_model", default=env("HYP_MODEL", "linear"))
    ap.add_argument("--hyp_rounds", type=int, default=int(env("HYP_ROUNDS", "20")))
    ap.add_argument("--hyp_lr", type=float, default=float(env("HYP_LR", "0.01")))
    ap.add_argument("--hyp_train_max", type=int, default=int(env("HYP_TRAIN_MAX", "300")))
    ap.add_argument("--hyp_pop_cands", type=int, default=int(env("HYP_POP_CANDS", "20")))
    a = ap.parse_args()
    if a.task.startswith("notes"):
        import textnet
        {"prep": textnet.cmd_prep, "server": textnet.cmd_server, "client": textnet.cmd_client}[a.role](a)
    else:
        {"prep": cmd_prep, "server": cmd_server, "client": cmd_client}[a.role](a)

if __name__ == "__main__":
    main()
