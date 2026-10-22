"""
textnet.py -- federated note generation as separate services (Docker)
=====================================================================
TASK=notes_diagnosis | notes_course

  prep    NOTEEVENTS -> text cohort, KG, client shards; downloads the
          generator once into /work/hf (clients then run offline)
  server  aggregates LoRA adapters (FedAvg / FedProx), keeps the best
          adapters (validation ROUGE-L), merges client metrics, writes results
  client  retrieval (any Docker arm, incl. fqp between client containers),
          LoRA training, generation and metric computation ON THE CLIENT;
          only adapter deltas and metric numbers leave the client
          (+ 2 example notes per client for the results file).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import fedrag_sim as F
import fl_net as N
import textgen as TG

def tdir(a):
    return os.path.join(a.work, a.task + ("_hidedx" if a.hide_dx else ""))

def ttag(a):
    sfx = F.fqp_suffix(a.arm, a.fqp_hops, a.fqp_decoys, a.fqp_repl) + ("_hidedx" if a.hide_dx else "") + F.meas_suffix(a)
    return f"{a.task_tag}__{a.fl}__t5lora__{a.arm}{sfx}__c{a.clients}__{a.partition}__seed{a.seed}"

def hf_env(a, offline):
    os.environ["HF_HOME"] = os.path.join(a.work, "hf")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

def cmd_prep(a):
    m = F.Meter()
    d = os.path.join(a.work, a.task)
    os.makedirs(d, exist_ok=True)
    cohort, kgp = os.path.join(d, "cohort.pkl"), os.path.join(d, "kg.pkl")
    if not os.path.exists(cohort):
        if F.find_table(a.mimic, "NOTEEVENTS.csv") is None:
            F.log("PREP CANNOT START: NOTEEVENTS.csv (or .csv.gz) is not in the MIMIC folder "
                  "(needed for note generation)")
            sys.exit(1)
        TG.cmd_notes(argparse.Namespace(mimic=a.mimic, section=a.task.split("_", 1)[1], max_target_chars=400,
                                        max_codes_per_visit=128, max_items=a.max_items,
                                        measurements=N.measurements_path(a), out=cohort))
    if not os.path.exists(kgp):
        F.cmd_kg(argparse.Namespace(cohort=cohort, kgdir=a.kgdir, user_data=None, out=kgp))
    hf_env(a, offline=False)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    AutoTokenizer.from_pretrained(a.tg_model)
    AutoModelForSeq2SeqLM.from_pretrained(a.tg_model)
    sd = os.path.join(d, N.split_tag(a))
    if not os.path.exists(os.path.join(sd, "DONE")):
        os.makedirs(sd, exist_ok=True)
        patients = pickle.load(open(cohort, "rb"))["patients"]
        for ci, c in enumerate(F.make_clients(patients, a.clients, a.partition, a.alpha, a.seed)):
            c = {"train": c["train"][:a.tg_max_train], "val": c["val"][:a.tg_max_val], "test": c["test"][:a.tg_max_test]}
            ids = sorted({i for s in c.values() for i in s})
            remap = {i: j for j, i in enumerate(ids)}
            pickle.dump({"patients": [patients[i] for i in ids],
                         "splits": {s: [remap[i] for i in v] for s, v in c.items()}},
                        open(os.path.join(sd, f"client_{ci}.pkl"), "wb"), protocol=4)
        open(os.path.join(sd, "DONE"), "w").write(json.dumps(m.read()))
    F.log("text prep done", m.read())

class S_:
    pass

def cmd_server(a):
    import torch
    torch.set_num_threads(a.threads)
    if a.arm not in N.DOCKER_ARMS or a.fl not in ("fedavg", "fedprox"):
        sys.exit("Docker text mode supports arms " + str(N.DOCKER_ARMS) + " and FL fedavg / fedprox")
    hf_env(a, offline=True)
    S = S_()
    S.lock = threading.Condition()
    _, model = TG.load_generator(a)
    S.g = TG.lora_state(model)
    S.n_train = sum(v.numel() for v in S.g.values())
    del model
    S.blobs = {0: F.pack_state(S.g, a.comm_dtype)}
    S.registered, S.updates, S.evals, S.tests, S.sources = 0, {}, {}, {}, {}
    S.best, S.best_blob, S.done, S.t0, S.bytes_in, S.bytes_out = None, None, False, time.time(), 0, 0
    S.kg = F.KG(pickle.load(open(os.path.join(a.work, a.task, "kg.pkl"), "rb"))) if a.arm == "server" else None
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, ttag(a))
    open(out + ".jsonl", "w").close()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, data, code=200):
            if isinstance(data, (dict, list)):
                data = json.dumps(data).encode()
            S.bytes_out += len(data)
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            if u.path == "/model":
                r = int(q["after_round"])
                with S.lock:
                    ok = S.lock.wait_for(lambda: r in S.blobs, timeout=60)
                return self._send(S.blobs[r]) if ok else self._send({"wait": True}, 202)
            if u.path == "/best":
                with S.lock:
                    ok = S.lock.wait_for(lambda: S.best_blob is not None and len(S.evals.get(a.rounds, {})) == 0
                                         and S.last_round == a.rounds, timeout=60)
                return self._send(S.best_blob) if ok else self._send({"wait": True}, 202)
            if u.path == "/directory":
                with S.lock:
                    ok = S.lock.wait_for(lambda: len(S.sources) == a.clients, timeout=60)
                return self._send(dict(S.sources)) if ok else self._send({"wait": True}, 202)
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            q = dict(urllib.parse.parse_qsl(u.query))
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n) if n else b""
            S.bytes_in += len(body)
            if u.path == "/register":
                with S.lock:
                    cid = S.registered
                    S.registered += 1
                return self._send({"client_id": cid})
            if u.path == "/source":
                d = json.loads(body)
                with S.lock:
                    S.sources[str(d["client"])] = {"url": d["url"], "bloom": d["bloom"]}
                    S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/kg_lookup":
                codes = json.loads(body)["codes"]
                return self._send({c: S.kg.popular(c)[:a.budget] for c in codes})
            if u.path == "/update":
                r, cid = int(q["round"]), int(q["client"])
                d = F.unpack_state(body)
                with S.lock:
                    S.updates.setdefault(r, {})[cid] = (int(q["n"]), d, float(q["loss"]))
                    if len(S.updates[r]) == a.clients:
                        ups = S.updates.pop(r)
                        W = float(sum(x[0] for x in ups.values()))
                        S.g = {k: S.g[k] + sum(x[1][k] * (x[0] / W) for x in ups.values()) for k in S.g}
                        S.blobs[r] = F.pack_state(S.g, a.comm_dtype)
                        S.blobs.pop(r - 2, None)
                        S.loss = float(np.mean([x[2] for x in ups.values()]))
                        S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/eval":
                r, cid = int(q["round"]), int(q["client"])
                with S.lock:
                    S.evals.setdefault(r, {})[cid] = json.loads(body)
                    if len(S.evals[r]) == a.clients:
                        ev = S.evals.pop(r)
                        val = TG.merge_text_metrics([e["val"] for e in ev.values()])
                        rec = {"round": r, "train_loss": S.loss, "val": val,
                               "resources": {"clients": {c: e["resources"] for c, e in ev.items()},
                                             "server_bytes_in": S.bytes_in, "server_bytes_out": S.bytes_out}}
                        with open(out + ".jsonl", "a") as fo:
                            fo.write(json.dumps(rec) + "\n")
                        F.log(f"round {r}: val rougeL={val.get('rougeL', 0):.3f} conceptF1={val.get('concept_f1', 0):.3f}")
                        if S.best is None or val.get("rougeL", 0) > S.best["val"].get("rougeL", 0):
                            S.best, S.best_blob = rec, S.blobs[r]
                        S.last_round, S.last_ev = r, ev
                        S.lock.notify_all()
                return self._send({"ok": True})
            if u.path == "/test":
                cid = int(q["client"])
                with S.lock:
                    S.tests[cid] = json.loads(body)
                    if len(S.tests) == a.clients:
                        T = [S.tests[c] for c in sorted(S.tests)]
                        test = TG.merge_text_metrics([t["test"] for t in T])
                        fq = [e.get("fqp") for e in S.last_ev.values() if e.get("fqp")]
                        cl = [t["resources"] for t in T]
                        summ = {"tag": ttag(a), "task": a.task, "kind": "text", "fl": a.fl,
                                "model": "t5-full" if a.lora_r <= 0 else "t5-lora", "arm": a.arm, "meas": int(getattr(a, "meas", 1)),
                                "clients": a.clients, "seed": a.seed, "partition": a.partition,
                                "hide_dx": int(a.hide_dx), "budget": a.budget, "rounds": a.rounds, "mode": "docker",
                                "best_round": S.best["round"], "val_at_best": S.best["val"], "test_text": test,
                                "fqp": ({**F.merge_fqp_stats(fq), "per_client": fq} if fq else None),
                                "generator": a.tg_model,
                                "resources": {"total": {"wall_s": round(time.time() - S.t0, 1),
                                                        "peak_rss_mb": max(c["peak_rss_mb"] for c in cl),
                                                        "client_peak_rss_mb": [c["peak_rss_mb"] for c in cl]},
                                              "trainable_params": int(S.n_train),
                                              "lora_mb": round(S.n_train * 4 / 2**20, 3),
                                              "total_mb_up": round(S.bytes_in / 2**20, 2),
                                              "total_mb_down": round(S.bytes_out / 2**20, 2),
                                              "gen_ms_per_item": float(np.mean([c["gen_ms_per_item"] for c in cl])),
                                              "threads": a.threads}}
                        json.dump(summ, open(out + ".summary.json", "w"), indent=1)
                        json.dump([e for t in T for e in t["examples"]], open(out + ".examples.json", "w"), indent=1)
                        F.log("summary written:", out + ".summary.json")
                        S.done = True
                return self._send({"ok": True})
            self._send({"error": "not found"}, 404)

    S.last_round = 0
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    F.log(f"text server up for {ttag(a)} (LoRA params {S.n_train})")
    while not S.done:
        time.sleep(1)
    time.sleep(3)
    srv.shutdown()

def cmd_client(a):
    import torch
    torch.set_num_threads(a.threads)
    hf_env(a, offline=True)
    C = N.Conn(a.server)
    cid = json.loads(C.call("POST", "/register", {}))["client_id"]
    d = os.path.join(a.work, a.task)
    sp = os.path.join(d, N.split_tag(a), f"client_{cid}.pkl")
    for _ in range(900):
        if os.path.exists(sp):
            break
        time.sleep(1)
    shard = pickle.load(open(sp, "rb"))
    kg = F.KG(pickle.load(open(os.path.join(d, "kg.pkl"), "rb")))
    names = pickle.load(open(os.path.join(d, "kg_names.pkl"), "rb"))
    cnames = TG.load_code_names(a.kgdir)
    matcher = TG.ConceptMatcher(names, a.kgdir)
    pts, splits = shard["patients"], shard["splits"]
    if not int(getattr(a, "meas", 1)):
        pts = F.MS.strip(pts)

    def q(p):
        return [c for c in p["visits"][0] if not (a.hide_dx and c.startswith("D_"))][:a.max_codes]
    m_ret = F.Meter()
    fst = None
    ret = {}
    qp = [{"visits": [q(p)], "y": 0} for p in pts]
    if a.arm in F.FQP_ARMS:
        import fqp as Q
        own = Q.build_sources(kg.kgc, a.clients, a.fqp_repl, a.seed, only=cid)[0][cid]
        eng = _fqp_engine(a, C, cid, kg, own)
        order = [(s, i) for s, ids in splits.items() for i in ids]
        covs, twos, fst = eng.coverage_batch([q(pts[i]) for _, i in order])
        df, ntr = F.Counter(), 0
        for (s, _), cv in zip(order, covs):
            if s == "train":
                df.update(cv.keys())
                ntr += 1
        for (s, i), cv, tw in zip(order, covs, twos):
            ret.setdefault(s, []).append(Q.rank_from_cov(cv, tw, df, ntr, a.budget, join_w=a.fqp_join_w, deg=kg.deg))
        fst["own_triples"] = int(own.n_triples)
        fst["completeness_vs_central_kg"] = float(np.mean(
            [x for x in (Q.completeness(kg, q(pts[i]), cv) for (_, i), cv in zip(order, covs)) if x is not None] or [0]))
    else:
        lookup, df, Nd = None, None, 1
        if a.arm == "server":
            codes = sorted({c for p in qp for c in F.flat(p)})
            table = json.loads(C.call("POST", "/kg_lookup", {"codes": codes}))
            lookup = lambda c: table.get(c, [])
        elif a.arm == "client":
            df, Nd = F.doc_freq(kg, [qp[i] for i in splits["train"]])
        rng = random.Random(a.seed * 1000 + cid)
        for s, ids in splits.items():
            ret[s] = [F.rank(a.arm, kg, F.flat(qp[i]), a.budget, df, Nd, rng, lookup) for i in ids]
    rows = {s: TG.build_rows([pts[i] for i in ids], ret.get(s, [[]] * len(ids)), kg, cnames, names, matcher, a.hide_dx)
            for s, ids in splits.items()}
    ret_res = m_ret.read()
    F.log(f"client {cid}: {len(rows['train'])}/{len(rows['val'])}/{len(rows['test'])} notes, retrieval {ret_res['wall_s']}s")

    tok, model = TG.load_generator(a)
    g = F.unpack_state(C.call("GET", "/model?after_round=0"))
    meter = F.Meter()
    for r in range(1, a.rounds + 1):
        t0 = time.time()
        st, loss = TG.train_local_text(model, tok, g, [(x[0], x[1]) for x in rows["train"]], a, r, cid, a.fl)
        tr_s = time.time() - t0
        C.call("POST", f"/update?round={r}&client={cid}&n={len(rows['train'])}&loss={loss}",
               F.pack_state({k: st[k] - g[k] for k in g}, a.comm_dtype))
        g = F.unpack_state(C.call("GET", f"/model?after_round={r}"))
        TG.load_lora(model, g)
        t1 = time.time()
        gens = TG.generate(model, tok, [x[0] for x in rows["val"]], a)
        val = TG.text_metrics(gens, [x[1] for x in rows["val"]], [x[2] for x in rows["val"]], matcher)
        rs = meter.read()
        C.call("POST", f"/eval?round={r}&client={cid}", {
            "val": val, "fqp": fst,
            "resources": {"train_wall_s": round(tr_s, 2), "gen_s": round(time.time() - t1, 2),
                          "rss_mb": rs["rss_mb"], "peak_rss_mb": rs["peak_rss_mb"], "cpu_s": rs["cpu_s"],
                          "retrieval_wall_s": ret_res["wall_s"], "bytes_sent": C.sent, "bytes_recv": C.recv}})
        F.log(f"client {cid} round {r}: loss={loss:.3f} rougeL={val.get('rougeL', 0):.3f} rss={rs['rss_mb']}MB")
    TG.load_lora(model, F.unpack_state(C.call("GET", "/best")))
    t1 = time.time()
    gens = TG.generate(model, tok, [x[0] for x in rows["test"]], a)
    gms = 1000 * (time.time() - t1) / max(1, len(gens))
    test = TG.text_metrics(gens, [x[1] for x in rows["test"]], [x[2] for x in rows["test"]], matcher)
    ex = [{"client": cid, "input": x[0][:600], "generated": gtxt, "reference": x[1]}
          for x, gtxt in list(zip(rows["test"], gens))[:2]]
    rs = meter.read()
    C.call("POST", f"/test?client={cid}", {"test": test, "examples": ex,
                                           "resources": {"gen_ms_per_item": gms, "peak_rss_mb": rs["peak_rss_mb"]}})
    F.log(f"client {cid} finished: test rougeL={test.get('rougeL', 0):.3f}")

def _fqp_engine(a, C, cid, kg, own):
    """same peer-to-peer federated query processing as fl_net.fqp_client_retrieval"""
    import fqp as Q
    import socket
    import urllib.request
    port = a.peer_port + (cid if a.peer_port_offset else 0)

    class PeerH(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.dumps({"a": own.answer_batch(json.loads(self.rfile.read(n))["q"])}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("0.0.0.0", port), PeerH)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = a.peer_host or socket.gethostbyname(socket.gethostname())
    C.call("POST", "/source", {"client": cid, "url": f"http://{host}:{port}", "bloom": own.bloom.to_json()})
    directory = json.loads(C.call("GET", "/directory"))
    summ = {int(k): Q.Bloom.from_json(v["bloom"]) for k, v in directory.items()}
    urls = {int(k): v["url"] for k, v in directory.items()}

    def transport(s, queries):
        if s == cid:
            return own.answer_batch(queries), 0, 0
        req = json.dumps({"q": queries}).encode()
        for i in range(30):
            try:
                rq = urllib.request.Request(urls[s] + "/subq", data=req, method="POST")
                with urllib.request.urlopen(rq, timeout=600) as resp:
                    raw = resp.read()
                return json.loads(raw)["a"], len(req), len(raw)
            except Exception:
                time.sleep(1 + i)
        raise RuntimeError(f"source {s} unreachable")
    return Q.FQPEngine(kg, summ, transport, home=cid, decoys=a.fqp_decoys,
                       hops=a.fqp_hops if a.arm == "fqp" else 1, join_top=a.fqp_join_top,
                       seed=a.seed * 31 + cid, sources_allowed={cid} if a.arm == "client_local" else None)
