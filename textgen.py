"""
textgen.py -- federated, retrieval-augmented clinical TEXT generation on MIMIC-III
=================================================================================

Task: for each hospital admission, generate the "Discharge Diagnosis" section
(or the "Brief Hospital Course", --section course) of the real MIMIC-III
discharge summary from

  * the admission's coded record, written out as text (ICD-9 diagnosis and
    procedure names, drug names), and
  * KG context retrieved by the chosen arm (none / random / server / client /
    client_local / fqp / client_kgc / server_ctx) -- the same retrievers,
    including federated query processing, as the prediction task.

--hide_dx 1 removes the diagnosis codes from both the text input and the
retrieval query: the model then has to infer the diagnoses from procedures,
drugs and retrieved knowledge, so retrieval matters much more.

Generator: Flan-T5-small (77M) with LoRA adapters; only the LoRA weights are
trained and exchanged (FedAvg / FedProx), so it runs on a CPU.

Metrics (all offline, no external LLM judge):
  ROUGE-1/2/L, BLEU, and UMLS concept-level, Ragas-style grounding metrics:
  concept precision / recall / F1 (generated vs reference concepts),
  faithfulness (generated concepts supported by the input context),
  hallucination rate (generated concepts in neither context nor reference),
  context recall (reference concepts present in the context = retrieval
  quality for generation). Optional BERTScore (--bertscore).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

import fedrag_sim as F

SECTIONS = {
    "diagnosis": r"discharge\s+diagnos[ie]s\s*:?(.*?)(?=\n\s*(?:discharge\s+condition|condition\s+at\s+discharge|"
                 r"discharge\s+instructions|followup|follow-up|discharge\s+status|major\s+surgical|$))",
    "course": r"brief\s+hospital\s+course\s*:?(.*?)(?=\n\s*(?:medications\s+on\s+admission|discharge\s+medications|"
              r"discharge\s+disposition|discharge\s+diagnos|$))",
}

def clean_note(t, max_chars):
    t = re.sub(r"\[\*\*[^\]]*\*\*\]", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    t = re.sub(r"\s*\n\s*", "; ", t)
    t = re.sub(r"^[;:\s\-]+", "", t)
    return t[:max_chars].strip(" ;")

def cmd_notes(a):
    import pandas as pd
    m = F.Meter()
    path = F.find_table(a.mimic, "NOTEEVENTS.csv")
    if path is None:
        raise SystemExit(f"ERROR: NOTEEVENTS.csv (or .csv.gz) not found in {a.mimic}")
    rx = re.compile(SECTIONS[a.section], re.I | re.S)
    targets = {}
    head = pd.read_csv(path, nrows=0)
    cols = {c.upper(): c for c in head.columns}
    use = [cols[c] for c in ("SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT") if c in cols]
    for chunk in pd.read_csv(path, usecols=use, chunksize=20000, low_memory=False):
        chunk.columns = [c.upper() for c in chunk.columns]
        chunk = chunk[chunk["CATEGORY"].astype(str).str.strip().str.lower() == "discharge summary"]
        for sid, h, txt in zip(chunk["SUBJECT_ID"], chunk["HADM_ID"], chunk["TEXT"]):
            if pd.isna(h) or not isinstance(txt, str):
                continue
            h = int(h)
            if h in targets:
                continue
            mt = rx.search(txt)
            if mt:
                tgt = clean_note(mt.group(1), a.max_target_chars)
                if len(tgt) >= 5:
                    targets[h] = (int(sid), tgt)
    F.log(f"{len(targets)} admissions with a '{a.section}' section")
    diag = F._read(a.mimic, "DIAGNOSES_ICD.csv")
    proc = F._read(a.mimic, "PROCEDURES_ICD.csv")
    pres = F._read(a.mimic, "PRESCRIPTIONS.csv")
    codes = defaultdict(list)
    for h, c in zip(diag["HADM_ID"], diag["ICD9_CODE"]):
        if h in targets and isinstance(c, str):
            codes[h].append("D_" + F.diag_to_dotted(c))
    for h, c in zip(proc["HADM_ID"], proc["ICD9_CODE"]):
        if h in targets and pd.notna(c):
            codes[h].append("P_" + F.proc_to_dotted(c))
    for h, d in zip(pres["HADM_ID"], pres["DRUG"]):
        if pd.notna(h) and int(h) in targets and isinstance(d, str):
            codes[int(h)].append("M_" + d.strip().lower())
    import measurements as MS
    meas = MS.load(getattr(a, "measurements", None))
    for h in targets:
        if h in meas:
            codes[h].extend(meas[h])
    items = []
    for h, (sid, tgt) in targets.items():
        cs = F._dedup(codes.get(h, []))[:a.max_codes_per_visit]
        if cs:
            items.append({"id": sid, "hadm": h, "visits": [cs], "y": 0, "next": [], "target": tgt})
    rng = random.Random(0)
    rng.shuffle(items)
    if a.max_items:
        items = items[:a.max_items]
    stats = {"task": f"notes_{a.section}", "items": len(items),
             "mean_target_words": float(np.mean([len(i["target"].split()) for i in items])) if items else 0,
             "mean_codes": float(np.mean([len(i["visits"][0]) for i in items])) if items else 0,
             "resources": m.read()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"task": f"notes_{a.section}", "patients": items, "stats": stats}, f, protocol=4)
    json.dump(stats, open(os.path.splitext(a.out)[0] + ".stats.json", "w"), indent=1)
    F.log("notes cohort:", json.dumps(stats))

def load_code_names(kgdir):
    names = {}
    for fn, pre in (("ICD9CM.csv", "D_"), ("ICD9PROC.csv", "P_")):
        p = F.kg_file(kgdir, fn)
        if p:
            with open(p, newline="", encoding="utf-8", errors="ignore") as f:
                for r in csv.DictReader(f):
                    names[pre + r["code"].strip()] = r["name"].strip()
    return names

def code_text(code, cnames):
    if code.startswith("M_"):
        return code[2:]
    if code[:2] in ("L_", "V_"):
        import measurements as MS
        return MS.finding_text(code)
    return cnames.get(code) or cnames.get(code.rstrip("0").rstrip(".")) or code[2:]

def make_source(item, retrieved, cnames, names, hide_dx):
    cs = item["visits"][0]
    dx = [code_text(c, cnames) for c in cs if c.startswith("D_")] if not hide_dx else []
    px = [code_text(c, cnames) for c in cs if c.startswith("P_")]
    mx = [code_text(c, cnames) for c in cs if c.startswith("M_")]
    lx = [code_text(c, cnames) for c in cs if c.startswith("L_")]
    vx = [code_text(c, cnames) for c in cs if c.startswith("V_")]
    kn = [names.get(r, r) for r in retrieved]
    parts = ["Write the discharge diagnoses."]
    if dx:
        parts.append("Coded diagnoses: " + "; ".join(dx[:25]) + ".")
    if px:
        parts.append("Procedures: " + "; ".join(px[:15]) + ".")
    if mx:
        parts.append("Medications: " + "; ".join(mx[:25]) + ".")
    if lx:
        parts.append("Abnormal laboratory results: " + "; ".join(lx[:15]) + ".")
    if vx:
        parts.append("Abnormal vital signs: " + "; ".join(vx[:8]) + ".")
    if kn:
        parts.append("Related medical knowledge: " + "; ".join(kn) + ".")
    return " ".join(parts)

_STOP = {"disease", "disorder", "pain", "history", "patient", "status", "other", "unspecified", "acute",
         "chronic", "left", "right", "normal", "stable", "secondary", "primary", "with", "without",
         "discharge", "diagnosis", "diagnoses", "admission", "hospital", "history of", "status post",
         "procedure", "medication", "treatment", "finding", "findings"}

class ConceptMatcher:
    """Dictionary n-gram matcher: text -> UMLS CUIs (names of the KG concepts
    around the cohort + ICD-9 names mapped to UMLS)."""

    def __init__(self, names, kgdir, max_n=6):
        self.d = {}
        for cui, nm in names.items():
            k = self.norm(nm)
            if len(k) >= 4 and k not in _STOP:
                self.d.setdefault(k, cui)
        p = F.kg_file(kgdir, "ICD9CM_to_UMLS.csv")
        icd2cui = {}
        if p:
            with open(p) as f:
                for r in csv.DictReader(f):
                    icd2cui[r["ICD9CM"].strip()] = r["UMLS"].strip()
        for code, nm in load_code_names(kgdir).items():
            cui = icd2cui.get(code[2:])
            k = self.norm(nm)
            if cui and len(k) >= 4:
                self.d.setdefault(k, cui)
        self.max_n = max_n

    @staticmethod
    def norm(s):
        return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))

    def __call__(self, text):
        toks = self.norm(text).split()
        out, i = set(), 0
        while i < len(toks):
            hit = 0
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                cui = self.d.get(" ".join(toks[i:i + n]))
                if cui:
                    out.add(cui)
                    hit = n
                    break
            i += max(hit, 1)
        return out

def text_metrics(gens, refs, ctx_sets, matcher, bertscore=False):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    m = defaultdict(list)
    for g, r, ctx in zip(gens, refs, ctx_sets):
        s = sc.score(r, g)
        for k in ("rouge1", "rouge2", "rougeL"):
            m[k].append(s[k].fmeasure)
        G, R = matcher(g), matcher(r)
        if G:
            m["concept_precision"].append(len(G & R) / len(G))
            m["faithfulness"].append(len(G & ctx) / len(G))
            m["hallucination"].append(len(G - ctx - R) / len(G))
        if R:
            m["concept_recall"].append(len(G & R) / len(R))
            m["context_recall"].append(len(R & ctx) / len(R))
        m["gen_words"].append(len(g.split()))
        m["empty"].append(float(not g.strip()))
    out = {k: float(np.mean(v)) for k, v in m.items() if v}
    p, rc = out.get("concept_precision", 0), out.get("concept_recall", 0)
    out["concept_f1"] = 2 * p * rc / (p + rc) if p + rc else 0.0
    try:
        import sacrebleu
        out["bleu"] = float(sacrebleu.corpus_bleu(gens, [refs]).score)
    except Exception:
        pass
    if bertscore:
        try:
            from bert_score import score as bs
            _, _, f1 = bs(gens, refs, lang="en", verbose=False)
            out["bertscore_f1"] = float(f1.mean())
        except Exception as e:
            F.log("BERTScore skipped:", e)
    out["n"] = len(gens)
    return out

def load_generator(a):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(a.tg_model)
    base = AutoModelForSeq2SeqLM.from_pretrained(a.tg_model)
    if a.lora_r <= 0:
        torch.manual_seed(a.seed)
        return tok, base
    targets = [n for n in ("q", "v") if any(x.endswith("." + n) for x, _ in base.named_modules())] or None
    cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                     target_modules=targets, task_type="SEQ_2_SEQ_LM")
    model = get_peft_model(base, cfg)
    torch.manual_seed(a.seed)
    return tok, model

def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

def load_lora(model, st):
    import torch
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in st:
                p.copy_(st[n])

def encode(tok, srcs, tgts, a):
    enc = tok(srcs, max_length=a.max_src, truncation=True, padding=True, return_tensors="pt")
    enc.pop("token_type_ids", None)
    if tgts is not None:
        lab = tok(tgts, max_length=a.max_tgt, truncation=True, padding=True, return_tensors="pt")["input_ids"]
        lab[lab == tok.pad_token_id] = -100
        enc["labels"] = lab
    return enc

def train_local_text(model, tok, g, items, a, rnd, ci, method):
    import torch
    load_lora(model, g)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.tg_lr)
    ref = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    rng = random.Random(a.seed * 7 + rnd * 13 + ci)
    tr = items[:]
    rng.shuffle(tr)
    losses = []
    for _ in range(a.local_epochs):
        for i in range(0, len(tr), a.tg_batch):
            b = tr[i:i + a.tg_batch]
            enc = encode(tok, [x[0] for x in b], [x[1] for x in b], a)
            loss = model(**enc).loss
            if method == "fedprox" and a.mu > 0:
                loss = loss + 0.5 * a.mu * sum(((p - ref[n]) ** 2).sum()
                                               for n, p in model.named_parameters() if p.requires_grad)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    return lora_state(model), (float(np.mean(losses)) if losses else None)

def generate(model, tok, srcs, a):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(srcs), a.tg_gen_batch):
            enc = encode(tok, srcs[i:i + a.tg_gen_batch], None, a)
            ids = model.generate(**enc, max_new_tokens=a.max_tgt, num_beams=a.num_beams, do_sample=False)
            outs += tok.batch_decode(ids, skip_special_tokens=True)
    return outs

def retrieve_all(a, kg, patients, clients, hide_dx):
    """-> per client {split: [retrieved CUIs per item]}, link-free (text only), fqp stats"""
    def q(p):
        cs = p["visits"][0]
        return [c for c in cs if not (hide_dx and c.startswith("D_"))][:a.max_codes]
    out, fq_parts = [], []
    gdf = gN = None
    if a.arm == "server_ctx":
        gdf, gN = F.doc_freq(kg, [{"visits": [q(patients[i])]} for c in clients for i in c["train"]])
    if a.arm == "client_kgc":
        kg.kgc_model, _ = F.train_kgc(kg, patients, clients, max(a.kgc_rounds, 10), a.seed)
    if a.arm in F.FQP_ARMS:
        import fqp as Q
        sources, summaries, part = F.fqp_setup(kg, a.clients, a)
    for ci, c in enumerate(clients):
        res = {}
        if a.arm in F.FQP_ARMS:
            eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci, decoys=a.fqp_decoys,
                              hops=a.fqp_hops if a.arm == "fqp" else 1, join_top=a.fqp_join_top,
                              seed=a.seed * 31 + ci, sources_allowed={ci} if a.arm == "client_local" else None)
            order = [(s, i) for s, ids in c.items() for i in ids]
            covs, twos, st = eng.coverage_batch([q(patients[i]) for _, i in order])
            df = Counter()
            ntr = 0
            for (s, _), cv in zip(order, covs):
                if s == "train":
                    df.update(cv.keys())
                    ntr += 1
            for (s, i), cv, tw in zip(order, covs, twos):
                res.setdefault(s, []).append(Q.rank_from_cov(cv, tw, df, ntr, a.budget, join_w=a.fqp_join_w, deg=kg.deg))
            st["completeness_vs_central_kg"] = float(np.mean(
                [x for x in (Q.completeness(kg, q(patients[i]), cv) for (_, i), cv in zip(order, covs)) if x is not None] or [0]))
            fq_parts.append(st)
        else:
            if a.arm in ("client", "client_kgc"):
                df, N = F.doc_freq(kg, [{"visits": [q(patients[i])]} for i in c["train"]])
            elif a.arm == "server_ctx":
                df, N = gdf, gN
            else:
                df, N = None, 1
            rng = random.Random(a.seed * 1000 + ci)
            for s, ids in c.items():
                res[s] = [F.rank(a.arm, kg, q(patients[i]), a.budget, df, N, rng) for i in ids]
        out.append(res)
    return out, (F.merge_fqp_stats(fq_parts) if fq_parts else None)

def merge_text_metrics(parts):
    """weighted mean of per-client text metrics (BLEU = weighted mean of client corpus BLEU)"""
    W = sum(p.get("n", 0) for p in parts) or 1
    keys = {k for p in parts for k in p if k != "n"}
    out = {k: float(sum(p[k] * p.get("n", 0) for p in parts if k in p) / W) for k in keys}
    out["n"] = W
    return out

def build_rows(items, retrieved, kg, cnames, names, matcher, hide_dx):
    rows = []
    for p, r in zip(items, retrieved):
        src = make_source(p, r, cnames, names, hide_dx)
        cs = p["visits"][0]
        ctx = {kg.code2cui[x] for x in cs if x in kg.code2cui and not (hide_dx and x.startswith("D_"))}
        rows.append((src, p["target"], ctx | set(r) | matcher(src)))
    return rows

def cmd_train_text(a):
    import torch
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    total = F.Meter()
    with open(a.cohort, "rb") as f:
        cohort = pickle.load(f)
    patients = cohort["patients"]
    if not int(getattr(a, "meas", 1)):
        import measurements as MS
        patients = MS.strip(patients)
    kg = F.KG(pickle.load(open(a.kg, "rb")))
    base = os.path.splitext(a.kg)[0]
    names = pickle.load(open(base + "_names.pkl", "rb")) if os.path.exists(base + "_names.pkl") else {}
    cnames = load_code_names(a.kgdir)
    matcher = ConceptMatcher(names, a.kgdir)
    clients = F.make_clients(patients, a.clients, a.partition, a.alpha, a.seed)
    if a.fl == "central":
        clients = [{s: [i for c in clients for i in c[s]] for s in ("train", "val", "test")}]
    for c in clients:
        c["train"] = c["train"][:a.tg_max_train]
        c["val"] = c["val"][:a.tg_max_val]
        c["test"] = c["test"][:a.tg_max_test]

    m_ret = F.Meter()
    retrieved, fq = retrieve_all(a, kg, patients, clients, a.hide_dx)
    ret_res = m_ret.read()
    data = []
    for ci, c in enumerate(clients):
        cd = {}
        for s, ids in c.items():
            rows = []
            for i, r in zip(ids, retrieved[ci].get(s, [[]] * len(ids))):
                p = patients[i]
                src = make_source(p, r, cnames, names, a.hide_dx)
                cs = p["visits"][0]
                ctx = {kg.code2cui[x] for x in cs if x in kg.code2cui and not (a.hide_dx and x.startswith("D_"))}
                ctx |= set(r) | matcher(src)
                rows.append((src, p["target"], ctx))
            cd[s] = rows
        data.append(cd)
    F.log(f"{tag(a)}: retrieval {ret_res['wall_s']}s; example input: {data[0]['train'][0][0][:300] if data[0]['train'] else ''}")

    tok, model = load_generator(a)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    g = lora_state(model)
    blob0 = len(F.pack_state(g, a.comm_dtype))
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, tag(a) + ".jsonl")
    open(log_path, "w").close()
    best, best_state, cum_up = None, None, 0
    local_states = [copy.deepcopy(g) for _ in data] if a.fl == "local" else None
    for rnd in range(1, a.rounds + 1):
        mr = F.Meter()
        states, ws, losses, up = [], [], [], 0
        for ci, cd in enumerate(data):
            start = local_states[ci] if a.fl == "local" else g
            st, ls = train_local_text(model, tok, start, [(x[0], x[1]) for x in cd["train"]], a, rnd, ci, a.fl)
            losses.append(ls)
            if a.fl == "local":
                local_states[ci] = st
                continue
            delta = {k: st[k] - g[k] for k in g}
            if a.fl != "central":
                blob = F.pack_state(delta, a.comm_dtype)
                up += len(blob)
                delta = F.unpack_state(blob)
            states.append(delta)
            ws.append(len(cd["train"]))
        if a.fl != "local":
            W = float(sum(ws))
            g = {k: g[k] + sum(s[k] * (w / W) for s, w in zip(states, ws)) for k in g}
        cum_up += up
        tr_res = mr.read()
        me = F.Meter()
        gens, refs, ctxs = [], [], []
        for ci, cd in enumerate(data):
            load_lora(model, local_states[ci] if a.fl == "local" else g)
            gens += generate(model, tok, [x[0] for x in cd["val"]], a)
            refs += [x[1] for x in cd["val"]]
            ctxs += [x[2] for x in cd["val"]]
        val = text_metrics(gens, refs, ctxs, matcher)
        ev = me.read()
        rec = {"round": rnd, "train_loss": float(np.mean([x for x in losses if x is not None])),
               "val": val, "resources": {"train": tr_res, "eval": ev, "bytes_up": up,
                                          "bytes_down": blob0 * len(data) if a.fl not in ("local", "central") else 0,
                                          "gen_ms_per_item": 1000 * ev["wall_s"] / max(1, len(gens))}}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        F.log(f"r{rnd:02d} loss={rec['train_loss']:.3f} val rougeL={val.get('rougeL', 0):.3f} "
              f"conceptF1={val.get('concept_f1', 0):.3f} faith={val.get('faithfulness', 0):.3f} "
              f"{tr_res['wall_s']}s rss={tr_res['rss_mb']}MB")
        if best is None or val.get("rougeL", 0) > best["val"].get("rougeL", 0):
            best = rec
            best_state = copy.deepcopy(local_states if a.fl == "local" else g)

    mt = F.Meter()
    gens, refs, ctxs, examples = [], [], [], []
    for ci, cd in enumerate(data):
        load_lora(model, best_state[ci] if a.fl == "local" else best_state)
        out = generate(model, tok, [x[0] for x in cd["test"]], a)
        gens += out
        refs += [x[1] for x in cd["test"]]
        ctxs += [x[2] for x in cd["test"]]
        for (src, ref, _), gen in list(zip(cd["test"], out))[:2]:
            examples.append({"client": ci, "input": src[:600], "generated": gen, "reference": ref})
    test = text_metrics(gens, refs, ctxs, matcher, a.bertscore)
    test_res = mt.read()
    summary = {"tag": tag(a), "task": cohort["task"], "kind": "text", "fl": a.fl, "model": ("t5-full" if a.lora_r <= 0 else "t5-lora"),
               "arm": a.arm, "meas": int(getattr(a, "meas", 1)), "clients": a.clients, "seed": a.seed, "partition": a.partition,
               "hide_dx": int(a.hide_dx), "budget": a.budget, "rounds": a.rounds,
               "best_round": best["round"], "val_at_best": best["val"], "test_text": test,
               "client_sizes": [{s: len(v) for s, v in cd.items()} for cd in data],
               "fqp": fq, "generator": a.tg_model,
               "resources": {"retrieval": ret_res, "total": total.read(), "test_generation": test_res,
                             "trainable_params": int(n_train), "all_params": int(n_all),
                             "lora_mb": round(n_train * 4 / 2**20, 3), "total_mb_up": round(cum_up / 2**20, 2),
                             "gen_ms_per_item": round(1000 * test_res["wall_s"] / max(1, len(gens)), 2),
                             "threads": a.threads}}
    json.dump(summary, open(os.path.join(a.out, tag(a) + ".summary.json"), "w"), indent=1)
    json.dump(examples, open(os.path.join(a.out, tag(a) + ".examples.json"), "w"), indent=1)
    F.log("done", tag(a), json.dumps({k: round(v, 4) for k, v in test.items()
                                       if k in ("rougeL", "bleu", "concept_f1", "faithfulness", "hallucination", "context_recall")}))

def tag(a):
    sfx = F.fqp_suffix(a.arm, a.fqp_hops, a.fqp_decoys, a.fqp_repl) + ("_hidedx" if a.hide_dx else "") + F.meas_suffix(a)
    return f"{a.task_tag}__{a.fl}__t5lora__{a.arm}{sfx}__c{a.clients}__{a.partition}__seed{a.seed}"

def add_text_args(p):
    p.add_argument("--kgdir", required=True, help="KG_mapping folder (ICD-9 names, UMLS)")
    p.add_argument("--tg_model", default=os.environ.get("TG_MODEL", "google/flan-t5-small"))
    p.add_argument("--hide_dx", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=8, help="LoRA rank; 0 = full fine-tuning")
    p.add_argument("--tg_lr", type=float, default=5e-4)
    p.add_argument("--tg_batch", type=int, default=8)
    p.add_argument("--tg_gen_batch", type=int, default=16)
    p.add_argument("--max_src", type=int, default=384)
    p.add_argument("--max_tgt", type=int, default=96)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--tg_max_train", type=int, default=400, help="train items per client per round (CPU budget)")
    p.add_argument("--tg_max_val", type=int, default=30)
    p.add_argument("--tg_max_test", type=int, default=100)
    p.add_argument("--bertscore", action="store_true")
