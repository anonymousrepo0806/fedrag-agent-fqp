
"""
fedrag_sim.py -- FMRAG / PARC on MIMIC-III, CPU-only, one machine
=================================================================

In-process simulation (server + all clients in one Python process). The
Docker version (fl_net.py) reuses everything here.

TASKS (--task)
  mortality_next  death in the last admission, from earlier admissions
  ade_next        adverse drug event (ICD9 E930-E949) in the last admission,
                  from earlier admissions
  ade             ADE detection in the same record; E930-E949 codes are
                  REMOVED from the inputs (they define the label)
  multitask       mortality_next + ade_next, two heads

RETRIEVAL ARMS (--arm)  same budget, model and FL loop
  none | random | server | client (FMRAG) | client_kgc | server_ctx |
  client_local | fqp
  fqp = FEDERATED QUERY PROCESSING over per-client KGs (fqp.py): source
  selection with Bloom summaries, decomposition, bound join, decoys.
  client_local = the client's own KG partition only (no federation).
  client_kgc = client-side retrieval re-ranked by the federated KG-completion
  model (DistMult). server_ctx = non-private reference.

FL METHODS (--fl)
  fedavg | fedprox | fedcurv | fedlwf | parc (EWC + KD) | fedasync |
  local | central

MODELS (--model)
  sage (GraphSAGE) | bag (no message passing) | retain (RETAIN) |
  transformer (G-BERT-lite, no pre-training)

EXTRAS
  --user_data DIR      each client adds its own local cases (client_<i>.csv)
  --kgc_rounds N       federated KG completion (DistMult) + MRR/Hits@k
  --hypotheses         client-side drug hypothesis generation (UMLS paths)
                       + evaluation against next-admission prescriptions
  --save_model         save the best model for the `infer` command

Subcommands: build, kg, retrieval, train, sweep, tables, infer.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import io
import json
import math
import os
import pickle
import random
try:
    import resource
except ImportError:
    resource = None
import sys
import time
import zlib
from collections import Counter, defaultdict, deque

import numpy as np

import measurements as MS

ARMS_ALL = ["none", "random", "server", "client", "client_kgc", "server_ctx", "client_local", "fqp", "sparql_ext"]
FQP_ARMS = ("fqp", "client_local")
ENGINE_ARMS = FQP_ARMS + ("sparql_ext",)
FL_ALL = ["fedavg", "fedprox", "fedcurv", "fedlwf", "parc", "fedasync", "local", "central"]
MODELS = ["sage", "bag", "retain", "transformer"]
TASK_HEADS = {"mortality_next": ["mortality"], "ade_next": ["ade"],
              "ade": ["ade"], "multitask": ["mortality", "ade"]}

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def rss_mb():
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except Exception:
        return float("nan")

def peak_rss_mb():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def cpu_secs():
    return time.process_time()

class Meter:
    def __init__(self):
        self.w0, self.c0 = time.time(), cpu_secs()

    def read(self):
        return {"wall_s": round(time.time() - self.w0, 3),
                "cpu_s": round(cpu_secs() - self.c0, 3),
                "rss_mb": round(rss_mb(), 1),
                "peak_rss_mb": round(peak_rss_mb(), 1)}

_USECOLS = {
    "ADMISSIONS.csv":     ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "HOSPITAL_EXPIRE_FLAG"],
    "DIAGNOSES_ICD.csv":  ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "PROCEDURES_ICD.csv": ["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
    "PRESCRIPTIONS.csv":  ["SUBJECT_ID", "HADM_ID", "DRUG"],
}

def find_table(root, name):
    """Case-insensitive NAME / NAME.gz in ROOT or one level below."""
    want = {name.lower(), name.lower() + ".gz"}
    if not os.path.isdir(root):
        return None
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower() in want:
                return os.path.join(d, f)
        if d.count(os.sep) - root.count(os.sep) >= 1:
            break
    return None

def _read(root, name):
    import pandas as pd
    path = find_table(root, name)
    if path is None:
        raise SystemExit(f"ERROR: {name} (or .csv.gz) not found in {root}")
    head = pd.read_csv(path, nrows=0)
    colmap = {c.upper(): c for c in head.columns}
    use = [colmap[c] for c in _USECOLS[name] if c in colmap]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    return df

def is_ade_code(raw):
    """ICD9-CM E930-E949: drugs causing adverse effects in therapeutic use
    (E950-E959 self-harm excluded)."""
    c = str(raw).strip().upper()
    if c.startswith("D_"):
        c = c[2:]
    if not c.startswith("E"):
        return False
    d = c[1:4]
    return d.isdigit() and 930 <= int(d) <= 949

def diag_to_dotted(raw):
    c = str(raw).strip().upper().replace(".", "")
    if c.startswith("E"):
        return c if len(c) <= 4 else c[:4] + "." + c[4:]
    return c if len(c) <= 3 else c[:3] + "." + c[3:]

def proc_to_dotted(raw):
    c = str(raw).strip().replace(".", "")
    return c if len(c) <= 2 else c[:2] + "." + c[2:]

def make_patient(pid, adm_codes, adm_died, task, max_per_visit=128):
    """adm_codes: chronological list of code lists; adm_died: list of 0/1."""
    adm_ade = [int(any(c.startswith("D_") and is_ade_code(c) for c in cs)) for cs in adm_codes]
    if task in ("mortality_next", "ade_next", "multitask"):
        if len(adm_codes) < 2:
            return None
        visits = [_dedup(cs)[:max_per_visit] for cs in adm_codes[:-1]]
        nxt = _dedup(adm_codes[-1])
        y = {"mortality_next": int(adm_died[-1]), "ade_next": adm_ade[-1],
             "multitask": [int(adm_died[-1]), adm_ade[-1]]}[task]
    elif task == "ade":
        visits = [_dedup([c for c in cs if not (c.startswith("D_") and is_ade_code(c))])[:max_per_visit]
                  for cs in adm_codes]
        nxt = []
        y = int(any(adm_ade))
    else:
        raise SystemExit("unknown task " + task)
    visits = [v for v in visits if v]
    if not visits:
        return None
    return {"id": pid, "visits": visits, "y": y, "next": nxt}

def yvec(p):
    return p["y"] if isinstance(p["y"], list) else [p["y"]]

def cmd_build(a):
    import pandas as pd
    m = Meter()
    log("reading MIMIC-III CSVs from", a.mimic)
    adm = _read(a.mimic, "ADMISSIONS.csv")
    diag = _read(a.mimic, "DIAGNOSES_ICD.csv")
    proc = _read(a.mimic, "PROCEDURES_ICD.csv")
    pres = None if a.no_drugs else _read(a.mimic, "PRESCRIPTIONS.csv")
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"], errors="coerce")
    codes_by_adm = defaultdict(list)
    for h, c in zip(diag["HADM_ID"], diag["ICD9_CODE"]):
        if pd.notna(c):
            codes_by_adm[h].append("D_" + diag_to_dotted(c))
    for h, c in zip(proc["HADM_ID"], proc["ICD9_CODE"]):
        if pd.notna(c):
            codes_by_adm[h].append("P_" + proc_to_dotted(c))
    if pres is not None:
        for h, d in zip(pres["HADM_ID"], pres["DRUG"]):
            if pd.notna(d) and pd.notna(h):
                codes_by_adm[h].append("M_" + str(d).strip().lower())
    del diag, proc, pres
    meas = MS.load(getattr(a, "measurements", None))
    if meas:
        for h, toks in meas.items():
            if h in codes_by_adm:
                codes_by_adm[h].extend(toks)
        log(f"added lab / vital findings to {sum(h in codes_by_adm for h in meas):,} admissions")
    patients = []
    excl = set()
    if getattr(a, "exclude_file", None) and os.path.exists(a.exclude_file):
        excl = {int(x) for x in open(a.exclude_file).read().split() if x.strip().isdigit()}
        log(f"excluding {len(excl)} subjects reserved as clients' local cases ({a.exclude_file})")
    adm = adm.sort_values(["SUBJECT_ID", "ADMITTIME"])
    for sid, g in adm.groupby("SUBJECT_ID"):
        if int(sid) in excl:
            continue
        hadms = list(g["HADM_ID"])
        died = list(g["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int))
        p = make_patient(int(sid), [codes_by_adm.get(h, []) for h in hadms], died,
                         a.task, a.max_codes_per_visit)
        if p:
            patients.append(p)
    heads = TASK_HEADS[a.task]
    Y = np.array([yvec(p) for p in patients]) if patients else np.zeros((0, len(heads)))
    stats = {"task": a.task, "patients": len(patients),
             "positive": {h: int(Y[:, j].sum()) for j, h in enumerate(heads)},
             "prevalence": {h: float(Y[:, j].mean()) if len(Y) else 0.0 for j, h in enumerate(heads)},
             "mean_visits": float(np.mean([len(p["visits"]) for p in patients])) if patients else 0,
             "mean_codes": float(np.mean([sum(map(len, p["visits"])) for p in patients])) if patients else 0,
             "resources": m.read()}
    log("cohort:", json.dumps(stats))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"task": a.task, "patients": patients, "stats": stats}, f, protocol=4)
    json.dump(stats, open(os.path.splitext(a.out)[0] + ".stats.json", "w"), indent=1)

def load_user_csv(path, task):
    """Local cases of ONE client (never leave the client). Header required:
      patient_id, visit_id, conditions, procedures, drugs, adverse_event[, died][, labs][, vitals]
    conditions/procedures = ICD-9 codes (dots optional), drugs = drug names,
    ';'-separated. labs / vitals (optional) = abnormal findings, e.g.
    'hyperkalemia;anemia' and 'tachycardia;hypoxemia' (names in measurements.FINDINGS).
    A patient's rows are ordered by visit_id."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["patient_id"]].append(r)

    def split(s):
        return [x.strip() for x in (s or "").split(";") if x.strip()]

    out = []
    for pid, rs in rows.items():
        rs.sort(key=lambda r: str(r.get("visit_id", "")).zfill(12))
        adm_codes, died = [], []
        for r in rs:
            cs = (["D_" + diag_to_dotted(c) for c in split(r.get("conditions"))]
                  + ["P_" + proc_to_dotted(c) for c in split(r.get("procedures"))]
                  + ["M_" + d.lower() for d in split(r.get("drugs"))]
                  + ["L_" + x.lower() for x in split(r.get("labs"))]
                  + ["V_" + x.lower() for x in split(r.get("vitals"))])
            if int(float(r.get("adverse_event") or 0)) and not any(is_ade_code(c) for c in cs):
                cs.append("D_E934.9")
            adm_codes.append(cs)
            died.append(int(float(r.get("died") or 0)))
        p = make_patient("user_" + str(pid), adm_codes, died, task)
        if p:
            p["user"] = True
            out.append(p)
    return out

def cmd_user_data(a):
    """Simulate each hospital's OWN local cases from MIMIC-III: a share of
    MIMIC patients is held back from the shared cohort and split across the
    clients, written as data/user_data/client_<i>.csv. Their SUBJECT_IDs go to
    reserved_subjects.txt so `build` leaves them out of the shared cohort."""
    import pandas as pd
    adm = _read(a.mimic, "ADMISSIONS.csv")
    diag = _read(a.mimic, "DIAGNOSES_ICD.csv")
    proc = _read(a.mimic, "PROCEDURES_ICD.csv")
    pres = _read(a.mimic, "PRESCRIPTIONS.csv")
    adm["ADMITTIME"] = pd.to_datetime(adm["ADMITTIME"], errors="coerce")
    subs = sorted(adm["SUBJECT_ID"].unique().tolist())
    rng = random.Random(a.seed)
    rng.shuffle(subs)
    chosen = set(subs[:int(round(a.fraction * len(subs)))])
    order = sorted(chosen, key=lambda _: rng.random())
    if getattr(a, "partition", "noniid") == "noniid":

        dsub = diag[diag["SUBJECT_ID"].isin(chosen)]
        chap = {s: Counter(icd9_chapter(str(c)) for c in g.dropna()).most_common(1)[0][0] if len(g.dropna()) else "none"
                for s, g in dsub.groupby("SUBJECT_ID")["ICD9_CODE"]}
        nrng = np.random.default_rng(a.seed)
        owner = {}
        for ch in sorted({chap.get(s, "none") for s in order}):
            grp = [s for s in order if chap.get(s, "none") == ch]
            for s, ci in zip(grp, nrng.choice(a.clients, size=len(grp), p=nrng.dirichlet([a.alpha] * a.clients))):
                owner[s] = int(ci)
        for i, s in enumerate(order):
            if i < a.clients:
                owner[s] = i
    else:
        owner = {s: i % a.clients for i, s in enumerate(order)}
    by = lambda df, col: {k: v for k, v in df[df["SUBJECT_ID"].isin(chosen)].groupby("HADM_ID")[col]}
    dx, px, rx = by(diag, "ICD9_CODE"), by(proc, "ICD9_CODE"), by(pres, "DRUG")
    os.makedirs(a.out, exist_ok=True)
    files = [open(os.path.join(a.out, f"client_{i}.csv"), "w", newline="", encoding="utf-8") for i in range(a.clients)]
    writers = [csv.writer(f) for f in files]
    for w in writers:
        w.writerow(["patient_id", "visit_id", "conditions", "procedures", "drugs", "adverse_event", "died",
                    "labs", "vitals"])
    mp = getattr(a, "measurements", None)
    if mp and not os.path.exists(mp) and any(find_table(a.mimic, t) for t in ("LABEVENTS.csv", "CHARTEVENTS.csv",
                                                                                  "LAB_FINDINGS.csv", "VITAL_FINDINGS.csv")):
        MS.cmd_measurements(argparse.Namespace(mimic=a.mimic, out=mp, kinds="labs,vitals", frac=0.3,
                                               chunksize=1_000_000))
    meas = MS.load(mp)
    counts = Counter()
    sub_adm = adm[adm["SUBJECT_ID"].isin(chosen)].sort_values(["SUBJECT_ID", "ADMITTIME"])
    for sid, g in sub_adm.groupby("SUBJECT_ID"):
        w = writers[owner[sid]]
        for k, (h, died) in enumerate(zip(g["HADM_ID"], g["HOSPITAL_EXPIRE_FLAG"].fillna(0).astype(int))):
            d = [str(c) for c in (dx.get(h, pd.Series(dtype=str))).dropna()]
            p = [str(c) for c in (px.get(h, pd.Series(dtype=str))).dropna()]
            m = sorted({str(x).strip() for x in (rx.get(h, pd.Series(dtype=str))).dropna()})
            ade = int(any(is_ade_code(c) for c in d))
            toks = meas.get(int(h), [])
            w.writerow([sid, k, ";".join(d), ";".join(p), ";".join(x.replace(";", ",") for x in m), ade, int(died),
                        ";".join(t[2:] for t in toks if t.startswith("L_")),
                        ";".join(t[2:] for t in toks if t.startswith("V_"))])
        counts[owner[sid]] += 1
    for f in files:
        f.close()
    with open(os.path.join(a.out, "reserved_subjects.txt"), "w") as f:
        f.write("\n".join(str(s) for s in sorted(chosen)))
    log(f"wrote local cases for {a.clients} clients ({len(chosen)} patients = {a.fraction:.0%} of MIMIC): "
        + ", ".join(f"client_{i}={counts[i]}" for i in range(a.clients)))
    log("they are excluded from the shared cohort: delete work/<task> and re-run so `build` picks this up")

def user_patients_for(user_dir, ci, task):
    if not user_dir:
        return []
    for name in (f"client_{ci}.csv", f"client{ci}.csv"):
        path = os.path.join(user_dir, name)
        if os.path.exists(path):
            ps = load_user_csv(path, task)
            log(f"client {ci}: {len(ps)} local user patients from {path}")
            return ps
    return []

_SKIP_RELS = {"same_as", "possibly_equivalent_to", "self"}

HYP_REL_W = {
    "may_treat": 1.00, "has_active_ingredient": 0.85, "is_a_risk_factor_of": 0.70,
    "may_cause": 0.55, "interacts_with": 0.40, "may_contraindicate": -1.50,
    "isa": 0.20, "is_a_subtype_of": 0.20, "belongs_to_drug_super-family": 0.30,
    "belongs_to_the_drug_family_of": 0.30,
}

HYP_INVERSE = ("may_treat", "has_active_ingredient", "isa", "is_a_subtype_of",
               "belongs_to_drug_super-family", "belongs_to_the_drug_family_of")
for _r in HYP_INVERSE:
    HYP_REL_W["~" + _r] = HYP_REL_W[_r]

def kg_file(kgdir, *names):
    """Find a KG resource in KG_mapping, umls/, resources/ or ../resources."""
    extra = os.environ.get("KG_RESOURCES", "/kg_resources")
    for n in names:
        for base in (kgdir, os.path.join(kgdir, "umls"), os.path.join(kgdir, "resources"),
                     os.path.join(kgdir, "..", "resources"), extra):
            p = os.path.join(base, n)
            if os.path.exists(p):
                return p
    return None

def build_hyp_kg(hyp_edges, kgdir):
    """CSR version of HypothesisKG (UMLS drug-relevant relations + ATC)."""
    atc_info, atc_names = {}, {}
    atc_csv = kg_file(kgdir, "ATC.csv")
    if atc_csv:
        with open(atc_csv, newline="", encoding="utf-8", errors="ignore") as f:
            for row in csv.DictReader(f):
                code = (row.get("code") or "").strip()
                if code:
                    atc_info[code] = {"name": row.get("name", ""), "class": row.get("parent_code", ""),
                                      "indication": (row.get("indication") or "")[:200]}
    cui2atc = {}
    a2u = kg_file(kgdir, "ATC_to_UMLS.csv")
    if a2u:
        with open(a2u, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                atc, cui = (row.get("ATC") or "").strip(), (row.get("UMLS") or "").strip()
                if atc and cui:
                    cui2atc[cui] = atc
                    if atc in atc_info:
                        atc_names[cui] = atc_info[atc]["name"]
    for (h, t), r in list(hyp_edges.items()):
        if r in HYP_INVERSE and (t, h) not in hyp_edges:
            hyp_edges[(t, h)] = "~" + r
    nodes = sorted({h for h, _ in hyp_edges} | {t for _, t in hyp_edges} | set(cui2atc))
    n2i = {x: i for i, x in enumerate(nodes)}
    rel_names = sorted(HYP_REL_W)
    rel2i = {r: i for i, r in enumerate(rel_names)}
    rows = sorted((n2i[h], n2i[t], rel2i[r]) for (h, t), r in hyp_edges.items())
    heads = np.array([r[0] for r in rows], dtype=np.int64)
    indptr = np.zeros(len(nodes) + 1, dtype=np.int64)
    np.add.at(indptr, heads + 1, 1)
    indptr = np.cumsum(indptr)
    is_drug = np.zeros(len(nodes), dtype=bool)
    for cui in cui2atc:
        is_drug[n2i[cui]] = True
    drug_meta = {cui: {"atc": atc, "class": atc_info.get(atc, {}).get("class", ""),
                       "indication": atc_info.get(atc, {}).get("indication", "")}
                 for cui, atc in cui2atc.items()}
    return {"nodes": nodes, "indptr": indptr,
            "indices": np.array([r[1] for r in rows], dtype=np.int32),
            "rel": np.array([r[2] for r in rows], dtype=np.int8),
            "rel_names": rel_names, "is_drug": is_drug, "drug_meta": drug_meta,
            "atc_names": atc_names}

def cmd_kg(a):
    m = Meter()
    with open(a.cohort, "rb") as f:
        cohort = pickle.load(f)
    user = []
    if getattr(a, "user_data", None) and os.path.isdir(a.user_data):
        for p in sorted(glob.glob(os.path.join(a.user_data, "client*.csv"))):
            user += load_user_csv(p, cohort["task"])
    pts = cohort["patients"] + user
    codes = {c for p in pts for v in p["visits"] for c in v} | {c for p in pts for c in p.get("next", [])}
    icd = {}
    with open(kg_file(a.kgdir, "ICD9CM_to_UMLS.csv")) as f:
        for r in csv.DictReader(f):
            icd[r["ICD9CM"].strip()] = r["UMLS"].strip()
    names_path = kg_file(a.kgdir, "concept_names.txt")
    name2cui = {}
    with open(names_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                name2cui.setdefault(parts[1].strip().lower(), parts[0])
    code2cui = {}
    for c in codes:
        if c[:2] in ("D_", "P_"):
            cui = icd.get(c[2:]) or (icd.get(c[2:].rstrip("0")) if "." in c else None)
        elif MS.is_measure(c):
            cui = MS.finding_cui(c)
        else:
            nm = c[2:]
            cui = (name2cui.get(nm) or name2cui.get(nm.split()[0])) if nm else None
        if cui:
            code2cui[c] = cui
    del name2cui
    by_type, tot_type = Counter(c[:2] for c in code2cui), Counter(c[:2] for c in codes)
    log("mapped codes to UMLS: " + ", ".join(f"{t}{by_type[t]}/{tot_type[t]}" for t in sorted(tot_type)))

    cohort_cuis = set(code2cui.values())
    adj, degree = defaultdict(set), Counter()
    kgc_trip, hyp_edges = [], {}
    n = 0
    with open(kg_file(a.kgdir, "umls.csv")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            rel, h, t = parts[0], parts[1], parts[2]
            if rel in HYP_REL_W:
                hyp_edges[(h, t)] = rel
            if rel in _SKIP_RELS or h == t:
                continue
            degree[h] += 1
            degree[t] += 1
            if h in cohort_cuis:
                adj[h].add(t)
            if t in cohort_cuis:
                adj[t].add(h)
            if h in cohort_cuis or t in cohort_cuis:
                kgc_trip.append((h, rel, t))
            n += 1
    wanted = set(adj) | {x for s in adj.values() for x in s}
    ents = sorted({h for h, _, _ in kgc_trip} | {t for _, _, t in kgc_trip})
    e2i = {e: i for i, e in enumerate(ents)}
    rels = sorted({r for _, r, _ in kgc_trip})
    r2i = {r: i for i, r in enumerate(rels)}
    trip = np.array([(e2i[h], r2i[r], e2i[t]) for h, r, t in kgc_trip], dtype=np.int32).reshape(-1, 3)
    del kgc_trip
    hyp = build_hyp_kg(hyp_edges, a.kgdir)
    del hyp_edges
    names = {}
    want_names = wanted | set(hyp["nodes"])
    with open(names_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] in want_names:
                names[parts[0]] = parts[1]
    for cui, nm in hyp.pop("atc_names").items():
        names.setdefault(cui, nm)
    kg = {"code2cui": code2cui, "adj": {k: sorted(v) for k, v in adj.items()},
          "degree": {k: degree[k] for k in wanted},
          "kgc": {"ents": ents, "rels": rels, "triples": trip}}
    stats = {"codes": len(codes), "mapped": len(code2cui), "mapped_by_type": dict(by_type),
             "triples_read": n, "cohort_cuis_with_neighbours": len(adj),
             "candidate_concepts": len(wanted), "kgc_triples": int(len(trip)),
             "kgc_entities": len(ents), "kgc_relations": len(rels),
             "hyp_nodes": len(hyp["nodes"]), "hyp_edges": int(len(hyp["indices"])),
             "hyp_drugs": int(hyp["is_drug"].sum()), "resources": m.read()}
    base = os.path.splitext(a.out)[0]
    with open(a.out, "wb") as f:
        pickle.dump(kg, f, protocol=4)
    with open(base + "_hyp.pkl", "wb") as f:
        pickle.dump(hyp, f, protocol=4)
    with open(base + "_names.pkl", "wb") as f:
        pickle.dump(names, f, protocol=4)
    json.dump(stats, open(base + ".stats.json", "w"), indent=1)
    log("kg:", json.dumps(stats))

SAFETY_RELS = ("may_contraindicate", "interacts_with")

class HypothesisEngine:
    """Port of client/hypothesis/drug_hypothesis.py with three fixes:
      * treatment / ingredient / hierarchy / drug-family relations are also
        walked in reverse: UMLS stores `may_treat` as drug -> condition, so
        the original forward search from a condition never reached the drugs
        that treat it;
      * contraindication / interaction edges are NOT used to reach candidate
        drugs (the original BFS followed them, so a drug contraindicated in
        the patient's condition could be proposed);
      * contraindications are checked against the patient's conditions AND
        current medications (original: medications only); contraindicated
        drugs are removed (counted in `removed_contraindicated`), drugs that
        interact with current medications are kept but flagged + penalised.
    Scoring as in the original: per path sum(relation weights)/len^0.8,
    summed per drug, normalised by #paths, clipped to [0,1]; score-0 drugs
    dropped; current drugs excluded. Returns CUIs (names added in reports)."""

    MAX_PATHS = 150

    def __init__(self, hyp, max_hops=3):
        self.h = hyp
        self.n2i = {x: i for i, x in enumerate(hyp["nodes"])}
        self.w = np.array([HYP_REL_W[r] for r in hyp["rel_names"]])
        self.max_hops = max_hops
        self.safety_ids = {hyp["rel_names"].index(r) for r in SAFETY_RELS}
        ci, ii = hyp["rel_names"].index("may_contraindicate"), hyp["rel_names"].index("interacts_with")
        N = len(hyp["nodes"])
        heads = np.repeat(np.arange(N), np.diff(hyp["indptr"]))
        tails, rel = hyp["indices"].astype(np.int64), hyp["rel"]
        self.contra, self.inter = set(), set()
        for code, dest in ((ci, self.contra), (ii, self.inter)):
            m = rel == code
            for h, t in zip(heads[m].tolist(), tails[m].tolist()):
                dest.add((h, t))
                dest.add((t, h))

    def _nbrs(self, i):
        s, e = self.h["indptr"][i], self.h["indptr"][i + 1]
        return self.h["indices"][s:e], self.h["rel"][s:e]

    def _paths(self, src):
        out, q, seen = [], deque([(src, [src], [])]), {src}
        while q:
            node, pn, pr = q.popleft()
            if len(pr) >= self.max_hops:
                continue
            nb, rl = self._nbrs(node)
            for t, r in zip(nb.tolist(), rl.tolist()):
                if r in self.safety_ids or t in seen:
                    continue
                seen.add(t)
                if self.h["is_drug"][t]:
                    out.append((pn + [t], pr + [r]))
                    if len(out) >= self.MAX_PATHS:
                        return out
                else:
                    q.append((t, pn + [t], pr + [r]))
        return out

    def generate(self, condition_cuis, drug_cuis, top_k=5, stats=None, feats=False):
        cur = {self.n2i[c] for c in drug_cuis if c in self.n2i}
        conds = [self.n2i[c] for c in _dedup(condition_cuis) if c in self.n2i]
        by_drug = defaultdict(list)
        for c in conds:
            for pn, pr in self._paths(c):
                s = float(sum(self.w[r] for r in pr)) / (len(pr) ** 0.8)
                by_drug[pn[-1]].append((s, pn, pr))
        hyps, removed = [], 0
        for d, ev in by_drug.items():
            if d in cur:
                continue
            if any((d, x) in self.contra for x in conds) or any((d, x) in self.contra for x in cur):
                removed += 1
                continue
            raw = sum(s for s, _, _ in ev)
            inter = [self.h["nodes"][x] for x in cur if (d, x) in self.inter]
            raw += HYP_REL_W["interacts_with"] * len(inter)
            score = max(0.0, min(1.0, raw / max(len(ev), 1)))
            if score <= 0:
                continue
            ev.sort(key=lambda e: -e[0])
            best = ev[0]
            meta = self.h["drug_meta"].get(self.h["nodes"][d], {})
            h = {"drug": self.h["nodes"][d], "atc": meta.get("atc", ""), "score": round(score, 4),
                 "interactions": inter[:5], "n_paths": len(ev),
                 "path": [self.h["nodes"][x] for x in best[1]],
                 "relations": [self.h["rel_names"][r] for r in best[2]]}
            if feats:
                rc = np.zeros(len(self.h["rel_names"]), dtype=np.float32)
                for _, _, pr in ev:
                    for r in pr:
                        rc[r] += 1
                lens = [len(pr) for _, _, pr in ev]
                h["f"] = [score, math.log1p(len(ev)), math.log1p(len({pn[0] for _, pn, _ in ev})), best[0],
                          float(min(lens)), float(np.mean(lens)), float(len(inter))] + (rc / len(ev)).tolist()
            hyps.append(h)
        if stats is not None:
            stats["removed_contraindicated"].append(removed)
        hyps.sort(key=lambda x: (-x["score"], -x["n_paths"], x["drug"]))
        return hyps[:top_k]

def run_hypotheses(engine, kg, patients, probs, thr, train_pts, top_k, max_patients, head=0,
                   rank_fn=None, max_cands=50, n_pop=20):
    """Hypotheses for the highest-risk test patients of ONE client. Ranked by
    the rule score (original) and, if rank_fn is given, by the learned
    federated ranker; both evaluated against the drugs actually prescribed at
    the next admission, next to a popularity baseline (most frequent
    next-admission drugs in the client's own training data)."""
    import hypo_learn as HL
    pop = Counter(kg.code2cui[c] for p in train_pts for c in p.get("next", [])
                  if c.startswith("M_") and c in kg.code2cui)
    pop_list = [c for c, _ in pop.most_common(300)]
    kgc_f = HL.kgc_scorer(kg)
    order = np.argsort(-probs[:, head]) if len(probs) else []
    chosen = [i for i in order if probs[i, head] >= thr][:max_patients]
    stats = defaultdict(list)
    examples = []
    t0 = time.time()
    for i in chosen:
        p = patients[i]
        cands, X, y, drugs, nxt = HL.patient_candidates(engine, kg, p, probs[i, head], pop, kgc_f, max_cands, flat,
                                                        n_pop if rank_fn is not None else 0, pop_list, stats)
        rule = [h["drug"] for h in cands if h["source"] == "kg"]
        hy = [h for h in cands if h["source"] == "kg"][:top_k]
        stats["n_hyp"].append(len(hy))
        stats["n_candidates"].append(len(cands))
        stats["has_hyp"].append(float(bool(hy)))
        if hy:
            stats["score"].append(float(np.mean([h["score"] for h in hy])))
            stats["inter"].append(float(np.mean([bool(h["interactions"]) for h in hy])))
            stats["path_len"].append(float(np.mean([len(h["relations"]) for h in hy])))
        learned = None
        if rank_fn is not None and len(cands):
            sc = rank_fn(X)
            learned = [cands[j]["drug"] for j in np.argsort(-sc)]
        if nxt:
            stats["candidate_recall"].append(float(bool(set(rule) & nxt)))
            stats["hybrid_candidate_recall"].append(float(bool({h["drug"] for h in cands} & nxt)))
            cur = set(drugs)
            base = [d for d in pop_list if d not in cur][:max(top_k, 50)]
            for name, lst in (("", rule), ("pop_", base)) + ((("learned_", learned),) if learned is not None else ()):
                for k_, v in HL.rank_metrics(lst, nxt, top_k).items():
                    stats[name + k_].append(v)
        if len(examples) < 5 and hy:
            ex = [{k: v for k, v in h.items() if k != "f"} for h in hy[:3]]
            if learned is not None:
                lt = {h["drug"]: h for h in cands}
                ex_l = [{k: v for k, v in lt[d].items() if k != "f"} for d in learned[:3]]
            else:
                ex_l = None
            examples.append({"risk": round(float(probs[i, head]), 4), "hypotheses": ex,
                             **({"learned_hypotheses": ex_l} if ex_l else {})})
    out = {k: float(np.mean(v)) for k, v in stats.items() if v}
    out.update({"patients": len(chosen), "evaluated_vs_next_admission": len(stats.get(f"P@{top_k}", [])),
                "secs": round(time.time() - t0, 2)})
    return out, examples

def hyp_training_data(engine, kg, train_pts, probs, head, max_patients, max_cands=50, seed=1, n_pop=20):
    """(X, y) of one client's training patients for the learned ranker."""
    import hypo_learn as HL
    pop = Counter(kg.code2cui[c] for p in train_pts for c in p.get("next", [])
                  if c.startswith("M_") and c in kg.code2cui)
    pop_list = [c for c, _ in pop.most_common(300)]
    kgc_f = HL.kgc_scorer(kg)
    idx = [i for i, p in enumerate(train_pts) if p.get("next")]
    random.Random(seed).shuffle(idx)
    Xs, ys = [], []
    for i in idx[:max_patients]:
        _, X, y, _, _ = HL.patient_candidates(engine, kg, train_pts[i], probs[i, head], pop, kgc_f, max_cands, flat,
                                              n_pop, pop_list)
        if y is not None and len(X):
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return np.zeros((0, 1), dtype=np.float32), np.zeros(0, dtype=np.float32)
    return np.concatenate(Xs), np.concatenate(ys)

def merge_hyp_stats(parts):
    keys = {k for p in parts for k in p if k not in ("patients", "evaluated_vs_next_admission", "secs")}
    out = {}
    for k in keys:
        w = [(p[k], p["evaluated_vs_next_admission"] if any(t in k for t in ("P@", "Hit@", "nDCG@", "MRR", "candidate_recall"))
              else p["patients"])
             for p in parts if k in p]
        W = sum(x[1] for x in w)
        if W:
            out[k] = float(sum(v * n for v, n in w) / W)
    out["patients"] = sum(p["patients"] for p in parts)
    out["evaluated_vs_next_admission"] = sum(p["evaluated_vs_next_admission"] for p in parts)
    out["secs"] = round(sum(p["secs"] for p in parts), 2)
    return out

class KG:
    def __init__(self, kg):
        self.code2cui = kg["code2cui"]
        self.adj = kg["adj"]
        self.deg = kg["degree"]
        self.kgc = kg.get("kgc")
        self._pop = {}
        self.kgc_model = None

    def neigh_of_code(self, code):
        cui = self.code2cui.get(code)
        return self.adj.get(cui, []) if cui else []

    def popular(self, code):
        r = self._pop.get(code)
        if r is None:
            r = sorted(self.neigh_of_code(code), key=lambda c: (-self.deg.get(c, 0), c))
            self._pop[code] = r
        return r

def candidate_coverage(kg, codes, exclude=()):
    cov = defaultdict(set)
    own = {kg.code2cui.get(c) for c in codes}
    for c in codes:
        for n in kg.neigh_of_code(c):
            if n not in own and n not in exclude:
                cov[n].add(c)
    return cov

def flat(p):
    return _dedup([c for v in p["visits"] for c in v])

def doc_freq(kg, patients):
    df = Counter()
    for p in patients:
        df.update(candidate_coverage(kg, flat(p)).keys())
    return df, len(patients)

def rank(arm, kg, codes, budget, df=None, n_docs=1, rng=None, lookup=None):
    if arm == "none" or budget <= 0:
        return []
    if arm == "server":
        lookup = lookup or kg.popular
        own = {kg.code2cui.get(c) for c in codes}
        lists = [lookup(c) for c in codes]
        out, seen, i = [], set(), 0
        while len(out) < budget and any(i < len(l) for l in lists):
            for l in lists:
                if i < len(l) and l[i] not in seen and l[i] not in own:
                    seen.add(l[i])
                    out.append(l[i])
                    if len(out) >= budget:
                        break
            i += 1
        return out
    cov = candidate_coverage(kg, codes)
    if not cov:
        return []
    if arm == "random":
        cands = sorted(cov)
        rng.shuffle(cands)
        return cands[:budget]
    if arm in ("client", "server_ctx", "client_kgc"):
        def base(n):
            return len(cov[n]) * (math.log((n_docs + 1) / (df.get(n, 0) + 1)) + 1.0)
        if arm == "client_kgc" and kg.kgc_model is not None:
            plaus = kg.kgc_model.plausibility(kg, cov)
            return sorted(cov, key=lambda n: (-base(n) * (0.5 + plaus.get(n, 0.5)), n))[:budget]
        return sorted(cov, key=lambda n: (-base(n), -kg.deg.get(n, 0), n))[:budget]
    raise ValueError(arm)

def _key(p):
    return "".join(map(str, yvec(p)))

_CHAPTERS = [(1, 139, "infectious"), (140, 239, "neoplasms"), (240, 279, "endocrine"), (280, 289, "blood"),
             (290, 319, "mental"), (320, 389, "nervous"), (390, 459, "circulatory"), (460, 519, "respiratory"),
             (520, 579, "digestive"), (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
             (710, 739, "musculoskeletal"), (740, 759, "congenital"), (760, 779, "perinatal"),
             (780, 799, "symptoms"), (800, 999, "injury")]

def icd9_chapter(code):
    c = code[2:] if code[:2] in ("D_", "P_") else code
    if c.startswith("V"):
        return "supplementary"
    if c.startswith("E"):
        return "external"
    try:
        n = int(c.split(".")[0])
    except ValueError:
        return "other"
    for lo, hi, name in _CHAPTERS:
        if lo <= n <= hi:
            return name
    return "other"

def main_chapter(p):
    """the patient's dominant ICD-9 diagnosis chapter ('hospital specialty')"""
    ch = Counter(icd9_chapter(c) for v in p["visits"] for c in v if c.startswith("D_"))
    return ch.most_common(1)[0][0] if ch else "none"

def partition_stats(patients, clients):
    """how non-IID a split is: sizes, label rates, chapter mix and its
    Jensen-Shannon divergence from the global chapter mix."""
    def js(p, q):
        m = 0.5 * (p + q)
        kl = lambda a, b: float(np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0])))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)
    chap = [main_chapter(p) for p in patients]
    names = sorted(set(chap))
    idx = {n: i for i, n in enumerate(names)}
    allv = np.zeros(len(names))
    for c in chap:
        allv[idx[c]] += 1
    allv /= max(1, allv.sum())
    per = []
    for c in clients:
        ids = [i for s in ("train", "val", "test") for i in c.get(s, []) if i < len(chap)]
        v = np.zeros(len(names))
        for i in ids:
            v[idx[chap[i]]] += 1
        Y = np.array([yvec(patients[i]) for i in ids]) if ids else np.zeros((0, 1))
        top = [names[j] for j in np.argsort(-v)[:3] if v[j] > 0]
        per.append({"n": len(ids), "label_rate": [round(float(x), 4) for x in (Y.mean(0) if len(Y) else [])],
                    "top_chapters": top, "js_divergence": round(js(v / max(1, v.sum()), allv), 4) if ids else None})
    sizes = [p["n"] for p in per]
    return {"clients": per, "mean_js_divergence": float(np.mean([p["js_divergence"] for p in per if p["js_divergence"] is not None])),
            "size_cv": float(np.std(sizes) / max(1e-9, np.mean(sizes))),
            "label_rate_std": [float(x) for x in np.std([p["label_rate"] for p in per if p["label_rate"]], 0)]}

def partition(patients, k, scheme, alpha, seed):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(patients))
    if k <= 1:
        return [list(idx)]
    if scheme == "noniid":

        keys = np.array([main_chapter(p) + "|" + _key(p) for p in patients])
        bins = [[] for _ in range(k)]
        for g in np.unique(keys):
            ii = idx[keys == g]
            rng.shuffle(ii)
            cuts = (np.cumsum(rng.dirichlet([alpha] * k)) * len(ii)).astype(int)[:-1]
            for b, part in enumerate(np.split(ii, cuts)):
                bins[b].extend(part.tolist())
        floor = max(1, int(min(0.02, 0.3 / k) * len(idx)))
        for b in range(k):
            while len(bins[b]) < floor:
                big = int(np.argmax([len(x) for x in bins]))
                if len(bins[big]) <= floor:
                    break
                bins[b].append(bins[big].pop(int(rng.integers(len(bins[big])))))
        return bins
    if scheme == "random":
        rng.shuffle(idx)
        floor = min(0.03, 0.5 / k)
        shares = floor + (1 - floor * k) * rng.dirichlet([1.0] * k)
        cuts = (np.cumsum(shares) * len(idx)).astype(int)[:-1]
        return [x.tolist() for x in np.split(idx, cuts)]
    if scheme == "dirichlet":
        bins = [[] for _ in range(k)]
        keys = np.array([_key(p) for p in patients])
        for lab in np.unique(keys):
            ii = idx[keys == lab]
            rng.shuffle(ii)
            cuts = (np.cumsum(rng.dirichlet([alpha] * k)) * len(ii)).astype(int)[:-1]
            for b, part in enumerate(np.split(ii, cuts)):
                bins[b].extend(part.tolist())
        return bins
    rng.shuffle(idx)
    return [x.tolist() for x in np.array_split(idx, k)]

def split_client(ids, patients, seed, fr=(0.7, 0.1, 0.2)):
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    groups = defaultdict(list)
    for i in ids:
        groups[_key(patients[i])].append(i)
    for lab in sorted(groups):
        ii = groups[lab]
        rng.shuffle(ii)
        n = len(ii)
        a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
        out["train"] += ii[:a]
        out["val"] += ii[a:b]
        out["test"] += ii[b:]
    return out

def make_clients(patients, k, scheme, alpha, seed):
    """Depends only on (k, scheme, alpha, seed) -> identical splits for every
    arm, FL method and model."""
    parts = partition(patients, k, scheme, alpha, seed)
    return [split_client(p, patients, seed + 17 * i) for i, p in enumerate(parts)]

def ir_metrics(ranked, gains, covers, n_hidden, ks):
    m = {}
    g = [gains.get(n, 0) for n in ranked]
    first = next((r + 1 for r, x in enumerate(g) if x > 0), None)
    m["MRR"] = 1.0 / first if first else 0.0
    ideal = sorted(gains.values(), reverse=True)
    for k in ks:
        top = g[:k]
        m[f"P@{k}"] = sum(1 for x in top if x > 0) / k
        m[f"R@{k}"] = sum(1 for x in top if x > 0) / len(gains)
        dcg = sum((2 ** x - 1) / math.log2(r + 2) for r, x in enumerate(top))
        idcg = sum((2 ** x - 1) / math.log2(r + 2) for r, x in enumerate(ideal[:k]))
        m[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
        cov = set()
        for n in ranked[:k]:
            cov |= covers.get(n, set())
        m[f"Cov@{k}"] = len(cov) / n_hidden
        m[f"Hit@{k}"] = 1.0 if any(x > 0 for x in top) else 0.0
    return m

def cmd_retrieval(a):
    """Masked-diagnosis IR evaluation for every retriever, incl. FQP."""
    m0 = Meter()
    with open(a.cohort, "rb") as f:
        patients = pickle.load(f)["patients"]
    kg = KG(pickle.load(open(a.kg, "rb")))
    results = []
    for k_clients in a.clients:
        for seed in a.seeds:
            clients = make_clients(patients, k_clients, a.partition, a.alpha, seed)
            if "client_kgc" in a.arms:
                kg.kgc_model = train_kgc(kg, patients, clients, a.kgc_rounds, seed)[0]
            fq = None
            sparql_eng = None
            if any(x in ENGINE_ARMS for x in a.arms):
                import fqp as Q
            if any(x in FQP_ARMS for x in a.arms):
                a.seed = seed
                if getattr(a, "tripart", "relation") == "random":
                    Q.build_sources = build_sources_random
                fq = fqp_setup(kg, k_clients, a)
            if "sparql_ext" in a.arms:
                if not a.sparql_cache:
                    raise ValueError("--sparql_cache is required when --arms includes sparql_ext "
                                      "(build one with `python sparql_ext.py build-cache ...` first)")
                import sparql_ext as SE
                sparql_eng = SE.SparqlExtEngine(a.sparql_cache, kg)
            rng_mask = random.Random(seed)
            gdf, gN = doc_freq(kg, [patients[i] for c in clients for i in c["train"]])
            agg = {arm: defaultdict(list) for arm in a.arms}
            lat = {arm: [] for arm in a.arms}
            fqst = {arm: [] for arm in a.arms}
            n_eval = n_reach = 0
            for ci, c in enumerate(clients):
                ldf, lN = doc_freq(kg, [patients[i] for i in c["train"]])
                items = []
                for i in c["test"]:
                    codes = flat(patients[i])
                    dx = [x for x in codes if x.startswith("D_") and x in kg.code2cui]
                    if len(dx) < 3:
                        continue
                    hidden = rng_mask.sample(dx, max(1, int(round(a.mask * len(dx)))))
                    query = [x for x in codes if x not in hidden]
                    qcuis = {kg.code2cui.get(q) for q in query}
                    gains, covers = {}, defaultdict(set)
                    for h in hidden:
                        hc = kg.code2cui[h]
                        if hc in qcuis:
                            continue
                        gains[hc] = 2
                        covers[hc].add(h)
                        if getattr(a, "rel", "kg") != "hidden":
                            for nb in kg.adj.get(hc, ()):
                                if nb not in qcuis:
                                    gains.setdefault(nb, 1)
                                    covers[nb].add(h)
                    n_hidden = len({h for v in covers.values() for h in v})
                    if not gains or n_hidden == 0:
                        continue
                    n_eval += 1
                    if set(gains) & set(candidate_coverage(kg, query)):
                        n_reach += 1
                    items.append((i, query, gains, covers, n_hidden))
                for arm in a.arms:
                    if arm in ENGINE_ARMS:
                        if arm == "sparql_ext":
                            eng = sparql_eng
                        else:
                            sources, summaries, _ = fq
                            eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci,
                                              decoys=a.fqp_decoys, hops=a.fqp_hops if arm == "fqp" else 1,
                                              join_top=a.fqp_join_top, seed=seed * 31 + ci,
                                              sources_allowed={ci} if arm == "client_local" else None)
                        if a.fqp_true_idf:
                            df, n_docs = (ldf, lN) if arm in FQP_ARMS else (gdf, gN)
                        else:
                            tr_cov, _, _ = eng.coverage_batch([flat(patients[i]) for i in c["train"]])
                            df = Counter()
                            for cov in tr_cov:
                                df.update(cov.keys())
                            n_docs = len(tr_cov)
                        t = time.perf_counter()
                        covs, twos, st = eng.coverage_batch([q for _, q, _, _, _ in items])
                        rankings = [Q.rank_from_cov(cv, tw, df, n_docs, a.depth, join_w=a.fqp_join_w, deg=kg.deg)
                                    for cv, tw in zip(covs, twos)]
                        if items:
                            lat[arm] += [(time.perf_counter() - t) / len(items)] * len(items)
                        st["completeness_vs_central_kg"] = float(np.mean(
                            [x for x in (Q.completeness(kg, q, cv) for (_, q, _, _, _), cv in zip(items, covs)) if x is not None] or [0]))
                        fqst[arm].append(st)
                    else:
                        df, N = (ldf, lN) if arm in ("client", "client_kgc") else (gdf, gN)
                        rankings = []
                        for i, query, _, _, _ in items:
                            t = time.perf_counter()
                            rankings.append(rank(arm, kg, query, a.depth, df, N, random.Random(seed * 7919 + i)))
                            lat[arm].append(time.perf_counter() - t)
                    for (i, query, gains, covers, n_hidden), ranked in zip(items, rankings):
                        for key, v in ir_metrics(ranked, gains, covers, n_hidden, a.k).items():
                            agg[arm][key].append(v)
            for arm in a.arms:
                row = {"clients": k_clients, "seed": seed, "arm": arm, "n_queries": n_eval,
                       "reachable": n_reach / max(1, n_eval),
                       "latency_ms_per_query": 1000 * float(np.mean(lat[arm])) if lat[arm] else None}
                row.update({key: float(np.mean(v)) for key, v in agg[arm].items()})
                if fqst[arm]:
                    row["fqp"] = merge_fqp_stats(fqst[arm])
                results.append(row)
                log(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "fqp"}))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"rows": results, "resources": m0.read()}, open(a.out, "w"), indent=1)
    log("saved", a.out)

class KGC:
    """DistMult over UMLS triples around the cohort; each client only trains
    on triples touching CUIs of its own patients."""

    def __init__(self, n_ent, n_rel, dim=32, seed=1):
        import torch
        g = torch.Generator().manual_seed(seed)
        self.E = (torch.randn(n_ent, dim, generator=g) * 0.1).requires_grad_()
        self.R = (torch.randn(n_rel, dim, generator=g) * 0.1).requires_grad_()
        self._e2i = {}

    def state(self):
        return {"E": self.E.detach().clone(), "R": self.R.detach().clone()}

    def load(self, st):
        import torch
        with torch.no_grad():
            self.E.copy_(st["E"])
            self.R.copy_(st["R"])

    def score(self, h, r, t):
        return (self.E[h] * self.R[r] * self.E[t]).sum(-1)

    def fit(self, trip, epochs=1, lr=0.01, neg=4, bs=1024, rng=None):
        import torch
        import torch.nn.functional as Fn
        opt = torch.optim.Adam([self.E, self.R], lr=lr)
        T = torch.from_numpy(trip.astype(np.int64))
        for _ in range(epochs):
            perm = torch.from_numpy(rng.permutation(len(T)))
            for i in range(0, len(T), bs):
                b = T[perm[i:i + bs]]
                h, r, t = b[:, 0], b[:, 1], b[:, 2]
                tn = torch.from_numpy(rng.integers(0, self.E.shape[0], size=(len(b), neg)))
                pos = self.score(h, r, t)
                ng = (self.E[h].unsqueeze(1) * self.R[r].unsqueeze(1) * self.E[tn]).sum(-1)
                loss = Fn.softplus(-pos).mean() + Fn.softplus(ng).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    def evaluate(self, trip, pool=50, rng=None):
        import torch
        if len(trip) == 0:
            return {}
        T = torch.from_numpy(trip.astype(np.int64))
        with torch.no_grad():
            pos = self.score(T[:, 0], T[:, 1], T[:, 2])
            neg_t = torch.from_numpy(rng.integers(0, self.E.shape[0], size=(len(T), pool - 1)))
            ng = (self.E[T[:, 0]].unsqueeze(1) * self.R[T[:, 1]].unsqueeze(1) * self.E[neg_t]).sum(-1)
            rank_ = 1 + (ng > pos.unsqueeze(1)).sum(1).float()
        return {"MRR": float((1 / rank_).mean()), "Hits@1": float((rank_ <= 1).float().mean()),
                "Hits@10": float((rank_ <= 10).float().mean())}

    def plausibility(self, kg, cov):
        """candidate -> mean over linking codes of sigmoid(max_r DistMult(code, r, candidate)),
        computed in one vectorised call per patient."""
        import torch
        hs, ts, cs, cands = [], [], [], []
        for n, codes in cov.items():
            j = self._e2i.get(n)
            if j is None:
                continue
            k = len(cands)
            got = False
            for c in codes:
                h = self._e2i.get(kg.code2cui.get(c))
                if h is not None:
                    hs.append(h)
                    ts.append(j)
                    cs.append(k)
                    got = True
            if got:
                cands.append(n)
        if not cands:
            return {}
        with torch.no_grad():
            H, T = self.E[torch.tensor(hs)], self.E[torch.tensor(ts)]
            s = torch.sigmoid(((H * T) @ self.R.T).max(1).values)
            idx = torch.tensor(cs)
            tot = torch.zeros(len(cands)).index_add_(0, idx, s)
            cnt = torch.zeros(len(cands)).index_add_(0, idx, torch.ones_like(s))
        return dict(zip(cands, (tot / cnt).tolist()))

def train_kgc(kg, patients, clients, rounds, seed, dim=32, local_epochs=1):
    info = kg.kgc
    trip = info["triples"]
    model = KGC(len(info["ents"]), len(info["rels"]), dim, seed)
    model._e2i = {e: i for i, e in enumerate(info["ents"])}
    if rounds <= 0 or len(trip) == 0:
        return model, []
    rng = np.random.default_rng(seed)
    local = []
    for c in clients:
        cuis = np.array(sorted({model._e2i[kg.code2cui[x]] for i in c["train"] for x in flat(patients[i])
                                if kg.code2cui.get(x) in model._e2i}), dtype=np.int64)
        t = trip[np.isin(trip[:, 0], cuis) | np.isin(trip[:, 2], cuis)]
        perm = rng.permutation(len(t))
        n_test = max(1, int(0.05 * len(t))) if len(t) > 20 else 0
        local.append((t[perm[n_test:]], t[perm[:n_test]]))
    hist = []
    for rnd in range(1, rounds + 1):
        g = model.state()
        states, ws = [], []
        for tr, _ in local:
            if len(tr) == 0:
                continue
            model.load(g)
            model.fit(tr, local_epochs, rng=rng)
            states.append(model.state())
            ws.append(len(tr))
        W = float(sum(ws))
        model.load({k: sum(s[k] * (w / W) for s, w in zip(states, ws)) for k in g})
        ev = [model.evaluate(te, rng=rng) for _, te in local if len(te)]
        m = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]} if ev else {}
        hist.append({"round": rnd, **m})
    log("KGC:", json.dumps(hist[-1]))
    return model, hist

def build_vocab(patients, kg):
    codes = sorted({c for p in patients for v in p["visits"] for c in v})
    concepts = sorted({n for nb in kg.adj.values() for n in nb})
    return {c: i for i, c in enumerate(codes + ["K_" + x for x in concepts])}

def build_graph(codes_by_visit, retrieved, kg, vocab, max_codes, links=None):
    """nodes = codes (+ retrieved concepts); edges = same-visit co-occurrence
    + code<->concept KG links; vis = visit index of each node (retrieved
    concepts form one extra 'knowledge visit' at the end, for RETAIN)."""
    nodes, pos, vis = [], {}, []
    for t, v in enumerate(codes_by_visit):
        for c in v:
            if c not in pos and c in vocab and len(nodes) < max_codes:
                pos[c] = len(nodes)
                nodes.append(c)
                vis.append(t)
    src, dst = [], []
    for v in codes_by_visit:
        ii = [pos[c] for c in v if c in pos]
        for x in range(len(ii)):
            for y in range(x + 1, len(ii)):
                src += [ii[x], ii[y]]
                dst += [ii[y], ii[x]]
    if retrieved:
        link = defaultdict(list)
        if links is not None:
            for r in retrieved:
                link[r] = [pos[c] for c in links.get(r, ()) if c in pos]
        else:
            for c in nodes:
                cui = kg.code2cui.get(c)
                if cui:
                    for nb in kg.adj.get(cui, ()):
                        link[nb].append(pos[c])
        kv = (max(vis) + 1) if vis else 0
        for r in retrieved:
            key = "K_" + r
            if key not in vocab:
                continue
            j = len(nodes)
            nodes.append(key)
            vis.append(kv)
            for i in link.get(r, []):
                src += [i, j]
                dst += [j, i]
    ids = [vocab[c] for c in nodes] or [0]
    vis = vis or [0]
    i32 = lambda x: np.asarray(x, dtype=np.int32)
    return (i32(ids), i32(src), i32(dst), i32(vis))

def prepare_split_data(patients, idx_lists, kg, vocab, arm, budget, max_codes,
                       df=None, N=1, rng=None, lookup=None):
    out, n_added, n_nodes, n_edges = {}, [], [], []
    for split, ids in idx_lists.items():
        items = []
        for i in ids:
            p = patients[i]
            ret = rank(arm, kg, flat(p)[:max_codes], budget, df, N, rng, lookup)
            g = build_graph(p["visits"], ret, kg, vocab, max_codes)
            n_added.append(len(ret))
            n_nodes.append(len(g[0]))
            n_edges.append(len(g[1]))
            items.append((g, yvec(p)))
        out[split] = items
    st = {"mean_concepts_added": float(np.mean(n_added)) if n_added else 0.0,
          "frac_with_retrieval": float(np.mean([x > 0 for x in n_added])) if n_added else 0.0,
          "mean_nodes": float(np.mean(n_nodes)) if n_nodes else 0.0,
          "mean_edges": float(np.mean(n_edges)) if n_edges else 0.0}
    return out, st

def prepare_split_data_fqp(patients, idx_lists, kg, vocab, budget, max_codes, engine, join_w=0.5):
    """FQP / client_local: candidates come from the federation (or only the
    client's own KG), local idf from the client's own training patients."""
    import fqp as Q
    order = [(s, i) for s, ids in idx_lists.items() for i in ids]
    qcodes = [flat(patients[i])[:max_codes] for _, i in order]
    covs, twos, fst = engine.coverage_batch(qcodes)
    df = Counter()
    n_train = 0
    for (s, _), cov in zip(order, covs):
        if s == "train":
            df.update(cov.keys())
            n_train += 1
    out = {s: [] for s in idx_lists}
    n_added, n_nodes, n_edges, comp = [], [], [], []
    for (s, i), q, cov, two in zip(order, qcodes, covs, twos):
        ret = Q.rank_from_cov(cov, two, df, n_train, budget, join_w=join_w, deg=kg.deg)
        g = build_graph(patients[i]["visits"], ret, kg, vocab, max_codes, links=cov)
        n_added.append(len(ret))
        n_nodes.append(len(g[0]))
        n_edges.append(len(g[1]))
        c = Q.completeness(kg, q, cov)
        if c is not None:
            comp.append(c)
        out[s].append((g, yvec(patients[i])))
    fst["completeness_vs_central_kg"] = float(np.mean(comp)) if comp else None
    st = {"mean_concepts_added": float(np.mean(n_added)) if n_added else 0.0,
          "frac_with_retrieval": float(np.mean([x > 0 for x in n_added])) if n_added else 0.0,
          "mean_nodes": float(np.mean(n_nodes)) if n_nodes else 0.0,
          "mean_edges": float(np.mean(n_edges)) if n_edges else 0.0}
    return out, st, fst

def build_sources_random(kgc, n_sources, repl, seed, only=None):
    import fqp as Q
    trip = kgc["triples"]
    rng = np.random.default_rng(seed + 991)
    r = max(1, min(repl, n_sources))
    rnd = rng.random((len(trip), n_sources))
    pick = np.argpartition(rnd, r - 1, axis=1)[:, :r]
    out = {}
    for s_ in range(n_sources):
        out[s_] = Q.Source(trip[(pick == s_).any(1)])
    return out, {0: []}

def fqp_setup(kg, n_sources, a):
    """per-client KGs (relation partition) + published Bloom summaries"""
    import fqp as Q
    sources, owners = Q.build_sources(kg.kgc, n_sources, a.fqp_repl, a.seed)
    summaries = {s: src.bloom for s, src in sources.items()}
    part = {"sources": n_sources, "replication": a.fqp_repl,
            "triples_per_source": [int(src.n_triples) for src in sources.values()],
            "entities_per_source": [int(len(src.ents)) for src in sources.values()],
            "relations_per_source": [sum(1 for ow in owners.values() if s in ow) for s in sources],
            "summary_bytes_per_source": [int(b.nbytes) for b in summaries.values()],
            "central_kg_triples": int(len(kg.kgc["triples"]))}
    return sources, summaries, part

def merge_fqp_stats(parts):
    keys = {k for p in parts for k, v in p.items() if isinstance(v, (int, float)) and v is not None}
    W = sum(p.get("queries", 0) for p in parts) or 1
    out = {}
    for k in keys:
        vals = [(p[k], p.get("queries", 0)) for p in parts if p.get(k) is not None]
        if k in ("requests", "total_mb", "secs", "queries"):
            out[k] = float(sum(v for v, _ in vals))
        elif vals:
            out[k] = float(sum(v * w for v, w in vals) / max(1, sum(w for _, w in vals)))
    return out

def collate(chunk):
    import torch
    ids, src, dst, batch, slot, sbt, pos, ys = [], [], [], [], [], [], [], []
    off = soff = 0
    T = L = 1
    for b, ((gi, gs, gd, gv), y) in enumerate(chunk):
        n = len(gi)
        nv = int(gv.max()) + 1
        ids.append(gi)
        src.append(gs + off)
        dst.append(gd + off)
        batch.append(np.full(n, b))
        slot.append(gv + soff)
        sbt.append(np.stack([np.full(nv, b), np.arange(nv)], 1))
        pos.append(np.arange(n))
        ys.append(y)
        off += n
        soff += nv
        T, L = max(T, nv), max(L, n)
    C = lambda x: torch.from_numpy(np.concatenate(x).astype(np.int64))
    return {"ids": C(ids), "src": C(src), "dst": C(dst), "batch": C(batch), "nb": len(chunk),
            "slot": C(slot), "slot_bt": C(sbt), "T": T, "pos": C(pos), "L": L,
            "y": torch.tensor(ys, dtype=torch.float)}

def make_model(kind, V, d, layers, dropout, n_out=1):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.emb = nn.Embedding(V, d)
            nn.init.normal_(self.emb.weight, std=0.1)
            if kind in ("sage", "bag"):
                L = 0 if kind == "bag" else layers
                self.self_lin = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
                self.nb_lin = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(L)])
                hin = 2 * d
            elif kind == "retain":
                self.rnn_a = nn.GRU(d, d, batch_first=True)
                self.rnn_b = nn.GRU(d, d, batch_first=True)
                self.fa = nn.Linear(d, 1)
                self.fb = nn.Linear(d, d)
                hin = d
            elif kind == "transformer":
                self.cls = nn.Parameter(torch.zeros(1, 1, d))
                self.posemb = nn.Embedding(512, d)
                enc = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True)
                self.enc = nn.TransformerEncoder(enc, max(1, layers))
                hin = d
            else:
                raise ValueError(kind)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hin, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_out))

        def forward(self, B):
            h = self.emb(B["ids"])
            n, nb = h.shape[0], B["nb"]
            if self.kind in ("sage", "bag"):
                if len(self.self_lin):
                    dst, src = B["dst"], B["src"]
                    deg = torch.zeros(n).index_add_(0, dst, torch.ones(dst.shape[0])).clamp(min=1).unsqueeze(1)
                    for sl, nl in zip(self.self_lin, self.nb_lin):
                        agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / deg
                        h = self.drop(torch.relu(sl(h) + nl(agg)))
                batch = B["batch"]
                cnt = torch.zeros(nb).index_add_(0, batch, torch.ones(n)).clamp(min=1).unsqueeze(1)
                mean = torch.zeros(nb, h.shape[1]).index_add_(0, batch, h) / cnt
                mx = torch.full((nb, h.shape[1]), -1e9).scatter_reduce(
                    0, batch.unsqueeze(1).expand_as(h), h, reduce="amax", include_self=True)
                z = torch.cat([mean, mx], 1)
            elif self.kind == "retain":
                S = B["slot_bt"].shape[0]
                cnt = torch.zeros(S).index_add_(0, B["slot"], torch.ones(n)).clamp(min=1).unsqueeze(1)
                v = torch.zeros(S, h.shape[1]).index_add_(0, B["slot"], h) / cnt
                Vt = torch.zeros(nb, B["T"], h.shape[1])
                Vt = Vt.index_put((B["slot_bt"][:, 0], B["slot_bt"][:, 1]), v)
                Vr = torch.flip(Vt, [1])
                ha, _ = self.rnn_a(Vr)
                hb, _ = self.rnn_b(Vr)
                z = (torch.softmax(self.fa(ha), 1) * torch.tanh(self.fb(hb)) * Vr).sum(1)
            else:
                X = torch.zeros(nb, B["L"], h.shape[1])
                X = X.index_put((B["batch"], B["pos"]), h + self.posemb(B["pos"].clamp(max=511)))
                mask = torch.ones(nb, B["L"], dtype=torch.bool)
                mask[B["batch"], B["pos"]] = False
                X = torch.cat([self.cls.expand(nb, 1, -1), X], 1)
                mask = torch.cat([torch.zeros(nb, 1, dtype=torch.bool), mask], 1)
                z = self.enc(X, src_key_padding_mask=mask)[:, 0]
            return self.head(z)

    return Net()

def predict(model, data, bs=512):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(data), bs):
            out.append(torch.sigmoid(model(collate(data[i:i + bs]))))
    return torch.cat(out).numpy() if out else np.zeros((0, 1))

def metrics(y, p, thr=None):
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score, matthews_corrcoef,
                                 brier_score_loss, log_loss, confusion_matrix)
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    m = {"n": int(len(y)), "pos": int(y.sum())}
    if len(y) == 0:
        return m
    m["brier"] = float(brier_score_loss(y, p))
    bins = np.minimum((p * 10).astype(int), 9)
    m["ece"] = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean()
                         for b in range(10) if (bins == b).any()))
    if len(set(y.tolist())) < 2:
        return m
    m["auroc"] = float(roc_auc_score(y, p))
    m["auprc"] = float(average_precision_score(y, p))
    m["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    m["f1_at_0.5"] = float(f1_score(y, (p >= 0.5).astype(int), zero_division=0))
    if thr is None:
        best = (-1.0, 0.5)
        for t in np.linspace(0.02, 0.98, 49):
            f = f1_score(y, (p >= t).astype(int), zero_division=0)
            if f > best[0]:
                best = (f, t)
        thr = best[1]
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    m.update({"thr": float(thr), "f1": float(f1_score(y, yp, zero_division=0)),
              "precision": float(precision_score(y, yp, zero_division=0)),
              "recall": float(recall_score(y, yp, zero_division=0)),
              "specificity": float(tn / max(1, tn + fp)), "npv": float(tn / max(1, tn + fn)),
              "accuracy": float((tp + tn) / len(y)),
              "balanced_accuracy": float(0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))),
              "mcc": float(matthews_corrcoef(y, yp)),
              "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)})
    return m

def pack_state(state, dtype="fp32"):
    import torch
    buf = io.BytesIO()
    cast = torch.float16 if dtype == "fp16" else torch.float32
    torch.save({k: v.to(cast) if v.is_floating_point() else v for k, v in state.items()}, buf)
    return zlib.compress(buf.getvalue(), 1)

def unpack_state(blob):
    import torch
    st = torch.load(io.BytesIO(zlib.decompress(blob)))
    return {k: v.float() if v.is_floating_point() else v for k, v in st.items()}

def fisher_diag(model, train, n_samples, bs=32):
    """Diagonal empirical Fisher (squared gradients of the task loss) at the
    current (= just-received global) parameters, from local training data."""
    import torch
    import torch.nn.functional as Fn
    fis = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    seen = 0
    for i in range(0, min(len(train), n_samples), bs):
        B = collate(train[i:i + bs])
        model.zero_grad()
        Fn.binary_cross_entropy_with_logits(model(B), B["y"], reduction="sum").backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fis[n] += p.grad.detach() ** 2
        seen += B["nb"]
    model.zero_grad()
    return {n: f / max(1, seen) for n, f in fis.items()}

def local_train(model, gstate, train, a, rnd, ci, method="fedavg", teacher=None):
    """One client's local update.
       fedprox : + mu/2 ||w - w_g||^2
       fedcurv : + lam_ewc/2 sum F (w - w_g)^2      (FedCurv / EWC anchored to global)
       fedlwf  : + lam_kd * KD(student, global teacher)
       parc    : + lam_ewc/2 sum F (w - w_g)^2 + lam_kd * KD   (PARC: EWC + KD)
    Returns (state, mean task loss, Fisher seconds)."""
    import torch
    import torch.nn.functional as Fn
    model.load_state_dict(gstate)
    tr = train[:]
    Y = np.array([y for _, y in tr], dtype=float).reshape(len(tr), -1)
    pos = Y.sum(0)
    pw = torch.tensor((len(tr) - pos) / np.maximum(1, pos), dtype=torch.float)
    hw = torch.ones(Y.shape[1])
    if Y.shape[1] == 2:
        hw[1] = a.ade_weight
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    t_f = time.time()
    fis = fisher_diag(model, tr, a.fisher_samples) if method in ("fedcurv", "parc") else None
    fisher_s = time.time() - t_f
    lam_quad = {"fedprox": a.mu, "fedcurv": a.lam_ewc, "parc": a.lam_ewc}.get(method, 0.0)
    lam_kd = a.lam_kd if method in ("fedlwf", "parc") else 0.0
    if lam_kd > 0 and teacher is not None:
        teacher.load_state_dict(gstate)
        teacher.eval()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    rng = random.Random(a.seed * 100003 + rnd * 101 + ci)
    losses = []
    T = a.kd_temp
    for _ in range(a.local_epochs):
        rng.shuffle(tr)
        for i in range(0, len(tr), a.batch):
            B = collate(tr[i:i + a.batch])
            out = model(B)
            l_task = (Fn.binary_cross_entropy_with_logits(out, B["y"], pos_weight=pw, reduction="none")
                      .mean(0) * hw).sum()
            loss = l_task
            if lam_quad > 0:
                q = 0.0
                for n, p in model.named_parameters():
                    d2 = (p - ref[n]) ** 2
                    q = q + ((fis[n] * d2).sum() if fis is not None else d2.sum())
                loss = loss + 0.5 * lam_quad * q
            if lam_kd > 0 and teacher is not None:
                with torch.no_grad():
                    qt = torch.sigmoid(teacher(B) / T)
                loss = loss + lam_kd * Fn.binary_cross_entropy_with_logits(out / T, qt) * T * T
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(l_task.detach()))
    return ({k: v.detach().clone() for k, v in model.state_dict().items()},
            (float(np.mean(losses)) if losses else None), round(fisher_s, 3))

def evaluate_heads(Yv, pv, Yt, pt, per, heads):
    """Pooled val picks each head's threshold; pooled + per-client test."""
    out = {}
    for j, h in enumerate(heads):
        val = metrics(Yv[:, j], pv[:, j])
        thr = val.get("thr", 0.5)
        test = metrics(Yt[:, j], pt[:, j], thr=thr)
        pc = [metrics(yy[:, j], pp[:, j], thr=thr) for yy, pp in per]
        au = [m["auroc"] for m in pc if "auroc" in m]
        out[h] = {"val": val, "test": test, "per_client": pc,
                  "worst_client_auroc": float(min(au)) if au else None,
                  "std_client_auroc": float(np.std(au)) if au else None}
    return out

def evaluate_round(model_for, data, heads):
    pv, pt, Yv, Yt, per = [], [], [], [], []
    for ci, cd in enumerate(data):
        mdl = model_for(ci)
        a_, b_ = predict(mdl, cd["val"]), predict(mdl, cd["test"])
        pv.append(a_.reshape(len(cd["val"]), len(heads)))
        pt.append(b_.reshape(len(cd["test"]), len(heads)))
        Yv += [y for _, y in cd["val"]]
        Yt += [y for _, y in cd["test"]]
        per.append((np.array([y for _, y in cd["test"]]).reshape(len(cd["test"]), len(heads)), pt[-1]))
    H = len(heads)
    return evaluate_heads(np.array(Yv).reshape(-1, H), np.concatenate(pv), np.array(Yt).reshape(-1, H),
                          np.concatenate(pt), per, heads)

def selection_score(heads_eval):
    v = [r["val"].get("auprc") for r in heads_eval.values()]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None

def selection_score_from_row(row, heads):
    v = [row["heads"][h]["val"].get("auprc") for h in heads if h in row.get("heads", {})]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None

def compute_training_efficiency(round_wall_s, per_round_client_secs, val_scores, test_scores=None,
                                patience=3, tol=1e-3):
    """Turns the per-round log any run already writes into the metrics asked
    for on top of it: total LOCAL training time per client (summed across
    all rounds, not just one round's snapshot), the wall-clock cost of
    reaching the best validation round (already selected elsewhere) and of
    an independent plateau-based convergence point, and simple
    accuracy-per-second efficiency ratios.

    round_wall_s          : [float] wall-clock seconds for round r (train+eval), in round order.
    per_round_client_secs : [[float]] round r's local-training seconds, one entry per client, in
                            client-index order (same contract as the "client_train_secs" field
                            every per-round log record already carries).
    val_scores            : [float|None] the primary selection metric (mean val AUPRC across
                            heads, same quantity best_round is already chosen from) per round.
    test_scores           : optional [float|None] primary head's test AUROC per round, used only
                            for the accuracy-per-second ratios.
    patience / tol        : a round only counts as "converged" once val_scores has failed to
                            improve by more than `tol` for `patience` consecutive rounds (standard
                            early-stopping plateau definition) -- independent of, and usually later
                            or equal to, whichever round best_round picks by raw best-so-far.
    """
    n_clients = max((len(s) for s in per_round_client_secs), default=0)
    per_client_total = [0.0] * n_clients
    for secs in per_round_client_secs:
        for ci, s in enumerate(secs):
            if s is not None:
                per_client_total[ci] += s

    cumulative_wall_s = []
    running = 0.0
    for w in round_wall_s:
        running += w or 0.0
        cumulative_wall_s.append(round(running, 3))

    best_round, best_score = None, None
    for i, v in enumerate(val_scores, start=1):
        if v is not None and (best_score is None or v > best_score):
            best_score, best_round = v, i

    convergence_round, best_so_far, plateau_run = None, None, 0
    for i, v in enumerate(val_scores, start=1):
        if v is None:
            continue
        if best_so_far is None or v > best_so_far + tol:
            best_so_far, plateau_run = v, 0
        else:
            plateau_run += 1
            if plateau_run >= patience and convergence_round is None:
                convergence_round = i - patience + 1
    if convergence_round is None:
        convergence_round = best_round

    def wall_at(rnd):
        return cumulative_wall_s[rnd - 1] if rnd and 1 <= rnd <= len(cumulative_wall_s) else None

    out = {
        "per_client_total_local_train_secs": [round(x, 3) for x in per_client_total],
        "total_local_train_secs_all_clients": round(sum(per_client_total), 3),
        "mean_local_train_secs_per_client": round(sum(per_client_total) / n_clients, 3) if n_clients else None,
        "cumulative_wall_s_by_round": cumulative_wall_s,
        "total_wall_s": cumulative_wall_s[-1] if cumulative_wall_s else None,
        "best_round": best_round,
        "time_to_best_round_s": wall_at(best_round),
        "convergence_round": convergence_round,
        "time_to_convergence_s": wall_at(convergence_round),
        "convergence_patience_rounds": patience,
        "convergence_tolerance": tol,
        "rounds_per_second": round(len(round_wall_s) / cumulative_wall_s[-1], 4)
                            if cumulative_wall_s and cumulative_wall_s[-1] else None,
    }
    if test_scores is not None:
        t_best = test_scores[best_round - 1] if best_round and best_round <= len(test_scores) else None
        t_conv = test_scores[convergence_round - 1] if convergence_round and convergence_round <= len(test_scores) else None
        w_best, w_conv = wall_at(best_round), wall_at(convergence_round)
        out["efficiency"] = {
            "test_auroc_at_best_round": t_best,
            "test_auroc_at_convergence": t_conv,
            "test_auroc_per_second_at_best": round(t_best / w_best, 6) if t_best and w_best else None,
            "test_auroc_per_second_at_convergence": round(t_conv / w_conv, 6) if t_conv and w_conv else None,
        }
    return out

def fqp_suffix(arm, hops=2, decoys=0, repl=2):
    """non-default FQP settings get their own run name"""
    if arm != "fqp" and arm != "client_local":
        return ""
    s = ""
    if arm == "fqp" and int(hops) != 2:
        s += f"_h{hops}"
    if arm == "fqp" and int(decoys):
        s += f"_d{decoys}"
    if int(repl) != 2:
        s += f"_r{repl}"
    return s

def run_tag(a):
    sfx = fqp_suffix(a.arm, getattr(a, "fqp_hops", 2), getattr(a, "fqp_decoys", 0), getattr(a, "fqp_repl", 2))
    return f"{a.task_tag}__{a.fl}__{a.model}__{a.arm}{sfx}{meas_suffix(a)}__c{a.clients}__{a.partition}__seed{a.seed}"

def meas_suffix(a):
    """ablation runs without lab / vital findings get their own name"""
    return "" if int(getattr(a, "meas", 1)) else "_nomeas"

def load_world(a):
    """cohort + per-client user data + KG + splits (shared by sim and docker)."""
    with open(a.cohort, "rb") as f:
        cohort = pickle.load(f)
    patients, task = cohort["patients"], cohort["task"]
    if not int(getattr(a, "meas", 1)):
        patients = MS.strip(patients)
    kg = KG(pickle.load(open(a.kg, "rb")))
    clients = make_clients(patients, a.clients, a.partition, a.alpha, a.seed)
    user_all = []
    for ci, c in enumerate(clients):
        ups = user_patients_for(getattr(a, "user_data", None), ci, task)
        if not int(getattr(a, "meas", 1)):
            ups = MS.strip(ups)
        start = len(patients) + len(user_all)
        user_all += ups
        c["train"] = c["train"] + list(range(start, start + len(ups)))
    return patients + user_all, task, kg, clients

def cmd_train(a):
    import torch
    torch.set_num_threads(a.threads)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    total = Meter()
    patients, task, kg, clients = load_world(a)
    heads = TASK_HEADS[task]
    vocab = build_vocab(patients, kg)
    part_stats = partition_stats(patients, clients)
    log(f"partition {a.partition}: sizes {[c['n'] for c in part_stats['clients']]}, "
        f"mean JS divergence of specialty mix {part_stats['mean_js_divergence']:.3f}")
    if a.fl == "central":
        clients = [{s: [i for c in clients for i in c[s]] for s in ("train", "val", "test")}]

    kgc_hist, kgc_res = [], None
    if a.arm == "client_kgc" or a.kgc_rounds > 0:
        mk_ = Meter()
        kg.kgc_model, kgc_hist = train_kgc(kg, patients, clients, max(a.kgc_rounds, 10), a.seed)
        kgc_res = mk_.read()

    m_ret = Meter()
    gdf = gN = None
    fqp_parts, fqp_part = [], None
    sparql_eng = None
    if a.arm in ENGINE_ARMS:
        import fqp as Q
    if a.arm in FQP_ARMS:
        sources, summaries, fqp_part = fqp_setup(kg, a.clients, a)
    if a.arm == "sparql_ext":
        if not a.sparql_cache:
            raise ValueError("--sparql_cache is required when --arm is sparql_ext "
                              "(build one with `python sparql_ext.py build-cache ...` first)")
        import sparql_ext as SE
        sparql_eng = SE.SparqlExtEngine(a.sparql_cache, kg)
    if a.arm == "server_ctx":
        gdf, gN = doc_freq(kg, [patients[i] for c in clients for i in c["train"]])
    data, gstats = [], []
    for ci, c in enumerate(clients):
        if a.arm in ENGINE_ARMS:
            if a.arm == "sparql_ext":
                eng = sparql_eng
            else:
                eng = Q.FQPEngine(kg, summaries, Q.local_transport(sources), home=ci, decoys=a.fqp_decoys,
                                  hops=a.fqp_hops if a.arm == "fqp" else 1, join_top=a.fqp_join_top,
                                  seed=a.seed * 31 + ci,
                                  sources_allowed={ci} if a.arm == "client_local" else None)
            cd, st, fst = prepare_split_data_fqp(patients, c, kg, vocab, a.budget, a.max_codes, eng,
                                                  join_w=a.fqp_join_w)
            data.append(cd)
            gstats.append(st)
            fqp_parts.append(fst)
            continue
        if a.arm in ("client", "client_kgc"):
            df, N = doc_freq(kg, [patients[i] for i in c["train"]])
        elif a.arm == "server_ctx":
            df, N = gdf, gN
        else:
            df, N = None, 1
        cd, st = prepare_split_data(patients, c, kg, vocab, a.arm, a.budget, a.max_codes,
                                    df, N, random.Random(a.seed * 1000 + ci))
        data.append(cd)
        gstats.append(st)
    ret_res = m_ret.read()
    log(f"{run_tag(a)}: retrieval+graphs {ret_res['wall_s']}s, "
        f"concepts/patient {np.mean([s['mean_concepts_added'] for s in gstats]):.1f}")

    mk = lambda: make_model(a.model, len(vocab), a.dim, a.layers, a.dropout, len(heads))
    model, evalm = mk(), mk()
    teacher = mk() if a.fl in ("fedlwf", "parc") else None
    n_params = sum(p.numel() for p in model.parameters())
    gstate = copy.deepcopy(model.state_dict())
    full_blob = len(pack_state(gstate, a.comm_dtype))
    local_states = [copy.deepcopy(gstate) for _ in data] if a.fl == "local" else None

    tag = run_tag(a)
    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, tag + ".jsonl")
    open(log_path, "w").close()
    best, best_score, best_state, cum_up, cum_down, rec = None, None, None, 0, 0, None
    sc_hist = []
    sc_window = 3
    speed = np.random.default_rng(a.seed).uniform(0.5, 2.0, len(data))
    asy = {"t": np.zeros(len(data)), "ver": [0] * len(data), "version": 0, "base": {}}

    import edge_controller as EC
    fleet = EC.EdgeFleet()

    for rnd in range(1, a.rounds + 1):
        mr = Meter()
        losses, fisher_secs, up, down, stales = [], [], 0, 0, []
        client_secs = [0.0] * len(data)
        if a.fl == "fedasync":

            for _ in range(len(data)):
                ci = int(np.argmin(asy["t"]))
                base = asy["base"].get(ci, gstate)
                fleet.get(ci, tag).start_round(rnd)
                st, ls, _ = local_train(model, base, data[ci]["train"], a, rnd, ci, "fedavg")
                client_secs[ci] += fleet.get(ci).end_round(rnd, loss=ls)
                losses.append(ls)
                blob = pack_state({k: st[k] - base[k] for k in base}, a.comm_dtype)
                up += len(blob)
                down += full_blob
                delta = unpack_state(blob)
                stale = asy["version"] - asy["ver"][ci]
                stales.append(stale)
                a_eff = a.async_alpha / (1.0 + a.async_b * stale)
                gstate = {k: gstate[k] + a_eff * delta[k] if gstate[k].is_floating_point() else gstate[k]
                          for k in gstate}
                asy["version"] += 1
                asy["t"][ci] += len(data[ci]["train"]) * speed[ci]
                asy["ver"][ci] = asy["version"]
                asy["base"][ci] = gstate
        else:
            states, weights = [], []
            method = a.fl if a.fl in ("fedprox", "fedcurv", "fedlwf", "parc") else "fedavg"
            for ci, cd in enumerate(data):
                fleet.get(ci, tag).start_round(rnd)
                start = local_states[ci] if a.fl == "local" else gstate
                st, ls, fs = local_train(model, start, cd["train"], a, rnd, ci, method, teacher)
                client_secs[ci] = fleet.get(ci).end_round(rnd, loss=ls)
                fisher_secs.append(fs)
                losses.append(ls)
                if a.fl == "local":
                    local_states[ci] = st
                    continue
                delta = {k: st[k] - gstate[k] for k in gstate}
                if a.fl != "central":
                    blob = pack_state(delta, a.comm_dtype)
                    up += len(blob)
                    down += full_blob
                    delta = unpack_state(blob)
                states.append(delta)
                weights.append(len(cd["train"]))
            if a.fl != "local":
                W = float(sum(weights))
                gstate = {k: gstate[k] + sum(s[k] * (w / W) for s, w in zip(states, weights))
                          if gstate[k].is_floating_point() else gstate[k] for k in gstate}
        cum_up += up
        cum_down += down
        train_res = mr.read()

        me = Meter()

        def model_for(ci):
            evalm.load_state_dict(local_states[ci] if a.fl == "local" else gstate)
            return evalm
        Hd = evaluate_round(model_for, data, heads)
        eval_res = me.read()
        prim = Hd[heads[0]]
        rec = {"round": rnd, "train_loss": float(np.mean([x for x in losses if x is not None])),
               "val": prim["val"], "test": prim["test"], "per_client_test": prim["per_client"],
               "worst_client_auroc": prim["worst_client_auroc"], "std_client_auroc": prim["std_client_auroc"],
               "heads": {h: {"val": Hd[h]["val"], "test": Hd[h]["test"],
                             "worst_client_auroc": Hd[h]["worst_client_auroc"],
                             "std_client_auroc": Hd[h]["std_client_auroc"]} for h in heads},
               "resources": {"train": train_res, "eval": eval_res, "client_train_secs": client_secs,
                             "fisher_secs": fisher_secs, "async_staleness": stales,
                             "bytes_up": up, "bytes_down": down,
                             "cum_bytes_up": cum_up, "cum_bytes_down": cum_down}}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"r{rnd:02d} loss={rec['train_loss']:.4f} " + " | ".join(
            f"{h} val_auprc={Hd[h]['val'].get('auprc', float('nan')):.3f} test_auroc={Hd[h]['test'].get('auroc', float('nan')):.3f}"
            for h in heads) + f" | {train_res['wall_s']}s rss={train_res['rss_mb']}MB")
        sc = selection_score(Hd)
        sc_hist.append(sc)
        valid_hist = [x for x in sc_hist[-sc_window:] if x is not None]
        sc_smoothed = float(np.mean(valid_hist)) if valid_hist else None
        if sc_smoothed is not None and (best_score is None or sc_smoothed > best_score):
            best, best_score = rec, sc_smoothed
            best_state = copy.deepcopy(local_states if a.fl == "local" else gstate)
    if best_state is None:
        best_state = copy.deepcopy(local_states if a.fl == "local" else gstate)

    evalm.load_state_dict(local_states[0] if a.fl == "local" else gstate)
    allt = [x for cd in data for x in cd["test"]]
    t0 = time.perf_counter()
    predict(evalm, allt)
    infer_ms = 1000 * (time.perf_counter() - t0) / max(1, len(allt))

    hyp_summary = None
    if a.hypotheses:
        hyp_summary = hypotheses_for_run(a, kg, patients, clients, data, best_state, evalm, best, heads, tag)

    summary = {"tag": tag, "task": task, "heads": heads, "fl": a.fl, "model": a.model, "arm": a.arm,
               "meas": int(getattr(a, "meas", 1)),
               "clients": a.clients, "seed": a.seed, "partition": a.partition,
               "budget": a.budget, "rounds": a.rounds, "local_epochs": a.local_epochs,
               "lam_ewc": a.lam_ewc, "lam_kd": a.lam_kd, "mu": a.mu,
               "user_data": bool(getattr(a, "user_data", None)),
               "best_round": best["round"] if best else None,
               "val_at_best": best["val"] if best else None,
               "test_at_best_val": best["test"] if best else None,
               "heads_at_best": best["heads"] if best else None,
               "per_client_test_at_best": best["per_client_test"] if best else None,
               "worst_client_auroc_at_best": best["worst_client_auroc"] if best else None,
               "std_client_auroc_at_best": best["std_client_auroc"] if best else None,
               "final_test": rec["test"] if rec else None,
               "final_heads": rec["heads"] if rec else None,
               "graph_stats": {k: float(np.mean([s[k] for s in gstats])) for k in gstats[0]},
               "client_sizes": [{s: len(cd[s]) for s in cd} for cd in data],
               "partition_stats": part_stats,
               "kgc": {"history": kgc_hist, "resources": kgc_res} if kgc_hist else None,
               "hypotheses": hyp_summary,
               "fqp": ({**merge_fqp_stats(fqp_parts), "partition": fqp_part, "hops": a.fqp_hops if a.arm == "fqp" else 1,
                        "per_client": fqp_parts} if fqp_parts else None),
               "resources": {"retrieval_and_graphs": ret_res, "total": total.read(),
                             "n_params": int(n_params), "model_mb": round(n_params * 4 / 2**20, 2),
                             "msg_down_mb": round(full_blob / 2**20, 3),
                             "total_mb_up": round(cum_up / 2**20, 2),
                             "total_mb_down": round(cum_down / 2**20, 2),
                             "mean_round_wall_s": None, "infer_ms_per_patient": round(infer_ms, 4),
                             "comm_dtype": a.comm_dtype, "threads": a.threads}}
    rows = [json.loads(l) for l in open(log_path)]
    summary["resources"]["mean_round_wall_s"] = round(float(np.mean(
        [r["resources"]["train"]["wall_s"] + r["resources"]["eval"]["wall_s"] for r in rows])), 3)
    summary["efficiency"] = compute_training_efficiency(
        round_wall_s=[r["resources"]["train"]["wall_s"] + r["resources"]["eval"]["wall_s"] for r in rows],
        per_round_client_secs=[r["resources"]["client_train_secs"] for r in rows],
        val_scores=[selection_score_from_row(r, heads) for r in rows],
        test_scores=[r["test"].get("auroc") for r in rows],
    )
    summary["edge_controller"] = {
        "per_edge_total_local_train_secs": {str(e): fleet.edges[e].total_local_train_secs for e in fleet.edges},
        "total_local_train_secs_all_edges": fleet.total_local_train_secs_all_edges(),
    }
    json.dump(summary, open(os.path.join(a.out, tag + ".summary.json"), "w"), indent=1)
    if a.save_model:
        import torch
        torch.save({"state": best_state[0] if a.fl == "local" else best_state, "vocab": vocab,
                    "model": a.model, "dim": a.dim, "layers": a.layers, "dropout": a.dropout,
                    "heads": heads, "task": task, "arm": a.arm, "budget": a.budget, "max_codes": a.max_codes,
                    "thresholds": {h: (best["heads"][h]["val"].get("thr", 0.5) if best else 0.5) for h in heads}},
                   os.path.join(a.out, tag + ".model.pt"))
    log("done", tag, "best round", summary["best_round"], "test",
        {k: round(v, 4) for k, v in (summary["test_at_best_val"] or {}).items() if k in ("auroc", "auprc", "f1")})

def hypotheses_for_run(a, kg, patients, clients, data, best_state, evalm, best, heads, tag):
    import hypo_learn as HL
    base = os.path.splitext(a.kg)[0]
    if not os.path.exists(base + "_hyp.pkl"):
        log("hypotheses skipped: re-run `kg` to build", base + "_hyp.pkl")
        return None
    engine = HypothesisEngine(pickle.load(open(base + "_hyp.pkl", "rb")), a.hyp_hops)
    head = heads.index("ade") if "ade" in heads else 0
    thr = best["heads"][heads[head]]["val"].get("thr", 0.5) if best else 0.5

    trained, mh = None, Meter()
    names = HL.feature_names(engine.h["rel_names"])
    if a.hyp_learn:
        cdata = []
        for ci, (c, cd) in enumerate(zip(clients, data)):
            evalm.load_state_dict(best_state[ci] if a.fl == "local" else best_state)
            probs = predict(evalm, cd["train"]).reshape(len(cd["train"]), len(heads))
            cdata.append(hyp_training_data(engine, kg, [patients[i] for i in c["train"]], probs, head,
                                           a.hyp_train_max, seed=a.seed + ci, n_pop=a.hyp_pop_cands))
        n_pos = int(sum(y.sum() for _, y in cdata))
        if n_pos:
            mode = a.fl if a.fl in ("local", "central") else "fedavg"
            trained = HL.train_federated(cdata, len(names), a.hyp_rounds, a.hyp_lr, a.hyp_model, a.seed, mode)
        log(f"hypothesis ranker: {sum(len(y) for _, y in cdata)} training pairs, {n_pos} positive, "
            f"{'trained' if trained else 'NOT trained (no positives)'}")
    parts, examples = [], []
    for ci, (c, cd) in enumerate(zip(clients, data)):
        evalm.load_state_dict(best_state[ci] if a.fl == "local" else best_state)
        probs = predict(evalm, cd["test"]).reshape(len(cd["test"]), len(heads))
        st, ex = run_hypotheses(engine, kg, [patients[i] for i in c["test"]], probs, thr,
                                [patients[i] for i in c["train"]], a.hyp_k, a.hyp_max, head,
                                HL.scorer(trained, ci) if trained else None, n_pop=a.hyp_pop_cands)
        parts.append(st)
        examples += [{"client": ci, **e} for e in ex]
    out = merge_hyp_stats(parts)
    out["head"] = heads[head]
    out["per_client"] = parts
    if trained:
        out["learned"] = {"model": a.hyp_model, "rounds": a.hyp_rounds, "train_loss": trained["history"],
                          "weights": HL.weights_report(trained, names), "resources": mh.read()}
    write_hyp_examples(base, examples, os.path.join(a.out, tag + ".hypotheses_examples.json"))
    log("hypotheses:", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()
                                   if k not in ("per_client", "learned")}))
    return out

def write_hyp_examples(kg_base, examples, path):
    names_p = kg_base + "_names.pkl"
    names = pickle.load(open(names_p, "rb")) if os.path.exists(names_p) else {}
    for e in examples:
        for h in e["hypotheses"] + e.get("learned_hypotheses", []):
            h["drug_name"] = names.get(h["drug"], h["drug"])
            h["path_names"] = [names.get(x, x) for x in h["path"]]
            h["interaction_names"] = [names.get(x, x) for x in h["interactions"]]
    with open(path, "w") as f:
        json.dump(examples[:20], f, indent=1)

def cmd_infer(a):
    """patient.json: {"visits": [["D_428.0", "P_37.22", "M_furosemide"], ...]}
    (D_/P_ = ICD-9 with dots, M_ = lowercase drug name)."""
    import torch
    try:
        ck = torch.load(a.model, weights_only=False)
    except TypeError:
        ck = torch.load(a.model)
    kg = KG(pickle.load(open(a.kg, "rb")))
    p = json.load(open(a.patient))
    model = make_model(ck["model"], len(ck["vocab"]), ck["dim"], ck["layers"], ck["dropout"], len(ck["heads"]))
    model.load_state_dict(ck["state"])
    arm = ck["arm"] if ck["arm"] in ("none", "server", "client") else "client"
    ret = rank(arm, kg, flat(p)[:ck["max_codes"]], ck["budget"], Counter(), 1, random.Random(0))
    g = build_graph(p["visits"], ret, kg, ck["vocab"], ck["max_codes"])
    probs = predict(model, [(g, [0] * len(ck["heads"]))])[0]
    base = os.path.splitext(a.kg)[0]
    names = pickle.load(open(base + "_names.pkl", "rb")) if os.path.exists(base + "_names.pkl") else {}
    out = {"risk": {h: round(float(probs[j]), 4) for j, h in enumerate(ck["heads"])},
           "high_risk": {h: bool(probs[j] >= ck["thresholds"][h]) for j, h in enumerate(ck["heads"])},
           "retrieved_concepts": [names.get(c, c) for c in ret]}
    if os.path.exists(base + "_hyp.pkl"):
        eng = HypothesisEngine(pickle.load(open(base + "_hyp.pkl", "rb")))
        fl = flat(p)
        hy = eng.generate([kg.code2cui[c] for c in fl if MS.is_condition(c) and c in kg.code2cui],
                          [kg.code2cui[c] for c in fl if c.startswith("M_") and c in kg.code2cui], a.k)
        for h in hy:
            h["drug_name"] = names.get(h["drug"], h["drug"])
            h["path_names"] = [names.get(x, x) for x in h["path"]]
        out["drug_hypotheses"] = hy
    print(json.dumps(out, indent=1))

def grid(name):
    main = [("fedavg", "sage", arm) for arm in ("none", "random", "server", "client", "client_kgc", "server_ctx",
                                                "client_local", "fqp", "sparql_ext")]
    fl = [(m, "sage", arm) for m in ("fedprox", "fedcurv", "fedlwf", "parc", "fedasync", "local", "central")
          for arm in ("none", "server", "client", "fqp", "sparql_ext")]
    models = [("fedavg", m, arm) for m in ("bag", "retain", "transformer")
              for arm in ("none", "server", "client", "fqp", "sparql_ext")]

    meas = [("fedavg", "sage", arm, 0) for arm in ("none", "server", "client", "fqp", "sparql_ext")]
    return {"main": main, "fl": fl, "models": models, "meas": meas, "baselines": fl + models,
            "all": main + fl + models + meas}[name]

def cmd_sweep(a):
    import subprocess
    for k in a.clients:
        for seed in a.seeds:
            for g in grid(a.grid):
                fl, model, arm = g[:3]
                meas = g[3] if len(g) > 3 else 1
                if fl == "central" and k != a.clients[0]:
                    continue
                if arm == "sparql_ext" and not a.sparql_cache:
                    log("skip (no --sparql_cache given, needed for arm sparql_ext)",
                        f"{a.task_tag}__{fl}__{model}__{arm}__c{k}__{a.partition}__seed{seed}")
                    continue
                sfx = fqp_suffix(arm, a.fqp_hops, a.fqp_decoys, a.fqp_repl) + ("" if meas else "_nomeas")
                tag = f"{a.task_tag}__{fl}__{model}__{arm}{sfx}__c{k}__{a.partition}__seed{seed}"
                if os.path.exists(os.path.join(a.out, tag + ".summary.json")) and not a.force:
                    log("skip (done)", tag)
                    continue
                cmd = [sys.executable, os.path.abspath(__file__), "train",
                       "--cohort", a.cohort, "--kg", a.kg, "--out", a.out,
                       "--arm", arm, "--fl", fl, "--model", model,
                       "--clients", str(k), "--seed", str(seed),
                       "--partition", a.partition, "--alpha", str(a.alpha),
                       "--rounds", str(a.rounds), "--local_epochs", str(a.local_epochs),
                       "--budget", str(a.budget), "--task_tag", a.task_tag,
                       "--threads", str(a.threads), "--comm_dtype", a.comm_dtype,
                       "--mu", str(a.mu), "--lam_ewc", str(a.lam_ewc), "--lam_kd", str(a.lam_kd),
                       "--kgc_rounds", str(a.kgc_rounds), "--fqp_repl", str(a.fqp_repl),
                       "--fqp_hops", str(a.fqp_hops), "--fqp_decoys", str(a.fqp_decoys), "--meas", str(meas)]
                if a.user_data:
                    cmd += ["--user_data", a.user_data]
                if arm == "sparql_ext":
                    cmd += ["--sparql_cache", a.sparql_cache]
                if a.hypotheses and arm in a.hyp_arms and fl == "fedavg" and model == "sage":
                    cmd += ["--hypotheses"]
                log("RUN", tag)
                subprocess.run(cmd, check=False)

def cmd_sweep_text(a):
    import subprocess
    import shlex
    combos = ([("fedavg", arm) for arm in ("none", "random", "server", "client", "client_local", "fqp")]
              + [(fl, arm) for fl in ("fedprox", "local", "central") for arm in ("none", "fqp")])
    for k in a.clients:
        for seed in a.seeds:
            for hd in a.hide_dx:
                for fl, arm in combos:
                    if fl == "central" and k != a.clients[0]:
                        continue
                    sfx = fqp_suffix(arm, a.fqp_hops, a.fqp_decoys, a.fqp_repl) + ("_hidedx" if hd else "")
                    t = f"{a.task_tag}__{fl}__t5lora__{arm}{sfx}__c{k}__{a.partition}__seed{seed}"
                    if os.path.exists(os.path.join(a.out, t + ".summary.json")) and not a.force:
                        log("skip (done)", t)
                        continue
                    cmd = [sys.executable, os.path.abspath(__file__), "train_text", "--cohort", a.cohort,
                           "--kg", a.kg, "--kgdir", a.kgdir, "--out", a.out, "--arm", arm, "--fl", fl,
                           "--clients", str(k), "--seed", str(seed), "--partition", a.partition,
                           "--rounds", str(a.rounds), "--local_epochs", str(a.local_epochs),
                           "--budget", str(a.budget), "--task_tag", a.task_tag, "--threads", str(a.threads),
                           "--hide_dx", str(hd), "--fqp_repl", str(a.fqp_repl), "--fqp_hops", str(a.fqp_hops),
                           "--fqp_decoys", str(a.fqp_decoys), "--mu", str(a.mu)] + shlex.split(a.extra)
                    log("RUN", t)
                    subprocess.run(cmd, check=False)

ARM_LABEL = {"none": "No RAG", "random": "Random KG context", "server": "Server-side RAG",
             "client": "Client-side RAG (FMRAG)", "client_kgc": "Client-side RAG + KGC",
             "server_ctx": "Server-side + patient context$^\\dagger$",
             "client_local": "Client-side, own KG only", "fqp": "Federated query processing (FQP)",
             "sparql_ext": "External FAIR-selected SPARQL source"}
FL_LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedcurv": "FedCurv", "fedlwf": "FedLwF",
            "parc": "PARC (EWC+KD)", "fedasync": "FedAsync", "local": "Local only", "central": "Centralised"}
MODEL_LABEL = {"sage": "", "bag": " (bag)", "retain": " (RETAIN)", "transformer": " (G-BERT-lite)"}

def ms(v, d=3):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return "--"
    return f"{np.mean(v):.{d}f}" + (f" $\\pm$ {np.std(v):.{d}f}" if len(v) > 1 else "")

def _flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, pre + k + "."))
        elif not isinstance(v, list):
            out[pre + k] = v
    return out

def _heads_of(r):
    return r.get("heads") or ["mortality"]

def _head_block(r, h):
    hb = r.get("heads_at_best") or {}
    if h in hb:
        return hb[h]
    if h == _heads_of(r)[0]:
        return {"test": r.get("test_at_best_val") or {}, "worst_client_auroc": r.get("worst_client_auroc_at_best")}
    return {}

def cmd_tables(a):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.results, "*.summary.json")))]
    text_rows = [r for r in rows if r.get("kind") == "text"]
    rows = [r for r in rows if r.get("kind") != "text"]
    for r in rows:
        r.setdefault("model", "sage")
        r["model"] = r["model"] if r["model"] in MODELS else "sage"
    tex = []
    if rows:
        flat_rows = [_flatten(r) for r in rows]
        keys = sorted({k for r in flat_rows for k in r})
        with open(os.path.join(a.results, "all_runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat_rows)

        def table(sel, title, label_fn, cols):
            ks = sorted({r["clients"] for r in sel})
            groups = _dedup([label_fn(r) for r in sel])
            tex.append(f"% {title}")
            tex.append("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\nClients & Setting & "
                       + " & ".join(c[0] for c in cols) + " \\\\\n\\midrule")
            for k in ks:
                lines = []
                for g in groups:
                    rr = [r for r in sel if r["clients"] == k and label_fn(r) == g]
                    if rr:
                        lines.append(f" & {g} & " + " & ".join(ms([c[1](r) for r in rr], c[2]) for c in cols) + " \\\\")
                if lines:
                    lines[0] = f"\\multirow{{{len(lines)}}}{{*}}{{{k}}}" + lines[0]
                    tex.extend(lines)
                    tex.append("\\midrule")
            tex[-1] = "\\bottomrule\n\\end{tabular}\n"

        def arm_label(r):
            lab = ARM_LABEL[r["arm"]]
            fq = r.get("fqp") or {}
            if r["arm"] == "fqp":
                lab += f" [{fq.get('hops', 2)} hop{'s' if fq.get('hops', 2) != 1 else ''}"
                lab += f", {int(fq.get('decoys_per_subquery') or 0)} decoys"
                lab += f", repl {(fq.get('partition') or {}).get('replication', 2)}]"
            if not int(r.get("meas", 1)):
                lab += " (codes only, no labs/vitals)"
            return lab
        label = lambda r: f"{FL_LABEL[r['fl']]}{MODEL_LABEL[r['model']]} / {arm_label(r)}"
        order = lambda r: (FL_ALL.index(r["fl"]), MODELS.index(r["model"]), ARMS_ALL.index(r["arm"]))
        for h in _dedup([h for r in rows for h in _heads_of(r)]):
            HT = lambda key, h=h: (lambda r: _head_block(r, h).get("test", {}).get(key))
            cols = [("AUROC", HT("auroc"), 3), ("AUPRC", HT("auprc"), 3), ("F1", HT("f1"), 3),
                    ("MCC", HT("mcc"), 3), ("Brier", HT("brier"), 3),
                    ("Worst-client AUROC", lambda r, h=h: _head_block(r, h).get("worst_client_auroc"), 3)]
            sel = [r for r in rows if h in _heads_of(r)]
            main = sorted([r for r in sel if r["fl"] == "fedavg" and r["model"] == "sage"], key=order)
            if main:
                table(main, f"[{h}] Main: FedAvg + GraphSAGE; test at best-validation round; mean $\\pm$ std over seeds",
                      arm_label, cols)
            base = sorted([r for r in sel if not (r["fl"] == "fedavg" and r["model"] == "sage")], key=order)
            if base:
                table(base, f"[{h}] Baselines (same splits, same retrieval)", label, cols[:4])
        ps = [r["partition_stats"] for r in rows if r.get("partition_stats")]
        if ps:
            tex.append(f"% Client heterogeneity ({rows[0].get('partition')}): mean JS divergence of specialty mix "
                       f"{np.mean([p['mean_js_divergence'] for p in ps]):.3f}, client-size CV "
                       f"{np.mean([p['size_cv'] for p in ps]):.2f}, std of client label rates "
                       f"{np.round(np.mean([p['label_rate_std'] for p in ps], 0), 3).tolist()}\n")
        res_cols = [("Retrieval (s)", lambda r: r["resources"]["retrieval_and_graphs"]["wall_s"], 1),
                    ("s / round", lambda r: r["resources"]["mean_round_wall_s"], 2),
                    ("Peak RAM (MB)", lambda r: r["resources"]["total"]["peak_rss_mb"], 0),
                    ("MB up (total)", lambda r: r["resources"]["total_mb_up"], 1),
                    ("Infer. ms/pt", lambda r: r["resources"]["infer_ms_per_patient"], 3),
                    ("Concepts/pt", lambda r: r["graph_stats"]["mean_concepts_added"], 1)]
        table(sorted(rows, key=order), "Computation and communication cost", label, res_cols)
        hy = [r for r in rows if r.get("hypotheses")]
        if hy:
            k = next((key.split("@")[1] for key in hy[0]["hypotheses"] if key.startswith("P@")), "5")
            G = lambda key: (lambda r: r["hypotheses"].get(key))
            table(sorted(hy, key=order), "Drug hypotheses (client-side, high-risk test patients). "
                  "P/Hit = hypothesised drugs prescribed at the next admission; pop = popularity baseline",
                  lambda r: ARM_LABEL[r["arm"]],
                  [(f"Learned P@{k}", G(f"learned_P@{k}"), 3), (f"Learned nDCG@{k}", G(f"learned_nDCG@{k}"), 3),
                   ("Learned MRR", G("learned_MRR"), 3),
                   (f"Rule P@{k}", G(f"P@{k}"), 3), (f"Rule nDCG@{k}", G(f"nDCG@{k}"), 3), ("Rule MRR", G("MRR"), 3),
                   (f"Pop. P@{k}", G(f"pop_P@{k}"), 3), (f"Pop. nDCG@{k}", G(f"pop_nDCG@{k}"), 3),
                   ("KG cand. recall", G("candidate_recall"), 3), ("Hybrid cand. recall", G("hybrid_candidate_recall"), 3),
                   ("Coverage", G("has_hyp"), 3), ("Interaction flag", G("inter"), 3),
                   ("Contraind. removed/pt", G("removed_contraindicated"), 2)])
        fq = [r for r in rows if r.get("fqp")]
        if fq:
            G = lambda key, m=1.0: (lambda r: (r["fqp"].get(key) * m) if r["fqp"].get(key) is not None else None)
            table(sorted(fq, key=order), "Federated query processing: per patient query (completeness = share of the "
                  "central-KG candidates found)", label,
                  [("Sources", G("sources_contacted_per_query"), 2), ("Sub-queries", G("subqueries_per_query"), 1),
                   ("Join sub-q.", G("join_subqueries_per_query"), 1), ("Wasted", G("wasted_contacts_per_query"), 3),
                   ("KB/query", lambda r: ((r["fqp"].get("req_bytes_per_query") or 0) + (r["fqp"].get("resp_bytes_per_query") or 0)) / 1024, 1),
                   ("ms/query", G("ms_per_query"), 2), ("Completeness", G("completeness_vs_central_kg"), 3),
                   ("Summaries (KB)", G("summary_bytes", 1 / 1024), 1)])
        kgr = [r for r in rows if r.get("kgc")]
        if kgr:
            G = lambda key: (lambda r: (r["kgc"]["history"][-1] if r["kgc"]["history"] else {}).get(key))
            table(sorted(kgr, key=order), "Federated KG completion (DistMult), held-out local triples, 50 candidates",
                  label, [("MRR", G("MRR"), 3), ("Hits@1", G("Hits@1"), 3), ("Hits@10", G("Hits@10"), 3)])
    if text_rows:
        flat_t = [_flatten(r) for r in text_rows]
        keys = sorted({k for r in flat_t for k in r})
        with open(os.path.join(a.results, "all_text_runs.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(flat_t)
        T = lambda key, m=1.0: (lambda r: (r["test_text"].get(key) * m) if r["test_text"].get(key) is not None else None)

        def tlabel(r):
            lab = f"{FL_LABEL[r['fl']]} / {ARM_LABEL[r['arm']]}"
            if r["arm"] == "fqp" and r.get("fqp"):
                lab += f" [{int(r['fqp'].get('decoys_per_subquery') or 0)} decoys]"
            return lab + (" (no dx input)" if r.get("hide_dx") else "") + ("" if int(r.get("meas", 1)) else " (no labs/vitals)")
        torder = lambda r: (r.get("hide_dx", 0), FL_ALL.index(r["fl"]), ARMS_ALL.index(r["arm"]))
        ks = sorted({r["clients"] for r in text_rows})
        cols = [("ROUGE-1", T("rouge1"), 3), ("ROUGE-2", T("rouge2"), 3), ("ROUGE-L", T("rougeL"), 3),
                ("BLEU", T("bleu"), 1), ("Concept F1", T("concept_f1"), 3), ("Faithfulness", T("faithfulness"), 3),
                ("Halluc.", T("hallucination"), 3), ("Context recall", T("context_recall"), 3),
                ("ms/note", lambda r: r["resources"]["gen_ms_per_item"], 0)]
        if any("bertscore_f1" in r["test_text"] for r in text_rows):
            cols.insert(4, ("BERTScore", T("bertscore_f1"), 3))
        tex.append("% Clinical note generation (MIMIC-III discharge diagnoses), Flan-T5-small + LoRA, "
                   "test with the best-validation adapters, mean $\\pm$ std over seeds")
        tex.append("\\begin{tabular}{ll" + "c" * len(cols) + "}\n\\toprule\nClients & Setting & "
                   + " & ".join(c[0] for c in cols) + " \\\\\n\\midrule")
        for k in ks:
            groups = _dedup([tlabel(r) for r in sorted(text_rows, key=torder) if r["clients"] == k])
            lines = []
            for gl in groups:
                rr = [r for r in text_rows if r["clients"] == k and tlabel(r) == gl]
                lines.append(f" & {gl} & " + " & ".join(ms([c[1](r) for r in rr], c[2]) for c in cols) + " \\\\")
            if lines:
                lines[0] = f"\\multirow{{{len(lines)}}}{{*}}{{{k}}}" + lines[0]
                tex.extend(lines)
                tex.append("\\midrule")
        tex[-1] = "\\bottomrule\n\\end{tabular}\n"
    rp = os.path.join(a.results, "retrieval.json")
    if os.path.exists(rp):
        R = json.load(open(rp))["rows"]
        mets = [m for m in ("P@10", "nDCG@10", "Cov@10", "MRR") if m in R[0]]
        ks = sorted({r["clients"] for r in R})
        arms = [x for x in ARMS_ALL if any(r["arm"] == x for r in R) and x != "none"]
        tex.append("% Retrieval (masked-diagnosis, graded relevance), mean $\\pm$ std over seeds")
        tex.append("\\begin{tabular}{ll" + "c" * (len(mets) + 1) + "}\n\\toprule\nClients & Retriever & "
                   + " & ".join(mets) + " & ms/query \\\\\n\\midrule")
        for k in ks:
            for j, arm in enumerate(arms):
                rr = [r for r in R if r["clients"] == k and r["arm"] == arm]
                first = f"\\multirow{{{len(arms)}}}{{*}}{{{k}}}" if j == 0 else ""
                tex.append(f"{first} & {ARM_LABEL[arm]} & " + " & ".join(ms([r[m] for r in rr]) for m in mets)
                           + " & " + ms([r["latency_ms_per_query"] for r in rr], 2) + " \\\\")
            tex.append("\\midrule")
        tex[-1] = "\\bottomrule\n\\end{tabular}"
        tex.append(f"% Upper bound: a relevant concept was in the 1-hop candidate pool for "
                   f"{np.mean([r['reachable'] for r in R]):.1%} of queries.")
    with open(a.out, "w") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(tex))

def add_train_args(p):
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--task_tag", default="mimic")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--comm_dtype", default="fp32", choices=["fp32", "fp16"])
    p.add_argument("--mu", type=float, default=0.01, help="FedProx proximal weight")
    p.add_argument("--lam_ewc", type=float, default=0.5, help="EWC / FedCurv weight (PARC: 0.5)")
    p.add_argument("--lam_kd", type=float, default=0.5, help="KD weight (PARC, FedLwF)")
    p.add_argument("--kgc_rounds", type=int, default=0, help="federated KG-completion rounds (client_kgc: >=10)")
    p.add_argument("--user_data", default=None, help="folder with client_<i>.csv local cases")
    p.add_argument("--fqp_repl", type=int, default=2, help="FQP: clients holding each KG relation")
    p.add_argument("--fqp_hops", type=int, default=2, help="FQP: 1 = no join, 2 = bound join across sources")
    p.add_argument("--fqp_decoys", type=int, default=0, help="FQP: decoy entities per remote sub-query")
    p.add_argument("--fqp_join_top", type=int, default=20, help="FQP: first-hop candidates used in the bound join")
    p.add_argument("--fqp_join_w", type=float, default=0.5, help="FQP: weight of the log-dampened 2-hop join term")
    p.add_argument("--sparql_cache", default=None,
                   help="sparql_ext: path to a cache built by `python sparql_ext.py build-cache` "
                        "(required when --arm/--arms includes sparql_ext)")
    p.add_argument("--meas", type=int, default=1, choices=[0, 1],
                   help="use lab / vital-sign findings (1) or diagnosis/procedure/drug codes only (0, ablation)")

def add_model_args(p):
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--max_codes", type=int, default=64)
    p.add_argument("--ade_weight", type=float, default=1.0, help="ADE head loss weight (multitask)")
    p.add_argument("--fisher_samples", type=int, default=200)
    p.add_argument("--kd_temp", type=float, default=2.0)
    p.add_argument("--async_alpha", type=float, default=0.6)
    p.add_argument("--async_b", type=float, default=0.5)
    p.add_argument("--hypotheses", action="store_true")
    p.add_argument("--hyp_k", type=int, default=5)
    p.add_argument("--hyp_max", type=int, default=100, help="max high-risk test patients per client")
    p.add_argument("--hyp_hops", type=int, default=3)
    p.add_argument("--hyp_learn", type=int, default=1, help="learned federated hypothesis ranker (1) or rules only (0)")
    p.add_argument("--hyp_model", default="linear", choices=["linear", "mlp"])
    p.add_argument("--hyp_rounds", type=int, default=20)
    p.add_argument("--hyp_lr", type=float, default=0.01)
    p.add_argument("--hyp_train_max", type=int, default=300, help="training patients per client for the ranker")
    p.add_argument("--hyp_pop_cands", type=int, default=20, help="popular local drugs added to the KG candidates (hybrid)")
    p.add_argument("--save_model", action="store_true")

def add_common(p):
    p.add_argument("--cohort", required=True)
    p.add_argument("--kg", required=True)
    p.add_argument("--partition", default="noniid", choices=["noniid", "random", "iid", "dirichlet"],
                   help="noniid (default): specialty + label + size skew; dirichlet: label skew only; "
                        "random: random patients, random sizes; iid: random, equal sizes")
    p.add_argument("--alpha", type=float, default=0.5)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    b = sp.add_parser("build")
    b.add_argument("--mimic", required=True)
    b.add_argument("--task", default="mortality_next", choices=list(TASK_HEADS))
    b.add_argument("--no_drugs", action="store_true")
    b.add_argument("--max_codes_per_visit", type=int, default=128)
    b.add_argument("--exclude_file", default=None, help="SUBJECT_IDs to leave out (reserved_subjects.txt)")
    b.add_argument("--measurements", default=None, help="measurements.pkl from the `measurements` command")
    b.add_argument("--out", required=True)

    ud = sp.add_parser("user_data", help="simulate each hospital's own local cases from MIMIC")
    ud.add_argument("--mimic", required=True)
    ud.add_argument("--clients", type=int, default=10)
    ud.add_argument("--fraction", type=float, default=0.10, help="share of MIMIC patients held back as local cases")
    ud.add_argument("--seed", type=int, default=0)
    ud.add_argument("--partition", default="noniid", choices=["noniid", "random"],
                    help="noniid: specialty-skewed Dirichlet(alpha) shares; random: round-robin")
    ud.add_argument("--alpha", type=float, default=0.5)
    ud.add_argument("--measurements", default=None, help="measurements.pkl -> labs / vitals columns")
    ud.add_argument("--out", required=True)

    me = sp.add_parser("measurements", help="labs (LABEVENTS) + vitals (CHARTEVENTS) -> abnormal findings")
    MS.add_args(me)

    k = sp.add_parser("kg")
    k.add_argument("--cohort", required=True)
    k.add_argument("--kgdir", required=True)
    k.add_argument("--user_data", default=None)
    k.add_argument("--out", required=True)

    r = sp.add_parser("retrieval")
    add_common(r)
    r.add_argument("--clients", type=int, nargs="+", default=[2, 4, 6, 10])
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    r.add_argument("--arms", nargs="+", default=["random", "server", "client", "client_kgc", "server_ctx",
                                                  "client_local", "fqp"])
    r.add_argument("--fqp_repl", type=int, default=2)
    r.add_argument("--fqp_hops", type=int, default=2)
    r.add_argument("--fqp_decoys", type=int, default=0)
    r.add_argument("--fqp_join_top", type=int, default=20)
    r.add_argument("--fqp_join_w", type=float, default=0.5)
    r.add_argument("--fqp_true_idf", action="store_true")
    r.add_argument("--sparql_cache", default=None,
                   help="path to a cache built by `python sparql_ext.py build-cache` "
                        "(required when --arms includes sparql_ext)")
    r.add_argument("--mask", type=float, default=0.3)
    r.add_argument("--rel", default="kg")
    r.add_argument("--tripart", default="relation")
    r.add_argument("--depth", type=int, default=100)
    r.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    r.add_argument("--kgc_rounds", type=int, default=10)
    r.add_argument("--out", required=True)

    t = sp.add_parser("train")
    add_common(t)
    add_train_args(t)
    add_model_args(t)
    t.add_argument("--arm", required=True, choices=ARMS_ALL)
    t.add_argument("--fl", default="fedavg", choices=FL_ALL)
    t.add_argument("--model", default="sage", choices=MODELS)
    t.add_argument("--clients", type=int, default=10)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--out", required=True)

    s = sp.add_parser("sweep")
    add_common(s)
    add_train_args(s)
    s.add_argument("--grid", default="all", choices=["main", "fl", "models", "meas", "baselines", "all"])
    s.add_argument("--clients", type=int, nargs="+", default=[2, 4, 6, 10])
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    s.add_argument("--hypotheses", action="store_true")
    s.add_argument("--hyp_arms", nargs="+", default=["none", "client"])
    s.add_argument("--force", action="store_true")
    s.add_argument("--out", required=True)

    tb = sp.add_parser("tables")
    tb.add_argument("--results", required=True)
    tb.add_argument("--out", required=True)

    nt = sp.add_parser("notes", help="build the text-generation cohort from NOTEEVENTS")
    nt.add_argument("--mimic", required=True)
    nt.add_argument("--section", default="diagnosis", choices=["diagnosis", "course"])
    nt.add_argument("--max_target_chars", type=int, default=400)
    nt.add_argument("--max_codes_per_visit", type=int, default=128)
    nt.add_argument("--max_items", type=int, default=0)
    nt.add_argument("--measurements", default=None, help="measurements.pkl -> abnormal findings in the prompt")
    nt.add_argument("--out", required=True)

    import textgen as TG
    tt = sp.add_parser("train_text", help="federated retrieval-augmented note generation")
    add_common(tt)
    add_train_args(tt)
    tt.add_argument("--max_codes", type=int, default=64)
    tt.add_argument("--arm", required=True, choices=ARMS_ALL)
    tt.add_argument("--fl", default="fedavg", choices=["fedavg", "fedprox", "local", "central"])
    tt.add_argument("--clients", type=int, default=10)
    tt.add_argument("--seed", type=int, default=1)
    tt.add_argument("--out", required=True)
    TG.add_text_args(tt)

    st = sp.add_parser("sweep_text", help="all text-generation runs")
    add_common(st)
    add_train_args(st)
    st.add_argument("--kgdir", required=True)
    st.add_argument("--clients", type=int, nargs="+", default=[10])
    st.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    st.add_argument("--hide_dx", type=int, nargs="+", default=[0, 1])
    st.add_argument("--extra", default="", help="extra flags passed to train_text, e.g. '--tg_max_train 200'")
    st.add_argument("--force", action="store_true")
    st.add_argument("--out", required=True)

    inf = sp.add_parser("infer")
    inf.add_argument("--model", required=True)
    inf.add_argument("--kg", required=True)
    inf.add_argument("--patient", required=True)
    inf.add_argument("--k", type=int, default=5)

    a = ap.parse_args()
    {"build": cmd_build, "kg": cmd_kg, "user_data": cmd_user_data, "retrieval": cmd_retrieval, "train": cmd_train,
     "sweep": cmd_sweep, "tables": cmd_tables, "infer": cmd_infer, "measurements": MS.cmd_measurements,
     "notes": lambda x: TG.cmd_notes(x), "train_text": lambda x: TG.cmd_train_text(x),
     "sweep_text": cmd_sweep_text}[a.cmd](a)

if __name__ == "__main__":
    main()
