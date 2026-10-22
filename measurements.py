"""
measurements.py -- laboratory values and vital signs as clinical FINDINGS
=========================================================================

MIMIC-III LABEVENTS (~28M rows) and CHARTEVENTS (~330M rows) are streamed
once, in chunks (CPU only, bounded memory), and turned into per-admission
abnormal-finding tokens:

  L_<finding>  from laboratory results, e.g. L_hyperkalemia, L_anemia
  V_<finding>  from bedside vital signs,  e.g. V_tachycardia, V_hypoxemia

Why findings instead of raw numbers: a finding is a UMLS concept (CUI), so it
(1) becomes a node of the patient graph like a diagnosis, (2) is a retrieval
entry point into the knowledge graph (hyperkalemia -> drugs that may_treat
it, related disorders), (3) is a condition for drug-hypothesis generation and
(4) reads naturally in the note-generation prompt. Units differ per lab and
the two MIMIC systems (CareVue / MetaVision), so labs use MIMIC's own
`FLAG = abnormal` and the direction (high / low) relative to the median of
that test's normal results; vitals use standard adult thresholds.

An admission gets a finding when at least `frac` (default 0.3) of that
measurement's readings during the stay are abnormal in that direction, so
a single outlier does not count.

Output: one cached pickle {HADM_ID: [tokens]} shared by every task, plus a
stats JSON (rows read, admissions with findings, prevalence of each finding).
Missing LABEVENTS / CHARTEVENTS -> that part is skipped with a warning.
"""

from __future__ import annotations

import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

FINDINGS = {
    "hyperkalemia": ("C0020461", "hyperkalemia"),
    "hypokalemia": ("C0020621", "hypokalemia"),
    "hypernatremia": ("C0020488", "hypernatremia"),
    "hyponatremia": ("C0020625", "hyponatremia"),
    "hyperglycemia": ("C0020456", "hyperglycemia"),
    "hypoglycemia": ("C0020615", "hypoglycemia"),
    "anemia": ("C0002871", "anemia"),
    "polycythemia": ("C0032461", "polycythemia"),
    "leukocytosis": ("C0023518", "leukocytosis"),
    "leukopenia": ("C0023530", "leukopenia"),
    "thrombocytopenia": ("C0040034", "thrombocytopenia"),
    "thrombocytosis": ("C0836924", "thrombocytosis"),
    "renal_impairment": ("C1565489", "renal impairment (raised creatinine)"),
    "azotemia": ("C0242528", "azotemia (raised urea)"),
    "lactic_acidosis": ("C0001125", "lactic acidosis"),
    "hyperbilirubinemia": ("C0020433", "hyperbilirubinemia"),
    "coagulopathy": ("C0005779", "coagulopathy (raised INR)"),
    "acidosis": ("C0001122", "acidosis (low pH)"),
    "alkalosis": ("C0002063", "alkalosis (high pH)"),
    "metabolic_acidosis": ("C0220981", "low bicarbonate"),
    "hypoalbuminemia": ("C0239981", "hypoalbuminemia"),
    "hypocalcemia": ("C0020598", "hypocalcemia"),
    "hypercalcemia": ("C0020437", "hypercalcemia"),
    "hypomagnesemia": ("C0151723", "hypomagnesemia"),
    "hypermagnesemia": ("C0151714", "hypermagnesemia"),
    "hypophosphatemia": ("C0085682", "hypophosphatemia"),
    "hyperphosphatemia": ("C0085681", "hyperphosphatemia"),
    "hyperchloremia": ("C0085679", "hyperchloremia"),
    "hypochloremia": ("C0085680", "hypochloremia"),
    "ast_increased": ("C0151904", "raised AST"),
    "alt_increased": ("C0438237", "raised liver enzymes (ALT)"),
    "troponin_increased": (None, "raised troponin T"),
    "hyperuricemia": ("C0740394", "hyperuricemia"),
    "hypercapnia": ("C0020440", "hypercapnia"),
    "hypoxemia": ("C0700292", "hypoxemia"),
    "tachycardia": ("C0039231", "tachycardia"),
    "bradycardia": ("C0428977", "bradycardia"),
    "hypotension": ("C0020649", "hypotension"),
    "hypertension": ("C0020538", "high blood pressure"),
    "tachypnea": ("C0231835", "tachypnea"),
    "fever": ("C0015967", "fever"),
    "hypothermia": ("C0020672", "hypothermia"),
}

LABS = {
    "potassium":   ([50971, 50822], "potassium", "hyperkalemia", "hypokalemia"),
    "sodium":      ([50983, 50824], "sodium", "hypernatremia", "hyponatremia"),
    "glucose":     ([50931, 50809], "glucose", "hyperglycemia", "hypoglycemia"),
    "hemoglobin":  ([51222, 50811], "hemoglobin", "polycythemia", "anemia"),
    "wbc":         ([51301, 51300], "white blood cells", "leukocytosis", "leukopenia"),
    "platelets":   ([51265], "platelet count", "thrombocytosis", "thrombocytopenia"),
    "creatinine":  ([50912], "creatinine", "renal_impairment", None),
    "urea":        ([51006], "urea nitrogen", "azotemia", None),
    "lactate":     ([50813], "lactate", "lactic_acidosis", None),
    "bilirubin":   ([50885], "bilirubin, total", "hyperbilirubinemia", None),
    "inr":         ([51237], "inr(pt)", "coagulopathy", None),
    "ph":          ([50820], "ph", "alkalosis", "acidosis"),
    "bicarbonate": ([50882], "bicarbonate", None, "metabolic_acidosis"),
    "albumin":     ([50862], "albumin", None, "hypoalbuminemia"),
    "calcium":     ([50893], "calcium, total", "hypercalcemia", "hypocalcemia"),
    "magnesium":   ([50960], "magnesium", "hypermagnesemia", "hypomagnesemia"),
    "phosphate":   ([50970], "phosphate", "hyperphosphatemia", "hypophosphatemia"),
    "chloride":    ([50902], "chloride", "hyperchloremia", "hypochloremia"),
    "ast":         ([50878], "asparate aminotransferase (ast)", "ast_increased", None),
    "alt":         ([50861], "alanine aminotransferase (alt)", "alt_increased", None),
    "troponin_t":  ([51003], "troponin t", "troponin_increased", None),
    "uric_acid":   ([51007], "uric acid", "hyperuricemia", None),
    "pco2":        ([50818], "pco2", "hypercapnia", None),
    "po2":         ([50821], "po2", None, "hypoxemia"),
}

VITALS = {
    "hr":   ([211, 220045], (0, 300), (100, "tachycardia"), (60, "bradycardia")),
    "sbp":  ([51, 442, 455, 6701, 220179, 220050], (0, 300), (160, "hypertension"), (90, "hypotension")),
    "rr":   ([615, 618, 220210, 224690], (0, 80), (24, "tachypnea"), None),
    "spo2": ([646, 220277], (0, 100), None, (92, "hypoxemia")),
    "temp": ([676, 223762, 678, 223761], (25, 45), (38.0, "fever"), (36.0, "hypothermia")),
}
TEMP_F = {678, 223761}

def log(*a):
    import time
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

def _find(root, name):
    want = {name.lower(), name.lower() + ".gz"}
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower() in want:
                return os.path.join(d, f)
        if d.count(os.sep) - root.count(os.sep) >= 1:
            break
    return None

def _reader(path, cols, chunksize):
    import pandas as pd
    head = pd.read_csv(path, nrows=0)
    cmap = {c.upper(): c for c in head.columns}
    use = [cmap[c] for c in cols if c in cmap]
    for ch in pd.read_csv(path, usecols=use, chunksize=chunksize, low_memory=False):
        ch.columns = [c.upper() for c in ch.columns]
        yield ch

def pd_map(ids, mapping):
    """vectorised int id -> name"""
    import pandas as pd
    return pd.Series(ids).map(mapping).to_numpy(dtype=object).astype(str)

def is_measure(code):
    return code[:2] in ("L_", "V_")

def is_condition(code):
    """codes that describe the patient's state (diagnoses + abnormal findings)"""
    return code[:2] in ("D_", "L_", "V_")

def finding_cui(code):
    return FINDINGS.get(code[2:], (None, None))[0] if is_measure(code) else None

def finding_text(code):
    f = FINDINGS.get(code[2:])
    return f[1] if f else code[2:].replace("_", " ")

def strip(patients):
    """ablation: the same patients without lab / vital findings"""
    out = []
    for p in patients:
        q = dict(p)
        q["visits"] = [[c for c in v if not is_measure(c)] for v in p["visits"]]
        q["visits"] = [v for v in q["visits"] if v] or [p["visits"][-1][:1]]
        q["next"] = [c for c in p.get("next", []) if not is_measure(c)]
        out.append(q)
    return out

def lab_items(mimic):
    """ITEMID -> lab name. Standard MIMIC-III ids, cross-checked / extended by
    D_LABITEMS labels (fluid = Blood) when that table is present."""
    item2lab = {i: lab for lab, (ids, _, _, _) in LABS.items() for i in ids}
    p = _find(mimic, "D_LABITEMS.csv")
    if p:
        import pandas as pd
        d = pd.read_csv(p)
        d.columns = [c.upper() for c in d.columns]
        lab_of_label = {spec[1]: lab for lab, spec in LABS.items()}
        for i, lbl, fl in zip(d["ITEMID"], d["LABEL"].astype(str).str.lower().str.strip(), d["FLUID"].astype(str)):
            if lbl in lab_of_label and fl.lower().startswith("blood"):
                item2lab.setdefault(int(i), lab_of_label[lbl])
        known = {int(i): l for i, l in zip(d["ITEMID"], d["LABEL"])}
        log("lab items used: " + ", ".join(f"{i}={known.get(i, '?')}" for i in sorted(item2lab)))
    return item2lab

def read_precomputed(mimic, name):
    """LAB_FINDINGS.csv / VITAL_FINDINGS.csv (HADM_ID, FINDINGS 'a;b;c') made in
    BigQuery with scripts/bigquery/*.sql -- same rules, no big download"""
    path = _find(mimic, name)
    if not path:
        return None
    import pandas as pd
    d = pd.read_csv(path)
    d.columns = [c.upper() for c in d.columns]
    pre = "L_" if name.startswith("LAB") else "V_"
    out = {}
    for h, f in zip(d["HADM_ID"], d["FINDINGS"]):
        if pd.notna(h) and isinstance(f, str) and f.strip():
            out[int(h)] = [pre + x.strip() for x in f.split(";") if x.strip()]
    log(f"using precomputed {name}: {len(out):,} admissions with findings")
    return out

def read_labs(mimic, chunksize, frac):
    pre = read_precomputed(mimic, "LAB_FINDINGS.csv")
    if pre is not None:
        return pre, {"lab_source": "LAB_FINDINGS.csv (BigQuery)", "admissions_with_lab_findings": len(pre)}
    path = _find(mimic, "LABEVENTS.csv")
    if not path:
        log("WARNING: LABEVENTS not found -> no laboratory findings")
        return {}, {}
    item2lab = lab_items(mimic)
    ids = np.array(sorted(item2lab))
    H, I, V, A = [], [], [], []
    n = 0
    for ch in _reader(path, ["HADM_ID", "ITEMID", "VALUENUM", "FLAG"], chunksize):
        n += len(ch)
        ch = ch[ch["ITEMID"].isin(ids) & ch["HADM_ID"].notna() & ch["VALUENUM"].notna()]
        H.append(ch["HADM_ID"].to_numpy(np.int64))
        I.append(ch["ITEMID"].to_numpy(np.int32))
        V.append(ch["VALUENUM"].to_numpy(np.float32))
        A.append(ch["FLAG"].astype(str).str.lower().str.startswith("abnormal").to_numpy())
        if len(H) % 5 == 0:
            log(f"  LABEVENTS rows read {n:,}")
    if not H:
        return {}, {"labevents_rows": n}
    H, I, V, A = map(np.concatenate, (H, I, V, A))
    labs = pd_map(I, item2lab)

    ref = {lab: float(np.median(V[(labs == lab) & ~A])) if np.any((labs == lab) & ~A) else float(np.median(V[labs == lab]))
           for lab in np.unique(labs)}
    refv = np.array([ref[l] for l in labs], dtype=np.float32)
    hi, lo = A & (V > refv), A & (V < refv)
    import pandas as pd
    g = pd.DataFrame({"h": H, "l": labs, "n": 1, "hi": hi.astype(np.int32), "lo": lo.astype(np.int32)}) \
        .groupby(["h", "l"], sort=False)[["n", "hi", "lo"]].sum()
    out = defaultdict(list)
    for (h, l), r in zip(g.index, g.itertuples(index=False)):
        _, _, f_hi, f_lo = LABS[l]
        if f_hi and r.hi / r.n >= frac:
            out[int(h)].append("L_" + f_hi)
        if f_lo and r.lo / r.n >= frac:
            out[int(h)].append("L_" + f_lo)
    st = {"labevents_rows": n, "lab_rows_used": int(len(H)), "lab_reference_medians": ref,
          "admissions_with_labs": int(g.index.get_level_values(0).nunique())}
    return out, st

def read_vitals(mimic, chunksize, frac):
    pre = read_precomputed(mimic, "VITAL_FINDINGS.csv")
    if pre is not None:
        return pre, {"vital_source": "VITAL_FINDINGS.csv (BigQuery)", "admissions_with_vital_findings": len(pre)}
    path = _find(mimic, "CHARTEVENTS.csv")
    if not path:
        log("WARNING: CHARTEVENTS not found -> no vital-sign findings")
        return {}, {}
    item2v = {i: v for v, spec in VITALS.items() for i in spec[0]}
    ids = np.array(sorted(item2v))
    parts, n = [], 0
    for ch in _reader(path, ["HADM_ID", "ITEMID", "VALUENUM", "ERROR"], chunksize):
        n += len(ch)
        ch = ch[ch["ITEMID"].isin(ids) & ch["HADM_ID"].notna() & ch["VALUENUM"].notna()]
        if "ERROR" in ch:
            ch = ch[ch["ERROR"].fillna(0) != 1]
        if not len(ch):
            continue
        val = ch["VALUENUM"].to_numpy(np.float32).copy()
        it = ch["ITEMID"].to_numpy()
        f = np.isin(it, list(TEMP_F))
        val[f] = (val[f] - 32) / 1.8
        vit = pd_map(it, item2v)
        rows = {"h": ch["HADM_ID"].to_numpy(np.int64), "v": vit, "n": np.ones(len(val), np.int32),
                "hi": np.zeros(len(val), np.int32), "lo": np.zeros(len(val), np.int32)}
        ok = np.ones(len(val), bool)
        for v, (_, (a, b), hi, lo) in VITALS.items():
            m = vit == v
            ok[m] = (val[m] > a) & (val[m] <= b)
            if hi:
                rows["hi"][m] = val[m] > hi[0]
            if lo:
                rows["lo"][m] = val[m] < lo[0]
        import pandas as pd
        df = pd.DataFrame(rows)[ok]
        parts.append(df.groupby(["h", "v"], sort=False)[["n", "hi", "lo"]].sum())
        if len(parts) % 10 == 0:
            log(f"  CHARTEVENTS rows read {n:,}")
            parts = [pd.concat(parts).groupby(level=[0, 1]).sum()]
    if not parts:
        return {}, {"chartevents_rows": n}
    import pandas as pd
    g = pd.concat(parts).groupby(level=[0, 1]).sum()
    out = defaultdict(list)
    for (h, v), r in zip(g.index, g.itertuples(index=False)):
        _, _, hi, lo = VITALS[v]
        if hi and r.hi / r.n >= frac:
            out[int(h)].append("V_" + hi[1])
        if lo and r.lo / r.n >= frac:
            out[int(h)].append("V_" + lo[1])
    st = {"chartevents_rows": n, "vital_rows_used": int(g["n"].sum()),
          "admissions_with_vitals": int(g.index.get_level_values(0).nunique())}
    return out, st

def cmd_measurements(a):
    import time
    t0 = time.time()
    labs, s1 = read_labs(a.mimic, a.chunksize, a.frac) if "labs" in a.kinds else ({}, {})
    vit, s2 = read_vitals(a.mimic, a.chunksize, a.frac) if "vitals" in a.kinds else ({}, {})
    out = defaultdict(list)
    for d in (labs, vit):
        for h, t in d.items():
            out[int(h)].extend(sorted(set(t)))
    cnt = Counter(t for ts in out.values() for t in ts)
    st = {**s1, **s2, "admissions_with_findings": len(out), "frac": a.frac, "kinds": a.kinds,
          "finding_counts": dict(cnt.most_common()), "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(dict(out), f, protocol=4)
    json.dump(st, open(os.path.splitext(a.out)[0] + ".stats.json", "w"), indent=1)
    log(f"measurements: {len(out):,} admissions with findings, {len(cnt)} finding types, "
        f"{st['seconds']}s -> {a.out}")
    return dict(out)

BQ = "`physionet-data.mimiciii_clinical.{}`"

def _case(col, mapping):
    by = defaultdict(list)
    for k, v in mapping.items():
        by[v].append(str(k))
    return "CASE " + " ".join(f"WHEN {col} IN ({', '.join(sorted(ks))}) THEN '{v}'" for v, ks in by.items()) + " END"

def sql_labs(frac=0.3):
    item2lab = {i: lab for lab, (ids, _, _, _) in LABS.items() for i in ids}
    hi = " UNION ALL ".join(f"SELECT '{l}' AS lab, '{s[2]}' AS f" for l, s in LABS.items() if s[2])
    lo = " UNION ALL ".join(f"SELECT '{l}' AS lab, '{s[3]}' AS f" for l, s in LABS.items() if s[3])
    return f"""-- LAB_FINDINGS.csv: abnormal laboratory findings per admission (same rules as measurements.py)
WITH l AS (
  SELECT hadm_id, {_case('itemid', item2lab)} AS lab, valuenum AS val,
         STARTS_WITH(LOWER(IFNULL(flag, '')), 'abnormal') AS abn
  FROM {BQ.format('labevents')}
  WHERE itemid IN ({', '.join(map(str, sorted(item2lab)))})
    AND hadm_id IS NOT NULL AND valuenum IS NOT NULL
),
ref AS (  -- median of each test's NORMAL results: abnormal values above it = high, below = low
  SELECT lab, APPROX_QUANTILES(IF(abn, NULL, val), 2 IGNORE NULLS)[OFFSET(1)] AS med FROM l GROUP BY lab
),
agg AS (
  SELECT l.hadm_id, l.lab, COUNT(*) AS n, COUNTIF(abn AND val > med) AS hi, COUNTIF(abn AND val < med) AS lo
  FROM l JOIN ref USING (lab) GROUP BY 1, 2
),
hi_map AS ({hi}),
lo_map AS ({lo})
SELECT hadm_id AS HADM_ID, STRING_AGG(f, ';' ORDER BY f) AS FINDINGS FROM (
  SELECT agg.hadm_id, hi_map.f FROM agg JOIN hi_map USING (lab) WHERE agg.hi / agg.n >= {frac}
  UNION ALL
  SELECT agg.hadm_id, lo_map.f FROM agg JOIN lo_map USING (lab) WHERE agg.lo / agg.n >= {frac}
) GROUP BY 1 ORDER BY 1
"""

def sql_vitals(frac=0.3):
    item2v = {i: v for v, spec in VITALS.items() for i in spec[0]}
    rng = " OR ".join(f"(vital = '{v}' AND val > {s[1][0]} AND val <= {s[1][1]})" for v, s in VITALS.items())
    hic = " OR ".join(f"(vital = '{v}' AND val > {s[2][0]})" for v, s in VITALS.items() if s[2])
    loc = " OR ".join(f"(vital = '{v}' AND val < {s[3][0]})" for v, s in VITALS.items() if s[3])
    hif = " ".join(f"WHEN '{v}' THEN '{s[2][1]}'" for v, s in VITALS.items() if s[2])
    lof = " ".join(f"WHEN '{v}' THEN '{s[3][1]}'" for v, s in VITALS.items() if s[3])
    hiv = ", ".join(f"'{v}'" for v, s in VITALS.items() if s[2])
    lov = ", ".join(f"'{v}'" for v, s in VITALS.items() if s[3])
    return f"""-- VITAL_FINDINGS.csv: abnormal vital-sign findings per admission (same rules as measurements.py)
-- replaces downloading CHARTEVENTS (~4 GB); the result is a few MB
WITH v AS (
  SELECT hadm_id, {_case('itemid', item2v)} AS vital,
         IF(itemid IN ({', '.join(map(str, sorted(TEMP_F)))}), (valuenum - 32) / 1.8, valuenum) AS val
  FROM {BQ.format('chartevents')}
  WHERE itemid IN ({', '.join(map(str, sorted(item2v)))})
    AND hadm_id IS NOT NULL AND valuenum IS NOT NULL AND (error IS NULL OR error != 1)
),
agg AS (
  SELECT hadm_id, vital, COUNT(*) AS n,
         COUNTIF({hic}) AS hi,
         COUNTIF({loc}) AS lo
  FROM v WHERE {rng}
  GROUP BY 1, 2
)
SELECT hadm_id AS HADM_ID, STRING_AGG(f, ';' ORDER BY f) AS FINDINGS FROM (
  SELECT hadm_id, CASE vital {hif} END AS f FROM agg WHERE vital IN ({hiv}) AND hi / n >= {frac}
  UNION ALL
  SELECT hadm_id, CASE vital {lof} END AS f FROM agg WHERE vital IN ({lov}) AND lo / n >= {frac}
) GROUP BY 1 ORDER BY 1
"""

def load(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

def add_args(p):
    p.add_argument("--mimic", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--kinds", default="labs,vitals")
    p.add_argument("--frac", type=float, default=0.3,
                   help="share of a stay's readings that must be abnormal to record the finding")
    p.add_argument("--chunksize", type=int, default=2_000_000)
