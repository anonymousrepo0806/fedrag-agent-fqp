"""
hypo_learn.py -- LEARNED, federated drug-hypothesis ranking
===========================================================

Candidate generation stays the knowledge-graph search of HypothesisEngine
(evidence paths from the patient's conditions to drugs, contraindicated
drugs removed). What is learned is HOW to rank those candidates.

Supervision (no manual labels): for a training patient, a candidate drug is
positive if it was actually prescribed at the NEXT admission.

Features per (patient, candidate drug):
  rule_score (original hand-weighted score), log #paths, log #patient
  conditions reaching the drug, best path score, min / mean path length,
  per-relation path counts (how the evidence reaches the drug), #interactions
  with current drugs, log local popularity of the drug at next admissions,
  the patient's predicted risk, and the federated KG-completion
  plausibility of (drug, may_treat, condition) when that model is trained.

Federated training: feature standardisation from federated sums (n, sum,
sum of squares -- no data), then FedAvg rounds of a linear (default,
interpretable) or small MLP ranker trained with class-weighted BCE.

Evaluation per test patient: P@k, Hit@k, nDCG@k, MRR for the learned
ranker vs the rule ranking vs a popularity baseline, plus candidate recall
(was any next-admission drug among the candidates at all = ranking ceiling).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

BASE = ["rule_score", "log_paths", "log_sources", "best_path", "min_len", "mean_len", "n_inter"]

EXTRA = ["log_pop", "risk", "kgc"]

def feature_names(rel_names):
    return BASE + [f"rel:{r}" for r in rel_names] + EXTRA

def kgc_scorer(kg):
    """(condition CUIs, drug CUI) -> sigmoid(max_c DistMult(drug, may_treat, c)), 0.5 if unavailable"""
    m = getattr(kg, "kgc_model", None)
    if m is None or "may_treat" not in kg.kgc["rels"]:
        return lambda conds, drug: 0.5
    import torch
    r = kg.kgc["rels"].index("may_treat")
    e2i = m._e2i

    def f(conds, drug):
        d = e2i.get(drug)
        cs = [e2i[c] for c in conds if c in e2i]
        if d is None or not cs:
            return 0.5
        with torch.no_grad():
            s = (m.E[d] * m.R[r] * m.E[cs]).sum(-1)
        return float(torch.sigmoid(s.max()))
    return f

def patient_candidates(engine, kg, p, risk, pop, kgc_f, max_cands, flat, n_pop=20, pop_list=None, stats=None):
    """-> candidates (KG candidates in rule order, then popular local drugs
    without KG evidence = hybrid candidate set), feature matrix, labels (or
    None), current drugs, next-admission drugs"""
    fl = flat(p)

    conds = [kg.code2cui[c] for c in fl if c[:2] in ("D_", "L_", "V_") and c in kg.code2cui]
    drugs = [kg.code2cui[c] for c in fl if c.startswith("M_") and c in kg.code2cui]
    hy = engine.generate(conds, drugs, max_cands, stats=stats, feats=True)
    for h in hy:
        h["source"] = "kg"
    if n_pop and pop_list:
        have, cur = {h["drug"] for h in hy}, set(drugs)
        nf = len(BASE) - 1 + len(engine.h["rel_names"])
        for d in [d for d in pop_list if d not in have and d not in cur][:n_pop]:
            hy.append({"drug": d, "atc": "", "score": 0.0, "interactions": [], "n_paths": 0, "path": [],
                       "relations": [], "source": "popularity", "f": [0.0] * (nf + 1)})
    nxt = {kg.code2cui[c] for c in p.get("next", []) if c.startswith("M_") and c in kg.code2cui}
    dim = len(feature_names(engine.h["rel_names"]))
    X = (np.array([h["f"] + [math.log1p(pop.get(h["drug"], 0)), float(risk), kgc_f(conds, h["drug"])]
                   for h in hy], dtype=np.float32) if hy else np.zeros((0, dim), dtype=np.float32))
    y = np.array([float(h["drug"] in nxt) for h in hy], dtype=np.float32) if nxt else None
    return hy, X, y, drugs, nxt

def make_ranker(dim, kind="linear", seed=1):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 1))
    return nn.Linear(dim, 1)

def _state(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

def local_fit(model, g, X, y, mean, std, epochs=1, lr=0.01, bs=512, seed=1):
    import torch
    import torch.nn.functional as Fn
    model.load_state_dict(g)
    if len(X) == 0:
        return _state(model), None
    Xt = torch.from_numpy((X - mean) / std)
    yt = torch.from_numpy(y)
    pw = torch.tensor([(len(y) - y.sum()) / max(1.0, y.sum())])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    losses = []
    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(X))
        for i in range(0, len(idx), bs):
            b = torch.from_numpy(idx[i:i + bs])
            loss = Fn.binary_cross_entropy_with_logits(model(Xt[b]).squeeze(-1), yt[b], pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    return _state(model), float(np.mean(losses))

def fed_norm(stats):
    """stats: list of (n, sum, sumsq) -> global mean, std (federated, no data)"""
    n = sum(s[0] for s in stats)
    if n == 0:
        return None, None
    s1 = sum(s[1] for s in stats)
    s2 = sum(s[2] for s in stats)
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)

def local_stats(X):
    return (len(X), X.sum(0).astype(np.float64), (X.astype(np.float64) ** 2).sum(0)) if len(X) else (0, 0.0, 0.0)

def fedavg(states, weights):
    W = float(sum(weights))
    return {k: sum(s[k] * (w / W) for s, w in zip(states, weights)) for k in states[0]}

def train_federated(client_data, dim, rounds=20, lr=0.01, kind="linear", seed=1, mode="fedavg"):
    """client_data: list of (X, y). mode: fedavg | local | central.
    Returns {"models": [state per client] or single, "mean", "std", "history"}."""
    mean, std = fed_norm([local_stats(X) for X, _ in client_data])
    if mean is None:
        return None
    model = make_ranker(dim, kind, seed)
    g = _state(model)
    hist = []
    if mode == "central":
        X = np.concatenate([X for X, _ in client_data])
        y = np.concatenate([y for _, y in client_data])
        for r in range(rounds):
            g, l = local_fit(model, g, X, y, mean, std, seed=seed + r, lr=lr)
            hist.append(l)
        return {"global": g, "mean": mean, "std": std, "history": hist, "kind": kind, "dim": dim}
    if mode == "local":
        locs = []
        for ci, (X, y) in enumerate(client_data):
            s = g
            for r in range(rounds):
                s, _ = local_fit(model, s, X, y, mean, std, seed=seed + 31 * ci + r, lr=lr)
            locs.append(s)
        return {"local": locs, "mean": mean, "std": std, "history": hist, "kind": kind, "dim": dim}
    for r in range(rounds):
        states, ws, ls = [], [], []
        for ci, (X, y) in enumerate(client_data):
            if len(X) == 0:
                continue
            s, l = local_fit(model, g, X, y, mean, std, seed=seed + 31 * ci + r, lr=lr)
            states.append(s)
            ws.append(len(X))
            ls.append(l)
        if states:
            g = fedavg(states, ws)
        hist.append(float(np.mean([x for x in ls if x is not None])) if ls else None)
    return {"global": g, "mean": mean, "std": std, "history": hist, "kind": kind, "dim": dim}

def scorer(trained, ci=0):
    import torch
    model = make_ranker(trained["dim"], trained["kind"])
    model.load_state_dict(trained["local"][ci] if "local" in trained else trained["global"])
    model.eval()
    mean, std = trained["mean"], trained["std"]

    def f(X):
        if len(X) == 0:
            return np.zeros(0)
        with torch.no_grad():
            return model(torch.from_numpy((X - mean) / std)).squeeze(-1).numpy()
    return f

def weights_report(trained, names):
    """standardised linear weights = which evidence predicts real prescriptions"""
    if trained.get("kind") != "linear" or "global" not in trained:
        return None
    w = trained["global"]["weight"].reshape(-1).tolist()
    return dict(sorted(zip(names, [round(x, 4) for x in w]), key=lambda kv: -abs(kv[1])))

def rank_metrics(ranked, rel, k):
    hits = [1.0 if d in rel else 0.0 for d in ranked]
    top = hits[:k]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(top))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {f"P@{k}": sum(top) / k, f"Hit@{k}": float(any(top)), f"nDCG@{k}": dcg / idcg if idcg else 0.0,
            "MRR": 1.0 / first if first else 0.0}
