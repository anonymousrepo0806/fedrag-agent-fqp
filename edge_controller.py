"""
edge_controller.py -- one object per client/"edge" (hospital) that owns
everything about that edge's own training progress: how long each round
took locally, its running loss/validation/test history, and which round
was its best or where it converged.

Both run modes build one EdgeController per client:
  - fedrag_sim.py (single process, all edges simulated in-memory): one
    EdgeController per simulated client inside cmd_train's round loop.
  - fl_net.py (real multi-process/Docker, one container per edge): the
    EdgeController living inside that container's own cmd_client process.

Either way the edge only ever reports its own round timing/loss through
this object; aggregation across edges (FedAvg etc.) and the
convergence/efficiency math that needs ALL edges' histories together stays
where it already lives, in fedrag_sim.compute_training_efficiency, which
EdgeController.efficiency_summary() below just forwards to for a single
edge's own slice of that computation.
"""

from __future__ import annotations

import time

class EdgeController:
    def __init__(self, edge_id, tag=""):
        self.edge_id = edge_id
        self.tag = tag
        self.history = []
        self._round_t0 = None

    def start_round(self, round_no):
        self._round_t0 = time.time()
        return self

    def end_round(self, round_no, loss=None, val_score=None, test_score=None, extra=None):
        if self._round_t0 is None:
            raise RuntimeError(f"edge {self.edge_id}: end_round called without start_round")
        train_secs = round(time.time() - self._round_t0, 3)
        self._round_t0 = None
        rec = {"round": round_no, "train_secs": train_secs, "loss": loss,
               "val_score": val_score, "test_score": test_score, "extra": extra or {}}
        self.history.append(rec)
        return train_secs

    @property
    def total_local_train_secs(self):
        return round(sum(r["train_secs"] for r in self.history), 3)

    @property
    def cumulative_train_secs_by_round(self):
        out, running = [], 0.0
        for r in self.history:
            running += r["train_secs"]
            out.append(round(running, 3))
        return out

    def best_round(self):
        scored = [r for r in self.history if r["val_score"] is not None]
        if not scored:
            return None
        return max(scored, key=lambda r: r["val_score"])["round"]

    def efficiency_summary(self, patience=3, tol=1e-3):
        """This edge's own view of compute_training_efficiency, using only
        this edge's local train time per round (not the whole federation's
        wall-clock, which only the coordinator/server can see)."""
        import fedrag_sim as F
        round_wall_s = [r["train_secs"] for r in self.history]
        val_scores = [r["val_score"] for r in self.history]
        test_scores = [r["test_score"] for r in self.history]
        return F.compute_training_efficiency(round_wall_s, [[s] for s in round_wall_s],
                                             val_scores, test_scores, patience=patience, tol=tol)

    def as_dict(self):
        return {"edge_id": self.edge_id, "tag": self.tag, "n_rounds": len(self.history),
                "total_local_train_secs": self.total_local_train_secs,
                "cumulative_train_secs_by_round": self.cumulative_train_secs_by_round,
                "best_round": self.best_round(), "history": self.history}

class EdgeFleet:
    """A coordinator-side registry of every edge's EdgeController, so the
    server/simulation driver can pull everyone's per-round train time into
    one `per_round_client_secs` matrix without each call site re-deriving
    client ordering itself."""

    def __init__(self):
        self.edges = {}

    def get(self, edge_id, tag=""):
        if edge_id not in self.edges:
            self.edges[edge_id] = EdgeController(edge_id, tag=tag)
        return self.edges[edge_id]

    def per_round_client_secs(self, n_rounds):
        ids = sorted(self.edges)
        out = []
        for r in range(n_rounds):
            out.append([self.edges[i].history[r]["train_secs"] if r < len(self.edges[i].history) else 0.0
                       for i in ids])
        return out

    def total_local_train_secs_all_edges(self):
        return round(sum(e.total_local_train_secs for e in self.edges.values()), 3)
