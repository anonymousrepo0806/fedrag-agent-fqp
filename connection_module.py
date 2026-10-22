"""
connection_module.py -- the one place that knows how an edge (client) talks
to the coordinator (server), so an architecture diagram has a single
"connection" box instead of HTTP calls scattered through fl_net.py.

Two implementations share the same call(method, path, data) surface:
  - HTTPConnectionModule : real process-to-process transport (fl_net.py's
    Docker/multi-process mode) -- one retrying HTTP client per edge,
    talking to the coordinator's http.server.
  - LocalConnectionModule : the in-process equivalent for fedrag_sim.py's
    single-process simulation, where "sending" a message is just a direct
    function call because client and server share memory. Formalizing it
    here means a future multi-process version of the simulation (or a
    notebook that wants to swap transports) only has to swap which
    ConnectionModule it builds, not rewrite the training loop.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

class ConnectionModule:
    """Common interface: call(method, path, data) -> raw response bytes."""

    def call(self, method, path, data=None, ctype="application/octet-stream", retries=900):
        raise NotImplementedError

    def send_update(self, round_no, client_id, n, loss, blob):
        return self.call("POST", f"/update?round={round_no}&client={client_id}&n={n}&loss={loss}", blob)

    def get_model(self, after_round):
        return self.call("GET", f"/model?after_round={after_round}")

    def send_eval(self, round_no, client_id, payload):
        return self.call("POST", f"/eval?round={round_no}&client={client_id}", payload)

class HTTPConnectionModule(ConnectionModule):
    """Retrying HTTP client used by an edge's own process to reach the
    coordinator process over the network -- fl_net.py's real multi-machine
    (or one-container-per-edge) transport."""

    def __init__(self, base):
        self.base = base.rstrip("/")
        self.sent = self.recv = 0

    def call(self, method, path, data=None, ctype="application/octet-stream", retries=900):
        if isinstance(data, (dict, list)):
            data, ctype = json.dumps(data).encode(), "application/json"
        for i in range(retries):
            try:
                req = urllib.request.Request(self.base + path, data=data, method=method,
                                             headers={"Content-Type": ctype})
                with urllib.request.urlopen(req, timeout=300) as r:
                    out = r.read()
                    self.sent += len(data or b"")
                    self.recv += len(out)
                    if r.status == 202:
                        continue
                    return out
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                time.sleep(min(5, 0.5 + i * 0.1))
        raise RuntimeError(f"{method} {path} failed")

class LocalConnectionModule(ConnectionModule):
    """In-process stand-in for HTTPConnectionModule: fedrag_sim.py's single
    process already holds the global model state in memory (`gstate`), so
    there is no real network hop between an edge and the coordinator --
    this just wraps that direct read/write behind the same call() surface,
    for callers (or diagrams) that want one "connection" abstraction
    regardless of which run mode is active. `state` is a dict shared with
    the caller (e.g. the simulation's own gstate dict) that this object
    reads and writes through "sent"/"recv" byte counters for symmetry
    with HTTPConnectionModule's instrumentation."""

    def __init__(self, server):
        self.server = server
        self.sent = self.recv = 0

    def call(self, method, path, data=None, ctype="application/octet-stream", retries=900):
        if isinstance(data, (dict, list)):
            data = json.dumps(data).encode()
        payload_len = len(data) if data else 0
        out = self.server.handle(method, path, data)
        self.sent += payload_len
        self.recv += len(out) if out else 0
        return out
