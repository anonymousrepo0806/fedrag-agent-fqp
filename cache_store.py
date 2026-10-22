"""
cache_store.py -- one place that owns every on-disk cache the project writes,
so an architecture diagram has a single "cache" box instead of scattered
pickle.dump calls in sparql_ext.py, fqp.py and fedrag_sim.py.

A cache entry is identified by a namespace ("sparql_ext", "fqp_bloom",
"kgc_model", ...) and a key (usually a content hash or a run tag). Callers
never see a file path; CacheStore turns (namespace, key) into one under
its root directory and does the pickling.

Used today by sparql_ext.py's build_cache (the external-FAIR-source
adjacency cache) and exposed for fqp.py / the KGC model / any future
federated-query cache to adopt the same way without inventing a new
on-disk layout each time.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time

class CacheStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _dir(self, namespace):
        d = os.path.join(self.root, namespace)
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, namespace, key):
        return os.path.join(self._dir(namespace), self._safe_key(key) + ".pkl")

    @staticmethod
    def _safe_key(key):
        if all(c.isalnum() or c in "-._" for c in key) and len(key) <= 120:
            return key
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def exists(self, namespace, key):
        return os.path.exists(self._path(namespace, key))

    def save(self, namespace, key, obj, meta=None):
        path = self._path(namespace, key)
        payload = {"namespace": namespace, "key": key, "saved_at": time.time(),
                   "meta": meta or {}, "data": obj}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        os.replace(tmp, path)
        return path

    def load(self, namespace, key, default=None):
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return default
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return payload["data"]

    def load_raw_path(self, namespace, key):
        """The data this entry would hold, saved directly with pickle.dump
        (no namespace/meta envelope) -- for caches written in the plain
        pickle format sparql_ext.py's build_cache used before CacheStore
        existed, so old cache files on disk keep loading unchanged."""
        return self._path(namespace, key)

    def save_plain(self, namespace, key, obj):
        """Like save(), but writes obj directly with no envelope -- for
        caches other tools (sparql_ext.py's CLI, a notebook, a user script)
        load back with a plain pickle.load rather than through CacheStore."""
        path = self._path(namespace, key)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, path)
        return path

    def load_plain(self, namespace, key, default=None):
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return default
        with open(path, "rb") as f:
            return pickle.load(f)

    def delete(self, namespace, key):
        path = self._path(namespace, key)
        if os.path.exists(path):
            os.remove(path)

    def list_keys(self, namespace):
        d = self._dir(namespace)
        return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".pkl"))

    def stats(self):
        out = {}
        if not os.path.isdir(self.root):
            return out
        for namespace in sorted(os.listdir(self.root)):
            d = os.path.join(self.root, namespace)
            if not os.path.isdir(d):
                continue
            files = [f for f in os.listdir(d) if f.endswith(".pkl")]
            out[namespace] = {"entries": len(files),
                               "bytes": sum(os.path.getsize(os.path.join(d, f)) for f in files)}
        return out

class FileCacheHandle:
    """A CacheStore entry addressed by an explicit file path instead of a
    (namespace, key) pair -- lets sparql_ext.py's CLI (`--out some/path.pkl`,
    chosen freely by the caller) use the same save/load/exists surface as
    the rest of the project without forcing every caller onto CacheStore's
    own directory layout."""

    def __init__(self, path):
        self.path = path

    def exists(self):
        return os.path.exists(self.path)

    def save_plain(self, obj):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, self.path)
        return self.path

    def load_plain(self, default=None):
        if not os.path.exists(self.path):
            return default
        with open(self.path, "rb") as f:
            return pickle.load(f)
