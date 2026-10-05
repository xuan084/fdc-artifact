"""Lock-v6 addendum: raw-data binding (new file; v5-frozen ``envs/pool_replay.py`` is not modified).

The v5 replay env reads ``$DATA_DIR/criteo_uplift_real/tidy.pkl`` without checking a
hash.  The v6 layer therefore freezes the sha256 of every raw data file a v6 task depends on and verifies it
(i) when the addendum is checked (eval start, resume = every start, analysis), and (ii) in the runner before any env
is built; the per-layer combined data hash is written into every result row and summary and resumed rows with a
different data hash are discarded.

DATA_FILES: logical name -> absolute path.  LAYER_DATA: which files each layer depends on (Criteo: the tidy pickle
actually read + its source csv.gz; Lenta: the tidy pickle).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

__all__ = ["DATA_FILES", "LAYER_DATA", "file_sha256", "data_hashes", "layer_data_sha", "data_drift"]

_ROOT = Path(__import__("os").environ.get("DATA_DIR", "data"))
DATA_FILES = {
    "criteo_tidy": str(_ROOT / "criteo_uplift_real" / "tidy.pkl"),
    "criteo_gz": str(_ROOT / "criteo_uplift_real" / "criteo-research-uplift-v2.1.csv.gz"),
    "lenta_tidy": str(_ROOT / "lenta" / "tidy.pkl"),
}
LAYER_DATA = {"CR9": ("criteo_tidy", "criteo_gz"), "LR9": ("lenta_tidy",)}
_CACHE: dict = {}


def file_sha256(path, chunk=1 << 22):
    """sha256 of a file; cached per process keyed by (path, size, mtime_ns) so repeated checks are cheap."""
    st = os.stat(path)
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        _CACHE[key] = h.hexdigest()
    return _CACHE[key]


def data_hashes(names=None):
    names = list(DATA_FILES) if names is None else list(names)
    return {n: {"path": DATA_FILES[n], "sha256": file_sha256(DATA_FILES[n])} for n in names}


def layer_data_sha(layer, frozen=None):
    """Combined hash of the layer's data files (current files; if ``frozen`` is given, they must match it)."""
    names = LAYER_DATA[layer]
    cur = {n: file_sha256(DATA_FILES[n]) for n in names}
    if frozen is not None:
        for n in names:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} ({DATA_FILES[n]}) sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()


def data_drift(frozen: dict) -> list[str]:
    bad = []
    for n, d in sorted((frozen or {}).items()):
        p = d.get("path")
        if not p or not Path(p).exists():
            bad.append(f"{n} (missing)")
        elif file_sha256(p) != d.get("sha256"):
            bad.append(n)
    return bad
