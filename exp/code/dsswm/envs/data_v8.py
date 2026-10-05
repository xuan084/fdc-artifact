"""Lock-v8 addendum: raw-data binding (new file; ``data_v6`` is untouched).

DATA_FILES_V8 = the v6/v7 Criteo + Lenta files plus the three blinded X5 RetailHero files written by
``ingest_v8_fresh.py`` (dev.pkl, eval_labels.pkl, eval_outcome.npy).  LAYER_DATA_V8 names the files each v8 layer
depends on.  Hashing a file reads its bytes only; no statistic of the X5 eval outcome is computed here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data_v6 import DATA_FILES, data_drift, file_sha256

__all__ = ["DATA_FILES_V8", "LAYER_DATA_V8", "X5_ROOT", "data_hashes_v8", "layer_data_sha_v8", "data_drift",
           "x5_provenance_check"]

X5_ROOT = (Path(__import__("os").environ.get("DATA_DIR", "data")) / "x5_retailhero")
DATA_FILES_V8 = dict(DATA_FILES)
DATA_FILES_V8.update({
    "x5_dev": str(X5_ROOT / "dev.pkl"),
    "x5_eval_labels": str(X5_ROOT / "eval_labels.pkl"),
    "x5_eval_outcome": str(X5_ROOT / "eval_outcome.npy"),
    "x5_provenance": str(X5_ROOT / "PROVENANCE.json"),       # external reviewer v8 review item 5: provenance metadata bound
})
LAYER_DATA_V8 = {"CR9": ("criteo_tidy", "criteo_gz"), "CR12": ("criteo_tidy", "criteo_gz"),
                 "X9": ("x5_dev", "x5_eval_labels", "x5_eval_outcome", "x5_provenance")}
_X5_PROV = {"x5_dev": "dev.pkl", "x5_eval_labels": "eval_labels.pkl", "x5_eval_outcome": "eval_outcome.npy"}


def data_hashes_v8(names=None):
    names = list(DATA_FILES_V8) if names is None else list(names)
    return {n: {"path": DATA_FILES_V8[n], "sha256": file_sha256(DATA_FILES_V8[n])} for n in names}


def layer_data_sha_v8(layer, frozen=None):
    names = LAYER_DATA_V8[layer]
    cur = {n: file_sha256(DATA_FILES_V8[n]) for n in names}
    if frozen is not None:
        for n in names:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} ({DATA_FILES_V8[n]}) sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()


def x5_provenance_check():
    """The three X5 files must match the sha256 recorded by the blinded ingest (PROVENANCE.json)."""
    prov = json.loads((X5_ROOT / "PROVENANCE.json").read_text())["files"]
    bad = [n for n, f in _X5_PROV.items() if file_sha256(DATA_FILES_V8[n]) != prov.get(f)]
    if bad:
        raise RuntimeError(f"X5 files differ from PROVENANCE.json: {bad}")
    return prov
