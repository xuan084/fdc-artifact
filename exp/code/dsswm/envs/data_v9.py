"""Lock-v9 addendum: raw-data binding (new file; data_v6 / data_v8 untouched).

DATA_FILES_V9 = DATA_FILES_V8 (Criteo, Lenta, the blinded X5 RetailHero files + PROVENANCE.json) plus the Hillstrom
tidy.pkl (block C).  LAYER_DATA_V9 names the files each v9 layer depends on.  Hashing reads bytes only.
"""
from __future__ import annotations

import hashlib
import json

from .data_v6 import data_drift, file_sha256
from .data_v8 import DATA_FILES_V8, LAYER_DATA_V8, X5_ROOT, x5_provenance_check
from .hillstrom_v9 import TIDY, TIDY_SHA256

__all__ = ["DATA_FILES_V9", "LAYER_DATA_V9", "X5_ROOT", "data_hashes_v9", "layer_data_sha_v9", "data_drift",
           "x5_provenance_check"]

DATA_FILES_V9 = dict(DATA_FILES_V8)
DATA_FILES_V9["hillstrom_tidy"] = str(TIDY)
LAYER_DATA_V9 = dict(LAYER_DATA_V8)
LAYER_DATA_V9.update({"HCZ6": ("hillstrom_tidy",), "HCR6": ("hillstrom_tidy",)})


def data_hashes_v9(names=None):
    names = list(DATA_FILES_V9) if names is None else list(names)
    out = {n: {"path": DATA_FILES_V9[n], "sha256": file_sha256(DATA_FILES_V9[n])} for n in names}
    if "hillstrom_tidy" in out and out["hillstrom_tidy"]["sha256"] != TIDY_SHA256:
        raise RuntimeError("Hillstrom tidy.pkl differs from its recorded sha256")
    return out


def layer_data_sha_v9(layer, frozen=None):
    names = LAYER_DATA_V9[layer]
    cur = {n: file_sha256(DATA_FILES_V9[n]) for n in names}
    if frozen is not None:
        for n in names:
            want = (frozen.get(n) or {}).get("sha256")
            if want != cur[n]:
                raise RuntimeError(f"raw data {n} ({DATA_FILES_V9[n]}) sha256 {cur[n][:16]} != frozen {str(want)[:16]}")
    return hashlib.sha256(json.dumps(cur, sort_keys=True).encode()).hexdigest()

