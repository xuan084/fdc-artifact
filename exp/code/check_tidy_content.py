"""Content check of the tidy tables (version independent; complements the pinned pickle-byte hashes).

The replay loaders pin the SHA-256 of the pickle BYTES (``dsswm/envs/{data_v6,lenta_v6,hillstrom_v9}.py``).  Pickle
bytes depend on the pandas / numpy versions, so a faithful re-ingest (``ingest_tidy.py``) can differ in bytes.  This
script hashes the CONTENT instead:

    content_sha256 = sha256 over, in order:
        b"tidy-content-v1\\n"
        the row count and column count (decimal, "\\n"-terminated)
        the index: its kind ("range" with start/stop/step, else the int64 little-endian values)
        for every column in order: name (utf-8) "\\n", numpy dtype str (e.g. "<f4") "\\n",
                                   the column's values as contiguous little-endian bytes of that dtype

Only numeric / bool columns occur in the three tables; any other dtype is an error.  ``EXPECTED`` holds the values
computed from our pickles (the files whose byte hashes are pinned in the loaders and listed in the README).

Usage (cwd = exp/code):
    python check_tidy_content.py [--data-dir DIR] [--table criteo|lenta|hillstrom|all] [--compare-pickle A B]
Exit code 0 iff every checked table matches.  --compare-pickle additionally runs pandas.testing.assert_frame_equal
(check_exact=True) between two pickles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIRS = {"criteo": "criteo_uplift_real", "lenta": "lenta", "hillstrom": "hillstrom"}

# From our pickles (pickle-byte sha256 in brackets, as pinned by the loaders / listed in the README).
EXPECTED = {
    "criteo": {"rows": 13979592, "cols": 16,
               "content_sha256": "1d80a35baf4c53a42edeca581ee61249ff5250c6d004b3cf9e238ea4e88e2b93",
               "pickle_sha256": "600a9a57a0e93a90c659552f482609321d97c58b1c9c90a6a391e05811546cd8"},
    "lenta": {"rows": 687029, "cols": 14,
              "content_sha256": "6e5f8246ef77f0aa20550e83ddf79d66e0b66f9ceeab67dbc2cc08bfbce472a8",
              "pickle_sha256": "6d075bbc306df3512f6bc3a795c94686984b4c99da3889c91778d25325565d43"},
    "hillstrom": {"rows": 64000, "cols": 13,
                  "content_sha256": "feb6b27af175ab59834f185bfb010d26683fcae293660d6006bb9a9536c444a9",
                  "pickle_sha256": "3dab9ed72bfdcaf51f6d993975a7ca052ef9355690489ca854217b7f0a68b296"},
}


def content_sha256(df: pd.DataFrame) -> str:
    h = hashlib.sha256(b"tidy-content-v1\n")
    h.update(f"{df.shape[0]}\n{df.shape[1]}\n".encode())
    idx = df.index
    if isinstance(idx, pd.RangeIndex):
        h.update(f"range {idx.start} {idx.stop} {idx.step}\n".encode())
    else:
        h.update(b"values\n")
        h.update(np.ascontiguousarray(idx.to_numpy().astype("<i8")).tobytes())
    for c in df.columns:
        v = df[c].to_numpy()
        if v.dtype.kind not in "biuf":
            raise TypeError(f"column {c!r} has non-numeric dtype {v.dtype}")
        v = np.ascontiguousarray(v.astype(v.dtype.newbyteorder("<"), copy=False))
        h.update(f"{c}\n{v.dtype.str}\n".encode())
        h.update(v.tobytes())
    return h.hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(table: str, path: Path) -> dict:
    df = pd.read_pickle(path)
    exp = EXPECTED[table]
    out = {"table": table, "path": path.name, "rows": int(df.shape[0]), "cols": int(df.shape[1]),
           "content_sha256": content_sha256(df), "pickle_sha256": sha256_file(path),
           "dtypes": {c: str(t) for c, t in df.dtypes.items()}}
    out["pickle_bytes_match"] = out["pickle_sha256"] == exp["pickle_sha256"]
    out["content_match"] = (exp["content_sha256"] is not None and out["content_sha256"] == exp["content_sha256"]
                            and out["rows"] == exp["rows"] and out["cols"] == exp["cols"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    ap.add_argument("--table", choices=("criteo", "lenta", "hillstrom", "all"), default="all")
    ap.add_argument("--compare-pickle", nargs=2, metavar=("A", "B"))
    a = ap.parse_args()
    ok = True
    if a.compare_pickle:
        x, y = (pd.read_pickle(p) for p in a.compare_pickle)
        try:
            pd.testing.assert_frame_equal(x, y, check_exact=True)
            print("assert_frame_equal: equal")
        except AssertionError as e:
            ok = False
            print(f"assert_frame_equal: DIFFERENT\n{e}")
    tables = ("criteo", "lenta", "hillstrom") if a.table == "all" else (a.table,)
    for t in tables:
        p = Path(a.data_dir) / DIRS[t] / "tidy.pkl"
        if not p.exists():
            print(f"{t}: {p} missing")
            ok = False
            continue
        r = check(t, p)
        ok &= r["content_match"]
        print(json.dumps({k: r[k] for k in ("table", "rows", "cols", "content_match", "pickle_bytes_match",
                                            "content_sha256")}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
