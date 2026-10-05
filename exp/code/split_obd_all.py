"""Open Bandit Dataset random/all: the blinded salted 50/50 row split read by lock v11 (reconstruction of the split step).

The split of 2026-10-05 was made by a one-off command whose code was not kept as a file; this script applies the
recorded rule (README "Data"; identical to ``split_obd_v12.py`` with the random/all salt and name) and checks the
outputs against the recorded SHA-256 values:

    row_id = the CSV's unnamed first column (must equal 0..1,374,326)
    dev iff int(sha256(f"dsswm-obd-2026-10-05|open_bandit|{row_id}")[:8], 16) / 2**32 < 0.5
    dev.pkl            all 90 columns of the dev rows ("Unnamed: 0" renamed to row_id), index reset
    eval_labels.pkl    every column except click, eval rows in original order, index reset
    eval_outcome.npy   click of the eval rows as int8, aligned with eval_labels.pkl

It also writes a minimal ``PROVENANCE.json`` ({"files": {...}} plus the rule).  Our original PROVENANCE.json is hash-
bound by lock v11 but is not shipped (it contains free text that identifies the authors' tooling); see the reproduction
runner's ``--provenance-by-files`` option and README "Reproduce end to end".

Usage (cwd = exp/code):  python split_obd_all.py [--zip PATH] [--out DIR]
    default zip  $DATA_DIR/open_bandit/raw/open_bandit_dataset.zip;  default out  $DATA_DIR/open_bandit
    (an existing dev.pkl / eval_labels.pkl / eval_outcome.npy in --out is never overwritten).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SALT, NAME, DEV_FRAC = "dsswm-obd-2026-10-05", "open_bandit", 0.5
MEMBER = "open_bandit_dataset/random/all/all.csv"
ZIP_SHA256 = "e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78"
EXPECTED = {"dev.pkl": "ac59e8b9b40ee29f026daf8b1057a455678d60fb4a5201e75ec6acd0157ee748",
            "eval_labels.pkl": "b27cd3b93d56e2a14ef6417f2f9d9623b5b125688578efa419aa22e7adaeb8af",
            "eval_outcome.npy": "cd0e23a339de3ed0aba6a14af1ab00f468a0c400966e0ad4c5b66dd190436ca8"}
N_DEV, N_EVAL = 686168, 688159


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def is_dev(keys):
    return np.array([int(hashlib.sha256(f"{SALT}|{NAME}|{k}".encode()).hexdigest()[:8], 16) / 2 ** 32 < DEV_FRAC
                     for k in keys], dtype=bool)


def main():
    dd = Path(os.environ.get("DATA_DIR", "data")) / "open_bandit"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default=str(dd / "raw" / "open_bandit_dataset.zip"))
    ap.add_argument("--out", default=str(dd))
    a = ap.parse_args()
    out = Path(a.out)
    if any((out / f).exists() for f in EXPECTED):
        raise SystemExit(f"{out} already holds split files; choose another --out")
    if sha(a.zip) != ZIP_SHA256:
        raise SystemExit("zip sha256 differs from the recorded value")
    with zipfile.ZipFile(a.zip) as z, z.open(MEMBER) as f:
        df = pd.read_csv(io.TextIOWrapper(f))
    df = df.rename(columns={"Unnamed: 0": "row_id"})
    if not np.array_equal(df["row_id"].to_numpy(), np.arange(len(df))):
        raise SystemExit("row_id is not the 0-based row index")
    dev = is_dev(df["row_id"].to_numpy())
    if int(dev.sum()) != N_DEV or int((~dev).sum()) != N_EVAL:
        raise SystemExit(f"split sizes {int(dev.sum())} / {int((~dev).sum())} differ from {N_DEV} / {N_EVAL}")
    out.mkdir(parents=True, exist_ok=True)
    d = df[dev].reset_index(drop=True)
    e = df[~dev].reset_index(drop=True)
    d.to_pickle(out / "dev.pkl")
    e.drop(columns=["click"]).to_pickle(out / "eval_labels.pkl")
    np.save(out / "eval_outcome.npy", e["click"].to_numpy().astype(np.int8))
    files = {f: sha(out / f) for f in EXPECTED}
    res = {f: {"sha256": files[f], "matches_recorded": files[f] == EXPECTED[f]} for f in EXPECTED}
    prov = {"name": NAME, "source_file": MEMBER, "source_zip_sha256": ZIP_SHA256, "salt": SALT, "dev_frac": DEV_FRAC,
            "split_rule": "dev iff int(sha256(f'{salt}|{name}|{row_key}')[:8], 16) / 2**32 < dev_frac, "
                          "name = 'open_bandit'", "row_key": "row_id = the CSV's unnamed first column",
            "files": files, "note": "minimal provenance record written by split_obd_all.py (not the original record)"}
    if not (out / "PROVENANCE.json").exists():
        (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
