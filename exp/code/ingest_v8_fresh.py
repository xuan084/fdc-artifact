"""Round-8 fresh-table ingest with a BLINDED dev/eval split (methodologist, 2026-10-04).  New file.

Datasets (downloaded 2026-10-04, raw files kept untouched under <root>/<name>/raw/):
  x5_retailhero  X5 RetailHero uplift (real SMS-campaign RCT; HF mirror pytorch-lifestream/retailhero-uplift):
                 uplift_train.csv.gz (client_id, treatment_flg, target) joined with clients.csv.gz (age, gender,
                 first_issue_date, first_redeem_date).
  megafon        MegaFon uplift competition (600k rows, X_1..X_50, treatment_group, conversion; documented by
                 scikit-uplift as a GENERATED SYNTHETIC dataset -> not eligible as "real data").

Blinding mechanism
  * split key = sha256(f"{SALT}|{dataset}|{row_key}") with row_key = client_id (X5) / 0-based raw row index (MegaFon);
    dev iff int(first 8 hex digits, 16) / 2^32 < DEV_FRAC.  SALT and DEV_FRAC are fixed here before any outcome is
    read; the split depends on identifiers only (outcome-free).
  * written files: dev.pkl (features, treatment, outcome; dev rows only), eval_labels.pkl (features, treatment; NO
    outcome), eval_outcome.npy (outcome vector aligned with eval_labels, chmod 444, sha256 in PROVENANCE.json).
  * this script computes outcome statistics on the DEV rows only; the eval outcome array is written straight from the
    parsed column without any reduction (no mean / count / print).  v8 code must read eval_outcome.npy only through a
    lock-gated loader (to be written with the v8 lock).
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("DATA_DIR", "data"))
SALT = "dsswm-v8-2026-10-04"
DEV_FRAC = 0.5


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dev_mask(dataset, keys):
    u = np.array([int(hashlib.sha256(f"{SALT}|{dataset}|{k}".encode()).hexdigest()[:8], 16) for k in keys],
                 dtype=np.float64) / 2 ** 32
    return u < DEV_FRAC


def write_blinded(name, df, treat_col, y_col, key_col, extra):
    out = ROOT / name
    dev = dev_mask(name, df[key_col].tolist())
    d = df[dev].reset_index(drop=True)
    e = df[~dev].reset_index(drop=True)
    d.to_pickle(out / "dev.pkl")
    e.drop(columns=[y_col]).to_pickle(out / "eval_labels.pkl")
    yo = out / "eval_outcome.npy"
    if yo.exists():
        os.chmod(yo, stat.S_IWUSR | stat.S_IRUSR)
    np.save(yo, e[y_col].to_numpy())
    os.chmod(yo, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    # dev-side statistics only
    dstats = {"n_dev": int(len(d)), "n_eval": int(len(e)), "dev_frac_realised": float(dev.mean()),
              "treat_share_dev": float(d[treat_col].mean()), "treat_share_eval_labels_only": float(e[treat_col].mean()),
              "outcome_rate_dev_by_arm": {int(a): float(g[y_col].mean()) for a, g in d.groupby(treat_col)}}
    prov = {"name": name, "created_utc": datetime.now(timezone.utc).isoformat(), "salt": SALT, "dev_frac": DEV_FRAC,
            "split_rule": "dev iff int(sha256(f'{salt}|{name}|{row_key}')[:8], 16) / 2**32 < dev_frac",
            "row_key": key_col, "treatment": treat_col, "outcome": y_col,
            "files": {f: sha_file(out / f) for f in ("dev.pkl", "eval_labels.pkl", "eval_outcome.npy")},
            "blinding": "eval outcomes only in eval_outcome.npy (read-only); no eval-outcome statistic computed",
            "dev_stats": dstats, **extra}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    return prov


def x5():
    raw = ROOT / "x5_retailhero" / "raw"
    tr = pd.read_csv(raw / "uplift_train.csv.gz")
    cl = pd.read_csv(raw / "clients.csv.gz")
    df = tr.merge(cl, on="client_id", how="left", validate="one_to_one")
    assert len(df) == len(tr) == 200039
    df = df.rename(columns={"treatment_flg": "treatment", "target": "outcome"})
    df["treatment"] = df["treatment"].astype(np.int8)
    df["outcome"] = df["outcome"].astype(np.int8)
    df["redeemed"] = df["first_redeem_date"].notna().astype(np.int8)
    return write_blinded("x5_retailhero", df, "treatment", "outcome", "client_id", {
        "source": "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift (X5 RetailHero, ods.ai)",
        "raw_sha256": {f: sha_file(raw / f) for f in ("uplift_train.csv.gz", "clients.csv.gz")},
        "real_data": True, "n_rows": int(len(df))})


def megafon():
    raw = ROOT / "megafon" / "raw"
    df = pd.read_csv(raw / "megafon_dataset.csv.gz")
    df["row_id"] = np.arange(len(df))
    df["treatment"] = (df["treatment_group"] == "treatment").astype(np.int8)
    df = df.drop(columns=["treatment_group"]).rename(columns={"conversion": "outcome"})
    return write_blinded("megafon", df, "treatment", "outcome", "row_id", {
        "source": "https://sklift.s3.eu-west-2.amazonaws.com/megafon_dataset.csv.gz (MegaFon uplift competition)",
        "raw_sha256": {"megafon_dataset.csv.gz": sha_file(raw / "megafon_dataset.csv.gz")},
        "real_data": False, "note": "documented by scikit-uplift as a generated synthetic dataset",
        "n_rows": int(len(df))})


if __name__ == "__main__":
    which = sys.argv[1:] or ["x5", "megafon"]
    for w in which:
        p = {"x5": x5, "megafon": megafon}[w]()
        print(w, json.dumps(p["dev_stats"]))
