"""v12: blinded salted 50/50 row split of the two unread Open Bandit campaigns random/women and random/men
(plan/v12_obd2_plan.md s2).  NEW file; run ONCE, before any click of either campaign is read.

For each campaign: read <campaign>.csv from the zip (python zipfile), check row_key = 0-based row index, assign
dev iff int(sha256(f'{salt}|{name}|{row_key}')[:8], 16) / 2**32 < 0.5, write
  open_bandit/<campaign>/dev.pkl          all columns of the dev rows (row_id renamed from the unnamed index)
  open_bandit/<campaign>/eval_labels.pkl  every column except click, eval rows in original order
  open_bandit/<campaign>/eval_outcome.npy int8 click aligned with eval_labels.pkl
(all chmod 444) and PROVENANCE.json (written once; split_created_utc recorded at split creation and never rewritten).
Only DEV clicks are summarised.  Eval: row count only.  Overlap (plan s2.1) uses the non-outcome timestamp column:
campaign rows vs the random/all eval half (recomputed from all.csv's index + timestamp columns with the v11 salt rule,
never opening the random/all eval files), vs the other campaign, and dev vs eval within a campaign.
The existing random/all files and the top-level PROVENANCE.json (hash-bound by lock v11) are not modified; the
pointer goes to open_bandit/V12_SPLITS.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("DATA_DIR", "data")) / "open_bandit"
ZIP = ROOT / "raw" / "open_bandit_dataset.zip"
CAMPAIGNS = {"women": {"salt": "dsswm-obd-women-2026-10-05", "name": "open_bandit_women"},
             "men": {"salt": "dsswm-obd-men-2026-10-05", "name": "open_bandit_men"}}
ALL_SALT, ALL_NAME = "dsswm-obd-2026-10-05", "open_bandit"
DEV_FRAC = 0.5
RULE = "dev iff int(sha256(f'{salt}|{name}|{row_key}')[:8], 16) / 2**32 < dev_frac"


def is_dev(salt, name, keys):
    return np.array([int(hashlib.sha256(f"{salt}|{name}|{k}".encode()).hexdigest()[:8], 16) / 2 ** 32 < DEV_FRAC
                     for k in keys], dtype=bool)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(z, member, usecols=None):
    with z.open(member) as f:
        return pd.read_csv(io.TextIOWrapper(f), usecols=usecols)


def ro(p):
    os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def main():
    z = zipfile.ZipFile(ZIP)
    for c in CAMPAIGNS:
        if (ROOT / c).exists():
            raise SystemExit(f"{ROOT / c} exists: the v12 split is never rewritten")
    # random/all eval-half timestamps (non-outcome columns only; v11 salt rule; random/all eval files untouched)
    al = read_csv(z, "open_bandit_dataset/random/all/all.csv", usecols=["Unnamed: 0", "timestamp"])
    if not np.array_equal(al["Unnamed: 0"].to_numpy(), np.arange(len(al))):
        raise SystemExit("random/all row_key check failed")
    al_dev = is_dev(ALL_SALT, ALL_NAME, al["Unnamed: 0"].to_numpy())
    if int((~al_dev).sum()) != 688159:
        raise SystemExit(f"random/all eval half size {(~al_dev).sum()} != 688159 (v11 PROVENANCE)")
    all_eval_ts = set(al.loc[~al_dev, "timestamp"].astype(str))
    all_ts_rng = [str(al["timestamp"].min()), str(al["timestamp"].max())]
    del al
    ts = {}
    prov = {}
    for c, cf in CAMPAIGNS.items():
        created = datetime.now(timezone.utc).isoformat()
        member = f"open_bandit_dataset/random/{c}/{c}.csv"
        df = read_csv(z, member)
        df = df.rename(columns={"Unnamed: 0": "row_id"})
        if not np.array_equal(df["row_id"].to_numpy(), np.arange(len(df))):
            raise SystemExit(f"{c}: row_key is not the 0-based row index; rule not silently changed")
        dev = is_dev(cf["salt"], cf["name"], df["row_id"].to_numpy())
        out = ROOT / c
        out.mkdir(parents=False, exist_ok=False)
        d = df[dev].reset_index(drop=True)
        e = df[~dev].reset_index(drop=True)
        d.to_pickle(out / "dev.pkl")
        e.drop(columns=["click"]).to_pickle(out / "eval_labels.pkl")
        np.save(out / "eval_outcome.npy", e["click"].to_numpy().astype(np.int8))
        del e
        for f in ("dev.pkl", "eval_labels.pkl", "eval_outcome.npy"):
            ro(out / f)
        files = {f: sha(out / f) for f in ("dev.pkl", "eval_labels.pkl", "eval_outcome.npy")}
        # ---- dev-only statistics (eval: row count only)
        item_ctr = d.groupby("item_id").click.mean()
        ts[c] = {"dev": set(d["timestamp"].astype(str)), "eval": set(df.loc[~dev, "timestamp"].astype(str)),
                 "all_rows": df["timestamp"].astype(str).to_numpy(), "dev_rows": d["timestamp"].astype(str).to_numpy(),
                 "range": [str(df["timestamp"].min()), str(df["timestamp"].max())]}
        prov[c] = {
            "name": cf["name"], "campaign": f"random/{c}",
            "split_created_utc": created,
            "split_created_note": "recorded once at split creation; this file is written once and never rewritten",
            "source_zip": str(ZIP), "source_zip_sha256": "e8ec18196582a5937381a1776382ca940689b90a18d2dcd1fb635be6df614d78",
            "source_file": f"{member} (uniform-random logging policy)",
            "salt": cf["salt"], "dev_frac": DEV_FRAC, "split_rule": RULE + f", name = '{cf['name']}'",
            "row_key": "row_id = the CSV's unnamed first column (equals the 0-based row index; verified)",
            "treatment": f"item_id ({int(df['item_id'].nunique())} items; uniform random) at position 1-3",
            "outcome": "click", "files": files,
            "file_notes": {"dev.pkl": f"all {d.shape[1]} columns (row_id renamed from the unnamed index)",
                           "eval_labels.pkl": "all columns except click, eval rows in original order",
                           "eval_outcome.npy": "int8 click, aligned with eval_labels.pkl rows; chmod 444"},
            "blinding": "eval outcomes only in eval_outcome.npy (read-only); no eval-outcome statistic computed; "
                        "dev-only statistics below; eval: row count and non-outcome overlap counts only",
            "plan": "iter_001/plan/v12_obd2_plan.md (committed 317619b4 / 5b3eb048 before this split)",
            "dev_stats": {
                "n_dev": int(len(d)), "n_eval": int((~dev).sum()), "n_rows_total": int(len(df)),
                "dev_frac_realised": float(dev.mean()),
                "propensity_values_dev": sorted(float(x) for x in d["propensity_score"].unique())[:10],
                "n_items_dev": int(d["item_id"].nunique()),
                "click_rate_dev": float(d["click"].mean()), "clicks_dev": int(d["click"].sum()),
                "click_rate_dev_by_position": {str(k): float(v) for k, v in d.groupby("position").click.mean().items()},
                "click_rate_dev_by_item_summary": {k: float(v) for k, v in item_ctr.describe().items()}},
            "real_data": True,
            "uses": [{"date": "2026-10-05", "who": "ds-swm methodologist (v12)",
                      "what": "blinded split (exp/code/split_obd_v12.py)", "eval_outcomes_touched": False}]}
        del d, df
        print(c, prov[c]["dev_stats"]["n_dev"], prov[c]["dev_stats"]["n_eval"], flush=True)
    # ---- overlap (non-outcome timestamp column only)
    for c in CAMPAIGNS:
        o = "men" if c == "women" else "women"
        rows = ts[c]["all_rows"]
        in_all_eval = np.fromiter((t in all_eval_ts for t in rows), bool, len(rows))
        in_other = np.fromiter((t in ts[o]["dev"] or t in ts[o]["eval"] for t in rows), bool, len(rows))
        dev_rows = ts[c]["dev_rows"]
        dev_in_eval = np.fromiter((t in ts[c]["eval"] for t in dev_rows), bool, len(dev_rows))
        prov[c]["overlap_non_outcome"] = {
            "timestamp_range": ts[c]["range"], "random_all_timestamp_range": all_ts_rng,
            "rows_sharing_exact_timestamp_with_random_all_eval_half": int(in_all_eval.sum()),
            "share_rows_sharing_timestamp_with_random_all_eval_half": float(in_all_eval.mean()),
            f"rows_sharing_exact_timestamp_with_{o}": int(in_other.sum()),
            f"share_rows_sharing_timestamp_with_{o}": float(in_other.mean()),
            "share_dev_rows_sharing_timestamp_with_own_eval_half": float(dev_in_eval.mean()),
            "note": "timestamps only (non-outcome); random/all eval membership recomputed from all.csv's index and "
                    "timestamp columns with the v11 salt rule (688,159 rows, matches PROVENANCE); user ids are absent, "
                    "so user overlap cannot be measured; separate campaigns have separate item sets, so a shared "
                    "timestamp means a concurrent impression, not a duplicated row"}
    for c in CAMPAIGNS:
        p = ROOT / c / "PROVENANCE.json"
        p.write_text(json.dumps(prov[c], indent=1))
        ro(p)
    ptr = {"written_utc": datetime.now(timezone.utc).isoformat(),
           "note": "v12 splits of random/women and random/men (plan iter_001/plan/v12_obd2_plan.md). The top-level "
                   "PROVENANCE.json is NOT amended because lock v11 binds its sha256.",
           "campaigns": {c: {"dir": str(ROOT / c), "provenance_sha256": sha(ROOT / c / "PROVENANCE.json")}
                         for c in CAMPAIGNS}}
    (ROOT / "V12_SPLITS.json").write_text(json.dumps(ptr, indent=1))
    print(json.dumps({c: prov[c]["overlap_non_outcome"] for c in CAMPAIGNS}, indent=1))


if __name__ == "__main__":
    main()
