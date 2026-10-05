"""Schema-aware FDC reproducibility check between an r5_cr_plugin_{a,b} block and its paired r5_cr_main_{a,b} block.

run_r5_cr_main_b.py nests billing_ok / schedule_digest and all run_stream fields under row["run_stream"], while the
plugin runner flattens them (top-level + rs_* fields). The built-in fdc_repro_check of run_r5_cr_plugin_a.py compares
top-level keys only and therefore reports a spurious full mismatch against block B. This post-hoc checker (not part of
the frozen block code; it reads results only, no analysis) compares the endpoint fields and every non-timing
run_stream field under both schemas.

Usage: python check_r5_fdc_repro_keymapped.py --task r5_cr_plugin_b --mode {pilot,full}
"""
import argparse
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
PAIR = {"r5_cr_plugin_a": "r5_cr_main_a", "r5_cr_plugin_b": "r5_cr_main_b"}
KEYS = ["N80_pen", "N80_raw", "completed", "fwer_event", "cert_k", "billing_ok", "schedule_digest", "n_cert_curve"]


def rs_fields(row):
    """Return run_stream fields from either schema (nested dict or flattened rs_* keys)."""
    if isinstance(row.get("run_stream"), dict):
        return dict(row["run_stream"])
    return {k[3:]: v for k, v in row.items() if k.startswith("rs_")}


def get(row, k):
    return row[k] if k in row else rs_fields(row).get(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(PAIR))
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    a = ap.parse_args()
    sub = "pilots" if a.mode == "pilot" else "full"
    load = lambda p: {x["seed"]: x for x in map(json.loads, p.read_text().splitlines())
                      if x.get("method") == "FDC" and x.get("error") is None}
    mine, other = load(RES / sub / a.task / "results.jsonl"), load(RES / sub / PAIR[a.task] / "results.jsonl")
    common = sorted(set(mine) & set(other))
    mis = []
    for s in common:
        for k in KEYS:
            if get(mine[s], k) != get(other[s], k):
                mis.append([s, k])
        r1, r2 = rs_fields(mine[s]), rs_fields(other[s])
        for k in sorted(set(r1) | set(r2)):
            if not k.startswith("sec") and r1.get(k) != r2.get(k):
                mis.append([s, "run_stream." + k])
    rep = {"task": a.task, "paired_task": PAIR[a.task], "mode": a.mode, "n_common": len(common),
           "n_mismatch_seeds": len({m[0] for m in mis}), "mismatches": mis[:50],
           "keys": KEYS + ["run_stream.* (excluding sec_*)"]}
    (RES / sub / a.task / "fdc_repro_check_keymapped.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("n_common", "n_mismatch_seeds")}))


if __name__ == "__main__":
    main()
