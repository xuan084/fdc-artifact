"""Artifact helper (added for the anonymous package; not part of any lock).

Recomputes the lock-v10 analyses (blocks A, B confirmatory; D descriptive) from the shipped sealed rows, without data,
without git history and without the lock gates (which cannot pass in a scrubbed copy, see README):
  1. for every v10 task, the content hash of exp/results/full/<task>/results.jsonl (error-free rows, timing excluded)
     must equal results_content_sha256 in exp/results/full/v10_seals/<task>.seal.json;
  2. the block analysis is recomputed with dsswm.stats.v10_analysis (same code path as run_v10.py --analyse);
  3. every numeric leaf under cells.<task>.comparisons / .methods and the block verdict is compared with the shipped
     exp/results/full/v10_analysis_<block>.json.
The replica status is taken from the shipped analysis (the replica rows are not shipped).
Usage (cwd = exp/code):  python recompute_v10_from_sealed_rows.py
"""
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
FULL = CODE.parents[1] / "exp/results/full"

from dsswm.stats import v10_analysis as VA  # noqa: E402
from dsswm.stats.v6_analysis import boot_idx  # noqa: E402
from dsswm.stats.v6_replica import content_hash  # noqa: E402


def leaves(d, pre=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{pre}.{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from leaves(v, f"{pre}[{i}]")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield pre, float(d)


def main():
    bad = 0
    for block in "ABD":
        shipped = json.loads((FULL / f"v10_analysis_{block}.json").read_text())
        rows_by = {}
        for t in shipped["tasks"]:
            rows = [json.loads(x) for x in (FULL / t / "results.jsonl").read_text().splitlines() if x.strip()]
            rows = [r for r in rows if r.get("error") is None]
            seal = json.loads((FULL / "v10_seals" / f"{t}.seal.json").read_text())
            ok = content_hash(rows) == seal["results_content_sha256"] and len(rows) == seal["n_rows"]
            bad += not ok
            print(f"[{'OK' if ok else 'MISMATCH'}] {t}: {len(rows)} rows, content hash "
                  f"{content_hash(rows)[:12]} vs seal {seal['results_content_sha256'][:12]}")
            rows_by[t] = rows
        if block in ("A", "B"):
            res = VA.analyse_block(block, rows_by, shipped["replica_status_combined"],
                                   seeds=VA.BLOCK_SEEDS[block], cells=VA.CELLS[block])
        else:
            seeds = VA.BLOCK_SEEDS["A"]
            idx = boot_idx(len(seeds))
            res = {"cells": {t: VA.analyse_cell(t, rows_by[t], seeds, VA.DESC_CELLS[t]["eps"], idx=idx)
                             for t in rows_by}}
        n = worst = 0
        for t, c in res["cells"].items():
            want = dict(leaves({k: shipped["cells"][t][k] for k in ("comparisons", "methods")}))
            for k, v in leaves({k: c[k] for k in ("comparisons", "methods")}):
                if k in want:
                    n += 1
                    worst = max(worst, abs(v - want[k]))
        v_new = res.get("decision", {}).get("verdict", "descriptive")
        v_old = shipped.get("decision", {}).get("verdict", "descriptive")
        ok = worst <= 1e-9 and v_new == v_old and n > 0
        bad += not ok
        print(f"[{'OK' if ok else 'MISMATCH'}] block {block}: verdict {v_new!r} (shipped {v_old!r}); "
              f"{n} numeric fields compared, max |diff| {worst:.3g}")
    print(f"v10 recompute: {bad} mismatch(es)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
