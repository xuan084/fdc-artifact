"""Check every shipped sealed row file against its seal (no data, no git history needed).

For each eval task whose rows are shipped (``exp/results/full/**/<task>/results.jsonl`` or ``results.jsonl.gz``):
  * SHA-256 of the (decompressed) row-file bytes;
  * ``results_content_sha256`` recomputed with ``dsswm.stats.v6_replica.content_hash`` (the function every seal
    module v6-v12 uses: canonical rows without timing fields, sorted) and compared with the seal;
  * the row count and the planned (method x eps x seed) matrix of the seal (complete, no duplicates, no extras);
  * the seal's ``addendum_sha256`` equals the ``sha256`` field of the shipped lock and every row's addendum_sha256.
What this does NOT check: that the seal itself was committed when it says (the seal_ref sidecars name commits of the
original history, which is not part of this copy); see exp/PROVENANCE_MANIFEST.md.

Usage (cwd = exp/code):  python check_sealed_rows.py [--json OUT]      exit 0 iff all shipped row files pass.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
FULL = WS / "exp" / "results" / "full"

from dsswm.stats.v6_replica import check_matrix, content_hash  # noqa: E402


def _seeds(spec):
    if isinstance(spec, list) and len(spec) == 3 and spec[1] - spec[0] + 1 == spec[2]:
        return list(range(spec[0], spec[1] + 1))
    if isinstance(spec, list):
        return list(spec)
    raise ValueError(spec)


def find_rows(task):
    hits = [p for p in FULL.rglob("results.jsonl*") if p.parent.name == task and p.name in
            ("results.jsonl", "results.jsonl.gz")]
    return sorted(hits)[0] if hits else None


def lock_sha_field(version):
    p = WS / "plan" / f"prereg_lock_v{version}_addendum.json"
    return json.loads(p.read_text()).get("sha256") if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out, ok = [], True
    for sp in sorted(FULL.glob("v*_seals/*.seal.json")):
        seal = json.loads(sp.read_text())
        task = seal["task_id"]
        rp = find_rows(task)
        rec = {"task": task, "seal": str(sp.relative_to(WS)), "rows_file": None}
        if rp is None:
            rec["status"] = "rows not shipped"
            out.append(rec)
            continue
        raw = gzip.open(rp, "rb").read() if rp.suffix == ".gz" else rp.read_bytes()
        rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        version = int(sp.parent.name.split("_")[0][1:])
        m = check_matrix(rows, seal["methods"], seal["eps"], _seeds(seal["seeds"]))
        lsha = lock_sha_field(version)
        rec.update({"rows_file": str(rp.relative_to(WS)), "rows_sha256": hashlib.sha256(raw).hexdigest(),
                    "file_sha256": hashlib.sha256(rp.read_bytes()).hexdigest(), "n_rows": len(rows),
                    "content_sha256": content_hash(rows), "seal_content_sha256": seal["results_content_sha256"],
                    "matrix_complete": bool(m["complete"]),
                    "lock_sha256_field": lsha,
                    "addendum_matches": seal.get("addendum_sha256") == lsha
                    and all(r.get("addendum_sha256") == lsha for r in rows)})
        good = (rec["content_sha256"] == rec["seal_content_sha256"] and len(rows) == seal["n_rows"]
                and rec["matrix_complete"] and rec["addendum_matches"])
        rec["status"] = "pass" if good else "FAIL"
        ok &= good
        out.append(rec)
        print(f"{task:18s} {rec['status']:5s} rows {len(rows):5d}  content {rec['content_sha256'][:16]}  "
              f"rows-file sha256 {rec['rows_sha256'][:16]}", flush=True)
    for r in out:
        if r["status"] == "rows not shipped":
            print(f"{r['task']:18s} rows not shipped (seal and analyses only)")
    n = sum(r["status"] == "pass" for r in out)
    print(f"sealed rows: {n} pass, {sum(r['status'] == 'FAIL' for r in out)} fail, "
          f"{sum(r['status'] == 'rows not shipped' for r in out)} not shipped")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
