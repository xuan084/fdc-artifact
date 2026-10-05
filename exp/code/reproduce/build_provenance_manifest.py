"""Build exp/PROVENANCE_MANIFEST.json: per lock / block, what THIS copy lets a reader verify, and what it does not.

Everything below is recomputed from the files of this copy (no git history, no network):
  * lock files: SHA-256 of the shipped file; whether the lock's own ``sha256`` field (the canonical hash every row,
    seal and analysis binds) can be recomputed from the shipped bytes (true only for an unedited lock);
  * code, inputs and frozen gate files bound by each lock (``code_sha256``, ``input_sha256``, ``frozen_gates``):
    how many shipped files are byte-identical to the bound hash, which differ (scrubbed copies), which are absent;
  * data files bound by each lock (``data_sha256``): byte equality under $DATA_DIR when the data are present;
  * sealed rows: content hash against the seal (``check_sealed_rows.py``), replica-report status;
  * the reproduction runner's report (``exp/results/reproduce/report.json``) if present.
Items that cannot be verified from this copy are listed per lock under ``not_verifiable`` with the reason.

Usage (cwd = exp/code):  python reproduce/build_provenance_manifest.py [--out ../PROVENANCE_MANIFEST.json]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp" / "results"
FULL = RES / "full"

from dsswm.stats import prereg  # noqa: E402
from dsswm.stats.v6_replica import content_hash  # noqa: E402

LOCKS = {5: "plan/prereg_lock.json", **{v: f"plan/prereg_lock_v{v}_addendum.json" for v in range(6, 13)}}
TASKS = {
    6: ["v6a_full", "v6b_full_a", "v6b_full_b", "v6c_full_a", "v6c_full_b"],
    7: ["v7a_full_a", "v7a_full_b", "v7b_full_a", "v7b_full_b"],
    8: ["v8a_full_a", "v8a_full_b", "v8b_full_a", "v8b_full_b", "v8c_full"],
    9: ["v9a_full_d", "v9a_full_k", "v9a_full_x", "v9b_full_d", "v9b_full_k", "v9b_full_x", "v9c_full_b", "v9c_full_a"],
    10: ["v10a_full_s16", "v10a_full_s32", "v10b_full_s16", "v10b_full_s32", "v10b_full_s64", "v10d_full_x5s64"],
    11: ["v11_obd_full"],
    12: ["v12_women_full", "v12_men_full"],
}
SUB = {11: "v11_obd", 12: "v12_obd"}

NOT_VERIFIABLE = [
    {"item": "lock chronology (when each lock was written and committed, and that it preceded every eval read of its "
             "block)", "why": "the lock gates anchor each lock to a single commit of the original git history; that "
                              "history is not part of the anonymous copy (it would identify the authors)"},
    {"item": "seal chronology (each seal committed right after its eval run, before any analysis)",
     "why": "seal_ref sidecars name commits of the original history, which is not included"},
    {"item": "absence of unrecorded eval access (no eval-half outcome read outside the logged, gated readers)",
     "why": "a negative cannot be shown from files; the shipped access logs (v11, v12, block G) list the recorded "
            "reads only, and earlier locks had no access log"},
    {"item": "byte identity of the scrubbed lock files with the locked originals, and recomputation of their "
             "canonical sha256 fields", "why": "scrubbing edited string leaves (paths, author / tool labels); the lock "
                                               "body therefore no longer hashes to its sha256 field. "
                                               "exp/lock_scrub_diff.json lists the edited leaves (authors' statement, "
                                               "checkable at de-anonymisation)"},
    {"item": "byte identity of the scrubbed code / input files with the bound originals",
     "why": "files that contained paths or tool names were edited; behaviour equality of the code that matters is "
            "checked instead by the reproduction runner (regenerated rows equal the sealed rows)"},
]


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def bound_files(d: dict | None, root: Path) -> dict:
    d = d or {}
    same, diff, absent = [], [], []
    for rel, want in sorted(d.items()):
        if isinstance(want, dict):
            want = want.get("sha256")
        got = sha(root / rel)
        (absent if got is None else same if got == want else diff).append(rel)
    return {"n_bound": len(d), "n_byte_identical": len(same), "n_differ": len(diff), "n_absent": len(absent),
            "differ": diff, "absent": absent}


def data_files(d: dict | None) -> dict:
    dd = os.environ.get("DATA_DIR")
    out = {"n_bound": len(d or {}), "checked": dd is not None and Path(dd).exists(), "files": {}}
    for name, ent in sorted((d or {}).items()):
        want = ent.get("sha256") if isinstance(ent, dict) else ent
        rec = {"sha256": want}
        if out["checked"]:
            path = ent.get("path", "") if isinstance(ent, dict) else ""
            if "$DATA_DIR/" in path:
                f = Path(dd) / path.split("$DATA_DIR/", 1)[-1]
            elif "/datasets/" in path:                      # unscrubbed locks name absolute paths
                f = Path(dd) / path.split("/datasets/", 1)[-1]
            else:
                f = None
            got = sha(f) if f is not None else None
            rec["present"] = got is not None
            rec["byte_identical"] = got == want if got else None
        out["files"][name] = rec
    return out


def rows_file(v, task):
    base = (FULL / SUB[v] / task) if v in SUB else (FULL / task)
    for n in ("results.jsonl", "results.jsonl.gz"):
        if (base / n).exists():
            return base / n
    return None


def task_record(v, task, repro):
    seal_p = FULL / f"v{v}_seals" / f"{task}.seal.json"
    seal = json.loads(seal_p.read_text()) if seal_p.exists() else None
    rec = {"task": task, "seal_shipped": seal is not None}
    rp = rows_file(v, task)
    rec["rows_shipped"] = rp is not None
    if rp is not None and seal is not None:
        raw = gzip.open(rp, "rb").read() if rp.suffix == ".gz" else rp.read_bytes()
        rows = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
        rec.update({"rows_file": str(rp.relative_to(WS)), "rows_sha256": hashlib.sha256(raw).hexdigest(),
                    "rows_content_matches_seal": content_hash(rows) == seal["results_content_sha256"],
                    "n_rows": len(rows)})
    base = (FULL / SUB[v] / task) if v in SUB else (FULL / task)
    rr = base / "replica_report.json"
    if rr.exists():
        rep = json.loads(rr.read_text())
        rec["replica_report_status"] = rep.get("status")
        rec["replica_report_content_matches_seal"] = (seal is not None and rep.get("results_content_sha256")
                                                      == seal.get("results_content_sha256"))
    if repro and task in repro:
        r = repro[task]
        rec["reproduction"] = {k: r.get(k) for k in ("status", "seeds", "n_jobs", "n_exact_match", "n_mismatch")}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "exp" / "PROVENANCE_MANIFEST.json"))
    ap.add_argument("--repro-report", default=str(RES / "reproduce" / "report.json"))
    a = ap.parse_args()
    repro = None
    if Path(a.repro_report).exists():
        rj = json.loads(Path(a.repro_report).read_text())
        repro = {t["task"]: t for t in rj["tasks"]}
    scrub = WS / "exp" / "lock_scrub_diff.json"
    scrub = json.loads(scrub.read_text())["locks"] if scrub.exists() else {}
    out = {"what": __doc__.split("\n")[0], "not_verifiable_from_this_copy": NOT_VERIFIABLE, "locks": {}}
    for v, rel in LOCKS.items():
        p = WS / rel
        lock = json.loads(p.read_text())
        canon = prereg.canonical_hash(lock)
        ent = {"lock_file": rel, "file_sha256_this_copy": sha(p), "sha256_field": lock.get("sha256"),
               "sha256_field_recomputes_from_this_copy": lock.get("sha256") == canon,
               "scrub_diff_leaves_changed": (scrub.get(Path(rel).name) or {}).get("leaves_changed"),
               "scrub_diff_non_string_leaves_changed": (scrub.get(Path(rel).name) or {}).get(
                   "non_string_leaves_changed"),
               "code": bound_files(lock.get("code_sha256"), CODE),
               "inputs": bound_files(lock.get("input_sha256"), WS),
               "frozen_gates": bound_files(lock.get("frozen_gates"), WS) if "frozen_gates" in lock else None,
               "data": data_files(lock.get("data_sha256")),
               "eval_tasks": [task_record(v, t, repro) for t in TASKS.get(v, [])]}
        out["locks"][f"v{v}"] = ent
        c = ent["code"]
        print(f"v{v:<3d} lock recomputes: {ent['sha256_field_recomputes_from_this_copy']!s:5s}  code "
              f"{c['n_byte_identical']}/{c['n_bound']} identical ({c['n_differ']} scrubbed, {c['n_absent']} absent)  "
              f"tasks {sum(t['rows_shipped'] for t in ent['eval_tasks'])}/{len(ent['eval_tasks'])} rows shipped, "
              f"{sum(bool(t.get('rows_content_matches_seal')) for t in ent['eval_tasks'])} match seal, "
              f"{sum((t.get('reproduction') or {}).get('status') == 'match' for t in ent['eval_tasks'])} reproduced")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
