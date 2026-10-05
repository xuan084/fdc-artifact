"""Lock-v7 addendum: replica rules (frozen before any v7 eval run; new file, reuses the v6 rule implementations).

R1 (rerun): an independent process re-runs the FIRST 10 seeds of each v7 eval task for every method and eps of the
   task; canonical rows (all fields except timing) must be identical (``v6_replica.check_r1``).
R2 (design identity): within every (seed, eps), ``schedule_digest`` is identical across the task's registered 50/50
   comparison group (``COMPARISON_GROUPS``), on the FULL main matrix (``v6_replica.check_r2``).
No R3 (v7 has no continuation of earlier streams).

Report integrity exactly as v6: the analysis RECOMPUTES R1/R2 from the actual rows and uses only the recomputed status;
the stored report is a cache that must reconcile exactly with the recomputation and its bindings (task, addendum,
code, data, main-row and replica-row content hashes); for eval tasks the current main rows must equal the git-sealed
content hash.
"""
from __future__ import annotations

import json

from .v6_replica import R1_N_SEEDS, check_matrix, check_r1, check_r2, combine, content_hash

GROUP_A = ("FDC-BF", "FDC-MR[front3]", "FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR")
GROUP_B = ("FDC-BF", "FDC-MR[front3]", "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR")
COMPARISON_GROUPS = {
    "v7a_full_a": (),
    "v7a_full_b": GROUP_A,
    "v7b_full_a": (),
    "v7b_full_b": GROUP_B,
    "v7a_pilot": GROUP_A,
    "v7b_pilot": GROUP_B,
}

__all__ = ["COMPARISON_GROUPS", "GROUP_A", "GROUP_B", "required_rules", "compute_checks", "build_report",
           "validate_report", "R1_N_SEEDS"]


def required_rules(task, is_eval):
    return ("R1", "R2")


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds):
    m = check_matrix(main_rows, methods, eps, seeds)
    if not m["complete"]:
        raise ValueError(f"{task}: planned main matrix incomplete: {m}")
    r1_seeds = list(seeds[:R1_N_SEEDS])
    mr = check_matrix(replica_rows, methods, eps, r1_seeds)
    if not mr["complete"]:
        raise ValueError(f"{task}: replica matrix (R1 seeds) incomplete: {mr}")
    if task not in COMPARISON_GROUPS:
        raise ValueError(f"{task}: no registered comparison group")
    return _norm([check_r1(main_rows, replica_rows, methods, eps, r1_seeds),
                  check_r2(main_rows, COMPARISON_GROUPS[task], seeds, eps)])


def _bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256):
    return {"task_id": task, "is_eval": bool(is_eval), "addendum_sha256": add_sha, "code_sha256": code_sha256,
            "data_sha256": data_sha256, "results_content_sha256": content_hash(main_rows),
            "replica_content_sha256": content_hash(replica_rows), "n_rows": len(main_rows),
            "n_replica_rows": len(replica_rows), "methods": list(methods), "eps": [float(e) for e in eps],
            "seeds": [int(seeds[0]), int(seeds[-1]), len(seeds)],
            "r1_seeds": [int(s) for s in seeds[:R1_N_SEEDS]], "required_rules": list(required_rules(task, is_eval))}


def build_report(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256):
    if is_eval and not add_sha:
        raise ValueError("replica report refused: eval task without a locked-addendum sha256")
    rep = combine(task, compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds))
    rep.update(_bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                         data_sha256))
    return rep


def validate_report(rep, task, is_eval, current_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                    data_sha256, sealed_content_sha256=None):
    if not isinstance(rep, dict):
        raise ValueError(f"{task}: no replica report")
    if is_eval and not add_sha:
        raise ValueError(f"{task}: eval analysis without a locked-addendum sha256")
    if is_eval and not sealed_content_sha256:
        raise ValueError(f"{task}: eval analysis without a verified eval seal")
    if sealed_content_sha256 is not None and content_hash(current_rows) != sealed_content_sha256:
        raise ValueError(f"{task}: current main rows differ from the git-sealed content hash")
    checks = compute_checks(task, is_eval, current_rows, replica_rows, methods, eps, seeds)
    status = combine(task, checks)["status"]
    want = _bindings(task, is_eval, current_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                     data_sha256)
    errs = [k for k, v in want.items() if _norm(rep.get(k)) != _norm(v)]
    if _norm(rep.get("checks")) != checks:
        errs.append("checks differ from the recomputation")
    if rep.get("status") != status:
        errs.append(f"status {rep.get('status')!r} != recomputed {status!r}")
    if errs:
        raise ValueError(f"{task}: replica report does not reconcile with the recomputation: {errs}")
    return status
