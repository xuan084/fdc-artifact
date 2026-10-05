"""Lock-v10 addendum: replica rules (frozen before any v10 eval run; new file; the v6 / v8 rule implementations are
reused unchanged).

R1  (rerun): an independent process re-runs the FIRST 10 seeds of each v10 eval task for every method of the task;
    canonical rows (all fields except the timing fields 'sec', 'rs_sec_cert', 'rs_sec_total', 'rs_sec_plan') must be
    identical (``v6_replica.check_r1``).
R2  (design identity): within every (seed, eps), ``schedule_digest`` is identical across ALL methods of the task (all
    four v10 methods use the frozen 50/50 design: balanced_alloc(S, A, 0.5)).
R2b (arrival identity): within every (seed, eps), ``arrival_digest`` (arrival segment sequence + every pool's record
    order of the schedule the engine actually consumed) is identical across all methods of the task.
Report integrity exactly as v7-v9 (recomputation on the actual rows, reconciliation of the stored report with every
binding, the git-sealed content hash for eval tasks).
"""
from __future__ import annotations

from .v6_replica import R1_N_SEEDS, check_matrix, check_r1, check_r2, combine, content_hash  # noqa: F401
from .v8_replica import _norm, check_r2b
from .v8_replica import _bindings as _bindings_v8
from .v10_analysis import METHODS

__all__ = ["groups_for", "compute_checks", "build_report", "validate_report", "R1_N_SEEDS", "check_r2b"]


def groups_for(task, methods=METHODS):
    """One frozen-design group per task: every registered v10 method (all on the 50/50 design)."""
    return (tuple(methods),)


def compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds):
    m = check_matrix(main_rows, methods, eps, seeds)
    if not m["complete"]:
        raise ValueError(f"{task}: planned main matrix incomplete: {m}")
    r1_seeds = list(seeds[:R1_N_SEEDS])
    mr = check_matrix(replica_rows, methods, eps, r1_seeds)
    if not mr["complete"]:
        raise ValueError(f"{task}: replica matrix (R1 seeds) incomplete: {mr}")
    checks = [check_r1(main_rows, replica_rows, methods, eps, r1_seeds)]
    for g in groups_for(task, methods):
        gg = tuple(x for x in g if x in methods)
        if len(gg) >= 2:
            checks.append(check_r2(main_rows, gg, seeds, eps))
    checks.append(check_r2b(main_rows, seeds, eps, methods))
    return _norm(checks)


def _bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256):
    b = _bindings_v8(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256)
    b["required_rules"] = ["R1", "R2", "R2b"]
    return b


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
