"""Lock-v8 addendum: replica rules (frozen before any v8 eval run; new file, reuses the v6 rule implementations).

R1  (rerun): an independent process re-runs the FIRST 10 seeds of each v8 eval task for every method and eps of the
    task; canonical rows (all fields except timing) must be identical (``v6_replica.check_r1``).
R2  (design identity): within every (seed, eps), ``schedule_digest`` is identical across each registered design group
    of the task (``DESIGN_GROUPS``: frozen 50/50 methods; adaptive methods -- their schedule carries no pre-planned arm,
    so the digest is the arrival + pool-order digest), on the FULL main matrix (``v6_replica.check_r2`` per group).
R2b (arrival identity, v8): within every (seed, eps), ``arrival_digest`` (sha256 of the arrival segment sequence and of
    every pool's record order, recomputed from ``make_schedule(env, seed, adaptive=True)``) is identical across ALL
    methods of the task.  This is what pairs the adaptive PJC rivals (different schedule digests) with FDC-BF.
Report integrity exactly as v7: recomputation on the actual rows, reconciliation of the stored report with every
binding, and for eval tasks the git-sealed content hash.
"""
from __future__ import annotations

import json

from .v6_replica import R1_N_SEEDS, check_matrix, check_r1, check_r2, combine, content_hash

# frozen 50/50 design groups (RECT-ck-HG / HC-WoR enter only if their tuned plan is 0.5 -- see lock frozen_configs)
_X9_HALF = ("FDC-BF", "FDC-MR[front3]", "FDC", "RECT-ck-HG-live", "RECT-ck-BF", "RECT-ck-BF+box")
_ADAPT_PJC = ("PJC-local", "PJC-menu")
DESIGN_GROUPS = {
    "v8a_pilot": None,      # filled by set_x9_groups() from the frozen X9 picks (same as v8a_full_a)
    "v8a_full_a": None,
    "v8a_full_b": (("B4-bal", "FIX-bal-fav", "Peace-fav-bal"),),
    "v8a_pilot_b": (("B4-bal", "FIX-bal-fav", "Peace-fav-bal"),),
    "v8b_full_a": (("FDC-BF", "FDC-MR[front3]", "FDC"),),
    "v8b_full_b": (("RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "RECT-ck-BF+box"), _ADAPT_PJC),
    "v8b_pilot": (("FDC-BF", "FDC-MR[front3]", "FDC", "RECT-ck-HG", "RECT-ck-HG-live", "HC-WoR", "RECT-ck-BF+box"),
                  _ADAPT_PJC),
    "v8c_full": (("FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "FIX-bal-fav", "Peace-fav-bal"), _ADAPT_PJC),
    "v8c_pilot": (("FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "FIX-bal-fav", "Peace-fav-bal"), _ADAPT_PJC),
}

__all__ = ["DESIGN_GROUPS", "x9_groups", "groups_for", "check_r2b", "required_rules", "compute_checks", "build_report",
           "validate_report", "R1_N_SEEDS"]


def x9_groups(rect_plan, hc_plan):
    half = list(_X9_HALF)
    if str(rect_plan) == "0.5":
        half.append("RECT-ck-HG")
    if str(hc_plan) == "0.5":
        half.append("HC-WoR")
    return (tuple(half), _ADAPT_PJC)


def groups_for(task, frozen_configs=None):
    g = DESIGN_GROUPS.get(task, "missing")
    if g == "missing":
        raise ValueError(f"{task}: no registered design group")
    if g is None:
        x9 = (frozen_configs or {}).get("X9") or {}
        try:
            g = x9_groups(x9["RECT-ck-HG"]["plan"], x9["HC-WoR"]["plan"])
        except (KeyError, TypeError) as e:
            raise ValueError(f"{task}: X9 design groups need the frozen X9 RECT-ck-HG / HC-WoR plans") from e
    return g


def check_r2b(rows, seeds, eps, methods):
    by = {}
    for r in rows:
        by.setdefault((int(r["seed"]), float(r["eps"])), {})[r["method"]] = r.get("arrival_digest")
    want = {(int(s), float(e)) for s in seeds for e in eps}
    bad = sorted(k for k, d in by.items() if set(d) != set(methods) or None in d.values() or len(set(d.values())) != 1)
    missing = sorted(want - set(by))
    return {"rule": "R2b", "pass": bool(by) and not bad and not missing, "n_streams": len(by),
            "bad": [list(k) for k in bad[:20]], "missing_streams": [list(k) for k in missing[:20]]}


def required_rules(task, is_eval):
    return ("R1", "R2", "R2b")


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds, frozen_configs=None):
    m = check_matrix(main_rows, methods, eps, seeds)
    if not m["complete"]:
        raise ValueError(f"{task}: planned main matrix incomplete: {m}")
    r1_seeds = list(seeds[:R1_N_SEEDS])
    mr = check_matrix(replica_rows, methods, eps, r1_seeds)
    if not mr["complete"]:
        raise ValueError(f"{task}: replica matrix (R1 seeds) incomplete: {mr}")
    checks = [check_r1(main_rows, replica_rows, methods, eps, r1_seeds)]
    for g in groups_for(task, frozen_configs):
        gg = tuple(x for x in g if x in methods)
        if len(gg) >= 2:
            checks.append(check_r2(main_rows, gg, seeds, eps))
    checks.append(check_r2b(main_rows, seeds, eps, methods))
    return _norm(checks)


def _bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256):
    return {"task_id": task, "is_eval": bool(is_eval), "addendum_sha256": add_sha, "code_sha256": code_sha256,
            "data_sha256": data_sha256, "results_content_sha256": content_hash(main_rows),
            "replica_content_sha256": content_hash(replica_rows), "n_rows": len(main_rows),
            "n_replica_rows": len(replica_rows), "methods": list(methods), "eps": [float(e) for e in eps],
            "seeds": [int(seeds[0]), int(seeds[-1]), len(seeds)],
            "r1_seeds": [int(s) for s in seeds[:R1_N_SEEDS]], "required_rules": list(required_rules(task, is_eval))}


def build_report(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256,
                 frozen_configs=None):
    if is_eval and not add_sha:
        raise ValueError("replica report refused: eval task without a locked-addendum sha256")
    rep = combine(task, compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds, frozen_configs))
    rep.update(_bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                         data_sha256))
    return rep


def validate_report(rep, task, is_eval, current_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                    data_sha256, sealed_content_sha256=None, frozen_configs=None):
    if not isinstance(rep, dict):
        raise ValueError(f"{task}: no replica report")
    if is_eval and not add_sha:
        raise ValueError(f"{task}: eval analysis without a locked-addendum sha256")
    if is_eval and not sealed_content_sha256:
        raise ValueError(f"{task}: eval analysis without a verified eval seal")
    if sealed_content_sha256 is not None and content_hash(current_rows) != sealed_content_sha256:
        raise ValueError(f"{task}: current main rows differ from the git-sealed content hash")
    checks = compute_checks(task, is_eval, current_rows, replica_rows, methods, eps, seeds, frozen_configs)
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
