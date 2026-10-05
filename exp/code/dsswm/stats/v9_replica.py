"""Lock-v9 addendum: replica rules (frozen before any v9 eval run; new file; the v6 / v8 rule implementations are
reused unchanged).

R1  (rerun): an independent process re-runs the FIRST 10 seeds of each v9 eval task for every method and eps of the
    task; canonical rows (all fields except timing) must be identical (``v6_replica.check_r1``).
R2  (design identity): within every (seed, eps), ``schedule_digest`` is identical across each registered frozen-design
    group of the task (``groups_for``), on the FULL main matrix.
R2b (arrival identity): within every (seed, eps), ``arrival_digest`` (arrival segment sequence + every pool's record
    order of the schedule the engine actually consumed) is identical across ALL methods of the task; the analysis
    also checks it across the tasks of a block (dense and K20 grids share the streams) and the cross-task result enters
    the replica status.
Report integrity exactly as v7 / v8 (recomputation on the actual rows, reconciliation of the stored report with every
binding, the git-sealed content hash for eval tasks).
"""
from __future__ import annotations

from .v6_replica import R1_N_SEEDS, check_matrix, check_r1, check_r2, combine, content_hash  # noqa: F401
from .v8_replica import _bindings, _norm, check_r2b

# frozen 50/50 design members of the CR9 / X9 tasks (RECT-ck-HG / HC-WoR join only if their frozen plan is 0.5)
_HALF_D = ("TU-FDC", "RECT-ck-BF-TU", "FDC-BF[K77]", "TU-LOC")
_HALF_K = ("FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "FDC-MR[front3]")
_HALF_X = ("FDC-HG", "FDC-LOC")
# Hillstrom: uniform 1/3 design members (rivals with the 'bal' plan share FDC-BF's design)
_C_BAL = {"HCZ6": ("FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "RECT-ck-HG[bal]"),
          "HCR6": ("FDC-BF", "RECT-ck-BF", "RECT-ck-BF+box", "RECT-ck-HG[bal]", "HC-WoR{prpl,0.5,-}[bal]")}
TASK_KIND = {"v9a_full_d": ("CR", "D"), "v9a_full_k": ("CR", "K"), "v9a_full_x": ("CR", "X"),
             "v9b_full_d": ("X9", "D"), "v9b_full_k": ("X9", "K"), "v9b_full_x": ("X9", "X"),
             "v9a_pilot_d": ("CR", "D"), "v9a_pilot_k": ("CR", "K"), "v9a_pilot_x": ("CR", "X"),
             "v9b_pilot_d": ("X9", "D"), "v9b_pilot_k": ("X9", "K"), "v9b_pilot_x": ("X9", "X"),
             "v9c_full_b": ("HCZ6", "C"), "v9c_full_a": ("HCR6", "C"),
             "v9c_pilot_b": ("HCZ6", "C"), "v9c_pilot_a": ("HCR6", "C")}

CROSS_PAIRS = (("TU-FDC@D", "FDC-BF@K"), ("HC-WoR@D", "HC-WoR@K"), ("PJC-local@D", "PJC-local@K"),
               ("TU-PJC@D", "PJC-local@K"), ("RECT-ck-BF-TU@D", "RECT-ck-BF+box@K"), ("RECT-ck-HG@D", "RECT-ck-HG@K"),
               ("FDC-BF[K77]@D", "FDC-BF@K"), ("TU-LOC@D", "FDC-LOC@K"))


def check_cross_grid(rows, seeds, eps, pairs=CROSS_PAIRS):
    """Cross-grid method-specific schedule identity (external reviewer v9 r1): for each pair (same method / same design on the
    dense and the K20 grid), (i) schedule_digest equal per (seed, eps) and (ii) the realised cell counts at every
    common visited block point equal ('block_counts'; this is what checks the count paths of the adaptive PJC rows,
    whose schedule digest does not encode the chosen arms).  Every pair needs >= 1 common block point per stream."""
    by = {(r["method"], int(r["seed"]), float(r["eps"])): r for r in rows}
    bad = []
    n = 0
    for a, b in pairs:
        for s in seeds:
            for e in eps:
                ra, rb = by.get((a, int(s), float(e))), by.get((b, int(s), float(e)))
                if ra is None or rb is None:
                    bad.append([a, b, int(s), "missing"])
                    continue
                n += 1
                if ra["schedule_digest"] != rb["schedule_digest"]:
                    bad.append([a, b, int(s), "schedule_digest"])
                ca, cb = ra.get("block_counts") or {}, rb.get("block_counts") or {}
                common = set(ca) & set(cb)
                if not common or any(ca[t] != cb[t] for t in common):
                    bad.append([a, b, int(s), "block_counts"])
    return {"rule": "R2c-cross-grid", "pass": n > 0 and not bad, "n_pairs_streams": n, "bad": bad[:20],
            "n_bad": len(bad)}


__all__ = ["CROSS_PAIRS", "check_cross_grid", "groups_for", "check_r2b", "compute_checks", "build_report", "validate_report", "R1_N_SEEDS", "TASK_KIND"]


def groups_for(task, frozen_configs):
    key, kind = TASK_KIND[task]
    if kind == "C":
        return (_C_BAL[key],)
    cfg = (frozen_configs or {}).get(key) or {}
    try:
        rect_half = str(cfg["RECT-ck-HG"]["plan"]) == "0.5"
        hc_half = str(cfg["HC-WoR"]["plan"]) == "0.5"
    except (KeyError, TypeError) as e:
        raise ValueError(f"{task}: design groups need the frozen {key} RECT-ck-HG / HC-WoR plans") from e
    base = {"D": _HALF_D, "K": _HALF_K, "X": _HALF_X}[kind]
    half = list(base)
    if kind in ("D", "K"):
        if rect_half:
            half.append("RECT-ck-HG")
        if hc_half:
            half.append("HC-WoR")
    return (tuple(half),)


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
