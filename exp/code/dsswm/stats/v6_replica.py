"""Lock-v6 addendum: replica rules R1-R3 (frozen before any eval run; new file).

R1 (rerun): after an eval task, an independent process re-runs the FIRST 10 SEEDS of the task's seed range for
   every method of the task and every eps of the task (block B: all three eps); canonical rows = all fields except
   TIMING_FIELDS must be identical.
R2 (design identity): within every (seed, eps), ``schedule_digest`` is identical across the methods of the task's
   registered 50/50 comparison group (COMPARISON_GROUPS; B excludes B4-bal because its frozen LR9 config is
   share = 0.4; adaptive / pool-design methods are not in any group).
R3 (block C prefix, v5 mapping): v5 k_stop = top-level ``rs_k_stop`` (r5_cr_main_a schema) or nested
   ``run_stream.k_stop`` (r5_cr_main_b schema); k_cut = k_stop, or the last checkpoint index of the v5 curve when v5
   never reached 12/15.  For every (rival, seed): v6 ``N80_pen`` == v5 ``N80_pen``; v6 ``k80`` == v5 k_stop (when v5
   completed); v6 ``n_cert_curve[:k_cut+1]`` == v5 ``n_cert_curve[:k_cut+1]`` (the v5 curve is PADDED to K entries
   after its stop; the padded tail is ignored); for each problem q with v5 ``cert_k[q] >= 0``: v6 cert_k[q] == v5
   cert_k[q]; with v5 cert_k[q] == -1: v6 cert_k[q] == -1 or > k_cut.  FDC: v6 ``N100_pen`` == ``r5_cr_fwer_audit``
   ``N100_pen`` (method 'FDC').  v5 files: r5_cr_main_{a,b}.
Status: 'pass' iff every applicable rule passes on complete inputs; otherwise 'fail' (with the offending keys).

Report integrity (v6 re-check item 1; final-check fix): the analysis RECOMPUTES R1-R3 (``compute_checks``) from the
actual replica rows, the current main rows and the frozen v5 references and uses only the recomputed status; the
stored report is a cache that must reconcile exactly (checks, status, bindings incl. the replica-row content hash).
 ``build_report`` refuses to write a report unless the task's FULL planned
matrix (methods x eps x seeds, each exactly once) is present; the report binds task id, locked-addendum sha256, code
hashes, data hashes and a content hash of the task's result rows (canonical, timing fields excluded).
``validate_report`` (used by the analysis) refuses a report whose task, addendum sha256 (None not allowed for eval),
code hashes, data hashes, seeds / eps / methods, rule set (R1, R2 and, for block-C eval tasks, R3) or content hash do
not match the CURRENT results; any appended / replaced / resumed row changes the content hash and invalidates it.
The runner additionally deletes an existing report whenever it writes new rows for the task.
"""

from __future__ import annotations

import hashlib
import json

TIMING_FIELDS = ("sec", "rs_sec_plan", "rs_sec_cert", "rs_sec_total")
R1_N_SEEDS = 10
COMPARISON_GROUPS = {
    "v6a_full": ("FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR"),
    "v6b_full_a": (),
    "v6b_full_b": ("FDC", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR"),
    "v6c_full_a": (),
    "v6c_full_b": ("FDC", "B4-bal"),
    # dev pilots (same groups)
    "v6a_pilot": ("FDC", "B4-bal", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR"),
    "v6b_pilot": ("FDC", "RECT-ck-HG", "RECT-ck-HG-live", "RECT-ck-Bern", "HC-WoR"),
    "v6c_pilot": ("FDC", "B4-bal"),
}

__all__ = ["TIMING_FIELDS", "R1_N_SEEDS", "COMPARISON_GROUPS", "canonical", "check_r1", "check_r2", "check_r3",
           "combine", "v5_k_stop", "content_hash", "check_matrix", "required_rules", "build_report", "validate_report",
           "compute_checks"]


def canonical(row):
    return {k: v for k, v in row.items() if k not in TIMING_FIELDS}


def _key(r):
    return (r["method"], float(r["eps"]), int(r["seed"]))


def check_r1(main_rows, replica_rows, methods, eps, seeds):
    want = {(m, float(e), int(s)) for m in methods for e in eps for s in seeds}
    main = {_key(r): canonical(r) for r in main_rows if _key(r) in want}
    rep = {}
    dup = []
    for r in replica_rows:
        k = _key(r)
        if k in rep:
            dup.append(k)
        rep[k] = canonical(r)
    missing = sorted(want - set(main)) + sorted(want - set(rep))
    diff = sorted(k for k in want & set(main) & set(rep) if main[k] != rep[k])
    ok = not missing and not diff and not dup
    return {"rule": "R1", "pass": ok, "n_compared": len(want), "missing": [list(k) for k in missing[:20]],
            "different": [list(k) for k in diff[:20]], "duplicates": [list(k) for k in dup[:20]]}


def check_r2(rows, group, seeds=None, eps=None):
    """Design identity; with seeds / eps given, every planned (seed, eps) stream must be present for every member."""
    if not group:
        return {"rule": "R2", "pass": True, "note": "no 50/50 comparison group in this task"}
    by = {}
    for r in rows:
        if r["method"] in group:
            by.setdefault((int(r["seed"]), float(r["eps"])), {})[r["method"]] = r["schedule_digest"]
    bad = sorted(k for k, d in by.items() if set(d) != set(group) or len(set(d.values())) != 1)
    missing = []
    if seeds is not None and eps is not None:
        missing = sorted({(int(s), float(e)) for s in seeds for e in eps} - set(by))
    return {"rule": "R2", "pass": bool(by) and not bad and not missing, "group": list(group), "n_streams": len(by),
            "bad": [list(k) for k in bad[:20]], "missing_streams": [list(k) for k in missing[:20]]}


def v5_k_stop(b):
    """k_stop of a v5 main-block record under either schema (None when v5 never reached 12/15)."""
    if "rs_k_stop" in b:
        return b["rs_k_stop"]
    rs = b.get("run_stream")
    if isinstance(rs, dict) and "k_stop" in rs:
        return rs["k_stop"]
    raise KeyError("v5 record has neither rs_k_stop nor run_stream.k_stop")


def check_r3(rows, v5_main_rows, v5_fwer_rows, rivals, seeds):
    v5 = {(r["method"], int(r["seed"])): r for r in v5_main_rows}
    fa = {int(r["seed"]): r for r in v5_fwer_rows if r["method"] == "FDC"}
    v6 = {(r["method"], int(r["seed"])): r for r in rows}
    bad = []
    for m in rivals:
        for s in seeds:
            a, b = v6.get((m, s)), v5.get((m, s))
            if a is None or b is None:
                bad.append([m, s, "missing"])
                continue
            if a["N80_pen"] != b["N80_pen"]:
                bad.append([m, s, "N80_pen"])
            ks = v5_k_stop(b)
            c5 = b["n_cert_curve"]
            k_cut = ks if ks is not None else len(c5) - 1
            if b.get("completed") and a["k80"] != ks:
                bad.append([m, s, "k80"])
            if a["n_cert_curve"][:k_cut + 1] != c5[:k_cut + 1]:
                bad.append([m, s, "n_cert_curve"])
            for q, (k6, k5) in enumerate(zip(a["cert_k"], b["cert_k"])):
                if (k5 >= 0 and k6 != k5) or (k5 < 0 and not (k6 < 0 or k6 > k_cut)):
                    bad.append([m, s, f"cert_k[{q}]"])
                    break
    if "FDC" in {r["method"] for r in rows}:
        for s in seeds:
            a = v6.get(("FDC", s))
            if a is None or s not in fa or a["N100_pen"] != fa[s]["N100_pen"]:
                bad.append(["FDC", s, "N100_pen vs r5_cr_fwer_audit"])
    return {"rule": "R3", "pass": not bad, "n_bad": len(bad), "bad": bad[:30]}


def combine(task, checks):
    ok = bool(checks) and all(c["pass"] for c in checks)
    return {"task_id": task, "status": "pass" if ok else "fail", "checks": checks}


# ---------------------------------------------------------------------------------------------- report integrity
def content_hash(rows):
    """sha256 of the task's result rows: canonical (timing excluded), sorted by (method, eps, seed)."""
    canon = sorted((json.dumps(canonical(r), sort_keys=True, default=str) for r in rows))
    return hashlib.sha256("\n".join(canon).encode()).hexdigest()


def check_matrix(rows, methods, eps, seeds):
    want = {(m, float(e), int(s)) for m in methods for e in eps for s in seeds}
    seen, dup, extra = set(), [], []
    for r in rows:
        k = _key(r)
        if k not in want:
            extra.append(k)
        elif k in seen:
            dup.append(k)
        seen.add(k)
    missing = sorted(want - seen)
    ok = not missing and not dup and not extra
    return {"complete": ok, "n_expected": len(want), "n_rows": len(rows), "missing": [list(k) for k in missing[:20]],
            "duplicates": [list(k) for k in dup[:20]], "extra": [list(k) for k in extra[:20]]}


def required_rules(task, is_eval):
    return ("R1", "R2", "R3") if (task.startswith("v6c_") and is_eval) else ("R1", "R2")


def _norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds, v5_main=None, v5_fwer=None):
    """Recompute R1-R3 from the ACTUAL rows (main results, replica rows, frozen v5 references).  Refuses when the main
    matrix or the replica matrix (R1 seeds x methods x eps) is incomplete, or when a block-C eval task lacks the v5
    references."""
    m = check_matrix(main_rows, methods, eps, seeds)
    if not m["complete"]:
        raise ValueError(f"{task}: planned main matrix incomplete: {m}")
    r1_seeds = list(seeds[:R1_N_SEEDS])
    mr = check_matrix(replica_rows, methods, eps, r1_seeds)
    if not mr["complete"]:
        raise ValueError(f"{task}: replica matrix (R1 seeds) incomplete: {mr}")
    checks = [check_r1(main_rows, replica_rows, methods, eps, r1_seeds),
              check_r2(main_rows, COMPARISON_GROUPS.get(task, ()), seeds, eps)]
    if "R3" in required_rules(task, is_eval):
        if v5_main is None or v5_fwer is None:
            raise ValueError(f"{task}: R3 needs the frozen v5 references")
        checks.append(check_r3(main_rows, v5_main, v5_fwer, [x for x in methods if x != "FDC"], seeds))
    return _norm(checks)


def _bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256,
              v5_main, v5_fwer):
    return {"task_id": task, "is_eval": bool(is_eval), "addendum_sha256": add_sha, "code_sha256": code_sha256,
            "data_sha256": data_sha256, "results_content_sha256": content_hash(main_rows),
            "replica_content_sha256": content_hash(replica_rows),
            "v5_reference_sha256": None if v5_main is None else content_hash(list(v5_main) + list(v5_fwer or [])),
            "n_rows": len(main_rows), "n_replica_rows": len(replica_rows), "methods": list(methods),
            "eps": [float(e) for e in eps], "seeds": [int(seeds[0]), int(seeds[-1]), len(seeds)],
            "r1_seeds": [int(s) for s in seeds[:R1_N_SEEDS]], "required_rules": list(required_rules(task, is_eval))}


def build_report(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256, data_sha256,
                 v5_main=None, v5_fwer=None):
    """The report is a CACHE of ``compute_checks`` plus bindings; it is never trusted on its own."""
    if is_eval and not add_sha:
        raise ValueError("replica report refused: eval task without a locked-addendum sha256")
    checks = compute_checks(task, is_eval, main_rows, replica_rows, methods, eps, seeds, v5_main, v5_fwer)
    rep = combine(task, checks)
    rep.update(_bindings(task, is_eval, main_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                         data_sha256, v5_main, v5_fwer))
    return rep


def validate_report(rep, task, is_eval, current_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                    data_sha256, v5_main=None, v5_fwer=None, sealed_content_sha256=None):
    """RECOMPUTE R1-R3 from the actual main rows, replica rows and v5 references and return the RECOMPUTED status
    ('pass' / 'fail').  The stored report must reconcile exactly with the recomputation (checks, status, every
    binding incl. the replica-row content hash); any disagreement, a missing report, or incomplete rows -> ValueError.
    The report's own status / pass fields never decide anything.  For eval tasks the current main rows must also
    equal the git-committed eval seal (``sealed_content_sha256``, from ``v6_seal.verify_seal``); R2 runs on the FULL
    main matrix (all planned seeds x eps), R1 on the first 10 seeds."""
    if not isinstance(rep, dict):
        raise ValueError(f"{task}: no replica report")
    if is_eval and not add_sha:
        raise ValueError(f"{task}: eval analysis without a locked-addendum sha256")
    if is_eval and not sealed_content_sha256:
        raise ValueError(f"{task}: eval analysis without a verified eval seal")
    if sealed_content_sha256 is not None and content_hash(current_rows) != sealed_content_sha256:
        raise ValueError(f"{task}: current main rows differ from the git-sealed content hash")
    checks = compute_checks(task, is_eval, current_rows, replica_rows, methods, eps, seeds, v5_main, v5_fwer)
    status = combine(task, checks)["status"]
    want = _bindings(task, is_eval, current_rows, replica_rows, methods, eps, seeds, add_sha, code_sha256,
                     data_sha256, v5_main, v5_fwer)
    errs = [k for k, v in want.items() if _norm(rep.get(k)) != _norm(v)]
    if _norm(rep.get("checks")) != checks:
        errs.append("checks differ from the recomputation")
    if rep.get("status") != status:
        errs.append(f"status {rep.get('status')!r} != recomputed {status!r}")
    if errs:
        raise ValueError(f"{task}: replica report does not reconcile with the recomputation: {errs}")
    return status
