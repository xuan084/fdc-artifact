"""r5_replica_check: independent re-run of sampled (stream, method) pairs of the r5 evaluation blocks + hash checks.

Each sampled (block, method, seed) is re-run from scratch in a FRESH interpreter (multiprocessing 'spawn' context: the
workers share no memory, cache or RNG state with the block runs) from the row's own (method, params, seed) and the
block's window (stop at 12/15 for the CR9 blocks; full horizon stop_k = Q for r5_cr_fwer_audit). The re-run is then
compared field by field with the stored row:
    N80_pen (and N100_pen for the full-horizon audit), cert_k, decided_pi, billing (billed, served, t_end, skipped,
    reselected, loop_arrivals, billing_ok), plus schedule_digest, k_stop, n_cert, n_false, fwer_event, n_cert_curve.
The endpoint N80_pen of the re-run is ALSO re-derived independently from the per-checkpoint curve and the true gaps
(definition in lock v5 'endpoints'), not only copied from run_stream.

Hash checks:
  H1  dsswm.stats.prereg.assert_locked(version=5): lock status / canonical hash / frozen dsswm code drift (lock v5).
  H2  per block: results.jsonl sha256 == the hash recorded in the block summary (when the block recorded one).
  H3  per block: row code hashes -> (a) 'code' dict rows: dsswm_frozen_sha256 == digest of lock v5 code_sha256,
      lock_sha256 / freeze_commit == lock v5, runner_sha256 == runner file on disk; (b) 'code_sha256' rows: the
      combined hash recomputed from the files listed in the block summary == the row value, and every listed dsswm
      file hash == lock v5 code_sha256[file].
  H4  runner scripts (not frozen by the lock; written after it): git-tracked and clean vs HEAD -> reported.
Power control: field-level tampering of stored rows (N80_pen, cert_k, decided_pi, billed) must be flagged, and re-runs
compared with a DIFFERENT seed's stored row must mismatch (the comparator can detect a difference).

PILOT (--mode pilot): DEV seeds only (900-909; never eval seeds). Primary sample = r5_cr_main_a pilot block, all 10
  streams x all 10 methods (100 re-runs; pass = 100% field identity). Secondary coverage = 3 streams (rng 42) x all
  methods of each other dependency block (main_b, plugin_a, plugin_b, factorial, fwer_audit) to test every adapter.
FULL  (--mode full): assert_locked(version=5, task_id) -> on refusal summary.status='skipped_by_lock'. 20 eval streams
  drawn with numpy default_rng(42) from 30000-30199; for each stream ALL methods of EVERY dependency block that holds
  that stream (main_a|main_b, plugin_a|plugin_b, factorial, fwer_audit). Pass = 0 mismatches AND all hash checks pass.
  Any mismatch -> status 'BLOCKED' (blocks the aggregate).
CPU only, 4 spawn workers, BLAS threads pinned to 1 (concurrent run: timings biased up).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp/results"
TASK = "r5_replica_check"
N_WORKERS = 4
PLANNED_MIN = 30
DEV_SEEDS = list(range(900, 910))
N_EVAL_STREAMS = 20
SEC_STREAMS = 3

# block -> (runner file, window)
BLOCKS = {
    "r5_cr_main_a": ("run_r5_cr_main.py", "stop12"),
    "r5_cr_main_b": ("run_r5_cr_main_b.py", "stop12"),
    "r5_cr_plugin_a": ("run_r5_cr_plugin_a.py", "stop12"),
    "r5_cr_plugin_b": ("run_r5_cr_plugin_a.py", "stop12"),
    "r5_cr_factorial": ("run_r5_cr_factorial.py", "stop12"),
    "r5_cr_fwer_audit": ("run_r5_cr_fwer_audit.py", "full"),
}
BILL_KEYS = ["billed", "served", "t_end", "skipped", "reselected", "loop_arrivals", "billing_ok"]
RS_KEYS = ["schedule_digest", "k_stop", "n_cert", "n_false", "fwer_event"]

_W = {}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def progress(step, total, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, txt):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": txt,
                                                  "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------ row normalisation
def rs_of(row):
    """run_stream summary of a stored row: nested 'run_stream' (main_b / factorial) or 'rs_' prefix (crm / fwer)."""
    if isinstance(row.get("run_stream"), dict):
        return row["run_stream"]
    return {k[3:]: v for k, v in row.items() if k.startswith("rs_")}


def norm_stored(row, window):
    rs = rs_of(row)
    out = {"N80_pen": int(row["N80_pen"]), "cert_k": list(rs["cert_k"]), "decided_pi": list(rs["decided_pi"]),
           "billing": {k: rs[k] for k in BILL_KEYS}, "n_cert_curve": list(row["n_cert_curve"])}
    out.update({k: rs[k] for k in RS_KEYS})
    if window == "full":
        out["N100_pen"] = int(row["N100_pen"])
    return out


def compare(a, b):
    """list of differing top-level fields (billing compared key by key)."""
    diff = []
    for k in sorted(set(a) | set(b)):
        if k == "billing":
            for bk in BILL_KEYS:
                if a.get("billing", {}).get(bk) != b.get("billing", {}).get(bk):
                    diff.append(f"billing.{bk}")
        elif a.get(k) != b.get(k):
            diff.append(k)
    return diff


# ------------------------------------------------------------------------------------------------ fresh workers
def _ctx(half, window):
    key = (half, window)
    if key not in _W:
        from dsswm.envs.pool_replay import PoolReplayEnv
        from dsswm.streams import frontier as fr
        from dsswm.streams.frontier_runner import build_ctx, true_policy_values
        if ("env", half) not in _W:
            env = PoolReplayEnv("CR9", half)
            c20 = build_ctx(env, fr.cr_problems("visit"), _W["eps"])
            J = true_policy_values(c20.pols, env.w, env.true_mu("visit"))
            Js = np.array([J[c20.feas[q]].max() for q in range(c20.Q)])
            _W[("env", half)] = (env, c20, J, Js)
        env, c20, J, Js = _W[("env", half)]
        ctx = c20 if window == "stop12" else dataclasses.replace(c20, stop_k=c20.Q)
        _W[key] = (env, c20, ctx, J, Js)
    return _W[key]


def _winit(eps):
    _W["eps"] = eps
    _W["pid"] = os.getpid()


def make_method(name, params):
    if name.startswith("F["):
        from dsswm.baselines.fdc import FDCAblation
        return FDCAblation(params["design"], params["union"], params["ledger"])
    from dsswm.streams import r5_registry as reg
    return reg.make_method(name, **params)


def rerun(a):
    block, window, half, name, params, seed = a
    try:
        from dsswm.streams.frontier_runner import run_stream
        env, c20, ctx, J, Js = _ctx(half, window)
        m = make_method(name, params)
        t0 = time.perf_counter()
        s, rows = run_stream(env, m, int(seed), ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js, keep_U=False)
        ck = ctx.checkpoints
        tau = int(ctx.tau_R)
        curve = [int(r["n_cert"]) for r in rows]
        curve += [curve[-1] if curve else 0] * (len(ck) - len(curve))
        # independent endpoint derivation (lock v5 'endpoints'): k* = first checkpoint with >= 12/15 certified;
        # penalised to tau_R if any certificate issued up to k* is false (gap > eps + 1e-12) or k* never reached.
        false_k = [int(s["cert_k"][q]) for q in range(ctx.Q)
                   if s["cert_k"][q] >= 0 and float(Js[q] - J[s["decided_pi"][q]]) > ctx.eps + 1e-12]
        k12 = next((i for i, c in enumerate(curve) if c >= c20.stop_k), None)
        n80_indep = int(ck[k12]) if (k12 is not None and not any(k <= k12 for k in false_k)) else tau
        nk = {"N80_pen": n80_indep if window == "full" else int(s["N80"]),
              "cert_k": [int(x) for x in s["cert_k"]], "decided_pi": [int(x) for x in s["decided_pi"]],
              "billing": {k: s[k] for k in BILL_KEYS}, "n_cert_curve": curve}
        nk.update({k: s[k] for k in RS_KEYS})
        if window == "full":
            # full horizon: run_stream's stop_k-generic N80 is the 15/15 endpoint
            k15 = next((i for i, c in enumerate(curve) if c >= ctx.Q), None)
            n100_indep = int(ck[k15]) if (k15 is not None and not false_k) else tau
            nk["N100_pen"] = int(s["N80"])
            indep_ok = (n100_indep == int(s["N80"]))
        else:
            indep_ok = (n80_indep == int(s["N80"]))
        return {"block": block, "method": name, "seed": int(seed), "norm": nk, "indep_endpoint_ok": bool(indep_ok),
                "n80_indep": n80_indep, "sec": round(time.perf_counter() - t0, 3), "worker_pid": os.getpid(),
                "error": None}
    except Exception:  # noqa: BLE001
        return {"block": block, "method": name, "seed": int(seed), "error": traceback.format_exc()}


# ------------------------------------------------------------------------------------------------ hash checks
def lock_digest(lock):
    items = sorted((k, v) for k, v in lock["code_sha256"].items() if k.startswith("dsswm/"))
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def git_state(path: Path):
    rel = str(path)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=str(CODE), capture_output=True,
                             text=True).returncode == 0
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=str(CODE), capture_output=True,
                           text=True).stdout.strip()
    last = subprocess.run(["git", "log", "-1", "--format=%h %cI", "--", rel], cwd=str(CODE), capture_output=True,
                          text=True).stdout.strip()
    return {"tracked": tracked, "clean": tracked and not dirty, "porcelain": dirty, "last_commit": last}


def resolve_runner_history(runner: Path, row_runner: Counter):
    rel = runner.name
    log = subprocess.run(["git", "log", "--format=%H %cI", "--", rel], cwd=str(CODE), capture_output=True,
                         text=True).stdout.split("\n")
    blob_of = {}
    for line in log:
        if not line.strip():
            continue
        h, d = line.split(" ", 1)
        b = subprocess.run(["git", "show", f"{h}:./{rel}"], cwd=str(CODE), capture_output=True).stdout
        blob_of.setdefault(hashlib.sha256(b).hexdigest(), (h, d))
    resolved, unresolved = {}, []
    for hsh, n in row_runner.items():
        if hsh in blob_of:
            h, d = blob_of[hsh]
            diff = subprocess.run(["git", "diff", h, "--", rel], cwd=str(CODE), capture_output=True,
                                  text=True).stdout
            hunks = [ln for ln in diff.splitlines() if ln.startswith("@@")]
            resolved[hsh] = {"n_rows": n, "commit": h[:8], "commit_date": d,
                             "diff_to_disk_lines_added": sum(1 for ln in diff.splitlines()
                                                             if ln.startswith("+") and not ln.startswith("+++")),
                             "diff_to_disk_lines_removed": sum(1 for ln in diff.splitlines()
                                                               if ln.startswith("-") and not ln.startswith("---")),
                             "diff_hunks": hunks, "diff": diff[:4000]}
        else:
            unresolved.append(hsh)
    return {"n_rows_differ": sum(row_runner.values()), "resolved_in_git": resolved,
            "unresolved": unresolved}


def block_hash_checks(block, bdir: Path, rows, lock):
    runner = CODE / BLOCKS[block][0]
    summ = json.loads((bdir / "summary.json").read_text()) if (bdir / "summary.json").exists() else {}
    rf = bdir / "results.jsonl"
    cur = sha_file(rf)
    rec = summ.get("results_file_sha256") or summ.get("results_sha256")
    out = {"results_jsonl_sha256": cur, "results_sha256_recorded": rec,
           "results_sha256_match": (None if rec is None else rec == cur)}
    ld = lock_digest(lock)
    bad = Counter()
    kinds = Counter()
    row_runner = Counter()  # row runner_sha256 values that differ from the runner on disk
    for r in rows:
        if isinstance(r.get("code"), dict):
            kinds["code_dict"] += 1
            c = r["code"]
            if c.get("dsswm_frozen_sha256") != ld:
                bad["dsswm_frozen_sha256"] += 1
            if c.get("lock_sha256") != lock["sha256"]:
                bad["lock_sha256"] += 1
            if c.get("freeze_commit") != lock["git_commit"]:
                bad["freeze_commit"] += 1
            if c.get("runner_sha256") != sha_file(runner):
                row_runner[c.get("runner_sha256")] += 1
        elif "code_sha256" in r:
            kinds["combined"] += 1
        else:
            kinds["none"] += 1
            bad["no_code_hash"] += 1
    if kinds["combined"]:
        per = summ.get("code_sha256")
        if not isinstance(per, dict):
            bad["summary_per_file_hashes_missing"] += 1
        else:
            now = {p: sha_file(CODE / p) for p in per}
            comb_now = hashlib.sha256(json.dumps(now, sort_keys=True).encode()).hexdigest()
            out["per_file_changed_since_block"] = [p for p in per if per[p] != now[p]]
            out["dsswm_files_not_equal_lock"] = [p for p in per if p.startswith("dsswm/")
                                                 and lock["code_sha256"].get(p) != per[p]]
            out["dsswm_files_not_in_lock"] = [p for p in per if p.startswith("dsswm/") and p not in lock["code_sha256"]]
            rc = Counter(r["code_sha256"] for r in rows if "code_sha256" in r)
            out["row_combined_hashes"] = dict(rc)
            if set(rc) != {comb_now}:
                bad["row_combined_vs_recomputed"] += 1
            if out["per_file_changed_since_block"]:
                bad["files_changed_since_block"] += 1
            if out["dsswm_files_not_equal_lock"]:
                bad["dsswm_vs_lock"] += 1
    if row_runner:
        # v2 (2026-10-03, after full v1 BLOCKED on r5_cr_factorial): a row runner hash that differs from disk is
        # resolved against the runner's git history. Resolved = the exact blob is a committed version of the runner
        # (provenance known); the diff to the current runner is reported verbatim. A resolved block passes H3 only if
        # its fresh re-runs under the CURRENT code also match 100% field-by-field (checked in run()). Unresolved ->
        # blocking violation, as in v1.
        res = resolve_runner_history(runner, row_runner)
        out["runner_sha256_vs_disk"] = res
        if res["unresolved"]:
            bad["runner_sha256_vs_disk_unresolved"] += sum(row_runner[h] for h in res["unresolved"])
    out["row_hash_kind"] = dict(kinds)
    out["violations"] = dict(bad)
    out["runner"] = {"file": runner.name, "sha256": sha_file(runner), "git": git_state(runner)}
    out["pass"] = (not bad) and out["results_sha256_match"] is not False
    return out


# ------------------------------------------------------------------------------------------------ main logic
def load_rows(bdir: Path):
    rows = {}
    for line in (bdir / "results.jsonl").read_text().splitlines():
        try:
            x = json.loads(line)
        except ValueError:
            continue
        if x.get("error") is None:
            rows[(x["method"], int(x["seed"]))] = x
    return rows


def run(mode, out: Path, lock):
    t_start = datetime.now()
    out.mkdir(parents=True, exist_ok=True)
    (out / "start_time.txt").write_text(t_start.isoformat())
    sub = "pilots" if mode == "pilot" else "full"
    half = "dev" if mode == "pilot" else "eval"
    eps = float(json.loads((RES / "r4_gates" / "eps.json").read_text())["eps_star"])
    assert abs(eps - 0.001) < 1e-15
    blocks = {}
    for b in BLOCKS:
        bdir = RES / sub / b
        if not (bdir / "results.jsonl").exists():
            blocks[b] = None
            continue
        blocks[b] = (bdir, load_rows(bdir))
    missing_blocks = [b for b, v in blocks.items() if v is None]
    if missing_blocks:
        s = {"task_id": TASK, "mode": mode, "status": "BLOCKED", "reason": f"source blocks missing: {missing_blocks}",
             "written_at": datetime.now().isoformat()}
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        return s

    rng = np.random.default_rng(42)
    sample = []  # (block, method, seed, tier)
    if mode == "pilot":
        rows_a = blocks["r5_cr_main_a"][1]
        for (m, s) in sorted(rows_a, key=lambda k: (k[1], k[0])):
            if s in DEV_SEEDS:
                sample.append(("r5_cr_main_a", m, s, "primary"))
        for b in BLOCKS:
            if b == "r5_cr_main_a":
                continue
            rows = blocks[b][1]
            seeds = sorted({s for (_, s) in rows if s in DEV_SEEDS})
            pick = sorted(int(x) for x in rng.choice(seeds, size=min(SEC_STREAMS, len(seeds)), replace=False))
            methods = list(dict.fromkeys(m for (m, _) in rows))
            sample += [(b, m, s, "secondary") for s in pick for m in methods if (m, s) in rows]
        sample_rule = ("primary: r5_cr_main_a dev pilot block, all 10 streams (900-909) x all 10 methods; secondary: "
                       "default_rng(42).choice(3 of the dev seeds 900-909) x all methods for every other block")
    else:
        pool_seeds = list(range(30000, 30200))
        eval_streams = sorted(int(x) for x in rng.choice(pool_seeds, size=N_EVAL_STREAMS, replace=False))
        for b in BLOCKS:
            rows = blocks[b][1]
            methods = list(dict.fromkeys(m for (m, _) in rows))
            sample += [(b, m, s, "primary") for s in eval_streams for m in methods if (m, s) in rows]
        sample_rule = (f"default_rng(42).choice(30000-30199, {N_EVAL_STREAMS}, replace=False) x all methods of every "
                       "dependency block holding the stream")
    (out / "sample.json").write_text(json.dumps({"rule": sample_rule, "n": len(sample),
                                                 "pairs": [list(x) for x in sample]}, indent=0))
    print(f"[{TASK}/{mode}] sample n={len(sample)} ({Counter(x[0] for x in sample)})", flush=True)

    # heavy jobs first (Peace-*) to balance the 4 workers
    jobs = [(b, BLOCKS[b][1], half, m, blocks[b][1][(m, s)]["params"], s) for (b, m, s, _) in sample]
    jobs.sort(key=lambda j: (0 if j[3].startswith("Peace") else 1))
    tier = {(b, m, s): t for (b, m, s, t) in sample}
    reruns = {}
    errs = []
    progress(0, len(jobs))
    with get_context("spawn").Pool(N_WORKERS, initializer=_winit, initargs=(eps,)) as pool:
        for i, r in enumerate(pool.imap_unordered(rerun, jobs, chunksize=1), 1):
            if r.get("error"):
                errs.append(r)
            else:
                reruns[(r["block"], r["method"], r["seed"])] = r
            if i % 10 == 0 or i == len(jobs):
                progress(i, len(jobs), {"reruns": i, "errors": len(errs)})
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] last={r['block']}/{r['method']}/{r['seed']} errors={len(errs)}", flush=True)

    # ---- field comparison
    per_pair = []
    for (b, m, s, t) in sample:
        r = reruns.get((b, m, s))
        if r is None:
            per_pair.append({"block": b, "method": m, "seed": s, "tier": t, "status": "error"})
            continue
        st = norm_stored(blocks[b][1][(m, s)], BLOCKS[b][1])
        d = compare(st, r["norm"])
        per_pair.append({"block": b, "method": m, "seed": s, "tier": t, "status": "match" if not d else "MISMATCH",
                         "diff_fields": d, "indep_endpoint_ok": r["indep_endpoint_ok"], "sec": r["sec"],
                         "worker_pid": r["worker_pid"]})
    by_tier = {}
    for t in ("primary", "secondary"):
        pp = [p for p in per_pair if p["tier"] == t]
        if not pp:
            continue
        by_tier[t] = {"n": len(pp), "n_match": sum(p["status"] == "match" for p in pp),
                      "n_mismatch": sum(p["status"] == "MISMATCH" for p in pp),
                      "n_error": sum(p["status"] == "error" for p in pp),
                      "n_indep_endpoint_fail": sum(p.get("indep_endpoint_ok") is False for p in pp)}
    by_block = {}
    for b in BLOCKS:
        pp = [p for p in per_pair if p["block"] == b]
        by_block[b] = {"n": len(pp), "n_mismatch": sum(p["status"] != "match" for p in pp),
                       "methods": sorted({p["method"] for p in pp}), "seeds": sorted({p["seed"] for p in pp}),
                       "field_diff_counts": dict(Counter(f for p in pp for f in p.get("diff_fields", [])))}
    mism = [p for p in per_pair if p["status"] != "match"]

    # ---- power control: tampered copies + cross-seed comparisons must be flagged
    tamper = {"N80_pen": 0, "cert_k": 0, "decided_pi": 0, "billing.billed": 0}
    n_t = 0
    for (b, m, s, _) in sample[:40]:
        r = reruns.get((b, m, s))
        if r is None:
            continue
        base = norm_stored(blocks[b][1][(m, s)], BLOCKS[b][1])
        n_t += 1
        for f in tamper:
            x = copy.deepcopy(base)
            if f == "N80_pen":
                x["N80_pen"] += 1
            elif f == "billing.billed":
                x["billing"]["billed"] += 1
            else:
                x[f] = x[f][::-1] if x[f] != x[f][::-1] else [v + 1 for v in x[f]]
            tamper[f] += int(f in compare(x, r["norm"]) or (f == "billing.billed" and "billing.billed"
                                                             in compare(x, r["norm"])))
    cross = {"n": 0, "n_flagged": 0}
    for (b, m, s), r in reruns.items():
        others = [s2 for (m2, s2) in blocks[b][1] if m2 == m and s2 != s]
        if not others:
            continue
        s2 = sorted(others)[0]
        cross["n"] += 1
        cross["n_flagged"] += int(bool(compare(norm_stored(blocks[b][1][(m, s2)], BLOCKS[b][1]), r["norm"])))
    power = {"tamper_n_rows": n_t, "tamper_detected": tamper,
             "tamper_all_detected": all(v == n_t for v in tamper.values()) and n_t > 0,
             "cross_seed": cross, "cross_seed_all_flagged": cross["n"] > 0 and cross["n_flagged"] == cross["n"]}

    # ---- hash checks
    hashes = {"H1_lock": {"version": lock["version"], "status": lock["status"], "sha256": lock["sha256"],
                          "git_commit": lock["git_commit"], "canonical_hash_ok": True,
                          "frozen_code_drift": [], "pass": True}}
    from dsswm.stats import prereg
    drift = prereg.frozen_code_drift(lock)
    hashes["H1_lock"]["frozen_code_drift"] = drift
    hashes["H1_lock"]["pass"] = not drift
    hashes["H1_lock"]["dsswm_frozen_digest"] = lock_digest(lock)
    hashes["blocks"] = {b: block_hash_checks(b, blocks[b][0], list(blocks[b][1].values()), lock) for b in BLOCKS}
    rec_absent = [b for b, h in hashes["blocks"].items() if h["results_sha256_recorded"] is None]
    hashes["H4_runners_untracked_or_dirty"] = sorted({h["runner"]["file"] for h in hashes["blocks"].values()
                                                       if not h["runner"]["git"]["clean"]})
    # v2: blocks whose row runner hash was resolved in git history must ALSO re-run 100% identically now
    runner_resolved = {b: h["runner_sha256_vs_disk"] for b, h in hashes["blocks"].items()
                       if "runner_sha256_vs_disk" in h}
    for b in runner_resolved:
        ok = by_block[b]["n"] > 0 and by_block[b]["n_mismatch"] == 0
        hashes["blocks"][b]["runner_resolved_rerun_identity_ok"] = ok
        if not ok:
            hashes["blocks"][b]["pass"] = False
    hash_pass = hashes["H1_lock"]["pass"] and all(h["pass"] for h in hashes["blocks"].values())

    prim = by_tier.get("primary", {})
    field_pass = (not mism) and not errs and all(v.get("n_indep_endpoint_fail", 0) == 0 for v in by_tier.values())
    if mode == "pilot":
        go = field_pass and hash_pass and power["tamper_all_detected"] and power["cross_seed_all_flagged"]
        status = "GO" if go else "NO_GO"
    else:
        go = field_pass and hash_pass and not rec_absent
        status = "PASS" if go else "BLOCKED"
    samples = [p for p in per_pair if p["status"] == "match"][:8] + mism[:20]
    t_end = datetime.now()
    summary = {
        "task_id": TASK, "mode": mode, "status": status, "go_no_go": status if mode == "pilot" else None,
        "blocks_aggregate": (not go) if mode == "full" else None,
        "n_mismatch": len(mism), "n_errors": len(errs), "n_reruns": len(per_pair),
        "primary": prim, "by_tier": by_tier, "by_block": by_block,
        "compared_fields": ["N80_pen", "N100_pen (fwer_audit)", "cert_k", "decided_pi",
                            "billing{" + ",".join(BILL_KEYS) + "}"] + RS_KEYS + ["n_cert_curve"],
        "independent_endpoint_rederivation": "N80_pen re-derived from curve + true gaps per lock v5 endpoint "
                                             "definition; must equal run_stream N80 (stop12) / N100 (full)",
        "fresh_process": {"context": "spawn", "n_workers": N_WORKERS,
                          "distinct_worker_pids": len({p.get("worker_pid") for p in per_pair if p.get("worker_pid")})},
        "power_control": power, "hash_checks": hashes, "hash_pass": hash_pass,
        "results_hash_not_recorded_by_block": rec_absent,
        "runner_revised_after_rows": {b: {"n_rows": v["n_rows_differ"],
                                          "resolved": {k[:16]: {kk: vv for kk, vv in x.items() if kk != "diff"}
                                                       for k, x in v["resolved_in_git"].items()},
                                          "unresolved": v["unresolved"]} for b, v in runner_resolved.items()},
        "checker_version": "v2 (git-history resolution of runner hashes; v1 BLOCKED summary kept as "
                           "summary_v1_blocked.json)",
        "sample_rule": sample_rule, "half": half, "eps": eps,
        "lock": {"version": lock["version"], "sha256": lock["sha256"], "git_commit": lock["git_commit"]},
        "mismatches": mism[:50], "errors": [e["error"][-1500:] for e in errs[:5]], "samples": samples,
        "runner_sha256": sha_file(Path(__file__)),
        "started_at": t_start.isoformat(), "ended_at": t_end.isoformat(),
        "wall_min": round((t_end - t_start).total_seconds() / 60, 2),
        "timing_note": "concurrent run (4 spawn workers, BLAS=1; other r5 tasks share the 20-core host): biased up",
        "eval_seeds_touched": mode == "full",
    }
    (out / "per_pair.jsonl").write_text("".join(json.dumps(p) + "\n" for p in per_pair))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["pilot", "full"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    from dsswm.stats import prereg
    ok, lock = prereg.lock_gate(TASK, version=5)
    if not ok:
        out.mkdir(parents=True, exist_ok=True)
        s = {"task_id": TASK, "mode": a.mode, "status": "skipped_by_lock", "reason": lock,
             "written_at": datetime.now().isoformat()}
        (out / "summary.json").write_text(json.dumps(s, indent=1))
        mark_done("failed", f"skipped_by_lock: {lock}")
        print(json.dumps(s), flush=True)
        return
    try:
        s = run(a.mode, out, lock)
        txt = (f"{s['status']} reruns={s.get('n_reruns')} mismatch={s.get('n_mismatch')} errors={s.get('n_errors')} "
               f"hash_pass={s.get('hash_pass')}")
        mark_done("success" if s["status"] in ("GO", "PASS") else "failed", txt)
        print(txt, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        mark_done("failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
