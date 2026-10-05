"""r4_setup_fp_stats: setup + pilot for the round-4 statistics layer.

New / changed modules: dsswm/stats/fp_eb.py (WSR20 WoR empirical-Bernstein CS, binary variance UCB, L1 functional
width), dsswm/stats/cp.py (+ one-sided CP; old two-sided clopper_pearson unchanged), dsswm/stats/acceptance_r4.py
(S1 / S2 / E[V/(R v 1)] / E3 / completion lower bound / paired log ratio / penalised cost / Holm / E-ms(d)),
dsswm/tests/test_fp_stats_r4.py.

pilot: (1) unit tests (new file + full suite); (2) WoR CS time-uniform coverage on 17 finite pools x 3 deltas with
10^4 exact WoR replications each (seed 42), coverage over the whole path t = 1..N; (3) var UCB >= true variance on
E_var; (4) cp_upper_one_sided(4, 200); (5) external reviewer FCR counterexample through acceptance_r4; (6) diagnostic: WSR20 CS
vs. the Lemma-L2 Bernstein CI (x_v = 12.165) at the r4 ledger budget on Hillstrom-like cells.
Pass: all unit tests pass AND min coverage >= 1 - delta - 0.005 AND |cp_upper_one_sided(4,200) - 0.04518| <= 1e-5.
full: rerun the unit tests and write code sha256 for the lock.
Usage: run_r4_setup_fp_stats.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
TASK = "r4_setup_fp_stats"
PY = sys.executable
CODE_FILES = ["dsswm/stats/fp_eb.py", "dsswm/stats/cp.py", "dsswm/stats/acceptance_r4.py",
              "dsswm/tests/test_fp_stats_r4.py", "run_r4_setup_fp_stats.py"]
REPS = 10_000
DELTAS = [0.05, 0.1, 0.2]

from dsswm.stats.acceptance_r4 import (reviewer_fcr_counterexample, e_v_over_r, layer_safety,  # noqa: E402
                                       s1_fwer)
from dsswm.stats.cp import cp_upper_one_sided  # noqa: E402
from dsswm.stats.fp_eb import delta_per_cell, var_ucb_binary, wor_mean_cs  # noqa: E402
from dsswm.theory_checks.mc_l1 import bernstein_mu_ci  # noqa: E402


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                       "final_progress": fp,
                                                       "timestamp": datetime.now().isoformat()}))


def run_tests():
    out = {}
    for name, target in [("fp_stats_r4", "dsswm/tests/test_fp_stats_r4.py"), ("full_suite", "dsswm/tests")]:
        t = time.time()
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", target], cwd=HERE,
                           capture_output=True, text=True, timeout=1800)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        out[name] = {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}
        if r.returncode != 0:
            out[name]["failures"] = [ln for ln in r.stdout.splitlines() if ln.startswith(("FAILED", "ERROR"))][:30]
    return out


# ------------------------------------------------------------------------------------------------- coverage
def pool_configs():
    rng = np.random.default_rng(42)
    cfgs = []
    for N in (200, 1000, 3500):            # 3500 ~ a Hillstrom S=6 cell (64000 / 18)
        for mu in (0.01, 0.05, 0.15, 0.3, 0.5):
            k = max(1, int(round(mu * N)))
            pool = np.zeros(N)
            pool[:k] = 1.0
            cfgs.append({"cid": f"bin_N{N}_mu{mu}", "kind": "binary", "pool": pool, "R": 1.0})
    for N, a, b in ((500, 0.5, 2.0), (1500, 2.0, 2.0)):
        R = 3.0
        cfgs.append({"cid": f"nonbin_N{N}_beta{a}_{b}", "kind": "nonbinary", "pool": R * rng.beta(a, b, N), "R": R})
    return cfgs


def run_cov(args):
    cfg, delta, seed = args
    rng = np.random.default_rng(seed)
    pool, R = cfg["pool"], cfg["R"]
    N = len(pool)
    m = pool.mean()
    sig2 = pool.var()
    miss = 0
    var_viol = 0
    chunk = max(200, int(4e6 // N))
    widths = {f: [] for f in (0.1, 0.5, 0.9)}
    done = 0
    while done < REPS:
        r = min(chunk, REPS - done)
        X = np.stack([rng.permutation(pool) for _ in range(r)])
        lo, hi = wor_mean_cs(X, N, delta, R=R)
        bad = ((lo > m + 1e-12) | (hi < m - 1e-12)).any(axis=1)
        miss += int(bad.sum())
        if cfg["kind"] == "binary":
            v = var_ucb_binary((lo, hi))
            var_viol += int(((v < sig2 - 1e-12).any(axis=1) & ~bad).sum())
        for f in widths:
            t = max(1, int(f * N)) - 1
            widths[f].append(hi[:, t] - lo[:, t])
        done += r
    cov = 1 - miss / REPS
    return {"cid": cfg["cid"], "kind": cfg["kind"], "N": N, "mu": float(m / R), "delta": delta, "reps": REPS,
            "coverage": cov, "miss": miss, "pass": bool(cov >= 1 - delta - 0.005),
            "var_ucb_violations_on_E_var": var_viol if cfg["kind"] == "binary" else None,
            **{f"med_width_t{int(f * 100)}pct": float(np.median(np.concatenate(widths[f]))) / R for f in widths}}


# ------------------------------------------------------------------------------------------------- diagnostic
def width_diag(seed=42):
    """Per-cell mean-CI width and sigma_bar^2 at the r4 ledger budget (delta_var = 0.01 over C_var = 48 cells):
    WSR20 time-uniform CS (delta = 0.01/48) vs Lemma-L2 Bernstein CI with x_v = ln(2 * 48 * 20 / 0.01) = 12.165."""
    rng = np.random.default_rng(seed)
    d_cell = delta_per_cell(0.01, 48)
    x_v = math.log(2 * 48 * 20 / 0.01)
    rows = []
    for N, mu in ((3500, 0.10), (3500, 0.15), (3500, 0.20), (1200, 0.15), (14000, 0.15)):
        pool = np.zeros(N)
        pool[: int(round(mu * N))] = 1.0
        X = np.stack([rng.permutation(pool) for _ in range(200)])
        lo, hi = wor_mean_cs(X, N, d_cell)
        for frac in (0.05, 0.1, 0.25, 0.5, 0.9):
            t = int(frac * N)
            muhat = X[:, :t].mean(axis=1)
            blo, bhi = bernstein_mu_ci(muhat, np.full(200, t), np.full(200, N), x_v)
            rows.append({"N": N, "mu": mu, "n": t, "frac": frac,
                         "wsr_width": float(np.median(hi[:, t - 1] - lo[:, t - 1])),
                         "bern_width": float(np.median(bhi - blo)),
                         "wsr_sig2_ucb": float(np.median(var_ucb_binary((lo[:, t - 1], hi[:, t - 1])))),
                         "bern_sig2_ucb": float(np.median(var_ucb_binary((blo, bhi)))),
                         "true_sig2": mu * (1 - mu)})
    return {"delta_cell": d_cell, "x_v_bernstein": x_v, "rows": rows}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "concurrent_run": True,
               "started_at": datetime.now().isoformat()}
    try:
        progress(0, 4, "tests")
        summary["unit_tests"] = run_tests()
        tests_ok = all(v["returncode"] == 0 for v in summary["unit_tests"].values())
        cpv = cp_upper_one_sided(4, 200)
        summary["cp_check"] = {"cp_upper_one_sided_4_200": cpv, "target": 0.04518,
                               "abs_err": abs(cpv - 0.04518), "pass": abs(cpv - 0.04518) <= 1e-5,
                               "cp_upper_one_sided_5_200": cp_upper_one_sided(5, 200)}
        nf, nc = reviewer_fcr_counterexample()
        saf = layer_safety(nf, nc)
        summary["reviewer_fcr_counterexample"] = {
            "fwer": s1_fwer(nf > 0)["rate"], "E_V_over_R": e_v_over_r(nf, nc), "fcr_old": saf["S2"]["fcr_old"],
            "fcr_old_boot_upper": saf["S2"]["upper"], "S1_pass": saf["S1"]["pass"], "S2_pass": saf["S2"]["pass"],
            "pass": abs(saf["S2"]["fcr_old"] - 0.0898) < 1e-4}
        summary["code_sha256"] = {f: sha256(HERE / f) for f in CODE_FILES}
        if args.mode == "pilot":
            progress(1, 4, "wor_cs_coverage")
            jobs = [(cfg, d, 42 + 1000 * i + j) for i, cfg in enumerate(pool_configs())
                    for j, d in enumerate(DELTAS)]
            with ProcessPoolExecutor(max_workers=min(4, args.workers)) as ex:
                cov = list(ex.map(run_cov, jobs))
            with open(out_dir / "wor_cs_coverage.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(cov[0].keys()))
                w.writeheader()
                w.writerows(cov)
            min_slack = min(c["coverage"] - (1 - c["delta"]) for c in cov)
            summary["wor_cs_coverage"] = {
                "n_runs": len(cov), "reps_per_run": REPS, "deltas": DELTAS,
                "min_coverage_minus_target": min_slack,
                "worst": min(cov, key=lambda c: c["coverage"] - (1 - c["delta"])),
                "all_pass": all(c["pass"] for c in cov),
                "var_ucb_violations_on_E_var_total": int(sum(c["var_ucb_violations_on_E_var"] or 0 for c in cov)),
                "by_delta_min_coverage": {str(d): min(c["coverage"] for c in cov if c["delta"] == d)
                                          for d in DELTAS}}
            progress(2, 4, "width_diag")
            summary["width_diag_ledger"] = width_diag()
            gate = (tests_ok and summary["wor_cs_coverage"]["all_pass"] and summary["cp_check"]["pass"]
                    and summary["wor_cs_coverage"]["var_ucb_violations_on_E_var_total"] == 0
                    and summary["reviewer_fcr_counterexample"]["pass"])
        else:
            gate = tests_ok and summary["cp_check"]["pass"] and summary["reviewer_fcr_counterexample"]["pass"]
        summary["gate_pass"] = bool(gate)
        summary["go_no_go"] = "GO" if gate else "NO_GO"
        summary["runtime_sec"] = round(time.time() - t0, 1)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        progress(4, 4, "done", {"gate_pass": bool(gate)})
        mark_done("success" if gate else "failed", f"{args.mode} gate_pass={gate}")
        print(json.dumps({k: summary[k] for k in ("gate_pass", "runtime_sec")}))
    except Exception as e:  # noqa: BLE001
        summary["error"] = traceback.format_exc()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", f"error: {e}")
        raise


if __name__ == "__main__":
    main()
