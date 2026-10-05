"""r4_setup_quadknap: setup + G-dp for dsswm/certify/quadknap.py (QFC enumeration certificate, min_t-DP conservative
upper bound with four separately reported gaps, rounded-up proposal with original-cost re-check).

pilot: unit tests (new file + full suite); external reviewer counterexample; 100 budget-boundary instances (S <= 8, seed 42) with
gap decomposition; naive round-up control (must show violations => the harness has power); enumeration == brute force
on 30 instances (1e-12); timings (single certificate for HR6 / CR9 / HR8 / CR12 and single DP for S=16 / CR12).
Pass: 0 violations of U_DP >= U_enum AND 100% proposals feasible under original costs AND enum == brute force.
full: 1000 random instances with S in {4, 6, 8} (gap decomposition table) + the pilot checks. Pass: 0 violations.
Usage: run_r4_setup_quadknap.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
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
TASK = "r4_setup_quadknap"
PY = sys.executable
CODE_FILES = ["dsswm/certify/quadknap.py", "dsswm/tests/test_quadknap.py", "run_r4_setup_quadknap.py"]
GAP_KEYS = ("gap_i_cost_rounding", "gap_ii_t_grid", "gap_iii_minmax", "gap_iv_low_order")

from dsswm.certify import quadknap as qk  # noqa: E402
from dsswm.streams.frontier import Problem, cr_problems, enumerate_policies, hr_problems  # noqa: E402
from dsswm.tests.test_quadknap import boundary_problem, brute_U, random_stats  # noqa: E402


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
    for name, target in [("quadknap", "dsswm/tests/test_quadknap.py"), ("full_suite", "dsswm/tests")]:
        t = time.time()
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", target], cwd=HERE,
                           capture_output=True, text=True, timeout=1800)
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
        out[name] = {"returncode": r.returncode, "tail": tail, "sec": round(time.time() - t, 1)}
    return out


def reviewer_check():
    dhat, V = np.array([0.0, -1.0]), np.array([0.5, 2.0])
    U_enum = float(np.max(dhat + np.sqrt(2 * V)))
    U_cont, t = qk.minmax_t_cont(dhat, V, 1.0)
    U_grid, tg = qk.minmax_t_grid(dhat, V, 1.0, np.geomspace(0.05, 50, qk.DEFAULT_T_POINTS))
    ok = abs(U_enum - 1) < 1e-12 and abs(U_cont - 13 / 12) < 1e-9 and U_grid >= 13 / 12 - 1e-12
    return {"U_enum": U_enum, "U_minmax_cont": U_cont, "t_star_cont": t, "U_dp_grid40": U_grid, "t_star_grid": tg,
            "gap_minmax": U_cont - U_enum, "pass": bool(ok)}


def one_instance(args):
    """Worker: one instance -> gap row (+ optional naive control)."""
    idx, seed, S, A, regime, kind = args
    rng = np.random.default_rng(seed)
    stats = random_stats(rng, S, A, regime=regime)
    pols = enumerate_policies(S, A)
    L1 = qk.l1_qstar(S, A, 15, 20, 0.04)
    if kind == "boundary":
        p = boundary_problem(rng, stats, pols, L1)
    else:
        kappa = np.concatenate([[0.0], rng.uniform(0.2, 2.0, size=A - 1)])
        p = Problem("rnd", "visit", tuple(kappa), float(rng.uniform(0.05, 1.0)))
    g = qk.gap_report(stats, p, L1, pols=pols)
    r = qk.qfc_certificate_enum(stats, [p], L1, 0.0, pols=pols, pi_hat=[g["pi_hat"]])[0]
    naive = qk.qfc_upper_dp(stats, p, L1, pi_hat=g["pi_hat"], rounding="naive_up", track=False)
    row = {"idx": idx, "kind": kind, "regime": regime, "S": S, "A": A, "budget": p.budget,
           "kappa": "|".join(f"{k:.4g}" for k in p.kappa), "L1": L1,
           **{k: g[k] for k in ("U_enum", "U_iv", "U_iii", "U_ii", "U_dp", "gap_total", *GAP_KEYS, "b_bar",
                                "t_cont", "t_grid_star", "U_enum_at_enum_argmax", "proposal_Jhat_shortfall",
                                "dp_vs_enum_relaxed_maxabs", "n_feasible", "n_relaxed")},
           "violation": int(g["violation"]), "proposal_feasible": int(g["proposal_feasible"]),
           "proposal_fallback": int(g["proposal_fallback"]), "relaxed_superset": int(g["relaxed_superset"]),
           "pi_hat_equals_enum_argmax": int(tuple(g["pi_hat"]) == tuple(g["pi_hat_enum"])),
           "enum_matches_gap_report": int(abs(r["U"] - g["U_enum"]) <= 1e-12 * max(1, abs(r["U"]))),
           "naive_up_U": naive["U_dp"], "naive_up_violation": int(naive["U_dp"] < g["U_enum"] - 1e-12),
           "rel_gap": g["gap_total"] / g["U_enum"] if g["U_enum"] > 0 else float("nan")}
    sample = None
    if idx < 5:
        sample = {"idx": idx, "problem": {"kappa": list(p.kappa), "budget": p.budget},
                  "w": stats.w.tolist(), "mu_hat": stats.mu_hat.tolist(), "n": stats.n.tolist(),
                  "N": stats.N.tolist(), "var_ucb": stats.var_ucb.tolist(), "L1": L1, "pi_hat": list(g["pi_hat"]),
                  "worst_challenger_enum": list(r["worst"]), "worst_dhat": r["worst_dhat"], "worst_V": r["worst_V"],
                  "worst_b": r["worst_b"], "gaps": {k: g[k] for k in GAP_KEYS}, "U_enum": g["U_enum"],
                  "U_dp": g["U_dp"]}
    return row, sample


def brute_check(n_inst=30, seed=4242):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in range(n_inst):
        S, A = [(6, 3), (5, 2), (4, 3), (7, 2)][i % 4]
        stats = random_stats(rng, S, A, regime="unit" if i % 2 else "cr")
        L1 = qk.l1_qstar(S, A, 15, 20, 0.04)
        probs = (hr_problems() if A == 3 else cr_problems())[i % 5::5]
        for r, p in zip(qk.qfc_certificate_enum(stats, probs, L1, 0.002), probs):
            bf = brute_U(stats, p, L1, r["pi_hat"])
            worst = max(worst, abs(bf - r["U"]) / max(1.0, abs(bf)))
    return {"n_instances": n_inst, "max_rel_abs_diff": worst, "pass": bool(worst <= 1e-12)}


def timings(seed=42):
    rng = np.random.default_rng(seed)
    out = {}
    layers = [("HR6_S6_A3", 6, 3, hr_problems()), ("CR9_S9_A2", 9, 2, cr_problems()),
              ("HR8_S8_A3", 8, 3, hr_problems()), ("CR12_S12_A2", 12, 2, cr_problems())]
    for name, S, A, probs in layers:
        stats = random_stats(rng, S, A, regime="cr")
        pols = enumerate_policies(S, A)
        L1 = qk.l1_qstar(S, A, len(probs), 20, 0.04)
        qk.qfc_certificate_enum(stats, probs, L1, 0.002, pols=pols)          # warm-up
        reps = 20 if S <= 9 else 5
        t = time.perf_counter()
        for _ in range(reps):
            qk.qfc_certificate_enum(stats, probs, L1, 0.002, pols=pols)
        dt = (time.perf_counter() - t) / reps * 1000
        out[name] = {"policies": int(A ** S), "problems": len(probs), "ms_all_problems": round(dt, 3),
                     "ms_per_certificate": round(dt / len(probs), 3)}
    for name, S, A, probs in [("S16_A3_DP", 16, 3, hr_problems()), ("CR12_S12_A2_DP", 12, 2, cr_problems())]:
        stats = random_stats(rng, S, A, regime="cr")
        L1 = qk.l1_qstar(S, A, len(probs), 20, 0.04)
        qk.qfc_upper_dp(stats, probs[0], L1)
        t = time.perf_counter()
        for p in probs:
            qk.qfc_upper_dp(stats, p, L1)
        dt = (time.perf_counter() - t) / len(probs) * 1000
        out[name] = {"policies": int(A ** S), "t_points": qk.DEFAULT_T_POINTS, "cost_units": qk.DEFAULT_UNITS,
                     "ms_per_dp_certificate_incl_proposal": round(dt, 3)}
    out["note"] = "concurrent run (up to 4 tasks share 20 cores); single-thread numpy (OMP/MKL/OPENBLAS=1)"
    return out


def summarize(rows):
    def agg(sel):
        if not sel:
            return {}
        d = {"n": len(sel), "violations": int(sum(r["violation"] for r in sel)),
             "proposal_feasible_rate": float(np.mean([r["proposal_feasible"] for r in sel])),
             "naive_up_violations": int(sum(r["naive_up_violation"] for r in sel)),
             "pi_hat_equals_enum_argmax_rate": float(np.mean([r["pi_hat_equals_enum_argmax"] for r in sel])),
             "max_dp_vs_enum_relaxed": float(np.nanmax([r["dp_vs_enum_relaxed_maxabs"] for r in sel])),
             "min_gap_component": float(min(min(r[k] for k in GAP_KEYS) for r in sel))}
        for k in (*GAP_KEYS, "gap_total"):
            v = np.array([r[k] for r in sel])
            d[k] = {"mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max())}
        rg = np.array([r["rel_gap"] for r in sel if np.isfinite(r["rel_gap"])])
        if rg.size:
            d["rel_gap_total"] = {"mean": float(rg.mean()), "median": float(np.median(rg)), "max": float(rg.max())}
        return d
    out = {"all": agg(rows)}
    for key in sorted({(r["kind"], r["regime"], r["S"]) for r in rows}):
        out[f"{key[0]}|{key[1]}|S{key[2]}"] = agg([r for r in rows if (r["kind"], r["regime"], r["S"]) == key])
    return out


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = sorted({(r["regime"], r["S"]) for r in rows})
    labels = [f"{g[0]}\nS={g[1]}" for g in groups]
    names = {"gap_i_cost_rounding": "(i) cost rounding", "gap_ii_t_grid": "(ii) t grid",
             "gap_iii_minmax": "(iii) min-max order", "gap_iv_low_order": "(iv) low-order globalisation"}
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(groups)), 3.6))
    bottom = np.zeros(len(groups))
    for k, c in zip(GAP_KEYS, colors):
        vals = []
        for g in groups:
            sel = [r for r in rows if (r["regime"], r["S"]) == g and np.isfinite(r["rel_gap"])]
            vals.append(np.mean([r[k] / r["U_enum"] for r in sel]) if sel else 0.0)
        ax.bar(labels, vals, bottom=bottom, color=c, label=names[k])
        bottom += np.array(vals)
    ax.set_ylabel("mean gap / U_enum")
    ax.set_title("Relaxation gaps of the min_t-DP upper bound (S <= 8, vs. enumeration)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "concurrent_run": True,
               "started_at": datetime.now().isoformat(), "workers": args.workers}
    try:
        progress(0, 5, "unit_tests")
        summary["unit_tests"] = run_tests()
        progress(1, 5, "reviewer_and_bruteforce")
        summary["reviewer_counterexample"] = reviewer_check()
        summary["enum_vs_bruteforce"] = brute_check()
        progress(2, 5, "instances")
        rng = np.random.default_rng(42)
        jobs = []
        for i in range(100):                                           # pilot: 100 boundary instances (S <= 8)
            S = int(rng.integers(3, 9))
            A = 3 if (i % 2 == 0 and S <= 6) else 2
            jobs.append((i, int(rng.integers(2 ** 31)), S, A, "unit" if i % 3 else "cr", "boundary"))
        if args.mode == "full":
            for i in range(1000):
                S = [4, 6, 8][i % 3]
                A = 3 if (S <= 6 and i % 2 == 0) else 2
                jobs.append((100 + i, int(rng.integers(2 ** 31)), S, A, "unit" if i % 2 else "cr", "random"))
        rows, samples = [], []
        with ProcessPoolExecutor(max_workers=min(4, args.workers)) as ex:
            for k, (row, sample) in enumerate(ex.map(one_instance, jobs, chunksize=5)):
                rows.append(row)
                if sample:
                    samples.append(sample)
                if k % 100 == 0:
                    progress(2, 5, "instances", {"done": k, "total": len(jobs)})
        with open(out_dir / "gap_decomposition.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        for s in samples:
            (out_dir / "samples" / f"instance_{s['idx']:03d}.json").write_text(json.dumps(s, indent=1))
        summary["gap_summary"] = summarize(rows)
        progress(3, 5, "timings")
        summary["timings_ms"] = timings()
        progress(4, 5, "plot")
        try:
            plot(rows, out_dir / "gap_stacked_bar.png")
        except Exception as e:  # noqa: BLE001
            summary["plot_error"] = repr(e)
        bnd = [r for r in rows if r["kind"] == "boundary"]
        crit = {
            "violations_U_dp_lt_U_enum": int(sum(r["violation"] for r in rows)),
            "boundary_instances": len(bnd),
            "proposal_feasible_rate": float(np.mean([r["proposal_feasible"] for r in rows])),
            "enum_eq_bruteforce": summary["enum_vs_bruteforce"]["pass"],
            "reviewer_13_12": summary["reviewer_counterexample"]["pass"],
            "dp_equals_enum_on_relaxed_set_per_t": bool(max(r["dp_vs_enum_relaxed_maxabs"] for r in rows) <= 1e-12),
            "relaxed_superset_all": bool(all(r["relaxed_superset"] for r in rows)),
            "naive_roundup_control_violations": int(sum(r["naive_up_violation"] for r in rows)),
            "unit_tests_pass": all(v["returncode"] == 0 for v in summary["unit_tests"].values()),
        }
        crit["pass"] = bool(crit["violations_U_dp_lt_U_enum"] == 0 and crit["proposal_feasible_rate"] == 1.0
                            and crit["enum_eq_bruteforce"] and crit["reviewer_13_12"] and crit["unit_tests_pass"]
                            and crit["relaxed_superset_all"] and crit["dp_equals_enum_on_relaxed_set_per_t"])
        summary["pass_criteria"] = crit
        summary["go_no_go"] = "GO" if crit["pass"] else "NO_GO"
        summary["code_sha256"] = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
        summary["wall_sec"] = round(time.time() - t0, 1)
        summary["finished_at"] = datetime.now().isoformat()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
        progress(5, 5, "done", {"pass": crit["pass"]})
        mark_done("success" if crit["pass"] else "failed",
                  f"G-dp {'PASS' if crit['pass'] else 'FAIL'}: violations={crit['violations_U_dp_lt_U_enum']}, "
                  f"proposal_feasible={crit['proposal_feasible_rate']:.3f}, brute={crit['enum_eq_bruteforce']}")
    except Exception as e:  # noqa: BLE001
        summary["error"] = traceback.format_exc()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str))
        mark_done("failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
