"""r2_setup_offgrid: round-1 off-grid infrastructure checks (methodology sections 1, 2.2, 2.3, 4.3).

pilot: unit tests; dev off-grid instances 600-699 (R0 / R1 / twin / dose 0.5, 3 streams); J-table timing per
       resolution f in {0.5, 1, 2, 4} on 100 dev problems (cached to exp/cache/jtables_f{f}); eta_loc per f
       (+ empirical lower bound of any valid eta_loc, + uniform-refinement diagnostic); B12 / B3-UI smoke runs.
full : eval manifests 10000-10127 (truth hashes only, no learner runs) + J-tables f in {1, 2} on all eval problems.

Usage: run_r2_setup_offgrid.py --mode {pilot,full}
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
TASK = "r2_setup_offgrid"
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache"
EPS, DELTA = 0.02, 0.05
DEV = "cuda" if torch.cuda.is_available() else "cpu"

from dsswm.certify.eta_loc import build_plans, eta_loc, j_values  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import (LADDER, _box_axes, cell_index_of, class_vectors, ladder_grid,  # noqa: E402
                                      sample_in_cells)
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.streams.generator import NL_DEFAULTS, generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance, manifest_row  # noqa: E402


def progress(step, total, phase, metric=None):
    (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES_ROOT / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES_ROOT / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def fkey(f):
    return f"{f:g}"


# ------------------------------------------------------------------------------------------------ phases
def run_tests(out_dir, log):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(HERE / "dsswm" / "tests")],
                       capture_output=True, text=True, cwd=str(HERE))
    (out_dir / "pytest.log").write_text(r.stdout + "\n" + r.stderr)
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    errors = int(m.group(1)) if (m := re.search(r"(\d+) error", tail)) else 0
    failed_ids = re.findall(r"^FAILED (\S+)", r.stdout, re.M)
    req = {}
    for name in ("test_eta_loc.py", "test_offgrid_crn.py", "test_no_truth_import.py", "test_r2_baselines.py"):
        req[name] = (HERE / "dsswm" / "tests" / name).exists() and not any(name in f for f in failed_ids)
    log(f"pytest: {passed} passed, {failed} failed, {errors} errors in {time.time() - t0:.0f}s")
    return {"passed": passed, "failed": failed, "errors": errors, "failed_ids": failed_ids, "required": req,
            "seconds": time.time() - t0}


def build_instances(seeds, out_dir, log, samples_dir):
    rows, errs = [], []
    variants = {"R0": dict(level="R0"), "R1": dict(level="R1"), "twin": dict(level="R0", twin=True),
                "dose0.5": dict(level="R0", dose=0.5)}
    t0 = time.time()
    n_built = 0
    for s in seeds:
        row = manifest_row(s)
        for name, kw in variants.items():
            for st in range(len(STREAMS)):
                try:
                    inst = make_offgrid_instance(s, stream=st, **kw)
                    assert len(inst.problems) == 15 and inst.env.n_steps == NL_DEFAULTS["n0"]
                    n_built += 1
                    if st == 0 and name in ("R0", "R1"):
                        row[f"{name}_init_y_rate"] = float(np.mean([y for o in inst.init_obs for (_, _, _, y, a) in
                                                                    o.outcomes if a] or [0]))
                except Exception as e:  # noqa: BLE001
                    errs.append({"seed": s, "variant": name, "stream": st, "error": repr(e)})
        rows.append(row)
    for s in seeds[:6]:
        r1, r0 = make_offgrid_instance(s, "R1"), make_offgrid_instance(s, "R0")
        tw = make_offgrid_instance(s, "R0", twin=True)
        (samples_dir / f"instance_{s}.json").write_text(json.dumps({
            "seed": s, "R1_truth": r1.truth, "R0_truth": r0.truth, "twin_truth": tw.truth,
            "first_problems": [q.public_dict() for q in r1.problems[:2]],
            "init_obs_first5_R0": [list(map(lambda x: x if not isinstance(x, tuple) else list(x),
                                            (o.t, o.loads, o.engaged, o.action, o.outcomes, o.next_engaged)))
                                   for o in r0.init_obs[:5]],
            "init_actions_equal_R0_R1": [o.action for o in r0.init_obs] == [o.action for o in r1.init_obs]},
            indent=1, default=str))
    log(f"built {n_built} (instance, variant, stream) objects for {len(seeds)} seeds in {time.time() - t0:.1f}s, "
        f"{len(errs)} errors")
    return rows, errs, n_built


def jtables(prop_by_aspace, probs, fs, log, cache_f32=(4,), timing_rows=None):
    """Raw J tables per f for every (seed, problem); cached. Returns {f: {pid: array}} lazily via paths."""
    out = {}
    for f in fs:
        ncl = NLClass(ladder_grid(f), device=DEV)
        params = ncl.torch_params()
        d = CACHE / f"jtables_f{fkey(f)}"
        d.mkdir(parents=True, exist_ok=True)
        times, mems = [], []
        for (seed, q, prop) in probs:
            p = d / f"{q.pid}_raw.npy"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H,
                               type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})())
            dt = time.perf_counter() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0.0
            np.save(p, raw.astype(np.float32) if f in cache_f32 else raw)
            times.append(dt)
            mems.append(mem)
            out.setdefault(f, {})[q.pid] = p
            if timing_rows is not None:
                timing_rows.append({"kind": "jtable", "f": f, "problem": q.pid, "Theta": ncl.B, "K": len(q.policies),
                                    "H": q.H, "sec": dt, "gpu_mb": mem, "raw_max": float(raw.max())})
        log(f"J-table f={fkey(f)} |Theta|={ncl.B}: mean {np.mean(times):.3f}s max {np.max(times):.3f}s "
            f"peak GPU {np.max(mems):.0f} MB over {len(probs)} problems")
        del params, ncl
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def refined_all_grid_error(prop, q, f_all, n, rng):
    """Diagnostic: empirical max_pi |J_theta - J_g| for g on a hypothetical grid refining ALL 12 coordinates by f_all
    (not enumerated: random g, random theta in its cell). A lower bound on any valid eta_loc at that resolution."""
    bx = np.array(_box_axes(ladder_grid(1)), float)
    base_n = np.array([3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2])          # points per axis in G_1 (vector order)
    npts = (base_n - 1) * f_all + 1
    k = np.floor(rng.random((n, 12)) * npts).astype(int)
    step = (bx[:, 1] - bx[:, 0]) / (npts - 1)
    g = bx[:, 0] + k * step
    lo = np.maximum(g - step / 2, bx[:, 0])
    hi = np.minimum(g + step / 2, bx[:, 1])
    th = lo + (hi - lo) * rng.random((n, 12))
    plans = build_plans(prop, q)
    Jg = j_values(prop, plans, g, q.utility.w, q.utility.w_ret)
    Jt = j_values(prop, plans, th, q.utility.w, q.utility.w_ret)
    return np.abs(Jt - Jg).max(1)


def eta_phase(probs, fs, n_eta_by_f, cq, log, rows, n_emp=200, seed=42):
    rng = np.random.default_rng(seed)
    stats = {}
    for f in fs:
        ncl = NLClass(ladder_grid(f), device=DEV)
        d = CACHE / f"jtables_f{fkey(f)}"
        e_med, e_min, viol, tight, emp, dec, tsec, lj = [], [], 0, [], [], [], [], []
        for (seed_, q, prop) in probs[: n_eta_by_f[f]]:
            t0 = time.perf_counter()
            eta, J, hb, info = eta_loc(prop, ncl, q, return_parts=True)
            tsec.append(time.perf_counter() - t0)
            np.save(d / f"{q.pid}_etaloc_raw.npy", eta)
            c = cq[q.pid]
            en = eta * c
            idx = rng.integers(ncl.B, size=n_emp)
            th = sample_in_cells(ncl, idx, rng)
            Jt = j_values(prop, build_plans(prop, q), th, q.utility.w, q.utility.w_ret)
            err = np.abs(Jt - J[idx]).max(1) * c
            viol += int((err > en[idx] + 1e-12).sum())
            tight.append(float(np.median(err / en[idx])))
            emp.append(float(err.max()))
            e_med.append(float(np.median(en)))
            e_min.append(float(en.min()))
            dec.append(np.asarray(info["parts"]["grad_term_by_coord"]) * c)
            lj.append(info["parts"]["L_J"] * c)
            rows.append({"kind": "eta_loc", "f": f, "problem": q.pid, "Theta": ncl.B, "sec": tsec[-1],
                         "eta_norm_median": e_med[-1], "eta_norm_min": e_min[-1], "eta_norm_max": float(en.max()),
                         "eta_over_eps_median": e_med[-1] / EPS, "emp_err_max": emp[-1],
                         "emp_err_median": float(np.median(err)), "tightness_median": tight[-1],
                         "violations": int((err > en[idx] + 1e-12).sum()), "draws": n_emp, "L_J": lj[-1],
                         "hess_term_max_norm": info["parts"]["hess_term_max"] * c})
        names = ["alpha0", "alpha1", "gamma0", "gamma1", "tauL0", "tauL1", "beta0", "beta1", "tauR0", "tauR1", "psi",
                 "lam"]
        stats[fkey(f)] = {"Theta": ncl.B, "n_problems": len(e_med), "eta_over_eps_median": float(np.median(e_med)) / EPS,
                          "eta_over_eps_min_cell_median": float(np.median(e_min)) / EPS,
                          "share_problems_median_eta_le_eps_over_4": float(np.mean(np.array(e_med) <= EPS / 4)),
                          "violations": viol, "draws": n_emp * len(e_med),
                          "tightness_median(err/eta)": float(np.median(tight)),
                          "empirical_err_max_median_over_problems": float(np.median(emp)),
                          "empirical_err_over_eps_median": float(np.median(emp)) / EPS,
                          "grad_term_by_coord_mean": dict(zip(names, np.mean(dec, 0).round(5).tolist())),
                          "L_J_median": float(np.median(lj)), "sec_per_problem_mean": float(np.mean(tsec))}
        log(f"eta_loc f={fkey(f)}: median eta/eps={stats[fkey(f)]['eta_over_eps_median']:.1f}, empirical sup err/eps "
            f"{stats[fkey(f)]['empirical_err_over_eps_median']:.2f}, violations {viol}/{n_emp * len(e_med)}, "
            f"{np.mean(tsec):.2f}s/problem")
    return stats


def smoke_baselines(log, samples_dir):
    """B12 (finite class, whole-trial TaS) and B3-UI run end-to-end on dev instance 600 (R0, f=1)."""
    from dsswm.baselines.b3_ui import certify_b3_ui
    from dsswm.baselines.b12_tas import B12TaS
    from dsswm.evidence.lr_set import SeqLRSet
    from dsswm.models.grid_ladder import class_vectors as cv
    out = []
    inst = make_offgrid_instance(600, "R0")
    env = inst.env
    ncl = NLClass(ladder_grid(1), device=DEV)
    prop = NLPropagator(2, 2, 2, env.aspace, 1.0, 0.3, device=DEV)
    tp = env.true_params()
    star = int(cell_index_of(np.concatenate([tp["alpha"], tp["gamma"], tp["tauL"], tp["beta"], tp["tauR"],
                                             [tp["psi_left"][0, 1]], [tp["lam"]]])[None], ncl)[0])
    LT, _ = prop.tables(ncl.torch_params())
    for q in inst.problems[:3]:
        raw = np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy")
        cq = 1.0 / raw.max()
        Jn = raw * cq
        u_max = cq * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)
        sim_rng = np.random.default_rng([600, 77, int(q.pid.split("q")[-1])])

        def sampler(k, n, q=q, cq=cq):       # harness side: real platform trials from s0
            ys, es = env.simulate_batch(q.policies[k], q.loads0, q.engaged0, q.H, n, sim_rng)
            return cq * (ys @ q.utility.w + q.utility.w_ret * es)
        t0 = time.perf_counter()
        b12 = B12TaS(Jn, u_max=u_max, delta=DELTA, eps=EPS)
        r = b12.run(sampler, q.H, max_steps=int(2.31e6))
        r.pop("logM")
        tr = float(Jn[star].max() - Jn[star, r["pi"]]) if r["pi"] is not None else None
        out.append({"method": "B12", "problem": q.pid, **r, "true_regret": tr, "theta_star_alive": bool(b12.mask()[star]),
                    "wall_s": time.perf_counter() - t0})
    # B3-UI on the real ledger (init + 300 random rounds), via the authenticated observations held by the ledger
    lr2 = SeqLRSet(prop, LT, DELTA)
    inst2 = make_offgrid_instance(600, "R0")
    obs = list(inst2.init_obs)
    h2 = inst2.env.handle()
    for o in obs:
        lr2.update(o)
    rng = np.random.default_rng(42)
    for _ in range(300):
        o = h2.step(int(rng.integers(inst2.env.aspace.n)))
        lr2.update(o)
        obs.append(o)
    for q in inst2.problems[:3]:
        raw = np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy")
        q.utility.c_q = 1.0 / raw.max()
        t0 = time.perf_counter()
        c = certify_b3_ui(prop, build_plans(prop, q), obs, lr2.log_num, cv(ncl)[lr2.mle()], ncl.grid, q.utility,
                          DELTA, EPS)
        Jn = raw * q.utility.c_q
        out.append({"method": "B3-UI", "problem": q.pid, "status": c["status"].value, "pi": c["pi"],
                    "r_bar": c["r_bar"], "beta_ui": c["beta"], "n_obs": len(obs),
                    "true_regret": float(Jn[star].max() - Jn[star, c["pi"]]) if c["pi"] is not None else None,
                    "v_hat_dist_to_truth_cell_point": float(np.abs(c["v_hat"] - cv(ncl)[star]).max()),
                    "wall_s": time.perf_counter() - t0})
    (samples_dir / "baseline_smoke.json").write_text(json.dumps(out, indent=1, default=str))
    for r in out:
        log(f"smoke {r['method']} {r['problem']}: status={r['status']} r_bar={r['r_bar']:.4f} "
            f"true_regret={r['true_regret']} steps={r.get('steps', '-')} wall={r['wall_s']:.1f}s")
    return out


def table_md(timing, stats):
    lines = ["| f | |Theta| | J-table sec/problem (mean / max) | GPU MB (peak) | eta_loc/eps median | empirical sup err/eps | eta_loc sec/problem |",
             "|---|---|---|---|---|---|---|"]
    for f in LADDER:
        t = [r for r in timing if r["kind"] == "jtable" and r["f"] == f]
        if not t:
            continue
        s = stats.get(fkey(f), {})
        lines.append(f"| {fkey(f)} | {t[0]['Theta']} | {np.mean([r['sec'] for r in t]):.3f} / "
                     f"{np.max([r['sec'] for r in t]):.3f} | {np.max([r['gpu_mb'] for r in t]):.0f} | "
                     f"{s.get('eta_over_eps_median', float('nan')):.1f} | "
                     f"{s.get('empirical_err_over_eps_median', float('nan')):.2f} | "
                     f"{s.get('sec_per_problem_mean', float('nan')):.2f} |")
    return "\n".join(lines) + "\n\n(timings: concurrent run with other tasks on the shared RTX 4090 / 20-core CPU)\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    samples = out_dir / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    T0 = time.time()
    log(f"start mode={args.mode} device={DEV} generator_hash={generator_hash()}")
    rows = []
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA,
               "generator_hash": generator_hash(), "concurrent_run": True}
    try:
        if args.mode == "pilot":
            progress(1, 6, "unit tests")
            tests = run_tests(out_dir, log)
            summary["unit_tests"] = tests
            progress(2, 6, "dev instances 600-699")
            seeds = list(range(600, 700))
            man, errs, n_built = build_instances(seeds, out_dir, log, samples)
            with open(out_dir / "dev_manifest.jsonl", "w") as fh:
                for r in man:
                    fh.write(json.dumps(r) + "\n")
            summary["offgrid_instances_built"] = n_built
            summary["offgrid_dev_seeds"] = [600, 699]
            summary["instance_build_errors"] = errs
            r0_eq = sum(r["R0"] == r["R1"] for r in man)
            summary["R0_equals_R1_count"] = int(r0_eq)
            progress(3, 6, "J-table timing per f")
            probs = []
            prop = None
            for s in range(600, 610):
                inst = make_offgrid_instance(s, "R1")
                if prop is None:
                    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3, device=DEV)
                byk = sorted(inst.problems, key=lambda q: q.meta["gen_index"])
                probs += [(s, q, prop) for q in byk[:10]]
            jt = jtables(None, probs, LADDER, log, timing_rows=rows)
            cq = {q.pid: 1.0 / float(np.load(jt[1][q.pid]).max()) for (_, q, _) in probs}
            over = max(float(np.load(jt[f][q.pid]).max()) * cq[q.pid] for f in LADDER for (_, q, _) in probs)
            summary["max_normalised_J_over_ladder"] = over
            progress(4, 6, "eta_loc per f")
            n_eta = {0.5: 100, 1: 100, 2: 100, 4: 20}
            stats = eta_phase(probs, LADDER, n_eta, cq, log, rows)
            summary["eta_loc"] = stats
            # uniform-refinement diagnostic (all 12 coordinates refined): lower bound on any valid eta_loc
            rng = np.random.default_rng(7)
            diag = {}
            for f_all in (1, 2, 4, 8, 16):
                e = []
                for (_, q, pr) in probs[:20]:
                    e.append(float(refined_all_grid_error(pr, q, f_all, 300, rng).max() * cq[q.pid]))
                diag[str(f_all)] = {"empirical_sup_err_over_eps_median": float(np.median(e)) / EPS,
                                    "share_problems_le_eps_over_4": float(np.mean(np.array(e) <= EPS / 4))}
                log(f"all-coordinate refinement x{f_all}: empirical sup err/eps median {np.median(e) / EPS:.2f}")
            summary["all_coordinate_refinement_diagnostic"] = diag
            progress(5, 6, "B12 / B3-UI smoke")
            try:
                summary["baseline_smoke"] = smoke_baselines(log, samples)
            except Exception as e:  # noqa: BLE001
                import traceback
                log("baseline smoke failed: " + traceback.format_exc())
                summary["baseline_smoke_error"] = repr(e)
            summary["b11"] = {"equivalent_to": "run_nl_acquisition_factorial.py::joint_TaskDirected (B5, crit='T')",
                              "action": "reuse round-0 tuned TaskDirected config; criterion extracted to "
                                        "dsswm/baselines/b11_taskfisher.py (unit-tested identical formula)"}
            summary["b3_ui"] = {"implemented": True, "module": "dsswm/baselines/b3_ui.py",
                                "note": "quadratic/linearised approximation of the continuous UI set (favourable)"}
            jt2 = [r["sec"] for r in rows if r["kind"] == "jtable" and r["f"] == 2]
            summary["metrics"] = {
                "unit_tests_passed": tests["passed"], "unit_tests_failed": tests["failed"] + tests["errors"],
                "eta_loc_violations": int(sum(s["violations"] for s in stats.values())),
                "eta_loc_draws": int(sum(s["draws"] for s in stats.values())),
                "jtable_sec_per_problem_by_f": {fkey(f): float(np.mean([r["sec"] for r in rows if r["kind"] == "jtable"
                                                                         and r["f"] == f])) for f in LADDER},
                "offgrid_instances_built": n_built}
            ok_tests = tests["failed"] == 0 and tests["errors"] == 0 and tests["passed"] > 0
            ok = ok_tests and max(jt2) <= 10 and len(errs) == 0 and summary["metrics"]["eta_loc_violations"] == 0
            summary["pass_criteria"] = {"all_unit_tests_pass": ok_tests, "jtable_f2_max_sec": float(max(jt2)),
                                        "jtable_f2_le_10s": bool(max(jt2) <= 10), "dev_instances_no_error": len(errs) == 0,
                                        "eta_loc_zero_violations": summary["metrics"]["eta_loc_violations"] == 0}
            summary["go_no_go"] = "GO" if ok else "NO_GO"
            (out_dir / "jtable_resolution_table.md").write_text(table_md(rows, stats))
        else:
            progress(1, 3, "eval manifests 10000-10127")
            seeds = list(range(10000, 10128))
            man = [manifest_row(s) for s in seeds]
            blob = "\n".join(json.dumps(r, sort_keys=True) for r in man) + "\n"
            (out_dir / "eval_manifest.jsonl").write_text(blob)
            summary["manifest_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
            progress(2, 3, "eval J-tables f in {1,2}")
            prop, probs = None, []
            for s in seeds:
                inst = make_offgrid_instance(s, "R1")
                if prop is None:
                    prop = NLPropagator(2, 2, 2, inst.env.aspace, 1.0, 0.3, device=DEV)
                probs += [(s, q, prop) for q in inst.problems]
            todo = [(s, q, p) for (s, q, p) in probs
                    if not all((CACHE / f"jtables_f{fkey(f)}" / f"{q.pid}_raw.npy").exists() for f in (1, 2))]
            jtables(None, todo, (1, 2), log, timing_rows=rows)
            complete = all((CACHE / f"jtables_f{fkey(f)}" / f"{q.pid}_raw.npy").exists() for f in (1, 2)
                           for (_, q, _) in probs)
            summary["jtable_cache_complete"] = bool(complete)
            summary["n_eval_problems"] = len(probs)
            summary["go_no_go"] = "GO" if complete else "NO_GO"
        with open(out_dir / "results.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, default=float) + "\n")
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"done in {time.time() - T0:.0f}s: {summary['go_no_go']}")
        progress(6, 6, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']}")
    except Exception as e:  # noqa: BLE001
        import traceback
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
