"""setup_env_core: environments, generator, evidence and certifier skeleton + non-triviality / timing checks.

Usage: run_setup_env_core.py --mode {pilot,full}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

from dsswm.certify.enum_exact import certify_enum  # noqa: E402
from dsswm.certify.lin_closed import certify_lin  # noqa: E402
from dsswm.envs.mis import make_mis_env  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.lin_class import LinClass, ridge_estimate  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_M_GRID, DEFAULT_NL_S_GRID, LIN_DEFAULTS, NL_DEFAULTS,  # noqa: E402
                                     generator_hash, make_lin_instance, make_nl_instance, nl_class_max_jtable)

TASK = "setup_env_core"


def progress(res_dir, step, total, phase, metric=None):
    (res_dir / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def lin_S_bound(prior):
    hi = [max(abs(a), abs(b)) for a, b in (prior["alpha"],) * 3 + (prior["beta"],) * 3 + (prior["gamma"],) * 3
          + (prior["psi"], prior["psi2"])]
    return float(np.sqrt(np.sum(np.square(hi))))


def run_lin(seeds, eps_list, out_rows, samples, static=False, H_override=None):
    cfg = dict(LIN_DEFAULTS)
    S = lin_S_bound(cfg["prior"])
    res = []
    for s in seeds:
        kw = {"static": static}
        if H_override:
            kw["H_choices"] = H_override
        inst = make_lin_instance(s, **kw)
        env = inst.env
        lc = LinClass(3, 3, env.aspace.nb, nmax=3, static=static)
        ell = EllipsoidSet(lc, sigma=cfg["sigma"], delta=0.05, S=S)
        led = EvidenceLedger([ell])
        for o in inst.init_obs:
            led.record(o, "initial")
        X = np.vstack([lc.obs_rows(o)[0] for o in inst.init_obs])
        y = np.concatenate([lc.obs_rows(o)[1] for o in inst.init_obs])
        th_hat, _ = ridge_estimate(X, y, 1.0)
        th = env.true_theta_vector()
        for q in inst.problems:
            Z = np.array([lc.z(p, q.loads0, q.H, q.utility, env.aspace) for p in q.policies])
            Jt, Jh = Z @ th, Z @ th_hat
            k = int(np.argmax(Jh))
            reg = float(Jt.max() - Jt[k])
            cert = certify_lin(Z, ell, eps_list[0])
            row = {"env": "E1-Static" if static else ("E1-Lin-H1" if H_override == (1,) else "E1-Lin"),
                   "instance": s, "problem": q.pid, "H": q.H, "n_policies": len(q.policies),
                   "point_regret": reg, "argmax_wrong": bool(reg > 1e-12),
                   **{f"misselect_eps{e}": bool(reg > e) for e in eps_list},
                   "J_true_spread": float(Jt.max() - Jt.min()), "r_bar_initial": cert["r_bar"],
                   "status_initial": cert["status"].value, "cert_pi_true_regret": float(Jt.max() - Jt[cert["pi_hat"]]),
                   "theta_star_in_set": bool(ell.contains(th)), "n0_obs": ell.n_obs}
            res.append(row)
            out_rows.append(row)
            if len(samples) < 4 and not static and H_override is None:
                samples.append({"env": "E1-Lin", "problem": q.public_dict(), "J_true": Jt.tolist(), "J_point": Jh.tolist(),
                                "theta_true": th.tolist(), "theta_hat": th_hat.tolist(), "sqrt_beta": ell.sqrt_beta(),
                                "ub_per_challenger": cert["ub"].tolist(), "pi_hat": cert["pi_hat"],
                                "binding": cert["binding"], "status": cert["status"].value})
    return res


def run_nl(seeds, eps_list, out_rows, samples, ncl, params, prop_cache, cache_dir, timings_all, env_name="E1-NL-S",
           grid=None, L=2, R=2, gen_kw=None):
    res = []
    LT = None
    for s in seeds:
        inst = make_nl_instance(s, grid=grid, nl_class=ncl, **(gen_kw or {}))
        env = inst.env
        key = (L, R)
        if key not in prop_cache:
            prop_cache[key] = NLPropagator(L, R, 2, env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"])
        prop = prop_cache[key]
        LT, _ = prop.tables(params)
        lr = SeqLRSet(prop, LT, 0.05)
        led = EvidenceLedger([lr])
        for o in inst.init_obs:
            led.record(o, "initial")
        kh = lr.mle()
        ti = inst.truth["theta_index"]
        mask = lr.mask().cpu().numpy()
        for q in inst.problems:
            tm = {}
            t0 = time.perf_counter()
            raw_u = type(q.utility)(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0)
            raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, raw_u, timings=tm)
            jt_s = time.perf_counter() - t0
            np.save(cache_dir / f"{env_name}_{q.pid}_raw.npy", raw.astype(np.float64))
            q.utility.c_q = 1.0 / float(raw.max())
            J = raw * q.utility.c_q
            Jt, Jh = J[ti], J[kh]
            k = int(np.argmax(Jh))
            reg = float(Jt.max() - Jt[k])
            cert = certify_enum(J, mask, kh, eps_list[0])
            row = {"env": env_name, "instance": s, "problem": q.pid, "H": q.H, "n_policies": len(q.policies),
                   "Theta_size": int(ncl.B), "jtable_seconds": jt_s, "jtable_plan_s": tm.get("plan_s"),
                   "jtable_gpu_s": tm.get("gpu_s"), "max_reachable_states": tm.get("max_states"),
                   "point_regret": reg, "argmax_wrong": bool(reg > 1e-12),
                   **{f"misselect_eps{e}": bool(reg > e) for e in eps_list},
                   "J_true_spread": float(Jt.max() - Jt.min()), "r_bar_initial": cert["r_bar"],
                   "status_initial": cert["status"].value, "set_size_initial": int(mask.sum()),
                   "theta_star_in_set": bool(mask[ti]), "c_q": q.utility.c_q}
            res.append(row)
            out_rows.append(row)
            timings_all.append(jt_s)
            if len(samples) < 8 and env_name == "E1-NL-S" and len([x for x in samples if x["env"] == env_name]) < 4:
                samples.append({"env": env_name, "problem": q.public_dict(), "J_true": Jt.tolist(), "J_mle": Jh.tolist(),
                                "theta_star_index": ti, "mle_index": kh, "truth": inst.truth,
                                "mle_params": {k2: np.asarray(v).tolist() for k2, v in ncl.params_at(kh).items()},
                                "set_size": int(mask.sum()), "status": cert["status"].value, "r_bar": cert["r_bar"],
                                "binding_policy": cert["binding"], "top_models": cert["top_models"],
                                "top_regrets": cert["top_regrets"]})
    return res


def mc_checks(n_mc=100_000):
    """Exact-vs-MC agreement on every environment variant (max |err| and max z)."""
    rng = np.random.default_rng(42)
    out = {}
    # Lin
    inst = make_lin_instance(0)
    lc = LinClass(3, 3, inst.env.aspace.nb)
    th = inst.env.true_theta_vector()
    errs, zs = [], []
    for q in inst.problems[:3]:
        for p in q.policies:
            J = lc.z(p, q.loads0, q.H, q.utility, inst.env.aspace) @ th
            u = q.utility.value_from_components(inst.env.simulate_batch(p, q.loads0, q.H, n_mc, rng))
            errs.append(abs(u.mean() - J)); zs.append(abs(u.mean() - J) / (u.std(ddof=1) / np.sqrt(n_mc)))
    out["E1-Lin"] = {"max_abs_err": float(max(errs)), "max_z": float(max(zs)), "n_checks": len(errs)}

    def nl_check(env, problems, L, R, name, k=2):
        prop = NLPropagator(L, R, 2, env.aspace, 1.0, 0.3)
        errs, zs = [], []
        for q in problems[:k]:
            Jt = prop.j_table(params_to_torch(env.true_params()), q.policies, q.loads0, q.engaged0, q.H, q.utility)[0]
            for kk, p in enumerate(q.policies):
                ys, es = env.simulate_batch(p, q.loads0, q.engaged0, q.H, n_mc, rng)
                u = q.utility.value_from_components(ys, es)
                errs.append(abs(u.mean() - Jt[kk])); zs.append(abs(u.mean() - Jt[kk]) / (u.std(ddof=1) / np.sqrt(n_mc)))
        out[name] = {"max_abs_err": float(max(errs)), "max_z": float(max(zs)), "n_checks": len(errs)}

    inst = make_nl_instance(0)
    nl_check(inst.env, inst.problems, 2, 2, "E1-NL-S")
    base = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in inst.truth.items() if k != "theta_index"}
    base["psi"] = [base["psi"]]
    for kind in ("m1", "m2", "m3"):
        env = make_mis_env(kind, base, 0.8, np.random.default_rng(1), L=2, R=2, nmax=2, seed=3)
        nl_check(env, inst.problems, 2, 2, f"E1-Mis-{kind}", k=1)
    inst = make_nl_instance(1, grid=DEFAULT_NL_M_GRID, H_choices=(6,), n_pol=(3, 4))
    nl_check(inst.env, inst.problems, 3, 3, "E1-NL-M", k=1)
    return out


def gpu_profile(prop, params, problem, res_dir):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    prop.j_table(params, problem.policies, problem.loads0, problem.engaged0, problem.H, problem.utility)
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20
    props = torch.cuda.get_device_properties(0)
    tot = props.total_memory / 2**20
    prof = {"gpu_name": props.name, "vram_total_mb": round(tot), "max_batch_size": int(params["alpha"].shape[0]),
            "batch_unit": "theta per J-table chunk (whole class fits in one chunk)",
            "vram_used_mb": round(peak, 1), "utilization_pct": round(100 * peak / tot, 2), "jtable_seconds": dt,
            "note": "Workload is tiny (|Theta| x reachable states x 2^P rows); VRAM is not the bottleneck, python "
                    "plan construction is. No benefit from larger batches."}
    (res_dir / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof, indent=1))
    return prof


def rate(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    args = ap.parse_args()
    t_start = time.time()
    start_iso = datetime.now().isoformat()
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    cache_dir = WS / "exp" / "cache" / "jtables"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    eps_list = [0.05, 0.02] if args.mode == "pilot" else [0.02, 0.05]
    n_inst = 10 if args.mode == "pilot" else 100
    seeds = list(range(0, n_inst))
    TOTAL = 7

    # 1. unit tests
    progress(res_root, 1, TOTAL, "unit_tests")
    r = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "dsswm" / "tests"), "-q", "-rA", "--tb=short",
                        f"--junitxml={out_dir / 'pytest_junit.xml'}"], capture_output=True, text=True, cwd=str(HERE))
    (out_dir / "pytest_output.txt").write_text(r.stdout + "\n" + r.stderr)
    import re
    m_pass = re.search(r"(\d+) passed", r.stdout)
    m_fail = re.search(r"(\d+) failed", r.stdout)
    tests = {"passed": int(m_pass.group(1)) if m_pass else 0, "failed": int(m_fail.group(1)) if m_fail else 0,
             "returncode": r.returncode}
    named = {
        "evidence_boundary": "test_evidence_boundary.py", "no_truth_import": "test_no_truth_import.py",
        "exact_vs_mc": "test_exact_vs_mc.py", "lin_enum_vs_closed": "test_lin_enum_vs_closed.py"}
    tests["required"] = {k: (f"FAILED dsswm/tests/{v}" not in r.stdout and v in r.stdout) for k, v in named.items()}

    rows, samples, timings = [], [], []
    # 2. E1-Lin + static controls
    progress(res_root, 2, TOTAL, "lin_nontriviality")
    lin = run_lin(seeds, eps_list, rows, samples)
    lin_static = run_lin(seeds, eps_list, rows, samples, static=True)
    lin_h1 = run_lin(seeds, eps_list, rows, samples, H_override=(1,))

    # 3. E1-NL-S
    progress(res_root, 3, TOTAL, "nl_s_jtables")
    ncl = NLClass(DEFAULT_NL_S_GRID)
    params = ncl.torch_params()
    prop_cache = {}
    nl = run_nl(seeds, eps_list, rows, samples, ncl, params, prop_cache, cache_dir, timings)
    nl_t = list(timings)
    nl_k0 = run_nl(seeds, eps_list, rows, samples, ncl, params, prop_cache, cache_dir, timings,
                   env_name="E1-NL-S-kappa0", gen_kw={"kappa_mode": "zero"})
    nl_kh = run_nl(seeds, eps_list, rows, samples, ncl, params, prop_cache, cache_dir, timings,
                   env_name="E1-NL-S-kappaHigh", gen_kw={"kappa_mode": "high"})

    # 4. E1-NL-M feasibility (3x3, reduced grid)
    progress(res_root, 4, TOTAL, "nl_m_feasibility")
    nclm = NLClass(DEFAULT_NL_M_GRID)
    paramsm = nclm.torch_params()
    tm_list = []
    nlm = run_nl(seeds[:3], eps_list, rows, samples, nclm, paramsm, prop_cache, cache_dir, tm_list, env_name="E1-NL-M",
                 grid=DEFAULT_NL_M_GRID, L=3, R=3, gen_kw={"H_choices": (6,), "K": 3})

    # 5. MC vs exact
    progress(res_root, 5, TOTAL, "mc_vs_exact")
    mc = mc_checks()

    # 6. GPU profile
    progress(res_root, 6, TOTAL, "gpu_profile")
    inst = make_nl_instance(0, nl_class=ncl)
    prof = gpu_profile(prop_cache[(2, 2)], params, inst.problems[0], res_root)

    # 7. aggregate
    progress(res_root, 7, TOTAL, "aggregate")
    with open(out_dir / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for k, smp in enumerate(samples):
        (out_dir / "samples" / f"sample_{k:02d}_{smp['env']}.json").write_text(json.dumps(smp, indent=1, default=str))

    e0 = eps_list[0]
    pilot_eps = 0.05
    mis_lin = rate(lin, f"misselect_eps{pilot_eps}")
    mis_nl = rate(nl, f"misselect_eps{pilot_eps}")
    jt_mean = float(np.mean(nl_t))
    jt_max = float(np.max(nl_t))

    def env_row(name, rs, theta_size, P_desc, H_desc):
        return {"Env": name, "Participants": P_desc, "H": H_desc, "|Theta|": theta_size,
                "J-table s/problem": (round(float(np.mean([r["jtable_seconds"] for r in rs])), 3)
                                      if rs and "jtable_seconds" in rs[0] else "closed form"),
                "Mis-selection rate (eps=0.05)": round(rate(rs, "misselect_eps0.05"), 3),
                "Mis-selection rate (eps=0.02)": round(rate(rs, "misselect_eps0.02"), 3),
                "argmax-wrong rate": round(rate(rs, "argmax_wrong"), 3),
                "median J spread": round(float(np.median([r["J_true_spread"] for r in rs])), 4),
                "initially CERTIFIED (eps=%.2f)" % e0: round(float(np.mean([r["status_initial"] == "CERTIFIED" for r in rs])), 3),
                "theta* in Theta_n0": round(float(np.mean([r["theta_star_in_set"] for r in rs])), 3),
                "n_problems": len(rs)}

    table = [
        env_row("E1-Lin", lin, "continuous (R^11, ellipsoid)", "3x3", "{4,6,8}"),
        env_row("E1-Lin H=1 (static control)", lin_h1, "continuous (R^11)", "3x3", "1"),
        env_row("E1-Static (no load)", lin_static, "continuous (R^11)", "3x3", "{4,6,8}"),
        env_row("E1-NL-S", nl, ncl.B, "2x2", "{6,8}"),
        env_row("E1-NL-S kappa=0 truth", nl_k0, ncl.B, "2x2", "{6,8}"),
        env_row("E1-NL-S kappa-high truth", nl_kh, ncl.B, "2x2", "{6,8}"),
        env_row("E1-NL-M (reduced grid)", nlm, nclm.B, "3x3", "6"),
    ]
    # instance-level spread of the mis-selection rate (instances are the statistical unit)
    from dsswm.stats.bootstrap import proportion_ci_by_cluster
    from dsswm.stats.cp import clopper_pearson

    def ci(rs):
        by = {}
        for r_ in rs:
            by.setdefault(r_["instance"], []).append(float(r_[f"misselect_eps{pilot_eps}"]))
        est, lo, hi = proportion_ci_by_cluster({k: np.array(v) for k, v in by.items()}, B=2000, seed=0)
        k_ = int(sum(sum(v) for v in by.values())); n_ = sum(len(v) for v in by.values())
        return {"rate": est, "cluster_boot_95": [lo, hi], "clopper_pearson_95": list(clopper_pearson(k_, n_))}

    pass_tests = tests["failed"] == 0 and tests["passed"] > 0 and all(tests["required"].values())
    pass_lin = 0.10 <= mis_lin <= 0.90
    pass_nl = 0.10 <= mis_nl <= 0.90
    pass_time = jt_max <= 10.0
    mc_max_err = max(v["max_abs_err"] for v in mc.values())
    mc_max_z = max(v["max_z"] for v in mc.values())
    go = pass_tests and pass_lin and pass_nl and pass_time and mc_max_z < 3.0

    summary = {
        "task_id": TASK, "mode": args.mode, "seed": 42, "instances": seeds, "problems_per_instance": 10,
        "generator_hash": generator_hash(),
        "metrics": {
            "unit_tests_passed": tests["passed"], "unit_tests_failed": tests["failed"],
            "required_unit_tests": tests["required"],
            "jtable_seconds_per_problem": {"E1-NL-S_mean": jt_mean, "E1-NL-S_max": jt_max,
                                           "E1-NL-M_mean": float(np.mean(tm_list)), "E1-NL-M_max": float(np.max(tm_list))},
            "point_swm_misselection_rate": {"E1-Lin": {f"eps{e}": rate(lin, f"misselect_eps{e}") for e in eps_list},
                                            "E1-NL-S": {f"eps{e}": rate(nl, f"misselect_eps{e}") for e in eps_list},
                                            "E1-Lin_ci": ci(lin), "E1-NL-S_ci": ci(nl)},
            "mc_vs_exact_max_abs_err": mc_max_err, "mc_vs_exact_max_z": mc_max_z, "mc_vs_exact": mc,
        },
        "pass_criteria": {"all_unit_tests_pass": pass_tests, "lin_misselection_in_[0.1,0.9]": pass_lin,
                          "nl_s_misselection_in_[0.1,0.9]": pass_nl, "jtable_le_10s": pass_time,
                          "mc_vs_exact_within_3se": mc_max_z < 3.0},
        "go_no_go": "GO" if go else "NO_GO",
        "env_table": table,
        "calibration": {
            "E1-Lin": {"sigma": LIN_DEFAULTS["sigma"], "prior": LIN_DEFAULTS["prior"], "n0": LIN_DEFAULTS["n0"],
                       "ellipsoid_S": lin_S_bound(LIN_DEFAULTS["prior"]), "beta": "theoretical (Abbasi-Yadkori 2011), lam=1",
                       "utility_scale": "c_q = 1/(2.4 * min(L,R) * sum_t w_t)"},
            "E1-NL-S": {"grid": DEFAULT_NL_S_GRID.to_dict(), "Theta_size": ncl.B, "c": NL_DEFAULTS["c"],
                        "rho_ret": NL_DEFAULTS["rho_ret"], "n0": NL_DEFAULTS["n0"], "nmax": 2,
                        "utility_scale": "class-max: c_q = 1/max_{theta in Theta, pi in Pi_q} J_raw (public, data-free)",
                        "lr_set": "predictable plug-in MLE numerator (first step: uniform mixture), delta=0.05"},
            "E1-NL-M": {"grid": DEFAULT_NL_M_GRID.to_dict(), "Theta_size": nclm.B},
        },
        "gpu_profile": prof,
        "wall_clock_s": time.time() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    # Markdown env table
    cols = list(table[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r_[c]) for c in cols) + " |" for r_ in table]
    (out_dir / "env_table.md").write_text("\n".join(md) + "\n")

    end_iso = datetime.now().isoformat()
    print(json.dumps({k: summary[k] for k in ("metrics", "pass_criteria", "go_no_go")}, indent=1, default=str))
    print("\n".join(md))
    return summary, start_iso, end_iso


if __name__ == "__main__":
    main()
