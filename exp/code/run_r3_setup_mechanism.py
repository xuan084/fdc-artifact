"""r3_setup_mechanism: design-orthogonal leverage module (Lambda_hat_perp, rho*, A_k, Rem bound) + write-ahead log.

pilot: unit tests (test_leverage + test_no_truth_import) and a read-out on 100 (instance, problem, ledger snapshot)
       triples -- dev seeds 734-737 x problems q0-q4 x ledger sizes {20, 50, 100, 200, 400} (n0 = 20 initial rounds
       + uniform random legal rounds on the real env, one chain per instance), E1-NL-S G_1 (|Theta| = 13824),
       eps = 0.02 -- plus 20 E1-Lin problems (seeds 734-737 x q0-q4, ledger n0 + 200 random rounds, eps_Lin = 0.05).
       Per triple the LEARNER process writes predictors.jsonl (flush + fsync) before the HARNESS writes
       results.jsonl (identity check, oracle Lambda_Delta, ||r||_xi, Rem-bound validity on sampled truths, eta).
       Pass: tests pass AND identity_err_max <= 1e-9 AND Lin dynamic Lambda_perp <= 1e-10; median sec/call reported.
full : rerun the unit tests and freeze the sha256 of the mechanism code.

Dev seeds only (734-737 < 10000). Concurrent run (other tasks share the 20 cores / the RTX 4090): timings are
"concurrent". Usage: run_r3_setup_mechanism.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
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
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
TASK = "r3_setup_mechanism"
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3mech"
EPS, EPS_LIN, DELTA = 0.02, 0.05, 0.05
SEEDS = (734, 735, 736, 737)
N_PROB = 5
SNAPSHOTS = (20, 50, 100, 200, 400)
LIN_EXTRA = 200
S_LIN = 2.8071337695236402      # ellipsoid_S of the round-1/2 locks (plan/history/round2/prereg_lock.json)
CODE_FILES = ["dsswm/mechanism/__init__.py", "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py",
              "dsswm/mechanism/oracle_check.py", "dsswm/stats/auc.py", "dsswm/tests/test_leverage.py",
              "run_r3_setup_mechanism.py"]

from dsswm.certify.lin_closed import certify_lin  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, regret_matrix  # noqa: E402
from dsswm.evidence.ellipsoid import EllipsoidSet  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.mechanism import oracle_check as oc  # noqa: E402   (harness side of this runner)
from dsswm.mechanism.leverage import LinLeverage, NLLeverage, select_pairs, theta_hat_index  # noqa: E402
from dsswm.mechanism.predictor_log import PredictorLog, ResultLog, join  # noqa: E402
from dsswm.models.lin_class import LinClass  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.streams.generator import (DEFAULT_NL_S_GRID, LIN_DEFAULTS, NL_DEFAULTS, generator_hash,  # noqa: E402
                                     make_lin_instance, make_nl_instance, nl_class_max_jtable)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


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
                                                       "final_progress": fp,
                                                       "timestamp": datetime.now().isoformat()}))


def code_hashes():
    out = {}
    for f in CODE_FILES:
        out[f] = hashlib.sha256((HERE / f).read_bytes()).hexdigest()
    h = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out, h


def run_tests(out_dir: Path):
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_leverage.py",
           "tests/test_no_truth_import.py"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=HERE / "dsswm", capture_output=True, text=True)
    (out_dir / "pytest_output.txt").write_text(r.stdout + "\n" + r.stderr)
    tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-1] if r.stdout.strip() else ""
    import re
    m_p = re.search(r"(\d+) passed", tail)
    m_f = re.search(r"(\d+) failed", tail)
    m_s = re.search(r"(\d+) skipped", tail)
    return {"returncode": r.returncode, "passed": int(m_p.group(1)) if m_p else 0,
            "failed": int(m_f.group(1)) if m_f else 0, "skipped": int(m_s.group(1)) if m_s else 0,
            "summary_line": tail, "wall_s": time.time() - t0, "ok": r.returncode == 0}


# ---------------------------------------------------------------------------------------------- NL J tables (GPU)
def precompute_jtables(dev):
    CACHE.mkdir(parents=True, exist_ok=True)
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    P = ncl.torch_params()
    t0 = time.time()
    n_new = 0
    for s in SEEDS:
        inst = make_nl_instance(s, nl_class=ncl)
        prop = NLPropagator(2, 2, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"],
                            device=dev)
        for q in inst.problems[:N_PROB]:
            path = CACHE / f"E1-NL-S_{q.pid}_g{generator_hash()}_raw.npy"
            if not path.exists():
                n_new += 1
            nl_class_max_jtable(q, prop, P, cache_path=str(path))
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 2 ** 20
    else:
        mem = 0.0
    return {"wall_s": time.time() - t0, "n_new": n_new, "peak_mem_mb": mem}


# ---------------------------------------------------------------------------------------------- NL worker (CPU)
def nl_worker(seed, out_dir, device="cpu", with_path_bound=True):
    torch.set_num_threads(1)
    t_start = time.time()
    ncl = NLClass(DEFAULT_NL_S_GRID, device=device)
    inst = make_nl_instance(seed, nl_class=ncl)
    prop = NLPropagator(2, 2, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"],
                        device=device)
    TH = NLLeverage(prop).theta_matrix(ncl.np_params)
    lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
    LT = prop.tables(ncl.torch_params())[0]
    lr = SeqLRSet(prop, LT, DELTA)
    J = {}
    for q in inst.problems[:N_PROB]:
        path = CACHE / f"E1-NL-S_{q.pid}_g{generator_hash()}_raw.npy"
        J[q.pid] = nl_class_max_jtable(q, prop, None, cache_path=str(path))
    # ---- harness: ground truth (evaluation only) ----
    truth = inst.env.true_params()
    k_star = int(inst.truth["theta_index"])
    plog = PredictorLog(out_dir / "predictors.jsonl")
    rlog = ResultLog(out_dir / "results.jsonl")
    handle = inst.env.handle()
    rng = np.random.default_rng([seed, 734])
    obs = list(inst.init_obs)
    for o in obs:
        lr.update(o)
    rows = []
    for n_snap in SNAPSHOTS:
        while len(obs) < n_snap:
            o = handle.step(int(rng.integers(inst.env.aspace.n)))
            lr.update(o)
            obs.append(o)
        ledger = list(obs[:n_snap])                    # learner sees data strictly before the certification time
        cum = lr.cum.cpu().numpy()
        mask = lr.mask().cpu().numpy()
        k_hat = theta_hat_index(cum, mask)
        cnt = lev.cell_counts(ledger)
        xi = cnt / cnt.sum()
        for q in inst.problems[:N_PROB]:
            Jc = J[q.pid]
            cert = certify_minimax(regret_matrix(Jc), mask, EPS)
            pi_hat = int(cert["pi"])
            pairs = select_pairs(Jc, mask, k_hat, pi_hat)
            key = {"instance": seed, "stream": 0, "method": "probe", "arm": "random_ledger",
                   "problem": f"{q.pid}@n{n_snap}"}
            # ---------------- LEARNER: predictors at certification time (no truth) ----------------
            pred = lev.predictors(q, ledger, TH[k_hat], pairs, cert["r_bar"], EPS, with_rem=True,
                                  with_path=with_path_bound)
            gaps = np.sort(Jc[k_hat])[::-1]
            prec = {**key, "n_ledger": n_snap, "set_size": int(mask.sum()), "status": cert["status"].value,
                    "pi_hat": pi_hat, "theta_hat": k_hat, "gap_hat_over_eps": float((gaps[0] - gaps[1]) / EPS)
                    if len(gaps) > 1 else None, **{k: v for k, v in pred.items()}}
            plog.write(prec)
            # ---------------- HARNESS: truth scoring after the predictor line is on disk ----------------
            res = {**key}
            ids = []
            for p in pred["pairs"]:
                chk = oc.nl_identity_check(lev, q, p["k1"], p["k2"], ncl.params_at(k_hat), truth, xi=xi, eps=EPS)
                ids.append({"label": p["label"], "identity_err": chk["identity_err"], "rem_exact": chk["rem_exact"],
                            "rem_gl": chk["rem_gl"], "first_order": chk["first_order"], "dJ_star": chk["dJ_star"],
                            "lambda_delta": chk["lambda_delta"], "r_norm_xi": chk["r_norm_xi"],
                            "plugin_consistency_err": chk["plugin_consistency_err"],
                            "rem_bar": p.get("rem_bar")})
            J_star = Jc[k_star]
            res.update({"pairs": ids, "identity_err_max": max(x["identity_err"] for x in ids),
                        "eta": float(np.max(np.abs(Jc[k_hat] - J_star))),
                        "true_regret": float(J_star.max() - J_star[pi_hat]),
                        "false_cert": bool(cert["status"].value == "CERTIFIED" and J_star.max() - J_star[pi_hat] > EPS),
                        "theta_hat_is_truth": bool(k_hat == k_star)})
            # Rem-bound validity on sampled off-grid truths inside theta_hat's cell box (only for 1 problem/snapshot)
            if q is inst.problems[0]:
                mu, _, _, a = lev.mu_box(TH[k_hat])
                k1 = pred["pairs"][0]["k1"]
                rb = lev.rem_bound(mu, a, q, k1)
                v = oc.rem_bound_validity(lev, q, k1, mu, a, rb["rem_bar"], n_samples=20, seed=seed + n_snap)
                res["rem_validity"] = v
            rlog.write(res)
            rows.append({"pred": prec, "res": res})
    return {"seed": seed, "n_rows": len(rows), "wall_s": time.time() - t_start}


# ---------------------------------------------------------------------------------------------- Lin
class _OpenLoop:
    def __init__(self, aspace, steps, H):
        self.aspace, self.steps, self.H = aspace, set(steps), H
        self.name = "T" + "".join("1" if t in self.steps else "0" for t in range(H))

    def act(self, t, loads, engaged):
        return self.aspace.index(((0, 0),)) if t in self.steps else self.aspace.index(())


def lin_worker(out_dir):
    from dsswm.streams.utilities import Utility
    rows, same_exp = [], []
    plog = PredictorLog(out_dir / "predictors_lin.jsonl")
    rlog = ResultLog(out_dir / "results_lin.jsonl")
    for s in SEEDS:
        inst = make_lin_instance(s)
        A = inst.env.aspace
        dyn = LinClass(LIN_DEFAULTS["L"], LIN_DEFAULTS["R"], A.nb, nmax=LIN_DEFAULTS["nmax"])
        sta = LinClass(LIN_DEFAULTS["L"], LIN_DEFAULTS["R"], A.nb, nmax=LIN_DEFAULTS["nmax"], static=True)
        ell = EllipsoidSet(dyn, sigma=LIN_DEFAULTS["sigma"], delta=DELTA, S=S_LIN)
        h = inst.env.handle()
        rng = np.random.default_rng([s, 7])
        obs = list(inst.init_obs) + [h.step(int(rng.integers(A.n))) for _ in range(LIN_EXTRA)]
        for o in obs:
            ell.update(o)
        Ld = LinLeverage(dyn, ref=dyn, sigma=LIN_DEFAULTS["sigma"])
        Ls = LinLeverage(sta, ref=dyn, sigma=LIN_DEFAULTS["sigma"])
        th = ell.theta_hat()
        th_star = inst.env.true_theta_vector()          # harness only
        for q in inst.problems[:N_PROB]:
            Z = np.stack([dyn.z(p, q.loads0, q.H, q.utility, A) for p in q.policies])
            cert = certify_lin(Z, ell, EPS_LIN)
            vals = Z @ th
            vals[cert["pi_hat"]] = -np.inf
            pairs = [("theta_hat_runner_up", cert["pi_hat"], int(np.argmax(vals))),
                     ("minimax_challenger", cert["pi_hat"], int(cert["binding"]) if cert["binding"] != cert["pi_hat"]
                      else int(np.argmax(vals)))]
            key = {"instance": s, "stream": 0, "method": "JPC-Lin", "arm": "random_ledger", "problem": q.pid}
            t0 = time.perf_counter()
            pd = Ld.predictors(q, obs, th, pairs, cert["r_bar"], EPS_LIN, A)
            ps = Ls.predictors(q, obs, th, pairs, cert["r_bar"], EPS_LIN, A)
            dt = time.perf_counter() - t0
            plog.write({**key, "dynamic": pd, "static_model": ps, "status": cert["status"].value})
            idc = [oc.lin_identity_check(Ld, q, k1, k2, th, th_star, A, dyn)["identity_err"] for _, k1, k2 in pairs]
            rlog.write({**key, "identity_err_max": max(idc)})
            rows.append({"lambda_dyn": pd["lambda_perp_max"], "lambda_static": ps["lambda_perp_max"],
                         "lambda_nonparam": max(p["lambda_perp_nonparam"] for p in pd["pairs"]),
                         "phi_perp_static": max(p["phi_perp"] for p in ps["pairs"]), "identity_err": max(idc),
                         "sec": dt, "A_k": max(p["A_k"] for p in pd["pairs"])})
        # Corollary C' demo: same-exposure / different-timing open-loop pairs
        H = 8
        pols = [_OpenLoop(A, range(3), H), _OpenLoop(A, (0, 1, 5), H), _OpenLoop(A, (0, 2, 4), H)]
        qq = type("Q", (), {"policies": pols, "loads0": np.zeros(6, dtype=np.int64), "H": H,
                            "utility": Utility(w=np.ones(H), w_ret=0.0, c_q=1.0), "pid": f"lin{s}_same_exposure"})()
        cnt = Ls.cell_counts(obs)
        xi = cnt / cnt.sum()
        from dsswm.mechanism.leverage import orth_leverage
        for k1, k2 in ((0, 1), (0, 2), (1, 2)):
            hv = Ls.contrast_leverage(qq, k1, k2, A)
            same_exp.append({"seed": s, "pair": [pols[k1].name, pols[k2].name],
                             "static_value_gap_coeff_max": float(np.abs(Ls.Phi.T @ hv).max()),
                             "lambda_static": orth_leverage(hv, xi, Ls.Phi, Ls.Phi_ref)["lambda_perp"],
                             "lambda_dynamic": orth_leverage(hv, xi, Ld.Phi, Ld.Phi_ref)["lambda_perp"]})
    return rows, same_exp


# ---------------------------------------------------------------------------------------------- summary
def q(x, p):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return float(np.quantile(x, p)) if len(x) else None


def summarise_nl(out_dir):
    rows = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl")
    P = [r["pred"] for r in rows]
    R = [r["res"] for r in rows]
    lam = [p["lambda_perp_max"] for p in P]
    phi = [max(x["phi_perp"] for x in p["pairs"]) for p in P]
    rem_eps = [max(x["rem_bar"] for x in p["pairs"]) / EPS for p in P]
    rem_path_eps = [max(x.get("rem_bar_path", np.nan) for x in p["pairs"]) / EPS for p in P]
    sec = [p["sec_per_call"] for p in P]
    eta = [r["eta"] for r in R]
    idm = [r["identity_err_max"] for r in R]
    need = [any(len(x["need_data"]) > 0 for x in p["pairs"]) for p in P]
    Ak = [x["A_k"] for p in P for x in p["pairs"]]
    Akp = [x["A_k_placebo_block"] for p in P for x in p["pairs"]]
    Akg = [x["A_k_placebo_global"] for p in P for x in p["pairs"]]
    rho = [x["rho_star"] for p in P for x in p["pairs"] if isinstance(x["rho_star"], (int, float))]
    lam_delta = [x["lambda_delta"] for r in R for x in r["pairs"]]
    val = [r["rem_validity"]["ratio"] for r in R if "rem_validity" in r]
    by_n = {}
    for p, r in zip(P, R):
        b = by_n.setdefault(p["n_ledger"], {"lam": [], "phi": [], "rem": [], "cert": 0, "n": 0, "need": 0})
        b["lam"].append(p["lambda_perp_max"])
        b["phi"].append(max(x["phi_perp"] for x in p["pairs"]))
        b["rem"].append(max(x["rem_bar"] for x in p["pairs"]) / EPS)
        b["cert"] += p["status"] == "CERTIFIED"
        b["n"] += 1
        b["need"] += any(len(x["need_data"]) > 0 for x in p["pairs"])
    by_n = {str(k): {"n": v["n"], "certified": v["cert"], "need_data_share": v["need"] / v["n"],
                     "lambda_perp_median": q(v["lam"], .5), "phi_perp_median": q(v["phi"], .5),
                     "rem_bar_over_eps_median": q(v["rem"], .5), "rem_bar_over_eps_p90": q(v["rem"], .9)}
            for k, v in sorted(by_n.items())}
    sub = [rm for rm, e in zip(rem_eps, eta) if e <= 2 * EPS]
    samples = sorted(rows, key=lambda r: r["pred"]["lambda_perp_max"])
    pick = [samples[i] for i in np.linspace(0, len(samples) - 1, 8).astype(int)]
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "nl_triples.json").write_text(json.dumps(pick, indent=1, default=str))
    return {
        "n_triples": len(rows), "identity_err_max": max(idm), "identity_err_median": q(idm, .5),
        "plugin_consistency_err_max": max(x["plugin_consistency_err"] for r in R for x in r["pairs"]),
        "lambda_perp": {"median": q(lam, .5), "p10": q(lam, .1), "p90": q(lam, .9), "max": q(lam, 1.0)},
        "phi_perp": {"median": q(phi, .5), "p10": q(phi, .1), "p90": q(phi, .9),
                     "T0iv_median_gt_0.3": (q(phi, .5) or 0) > 0.3},
        "rem_bar_over_eps": {"median": q(rem_eps, .5), "p90": q(rem_eps, .9), "min": q(rem_eps, 0.0),
                             "p90_on_eta_le_2eps": q(sub, .9), "n_eta_le_2eps": len(sub),
                             "T0ii_preview_pass": (q(sub, .9) is not None and q(sub, .9) <= 0.5)},
        "rem_bar_path_over_eps_median": q(rem_path_eps, .5),
        "rem_bound_validity_ratio_max": max(val) if val else None, "n_rem_validity_checks": len(val),
        "need_data_share": float(np.mean(need)),
        "A_k": {"median": q(Ak, .5), "placebo_block_median": q(Akp, .5), "placebo_global_median": q(Akg, .5)},
        "rho_star": {"median": q(rho, .5), "share_nonpositive": float(np.mean([x <= 0 for x in rho])) if rho else None},
        "lambda_delta_oracle": {"median": q(lam_delta, .5), "p90": q(lam_delta, .9)},
        "eta": {"median": q(eta, .5), "share_le_2eps": float(np.mean([e <= 2 * EPS for e in eta]))},
        "sec_per_call_cpu": {"median": q(sec, .5), "p90": q(sec, .9), "max": q(sec, 1.0),
                             "note": "learner process, 1 CPU thread, concurrent run (4 workers + other tasks)"},
        "certified_share": float(np.mean([p["status"] == "CERTIFIED" for p in P])),
        "false_cert": int(sum(r["false_cert"] for r in R)),
        "by_ledger_size": by_n,
    }


def gpu_timing(seed=734, n=10):
    """sec/call of the learner predictors on CUDA (B = 1; latency, not throughput)."""
    if not torch.cuda.is_available():
        return None
    dev = "cuda"
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    inst = make_nl_instance(seed, nl_class=ncl)
    prop = NLPropagator(2, 2, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], device=dev)
    TH = NLLeverage(prop).theta_matrix(ncl.np_params)
    lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
    obs = list(inst.init_obs)
    ts = []
    for i in range(n):
        qq = inst.problems[i % N_PROB]
        path = CACHE / f"E1-NL-S_{qq.pid}_g{generator_hash()}_raw.npy"
        Jc = nl_class_max_jtable(qq, prop, None, cache_path=str(path))
        pairs = select_pairs(Jc, np.ones(ncl.B, bool), 0, int(np.argmax(Jc[0])))
        r = lev.predictors(qq, obs, TH[0], pairs, 0.05)
        ts.append(r["sec_per_call"])
    return {"median": float(np.median(ts[1:])), "first_call": ts[0],
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out_dir = RES_ROOT / ("pilots" if args.mode == "pilot" else "full") / TASK
    out_dir.mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "started_at": datetime.now().isoformat(),
               "dev_seeds": list(SEEDS), "eps": EPS, "eps_lin": EPS_LIN, "generator_hash": generator_hash()}
    try:
        progress(0, 4, "unit_tests")
        tests = run_tests(out_dir)
        summary["unit_tests"] = tests
        summary["unit_tests_passed"] = tests["ok"]
        log(f"unit tests: {tests['summary_line']}")
        files, h = code_hashes()
        summary["code_sha256"] = {"files": files, "combined": h}
        if args.mode == "full":
            summary["pass"] = tests["ok"]
            (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            mark_done("success" if tests["ok"] else "failed", f"full: tests {tests['summary_line']}; code {h[:16]}")
            return
        for f in ("predictors.jsonl", "results.jsonl", "predictors_lin.jsonl", "results_lin.jsonl"):
            (out_dir / f).unlink(missing_ok=True)
        progress(1, 4, "jtables_gpu")
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        summary["jtables"] = precompute_jtables(dev)
        log(f"J tables: {summary['jtables']}")
        progress(2, 4, "nl_triples")
        with ProcessPoolExecutor(max_workers=min(args.workers, 4)) as ex:
            futs = [ex.submit(nl_worker, s, out_dir) for s in SEEDS]
            summary["nl_workers"] = [f.result() for f in futs]
        summary["nl"] = summarise_nl(out_dir)
        log(f"NL: identity_err_max={summary['nl']['identity_err_max']:.2e} "
            f"sec/call median={summary['nl']['sec_per_call_cpu']['median']:.3f}")
        progress(3, 4, "lin")
        lin_rows, same_exp = lin_worker(out_dir)
        summary["lin"] = {"n_problems": len(lin_rows),
                          "lin_lambda_perp_max": max(r["lambda_dyn"] for r in lin_rows),
                          "lin_static_lambda_perp_median": q([r["lambda_static"] for r in lin_rows], .5),
                          "lin_nonparam_lambda_perp_median": q([r["lambda_nonparam"] for r in lin_rows], .5),
                          "lin_static_phi_perp_median": q([r["phi_perp_static"] for r in lin_rows], .5),
                          "identity_err_max": max(r["identity_err"] for r in lin_rows),
                          "sec_per_call_median": q([r["sec"] for r in lin_rows], .5),
                          "same_exposure": same_exp,
                          "same_exposure_lambda_static_min": min(r["lambda_static"] for r in same_exp),
                          "same_exposure_lambda_dynamic_max": max(r["lambda_dynamic"] for r in same_exp)}
        summary["gpu_timing"] = gpu_timing()
        nl_s = summary["nl"]
        summary["metrics"] = {
            "unit_tests_passed": tests["ok"],
            "identity_err_max": max(nl_s["identity_err_max"], summary["lin"]["identity_err_max"]),
            "lin_lambda_perp_max": summary["lin"]["lin_lambda_perp_max"],
            "sec_per_call": nl_s["sec_per_call_cpu"]["median"]}
        m = summary["metrics"]
        summary["pass_criteria"] = {
            "tests_pass": tests["ok"], "identity_err_le_1e-9": m["identity_err_max"] <= 1e-9,
            "lin_dynamic_lambda_perp_le_1e-10": m["lin_lambda_perp_max"] <= 1e-10,
            "median_sec_per_call_le_2s": m["sec_per_call"] <= 2.0}
        summary["pass"] = all(summary["pass_criteria"][k] for k in
                              ("tests_pass", "identity_err_le_1e-9", "lin_dynamic_lambda_perp_le_1e-10"))
        summary["go_no_go"] = "GO" if summary["pass"] else "NO_GO"
        summary["wall_s"] = time.time() - t0
        summary["timing_note"] = "concurrent run (other round-3 setup tasks share CPU/GPU)"
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        if torch.cuda.is_available():
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20,
                "max_batch_size": "n/a (class-wide J tables in one batch, B = 13824 theta)",
                "vram_used_mb": torch.cuda.max_memory_allocated() / 2 ** 20,
                "utilization_pct": None, "note": "CPU-dominated task; GPU only for J tables and the GPU timing probe"}))
        progress(4, 4, "done", {"identity_err_max": m["identity_err_max"], "sec_per_call": m["sec_per_call"]})
        mark_done("success" if summary["pass"] else "failed",
                  f"tests {tests['summary_line']}; identity_err_max={m['identity_err_max']:.2e}; "
                  f"lin_lambda_perp_max={m['lin_lambda_perp_max']:.2e}; sec/call={m['sec_per_call']:.3f}")
        log(f"done in {time.time() - t0:.0f}s pass={summary['pass']}")
    except Exception as e:  # noqa: BLE001
        summary["error"] = traceback.format_exc()
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        mark_done("failed", f"exception: {e!r}")
        raise


if __name__ == "__main__":
    main()
