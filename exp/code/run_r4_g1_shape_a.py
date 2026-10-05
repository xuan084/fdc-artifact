"""r4_g1_shape_a: Gate G1-shape part A (lifeline 1) + G-size on CR9 visit, dev half D_CR (methodology s3 / R1-R3).

Main layer (gate): CR9 visit, 15 budgets, eps = eps* from exp/results/r4_gates/eps.json, 100 dev permutation streams
(seeds 900-999, identical to G-eps), methods {QFC, B1, B4, A-Ney, A-XY, B5}. All share the pool protocol, the K = 20
checkpoints, billing of every arrival and the 12/15 stop (streams.frontier_runner.run_stream).
  * G1-shape (part A): paired geometric-mean N80 ratio QFC / X <= 0.67 for X in {B1, B4} (point estimate; censored
    N80 = tau_R). B2 / B3 / Peace are parts B / C.
  * G-size: QFC uncensored share >= 0.7 (strict: reached 12/15 before tau_R and no eps-wrong certification).
  * A-share (descriptive): N80 ratio A-Ney / QFC and A-XY / QFC, expected in [0.9, 1.1].
  * B5 is asymptotic (not finite-sample valid): descriptive only.
Secondary read-outs (record only, NOT gates), first 30 dev streams (900-929):
  * HR8 visit (Hillstrom hs3 x category): QFC, B1, B4 at eps = 0.0125 (no eps*_HR8 exists; 0.0125 is the setup /
    timing value and the "6/15 non-trivial" diagnostic value of R1).
  * CR12 visit at eps*: QFC (exact enumeration over 4096) vs QFC-DP (conservative min_t-DP upper bound U_DP >= U_q with
    the DP proposal), plus B1 / B4 for the S-S comparison (CR9 vs CR12).

Statistics: paired log ratios per stream; 95% percentile stream-bootstrap CI (B = 10^4, seed 42); FWER = share of
streams with >= 1 eps-wrong certification (+ one-sided CP upper bound); FCR_old = sum false / sum certified.
Frozen allocations A-Ney / A-XY are read from r4_setup_baselines_a (estimated on D_CR, frozen before this task).
Only dev halves and dev seeds are touched. CPU only, <= 4 workers, BLAS threads = 1; timings are from a concurrent run.

Usage: run_r4_g1_shape_a.py --mode {pilot,full} [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
RES = WS / "exp" / "results"
TASK = "r4_g1_shape_a"
DEV_SEEDS = list(range(900, 1000))
SEC_SEEDS = list(range(900, 930))
HR8_EPS = 0.0125
GATE_RATIO = 0.67
GSIZE_MIN = 0.7
B_BOOT = 10_000
MAIN_METHODS = ["QFC", "B1", "B4", "A-Ney", "A-XY", "B5"]
GATE_OPPONENTS = ["B1", "B4"]
FILES = ["run_r4_g1_shape_a.py", "dsswm/streams/frontier_runner.py", "dsswm/streams/frontier.py",
         "dsswm/baselines/frontier_common.py", "dsswm/baselines/clucb_joint.py", "dsswm/baselines/uniform_rs.py",
         "dsswm/baselines/frozen_alloc.py", "dsswm/baselines/maq_desc.py", "dsswm/certify/quadknap.py",
         "dsswm/envs/pool_replay.py", "dsswm/stats/fp_eb.py"]

from dsswm.baselines.clucb_joint import CLUCBJoint  # noqa: E402
from dsswm.baselines.frontier_common import Method, QFCMethod  # noqa: E402
from dsswm.baselines.maq_desc import MaqQini  # noqa: E402
from dsswm.baselines.uniform_rs import UniformRS  # noqa: E402
from dsswm.envs.pool_replay import PoolReplayEnv  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402
from dsswm.streams.frontier_runner import build_ctx, run_stream, true_policy_values  # noqa: E402

ENVS: dict = {}
CTX: dict = {}
TRUTH: dict = {}
ALLOCS: dict = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def progress(step, total, phase, metric=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    pid = RES / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = RES / f"{TASK}_PROGRESS.json"
    fp = json.loads(pf.read_text()) if pf.exists() else {}
    (RES / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                 "final_progress": fp, "timestamp": datetime.now().isoformat()}))


# ------------------------------------------------------------------------------------------------ QFC-DP (CR12)
class QFCDPMethod(Method):
    """QFC with the conservative min_t-DP upper bound (quadknap.qfc_upper_dp, U_DP >= U_q) and the DP proposal
    (argmax J_hat under rounded-up costs, re-checked). Pool allocation as QFC. Record-only CR12 read-out."""
    name = "QFC-DP"
    alloc_kind = "pool"
    validity = "rigorous"

    def setup(self, ctx):
        from dsswm.baselines.frontier_common import qfc_default_params
        self.params = qfc_default_params(ctx.S, ctx.A, ctx.Q, K=len(ctx.checkpoints))
        self._idx = {tuple(int(x) for x in r): i for i, r in enumerate(ctx.pols)}
        self.n_fallback = 0

    def certify(self, ctx, st):
        from dsswm.certify.quadknap import make_stats, qfc_upper_dp
        stats = make_stats(ctx.w, st.mu_hat, st.n, N=st.N, x_v=self.params["x_v"], binary=ctx.binary, R=ctx.R)
        out = []
        for p in ctx.problems:
            r = qfc_upper_dp(stats, p, self.params["L1"], track=False)
            if r["proposal"] is not None and r["proposal"]["fallback"]:
                self.n_fallback += 1
            U = float(r["U_dp"])
            out.append((bool(U <= ctx.eps), int(self._idx[tuple(r["pi_hat"])]), U))
        return out


def make_method(name):
    if name == "QFC":
        return QFCMethod()
    if name == "B1":
        return CLUCBJoint()
    if name == "B4":
        return UniformRS()
    if name == "B5":
        return MaqQini()
    if name == "A-Ney":
        return QFCMethod("A-Ney", alloc_p=ALLOCS["A-Ney"])
    if name == "A-XY":
        return QFCMethod("A-XY", alloc_p=ALLOCS["A-XY"])
    if name == "QFC-DP":
        return QFCDPMethod()
    raise KeyError(name)


def job(args):
    layer, name, seed = args
    env, ctx = ENVS[layer], CTX[layer]
    J, Js = TRUTH[layer]
    try:
        m = make_method(name)
        s, rows = run_stream(env, m, seed, ctx.problems, ctx.eps, ctx=ctx, J_true=J, J_star=Js)
        s["layer"] = layer
        s["tau_R"] = int(ctx.tau_R)
        s["censored_strict"] = bool(s["censored"] or s["N80"] >= ctx.tau_R)
        if name == "QFC-DP":
            s["dp_fallbacks"] = int(m.n_fallback)
        for r in rows:
            r["layer"] = layer
        return s, rows, None
    except Exception:  # noqa: BLE001
        return {"layer": layer, "method": name, "perm_seed": seed}, [], traceback.format_exc()


# ------------------------------------------------------------------------------------------------ statistics
def cp_upper(k, n, alpha=0.05):
    return 1.0 if k >= n else float(sps.beta.ppf(1 - alpha, k + 1, n - k))


def cp_lower(k, n, alpha=0.05):
    return 0.0 if k <= 0 else float(sps.beta.ppf(alpha, k, n - k + 1))


def boot_mean_ci(x, B=B_BOOT, seed=42):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    bm = x[idx].mean(1)
    return float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975)), bm


def method_table(S, layer, name):
    rr = sorted([s for s in S if s["layer"] == layer and s["method"] == name], key=lambda s: s["perm_seed"])
    if not rr:
        return None
    n = len(rr)
    unc = sum(not s["censored_strict"] for s in rr)
    ev = sum(s["fwer_event"] for s in rr)
    nc = sum(s["n_cert"] for s in rr)
    nf = sum(s["n_false"] for s in rr)
    n80 = np.array([s["N80"] for s in rr], dtype=float)
    unc_n80 = n80[[not s["censored_strict"] for s in rr]]
    secs = np.array([s["sec_total"] for s in rr])
    return {"n_streams": n, "validity": rr[0]["validity"], "alloc_kind": rr[0]["alloc_kind"],
            "uncensored_share": unc / n, "uncensored_cp_lower_1s": cp_lower(unc, n),
            "fwer_events": int(ev), "fwer": ev / n, "fwer_cp_upper_1s": cp_upper(ev, n),
            "certs_total": int(nc), "false_total": int(nf), "fcr_old": nf / nc if nc else 0.0,
            "N80_geomean_censored_at_tauR": float(np.exp(np.log(n80).mean())),
            "N80_median": float(np.median(n80)),
            "N80_uncensored_median": float(np.median(unc_n80)) if unc_n80.size else None,
            "N80_median_frac_tauR": float(np.median(n80) / rr[0]["tau_R"]),
            "k_stop_hist": {str(k): int(v) for k, v in zip(*np.unique([-1 if s["k_stop"] is None else s["k_stop"]
                                                                        for s in rr], return_counts=True))},
            "sec_per_stream_mean": round(float(secs.mean()), 3), "sec_per_stream_max": round(float(secs.max()), 3),
            "billing_ok_all": all(s["billing_ok"] for s in rr), "skipped_total": int(sum(s["skipped"] for s in rr)),
            "reselected_total": int(sum(s["reselected"] for s in rr))}


def paired_ratio(S, layer, num, den, seed=42):
    a = {s["perm_seed"]: s for s in S if s["layer"] == layer and s["method"] == num}
    b = {s["perm_seed"]: s for s in S if s["layer"] == layer and s["method"] == den}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    lr = np.array([np.log(a[k]["N80"] / b[k]["N80"]) for k in seeds])
    lo, hi, bm = boot_mean_ci(lr, seed=seed)
    both_cens = sum(a[k]["censored_strict"] and b[k]["censored_strict"] for k in seeds)
    return {"num": num, "den": den, "n_pairs": len(seeds), "geomean_ratio": float(np.exp(lr.mean())),
            "ci95_ratio": [float(np.exp(lo)), float(np.exp(hi))], "mean_log_ratio": float(lr.mean()),
            "sd_log_ratio": float(lr.std(ddof=1)) if len(lr) > 1 else 0.0,
            "p_boot_1s_ratio_ge_0.80": float((bm >= np.log(0.8)).mean()),
            "p_boot_1s_ratio_ge_0.67": float((bm >= np.log(GATE_RATIO)).mean()),
            "share_streams_num_faster": float((lr < 0).mean()), "share_ties": float((lr == 0).mean()),
            "n_both_censored": int(both_cens),
            "ratio_quantiles": {q: float(np.exp(np.quantile(lr, float(q)))) for q in ("0.1", "0.5", "0.9")}}


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-streams", type=int, default=100)
    ap.add_argument("--n-sec", type=int, default=30)
    a = ap.parse_args()
    out = RES / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out / "samples").mkdir(parents=True, exist_ok=True)
    for f in ("results.jsonl", "streams.jsonl"):
        if (out / f).exists():
            (out / f).unlink()
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = time.time()
    summary = {"task_id": TASK, "mode": a.mode, "started_at": datetime.now().isoformat(),
               "timing_note": "concurrent run (<=4 workers, other r4 tasks sharing the 20-core host)"}
    try:
        eps_doc = json.loads((RES / "r4_gates" / "eps.json").read_text())
        eps_star = float(eps_doc["eps_star"])
        assert eps_doc["G_eps"] == "PASS" and eps_doc["layer"] == "CR9"
        base_a = json.loads((RES / "pilots" / "r4_setup_baselines_a" / "summary.json").read_text())
        ALLOCS["A-Ney"] = np.array(base_a["CR9"]["frozen_A_Ney"])
        ALLOCS["A-XY"] = np.array(base_a["CR9"]["frozen_A_XY"][str(eps_star)]["p"])
        summary.update({"eps_star": eps_star, "eps_source": "exp/results/r4_gates/eps.json",
                        "frozen_allocs_source": "exp/results/pilots/r4_setup_baselines_a/summary.json (CR9)",
                        "frozen_A_XY_info": {k: v for k, v in base_a["CR9"]["frozen_A_XY"][str(eps_star)].items()
                                             if k != "p"}})
        progress(0, 4, "build_envs")
        t0 = time.time()
        for layer, eps, probs in (("CR9", eps_star, fr.cr_problems("visit")), ("CR12", eps_star, fr.cr_problems("visit")),
                                  ("HR8", HR8_EPS, fr.hr_problems("visit"))):
            env = PoolReplayEnv(layer, "dev")
            ctx = build_ctx(env, probs, eps)
            J = true_policy_values(ctx.pols, env.w, env.true_mu("visit"))
            Js = np.array([J[ctx.feas[q]].max() for q in range(ctx.Q)])
            ENVS[layer], CTX[layer], TRUTH[layer] = env, ctx, (J, Js)
        summary["sec_build_envs"] = round(time.time() - t0, 1)
        summary["layers"] = {L: {"S": CTX[L].S, "A": CTX[L].A, "P": int(CTX[L].P), "Q": CTX[L].Q, "eps": CTX[L].eps,
                                 "tau_R_dev": int(CTX[L].tau_R), "replan": int(CTX[L].replan),
                                 "stop_k": int(CTX[L].stop_k), "checkpoints": CTX[L].checkpoints.tolist()}
                             for L in CTX}
        seeds = DEV_SEEDS[:a.n_streams]
        sec_seeds = SEC_SEEDS[:a.n_sec]
        jobs = [("CR9", m, sd) for sd in seeds for m in MAIN_METHODS]
        jobs += [("CR12", m, sd) for sd in sec_seeds for m in ("QFC", "QFC-DP", "B1", "B4")]
        jobs += [("HR8", m, sd) for sd in sec_seeds for m in ("QFC", "B1", "B4")]
        summ_all, errors = [], []
        progress(1, 4, "streams", {"n_jobs": len(jobs)})
        t0 = time.time()
        done = 0
        with get_context("fork").Pool(a.workers) as pool:
            for s, rows, err in pool.imap_unordered(job, jobs, chunksize=1):
                done += 1
                if err:
                    errors.append({**s, "traceback": err})
                    continue
                summ_all.append(s)
                with open(out / "results.jsonl", "a") as f:
                    for r in rows:
                        f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
                with open(out / "streams.jsonl", "a") as f:
                    f.write(json.dumps(s, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
                if s["perm_seed"] in (900, 901, 902):
                    (out / "samples" / f"{s['layer']}_{s['method']}_{s['perm_seed']}.json").write_text(json.dumps(
                        {"summary": s, "trajectory": rows}, default=lambda o: o.tolist() if hasattr(o, "tolist")
                        else str(o)))
                if done % 40 == 0:
                    progress(1, 4, "streams", {"done": done, "n_jobs": len(jobs), "errors": len(errors)})
        summary["sec_streams"] = round(time.time() - t0, 1)
        summary["n_errors"] = len(errors)
        summary["errors"] = errors[:5]
        progress(2, 4, "stats")
        # ---------------- tables
        tables = {L: {m: method_table(summ_all, L, m) for m in ms}
                  for L, ms in (("CR9", MAIN_METHODS), ("CR12", ["QFC", "QFC-DP", "B1", "B4"]),
                                ("HR8", ["QFC", "B1", "B4"]))}
        summary["tables"] = tables
        ratios = {f"QFC/{x}": paired_ratio(summ_all, "CR9", "QFC", x) for x in MAIN_METHODS if x != "QFC"}
        ratios["A-Ney/QFC"] = paired_ratio(summ_all, "CR9", "A-Ney", "QFC")
        ratios["A-XY/QFC"] = paired_ratio(summ_all, "CR9", "A-XY", "QFC")
        summary["CR9_ratios"] = ratios
        # QFC reproduces G-eps exactly on the same streams?
        geps = {d["perm_seed"]: d for d in json.loads(
            (RES / "pilots" / "r4_g_eps_select" / "samples" / "cr9_streams_at_eps_star.json").read_text())}
        qfc = {s["perm_seed"]: s for s in summ_all if s["layer"] == "CR9" and s["method"] == "QFC"}
        mism = [k for k in qfc if k in geps and int(geps[k]["N80"]) != int(qfc[k]["N80"])]
        summary["G_eps_reproduction"] = {"n_checked": sum(k in geps for k in qfc), "n_N80_mismatch": len(mism),
                                         "mismatch_seeds": mism[:10]}
        sec = {"CR12": {f"QFC/{x}": paired_ratio(summ_all, "CR12", "QFC", x) for x in ("B1", "B4", "QFC-DP")},
               "HR8": {f"QFC/{x}": paired_ratio(summ_all, "HR8", "QFC", x) for x in ("B1", "B4")},
               "CR9_first30": {f"QFC/{x}": paired_ratio([s for s in summ_all if s["perm_seed"] in sec_seeds],
                                                         "CR9", "QFC", x) for x in ("B1", "B4")}}
        # CR12 enum vs DP: per-checkpoint U gap on the same streams (identical pool schedules)
        summary["secondary_record_only"] = sec
        # ---------------- gate
        qt = tables["CR9"]["QFC"]
        gate = {}
        for x in GATE_OPPONENTS:
            r = ratios[f"QFC/{x}"]
            gate[x] = {"geomean_ratio": r["geomean_ratio"], "ci95": r["ci95_ratio"],
                       "pass_le_0.67": bool(r["geomean_ratio"] <= GATE_RATIO)}
        g1 = all(v["pass_le_0.67"] for v in gate.values())
        gsize = qt["uncensored_share"] >= GSIZE_MIN
        summary["gate"] = {"G1_shape_partA": {"rule": "paired geometric-mean N80 ratio QFC/X <= 0.67 (point estimate)"
                                              " for X in {B1, B4}; censored N80 = tau_R",
                                              "by_opponent": gate, "pass": bool(g1)},
                           "G_size": {"rule": "QFC strict uncensored share >= 0.7", "value": qt["uncensored_share"],
                                      "pass": bool(gsize)},
                           "QFC_dev_safety": {"fwer_events": qt["fwer_events"], "fcr_old": qt["fcr_old"]},
                           "A_share": {k: ratios[k]["geomean_ratio"] for k in ("A-Ney/QFC", "A-XY/QFC")},
                           "note": "part A only; G1-shape overall needs B2 (part B), B3 (part B), Peace (part C); "
                                   "final verdict in r4_gate_decision"}
        summary["pass"] = bool(g1 and gsize and len(errors) == 0)
        summary["go_no_go"] = "GO" if summary["pass"] else "NO_GO"
        summary["unit_prices_sec_per_stream"] = {L: {m: t["sec_per_stream_mean"] for m, t in tables[L].items() if t}
                                                 for L in tables}
        summary["integrity"] = {"eval_seeds_touched": False, "dev_seeds": [seeds[0], seeds[-1]],
                                "secondary_seeds": [sec_seeds[0], sec_seeds[-1]],
                                "billing_ok_all": all(s["billing_ok"] for s in summ_all)}
        summary["code_sha256"] = {f: sha(CODE / f) for f in FILES}
        summary["wall_min"] = round((time.time() - t_start) / 60, 2)
        summary["finished_at"] = datetime.now().isoformat()
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist()
                                                     if hasattr(o, "tolist") else str(o)))
        progress(4, 4, "done")
        mark_done("success", f"G1A={'PASS' if g1 else 'FAIL'} Gsize={'PASS' if gsize else 'FAIL'} "
                             f"QFC/B1={ratios['QFC/B1']['geomean_ratio']:.3f} QFC/B4={ratios['QFC/B4']['geomean_ratio']:.3f}")
        print(json.dumps(summary["gate"], indent=1))
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        (out / "error.txt").write_text(tb)
        mark_done("failed", tb[-500:])
        raise


if __name__ == "__main__":
    main()
