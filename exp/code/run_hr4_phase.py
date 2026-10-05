"""HR4 resolution phase transition, one resolution chunk (cand_f / cand_a; methodology 2.2 / 6.1 / Figure 3).

Serves hr4_phase_fine (f = lock['hr4_fine_f'], i.e. 4 because f* = 2) and hr4_phase_coarse (f = 0.5): the task id
selects the f-levels; f = 1 and f = f* on the SAME instances come from hr1_r2 / hr1_r3 and are merged in analysis.

Off-grid truth (R3 generator: continuous theta*), learner grid G_f, local inflation. Methods per (instance, stream):
  JPC_infl   joint LR set on G_f + inflated minimax certificate (R_grid + 2 max_alive eta_loc <= eps) + DDA;
             honest refusal when the data-free floor 2 min_g eta_loc > eps holds
  B8_infl    full identification (J-width over alive set < eps/2), then the inflated certificate (HR5 arm)
  B3         G_1 linearisation + Wald chi2_12 (reference for 'stream_ratio_vs_B3_by_f'; f-independent, not inflated)
  JPC        uncorrected JPC on G_f (diagnostic only, --diag-streams): completion / FCR / theta*-cell survival as a
             function of h/h* -- the only arm on which a completion-vs-h curve is non-degenerate when the floor binds
Every grid row records h (median cell half-diagonal of G_f, raw coords) and h_theta* (half-diagonal of the theta*
cell), the literal h*(q) = eps / (L_J(q) sqrt(cond I_xi)) with I_xi = realised design Fisher of that method's ledger
(all rounds incl. n0) at the grid MLE, plus the identified-subspace and cond=1 variants (prereg_lock h_star_method),
h/h*, completion, new steps, eta_loc (min / theta*-cell / alive max).
Code paths for the methods, scoring, J tables and eta_loc are imported unchanged from run_hr1_chunk.py.

Pilot: dev instances 690-694 x 10 problems x 1 stream (seed 42).  Full: lock range x 3 streams x 15 problems
(requires assert_locked()). Incremental results.jsonl; finished (instance, stream, method) units skipped on restart.

Usage: run_hr4_phase.py --task hr4_phase_fine --mode {pilot,full} [--workers 2] [--smoke] [--diag-streams 0]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hr1_chunk as base  # noqa: E402
from dsswm.baselines.glm_linearised import fisher_rounds  # noqa: E402
from dsswm.certify.eta_loc import build_plans, eta_loc, j_values  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import cell_index_of, half_widths, ladder_grid  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402
from dsswm.streams.offgrid import make_offgrid_instance  # noqa: E402

WS, RES_ROOT, CACHE, LOCK = base.WS, base.RES_ROOT, base.CACHE, base.LOCK
EPS, DELTA, N0 = base.EPS, base.DELTA, base.N0
C_KNOWN, RHO_RET, NMAX = base.C_KNOWN, base.RHO_RET, base.NMAX
fkey = base.fkey
PILOT_SEEDS = [690, 691, 692, 693, 694]
PILOT_NPROB = 10
GRID_METHODS = ["JPC_infl", "B8_infl", "JPC"]
MEM_BUDGET = 6e7          # NLPropagator chunk budget: f=4 J-table peak 5.5 GB at 1.5e8 -> keep <= 5 GB on shared 4090


# ------------------------------------------------------------------------------------ phase 1: tables (GPU)
J_CHUNK = 93312           # classes per prop.tables() call: f=4 J-table GPU peak 5.5 GB (one call) -> 1.4 GB (verified
                          # bit-identical up to 1e-14 against the single-call table on dev problem nl695_q0)


def j_table_chunked(prop, params, B, q):
    out = []
    for s in range(0, B, J_CHUNK):
        ps = {k: (v[s:s + J_CHUNK] if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v)
              for k, v in params.items()}
        out.append(prop.j_table(ps, q.policies, q.loads0, q.engaged0, q.H, base.U1(q)))
    return np.concatenate(out)


def precompute(seeds, n_prob, fs, dev, log):
    """G_1 J tables (c_q) + G_f J tables, eta_loc, L_J for every f in fs; truth-free, cached, file-locked.
    Harness only: J_true at the continuous theta*. Same cache files / format as run_hr1_chunk.precompute."""
    prop = NLPropagator(2, 2, NMAX, make_offgrid_instance(seeds[0], "R3").env.aspace, C_KNOWN, RHO_RET, device=dev,
                        mem_budget_elems=MEM_BUDGET)
    meta = {"cq": {}, "jtrue": {}, "eta_min_over_eps": {}, "sec": {}, "h": {}}
    probs = []
    for s in seeds:
        inst = make_offgrid_instance(s, "R3", stream=0)
        for q in inst.problems[:n_prob]:
            probs.append((s, q, base.vstar_of(inst.truth)))
    for f, need_eta in [(1, False)] + [(f, True) for f in fs]:
        ncl = NLClass(ladder_grid(f), device=dev)
        W = half_widths(ncl)
        meta["h"][fkey(f)] = float(np.median(np.linalg.norm(W, axis=1)))
        params = None
        d = CACHE / f"jtables_f{fkey(f)}"
        d.mkdir(parents=True, exist_ok=True)
        t_j, t_e = [], []
        with open(d / ".hr1_chunk.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for (s, q, v) in probs:
                pj = d / f"{q.pid}_raw.npy"
                if not pj.exists():
                    params = params or ncl.torch_params()
                    t0 = time.perf_counter()
                    raw = j_table_chunked(prop, params, ncl.B, q)
                    base.atomic_save(pj, raw.astype(np.float32) if f == 4 else raw)
                    t_j.append(time.perf_counter() - t0)
                pe = d / f"{q.pid}_etaloc_raw.npy"
                if need_eta and (not pe.exists() or not (d / f"{q.pid}_LJ.npy").exists()):
                    t0 = time.perf_counter()
                    eta, _, _, info = eta_loc(prop, ncl, q, return_parts=True)
                    base.atomic_save(d / f"{q.pid}_LJ.npy", np.array([info["parts"]["L_J"]]))
                    base.atomic_save(pe, eta)
                    t_e.append(time.perf_counter() - t0)
            fcntl.flock(lk, fcntl.LOCK_UN)
        meta["sec"][fkey(f)] = {"jtable_new_mean": float(np.mean(t_j)) if t_j else None, "n_jtable_new": len(t_j),
                                "eta_new_mean": float(np.mean(t_e)) if t_e else None, "n_eta_new": len(t_e),
                                "Theta": int(ncl.B), "h_cell_halfdiag_median": meta["h"][fkey(f)]}
        log(f"tables f={fkey(f)} |Theta|={ncl.B} h={meta['h'][fkey(f)]:.3f}: new J {len(t_j)} "
            f"({meta['sec'][fkey(f)]['jtable_new_mean']}), new eta {len(t_e)} ({meta['sec'][fkey(f)]['eta_new_mean']})")
        del ncl, params
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for (s, q, v) in probs:
        c = 1.0 / float(np.load(CACHE / "jtables_f1" / f"{q.pid}_raw.npy").max())
        meta["cq"][q.pid] = c
        meta["jtrue"][q.pid] = j_values(prop, build_plans(prop, q), v[None], q.utility.w, q.utility.w_ret)[0] * c
        for f in fs:
            meta["eta_min_over_eps"].setdefault(fkey(f), []).append(
                float(np.load(CACHE / f"jtables_f{fkey(f)}" / f"{q.pid}_etaloc_raw.npy").min()) * c / EPS)
    if torch.cuda.is_available():
        meta["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20
    del prop
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return meta


# ------------------------------------------------------------------------------------ phase 2: learners (CPU)
_ORIG_INIT_LR = base._init_lr


def _init_lr_tracked(arm):
    """Same as run_hr1_chunk._init_lr; additionally exposes the method's LR set and its observation list on the arm
    (read-only; used by ArmH.row to compute the realised-design Fisher / h* at the moment each row is written)."""
    inst, env, h, lr, led = _ORIG_INIT_LR(arm)
    arm._lr, arm._obs = lr, list(inst.init_obs)
    rec = led.record

    def record(obs, tag, problem_id=None, cost=1.0):
        rec(obs, tag, problem_id, cost)
        arm._obs.append(obs)
    led.record = record
    return inst, env, h, lr, led


class ArmH(base.Arm):
    """run_hr1_chunk.Arm + per-row resolution fields (h, h*, h/h*). Harness-side: theta* only enters h_theta*."""

    def __init__(self, seed, stream, f, n_prob, meta, need_lr=True, need_eta=True):
        super().__init__(seed, stream, f, n_prob, meta, need_lr=need_lr, need_eta=need_eta)
        self.h = float(meta["h"][fkey(f)])
        W = half_widths(self.ncl)
        self.h_star_cell = float(np.linalg.norm(W[self.star]))
        d = CACHE / f"jtables_f{fkey(f)}"
        self.LJ = []
        for q, c in zip(self.problems, self.cq):
            p = d / f"{q.pid}_LJ.npy"
            self.LJ.append(float(np.load(p)[0]) * c if (need_eta and p.exists()) else None)
        self._lr, self._obs = None, []

    def hstar_fields(self, k):
        if self._lr is None or self.LJ[k] is None:
            return {}
        pairs = {}
        for o in self._obs:
            key = (self.prop.codec.encode(o.loads, o.engaged), int(o.action))
            pairs[key] = pairs.get(key, 0) + 1
        kh = self._lr.mle()
        d = self.V.shape[1]
        I = np.zeros((d, d))
        for (code, a), n in pairs.items():
            I += n * fisher_rounds(self.V[kh], self.prop.codec, self.aspace, code, [a], C_KNOWN, 2, 2)[0]
        ev = np.linalg.eigvalsh(I)
        lmax = float(ev.max())
        cond = lmax / float(max(ev.min(), 1e-12 * lmax))
        idn = ev[ev >= 1e-6 * lmax]
        cond_id = float(idn.max() / idn.min())
        LJ = self.LJ[k]
        hs, hs_id, hs_1 = EPS / (LJ * math.sqrt(cond)), EPS / (LJ * math.sqrt(cond_id)), EPS / LJ
        return {"h": self.h, "h_theta_star_cell": self.h_star_cell, "L_J_norm": LJ, "fisher_cond": cond,
                "fisher_cond_identified": cond_id, "n_identified_dirs": int(len(idn)), "n_rounds_design": len(self._obs),
                "h_star": hs, "h_star_identified": hs_id, "h_star_cond1": hs_1,
                "log_h_over_hstar": math.log(self.h / hs), "log_h_over_hstar_identified": math.log(self.h / hs_id),
                "log_h_over_hstar_cond1": math.log(self.h / hs_1),
                "eta_loc_star_cell": float(self.eta[k][self.star]), "eta_loc_min": float(self.eta[k].min())}

    def row(self, k, method, status, pi, steps, extra=None, charged=None):
        r = super().row(k, method, status, pi, steps, extra, charged)
        r["kind"] = "hr4"
        r["level"] = f"R3_f{fkey(self.f)}"
        r.update(self.hstar_fields(k))
        return r


def run_unit(seed, stream, group, methods, n_prob, meta, f, want_samples):
    base._init_lr = _init_lr_tracked                     # loky workers re-import run_hr1_chunk: patch every call
    torch.set_num_threads(1)
    t0 = time.time()
    rows, samples, errors = [], [], []
    try:
        arm = ArmH(seed, stream, f, n_prob, meta) if group == "grid" else ArmH(seed, stream, 1, n_prob, meta,
                                                                                need_eta=False)
    except Exception as e:  # noqa: BLE001
        return rows, samples, [{"instance": seed, "stream": stream, "method": group, "f": f, "error": repr(e),
                                "tb": traceback.format_exc()}], {"instance": seed, "stream": stream, "group": group,
                                                                 "f": f, "methods": methods, "sec": time.time() - t0}
    t_arm = time.time() - t0
    msec = {}
    for m in methods:
        tm = time.time()
        try:
            if m in ("JPC", "JPC_infl"):
                r, s = base.run_jpc(arm, m == "JPC_infl", want_samples)
                samples += s
            elif m == "B8_infl":
                r = base.run_b8(arm)
            elif m == "B3":
                r, s = base.run_b3(arm, "B3", want_samples)
                samples += s
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "stream": stream, "method": m, "f": f, "error": repr(e),
                           "tb": traceback.format_exc()})
        msec[m] = time.time() - tm
    return rows, samples, errors, {"instance": seed, "stream": stream, "group": group, "f": f, "methods": methods,
                                   "arm_sec": t_arm, "method_sec": msec, "sec": time.time() - t0,
                                   "rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}


# ------------------------------------------------------------------------------------ summaries
def logistic_fit(x, y, n_boot=1000, seed=42):
    """Completion ~ sigmoid(a + b x), x = log(h/h*); inflection x0 = -a/b with a percentile bootstrap CI.
    Returns status 'degenerate' when y is constant (no transition observable in the data)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) < 4 or y.min() == y.max():
        return {"status": "degenerate", "n": int(len(y)), "completion": float(y.mean()) if len(y) else None,
                "x_range": [float(x.min()), float(x.max())] if len(x) else None}
    from scipy.optimize import minimize

    def fit(xx, yy):
        def nll(p):
            z = p[0] + p[1] * xx
            return float(np.sum(np.logaddexp(0, z) - yy * z) + 1e-4 * (p ** 2).sum())
        p = minimize(nll, np.zeros(2), method="BFGS").x
        return p
    a, b = fit(x, y)
    rng = np.random.default_rng(seed)
    x0s = []
    for _ in range(n_boot):
        i = rng.integers(len(x), size=len(x))
        if y[i].min() == y[i].max():
            continue
        aa, bb = fit(x[i], y[i])
        if abs(bb) > 1e-9:
            x0s.append(-aa / bb)
    x0 = -a / b if abs(b) > 1e-9 else None
    ci = [float(np.quantile(x0s, .025)), float(np.quantile(x0s, .975))] if len(x0s) > 20 else None
    return {"status": "ok", "n": int(len(y)), "a": float(a), "b": float(b), "inflection_log_h_over_hstar": x0,
            "inflection_ci95": ci, "n_boot_ok": len(x0s),
            "inflection_in_[h*/2,2h*]": (bool(ci[0] >= -math.log(2) and ci[1] <= math.log(2)) if ci else None)}


def summarize(rows, n_prob):
    out = {}
    for key in sorted({(r["method"], r["f"]) for r in rows}, key=lambda t: (t[1], t[0])):
        m, f = key
        R = [r for r in rows if (r["method"], r["f"]) == key]
        cert = [r for r in R if r["status"] == "CERTIFIED"]
        nfc = sum(r["false_cert"] for r in R)
        lo, hi = clopper_pearson(nfc, len(R), 0.05)
        units = sorted({(r["instance"], r["stream"]) for r in R})
        alive = [r["theta_cell_alive"] for r in R if r.get("theta_cell_alive") is not None]
        rec = {"method": m, "f": f, "n": len(R), "completion": len(cert) / len(R),
               "status_counts": {s: sum(r["status"] == s for r in R) for s in sorted({r["status"] for r in R})},
               "refused_floor": int(sum(bool(r.get("refused_floor")) for r in R)),
               "stream_steps_charged_mean": float(np.mean([sum(r["new_env_steps"] for r in R
                                                               if (r["instance"], r["stream"]) == u) for u in units])),
               "steps_consumed_median": float(np.median([r["steps_consumed"] for r in R])),
               "false_certs": int(nfc), "fcr_cp_upper": hi, "fcr_cp": [lo, hi],
               "theta_cell_survival": float(np.mean(alive)) if alive else None,
               "wall_clock_per_problem_s": float(np.mean([r["wall_clock_s"] for r in R]))}
        if any("h_star" in r for r in R):
            H = [r for r in R if "h_star" in r]
            for nm in ("", "_identified", "_cond1"):
                v = np.array([r[f"log_h_over_hstar{nm}"] for r in H])
                rec[f"log_h_over_hstar{nm}"] = {"median": float(np.median(v)), "q10": float(np.quantile(v, .1)),
                                                "q90": float(np.quantile(v, .9))}
            rec["h"] = H[0]["h"]
            rec["h_star_median"] = float(np.median([r["h_star"] for r in H]))
            rec["share_h_le_hstar"] = float(np.mean([r["log_h_over_hstar"] <= 0 for r in H]))
            rec["eta_loc_min_over_eps_median"] = float(np.median([r["eta_loc_min"] for r in H])) / EPS
            rec["eta_loc_star_cell_over_eps_median"] = float(np.median([r["eta_loc_star_cell"] for r in H])) / EPS
            y = [float(r["status"] == "CERTIFIED") for r in H]
            rec["logistic_completion_vs_log_h_over_hstar"] = logistic_fit([r["log_h_over_hstar"] for r in H], y)
            rec["logistic_completion_vs_log_h_over_hstar_cond1"] = logistic_fit(
                [r["log_h_over_hstar_cond1"] for r in H], y)
        out[f"{m}@f{fkey(f)}"] = rec
    return out


def ratios(rows, summ, fs):
    """Instance-level log stream-step ratios (descriptive, per chunk). Binding values come from the analysis merge."""
    insts = sorted({r["instance"] for r in rows})

    def tot(m, f, i):
        S = sorted({r["stream"] for r in rows if r["instance"] == i and r["method"] == m and r["f"] == f})
        return float(np.mean([sum(r["new_env_steps"] for r in rows if r["method"] == m and r["f"] == f
                                  and r["instance"] == i and r["stream"] == s) for s in S])) if S else None
    out = {}
    for f in fs:
        for num, den, fd in (("JPC_infl", "B3", 1), ("JPC_infl", "B8_infl", f), ("JPC", "B3", 1)):
            lr_ = np.array([math.log((tot(num, f, i) + 1) / (tot(den, fd, i) + 1)) for i in insts
                            if tot(num, f, i) is not None and tot(den, fd, i) is not None])
            if not len(lr_):
                continue
            rng = np.random.default_rng(42)
            bs = [lr_[rng.integers(len(lr_), size=len(lr_))].mean() for _ in range(2000)]
            out[f"{num}_over_{den}@f{fkey(f)}"] = {
                "ratio_geo_mean": float(math.exp(lr_.mean())),
                "ci95_chunk_bootstrap2000": [float(math.exp(np.quantile(bs, .025))),
                                             float(math.exp(np.quantile(bs, .975)))],
                "per_instance": [float(math.exp(x)) for x in lr_],
                "note": "censored rows charged T_max=3000/problem; ratio of charged stream totals"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hr4_phase_fine")
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--diag-streams", default="0", help="streams on which the uncorrected JPC diagnostic runs")
    ap.add_argument("--no-b3", action="store_true", help="skip the B3 reference (taken from hr1_r3 in analysis)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    TASK = args.task
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    assert lock.get("version") == 2, "prereg lock v2 required"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    fstar = int(lock["f_star"])
    if TASK == "hr4_phase_fine":
        fs = [float(lock.get("hr4_fine_f", 2 if fstar == 4 else 4))]
    elif TASK == "hr4_phase_coarse":
        fs = [0.5]
    else:
        raise ValueError(TASK)
    fs = [int(f) if float(f).is_integer() else f for f in fs]
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else "") + args.tag)
    diag_streams = {int(x) for x in args.diag_streams.split(",") if x != ""}
    quiet = args.smoke or bool(args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.jsonl"

    def progress(step, total, phase, metric=None):
        if quiet:
            return
        (RES_ROOT / f"{TASK}_PROGRESS.json").write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summary):
        if quiet:
            return
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

    if not quiet:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if pilot:
        seeds, n_prob, streams = PILOT_SEEDS, PILOT_NPROB, [0]
        assert max(seeds) < 10000, "dev seeds only in pilot"
    else:
        rng_ = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds = [s for a, b in rng_ for s in range(a, b + 1)]
        n_prob, streams = int(lock["eval_manifest"]["K_problems"]), [0, 1, 2]
    if args.smoke:
        seeds, n_prob = seeds[:1], 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "level": "R3 (off-grid truth, local inflation)",
               "f_levels": fs, "f_star": fstar, "eps": EPS, "delta": DELTA, "instances": [seeds[0], seeds[-1]],
               "n_instances": len(seeds), "n_problems_per_stream": n_prob, "streams": streams,
               "eval_seeds_touched": not pilot, "generator_hash": generator_hash(), "concurrent_run": True,
               "tmax_stepwise": base.TMAX_STEP, "lock_status": lock.get("status"),
               "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional"),
               "methods": GRID_METHODS + ([] if args.no_b3 else ["B3"]),
               "note_merge": "f=1 and f*=2 points of the HR4 curve come from hr1_r2 / hr1_r3 on the same instances",
               "note_hstar": "literal h* per prereg_lock.h_star_method; h = median G_f cell half-diagonal (raw coords)"}
    log(f"start task={TASK} mode={args.mode} f={fs} device={dev} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) x "
        f"{n_prob} problems x streams {streams}, lock={lock.get('status')}, generator_hash={generator_hash()}")
    try:
        progress(1, 4, "gpu: J tables + eta_loc")
        meta = precompute(seeds, n_prob if streams == [0] else 15, fs, dev, log)
        summary["eta_loc_min_over_eps"] = {fk: {"median": float(np.median(v)), "min": float(np.min(v)),
                                                "share_floor_binding": float(np.mean(np.array(v) > 0.5))}
                                           for fk, v in meta["eta_min_over_eps"].items()}
        summary["table_sec"] = meta["sec"]
        summary["gpu_peak_mb"] = meta.get("gpu_peak_mb")
        log(f"min_g eta_loc/eps by f: {summary['eta_loc_min_over_eps']}; gpu peak {meta.get('gpu_peak_mb')} MB")
        done_units = set()
        if res_path.exists():
            cnt = {}
            for x in open(res_path):
                r = json.loads(x)
                u = (r["instance"], r["stream"], r["method"], r["f"])
                cnt[u] = cnt.get(u, 0) + 1
            done_units = {u for u, c in cnt.items() if c >= n_prob}
            log(f"resume: {len(done_units)} finished units")
        jobs = []
        for s in seeds:
            for st in streams:
                for f in fs:
                    ms = [m for m in GRID_METHODS if not (m == "JPC" and st not in diag_streams)
                          and (s, st, m, f) not in done_units]
                    if ms:
                        jobs.append((s, st, "grid", ms, f))
                if not args.no_b3 and (s, st, "B3", 1) not in done_units:
                    jobs.append((s, st, "B3", ["B3"], 1))
        progress(2, 4, "cpu: methods", {"jobs": len(jobs)})
        from joblib import Parallel, delayed
        errors, samples, unit_sec = [], [], []
        n_done = 0
        gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
            delayed(run_unit)(s, st, g, ms, n_prob, meta, f, s == seeds[0] and st == 0) for (s, st, g, ms, f) in jobs)
        with open(res_path, "a") as fh:
            for rr, ss, ee, us in gen:
                bad = {(e["instance"], e["stream"], e["method"]) for e in ee}
                for r in rr:
                    if (r["instance"], r["stream"], r["method"]) not in bad:
                        fh.write(json.dumps(r, default=float) + "\n")
                fh.flush()
                errors += ee
                samples += ss
                unit_sec.append(us)
                n_done += 1
                for e in ee:
                    log(f"ERROR {e['instance']} s{e['stream']} {e['method']} f={e['f']}: {e['error']}\n{e['tb']}")
                log(f"unit {n_done}/{len(jobs)}: inst {us['instance']} s{us['stream']} {us['group']} f={us['f']} "
                    f"{us['sec']:.1f}s (arm {us.get('arm_sec', 0):.1f}s, {us.get('method_sec')}) rss {us.get('rss_gb')} GB")
                progress(2, 4, "cpu: methods", {"jobs_done": n_done, "jobs": len(jobs)})
        rows = [json.loads(x) for x in open(res_path)]
        (out_dir / "samples" / "samples.json").write_text(json.dumps(samples, indent=1, default=float))
        (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))
        progress(3, 4, "summary")
        summ = summarize(rows, n_prob)
        rat = ratios(rows, summ, fs)
        eb = [r for r in rows if "env_n_steps" in r]
        eb_ok = all(r["env_n_steps"] == r["lr_n_rounds"] for r in eb)
        summary.update({"by_method": summ, "ratios": rat, "crashes": len(errors), "unit_sec": unit_sec,
                        "n_rows": len(rows),
                        "evidence_boundary": {"rows_checked": len(eb), "env_n_steps_eq_lr_n_rounds": eb_ok}})
        # timing projection: full = 32 instances x 3 streams x 15 problems; JPC diagnostic on |diag_streams| streams
        n_inst_full = 32
        per = {}
        for us in unit_sec:
            for m, sec in (us.get("method_sec") or {}).items():
                per.setdefault(m, []).append(sec)
            per.setdefault(f"arm_{us['group']}", []).append(us.get("arm_sec", 0.0))
        n_full_streams = {"JPC": len(diag_streams)}
        cpu = {m: float(np.mean(v)) / n_prob * 15 * n_inst_full * n_full_streams.get(m, 3) if not m.startswith("arm")
               else float(np.mean(v)) * n_inst_full * 3 for m, v in per.items()}
        cpu_sec = sum(cpu.values())
        tj = sum((meta["sec"].get(fkey(f), {}).get("jtable_new_mean") or 0) +
                 (meta["sec"].get(fkey(f), {}).get("eta_new_mean") or 0) for f in fs) + \
            (meta["sec"].get("1", {}).get("jtable_new_mean") or 0)
        gpu_min = tj * n_inst_full * 15 / 60
        proj = cpu_sec / args.workers / 60 + gpu_min
        summary["timing_projection"] = {"cpu_sec_by_component": cpu, "projected_full_cpu_sec": cpu_sec,
                                        "workers": args.workers, "projected_full_cpu_wall_min": cpu_sec / args.workers / 60,
                                        "gpu_table_sec_per_problem": tj, "projected_full_gpu_table_min": gpu_min,
                                        "projected_full_total_min": proj,
                                        "peak_rss_gb_per_worker": max([u.get("rss_gb") or 0 for u in unit_sec],
                                                                      default=None),
                                        "note": "GPU tables serial before CPU phase; concurrent run (4 tasks on 20 cores)"}
        pc = {"zero_crashes": len(errors) == 0, "evidence_boundary_assertions_hold": eb_ok,
              "runs_end_to_end_every_f": all(f"{m}@f{fkey(f)}" in summ for f in fs for m in ("JPC_infl", "B8_infl")),
              "projected_full_wall_le_60_min": proj <= 60}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for k_, v in summ.items():
            log(f"{k_:14s} completion {v['completion']:.2f} {v['status_counts']} refused {v['refused_floor']} "
                f"stream steps {v['stream_steps_charged_mean']:.0f} FCR {v['false_certs']}/{v['n']} "
                f"theta-cell {v['theta_cell_survival']} log(h/h*) {v.get('log_h_over_hstar', {}).get('median')}")
        for k_, v in rat.items():
            log(f"ratio {k_}: {v['ratio_geo_mean']:.3f} CI {[round(x, 3) for x in v['ci95_chunk_bootstrap2000']]}")
        log(f"timing projection full: {proj:.1f} min; done in {time.time() - T0:.0f}s: {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}; projected full {proj:.1f} min")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
