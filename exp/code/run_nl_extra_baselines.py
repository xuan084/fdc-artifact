"""nl_extra_baselines: extra baselines B7-B10 on E1-NL-S (kappa_dyn = high), same protocol / instance seeds /
noise seeds / problem streams as nl_reuse_kappa_high (common random numbers: every method runs on its own fresh
copy of the platform with the same seeds).

Usage: run_nl_extra_baselines.py --mode {pilot,full} [--workers 4] [--instances N] [--problems K] [--tmax T]

Methods (all use the same finite class, the same shared n0=20 initial rounds and the same per-step platform):
  JPC        re-run of the main method (joint LR set + exact minimax certificate + DDA), for pairing / reproducibility
  B7         minimax-regret robust planning on Theta_t (initial data only), NO acquisition. Acts on every problem;
             reports true regret and the zero-cost certifiable rate (R_bar <= eps).
  B8         full identification, then plan: probe until  max_{pi in Pi_q} [max_{theta,theta' in Theta_t}
             |J_theta(pi) - J_theta'(pi)|] < tau  with tau = eps/2 (=> argmax_theta_hat has regret <= eps for every
             theta in Theta_t, i.e. the same eps-guarantee), ledger reused across problems. Probes: the same DDA LP,
             with the models farthest from theta_hat in J-value (not in decision regret) as targets.
  B8tau      same, tau = eps (favourable to B8: only a 2*eps guarantee at stop; status reported by the certificate).
  B8fam      full identification over Pi_family = union of all candidate policies of the instance's stream,
             tau = eps/2 (classic 'identify the model, then plan everything').
  B9-{5,20,100}  ensemble-consistency stopping: online Poisson-bootstrap ensemble of finite-class MLEs; stop and
             claim certification as soon as every member's argmax policy agrees. Probes: DDA LP against the
             disagreeing members. Not a coverage guarantee -> FCR expected above delta.
  B10        reward-free / multi-policy evaluation: problem-agnostic G-optimal coverage design over the interaction
             library (every observable state x legal action), linearised Fisher at theta_hat:
             a_t = argmin_a max_{x in library} <F_x, (V + F_a)^{-1}>; stopping = plug-in of the collected data into
             the same LR set + exact certificate (sound).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")      # <= 4 CPU workers in total; each worker single-threaded

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

import run_nl_reuse_kappa_high as R  # noqa: E402  (same Ctx, JPC, score_row, J-table cache names)
from dsswm.acquire.nl_kl_dda import dda_choose  # noqa: E402
from dsswm.baselines.glm_linearised import fisher_rounds  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, trichotomy  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.bootstrap import cluster_bootstrap  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.stats.km_rmst import kaplan_meier, rmst  # noqa: E402
from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, generator_hash  # noqa: E402

TASK = "nl_extra_baselines"
EPS, DELTA, TOP_M, LOG_THR = R.EPS, R.DELTA, R.TOP_M, R.LOG_THR
C_KNOWN = R.C_KNOWN
DEFAULT_METHODS = "JPC,B7,B8,B8tau,B8fam,B9-5,B9-20,B9-100,B10"


def progress(res_root, step, total, phase, metric=None):
    (res_root / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
        "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))


def mark_done(res_root, status, summary):
    pid = res_root / f"{TASK}.pid"
    if pid.exists():
        pid.unlink()
    pf = res_root / f"{TASK}_PROGRESS.json"
    fp = {}
    if pf.exists():
        try:
            fp = json.loads(pf.read_text())
        except ValueError:
            pass
    (res_root / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary,
                                                        "final_progress": fp, "timestamp": datetime.now().isoformat()}))


def _init(ctx):
    inst = ctx.fresh_instance()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    led = EvidenceLedger([lr])
    for o in inst.init_obs:
        led.record(o, "initial")
    return inst, env, h, lr, led


# ------------------------------------------------------------------ B7: minimax-regret robust planning, no acquisition
def run_b7(ctx):
    inst, env, h, lr, led = _init(ctx)
    mask = lr.mask().numpy()
    rows = []
    for k, q in enumerate(ctx.problems):
        cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
        ok = cert["status"].value == "CERTIFIED"
        rows.append(R.score_row(ctx, k, "B7", "CERTIFIED" if ok else "ACT_UNCERTIFIED", cert["pi"], 0, not ok,
                                {"r_bar_end": cert["r_bar"], "set_size_end": int(mask.sum()),
                                 "theta_star_in_set": bool(mask[ctx.ti]), "zero_cost_cert": ok, "H": q.H,
                                 "n_policies": len(q.policies), "rollouts": 0, "wall_clock_s": 0.0,
                                 "note": "acts with the minimax-regret policy without acquiring data"}))
    assert env.n_steps == NL_DEFAULTS["n0"]
    return rows


# ------------------------------------------------------------------ B8: full identification, then plan
def run_b8(ctx, tmax, tau, family="problem", method="B8", samples=None):
    inst, env, h, lr, led = _init(ctx)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 808, int(round(tau * 1e4)), family == "stream"])
    Jfam = np.concatenate(ctx.J, 1) if family == "stream" else None
    rows = []
    n_obs = NL_DEFAULTS["n0"]
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        J = Jfam if family == "stream" else ctx.J[k]
        steps, status = 0, None
        traj = []
        width = None
        while True:
            mask = lr.mask().numpy()
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                status = "MODEL_CONFLICT"
                break
            Jm = J[idx]
            width = float((Jm.max(0) - Jm.min(0)).max())
            if width < tau:
                cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
                status = "CERTIFIED" if cert["status"].value == "CERTIFIED" else "IDENTIFIED_UNCERTIFIED"
                break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            dist = np.abs(Jm - J[kh][None]).max(1)
            order = np.argsort(-dist)
            blk = [int(idx[o]) for o in order[:TOP_M] if dist[o] > tau / 2] or [int(idx[o]) for o in order[:TOP_M]]
            margins = LOG_THR - lr.log_ratio().numpy()[blk]
            code = ctx.prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk], ctx.pe[blk],
                                 margins, ctx.inc, ctx.prop, ctx.legal, rng)
            led.record(h.step(a), "probe", q.pid)
            steps += 1
            if samples is not None and len(traj) < 200 and steps % 5 == 1:
                traj.append({"step": steps, "action": a, "set_size": int(mask.sum()), "id_width": width,
                             "mode": info["mode"]})
        n_obs += steps
        assert lr.n_rounds == n_obs == env.n_steps, "step accounting mismatch"
        mask = lr.mask().numpy()
        cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
        # plan: argmax under theta_hat (identified model); also report the minimax policy's certificate
        pi = int(np.argmax(ctx.J[k][lr.mle()]))
        extra = {"tau": tau, "family": family, "id_width_end": width, "r_bar_end": cert["r_bar"],
                 "cert_status_end": cert["status"].value, "minimax_pi": cert["pi"],
                 "set_size_end": int(mask.sum()), "theta_star_in_set": bool(mask[ctx.ti]),
                 "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED"), "wall_clock_s": time.perf_counter() - t0,
                 "rollouts": 0, "H": q.H, "n_policies": len(q.policies)}
        rows.append(R.score_row(ctx, k, method, status, pi, steps, status in ("NEED_DATA", "MODEL_CONFLICT"), extra))
        if samples is not None and k < 3 and ctx.seed < 2:
            samples.append({"instance": ctx.seed, "problem_index": k, "method": method, "status": status,
                            "steps": steps, "trajectory_head": traj[:40], "J_true": ctx.J[k][ctx.ti].tolist(),
                            "policy": pi, "true_regret": rows[-1]["true_regret"]})
    return rows


# ------------------------------------------------------------------ B9: ensemble-consistency stopping
def run_b9(ctx, tmax, M, samples=None):
    inst, env, h, lr, led = _init(ctx)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 909, M])
    wrng = np.random.default_rng([ctx.seed, ctx.noise, 919, M])
    B = ctx.ncl.B
    tie = wrng.random((M, B)) * 1e-9                 # member-specific random tie-break (random init analogue)
    cum = np.zeros((M, B))

    def add(obs):
        ll = ctx.prop.loglik(ctx.LT, [obs])[:, 0].numpy()
        w = wrng.poisson(1.0, M).astype(float)       # online (Poisson) bootstrap weights
        cum[:] += w[:, None] * ll[None]

    for o in inst.init_obs:
        add(o)
    method = f"B9-{M}"
    rows = []
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        J = ctx.J[k]
        steps, status, traj = 0, None, []
        while True:
            km = np.argmax(cum + tie, 1)
            pis = np.argmax(J[km], 1)
            vals, cnt = np.unique(pis, return_counts=True)
            maj = int(vals[np.argmax(cnt)])
            if len(vals) == 1:
                status = "CERTIFIED"
                break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            blk = list(dict.fromkeys(int(x) for x in km[pis != maj]))[:TOP_M]
            margins = np.ones(len(blk))
            code = ctx.prop.codec.encode(*h.observable_state())
            a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk], ctx.pe[blk],
                                 margins, ctx.inc, ctx.prop, ctx.legal, rng)
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            add(obs)
            steps += 1
            if samples is not None and len(traj) < 200 and steps % 5 == 1:
                traj.append({"step": steps, "action": a, "agree_frac": float(cnt.max() / M), "n_distinct_pi": len(vals)})
        mask = lr.mask().numpy()
        cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
        km = np.argmax(cum + tie, 1)
        extra = {"ensemble_size": M, "agree_frac_end": float(np.mean(np.argmax(J[km], 1) == maj)),
                 "n_distinct_members": int(len(np.unique(km))), "lr_cert_status_at_stop": cert["status"].value,
                 "lr_r_bar_at_stop": cert["r_bar"], "theta_star_in_set": bool(mask[ctx.ti]),
                 "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED"), "wall_clock_s": time.perf_counter() - t0,
                 "rollouts": 0, "H": q.H, "n_policies": len(q.policies)}
        rows.append(R.score_row(ctx, k, method, status, maj, steps, status != "CERTIFIED", extra))
        if samples is not None and k < 3 and ctx.seed < 2:
            samples.append({"instance": ctx.seed, "problem_index": k, "method": method, "status": status,
                            "steps": steps, "trajectory_head": traj[:40], "J_true": J[ctx.ti].tolist(),
                            "policy": maj, "true_regret": rows[-1]["true_regret"],
                            "false_cert": rows[-1]["false_cert"]})
    return rows


# ------------------------------------------------------------------ B10: reward-free G-optimal coverage + plug-in cert
def run_b10(ctx, tmax, samples=None):
    inst, env, h, lr, led = _init(ctx)
    rng = np.random.default_rng([ctx.seed, ctx.noise, 1010])
    codec, nA = ctx.prop.codec, ctx.aspace.n
    d = ctx.vecs.shape[1]
    hist = {}
    for o in inst.init_obs:
        key = codec.encode(o.loads, o.engaged) * nA + o.action
        hist[key] = hist.get(key, 0) + 1
    lib_cache = {}

    def lib(kh):
        if kh not in lib_cache:
            if len(lib_cache) > 8:
                lib_cache.clear()
            v = ctx.vecs[kh]
            F = np.stack([fisher_rounds(v, codec, ctx.aspace, c, np.arange(nA), C_KNOWN, 2, 2)
                          for c in range(codec.size)]).reshape(codec.size * nA, d, d)
            lib_cache[kh] = F
        return lib_cache[kh]

    rows = []
    kh_prev, V = None, None
    for k, q in enumerate(ctx.problems):
        t0 = time.perf_counter()
        steps, status, traj = 0, None, []
        g_val = None
        while True:
            mask = lr.mask().numpy()
            cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
            if cert["status"].value in ("CERTIFIED", "MODEL_CONFLICT"):
                status = cert["status"].value
                break
            if not ctx.all_singletons and steps % 10 == 0:
                amb, _, _ = trichotomy(ctx.Reg[k], mask, ctx.gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= tmax:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            Flib = lib(kh)
            if kh != kh_prev:
                keys = np.array(list(hist.keys()))
                cnts = np.array([hist[x] for x in keys], float)
                V = np.eye(d) + np.einsum("k,kde->de", cnts, Flib[keys])
                kh_prev = kh
            code = codec.encode(*h.observable_state())
            if rng.random() < 0.05:
                a = int(rng.choice(ctx.legal))
            else:
                Fl = Flib.reshape(-1, d * d)
                best, best_val = None, np.inf
                for a_c in ctx.legal:
                    Vi = np.linalg.inv(V + Flib[code * nA + a_c])
                    val = float((Fl @ Vi.reshape(-1)).max())
                    if val < best_val - 1e-15:
                        best, best_val = int(a_c), val
                a, g_val = best, best_val
            led.record(h.step(a), "probe", q.pid)
            key = code * nA + a
            hist[key] = hist.get(key, 0) + 1
            V = V + Flib[key]
            steps += 1
            if samples is not None and len(traj) < 200 and steps % 5 == 1:
                traj.append({"step": steps, "action": a, "set_size": int(mask.sum()), "r_bar": cert["r_bar"],
                             "G_value": g_val})
        mask = lr.mask().numpy()
        cert = certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)
        extra = {"r_bar_end": cert["r_bar"], "set_size_end": int(mask.sum()), "theta_star_in_set": bool(mask[ctx.ti]),
                 "G_value_last": g_val, "zero_cost_cert": bool(steps == 0 and status == "CERTIFIED"),
                 "wall_clock_s": time.perf_counter() - t0, "rollouts": 0, "H": q.H, "n_policies": len(q.policies)}
        rows.append(R.score_row(ctx, k, "B10", status, cert["pi"], steps, status != "CERTIFIED", extra))
        if samples is not None and k < 3 and ctx.seed < 2:
            samples.append({"instance": ctx.seed, "problem_index": k, "method": "B10", "status": status,
                            "steps": steps, "trajectory_head": traj[:40], "policy": cert["pi"],
                            "true_regret": rows[-1]["true_regret"]})
    return rows


# ------------------------------------------------------------------ one instance
def run_instance(seed, noise, n_problems, tmax, cache_dir, methods):
    t0 = time.time()
    torch.set_num_threads(1)
    ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
    ctx = R.Ctx(seed, noise, n_problems, cache_dir, ncl)
    rows, samples, errors, timing = [], [], [], {}
    for m in methods:
        tm = time.time()
        try:
            if m == "JPC":
                r, _ = R.run_jpc(ctx, tmax)
            elif m == "B7":
                r = run_b7(ctx)
            elif m == "B8":
                r = run_b8(ctx, tmax, EPS / 2, "problem", "B8", samples=samples)
            elif m == "B8tau":
                r = run_b8(ctx, tmax, EPS, "problem", "B8tau")
            elif m == "B8fam":
                r = run_b8(ctx, tmax, EPS / 2, "stream", "B8fam")
            elif m.startswith("B9-"):
                r = run_b9(ctx, tmax, int(m.split("-")[1]), samples=samples)
            elif m == "B10":
                r = run_b10(ctx, tmax, samples=samples)
            else:
                raise ValueError(m)
            rows += r
        except Exception as e:  # noqa: BLE001 - record and continue with the other methods
            errors.append({"instance": seed, "noise": noise, "method": m, "error": repr(e), "tb": traceback.format_exc()})
        timing[m] = time.time() - tm
    return {"seed": seed, "noise": noise, "rows": rows, "samples": samples, "errors": errors, "timing": timing,
            "wall_s": time.time() - t0}


# ------------------------------------------------------------------ aggregation
def summarize(rows, methods, tau):
    out = {}
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs:
            continue
        steps = np.array([min(r["new_env_steps"], tau) for r in rs], float)
        cens = np.array([r["censored"] or r["new_env_steps"] > tau for r in rs])
        n_cert = int(sum(r["status"] == "CERTIFIED" for r in rs))
        n_false = int(sum(r["false_cert"] for r in rs))
        lo, hi = clopper_pearson(n_false, n_cert) if n_cert else (0.0, 1.0)
        reg = [r["true_regret"] for r in rs if r["true_regret"] is not None]
        statuses = sorted({r["status"] for r in rs})
        d = {"n": len(rs), "completion_rate": float(np.mean(~cens)),
             "rmst_new_env_steps": None if m == "B7" else rmst(steps, cens, tau),
             "mean_steps": float(steps.mean()), "median_steps": float(np.median(steps)),
             "total_steps": int(sum(r["new_env_steps"] for r in rs)),
             "n_certified": n_cert, "n_false_cert": n_false, "fcr": n_false / n_cert if n_cert else None,
             "fcr_cp_lower": lo, "fcr_cp_upper": hi,
             "true_regret_mean": float(np.mean(reg)) if reg else None,
             "true_regret_max": float(np.max(reg)) if reg else None,
             "frac_regret_gt_eps": float(np.mean([x > EPS for x in reg])) if reg else None,
             "zero_cost_cert_frac": float(np.mean([bool(r.get("zero_cost_cert")) for r in rs])),
             "theta_star_in_set_frac": float(np.mean([r["theta_star_in_set"] for r in rs if "theta_star_in_set" in r]))
             if any("theta_star_in_set" in r for r in rs) else None,
             "wall_clock_s_total": float(sum(r.get("wall_clock_s", 0.0) or 0.0 for r in rs)),
             "status_counts": {s: int(sum(r["status"] == s for r in rs)) for s in statuses}}
        if m == "B7":
            d["certifiable_rate"] = d["completion_rate"]
            d["note"] = "never acquires data: uncertified problems have infinite time-to-certificate (RMST n/a)"
        if m.startswith("B9"):
            d["fcr_exceeds_delta"] = bool(d["fcr"] is not None and d["fcr"] > DELTA)
            d["fcr_cp_lower_exceeds_delta"] = bool(lo > DELTA)
            d["lr_set_would_certify_at_stop"] = float(np.mean([r["lr_cert_status_at_stop"] == "CERTIFIED"
                                                               for r in rs if r["status"] == "CERTIFIED"])) if n_cert else None
        if m.startswith("B8"):
            d["identified_but_uncertified"] = int(sum(r["status"] == "IDENTIFIED_UNCERTIFIED" for r in rs))
        out[m] = d
    return out


def km_table(rows, methods, tau):
    out = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        if not rs or m == "B7":
            continue
        st = np.array([min(r["new_env_steps"], tau) for r in rs], float)
        ce = np.array([r["censored"] or r["new_env_steps"] > tau for r in rs])
        ut, sv = kaplan_meier(st, ce)
        out += [{"method": m, "steps": float(t), "surv": float(s)} for t, s in zip(ut, sv)]
    return out


def paired(rows, baselines):
    idx = {}
    for r in rows:
        idx.setdefault((r["instance"], r["noise_seed"], r["problem_index"]), {})[r["method"]] = r
    out = {}
    for b in baselines:
        pp = [(d["JPC"]["new_env_steps"], d[b]["new_env_steps"], key[0], key[1], d[b]["censored"])
              for key, d in sorted(idx.items()) if "JPC" in d and b in d]
        if not pp:
            continue
        a = np.array([(x[0], x[1]) for x in pp], float)
        ratio1 = (a[:, 0] + 1) / (a[:, 1] + 1)
        nz = np.max(a, 1) > 0
        rnz = a[nz, 0] / np.maximum(a[nz, 1], 1)
        per = {}
        for x in pp:
            per.setdefault((x[2], x[3]), []).append((x[0], x[1]))
        per = {k: np.array(v, float) for k, v in per.items()}
        stat = lambda arrs: float(np.log(max(np.concatenate(arrs)[:, 0].sum(), 1.0) /  # noqa: E731
                                         max(np.concatenate(arrs)[:, 1].sum(), 1.0)))
        est, lo, hi = cluster_bootstrap(per, stat, B=2000, seed=1)
        out[b] = {"n_problems": len(pp), "median_ratio_plus1": float(np.median(ratio1)),
                  "q25_q75_ratio_plus1": [float(np.quantile(ratio1, 0.25)), float(np.quantile(ratio1, 0.75))],
                  "ties_both_zero": int((~nz).sum()), "n_nontrivial": int(nz.sum()),
                  "nontrivial_median_ratio": float(np.median(rnz)) if nz.any() else None,
                  "frac_JPC_strictly_fewer_nontrivial": float(np.mean(a[nz, 0] < a[nz, 1])) if nz.any() else None,
                  "stream_total_ratio": float(math.exp(est)), "stream_total_ratio_ci95": [math.exp(lo), math.exp(hi)],
                  "JPC_total_steps": int(a[:, 0].sum()), "baseline_total_steps": int(a[:, 1].sum()),
                  "baseline_censored": int(sum(x[4] for x in pp)),
                  "note": "censored baseline problems enter at their cap (lower bound) -> ratio conservative for JPC"}
    return out


def jpc_repro_check(rows, ref_path):
    """JPC here must reproduce nl_reuse_kappa_high exactly (same seeds, same code path)."""
    if not ref_path.exists():
        return None
    ref = {}
    for line in ref_path.open():
        r = json.loads(line)
        if r.get("method") == "JPC" and r.get("ptype", 1) == 1:
            ref[(r["instance"], r["noise_seed"], r["problem_index"])] = (r["new_env_steps"], r["certified_policy"])
    mine = {(r["instance"], r["noise_seed"], r["problem_index"]): (r["new_env_steps"], r["certified_policy"])
            for r in rows if r["method"] == "JPC"}
    common = [k for k in mine if k in ref]
    same = sum(mine[k] == ref[k] for k in common)
    return {"n_compared": len(common), "n_identical": int(same)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--tmax", type=int, default=3000, help="per-problem cap for per-step methods (= nl_reuse_kappa_high)")
    ap.add_argument("--methods", default=DEFAULT_METHODS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(1)
    res_root = WS / "exp" / "results"
    out_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    cache_dir = WS / "exp" / "cache" / "jtables"
    (res_root / f"{TASK}.pid").write_text(str(os.getpid()))
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    n_inst = args.instances or (10 if args.mode == "pilot" else 32)
    n_prob = args.problems or (10 if args.mode == "pilot" else 15)
    seeds = list(range(n_inst)) if args.mode == "pilot" else list(range(10000, 10000 + n_inst))
    noises = [42] if args.mode == "pilot" else [42, 123, 456]
    methods = args.methods.split(",")
    log(f"start mode={args.mode} instances={n_inst} problems={n_prob} noises={noises} methods={methods} eps={EPS} "
        f"delta={DELTA} tmax={args.tmax} generator_hash={generator_hash()} (concurrent run, {args.workers} workers)")
    jobs = [(s, nz) for nz in noises for s in seeds]
    progress(res_root, 0, len(jobs), "instances", {"done": 0, "of": len(jobs)})
    t_run = time.time()
    results = []
    for res in Parallel(n_jobs=args.workers, return_as="generator_unordered", verbose=0)(
            delayed(run_instance)(s, nz, n_prob, args.tmax, cache_dir, methods) for s, nz in jobs):
        results.append(res)
        steps = {m: sum(r["new_env_steps"] for r in res["rows"] if r["method"] == m) for m in methods}
        log(f"instance {res['seed']}/{res['noise']} done in {res['wall_s']:.0f}s "
            f"timing={ {k: round(v) for k, v in res['timing'].items()} } total_steps={steps} errors={len(res['errors'])}")
        for e in res["errors"]:
            log(f"ERROR {e['method']} inst {e['instance']}: {e['error']}\n{e['tb']}")
        progress(res_root, len(results), len(jobs), "instances", {"done": len(results), "of": len(jobs)})
    rows = [r for res in results for r in res["rows"]]
    samples = [s for res in results for s in res["samples"]]
    errors = [e for res in results for e in res["errors"]]
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    (out_dir / "samples" / "trajectories.json").write_text(json.dumps(samples[:30], indent=1, default=float))
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=1))

    tau = args.tmax
    summ = summarize(rows, methods, tau)
    with open(out_dir / "km.csv", "w") as f:
        f.write("method,steps,surv\n")
        for r in km_table(rows, methods, tau):
            f.write(f"{r['method']},{r['steps']},{r['surv']}\n")
    pr = paired(rows, [m for m in methods if m not in ("JPC", "B7")])
    ref_dir = res_root / ("pilots" if args.mode == "pilot" else "full") / "nl_reuse_kappa_high"
    repro = jpc_repro_check(rows, ref_dir / "results.jsonl")
    # B7 vs JPC: true regret paired on the same problems
    b7 = summ.get("B7", {})
    # pilot gate (pre-registered): JPC steps / B8 steps median < 0.5 AND every baseline runs without crash
    b8 = pr.get("B8")
    gate_ratio = bool(b8 and b8["median_ratio_plus1"] < 0.5)
    gate_ratio_nontriv = bool(b8 and b8["nontrivial_median_ratio"] is not None and b8["nontrivial_median_ratio"] < 0.5)
    gate_total = bool(b8 and b8["stream_total_ratio_ci95"][1] < 0.5)
    no_crash = len(errors) == 0 and all(m in summ for m in methods)
    susp = []
    for b, d in pr.items():
        if d["stream_total_ratio"] < 0.2:
            susp.append(f"stream-total new steps JPC/{b} = {d['stream_total_ratio']:.3g} (savings > 5x)")
    for m in ("B8", "B8tau", "B8fam", "B10"):
        if m in summ and summ[m]["n_false_cert"] > 0:
            susp.append(f"{m} (sound by construction) has {summ[m]['n_false_cert']} false certificates")
    summary = {
        "task_id": TASK, "mode": args.mode, "env": R.ENV_NAME, "kappa_dyn": "high", "eps": EPS, "delta": DELTA,
        "n_instances": n_inst, "noise_seeds": noises, "problems_per_instance": n_prob, "instance_seeds": seeds,
        "generator_hash": generator_hash(), "tmax_step_methods": args.tmax, "rmst_horizon": tau,
        "methods": summ, "paired_vs_JPC": pr, "jpc_reproducibility_vs_nl_reuse_kappa_high": repro,
        "cand_c_decision_vs_full_ident": {
            "ratio_JPC_over_B8_median_plus1": b8["median_ratio_plus1"] if b8 else None,
            "ratio_JPC_over_B8_nontrivial_median": b8["nontrivial_median_ratio"] if b8 else None,
            "ratio_JPC_over_B8_stream_total": b8["stream_total_ratio"] if b8 else None,
            "ratio_JPC_over_B8_stream_total_ci95": b8["stream_total_ratio_ci95"] if b8 else None,
            "ratio_JPC_over_B8tau_stream_total": pr.get("B8tau", {}).get("stream_total_ratio"),
            "ratio_JPC_over_B8fam_stream_total": pr.get("B8fam", {}).get("stream_total_ratio")},
        "B7_robust_no_acquisition": {"certifiable_rate": b7.get("certifiable_rate"),
                                     "true_regret_mean": b7.get("true_regret_mean"),
                                     "frac_regret_gt_eps": b7.get("frac_regret_gt_eps"),
                                     "JPC_true_regret_mean": summ.get("JPC", {}).get("true_regret_mean")},
        "B9_fcr": {m: {"fcr": summ[m]["fcr"], "cp": [summ[m]["fcr_cp_lower"], summ[m]["fcr_cp_upper"]],
                       "exceeds_delta": summ[m]["fcr_exceeds_delta"]} for m in summ if m.startswith("B9")},
        "gate": {"literal_median_ratio_JPC_B8_lt_0.5": gate_ratio,
                 "nontrivial_median_ratio_JPC_B8_lt_0.5": gate_ratio_nontriv,
                 "stream_total_ratio_JPC_B8_ci_upper_lt_0.5": gate_total, "all_baselines_ran_without_crash": no_crash},
        "suspicious_flags": susp, "errors": len(errors), "wall_clock_s": time.time() - t_run,
        "notes": [
            f"Concurrent run ({args.workers} workers, other tasks on the same machine): wall-clock numbers are inflated.",
            "B8 target set = models farthest from theta_hat in J value (identification), not in decision regret; same "
            "DDA LP machinery, same LR set, same ledger reuse as JPC -> differences isolate the stopping target.",
            "B8 tau=eps/2 gives the same eps-guarantee as the certificate; B8tau (tau=eps) is the favourable variant.",
            "B8fam knows the whole stream's candidate policies in advance (favourable).",
            "B9 uses an online Poisson bootstrap of finite-class MLEs; it claims certification on member agreement.",
            "B10 acquisition is problem-agnostic (reward-free G-optimal coverage on linearised Fisher at theta_hat); "
            "its stopping uses the same sound LR-set certificate (plug-in of the collected data).",
            "Ratios use (steps+1) for the literal per-problem median; per-problem medians are tie-dominated because "
            "many problems are zero-cost for every set-based method; stream totals are the stream-level unit.",
        ],
    }
    go = (gate_ratio or gate_ratio_nontriv or gate_total) and no_crash
    summary["go_no_go"] = "GO" if (gate_ratio and no_crash) else ("GO_RESTATED" if go else "NO_GO")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"summary gate={summary['gate']} go={summary['go_no_go']} repro={repro} susp={susp}")
    mark_done(res_root, "success", f"{summary['go_no_go']}; JPC/B8 median(+1)={b8['median_ratio_plus1'] if b8 else None}, "
                                   f"stream total={b8['stream_total_ratio'] if b8 else None}")
    return summary


if __name__ == "__main__":
    main()
