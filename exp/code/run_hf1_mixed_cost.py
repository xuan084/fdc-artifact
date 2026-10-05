"""hf1_mixed_cost: HF1 -- in-class cost of the mixed numerator (methodology 4.2, hypotheses HF1).

Usage: run_hf1_mixed_cost.py --mode {pilot,full} [--workers 4] [--instances a-b] [--problems K] [--levels R0,R2]

Setting: dynamic E1-NL-S class G_1 (|Theta| = 13824), eta = 0 (no registered misspecification).
  R0 (primary)   on-grid truth (streams.offgrid level R0): the class is well specified, theta* is a grid point.
  R2 (secondary) continuous R1 truth, learner grid G_1, locally inflated certificate (2 max_alive eta_loc <= eps).
                 Here "theta* coverage" is coverage of the grid cell containing theta*; the class is only
                 approximately specified, so MODEL_CONFLICT is reported but is not strictly a false alarm.
Problem-stream protocol (methodology 3): n0 = 20 shared real rounds, then K Type-1 problems of stream s
(STREAMS[s] = (noise seed, order permutation)); every arm runs on its own fresh platform copy (CRN) and reuses its
OWN ledger across the stream. All arms use the pre-registered DDA acquisition, T_max = 3000 new steps / problem.
Arms:
  plugin      JPC on the in-class plug-in SeqLRSet (baseline; falsification scope {}).
  mixed       JPC on MixedLRSet {pool, reg(load bucket), ext}, weights 1/3 (cost <= log 3 nats).
  mixed_half  JPC on MixedLRSet {pool, ext}, weights 1/2 (the 1/2-1/2 fallback, cost <= log 2).
  For mixed arms: empty Theta_t -> MODEL_CONFLICT -> B2 fallback (residual + EB whole-trial LUCB, steps charged);
  replay check (dsswm.evidence.replay, same components) before every zero-cost certificate from the reused ledger.
Metrics: per-instance stream ratio (sum of charged new steps, arm / plugin; geometric mean + instance bootstrap CI),
theta* (cell) coverage at every problem end, MODEL_CONFLICT false-alarm rate (per problem and per stream, CP),
replay triggers / conflicts, FCR, completion, zero-cost rate, min_gap_vs_pool (must be >= -log 3 / -log 2).
Pilot: dev seeds 640-649 x stream 0 x 10 problems, R0 (+ R2 secondary on the same dev seeds).
Full: assert_locked(); eval 10000-10047 x 3 streams x 15 problems at R0; first 24 instances at R2.
CPU only (the gpu slot is a scheduling token); timings measured under concurrent runs.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
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

from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables, dda_choose  # noqa: E402
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.minimax_enum import (certify_minimax, group_ids, observable_signature, regret_matrix,  # noqa: E402
                                        trichotomy)
from dsswm.certify.minimax_infl import certify_minimax_infl, inflation_floor  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import load_bucket  # noqa: E402
from dsswm.evidence.replay import replay_check  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator, params_to_torch  # noqa: E402
from dsswm.models.grid_ladder import cell_index_of, ladder_grid  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import NL_DEFAULTS, generator_hash, nl_class_max_jtable  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

TASK = "hf1_mixed_cost"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
C_KNOWN, RHO_RET, NMAX, N0 = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"], NL_DEFAULTS["n0"]
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = 2_310_000
ARMS = ("plugin", "mixed", "mixed_half")
ARM_COMPONENTS = {"mixed": ("pool", "reg", "ext"), "mixed_half": ("pool", "ext")}
ARM_COST = {"plugin": 0.0, "mixed": math.log(3.0), "mixed_half": math.log(2.0)}
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"      # G_1 class tables (truth-free) + eta_loc (R2)
PLANNED_MIN = {"pilot": 10, "full": 50}
LEVEL_IDS = {"R0": 0, "R2": 2}
# R2 certificate: 'unc' (uncorrected G_1 minimax; default) or 'infl' (2 max_alive eta_loc <= eps). On dev seeds the
# inflated certificate hits the data-free floor (2 min eta_loc > eps) on every problem (hr1_r2_* pilots: 20/20
# refusals), so every arm would tie at T_max; the in-class cost at R2 is therefore measured with the uncorrected
# certificate and the floor-refusal rate is reported per problem (it is arm-independent).
R2_CERT = os.environ.get("HF1_R2_CERT", "unc")


# ------------------------------------------------------------------ scheduler protocol
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


def update_gpu_progress(status, start_iso, wall_min, snapshot, planned):
    p = WS / "exp" / "gpu_progress.json"
    lock = WS / "exp" / "gpu_progress.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        d = json.loads(p.read_text()) if p.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        key = "completed" if status == "success" else "failed"
        if TASK not in d.setdefault(key, []):
            d[key].append(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": planned, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


# ------------------------------------------------------------------ learner tables (shared per worker)
_W = {}


def worker_tables(aspace):
    if "ncl" not in _W:
        torch.set_num_threads(1)
        ncl = NLClass(ladder_grid(1), device="cpu")
        assert ncl.B == 13824, ncl.B
        _W["ncl"] = ncl
        _W["params"] = ncl.torch_params()
        prop = NLPropagator(2, 2, NMAX, aspace, C_KNOWN, RHO_RET, device="cpu")
        _W["prop"] = prop
        _W["LT"] = prop.tables(_W["params"])[0]
        _W["LT_np"] = _W["LT"].numpy()
        _W["py"], _W["pe"] = class_prob_tables(ncl.np_params, C_KNOWN, NMAX, aspace.nb)
        _W["inc"] = IncidenceIndex(prop.codec, aspace, NMAX)
        _W["gid"] = group_ids(observable_signature(_W["py"], _W["pe"], max_level=1))
        _W["all_single"] = len(np.unique(_W["gid"])) == ncl.B
    return _W


def vstar_of(truth):
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


class Ctx:
    def __init__(self, seed, level, stream, K):
        self.seed, self.level, self.stream = seed, level, stream
        self.noise = STREAMS[stream][0]
        inst = make_offgrid_instance(seed, level, stream=stream)
        self.aspace = inst.env.aspace
        W = worker_tables(self.aspace)
        self.ncl, self.prop, self.LT, self.LT_np = W["ncl"], W["prop"], W["LT"], W["LT_np"]
        self.py, self.pe, self.inc, self.gid, self.all_single = W["py"], W["pe"], W["inc"], W["gid"], W["all_single"]
        self.problems = inst.problems[:K]
        self.legal = np.arange(self.aspace.n)
        self.J, self.Reg, self.eta = [], [], []
        for q in self.problems:
            path = JT_CACHE / f"{q.pid}_raw.npy"
            J = nl_class_max_jtable(q, self.prop, W["params"], cache_path=str(path) if path.exists() else None)
            self.J.append(np.asarray(J, float))
            self.Reg.append(regret_matrix(self.J[-1]))
            if level == "R2":
                self.eta.append(np.load(JT_CACHE / f"{q.pid}_etaloc_raw.npy").astype(float) * q.utility.c_q)
            else:
                self.eta.append(None)
        # harness only (scoring): true J at theta* (exact propagator), theta* grid cell
        LTt, EYt = self.prop.tables(params_to_torch(inst.env.true_params(), device="cpu"))
        plans = [[self.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] for q in self.problems]
        self.Jt = [np.array([float(self.prop._run_plan(pl, es, LTt, EYt, q.utility.w, q.utility.w_ret, q.utility.c_q)[0])
                             for pl, es in pp]) for q, pp in zip(self.problems, plans)]
        self.star = int(cell_index_of(vstar_of(inst.truth)[None], self.ncl)[0])
        self.eta_star = [float(np.abs(Jt - J[self.star]).max()) for Jt, J in zip(self.Jt, self.J)]

    def fresh(self):
        return make_offgrid_instance(self.seed, self.level, stream=self.stream)


# ------------------------------------------------------------------ B2 fallback (MODEL_CONFLICT)
def trial_sampler(env, q, key, store=None):
    counters = {}

    def sampler(a, n):
        c = counters.get(a, 0)
        counters[a] = c + 1
        r = np.random.default_rng([*key, a, c])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        if store is not None:
            st = store.setdefault(a, ([], []))
            st[0].append(np.asarray(ys)); st[1].append(np.asarray(es))
        return q.utility.value_from_components(ys, es)
    return sampler


def run_b2(ctx, env, k, history, key):
    q = ctx.problems[k]
    keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
    prior = {}
    for a, kk in enumerate(keys):
        if kk in history:
            ys, es = history[kk]
            prior[a] = list(q.utility.value_from_components(ys, es))
    store = {}
    smp = trial_sampler(env, q, key, store)
    res = lucb(smp, len(q.policies), q.H, EPS, DELTA, float(q.meta["u_max"]), TMAX_TRIAL, kind="eb", prior=prior)
    for a, kk in enumerate(keys):
        if a in store:
            ys = np.concatenate(store[a][0]); es = np.concatenate(store[a][1])
            n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
            ys, es = ys[:n_used], es[:n_used]
            if kk in history:
                ys = np.concatenate([history[kk][0], ys]); es = np.concatenate([history[kk][1], es])
            history[kk] = (ys, es)
    return res


# ------------------------------------------------------------------ one arm over one problem stream
def certify(ctx, k, mask):
    if ctx.level == "R2" and R2_CERT == "infl":
        return certify_minimax_infl(ctx.Reg[k], mask, ctx.eta[k], EPS, TOP_M)
    return certify_minimax(ctx.Reg[k], mask, EPS, TOP_M)


def run_arm(ctx, arm, want_samples=False):
    inst = ctx.fresh()
    env = inst.env
    h = env.handle()
    mixed = arm != "plugin"
    if mixed:
        E = MixedLRSet(ctx.prop, ctx.LT, DELTA, components=ARM_COMPONENTS[arm], regime_fn=load_bucket, name=arm)
    else:
        E = SeqLRSet(ctx.prop, ctx.LT, DELTA)
    ledger = []
    alarm = None

    def feed(o, pid):
        nonlocal alarm
        if mixed:
            E.update(o, problem_id=pid)
            if alarm is None and E.conflict_round is not None:
                alarm = {"round": int(E.conflict_round), "problem_k": None, "source": "incremental",
                         "component_adv": {c: round(float(v), 3) for c, v in E.component_advantage().items()}}
        else:
            E.update(o)
        ledger.append((o, pid))

    for o in inst.init_obs:
        feed(o, None)
    if alarm is not None:
        alarm["problem_k"] = 0
    rng = np.random.default_rng([ctx.seed, ctx.noise, LEVEL_IDS[ctx.level], 101])   # same DDA stream for every arm
    b2_hist = {}
    rows, samples = [], []
    cum_new = 0
    for k, q in enumerate(ctx.problems):
        Reg, eta = ctx.Reg[k], ctx.eta[k]
        t0 = time.perf_counter()
        steps, status, traj, replay, refused = 0, None, [], None, False
        while True:
            lrat = E.log_ratio().numpy()
            mask = lrat < LOG_THR
            if not mask.any():
                status = "MODEL_CONFLICT"
                break
            if R2_CERT == "infl" and eta is not None and inflation_floor(eta, mask, EPS):
                status, refused = "NEED_DATA", True
                break
            cert = certify(ctx, k, mask)
            sv = cert["status"].value
            if mixed and sv == "CERTIFIED" and steps == 0 and k > 0:
                rc = replay_check(ctx.prop, ctx.LT, ledger, q.pid, DELTA, components=ARM_COMPONENTS[arm],
                                  regime_fn=load_bucket, env_handle=h)
                replay = {"conflict": bool(rc["conflict"]), "size": int(rc["size"]), "n_replayed": int(rc["n_replayed"]),
                          "mask_equal_incremental": bool(np.array_equal(rc["mask"].numpy(), mask))}
                if rc["conflict"]:
                    sv = "MODEL_CONFLICT"
                    if alarm is None:
                        alarm = {"round": len(ledger), "problem_k": k, "source": "replay", "component_adv": None}
            if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                status = sv
                break
            if not ctx.all_single and steps % 10 == 0:
                amb, _, _ = trichotomy(Reg, mask, ctx.gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            code = ctx.prop.codec.encode(*h.observable_state())
            blk = cert["blocking"] or certify_minimax(Reg, mask, 0.0, TOP_M)["blocking"]
            if not blk and eta is not None:
                blk = [int(i) for i in np.flatnonzero(mask)[np.argsort(-eta[mask])[:TOP_M]]]
            kh = E.mle()
            if not blk:
                a = int(rng.choice(ctx.legal)); mode = "random"
            else:
                margins = LOG_THR - lrat[blk]
                a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk],
                                     ctx.pe[blk], margins, ctx.inc, ctx.prop, ctx.legal, rng)
                mode = info["mode"]
            obs = h.step(a)
            steps += 1
            feed(obs, q.pid)
            if alarm is not None and alarm["problem_k"] is None:
                alarm["problem_k"] = k
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "star_alive": bool(mask[ctx.star]), "mode": mode})
        assert env.n_steps == len(ledger) == E.n_rounds, "every evidence round must be a real platform round"
        cum_new += steps
        Jtq = ctx.Jt[k]
        mask = (E.log_ratio().numpy() < LOG_THR)
        pi = cert["pi"] if status == "CERTIFIED" else None
        charged, fallback = None, None
        model_status = status
        if status == "MODEL_CONFLICT":
            res = run_b2(ctx, env, k, b2_hist, (ctx.seed, ctx.noise, LEVEL_IDS[ctx.level], k, 55, ARMS.index(arm)))
            fb_pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
            fallback = {"method": "B2", "status": res["status"], "steps": int(res["steps"]), "pulls": int(res["pulls"]),
                        "n_reused": int(res.get("n_reused", 0)), "pi": fb_pi,
                        "true_regret": float(Jtq.max() - Jtq[fb_pi]) if fb_pi is not None else None}
            if res["status"] == "CERTIFIED":
                status, pi, charged = "CERTIFIED", fb_pi, steps + int(res["steps"])
            else:
                pi, charged = None, max(steps + int(res["steps"]), TMAX_TRIAL)
        cert_ok = status == "CERTIFIED"
        if charged is None:
            charged = steps if cert_ok else max(steps, TMAX_STEP)
        tr = float(Jtq.max() - Jtq[pi]) if (cert_ok and pi is not None) else None
        row = {"task": TASK, "level": ctx.level, "instance": ctx.seed, "stream": ctx.stream, "noise_seed": ctx.noise,
               "eta_level": 0.0, "problem": q.pid, "k": k, "method": arm, "design": "DDA",
               "status": status, "model_status": model_status, "new_env_steps": int(charged),
               "steps_consumed_model": int(steps), "censored": not cert_ok, "certified_policy": pi, "true_regret": tr,
               "false_cert": bool(cert_ok and tr is not None and tr > EPS), "zero_cost": bool(cert_ok and charged == 0),
               "model_conflict": model_status == "MODEL_CONFLICT", "refused_floor": refused,
               "theta_cell_alive": bool(mask[ctx.star]) if mask.any() else False, "set_size": int(mask.sum()),
               "eta_q_theta_star_cell": ctx.eta_star[k],
               "eta_loc_alive_max": (float(eta[mask].max()) if (eta is not None and mask.any()) else None),
               "true_best": int(np.argmax(Jtq)), "true_top2_gap": float(np.sort(Jtq)[-1] - np.sort(Jtq)[-2]),
               "H": q.H, "n_policies": len(q.policies), "eps": EPS, "delta": DELTA,
               "fallback": fallback, "replay": replay, "alarm_active": bool(alarm is not None),
               "falsification_scope": (E.falsification_scope() if mixed else {"components": []}),
               "cum_new_steps_model": int(cum_new), "lr_n_rounds": int(E.n_rounds), "env_n_steps": int(env.n_steps),
               "rollouts": 0, "wall_clock_s": time.perf_counter() - t0}
        rows.append(row)
        if want_samples and k < 3:
            samples.append({**row, "J_true": np.round(Jtq, 4).tolist(),
                            "J_class_at_theta_star_cell": np.round(ctx.J[k][ctx.star], 4).tolist(),
                            "trajectory_head": traj})
    stream = {"level": ctx.level, "instance": ctx.seed, "stream": ctx.stream, "method": arm, "alarm": alarm,
              "stream_charged_steps": int(sum(r["new_env_steps"] for r in rows)),
              "stream_model_steps": int(cum_new),
              "min_gap_vs_pool": float(E.min_gap_vs_pool) if mixed else 0.0,
              "cost_bound": -ARM_COST[arm],
              "component_adv_final": ({c: round(float(v), 3) for c, v in E.component_advantage().items()}
                                      if mixed else None),
              "final_star_alive": bool((E.log_ratio().numpy() < LOG_THR)[ctx.star])}
    return rows, samples, stream


def run_unit(seed, level, stream, K, want_samples):
    t0 = time.time()
    out = {"seed": seed, "level": level, "stream": stream, "rows": [], "streams": [], "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, level, stream, K)
        out["ctx"] = {"level": level, "instance": seed, "stream": stream, "star": ctx.star, "r2_cert": R2_CERT if level == "R2" else None,
                      "infl_floor_refusal": ([bool(2.0 * float(e.min()) > EPS) for e in ctx.eta] if level == "R2" else None),
                      "eta_theta_star_cell_over_eps_max": max(ctx.eta_star) / EPS, "all_single": bool(ctx.all_single),
                      "ctx_s": time.time() - t0}
        for arm in ARMS:
            try:
                ta = time.time()
                rows, smp, st = run_arm(ctx, arm, want_samples=want_samples)
                st["wall_s"] = time.time() - ta
                out["rows"] += rows
                out["samples"] += smp
                out["streams"].append(st)
            except Exception as e:  # noqa: BLE001
                out["errors"].append({"where": arm, "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "unit", "error": repr(e), "tb": traceback.format_exc()[-1500:]})
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ analysis
def _cp(k, n):
    if n == 0:
        return None, None
    lo, hi = clopper_pearson(int(k), int(n))
    return float(lo), float(hi)


def _med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def ratio_stats(streams, arm, level, B=10000, seed=42):
    """Per (instance, stream) ratio of stream charged steps arm / plugin; geometric mean, cluster bootstrap over
    instances (B = 1e4). +1 smoothing guards zero-step streams (none expected: problem 0 always needs data)."""
    P = {(s["instance"], s["stream"]): s["stream_charged_steps"] for s in streams
         if s["method"] == "plugin" and s["level"] == level}
    A = {(s["instance"], s["stream"]): s["stream_charged_steps"] for s in streams
         if s["method"] == arm and s["level"] == level}
    keys = sorted(set(P) & set(A))
    if not keys:
        return None
    lr = {k: math.log((A[k] + 1.0) / (P[k] + 1.0)) for k in keys}
    inst = sorted({k[0] for k in keys})
    per_inst = np.array([np.mean([lr[k] for k in keys if k[0] == i]) for i in inst])
    rng = np.random.default_rng(seed)
    bs = per_inst[rng.integers(0, len(inst), size=(B, len(inst)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hi90 = np.percentile(bs, 95.0)
    tot_a, tot_p = sum(A[k] for k in keys), sum(P[k] for k in keys)
    return {"n_streams": len(keys), "n_instances": len(inst), "ratio_geo_mean": float(math.exp(per_inst.mean())),
            "ci95": [float(math.exp(lo)), float(math.exp(hi))], "ci_upper_one_sided_95": float(math.exp(hi90)),
            "ratio_of_totals": tot_a / tot_p if tot_p else None,
            "per_stream_ratio": {f"{k[0]}_s{k[1]}": round(math.exp(lr[k]), 4) for k in keys},
            "frac_streams_ratio_gt_1": float(np.mean([lr[k] > 0 for k in keys]))}


def summarize(rows, streams, levels, mode, wall):
    S = {"task": TASK, "mode": mode, "eps": EPS, "delta": DELTA, "levels": {}, "hf1": {},
         "n_rows": len(rows), "n_streams": len(streams), "wall_s": wall,
         "timing_note": "wall clock measured under concurrent runs (up to 4 tasks share 20 cores; 4 workers here)"}
    for lv in levels:
        L = {}
        for arm in ARMS:
            R = [r for r in rows if r["level"] == lv and r["method"] == arm]
            ST = [s for s in streams if s["level"] == lv and s["method"] == arm]
            if not R:
                continue
            n_cert = sum(r["status"] == "CERTIFIED" for r in R)
            fc = sum(r["false_cert"] for r in R)
            mc = sum(r["model_conflict"] for r in R)
            al = [s for s in ST if s["alarm"] is not None]
            cov = sum(r["theta_cell_alive"] for r in R)
            rp = [r["replay"] for r in R if r["replay"] is not None]
            c = {"n_problems": len(R), "n_streams": len(ST), "n_cert": n_cert, "completion": n_cert / len(R),
                 "n_false_cert": int(fc), "fcr": fc / n_cert if n_cert else None, "fcr_cp": _cp(fc, n_cert),
                 "status_counts": {k2: sum(r["status"] == k2 for r in R) for k2 in sorted({r["status"] for r in R})},
                 "model_status_counts": {k2: sum(r["model_status"] == k2 for r in R)
                                         for k2 in sorted({r["model_status"] for r in R})},
                 "stream_charged_steps_median": _med([s["stream_charged_steps"] for s in ST]),
                 "stream_charged_steps_mean": float(np.mean([s["stream_charged_steps"] for s in ST])),
                 "stream_model_steps_median": _med([s["stream_model_steps"] for s in ST]),
                 "zero_cost_rate": sum(r["zero_cost"] for r in R) / len(R),
                 "coverage_theta_star_cell": cov / len(R), "coverage_cp": _cp(cov, len(R)),
                 "coverage_stream_final": float(np.mean([s["final_star_alive"] for s in ST])) if ST else None,
                 "false_alarm_problems": int(mc), "false_alarm_rate_problem": mc / len(R),
                 "false_alarm_problem_cp": _cp(mc, len(R)),
                 "false_alarm_streams": len(al), "false_alarm_rate_stream": len(al) / len(ST) if ST else None,
                 "false_alarm_stream_cp": _cp(len(al), len(ST)),
                 "alarm_sources": {src: sum(a["alarm"]["source"] == src for a in al) for src in ("incremental", "replay")},
                 "replay_triggers": len(rp), "replay_conflicts": sum(x["conflict"] for x in rp),
                 "replay_mask_equal_incremental": (all(x["mask_equal_incremental"] for x in rp) if rp else None),
                 "n_fallback": sum(r["fallback"] is not None for r in R),
                 "min_gap_vs_pool_min": float(min(s["min_gap_vs_pool"] for s in ST)) if ST else None,
                 "cost_bound": -ARM_COST[arm],
                 "cost_bound_respected": all(s["min_gap_vs_pool"] >= -ARM_COST[arm] - 1e-9 for s in ST),
                 "refused_floor": sum(r["refused_floor"] for r in R)}
            if arm != "plugin":
                c["ratio_vs_plugin"] = ratio_stats(streams, arm, lv)
                c["component_adv_final_median"] = {
                    comp: _med([s["component_adv_final"][comp] for s in ST if s["component_adv_final"]])
                    for comp in ARM_COMPONENTS[arm]}
            L[arm] = c
        S["levels"][lv] = L
    r0 = S["levels"].get("R0", {})
    for arm in ("mixed", "mixed_half"):
        if arm not in r0:
            continue
        c = r0[arm]
        rs = c["ratio_vs_plugin"] or {}
        fa_hi = c["false_alarm_problem_cp"][1]
        S["hf1"][arm] = {
            "stream_ratio_mixed_over_plugin": rs.get("ratio_geo_mean"), "ratio_ci95": rs.get("ci95"),
            "ratio_point_le_1.15": (rs.get("ratio_geo_mean") or 9) <= 1.15,
            "ratio_ci_upper_le_1.25": (rs.get("ci95") or [9, 9])[1] <= 1.25,
            "coverage": c["coverage_theta_star_cell"], "coverage_ge_1_minus_delta": c["coverage_theta_star_cell"] >= 1 - DELTA,
            "false_alarm_rate_problem": c["false_alarm_rate_problem"], "false_alarm_problem_cp_upper": fa_hi,
            "false_alarm_rate_stream": c["false_alarm_rate_stream"],
            "false_alarm_le_delta": c["false_alarm_rate_problem"] <= DELTA,
            "replay_triggers": c["replay_triggers"], "replay_conflicts": c["replay_conflicts"],
            "pilot_pass": bool((rs.get("ratio_geo_mean") or 9) <= 1.25 and fa_hi is not None and fa_hi <= 0.10),
            "pilot_pass_rule": "mixed/plug-in stream ratio <= 1.25 AND false alarm CP upper <= 0.10 (per problem)"}
    return S


def plots(streams, out_dir, levels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(levels), figsize=(4.8 * len(levels), 4), squeeze=False)
    for ax, lv in zip(axs[0], levels):
        data, labels = [], []
        for arm in ("mixed", "mixed_half"):
            rs = ratio_stats(streams, arm, lv)
            if rs:
                data.append(np.log(list(rs["per_stream_ratio"].values())))
                labels.append(f"{arm}\nGM={rs['ratio_geo_mean']:.3f}")
        if data:
            ax.boxplot(data, tick_labels=labels)
            for i, d in enumerate(data):
                ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).uniform(-.08, .08, len(d)), d, s=12,
                           alpha=.7)
        ax.axhline(0, color="k", lw=.8)
        ax.axhline(math.log(1.15), color="tab:orange", ls="--", lw=.8, label="log 1.15")
        ax.axhline(math.log(1.25), color="tab:red", ls="--", lw=.8, label="log 1.25")
        ax.set_ylabel("log(stream steps / plug-in)")
        ax.set_title(f"HF1 in-class cost, {lv} (eta=0)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "stream_ratio_box.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ main
def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--r2-instances", type=int, default=None, help="number of leading instances run at R2")
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--levels", default="R0,R2")
    ap.add_argument("--streams", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--r2-cert", choices=["unc", "infl"], default="unc")
    ap.add_argument("--smoke", action="store_true", help="no DONE marker / gpu_progress update")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    global R2_CERT
    R2_CERT = os.environ["HF1_R2_CERT"] = args.r2_cert
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        rng_ = lock["eval_manifest"]["per_task_ranges"].get(TASK, [[10000, 10047]])[0]
        seeds = list(range(rng_[0], rng_[1] + 1))
    else:
        seeds = list(range(640, 650))
    if args.instances:
        seeds = parse_range(args.instances)
    if pilot:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    else:
        assert all(s >= 10000 for s in seeds)
    K = args.problems or (10 if pilot else 15)
    levels = args.levels.split(",")
    streams_ids = [int(x) for x in (args.streams or ("0" if pilot else "0,1,2")).split(",")]
    n_r2 = args.r2_instances or (len(seeds) if pilot else 24)
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_f, done_f, log_f = out_dir / "results.jsonl", out_dir / "units_done.jsonl", out_dir / "run.log"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_f, "a") as fh:
            fh.write(line + "\n")

    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    T0 = time.time()
    progress(0, 1, "start")
    status = "success"
    try:
        done = set()
        if done_f.exists():
            for ln in done_f.read_text().splitlines():
                u = json.loads(ln)
                done.add((u["seed"], u["level"], u["stream"]))
        units = [(s, lv, st) for lv in levels for s in (seeds if lv == "R0" else seeds[:n_r2]) for st in streams_ids]
        jobs = [u for u in units if u not in done]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}; R2 first {n_r2}) levels={levels} "
            f"streams={streams_ids} K={K} arms={ARMS} T_max={TMAX_STEP} workers={args.workers}; units todo={len(jobs)} "
            f"done={len(done)}; generator_hash={generator_hash()}")
        total = len(units)
        if jobs and not args.resummarize:
            sample_units = {(seeds[0], lv, streams_ids[0]) for lv in levels} | {(seeds[1], "R0", streams_ids[0])}
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(run_unit)(s, lv, st, K, (s, lv, st) in sample_units) for (s, lv, st) in jobs)
            n_err = 0
            for i, out in enumerate(gen):
                with open(res_f, "a") as fh:
                    for r in out["rows"]:
                        fh.write(json.dumps({"rec": "row", **r}, default=str) + "\n")
                    for st in out["streams"]:
                        fh.write(json.dumps({"rec": "stream", **st}, default=str) + "\n")
                    if "ctx" in out:
                        fh.write(json.dumps({"rec": "ctx", **out["ctx"]}) + "\n")
                for sm in out["samples"]:
                    (out_dir / "samples" / f"{out['level']}_i{out['seed']}_s{out['stream']}_{sm['method']}_k{sm['k']}.json"
                     ).write_text(json.dumps(sm, indent=1, default=str))
                for e in out["errors"]:
                    n_err += 1
                    log(f"ERROR unit {out['seed']}/{out['level']}/{out['stream']} {e['where']}: {e['error']}\n{e['tb']}")
                if not out["errors"]:
                    with open(done_f, "a") as fh:
                        fh.write(json.dumps({"seed": out["seed"], "level": out["level"], "stream": out["stream"],
                                             "wall_s": out["wall_s"]}) + "\n")
                stp = {s["method"]: s["stream_charged_steps"] for s in out["streams"]}
                al = {s["method"]: (s["alarm"]["round"] if s["alarm"] else None) for s in out["streams"]}
                log(f"unit {i + 1}/{len(jobs)} {out['seed']} {out['level']} s{out['stream']} {out['wall_s']:.0f}s "
                    f"steps={stp} alarm={al} fc={[sum(r['false_cert'] for r in out['rows'] if r['method'] == a) for a in ARMS]}")
                progress(len(done) + i + 1, total, "units", {"errors": n_err})
        rows, streams, ctxs = [], [], []
        for ln in res_f.read_text().splitlines():
            d = json.loads(ln)
            rec = d.pop("rec")
            if rec == "row":
                rows.append(d)
            elif rec == "stream":
                streams.append(d)
            elif rec == "ctx":
                ctxs.append(d)
        summ = summarize(rows, streams, levels, args.mode, time.time() - T0)
        fl = [x for c in ctxs if c.get("infl_floor_refusal") for x in c["infl_floor_refusal"]]
        summ["r2_cert"] = R2_CERT
        summ["r2_infl_floor_refusal_rate"] = (sum(fl) / len(fl)) if fl else None
        summ["r2_note"] = ("R2 rows use the uncorrected G_1 certificate (class only approximately specified: "
                           "MODEL_CONFLICT at R2 is not a false alarm in the strict sense; coverage = theta* grid cell). "
                           "The inflated certificate is refused data-free on r2_infl_floor_refusal_rate of the problems.")
        crashed = sum(1 for ln in log_f.read_text().splitlines() if "ERROR unit" in ln) if log_f.exists() else 0
        summ.update({"seeds": [seeds[0], seeds[-1]], "n_r2_instances": n_r2, "K": K, "streams": streams_ids,
                     "T_max_step": TMAX_STEP, "T_max_trial": TMAX_TRIAL, "generator_hash": generator_hash(),
                     "eval_seeds_touched": not pilot, "n_unit_errors": crashed})
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(streams, out_dir, levels)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; hf1={json.dumps(summ['hf1'], default=str)}; wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"hf1": summ["hf1"], "n_unit_errors": crashed}, default=str)[:1500]
        if crashed:
            status = "success" if rows else "failed"
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    if args.smoke:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
        return 0 if status == "success" else 1
    mark_done(status, result_summary)
    update_gpu_progress(status, start_iso, wall / 60,
                        {"env": "E1-NL-S dynamic class G_1 (|Theta|=13824), eta=0; R0 primary + R2 (inflated) secondary",
                         "mode": args.mode, "instances": len(seeds), "r2_instances": n_r2, "K": K,
                         "streams": streams_ids, "arms": list(ARMS), "workers": args.workers, "gpu_count": 0,
                         "note": "CPU only, concurrent with other tasks"},
                        PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
