"""hr3_r1_uncorrected: HR3 off-grid diagnosis at R1 (+ HF3 discretization arm). Methodology 2.2 / 2.4 / 5 / 6.

Level R1: continuous truth (streams.offgrid, R1 box), learner grid G_1 (DEFAULT_NL_S_GRID, |Theta| = 13824),
NO inflation (unsafe arm, diagnostic only). Two arms on CRN-paired platform copies (same instance, stream, noise):
  JPC        plug-in numerator SeqLRSet (falsification scope {}), uncorrected minimax certificate, DDA acquisition.
  JPC-mixed  MixedLRSet {pool, reg(load bucket), ext} (1/3 each) + replay check before every zero-cost certificate
             from the reused ledger. Theta_t empty -> MODEL_CONFLICT (alarm).
For both arms an empty set triggers the B2 fallback (residual + empirical-Bernstein whole-trial LUCB) on that problem;
fallback certificates are reported separately and are NOT model certificates (HR3 FCR is over model certificates).
Each arm re-uses its OWN ledger over the problem stream; n0 = 20 shared initial rounds; T_max = 3000 new steps/problem.

Per problem (harness only, never seen by the learner):
  eta_dec      = max_pi |J_true - J_{g°}|, g° = KL projection of theta* onto G_1 over the arm's realised design
                 (exact per-round KL, same code path as g0_resolution_gate) at that arm's stopping time;
  eta_dec_nearest = same with the coordinate-wise nearest grid point (= the R0 snap);
  true regret of the certified policy at the continuous theta* (exact propagator), false_cert = regret > eps.
Readouts:
  HR3  FCR (model certificates of JPC) by eta_dec/eps bin {[0,.25),[.25,.5),[.5,1),[1,2),[2,inf)}; Jonckheere trend
       test (one-sided, increasing; permutation of bin labels over certified problems + normal approximation);
       CP bounds in the pooled bin eta_dec >= eps/2 (pass: CP lower > delta).
  HF3  detection rate of pure discretization error by the mixed numerator: share of JPC false certificates whose
       CRN-paired JPC-mixed problem was turned into MODEL_CONFLICT at or before that problem (pass: <= 2 delta,
       expected negative / ~0 power); plus per-problem and per-stream alarm rates of JPC-mixed.
Pilot: dev instances 680-689 x 10 problems x stream 0 (seed 42). Full: lock range (10000-10047) x 3 streams x 15
problems (requires assert_locked()). Incremental results.jsonl + units_done.jsonl (unit = (instance, stream)).
CPU only (gpu slot is a scheduling token); timings measured under concurrent runs.

Usage: run_hr3_r1_uncorrected.py --mode {pilot,full} [--workers 4] [--instances a-b] [--problems K] [--streams 0,1,2]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

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

from dsswm.acquire.nl_kl_dda import (IncidenceIndex, class_prob_tables, dda_choose, kl_features,  # noqa: E402
                                     single_prob_tables)
from dsswm.baselines.whole_trial_bai import lucb  # noqa: E402
from dsswm.certify.eta_loc import build_plans, j_values  # noqa: E402
from dsswm.certify.minimax_enum import (certify_minimax, group_ids, observable_signature, regret_matrix,  # noqa: E402
                                        trichotomy)
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import load_bucket  # noqa: E402
from dsswm.evidence.replay import replay_check  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import cell_index_of, class_vectors, ladder_grid  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import NL_DEFAULTS, generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

TASK = "hr3_r1_uncorrected"
LEVEL = "R1"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
TMAX_TRIAL = 2_310_000
N0 = NL_DEFAULTS["n0"]
C_KNOWN, RHO_RET, NMAX = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"]
ARMS = ("JPC", "JPC-mixed")
BIN_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, math.inf)          # in units of eps
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"
PILOT_SEEDS = list(range(680, 690))
PLANNED_MIN = {"pilot": 10, "full": 45}


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


# ------------------------------------------------------------------ helpers
def vstar_of(truth: dict) -> np.ndarray:
    return np.concatenate([truth["alpha"], truth["gamma"], truth["tauL"], truth["beta"], truth["tauR"],
                           [truth["psi"]], [truth["lam"]]]).astype(float)


def U1(q):
    return type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": 1.0})()


def _u_max(q, c):
    return c * (2 * float(q.utility.w.sum()) + q.utility.w_ret * 4)


def precompute_tables(seeds, streams, n_prob, log):
    """Truth-free G_1 J tables (cache shared with every other round-1 task); compute only the missing ones."""
    need = []
    for s in seeds:
        for st in streams:
            inst = make_offgrid_instance(s, LEVEL, stream=st)
            qs = inst.problems[:n_prob]
            for q in qs:
                if not (JT_CACHE / f"{q.pid}_raw.npy").exists() and q.pid not in {x.pid for x in need}:
                    need.append(q)
    if not need:
        log("G_1 J tables: all cached")
        return 0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inst = make_offgrid_instance(seeds[0], LEVEL)
    prop = NLPropagator(2, 2, NMAX, inst.env.aspace, C_KNOWN, RHO_RET, device=dev)
    ncl = NLClass(ladder_grid(1), device=dev)
    params = ncl.torch_params()
    JT_CACHE.mkdir(parents=True, exist_ok=True)
    with open(JT_CACHE / ".hr1_chunk.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for q in need:
            pj = JT_CACHE / f"{q.pid}_raw.npy"
            if pj.exists():
                continue
            raw = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H, U1(q))
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            with open(tmp, "wb") as fh:
                np.save(fh, raw)
            os.replace(tmp, pj)
        fcntl.flock(lk, fcntl.LOCK_UN)
    log(f"G_1 J tables: computed {len(need)} on {dev}")
    del prop, ncl, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return len(need)


_W = {}


def worker_class():
    if "ncl" not in _W:
        torch.set_num_threads(1)
        ncl = NLClass(ladder_grid(1), device="cpu")
        _W["ncl"] = ncl
        _W["V"] = class_vectors(ncl)
        _W["params"] = ncl.torch_params()
    return _W


class Ctx:
    """Learner tables for one (instance, stream) on G_1 + harness truth for scoring only."""

    def __init__(self, seed, stream, n_prob):
        W = worker_class()
        self.seed, self.stream = seed, stream
        self.noise = STREAMS[stream][0]
        inst = make_offgrid_instance(seed, LEVEL, stream=stream)
        self.truth = inst.truth
        self.problems = inst.problems[:n_prob]
        self.aspace = inst.env.aspace
        self.ncl, self.V = W["ncl"], W["V"]
        self.prop = NLPropagator(2, 2, NMAX, self.aspace, C_KNOWN, RHO_RET, device="cpu")
        self.LT = self.prop.tables(W["params"])[0]
        self.LT_np = self.LT.numpy()
        self.py, self.pe = class_prob_tables(self.ncl.np_params, C_KNOWN, NMAX, self.aspace.nb)
        self.inc = IncidenceIndex(self.prop.codec, self.aspace, NMAX)
        self.legal = np.arange(self.aspace.n)
        self.gid = group_ids(observable_signature(self.py, self.pe, max_level=1))
        self.all_single = len(np.unique(self.gid)) == self.ncl.B
        vst = vstar_of(inst.truth)
        self.star = int(cell_index_of(vst[None], self.ncl)[0])                 # nearest grid point (harness)
        tp = inst.env.true_params()
        tpy, tpe = single_prob_tables(tp, C_KNOWN, NMAX, self.aspace.nb)
        self.Fkl = kl_features(tpy, tpe, self.py, self.pe)                     # (nf, B), harness only
        self.J, self.Reg, self.Jt, self.cq = [], [], [], []
        for q in self.problems:
            raw = np.load(JT_CACHE / f"{q.pid}_raw.npy").astype(float)
            c = 1.0 / float(raw.max())
            J = raw * c
            self.J.append(J)
            self.Reg.append(regret_matrix(J))
            plans = build_plans(self.prop, q)
            self.Jt.append(j_values(self.prop, plans, vst[None], q.utility.w, q.utility.w_ret)[0] * c)
            self.cq.append(c)

    def fresh(self):
        return make_offgrid_instance(self.seed, LEVEL, stream=self.stream)

    def gcirc(self, all_obs):
        pairs = {}
        for o in all_obs:
            key = (self.prop.codec.encode(o.loads, o.engaged), int(o.action))
            pairs[key] = pairs.get(key, 0) + 1
        cnt = np.zeros(self.inc.nf)
        for (code, a), n in pairs.items():
            cnt += n * self.inc.S[code * self.inc.nA + a]
        kl = cnt @ self.Fkl
        gc = int(np.argmin(kl))
        return gc, float(kl[gc]), float(kl[self.star])


def run_b2(ctx, env, k, history):
    """B2 fallback on problem k (residual + EB whole-trial LUCB, re-using this arm's stored whole trials)."""
    q, c = ctx.problems[k], ctx.cq[k]
    keys = [(p.name, tuple(int(x) for x in q.loads0), tuple(int(x) for x in q.engaged0), q.H) for p in q.policies]
    prior = {}
    for a, key in enumerate(keys):
        if key in history:
            ys, es = history[key]
            prior[a] = list(c * (ys.astype(float) @ q.utility.w + q.utility.w_ret * es.astype(float)))
    counters, raw_store = {}, {}

    def sampler(a, n):
        cc = counters.get(a, 0)
        counters[a] = cc + 1
        r = np.random.default_rng([ctx.seed, ctx.noise, k, a, cc, 55])
        ys, es = env.simulate_batch(q.policies[a], q.loads0, q.engaged0, q.H, n, r)
        st = raw_store.setdefault(a, ([], []))
        st[0].append(ys.astype(np.int8))
        st[1].append(es.astype(np.int8))
        return c * (ys @ q.utility.w + q.utility.w_ret * es)

    res = lucb(sampler, len(q.policies), q.H, EPS, DELTA, _u_max(q, c), TMAX_TRIAL, kind="eb", prior=prior)
    for a, key in enumerate(keys):
        if a in raw_store:
            ys = np.concatenate(raw_store[a][0])
            es = np.concatenate(raw_store[a][1])
            n_used = int(res["n_per_arm"][a]) - len(prior.get(a, []))
            ys, es = ys[:n_used], es[:n_used]
            if key in history:
                ys = np.concatenate([history[key][0], ys])
                es = np.concatenate([history[key][1], es])
            history[key] = (ys, es)
    return res


# ------------------------------------------------------------------ one arm over the problem stream
def run_arm(ctx: Ctx, arm: str, want_samples=False):
    inst = ctx.fresh()
    env = inst.env
    h = env.handle()
    mixed = arm == "JPC-mixed"
    E = MixedLRSet(ctx.prop, ctx.LT, DELTA, regime_fn=load_bucket, name="main") if mixed else \
        SeqLRSet(ctx.prop, ctx.LT, DELTA)
    ledger, all_obs = [], []
    alarm = None

    def feed(o, pid):
        nonlocal alarm
        if mixed:
            E.update(o, problem_id=pid)
        else:
            E.update(o)
        ledger.append((o, pid))
        all_obs.append(o)
        if alarm is None and E.size() == 0:
            alarm = {"round": int(E.n_rounds), "new_steps": int(max(0, E.n_rounds - N0)), "problem_k": None}

    for o in inst.init_obs:
        feed(o, None)
    if alarm is not None:
        alarm["problem_k"] = 0
    rng = np.random.default_rng([ctx.seed, ctx.noise, 101])            # same DDA rng stream for both arms (CRN)
    b2_hist, rows, samples = {}, [], []
    cum_new = 0
    for k, q in enumerate(ctx.problems):
        Reg, J, Jt = ctx.Reg[k], ctx.J[k], ctx.Jt[k]
        t0 = time.perf_counter()
        steps, status, traj, replay, n_explore = 0, None, [], None, 0
        while True:
            lrat = E.log_ratio().numpy()
            mask = lrat < LOG_THR
            if not mask.any():
                status = "MODEL_CONFLICT"
                break
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if mixed and sv == "CERTIFIED" and steps == 0 and k > 0:
                rc = replay_check(ctx.prop, ctx.LT, ledger, q.pid, DELTA, regime_fn=load_bucket, env_handle=h)
                replay = {"conflict": bool(rc["conflict"]), "size": int(rc["size"]),
                          "n_replayed": int(rc["n_replayed"]),
                          "mask_equal_incremental": bool(np.array_equal(rc["mask"].numpy(), mask))}
                if rc["conflict"]:
                    sv = "MODEL_CONFLICT"
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
            kh = E.mle()
            if not blk:
                a, mode = int(rng.choice(ctx.legal)), "random"
            else:
                margins = LOG_THR - lrat[blk]
                a, info = dda_choose(code, ctx.py[kh:kh + 1], ctx.pe[kh:kh + 1], ctx.LT_np[kh], ctx.py[blk],
                                     ctx.pe[blk], margins, ctx.inc, ctx.prop, ctx.legal, rng)
                mode = info["mode"]
            n_explore += int(mode != "dda")
            obs = h.step(a)
            steps += 1
            feed(obs, q.pid)
            if alarm is not None and alarm["problem_k"] is None:
                alarm["problem_k"] = k
            if want_samples and len(traj) < 30:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "nearest_alive": bool(mask[ctx.star]), "mode": mode})
        assert env.n_steps == len(all_obs) == E.n_rounds, "every evidence round must be a real platform round"
        cum_new += steps
        mask = E.log_ratio().numpy() < LOG_THR
        model_status = status
        model_pi = cert["pi"] if status == "CERTIFIED" else None
        model_tr = float(Jt.max() - Jt[model_pi]) if model_pi is not None else None
        gc, kl_gc, kl_near = ctx.gcirc(all_obs)
        eta_dec = float(np.abs(Jt - J[gc]).max())
        eta_near = float(np.abs(Jt - J[ctx.star]).max())
        pi, charged, fallback = model_pi, None, None
        if status == "MODEL_CONFLICT":
            res = run_b2(ctx, env, k, b2_hist)
            fb_pi = int(res["pi"]) if res["status"] == "CERTIFIED" else None
            fallback = {"method": "B2", "status": res["status"], "steps": int(res["steps"]), "pulls": int(res["pulls"]),
                        "n_reused": int(res["n_reused"]), "pi": fb_pi,
                        "true_regret": float(Jt.max() - Jt[fb_pi]) if fb_pi is not None else None}
            if res["status"] == "CERTIFIED":
                status, pi, charged = "CERTIFIED", fb_pi, steps + int(res["steps"])
            else:
                pi, charged = None, max(steps + int(res["steps"]), TMAX_TRIAL)
        cert_ok = status == "CERTIFIED"
        if charged is None:
            charged = steps if cert_ok else max(steps, TMAX_STEP)
        tr = float(Jt.max() - Jt[pi]) if (cert_ok and pi is not None) else None
        row = {"task": TASK, "level": LEVEL, "f": 1, "instance": ctx.seed, "stream": ctx.stream, "noise_seed": ctx.noise,
               "problem": q.pid, "gen_index": int(q.meta.get("gen_index", -1)), "k": k, "method": arm,
               "design": "DDA", "status": status, "model_status": model_status,
               "new_env_steps": int(charged), "steps_consumed_model": int(steps), "censored": not cert_ok,
               "certified_policy": pi, "true_regret": tr, "false_cert": bool(cert_ok and tr is not None and tr > EPS),
               "model_cert": model_status == "CERTIFIED", "model_pi": model_pi, "model_true_regret": model_tr,
               "model_false_cert": bool(model_tr is not None and model_tr > EPS),
               "zero_cost": bool(cert_ok and charged == 0),
               "eta_dec": eta_dec, "eta_dec_over_eps": eta_dec / EPS, "eta_dec_nearest": eta_near,
               "gcirc": gc, "gcirc_eq_nearest": bool(gc == ctx.star), "kl_gcirc": kl_gc, "kl_nearest": kl_near,
               "gcirc_argmax_true_regret": float(Jt.max() - Jt[int(np.argmax(J[gc]))]),
               "nearest_argmax_true_regret": float(Jt.max() - Jt[int(np.argmax(J[ctx.star]))]),
               "n_policies_regret_gt_eps": int(np.sum(Jt.max() - Jt > EPS)),
               "eta_loc": None, "theta_cell_alive": bool(mask[ctx.star]), "gcirc_cell_alive": bool(mask[gc]),
               "set_size": int(mask.sum()), "true_best": int(np.argmax(Jt)),
               "true_top2_gap": float(np.sort(Jt)[-1] - np.sort(Jt)[-2]) if len(Jt) > 1 else None,
               "H": q.H, "n_policies": len(q.policies), "eps": EPS, "delta": DELTA,
               "fallback": fallback, "replay": replay, "alarm_active": bool(alarm is not None),
               "falsification_scope": (E.falsification_scope() if mixed else {"components": []}),
               "cum_new_steps_model": int(cum_new), "explore_steps": n_explore,
               "lr_n_rounds": int(E.n_rounds), "env_n_steps": int(env.n_steps), "rollouts": 0, "dual_calls": 0,
               "wall_clock_s": time.perf_counter() - t0}
        rows.append(row)
        if want_samples and (k < 2 or row["model_false_cert"]) and len(samples) < 4:
            samples.append({**row, "J_true": np.round(Jt, 4).tolist(), "J_gcirc": np.round(J[gc], 4).tolist(),
                            "J_nearest": np.round(J[ctx.star], 4).tolist(), "trajectory_head": traj})
    stream = {"instance": ctx.seed, "stream": ctx.stream, "method": arm, "alarm": alarm,
              "stream_charged_steps": int(sum(r["new_env_steps"] for r in rows)), "stream_model_steps": int(cum_new),
              "min_gap_vs_pool": float(E.min_gap_vs_pool) if mixed else None,
              "component_adv_final": ({c: round(float(v), 3) for c, v in E.component_advantage().items()}
                                      if mixed else None),
              "truth_sha256": ctx.truth.get("sha256")}
    return rows, samples, stream


def run_unit(seed, stream, n_prob, want_samples):
    torch.set_num_threads(1)
    t0 = time.time()
    out = {"seed": seed, "stream": stream, "rows": [], "streams": [], "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, stream, n_prob)
    except Exception as e:  # noqa: BLE001
        out["errors"].append({"where": "ctx", "error": repr(e), "tb": traceback.format_exc()})
        out["wall_s"] = time.time() - t0
        return out
    out["ctx_s"] = time.time() - t0
    for arm in ARMS:
        try:
            r, s, st = run_arm(ctx, arm, want_samples)
            out["rows"] += r
            out["samples"] += s
            out["streams"].append(st)
        except Exception as e:  # noqa: BLE001
            out["errors"].append({"where": arm, "error": repr(e), "tb": traceback.format_exc()})
    out["wall_s"] = time.time() - t0
    return out


# ------------------------------------------------------------------ statistics
def _bin(x):
    for i in range(len(BIN_EDGES) - 1):
        if BIN_EDGES[i] <= x < BIN_EDGES[i + 1]:
            return i
    return len(BIN_EDGES) - 2


def bin_label(i):
    lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
    return f"[{lo:g},{hi:g})" if math.isfinite(hi) else f"[{lo:g},inf)"


def jt_binary(bins, y, nb):
    """Jonckheere-Terpstra statistic for binary y over ordered groups (ties count 1/2)."""
    n1 = np.array([np.sum((bins == g) & (y == 1)) for g in range(nb)], float)
    n0 = np.array([np.sum((bins == g) & (y == 0)) for g in range(nb)], float)
    s = 0.0
    for i in range(nb):
        for j in range(i + 1, nb):
            ni, nj = n0[i] + n1[i], n0[j] + n1[j]
            # pairs (a in i, b in j): a<b iff a=0,b=1 ; a==b counts 1/2
            s += n0[i] * n1[j] + 0.5 * (n0[i] * n0[j] + n1[i] * n1[j])
            _ = ni, nj
    return s


def jonckheere(bins, y, nb, n_perm=20000, seed=42):
    bins, y = np.asarray(bins, int), np.asarray(y, int)
    n = len(y)
    if n < 2 or len(np.unique(bins)) < 2:
        return {"n": int(n), "jt": None, "p_perm": None, "p_normal": None, "note": "fewer than 2 non-empty bins"}
    obs = jt_binary(bins, y, nb)
    if y.min() == y.max():
        return {"n": int(n), "jt": obs, "p_perm": 1.0, "p_normal": 1.0, "note": "outcome constant (no false certs or all)"}
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        ge += jt_binary(bins, rng.permutation(y), nb) >= obs - 1e-12
    p_perm = (ge + 1) / (n_perm + 1)
    sizes = np.array([np.sum(bins == g) for g in range(nb)], float)
    mean = (n ** 2 - np.sum(sizes ** 2)) / 4.0
    var = (n ** 2 * (2 * n + 3) - np.sum(sizes ** 2 * (2 * sizes + 3))) / 72.0     # ties-uncorrected
    from scipy.stats import norm
    z = (obs - mean) / math.sqrt(var) if var > 0 else 0.0
    return {"n": int(n), "jt": obs, "jt_mean_null": mean, "z": z, "p_perm": float(p_perm),
            "p_normal": float(1 - norm.cdf(z)), "n_perm": n_perm,
            "note": "units = certified problems (permutation ignores (instance, stream) clustering; pilot readout)"}


def fcr_table(R, cert_key, fc_key, eta_key="eta_dec_over_eps"):
    nb = len(BIN_EDGES) - 1
    tab = []
    for g in range(nb):
        rr = [r for r in R if _bin(r[eta_key]) == g]
        cc = [r for r in rr if r[cert_key]]
        kf = sum(bool(r[fc_key]) for r in cc)
        lo, hi = clopper_pearson(kf, len(cc))
        tab.append({"bin": bin_label(g), "n_problems": len(rr), "n_cert": len(cc), "n_false_cert": kf,
                    "fcr": kf / len(cc) if cc else None, "cp_lower": lo, "cp_upper": hi,
                    "cert_rate": len(cc) / len(rr) if rr else None,
                    "eta_dec_over_eps_median": float(np.median([r[eta_key] for r in rr])) if rr else None})
    return tab


def summarize(rows, streams, mode, wall):
    out = {"task": TASK, "mode": mode, "level": LEVEL, "grid": "G_1 (|Theta|=13824)", "eps": EPS, "delta": DELTA,
           "bins_eta_dec_over_eps": [bin_label(i) for i in range(len(BIN_EDGES) - 1)], "arms": {}}
    for arm in ARMS:
        R = [r for r in rows if r["method"] == arm]
        if not R:
            continue
        mc = [r for r in R if r["model_cert"]]
        kmf = sum(r["model_false_cert"] for r in mc)
        allc = [r for r in R if r["status"] == "CERTIFIED"]
        kf = sum(r["false_cert"] for r in allc)
        ed = np.array([r["eta_dec_over_eps"] for r in R])
        hi_bin = [r for r in mc if r["eta_dec_over_eps"] >= 0.5]
        k_hi = sum(r["model_false_cert"] for r in hi_bin)
        lo_hi, up_hi = clopper_pearson(k_hi, len(hi_bin))
        lo_bin = [r for r in mc if r["eta_dec_over_eps"] < 0.5]
        k_lo = sum(r["model_false_cert"] for r in lo_bin)
        jt = jonckheere([_bin(r["eta_dec_over_eps"]) for r in mc], [int(r["model_false_cert"]) for r in mc],
                        len(BIN_EDGES) - 1)
        st = [s for s in streams if s["method"] == arm]
        status_counts = {}
        for r in R:
            status_counts[r["model_status"]] = status_counts.get(r["model_status"], 0) + 1
        out["arms"][arm] = {
            "n_problems": len(R), "n_streams": len(st), "model_status_counts": status_counts,
            "n_model_cert": len(mc), "n_model_false_cert": kmf, "model_fcr": kmf / len(mc) if mc else None,
            "model_fcr_cp": clopper_pearson(kmf, len(mc)),
            "n_cert_incl_fallback": len(allc), "n_false_cert_incl_fallback": kf,
            "completion": len(allc) / len(R),
            "fcr_by_eta_dec_bin": fcr_table(R, "model_cert", "model_false_cert"),
            "fcr_by_eta_dec_nearest_bin": fcr_table([{**r, "eta_near_over_eps": r["eta_dec_nearest"] / EPS} for r in R],
                                                    "model_cert", "model_false_cert", "eta_near_over_eps"),
            "jonckheere": jt,
            "pooled_eta_dec_ge_half_eps": {"n_cert": len(hi_bin), "n_false_cert": k_hi,
                                           "fcr": k_hi / len(hi_bin) if hi_bin else None,
                                           "cp_lower": lo_hi, "cp_upper": up_hi},
            "pooled_eta_dec_lt_half_eps": {"n_cert": len(lo_bin), "n_false_cert": k_lo,
                                           "fcr": k_lo / len(lo_bin) if lo_bin else None},
            "eta_dec_over_eps": {"median": float(np.median(ed)), "p10": float(np.quantile(ed, .1)),
                                 "p90": float(np.quantile(ed, .9)), "max": float(ed.max()),
                                 "share_ge_half": float(np.mean(ed >= 0.5)), "share_ge_1": float(np.mean(ed >= 1.0))},
            "eta_dec_computed_for_all": bool(all(r["eta_dec"] is not None and np.isfinite(r["eta_dec"]) for r in R)),
            "gcirc_eq_nearest_rate": float(np.mean([r["gcirc_eq_nearest"] for r in R])),
            "share_false_cert_possible": float(np.mean([r["n_policies_regret_gt_eps"] > 0 for r in R])),
            "share_gcirc_argmax_regret_gt_eps": float(np.mean([r["gcirc_argmax_true_regret"] > EPS for r in R])),
            "share_nearest_argmax_regret_gt_eps": float(np.mean([r["nearest_argmax_true_regret"] > EPS for r in R])),
            "model_cert_true_regret_over_eps_max": (float(max(r["model_true_regret"] for r in mc)) / EPS) if mc else None,
            "nearest_cell_survival": float(np.mean([r["theta_cell_alive"] for r in R])),
            "gcirc_cell_survival": float(np.mean([r["gcirc_cell_alive"] for r in R])),
            "zero_cost_rate": float(np.mean([r["zero_cost"] for r in R])),
            "stream_charged_steps_median": float(np.median([s["stream_charged_steps"] for s in st])) if st else None,
            "stream_model_steps_median": float(np.median([s["stream_model_steps"] for s in st])) if st else None,
            "median_steps_model_cert": float(np.median([r["steps_consumed_model"] for r in mc])) if mc else None,
            "mean_true_regret_model_cert": float(np.mean([r["model_true_regret"] for r in mc])) if mc else None,
            "alarm_streams": sum(1 for s in st if s["alarm"]),
            "wall_s_per_problem_median": float(np.median([r["wall_clock_s"] for r in R])),
        }
    # ---- HF3: paired detection of plug-in false certificates by the mixed numerator (CRN)
    by = {(r["instance"], r["stream"], r["k"], r["method"]): r for r in rows}
    fcs = [r for r in rows if r["method"] == "JPC" and r["model_false_cert"]]
    det = []
    for r in fcs:
        m = by.get((r["instance"], r["stream"], r["k"], "JPC-mixed"))
        if m is None:
            continue
        det.append(bool(m["model_status"] == "MODEL_CONFLICT" or m["alarm_active"]))
    kd = sum(det)
    lo_d, up_d = clopper_pearson(kd, len(det))
    M = [r for r in rows if r["method"] == "JPC-mixed"]
    Ms = [s for s in streams if s["method"] == "JPC-mixed"]
    n_conf = sum(r["model_status"] == "MODEL_CONFLICT" for r in M)
    mixed_fc = sum(r["model_false_cert"] for r in M)
    out["hf3_discretization"] = {
        "n_plugin_false_cert": len(fcs), "n_detected": kd,
        "mixed_detection_rate": kd / len(det) if det else None, "detection_cp": [lo_d, up_d],
        "threshold_detect_le": 2 * DELTA,
        "pass_detect_le_2delta": (kd / len(det) <= 2 * DELTA) if det else None,
        "mixed_alarm_rate_per_problem": n_conf / len(M) if M else None,
        "mixed_alarm_rate_per_problem_cp": clopper_pearson(n_conf, len(M)) if M else None,
        "mixed_alarm_streams": sum(1 for s in Ms if s["alarm"]), "n_streams": len(Ms),
        "mixed_model_false_cert": mixed_fc,
        "replay_checks": sum(1 for r in M if r["replay"]),
        "replay_conflicts": sum(1 for r in M if r["replay"] and r["replay"]["conflict"]),
        "min_gap_vs_pool_min": float(min(s["min_gap_vs_pool"] for s in Ms)) if Ms else None,
        "log_w_pool": math.log(1 / 3),
        "note": "detection = paired JPC-mixed problem is MODEL_CONFLICT or the mixed alarm was already raised by then",
    }
    J = out["arms"].get("JPC", {})
    hr3 = {"jonckheere_p_perm": (J.get("jonckheere") or {}).get("p_perm"),
           "jonckheere_p_normal": (J.get("jonckheere") or {}).get("p_normal"),
           "cp_lower_eta_ge_half_eps": (J.get("pooled_eta_dec_ge_half_eps") or {}).get("cp_lower"),
           "n_cert_eta_ge_half_eps": (J.get("pooled_eta_dec_ge_half_eps") or {}).get("n_cert"),
           "thresholds": {"jonckheere_p_lt": 0.05, "cp_lower_gt": DELTA}}
    hr3["pass_monotone"] = hr3["jonckheere_p_perm"] is not None and hr3["jonckheere_p_perm"] < 0.05
    hr3["pass_cp_lower"] = hr3["cp_lower_eta_ge_half_eps"] is not None and hr3["cp_lower_eta_ge_half_eps"] > DELTA
    tab = J.get("fcr_by_eta_dec_bin") or []
    hr3["falsified_all_bins_le_delta"] = bool(tab) and all((b["fcr"] or 0.0) <= DELTA for b in tab if b["n_cert"])
    hr3["note_pilot"] = "pilot n is small: HR3 is evaluated on the full eval set only; this is a smoke/timing readout"
    out["hr3"] = hr3
    out.update({"n_rows": len(rows), "n_streams": len(streams), "wall_s": wall,
                "timing_note": "CPU-only, 4 workers, measured while other tasks ran concurrently (per-problem times are inflated)"})
    return out


def plots(rows, out_dir, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    mids = [0.125, 0.375, 0.75, 1.5, 3.0]
    for arm, mk in (("JPC", "o-"), ("JPC-mixed", "s--")):
        tab = summ["arms"].get(arm, {}).get("fcr_by_eta_dec_bin", [])
        xs = [mids[i] for i, b in enumerate(tab) if b["n_cert"]]
        ys = [b["fcr"] for b in tab if b["n_cert"]]
        lo = [b["fcr"] - b["cp_lower"] for b in tab if b["n_cert"]]
        hi = [b["cp_upper"] - b["fcr"] for b in tab if b["n_cert"]]
        if xs:
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, capsize=3, label=f"R1 {arm} (model certs)")
    try:
        hs = json.loads((RES_ROOT / "pilots" / "hf2b_m1_m3_mixed" / "summary.json").read_text())
        for kd, mk in (("m1", "^:"), ("m3", "v:")):
            xs, ys = [], []
            for e in (0, 1, 2):
                c = hs["cells"].get(f"{kd}_eta{e}", {}).get("plugin")
                if c and c.get("n_cert"):
                    xs.append(max(e, 0.06))
                    ys.append(c["fcr"])
            if xs:
                ax.plot(xs, ys, mk, alpha=.6, label=f"{kd} plug-in (eta_min/eps, hf2b pilot)")
    except Exception:  # noqa: BLE001
        pass
    ax.axhline(DELTA, color="grey", lw=.8, ls=":")
    ax.axvline(0.5, color="grey", lw=.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("measured eta_dec / eps (bin mid)")
    ax.set_ylabel("FCR")
    ax.set_title("R1 uncorrected: FCR vs eta_dec")
    ax.legend(fontsize=7)
    ax = axes[1]
    for arm, c in (("JPC", "C0"), ("JPC-mixed", "C1")):
        R = [r for r in rows if r["method"] == arm and r["model_cert"]]
        x = [max(r["eta_dec_over_eps"], 1e-3) for r in R]
        y = [r["model_true_regret"] / EPS for r in R]
        ax.scatter(x, y, s=10, alpha=.6, c=c, label=arm)
    ax.axhline(1.0, color="grey", lw=.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("eta_dec / eps")
    ax.set_ylabel("true regret / eps (model-certified)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "fcr_vs_eta_dec.png", dpi=130)
    plt.close(fig)


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", default=None)
    ap.add_argument("--problems", type=int, default=None)
    ap.add_argument("--streams", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--no-protocol", action="store_true", help="smoke runs: do not write DONE / gpu_progress")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    if pilot:
        seeds = PILOT_SEEDS
    else:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        r = lock["eval_manifest"]["per_task_ranges"][TASK][0]
        seeds = list(range(r[0], r[1] + 1))
    if args.instances:
        seeds = parse_range(args.instances)
    if pilot:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    else:
        assert all(s >= 10000 for s in seeds)
    K = args.problems or (10 if pilot else 15)
    streams_ids = [int(x) for x in (args.streams or ("0" if pilot else "0,1,2")).split(",")]
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_f, done_f, log_f = out_dir / "results.jsonl", out_dir / "units_done.jsonl", out_dir / "run.log"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_f, "a") as fh:
            fh.write(line + "\n")

    proto = not args.no_protocol
    if proto:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
        progress(0, 1, "start")
    start_iso = datetime.now().isoformat()
    T0 = time.time()
    status = "success"
    result_summary = ""
    try:
        n_new_tables = precompute_tables(seeds, streams_ids, K, log)
        gpu_peak = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
        if proto:
            (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps({
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_total_mb": (torch.cuda.get_device_properties(0).total_memory / 2 ** 20
                                  if torch.cuda.is_available() else None),
                "max_batch_size": None, "vram_used_mb": gpu_peak, "utilization_pct": None,
                "note": f"CPU-only learners; GPU only for missing G_1 J tables ({n_new_tables} computed); no batch "
                        "probing applicable (exact enumeration, no NN)"}))
        done = set()
        if done_f.exists():
            for ln in done_f.read_text().splitlines():
                u = json.loads(ln)
                done.add((u["seed"], u["stream"]))
        jobs = [(s, st) for s in seeds for st in streams_ids if (s, st) not in done]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} ({len(seeds)}) streams={streams_ids} K={K} arms={ARMS} "
            f"T_max={TMAX_STEP} workers={args.workers}; units todo={len(jobs)} done={len(done)}; "
            f"generator_hash={generator_hash()}")
        total = len(jobs) + len(done)
        if jobs and not args.resummarize:
            want = {(seeds[0], streams_ids[0]), (seeds[min(1, len(seeds) - 1)], streams_ids[0])}
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(run_unit)(s, st, K, (s, st) in want) for (s, st) in jobs)
            n_err = 0
            for i, out in enumerate(gen):
                with open(res_f, "a") as fh:
                    for r in out["rows"]:
                        fh.write(json.dumps({"rec": "row", **r}, default=str) + "\n")
                    for st in out["streams"]:
                        fh.write(json.dumps({"rec": "stream", **st}, default=str) + "\n")
                for sm in out["samples"]:
                    (out_dir / "samples" / f"i{out['seed']}_s{out['stream']}_{sm['method']}_k{sm['k']}.json"
                     ).write_text(json.dumps(sm, indent=1, default=str))
                for e in out["errors"]:
                    n_err += 1
                    log(f"ERROR unit {out['seed']}/{out['stream']} {e['where']}: {e['error']}\n{e['tb']}")
                if not out["errors"]:
                    with open(done_f, "a") as fh:
                        fh.write(json.dumps({"seed": out["seed"], "stream": out["stream"],
                                             "wall_s": out["wall_s"]}) + "\n")
                fc = {a: sum(r["model_false_cert"] for r in out["rows"] if r["method"] == a) for a in ARMS}
                al = [s["alarm"]["round"] if s["alarm"] else None for s in out["streams"] if s["method"] == "JPC-mixed"]
                ed = [round(r["eta_dec_over_eps"], 2) for r in out["rows"] if r["method"] == "JPC"]
                log(f"unit {i + 1}/{len(jobs)} {out['seed']} s{out['stream']} {out['wall_s']:.0f}s "
                    f"(ctx {out.get('ctx_s', 0):.0f}s) model_fc={fc} mixed_alarm={al} eta_dec/eps(JPC)={ed}")
                if proto:
                    progress(len(done) + i + 1, total, "units", {"errors": n_err})
        rows, streams = [], []
        for ln in res_f.read_text().splitlines():
            d = json.loads(ln)
            rec = d.pop("rec")
            (rows if rec == "row" else streams).append(d)
        summ = summarize(rows, streams, args.mode, time.time() - T0)
        crashed = sum(1 for ln in log_f.read_text().splitlines() if ln.startswith("[") and "ERROR unit" in ln)
        jt = summ["arms"].get("JPC", {})
        summ.update({"seeds": [seeds[0], seeds[-1]], "K": K, "streams": streams_ids, "T_max_step": TMAX_STEP,
                     "T_max_trial": TMAX_TRIAL, "generator_hash": generator_hash(), "eval_seeds_touched": not pilot,
                     "n_unit_errors_logged": crashed})
        n_runs = len(rows)
        per_unit = [json.loads(ln)["wall_s"] for ln in done_f.read_text().splitlines()] if done_f.exists() else []
        if per_unit:
            full_units = 48 * 3
            full_k_scale = 15 / K
            summ["full_projection_min"] = float(np.mean(per_unit) * full_k_scale * full_units / args.workers / 60)
            summ["full_projection_note"] = ("mean unit wall x (15/K) x 144 units / workers; stream-0 pilot problems "
                                            "only, concurrent CPU load")
        summ["pilot_gate"] = {
            "end_to_end": n_runs > 0 and crashed == 0, "n_problem_method_runs": n_runs,
            "eta_dec_computed_every_problem": all(a.get("eta_dec_computed_for_all") for a in summ["arms"].values()),
            "fcr_by_bin_table_produced": bool(jt.get("fcr_by_eta_dec_bin")),
        }
        summ["pilot_gate"]["pass"] = all(summ["pilot_gate"][k] for k in ("end_to_end", "eta_dec_computed_every_problem",
                                                                         "fcr_by_bin_table_produced"))
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(rows, out_dir, summ)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; pilot_gate={summ['pilot_gate']}; wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"pilot_gate": summ["pilot_gate"], "hr3": summ["hr3"],
                                     "hf3": summ["hf3_discretization"]}, default=str)[:1500]
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    if proto:
        mark_done(status, result_summary)
        update_gpu_progress(status, start_iso, wall / 60,
                            {"env": "E1-NL-S R1 off-grid truth, learner G_1 (|Theta|=13824), uncorrected",
                             "mode": args.mode, "instances": len(seeds), "K": K, "streams": streams_ids,
                             "arms": list(ARMS), "workers": args.workers, "gpu_count": 0,
                             "note": "CPU only, concurrent with other tasks"}, PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
