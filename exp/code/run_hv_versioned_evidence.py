"""hv_versioned_evidence: cand_g pilot -- versioned joint evidence (VJE) vs naive reuse vs detect-reset.

Usage: run_hv_versioned_evidence.py --mode {pilot,full} [--workers 4] [--instances a-b] [--n-old 20,1000]
                                    [--k-true 0,1] [--problems K] [--streams 0]

Setting (methodology 2.2 / 4.2, candidates.json cand_g): dynamic E1-NL-S class G_1 (|Theta| = 13824), on-grid R0
truth theta*_old. OLD epoch: n_old real rounds (n0 = 20 shared initial rounds + n_old - 20 uniform legal rounds).
KNOWN epoch boundary, then an IN-CLASS change of k_true in {0, 1} participants: with k_true = 1 one participant
slot (left tuple (alpha, gamma, tauL) or right tuple (beta, tauR)) jumps to a different grid tuple; global
(psi, lam) are shared. The change is drawn by the harness (rng [seed, stream, 77]); the same draw is used for
k_true = 0 and 1 (paired), and the platform state (loads / engaged / noise rng) continues across the boundary.
NEW epoch: a stream of K Type-1 problems (offgrid R0 generator, stream order perm), pre-registered DDA acquisition,
T_max = 3000 new steps / problem, each arm reuses its OWN ledger across the stream (CRN: one fresh platform per arm,
identical old-epoch actions; the DDA rng seed is shared).
Arms (all decisions are minimax-regret certificates on G_1 with exact class J tables, eps = 0.02, delta = 0.05):
  naive        SeqLRSet over ALL rounds, ignores the boundary (falsification scope {}). Empty set -> MODEL_CONFLICT,
               censored at T_max (no fallback).
  detect_reset DetectReset(MixedLRSet {pool, reg(load bucket), ext}): when the mixed set becomes empty the evidence
               is cleared and a fresh mixed set starts from the next real round. Does not use the boundary.
  vje          VersionedJointLR (k = 1): joint (theta_old, theta_new) evidence, theta_new = theta_old except on ONE
               participant slot; decisions on the projection Theta_new. Uses the known boundary.
  reset_known  fresh SeqLRSet started exactly at the known boundary (old data discarded): the "version everything"
               comparator; isolates what VJE retains from the old epoch.
Metrics: naive FCR by n_old (HV1), stream ratio vje / detect_reset (HV3; instance-cluster bootstrap on log scale),
vje / reset_known, coverage of the true (versioned) hypothesis, boundary (data-free) certificate survival on
problems whose old-epoch certificate is still eps-correct under the new truth (unaffected_cert_survival),
alarm rounds of detect_reset, completion, zero-cost rate.
Pilot: dev seeds 676-680 x stream 0 x 10 problems x n_old {20, 1000} x k_true {0, 1}.
Full: assert_locked(); eval 10000-10023 x 3 streams x 10 problems x n_old {20, 200, 1000} x k_true {0, 1}.
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
from dsswm.certify.minimax_enum import certify_minimax, group_ids, observable_signature, regret_matrix  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.evidence.mixed_lr import MixedLRSet  # noqa: E402
from dsswm.evidence.regimes import load_bucket  # noqa: E402
from dsswm.evidence.versioned import DetectReset, VersionedJointLR  # noqa: E402
from dsswm.exact.nl_propagate import NLPropagator  # noqa: E402
from dsswm.models.grid_ladder import ladder_grid  # noqa: E402
from dsswm.models.nl_class import NLClass  # noqa: E402
from dsswm.stats.cp import clopper_pearson  # noqa: E402
from dsswm.streams.generator import NL_DEFAULTS, generator_hash, nl_class_max_jtable  # noqa: E402
from dsswm.streams.offgrid import STREAMS, make_offgrid_instance  # noqa: E402

TASK = "hv_versioned_evidence"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
C_KNOWN, RHO_RET, NMAX, N0 = NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"], NL_DEFAULTS["nmax"], NL_DEFAULTS["n0"]
LOG_THR = math.log(1.0 / DELTA)
TMAX_STEP = 3000
ARMS = ("naive", "detect_reset", "vje", "reset_known")
RES_ROOT = WS / "exp" / "results"
JT_CACHE = WS / "exp" / "cache" / "jtables_f1"
PLANNED_MIN = {"pilot": 12, "full": 35}
CHANGE_RULE = os.environ.get("HV_CHANGE_RULE", "salient")


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
    return _W


# ------------------------------------------------------------------ evidence adapters (uniform interface)
class NaiveArm:
    def __init__(self, W):
        self.E = SeqLRSet(W["prop"], W["LT"], DELTA)

    def feed(self, o, pid):
        self.E.update(o)

    def boundary(self):
        pass

    def log_ratio(self):
        return self.E.log_ratio().numpy()

    def mle(self):
        return self.E.mle()

    @property
    def n_rounds(self):
        return self.E.n_rounds


class ResetKnownArm(NaiveArm):
    def __init__(self, W):
        self.W = W
        super().__init__(W)
        self.n_total = 0

    def feed(self, o, pid):
        self.E.update(o)
        self.n_total += 1

    def boundary(self):
        self.E = SeqLRSet(self.W["prop"], self.W["LT"], DELTA)

    @property
    def n_rounds(self):
        return self.n_total


class DetectResetArm:
    def __init__(self, W):
        self.D = DetectReset(lambda: MixedLRSet(W["prop"], W["LT"], DELTA, regime_fn=load_bucket, name="main"))
        self.n_total = 0

    def feed(self, o, pid):
        self.D.update(o, problem_id=pid)
        self.n_total += 1

    def boundary(self):
        pass

    def log_ratio(self):
        if self.D.cur.n_rounds == 0:              # just reset: nothing excluded
            return np.zeros(self.D.cur.B)
        return self.D.cur.log_ratio().numpy()

    def mle(self):
        return self.D.cur.mle()

    @property
    def n_rounds(self):
        return self.n_total


class VJEArm:
    def __init__(self, W):
        self.V = VersionedJointLR(W["ncl"], W["prop"], W["LT"], DELTA)
        self.flat_idx = self.V.newidx.reshape(-1)

    def feed(self, o, pid):
        self.V.update(o)

    def boundary(self):
        self.V.start_new_epoch()

    def _profile(self):
        """Profile log-likelihood of theta_new: max over alive (theta_old, change) pairs that map to it."""
        V = self.V
        if V.epoch == 0:
            return V.cum_old
        jt = V.joint().reshape(-1)
        prof = torch.full((V.B,), -float("inf"), dtype=jt.dtype)
        return prof.scatter_reduce(0, self.flat_idx, jt, reduce="amax", include_self=True)

    def log_ratio(self):
        return (self.V.log_num() - self._profile()).numpy()

    def mle(self):
        V = self.V
        if V.epoch == 0:
            return int(torch.argmax(V.cum_old))
        flat = int(torch.argmax(V.joint()))
        return int(V.newidx_np.reshape(-1)[flat])

    @property
    def n_rounds(self):
        return self.V.n_old + self.V.n_new


ARM_CLS = {"naive": NaiveArm, "detect_reset": DetectResetArm, "vje": VJEArm, "reset_known": ResetKnownArm}


# ------------------------------------------------------------------ harness: change draw (truth side)
def draw_change(ncl, ti_old, seed, stream, J=None, rule="salient"):
    """One participant slot s (0..L+R-1, uniform) and a different digit; returns (slot, new_index).
    rule 'random' : digit uniform among the other digits of slot s.
    rule 'salient': digit maximising sum_k max_pi |J_new - J_old| over the stream's problems (harness side, class J
                    tables at the two grid points); makes the change decision-relevant more often (HV1 power)."""
    rng = np.random.default_rng([seed, stream, 77])
    radices = ncl._radices
    S = len(radices) - 1
    strides = np.ones(len(radices), dtype=np.int64)
    for s in range(len(radices) - 2, -1, -1):
        strides[s] = strides[s + 1] * radices[s + 1]
    s = int(rng.integers(S))
    cur = (ti_old // strides[s]) % radices[s]
    others = [d for d in range(radices[s]) if d != cur]
    d = int(rng.choice(others))
    if rule == "salient" and J is not None:
        score = [sum(float(np.abs(Jq[ti_old + (dd - cur) * strides[s]] - Jq[ti_old]).max()) for Jq in J)
                 for dd in others]
        d = others[int(np.argmax(score))]
    return s, int(ti_old + (d - cur) * strides[s])


def apply_params(env, p):
    """Harness-only: move the platform's participant parameters to grid point p (state / rng continue)."""
    L = env.L
    env._alpha = np.asarray(p["alpha"], float).copy()
    env._beta = np.asarray(p["beta"], float).copy()
    env._gamma = np.asarray(p["gamma"], float).copy()
    env._tau = np.concatenate([np.asarray(p["tauL"], float), np.asarray(p["tauR"], float)])
    assert env._alpha.shape[0] == L


class Ctx:
    def __init__(self, seed, stream, K, n_old, k_true):
        self.seed, self.stream, self.n_old, self.k_true = seed, stream, n_old, k_true
        self.noise = STREAMS[stream][0]
        inst = make_offgrid_instance(seed, "R0", stream=stream)
        self.aspace = inst.env.aspace
        W = worker_tables(self.aspace)
        self.W = W
        self.ncl, self.prop = W["ncl"], W["prop"]
        self.problems = inst.problems[:K]
        self.legal = np.arange(self.aspace.n)
        t = inst.truth
        self.ti_old = int(self.ncl.index_of(tuple(t["alpha"]), tuple(t["beta"]), tuple(t["gamma"]), tuple(t["tauL"]),
                                            tuple(t["tauR"]), t["psi"], t["lam"]))
        self.J, self.Reg = [], []
        for q in self.problems:
            path = JT_CACHE / f"{q.pid}_raw.npy"
            J = nl_class_max_jtable(q, self.prop, W["params"], cache_path=str(path) if path.exists() else None)
            if not path.exists():
                tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
                np.save(tmp, np.asarray(J, float) / q.utility.c_q)
                os.replace(tmp, path)
            self.J.append(np.asarray(J, float))
            self.Reg.append(regret_matrix(self.J[-1]))
        self.change_slot, ti_alt = draw_change(self.ncl, self.ti_old, seed, stream, self.J, CHANGE_RULE)
        self.ti_new = ti_alt if k_true == 1 else self.ti_old
        p_old, p_new = self.ncl.params_at(self.ti_old), self.ncl.params_at(self.ti_new)
        self.change_desc = {k: [np.round(np.asarray(p_old[k]), 3).tolist(), np.round(np.asarray(p_new[k]), 3).tolist()]
                            for k in ("alpha", "gamma", "tauL", "beta", "tauR") if not np.allclose(p_old[k], p_new[k])}
        # truth on the grid: class J rows are exact
        self.Jt_new = [J[self.ti_new] for J in self.J]
        self.Jt_old = [J[self.ti_old] for J in self.J]
        self.decision_changed = [float(Jn.max() - Jn[int(np.argmax(Jo))]) > EPS
                                 for Jo, Jn in zip(self.Jt_old, self.Jt_new)]
        self.dJ_max = [float(np.abs(Jn - Jo).max()) for Jo, Jn in zip(self.Jt_old, self.Jt_new)]

    def fresh(self):
        return make_offgrid_instance(self.seed, "R0", stream=self.stream)


# ------------------------------------------------------------------ one arm over one (old epoch, boundary, stream)
def run_arm(ctx, arm, old_cert=None, want_samples=False):
    inst = ctx.fresh()
    env = inst.env
    h = env.handle()
    A = ARM_CLS[arm](ctx.W)
    n_alarm_before = 0
    for o in inst.init_obs[:ctx.n_old]:
        A.feed(o, None)
    arng = np.random.default_rng([ctx.seed, ctx.stream, 12, 5])          # identical old-epoch actions for every arm
    for _ in range(max(0, ctx.n_old - len(inst.init_obs))):
        A.feed(h.step(int(arng.integers(h.aspace.n))), None)
    assert ctx.n_old >= len(inst.init_obs) and env.n_steps == ctx.n_old == A.n_rounds
    n_old_real = env.n_steps
    if arm == "detect_reset":
        n_alarm_before = len(A.D.alarms)
    # old-epoch certificates (naive arm only computes them; they equal the old SeqLRSet at the boundary)
    boundary_old = None
    if arm == "naive":
        m = A.log_ratio() < LOG_THR
        boundary_old = []
        for k in range(len(ctx.problems)):
            c = certify_minimax(ctx.Reg[k], m, EPS, TOP_M)
            boundary_old.append(c["pi"] if c["status"].value == "CERTIFIED" else None)
    # ---- known epoch boundary
    A.boundary()
    apply_params(env, ctx.ncl.params_at(ctx.ti_new))
    # boundary (data-free) certificate survival on the new epoch
    mask_b = A.log_ratio() < LOG_THR
    boundary_cert = []
    for k in range(len(ctx.problems)):
        c = certify_minimax(ctx.Reg[k], mask_b, EPS, TOP_M) if mask_b.any() else None
        ok = c is not None and c["status"].value == "CERTIFIED"
        pi = c["pi"] if ok else None
        boundary_cert.append({"certified": ok, "pi": pi,
                              "correct_new": (bool(ctx.Jt_new[k].max() - ctx.Jt_new[k][pi] <= EPS) if ok else None)})
    boundary_set = int(mask_b.sum())
    rng = np.random.default_rng([ctx.seed, ctx.noise, 101, ctx.n_old])  # same DDA stream for every arm
    rows, samples = [], []
    cum_new = 0
    for k, q in enumerate(ctx.problems):
        Reg = ctx.Reg[k]
        t0 = time.perf_counter()
        steps, status, traj = 0, None, []
        while True:
            lrat = A.log_ratio()
            mask = lrat < LOG_THR
            if not mask.any():
                status = "MODEL_CONFLICT"
                break
            cert = certify_minimax(Reg, mask, EPS, TOP_M)
            if cert["status"].value == "CERTIFIED":
                status = "CERTIFIED"
                break
            if steps >= TMAX_STEP:
                status = "NEED_DATA"
                break
            code = ctx.prop.codec.encode(*h.observable_state())
            blk = cert["blocking"] or certify_minimax(Reg, mask, 0.0, TOP_M)["blocking"]
            kh = A.mle()
            if not blk:
                a = int(rng.choice(ctx.legal)); mode = "random"
            else:
                margins = LOG_THR - lrat[blk]
                a, info = dda_choose(code, ctx.W["py"][kh:kh + 1], ctx.W["pe"][kh:kh + 1], ctx.W["LT_np"][kh],
                                     ctx.W["py"][blk], ctx.W["pe"][blk], margins, ctx.W["inc"], ctx.prop, ctx.legal,
                                     rng)
                mode = info["mode"]
            obs = h.step(a)
            steps += 1
            A.feed(obs, q.pid)
            if want_samples and len(traj) < 25:
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "true_new_alive": bool(mask[ctx.ti_new]), "mode": mode})
        assert env.n_steps == A.n_rounds, "every evidence round must be a real platform round"
        cum_new += steps
        Jtq = ctx.Jt_new[k]
        mask = A.log_ratio() < LOG_THR
        cert_ok = status == "CERTIFIED"
        pi = cert["pi"] if cert_ok else None
        charged = steps if cert_ok else max(steps, TMAX_STEP)
        tr = float(Jtq.max() - Jtq[pi]) if cert_ok else None
        row = {"task": TASK, "instance": ctx.seed, "stream": ctx.stream, "noise_seed": ctx.noise, "n_old": ctx.n_old,
               "k_true": ctx.k_true, "change_slot": ctx.change_slot if ctx.k_true else None,
               "problem": q.pid, "k": k, "method": arm, "design": "DDA", "status": status,
               "new_env_steps": int(charged), "steps_consumed": int(steps), "censored": not cert_ok,
               "certified_policy": pi, "true_regret": tr, "false_cert": bool(cert_ok and tr > EPS),
               "zero_cost": bool(cert_ok and steps == 0), "model_conflict": status == "MODEL_CONFLICT",
               "true_new_alive": bool(mask[ctx.ti_new]) if mask.any() else False,
               "true_old_alive": bool(mask[ctx.ti_old]) if mask.any() else False,
               "vje_covers_true_pair": (bool(A.V.contains(ctx.ti_old, ctx.ti_new)) if arm == "vje" else None),
               "set_size": int(mask.sum()), "decision_changed": bool(ctx.decision_changed[k]),
               "dJ_max_old_new": ctx.dJ_max[k], "true_best": int(np.argmax(Jtq)),
               "true_top2_gap": float(np.sort(Jtq)[-1] - np.sort(Jtq)[-2]), "H": q.H, "n_policies": len(q.policies),
               "boundary_cert": boundary_cert[k]["certified"], "boundary_cert_correct": boundary_cert[k]["correct_new"],
               "eps": EPS, "delta": DELTA, "cum_new_steps": int(cum_new), "rollouts": 0,
               "wall_clock_s": time.perf_counter() - t0}
        rows.append(row)
        if want_samples and k < 3:
            samples.append({**row, "J_true_new": np.round(Jtq, 4).tolist(),
                            "J_true_old": np.round(ctx.Jt_old[k], 4).tolist(), "trajectory_head": traj})
    st = {"instance": ctx.seed, "stream": ctx.stream, "n_old": ctx.n_old, "k_true": ctx.k_true, "method": arm,
          "change_slot": ctx.change_slot if ctx.k_true else None, "change": ctx.change_desc,
          "n_old_real": int(n_old_real), "boundary_set_size": boundary_set,
          "boundary_cert": boundary_cert,
          "stream_charged_steps": int(sum(r["new_env_steps"] for r in rows)), "stream_model_steps": int(cum_new),
          "n_false_cert": int(sum(r["false_cert"] for r in rows)),
          "final_true_new_alive": bool(rows[-1]["true_new_alive"]),
          "detect_reset_alarms": (list(A.D.alarms) if arm == "detect_reset" else None),
          "detect_reset_alarms_old_epoch": (n_alarm_before if arm == "detect_reset" else None),
          "vje_changed_slots_alive": (A.V.changed_slots_alive() if arm == "vje" else None)}
    if boundary_old is not None:
        st["old_epoch_cert_pi"] = boundary_old
    return rows, samples, st


def run_unit(seed, stream, K, n_old, k_true, want_samples):
    t0 = time.time()
    out = {"seed": seed, "stream": stream, "n_old": n_old, "k_true": k_true, "rows": [], "streams": [],
           "samples": [], "errors": []}
    try:
        ctx = Ctx(seed, stream, K, n_old, k_true)
        out["ctx"] = {"instance": seed, "stream": stream, "n_old": n_old, "k_true": k_true, "ti_old": ctx.ti_old,
                      "ti_new": ctx.ti_new, "change_slot": ctx.change_slot, "change": ctx.change_desc,
                      "decision_changed": ctx.decision_changed, "dJ_max": ctx.dJ_max, "ctx_s": time.time() - t0}
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
        return [None, None]
    lo, hi = clopper_pearson(int(k), int(n))
    return [float(lo), float(hi)]


def ratio_stats(streams, num, den, n_old, k_true, B=10000, seed=42):
    P = {(s["instance"], s["stream"]): s["stream_charged_steps"] for s in streams
         if s["method"] == den and s["n_old"] == n_old and s["k_true"] == k_true}
    A = {(s["instance"], s["stream"]): s["stream_charged_steps"] for s in streams
         if s["method"] == num and s["n_old"] == n_old and s["k_true"] == k_true}
    keys = sorted(set(P) & set(A))
    if not keys:
        return None
    lr = {k: math.log((A[k] + 1.0) / (P[k] + 1.0)) for k in keys}
    inst = sorted({k[0] for k in keys})
    per_inst = np.array([np.mean([lr[k] for k in keys if k[0] == i]) for i in inst])
    rng = np.random.default_rng(seed)
    bs = per_inst[rng.integers(0, len(inst), size=(B, len(inst)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n_streams": len(keys), "n_instances": len(inst), "ratio_geo_mean": float(math.exp(per_inst.mean())),
            "ci95": [float(math.exp(lo)), float(math.exp(hi))],
            "ci_upper_one_sided_95": float(math.exp(np.percentile(bs, 95.0))),
            "per_stream_ratio": {f"{k[0]}_s{k[1]}": round(math.exp(lr[k]), 4) for k in keys}}


def summarize(rows, streams, cells, mode, wall):
    S = {"task": TASK, "mode": mode, "eps": EPS, "delta": DELTA, "cells": {}, "n_rows": len(rows),
         "n_streams": len(streams), "wall_s": wall,
         "timing_note": "并发运行：wall clock measured with up to 4 tasks sharing 20 cores; 4 workers here"}
    for (n_old, k_true) in cells:
        C = {}
        for arm in ARMS:
            R = [r for r in rows if r["n_old"] == n_old and r["k_true"] == k_true and r["method"] == arm]
            ST = [s for s in streams if s["n_old"] == n_old and s["k_true"] == k_true and s["method"] == arm]
            if not R:
                continue
            n_cert = sum(r["status"] == "CERTIFIED" for r in R)
            fc = sum(r["false_cert"] for r in R)
            Rch = [r for r in R if r["decision_changed"]]
            fc_ch = sum(r["false_cert"] for r in Rch)
            nc_ch = sum(r["status"] == "CERTIFIED" for r in Rch)
            # unaffected certificates: old-epoch certificate (naive at the boundary) still eps-correct under new truth
            surv_n, surv_k = 0, 0
            for s in ST:
                old = next((x for x in streams if x["method"] == "naive" and x["instance"] == s["instance"]
                            and x["stream"] == s["stream"] and x["n_old"] == n_old and x["k_true"] == k_true), None)
                if old is None:
                    continue
                for k, pi in enumerate(old["old_epoch_cert_pi"]):
                    row = next(r for r in R if r["instance"] == s["instance"] and r["stream"] == s["stream"]
                               and r["k"] == k)
                    if pi is None or row["true_best"] is None:
                        continue
                    if row["decision_changed"]:
                        continue
                    surv_n += 1
                    surv_k += int(bool(s["boundary_cert"][k]["certified"] and s["boundary_cert"][k]["correct_new"]))
            bc = [b for s in ST for b in s["boundary_cert"]]
            c = {"n_problems": len(R), "n_streams": len(ST), "n_cert": n_cert, "completion": n_cert / len(R),
                 "n_false_cert": int(fc), "fcr": fc / n_cert if n_cert else None, "fcr_cp": _cp(fc, n_cert),
                 "decision_changed_problems": len(Rch), "fcr_on_decision_changed": fc_ch / nc_ch if nc_ch else None,
                 "status_counts": {x: sum(r["status"] == x for r in R) for x in sorted({r["status"] for r in R})},
                 "stream_charged_steps_median": float(np.median([s["stream_charged_steps"] for s in ST])),
                 "stream_charged_steps_mean": float(np.mean([s["stream_charged_steps"] for s in ST])),
                 "zero_cost_rate": sum(r["zero_cost"] for r in R) / len(R),
                 "coverage_true_new": sum(r["true_new_alive"] for r in R) / len(R),
                 "coverage_true_new_cp": _cp(sum(r["true_new_alive"] for r in R), len(R)),
                 "boundary_set_size_median": float(np.median([s["boundary_set_size"] for s in ST])),
                 "boundary_cert_rate": sum(b["certified"] for b in bc) / len(bc) if bc else None,
                 "boundary_cert_wrong": int(sum(b["certified"] and not b["correct_new"] for b in bc)),
                 "unaffected_cert_survival": (surv_k / surv_n) if surv_n else None,
                 "unaffected_cert_n": surv_n}
            if arm == "detect_reset":
                c["alarm_streams"] = sum(bool(s["detect_reset_alarms"]) for s in ST)
                c["alarm_rounds"] = [s["detect_reset_alarms"] for s in ST]
                c["alarms_old_epoch"] = [s["detect_reset_alarms_old_epoch"] for s in ST]
            if arm == "vje":
                c["vje_covers_true_pair_rate"] = sum(bool(r["vje_covers_true_pair"]) for r in R) / len(R)
                c["vje_changed_slots_alive"] = [s["vje_changed_slots_alive"] for s in ST]
                c["vje_identifies_changed_slot"] = (
                    float(np.mean([s["vje_changed_slots_alive"] == [s["change_slot"]] for s in ST]))
                    if k_true == 1 else None)
            C[arm] = c
        C["ratios"] = {"vje_over_detect_reset": ratio_stats(streams, "vje", "detect_reset", n_old, k_true),
                       "vje_over_reset_known": ratio_stats(streams, "vje", "reset_known", n_old, k_true),
                       "vje_over_naive": ratio_stats(streams, "vje", "naive", n_old, k_true),
                       "detect_reset_over_reset_known": ratio_stats(streams, "detect_reset", "reset_known", n_old,
                                                                    k_true)}
        S["cells"][f"n_old={n_old},k_true={k_true}"] = C
    # HV1: naive FCR monotone in n_old (k_true = 1 is the informative cell)
    hv1 = {}
    for kt in sorted({c[1] for c in cells}):
        seq = []
        for n_old in sorted({c[0] for c in cells if c[1] == kt}):
            c = S["cells"][f"n_old={n_old},k_true={kt}"]["naive"]
            seq.append({"n_old": n_old, "fcr": c["fcr"], "fcr_cp": c["fcr_cp"], "n_cert": c["n_cert"],
                        "n_false_cert": c["n_false_cert"], "fcr_on_decision_changed": c["fcr_on_decision_changed"]})
        f = [x["fcr"] if x["fcr"] is not None else 0.0 for x in seq]
        hv1[f"k_true={kt}"] = {"by_n_old": seq, "monotone_nondecreasing": bool(all(a <= b for a, b in zip(f, f[1:]))),
                               "strict_increase_ends": bool(len(f) >= 2 and f[-1] > f[0])}
    S["HV1"] = hv1
    hv3 = {}
    for (n_old, kt) in cells:
        r = S["cells"][f"n_old={n_old},k_true={kt}"]["ratios"]["vje_over_detect_reset"]
        if r:
            hv3[f"n_old={n_old},k_true={kt}"] = {"ratio": r["ratio_geo_mean"], "ci95": r["ci95"],
                                                 "pass_ci_upper_lt_0.8": r["ci95"][1] < 0.8,
                                                 "falsified_ci_within_0.9_1.1": bool(r["ci95"][0] >= 0.9
                                                                                     and r["ci95"][1] <= 1.1)}
    S["HV3"] = hv3
    return S


def plots(S, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    cells = S["cells"]
    keys = list(cells)
    x = np.arange(len(keys))
    for j, arm in enumerate(ARMS):
        axs[0].bar(x + 0.2 * j - 0.3, [cells[k][arm]["stream_charged_steps_mean"] for k in keys], 0.2, label=arm)
        axs[1].bar(x + 0.2 * j - 0.3, [cells[k][arm]["fcr"] or 0.0 for k in keys], 0.2, label=arm)
    for ax, t in zip(axs, ("mean stream charged steps", "FCR")):
        ax.set_xticks(x, keys, rotation=20, fontsize=7)
        ax.set_title(t)
        ax.legend(fontsize=7)
    axs[1].axhline(DELTA, color="k", ls="--", lw=.8)
    fig.tight_layout()
    fig.savefig(out_dir / "hv_overview.png", dpi=130)
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
    ap.add_argument("--problems", type=int, default=10)
    ap.add_argument("--n-old", default=None)
    ap.add_argument("--k-true", default="0,1")
    ap.add_argument("--streams", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="no DONE marker / gpu_progress update")
    ap.add_argument("--change-rule", choices=["salient", "random"], default="salient")
    args = ap.parse_args()
    global CHANGE_RULE
    CHANGE_RULE = os.environ["HV_CHANGE_RULE"] = args.change_rule
    pilot = args.mode == "pilot"
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
        rng_ = lock["eval_manifest"]["per_task_ranges"].get(TASK, [[10000, 10023]])[0]
        seeds = list(range(rng_[0], rng_[1] + 1))
    else:
        seeds = list(range(676, 681))
    if args.instances:
        seeds = parse_range(args.instances)
    if pilot:
        assert all(600 <= s <= 699 for s in seeds), "pilot uses dev seeds 600-699 only"
    else:
        assert all(s >= 10000 for s in seeds)
    K = args.problems
    n_olds = [int(x) for x in (args.n_old or ("20,1000" if pilot else "20,200,1000")).split(",")]
    k_trues = [int(x) for x in args.k_true.split(",")]
    streams_ids = [int(x) for x in (args.streams or ("0" if pilot else "0,1,2")).split(",")]
    cells = [(n, k) for n in n_olds for k in k_trues]
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + args.tag)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    res_f, done_f, log_f = out_dir / "results.jsonl", out_dir / "units_done.jsonl", out_dir / "run.log"

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        with open(log_f, "a") as fh:
            fh.write(line + "\n")

    if not args.smoke:
        (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    start_iso = datetime.now().isoformat()
    T0 = time.time()
    if not args.smoke:
        progress(0, 1, "start")
    status = "success"
    result_summary = ""
    try:
        done = set()
        if done_f.exists():
            for ln in done_f.read_text().splitlines():
                u = json.loads(ln)
                done.add((u["seed"], u["stream"], u["n_old"], u["k_true"]))
        units = [(s, st, n, k) for (n, k) in cells for s in seeds for st in streams_ids]
        # heaviest first (n_old = 1000 has the longest old epoch; VJE joint is the same size)
        units.sort(key=lambda u: -u[2])
        jobs = [u for u in units if u not in done]
        log(f"mode={args.mode} seeds={seeds[0]}-{seeds[-1]} streams={streams_ids} K={K} n_old={n_olds} "
            f"k_true={k_trues} arms={ARMS} T_max={TMAX_STEP} workers={args.workers}; units todo={len(jobs)} "
            f"done={len(done)}; generator_hash={generator_hash()}")
        total = len(units)
        if jobs:
            sample_units = {(seeds[0], streams_ids[0], n, k) for (n, k) in cells}
            gen = Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                delayed(run_unit)(s, st, K, n, k, (s, st, n, k) in sample_units) for (s, st, n, k) in jobs)
            n_err = 0
            for i, out in enumerate(gen):
                with open(res_f, "a") as fh:
                    for r in out["rows"]:
                        fh.write(json.dumps({"rec": "row", **r}, default=str) + "\n")
                    for st in out["streams"]:
                        fh.write(json.dumps({"rec": "stream", **st}, default=str) + "\n")
                    if "ctx" in out:
                        fh.write(json.dumps({"rec": "ctx", **out["ctx"]}, default=str) + "\n")
                for sm in out["samples"]:
                    (out_dir / "samples" / f"i{out['seed']}_s{out['stream']}_n{out['n_old']}_k{out['k_true']}_"
                     f"{sm['method']}_q{sm['k']}.json").write_text(json.dumps(sm, indent=1, default=str))
                for e in out["errors"]:
                    n_err += 1
                    log(f"ERROR unit {out['seed']}/{out['stream']}/{out['n_old']}/{out['k_true']} {e['where']}: "
                        f"{e['error']}\n{e['tb']}")
                if not out["errors"]:
                    with open(done_f, "a") as fh:
                        fh.write(json.dumps({"seed": out["seed"], "stream": out["stream"], "n_old": out["n_old"],
                                             "k_true": out["k_true"], "wall_s": out["wall_s"]}) + "\n")
                stp = {s["method"]: s["stream_charged_steps"] for s in out["streams"]}
                fcs = {s["method"]: s["n_false_cert"] for s in out["streams"]}
                al = [s["detect_reset_alarms"] for s in out["streams"] if s["method"] == "detect_reset"]
                log(f"unit {i + 1}/{len(jobs)} {out['seed']} s{out['stream']} n_old={out['n_old']} "
                    f"k={out['k_true']} {out['wall_s']:.0f}s steps={stp} fc={fcs} dr_alarms={al}")
                if not args.smoke:
                    progress(len(done) + i + 1, total, "units", {"errors": n_err})
        rows, streams, ctxs = [], [], []
        for ln in res_f.read_text().splitlines():
            d = json.loads(ln)
            rec = d.pop("rec")
            {"row": rows, "stream": streams, "ctx": ctxs}[rec].append(d)
        summ = summarize(rows, streams, cells, args.mode, time.time() - T0)
        crashed = sum(1 for ln in log_f.read_text().splitlines() if "ERROR unit" in ln) if log_f.exists() else 0
        summ.update({"seeds": [seeds[0], seeds[-1]], "K": K, "streams": streams_ids, "n_old": n_olds,
                     "k_true": k_trues, "T_max_step": TMAX_STEP, "generator_hash": generator_hash(),
                     "eval_seeds_touched": not pilot, "n_unit_errors": crashed, "change_rule": CHANGE_RULE,
                     "change_draws": [{k: c[k] for k in ("instance", "stream", "n_old", "k_true", "change_slot",
                                                         "change", "decision_changed")} for c in ctxs]})
        (out_dir / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
        try:
            plots(summ, out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed: {e!r}")
        log(f"summary written; HV1={json.dumps(summ['HV1'], default=str)}; HV3={json.dumps(summ['HV3'], default=str)}; "
            f"wall {time.time() - T0:.0f}s")
        result_summary = json.dumps({"HV1": summ["HV1"], "HV3": summ["HV3"], "n_unit_errors": crashed},
                                    default=str)[:1500]
        if crashed and not rows:
            status = "failed"
    except Exception as e:  # noqa: BLE001
        status = "failed"
        result_summary = f"{e!r}"
        log(f"FATAL {e!r}\n{traceback.format_exc()}")
    wall = time.time() - T0
    if args.smoke:
        return 0 if status == "success" else 1
    mark_done(status, result_summary)
    update_gpu_progress(status, start_iso, wall / 60,
                        {"env": "E1-NL-S G_1 (|Theta|=13824), R0 truth, in-class change at known boundary",
                         "mode": args.mode, "instances": len(seeds), "K": K, "streams": streams_ids,
                         "n_old": n_olds, "k_true": k_trues, "arms": list(ARMS), "workers": args.workers,
                         "gpu_count": 0, "note": "CPU only, concurrent with other tasks"},
                        PLANNED_MIN[args.mode])
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
