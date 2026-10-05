"""nl_component_ablations: one-component-at-a-time ablations of JPC on E1-NL-S (kappa_high, eta = 0).

FULL JPC (reference arm): joint sequential-LR set over the finite class + exact-enumeration certificate +
DDA (top-m = 5 worst models, KL / cost, Track-and-Stop LP allocation) over the library
  (a) 1-step probes from the current state                       cost 1
  (a') steer-then-probe: deterministic shortest load path (d <= 2 rounds, known load rule) then a 1-step probe,
       cost = 1 + c_steer with c_steer = d (reachability-graph shortest path)
  (b) open-loop 2-step 'pressure-then-observe' sequences          cost 2   (receding horizon: first action)
  (c) whole candidate trials (reset to s_0 is free, as for B1)    cost H
+ cross-problem evidence ledger (Theta_t, platform state persist along the K-problem stream).

Arms (each changes exactly one component):
  no_csteer   steer-then-probe elements priced at probe length only (cost 1, steering rounds ignored)
  single_step library = 1-step probes + whole trials (no 2-step sequences, no steer-then-probe)
  top_m1      m = 1 worst model in the DDA allocation
  dual_bnb    certificate = participant-copy Lagrangian dual + BnB on 1 hub (sound UB); exact R_bar still
              computed as a free pre-screen (UB >= exact, so dual is only evaluated when exact <= eps: this gives
              the identical stopping time as evaluating the dual at every step); dual wall-clock is also sampled
              every 25 decisions to project the cost of running it at every step
  no_reuse    every problem restarts from the shared n0 = 20 initial rounds (fresh Theta, fresh platform)
  static      parameter-count-matched static model class: same grid (13824 points, same parameters) but the
              load enters neither the outcome logit (g(n) = 1) nor retention (g_ret(n) = 1): no fatigue dynamics
  audit       AGC gate: delta split delta_model = delta_audit = delta/2; after every model certificate run
              N_AUDIT on-policy whole trials of pi_hat and of the binding challenger, PPI-style residual test
              (model prediction as control variate, Hoeffding, union over K(K-1)/2 pairs); reject -> MODEL_CONFLICT
kappa = 0 controls (cand_c 'static gap < 1.2x' criterion and H4 control): full, single_step, static.

Usage: run_nl_component_ablations.py --mode pilot [--workers 4]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import deque  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

import run_nl_acquisition_factorial as F  # noqa: E402  (shared J-table cache, NumpyLR, worker globals, stats)

TASK = "nl_component_ablations"
EPS = 0.02
DELTA = 0.05
NOISE_SEED = 42
T_MAX = 600
D_STEER = 2
N_AUDIT = 2
RES_ROOT = WS / "exp" / "results"
CACHE_A = WS / "exp" / "cache" / "nl_ablations"

BASE = {"m": 5, "explore": 0.05, "two": True, "steer": True, "steer_cost": "length+steer", "reuse": True,
        "cert": "exact", "model": "nl", "audit": False}
ARMS = {
    "full": {},
    "no_csteer": {"steer_cost": "length"},
    "single_step": {"two": False, "steer": False},
    "top_m1": {"m": 1},
    "dual_bnb": {"cert": "dual"},
    "no_reuse": {"reuse": False},
    "static": {"model": "static"},
    "audit": {"audit": True},
    # supplementary fresh-start protocol (no ledger reuse; every problem from n0, as in nl_acquisition_factorial):
    # isolates per-problem acquisition, where the reuse stream certifies ~60% of problems at zero cost
    "nr_single_step": {"reuse": False, "two": False, "steer": False},
    "nr_no_csteer": {"reuse": False, "steer_cost": "length"},
    "nr_top_m1": {"reuse": False, "m": 1},
}
KAPPA_ZERO_ARMS = ["full", "single_step", "static", "no_reuse", "nr_single_step"]
MAIN_ARMS = ["full", "no_csteer", "single_step", "top_m1", "dual_bnb", "no_reuse", "static", "audit"]
NR_ARMS = ["nr_single_step", "nr_no_csteer", "nr_top_m1"]


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


# ---------------------------------------------------------------------------------------------- precompute (GPU)
def static_params(params):
    import torch
    N = 3
    p = dict(params)
    p["g"] = torch.ones(N, dtype=torch.float64, device=params["alpha"].device)
    p["g_ret"] = torch.ones(N, dtype=torch.float64, device=params["alpha"].device)
    return p


def precompute(seeds, n_prob, log):
    import torch
    from dsswm.acquire.nl_factor_acq import FactorModel
    from dsswm.exact.nl_contrib import contrib_table
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, make_nl_instance
    from dsswm.streams.utilities import Utility
    t0 = time.perf_counter()
    jt = F.precompute(seeds, n_prob, log)                 # NL-class J tables (shared nl_factorial cache)
    CACHE_A.mkdir(parents=True, exist_ok=True)
    grid = DEFAULT_NL_S_GRID
    ncl = NLClass(grid)
    params = ncl.torch_params()
    sp = static_params(params)
    timing = {"nl_jtables": jt, "static_s": 0.0, "contrib_s": 0.0, "contrib_max_err": 0.0, "static_vs_nl_maxdiff": []}
    prop = None
    for s in seeds:
        inst = make_nl_instance(s, noise_seed=NOISE_SEED, grid=grid, nl_class=ncl, kappa_mode="high")
        if prop is None:
            prop = NLPropagator(grid.L, grid.R, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"],
                                NL_DEFAULTS["rho_ret"])
            if not (CACHE_A / "static_FT.npy").exists():
                LTs, _ = prop.tables(sp)
                LTs = LTs.cpu().numpy()
                np.save(CACHE_A / "static_LT.npy", LTs)
                np.save(CACHE_A / "static_FT.npy", FactorModel(prop).factor_table(LTs))
        for k in range(n_prob):
            q = inst.problems[k]
            c_q = float(np.load(F.CACHE / f"khigh_{q.pid}_cq.npy")[0])
            J = np.load(F.CACHE / f"khigh_{q.pid}_J.npy")
            ps, pc = CACHE_A / f"static_{q.pid}_J.npy", CACHE_A / f"contrib_{q.pid}.npy"
            if not ps.exists():
                t1 = time.perf_counter()
                raw = prop.j_table(sp, q.policies, q.loads0, q.engaged0, q.H, Utility(w=q.utility.w, w_ret=q.utility.w_ret,
                                                                                      c_q=1.0))
                np.save(ps, raw * c_q)          # same public class-max scale as the NL class -> eps in same units
                timing["static_s"] += time.perf_counter() - t1
            if not pc.exists():
                t1 = time.perf_counter()
                C = contrib_table(prop, params, q.policies, q.loads0, q.engaged0, q.H,
                                  Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=c_q))
                timing["contrib_s"] += time.perf_counter() - t1
                timing["contrib_max_err"] = max(timing["contrib_max_err"], float(np.abs(C.sum(-1) - J).max()))
                np.save(pc, C)
        log(f"precompute (static + contrib): instance {s} done")
    torch.cuda.synchronize()
    timing["total_s"] = time.perf_counter() - t0
    return timing


# ---------------------------------------------------------------------------------------------- worker globals
_W: dict = {}


def wglobals():
    if _W:
        return _W
    G = F._globals()
    from dsswm.certify.dual_bnb import block_digits
    prop, fm = G["prop"], G["fm"]
    aspace = prop.aspace
    _W.update(G)
    _W["LT_static"] = np.load(CACHE_A / "static_LT.npy")
    _W["FT_static"] = np.load(CACHE_A / "static_FT.npy")
    _W["digits"] = block_digits(G["ncl"]._radices, G["ncl"].B)
    _W["radices"] = [int(r) for r in G["ncl"]._radices]
    # --- steering graph over load configurations (known deterministic load rule, incentive-free matchings)
    P, N = prop.P, prop.N
    steer_actions = [a for a in range(aspace.n) if not aspace.actions[a].incentives]
    nlc = prop.codec.n_load_codes
    loads_of = np.array([(lc // prop.codec._pow_n) % N for lc in range(nlc)])
    nxt = np.zeros((nlc, len(steer_actions)), dtype=np.int64)
    for lc in range(nlc):
        for ai, a in enumerate(steer_actions):
            m = np.concatenate([aspace.matched_left(a), aspace.matched_right(a)])
            nl = np.where(m, np.minimum(loads_of[lc] + 1, prop.nmax), np.maximum(loads_of[lc] - 1, 0))
            nxt[lc, ai] = int(nl @ prop.codec._pow_n)
    paths = {}
    for src in range(nlc):
        prev = {src: None}
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for ai in range(len(steer_actions)):
                v = int(nxt[u, ai])
                if v not in prev:
                    prev[v] = (u, steer_actions[ai])
                    dq.append(v)
        for tgt, pv in prev.items():
            if tgt == src:
                continue
            seq, cur = [], tgt
            while prev[cur] is not None:
                u, a = prev[cur]
                seq.append((u, a))
                cur = u
            seq.reverse()
            if len(seq) <= D_STEER:
                paths.setdefault(src, []).append((tgt, seq))     # seq: [(load_code_before, action), ...]
    _W["steer_paths"] = paths
    _W["loads_of"] = loads_of
    return _W


# ---------------------------------------------------------------------------------------------- one stream
class Stream:
    def __init__(self, seed, arm, kappa, trace=False):
        W = wglobals()
        self.W, self.prop, self.fm, self.ncl = W, W["prop"], W["fm"], W["ncl"]
        self.seed, self.arm, self.kappa = seed, arm, kappa
        self.cfg = {**BASE, **ARMS[arm]}
        static = self.cfg["model"] == "static"
        self.LTm = W["LT_static"] if static else W["LT"]
        self.FTm = W["FT_static"] if static else W["FT"]
        self.delta_model = DELTA / 2 if self.cfg["audit"] else DELTA
        self.rng = np.random.default_rng([seed, list(ARMS).index(arm), 0 if kappa == "high" else 1])
        self.trace_on = trace
        self._fresh()

    def _fresh(self):
        from dsswm.streams.generator import make_nl_instance
        inst = make_nl_instance(self.seed, noise_seed=NOISE_SEED, grid=self.W["grid"], nl_class=self.ncl,
                                kappa_mode=self.kappa)
        self.inst = inst
        self.h = inst.env.handle()
        self._truth_idx = int(inst.truth["theta_index"])   # evaluation only
        self.lr = F.NumpyLR(self.prop, self.LTm, self.delta_model)
        for o in inst.init_obs:
            self.lr.update(o)
        self._ph = {}

    # ---------------------------------------------------------------- problem setup
    def set_problem(self, pidx):
        self.pidx = pidx
        self.q = self.inst.problems[pidx]
        pid = self.q.pid
        self.Jtrue = np.load(F.CACHE / f"khigh_{pid}_J.npy")          # class J (truth row used for scoring only)
        self.J = np.load(CACHE_A / f"static_{pid}_J.npy") if self.cfg["model"] == "static" else self.Jtrue
        self.C = np.load(CACHE_A / f"contrib_{pid}.npy") if self.cfg["cert"] == "dual" else None
        self.c_q = float(np.load(F.CACHE / f"khigh_{pid}_cq.npy")[0])
        self.K, self.H = self.J.shape[1], self.q.H
        m = min(self.prop.L, self.prop.R)
        self.u_max = self.c_q * (m * float(self.q.utility.w.sum()) + self.q.utility.w_ret * self.prop.P)
        self._trial = {}
        self.n_start = self.h.n_steps

    def new_steps(self):
        return self.h.n_steps - self.n_start

    def state_code(self):
        ld, en = self.h.observable_state()
        return self.prop.codec.encode(ld, en)

    def step(self, a):
        obs = self.h.step(int(a))
        self.lr.update(obs)
        return obs

    # ---------------------------------------------------------------- certificates
    def cert(self):
        t = time.perf_counter()
        mask = self.lr.mask()
        mle = self.lr.mle()
        k_hat = int(np.argmax(self.J[mle]))
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            out = {"r_bar": np.inf, "k_hat": k_hat, "mle": mle, "idx": idx, "reg": np.zeros(0), "mask": mask}
        else:
            Jm = self.J[idx]
            reg = Jm.max(1) - Jm[:, k_hat]
            out = {"r_bar": float(max(reg.max(), 0.0)), "k_hat": k_hat, "mle": mle, "idx": idx, "reg": reg, "mask": mask}
        self.st["exact_s"] += time.perf_counter() - t
        self.st["n_cert_checks"] += 1
        return out

    def dual(self, c):
        from dsswm.certify.dual_bnb import certify_dual
        t = time.perf_counter()
        r = certify_dual(self.C, c["mask"], c["k_hat"], self.W["digits"], self.W["radices"], EPS, n_hubs=1,
                         exact=False)
        dt = time.perf_counter() - t
        self.st["dual_s"] += dt
        self.st["n_dual"] += 1
        self.st["dual_times"].append(dt)
        gap = r["r_bar_ub"] - c["r_bar"]
        self.st["dual_max_gap"] = max(self.st["dual_max_gap"], gap)
        if gap < -1e-9:
            self.st["dual_soundness_violations"] += 1
        return r

    # ---------------------------------------------------------------- DDA
    def ph(self, mle):
        if mle not in self._ph:
            self._ph = {mle: np.exp(self.fm.row_logp(self.FTm[mle]))}
        return self._ph[mle]

    def trial_rows(self, mle):
        if mle in self._trial:
            return self._trial[mle]
        PH, fm, A = self.ph(mle), self.fm, self.fm.A
        out = []
        s0 = self.prop.codec.encode(self.q.loads0, self.q.engaged0)
        for pi in self.q.policies:
            codes, probs = np.array([s0]), np.array([1.0])
            rows_all, w_all = [], []
            for t in range(self.H):
                acts = np.array([pi.act(t, *self.prop.codec.decode(int(c))) for c in codes])
                rows = codes * A + acts
                rows_all.append(rows)
                w_all.append(probs)
                pn = (probs[:, None] * PH[rows]).ravel()
                nx = fm.NEXT[rows].ravel()
                keep = pn > 1e-12
                u, inv = np.unique(nx[keep], return_inverse=True)
                pr = np.zeros(len(u))
                np.add.at(pr, inv, pn[keep])
                codes, probs = u, pr
            r = np.concatenate(rows_all)
            wv = np.concatenate(w_all)
            ur, inv = np.unique(r, return_inverse=True)
            ww = np.zeros(len(ur))
            np.add.at(ww, inv, wv)
            out.append((ur, ww))
        self._trial = {mle: out}
        return out

    def alternatives(self, c):
        m = self.cfg["m"]
        reg, idx = c["reg"], c["idx"]
        cand = np.flatnonzero(reg > EPS)
        if cand.size == 0:
            cand = np.argsort(-reg)[:m]
        order = cand[np.argsort(-reg[cand])][:m]
        alts = idx[order]
        alts = alts[alts != c["mle"]]
        if alts.size == 0:
            alts = idx[np.argsort(-reg)[:1]]
        return alts

    def decide(self, c):
        from scipy.optimize import linprog
        fm, A, cfg = self.fm, self.fm.A, self.cfg
        mle = c["mle"]
        alts = self.alternatives(c)
        need = np.maximum(self.lr.thr - self.lr.log_ratio()[alts], 0.05)
        code = self.state_code()
        ld, en = self.h.observable_state()
        nlc = self.prop.codec.n_load_codes
        ecode = code // nlc
        R1 = code * A + np.arange(A)
        parts = [R1]
        if cfg["two"]:
            P1 = self.ph(mle)[R1]
            nx = fm.NEXT[R1]
            keep = P1 > 1e-12
            S2, inv = np.unique(nx[keep], return_inverse=True)
            P2 = np.zeros((A, len(S2)))
            ai = np.repeat(np.arange(A)[:, None], 64, 1)[keep]
            np.add.at(P2, (ai, inv), P1[keep])
            R2 = (S2[:, None] * A + np.arange(A)[None]).ravel()
            parts.append(R2)
        steer = []
        if cfg["steer"]:
            lc0 = code % nlc
            for tgt, seq in self.W["steer_paths"].get(lc0, []):
                prow = np.array([(lc + nlc * ecode) * A + a for lc, a in seq])     # steering rounds (engagement held)
                trow = (tgt + nlc * ecode) * A + np.arange(A)                     # probe rows at the target
                steer.append((tgt, seq, prow, trow))
                parts += [prow, trow]
        trials = self.trial_rows(mle)
        parts += [r for r, _ in trials]
        U = np.unique(np.concatenate(parts))
        pos = lambda rows: np.searchsorted(U, rows)
        kl = fm.row_kl(self.FTm[mle], self.FTm[alts], rows=U).T / need[None]     # (|U|, m)
        vals, costs, els = [], [], []
        f1 = kl[pos(R1)]
        vals.append(f1); costs += [1.0] * A; els += [("one", a) for a in range(A)]
        if cfg["two"]:
            f2 = kl[pos(R2)].reshape(len(S2), A, -1)
            v2 = f1[:, None] + np.tensordot(P2, f2, axes=(1, 0))
            vals.append(v2.reshape(A * A, -1)); costs += [2.0] * (A * A)
            els += [("two", a1) for a1 in range(A) for _ in range(A)]
        for tgt, seq, prow, trow in steer:
            v = kl[pos(prow)].sum(0)[None] + kl[pos(trow)]
            vals.append(v)
            d = len(seq)
            costs += [1.0 + d if cfg["steer_cost"] == "length+steer" else 1.0] * A
            els += [("steer", (tgt, tuple(a for _, a in seq), a2)) for a2 in range(A)]
        for k, (r, w) in enumerate(trials):
            vals.append(np.tensordot(w, kl[pos(r)], axes=(0, 0))[None]); costs.append(float(self.H))
            els.append(("trial", k))
        V = np.concatenate(vals, 0)
        cost = np.array(costs)
        n, m = V.shape
        cvec = np.zeros(n + 1); cvec[-1] = -1.0
        res = linprog(cvec, A_ub=np.hstack([-V.T, np.ones((m, 1))]), b_ub=np.zeros(m),
                      A_eq=np.concatenate([cost, [0.0]])[None], b_eq=[1.0],
                      bounds=[(0, None)] * n + [(None, None)], method="highs")
        if res.status == 0:
            frac = np.maximum(res.x[:n] * cost, 0)
            x = int(self.rng.choice(n, p=frac / frac.sum())) if frac.sum() > 0 else int(np.argmax(V.min(1) / cost))
        else:
            x = int(np.argmax(V.min(1) / cost))
        return els[x]

    # ---------------------------------------------------------------- execution
    def run_trial(self, k, stop_fn=None):
        pi = self.q.policies[k]
        self.h.reset_to(self.q.loads0, self.q.engaged0)
        ysum = np.zeros(self.H)
        last = None
        for t in range(self.H):
            ld, en = self.h.observable_state()
            last = self.step(pi.act(t, ld, en))
            ysum[t] = sum(o[3] for o in last.outcomes)
            if stop_fn is not None and stop_fn():
                return None, True
            if self.new_steps() >= T_MAX:
                return None, False
        return self.c_q * float(ysum @ self.q.utility.w + self.q.utility.w_ret * sum(last.next_engaged)), False

    def audit(self, c):
        """AGC: on-policy trials of pi_hat and the binding challenger; residuals around the (fixed) theta-hat."""
        k_hat, mle, idx = c["k_hat"], c["mle"], c["idx"]
        Jm = self.J[idx]
        gaps = Jm - Jm[:, k_hat:k_hat + 1]
        gaps[:, k_hat] = -np.inf
        k_ch = int(np.argmax(gaps.max(0)))
        pred = self.J[mle]
        res = {k_hat: [], k_ch: []}
        for _ in range(N_AUDIT):
            for k in (k_hat, k_ch):
                u, _ = self.run_trial(k)
                if u is None:                       # budget exhausted mid-audit
                    return "NEED_DATA", k_ch, None
                res[k].append(u - pred[k])
        npairs = self.K * (self.K - 1) / 2
        rad = self.u_max * math.sqrt(math.log(4 * npairs / (DELTA / 2)) / (2 * N_AUDIT))
        lb = (pred[k_ch] - pred[k_hat]) + (np.mean(res[k_ch]) - np.mean(res[k_hat])) - 2 * rad
        self.st["audit_lb"] = float(lb)
        self.st["audit_rad"] = float(rad)
        return ("MODEL_CONFLICT" if lb > EPS else "CERTIFIED"), k_ch, float(lb)

    def go_problem(self):
        t0 = time.perf_counter()
        cfg = self.cfg
        self.st = {"one": 0, "two": 0, "steer": 0, "trial": 0, "explore": 0, "decisions": 0, "steer_steps": 0,
                   "steer_arrive_fail": 0, "exact_s": 0.0, "n_cert_checks": 0, "dual_s": 0.0, "n_dual": 0,
                   "dual_times": [], "dual_max_gap": -np.inf, "dual_soundness_violations": 0,
                   "compute_unknown_checks": 0, "audit_steps": 0, "dec_s": 0.0}
        size0 = int(self.lr.mask().sum())
        status, trace = None, ([] if self.trace_on else None)
        stop_now = lambda: (lambda c: c["r_bar"] <= EPS or c["idx"].size == 0)(self.cert())
        while True:
            c = self.cert()
            if c["idx"].size == 0:
                status = "MODEL_CONFLICT"
                break
            if c["r_bar"] <= EPS:
                if cfg["cert"] == "dual":
                    d = self.dual(c)
                    if d["r_bar_ub"] <= EPS:
                        status = "CERTIFIED"
                        break
                    self.st["compute_unknown_checks"] += 1
                elif cfg["audit"]:
                    s0 = self.h.n_steps
                    st, k_ch, lb = self.audit(c)
                    self.st["audit_steps"] += self.h.n_steps - s0
                    self.st["audit_challenger"] = k_ch
                    status = st
                    break
                else:
                    status = "CERTIFIED"
                    break
            elif cfg["cert"] == "dual" and self.st["decisions"] % 25 == 0:
                self.dual(c)                                     # wall-clock sample + soundness check
            if self.new_steps() >= T_MAX:
                status = "COMPUTE_UNKNOWN" if (cfg["cert"] == "dual" and c["r_bar"] <= EPS) else "NEED_DATA"
                break
            td = time.perf_counter()
            self.st["decisions"] += 1
            if self.rng.random() < cfg["explore"]:
                el = ("one", int(self.rng.integers(self.fm.A)))
                self.st["explore"] += 1
            else:
                el = self.decide(c)
            self.st["dec_s"] += time.perf_counter() - td
            kind, arg = el
            self.st[kind] += 1
            if trace is not None and len(trace) < 120:
                trace.append({"step": self.new_steps(), "set_size": int(c["idx"].size), "r_bar": c["r_bar"],
                              "k_hat": c["k_hat"], "mle": c["mle"], "element": [kind, str(arg)]})
            if kind in ("one", "two"):
                self.step(arg)
            elif kind == "steer":
                tgt, seq, a2 = arg
                hit = False
                for a in seq:
                    self.step(a)
                    self.st["steer_steps"] += 1
                    if stop_now() or self.new_steps() >= T_MAX:
                        hit = True
                        break
                if not hit:
                    if self.state_code() % self.prop.codec.n_load_codes != tgt:
                        self.st["steer_arrive_fail"] += 1
                    self.step(a2)
            else:
                self.run_trial(arg, stop_now)
        c = self.cert()
        k = c["k_hat"]
        if status == "CERTIFIED" and cfg["audit"]:
            pass
        jt = self.Jtrue[self._truth_idx]
        regret = float(jt.max() - jt[k])
        certified = status == "CERTIFIED"
        dts = self.st.pop("dual_times")
        out = {"instance": self.seed, "kappa": self.kappa, "arm": self.arm, "problem": self.q.pid, "pidx": self.pidx,
               "H": self.H, "K": self.K, "noise_seed": NOISE_SEED, "status": status,
               "new_env_steps": int(self.new_steps()), "censored": not certified, "t_max": T_MAX,
               "certified_policy": int(k), "true_regret": regret, "false_cert": bool(certified and regret > EPS),
               "final_r_bar": float(c["r_bar"]), "set_size_start": size0, "set_size_end": int(c["idx"].size),
               "theta_star_in_set": (bool(c["mask"][self._truth_idx]) if self.cfg["model"] == "nl" else None),
               "zero_cost": bool(certified and self.new_steps() == 0), "wall_clock_s": time.perf_counter() - t0,
               "rollouts": 0, **{k2: (float(v) if isinstance(v, (float, np.floating)) else v) for k2, v in self.st.items()},
               "dual_time_median": float(np.median(dts)) if dts else None}
        if self.cfg["cert"] == "dual" and dts:
            out["dual_projected_every_step_s"] = float(np.median(dts)) * out["n_cert_checks"]
        if trace is not None:
            out["trace"] = trace
        return out

    def run(self, n_prob):
        rows = []
        for p in range(n_prob):
            if p > 0 and not self.cfg["reuse"]:
                self._fresh()
            self.set_problem(p)
            rows.append(self.go_problem())
        return rows


def run_job(job):
    try:
        s = Stream(job["seed"], job["arm"], job["kappa"], trace=job.get("trace", False))
        return {"ok": True, "rows": s.run(job["n_prob"]), **{k: job[k] for k in ("seed", "arm", "kappa")}}
    except Exception as e:  # noqa: BLE001 - never kill the batch
        return {"ok": False, "error": repr(e), "traceback": traceback.format_exc(), **{k: job[k] for k in ("seed", "arm", "kappa")}}


# ---------------------------------------------------------------------------------------------- analysis
def arm_stats(sub):
    cert = [r for r in sub if r["status"] == "CERTIFIED"]
    fc = sum(r["false_cert"] for r in sub)
    steps = np.array([r["new_env_steps"] for r in sub])
    st = {}
    for s in ("CERTIFIED", "NEED_DATA", "COMPUTE_UNKNOWN", "MODEL_CONFLICT"):
        st[s] = int(sum(r["status"] == s for r in sub))
    out = {"n": len(sub), "rmst": F._rmst(sub, T_MAX), "completion": len(cert) / len(sub),
           "median_steps": float(np.median(steps)), "mean_steps": float(steps.mean()),
           "total_steps": int(steps.sum()), "false_cert": int(fc), "fcr": fc / max(len(cert), 1),
           "fcr_cp95": F._cp(fc, max(len(cert), 1)), "status_counts": st,
           "compute_unknown_rate": st["COMPUTE_UNKNOWN"] / len(sub),
           "zero_cost": int(sum(r["zero_cost"] for r in sub)),
           "mean_wall_s": float(np.mean([r["wall_clock_s"] for r in sub])),
           "mean_cert_s": float(np.mean([r["exact_s"] + r["dual_s"] for r in sub])),
           "element_mix": {k: float(np.mean([r[k] for r in sub])) for k in ("one", "two", "steer", "trial", "explore")},
           "mean_steer_steps": float(np.mean([r["steer_steps"] for r in sub])),
           "steer_arrive_fail": int(sum(r["steer_arrive_fail"] for r in sub))}
    ts = [r["theta_star_in_set"] for r in sub if r["theta_star_in_set"] is not None]
    out["theta_star_in_set_rate"] = float(np.mean(ts)) if ts else None
    if any(r["n_dual"] for r in sub):
        dt = [r["dual_time_median"] for r in sub if r["dual_time_median"] is not None]
        out["dual"] = {"n_dual_calls": int(sum(r["n_dual"] for r in sub)),
                       "soundness_violations": int(sum(r["dual_soundness_violations"] for r in sub)),
                       "max_ub_minus_exact": float(max(r["dual_max_gap"] for r in sub if r["n_dual"])),
                       "compute_unknown_checks": int(sum(r["compute_unknown_checks"] for r in sub)),
                       "dual_s_median_per_call": float(np.median(dt)),
                       "exact_s_median_per_call": float(np.median([r["exact_s"] / max(r["n_cert_checks"], 1) for r in sub])),
                       "mean_dual_s_per_problem_actual": float(np.mean([r["dual_s"] for r in sub])),
                       "mean_dual_s_per_problem_projected_every_step": float(np.mean([r.get("dual_projected_every_step_s", 0.0) for r in sub])),
                       "mean_exact_s_per_problem": float(np.mean([r["exact_s"] for r in sub]))}
    if any(r["audit_steps"] for r in sub):
        out["audit"] = {"mean_audit_steps": float(np.mean([r["audit_steps"] for r in sub])),
                        "n_rejections": st["MODEL_CONFLICT"],
                        "audit_lb_median": float(np.median([r["audit_lb"] for r in sub if "audit_lb" in r])),
                        "audit_radius": float(np.median([r["audit_rad"] for r in sub if "audit_rad" in r]))}
    return out


def paired_vs(rows, arm, ref="full", kappa="high"):
    import pandas as pd
    d = pd.DataFrame([{k: r[k] for k in ("instance", "pidx", "arm", "new_env_steps", "censored", "kappa")}
                      for r in rows if r["kappa"] == kappa and r["arm"] in (arm, ref)])
    pv = d.pivot_table(index=["instance", "pidx"], columns="arm", values="new_env_steps")
    pv = pv.dropna()
    a, b = pv[arm].values, pv[ref].values
    by = {i: g for i, g in pv.groupby(level=0)}

    def rmst_ratio(gs):
        g = pd.concat(gs)
        ra = [{"new_env_steps": x, "censored": x >= T_MAX} for x in g[arm]]
        rb = [{"new_env_steps": x, "censored": x >= T_MAX} for x in g[ref]]
        return F._rmst(ra, T_MAX) / max(F._rmst(rb, T_MAX), 1e-9)

    def tot_ratio(gs):
        g = pd.concat(gs)
        return float(g[arm].sum() / max(g[ref].sum(), 1e-9))

    out = {"n_pairs": int(len(pv)), "mean_delta_steps": float(np.mean(a - b)), "median_delta_steps": float(np.median(a - b)),
           "median_ratio_1p": float(np.median((a + 1) / (b + 1))), "frac_arm_fewer": float(np.mean(a < b)),
           "frac_arm_more": float(np.mean(a > b)),
           "rmst_ratio_boot": F._boot(by, rmst_ratio), "total_steps_ratio_boot": F._boot(by, tot_ratio)}
    from scipy.stats import wilcoxon
    try:
        out["wilcoxon_log1p_p"] = float(wilcoxon(np.log1p(a), np.log1p(b)).pvalue)
    except ValueError:
        out["wilcoxon_log1p_p"] = None
    return out


def analyse(rows, timing):
    per = {"kappa_high": {}, "kappa_zero": {}}
    for kap, key in (("high", "kappa_high"), ("zero", "kappa_zero")):
        for arm in ARMS:
            sub = [r for r in rows if r["kappa"] == kap and r["arm"] == arm]
            if sub:
                per[key][arm] = arm_stats(sub)
    vs = {a: paired_vs(rows, a) for a in MAIN_ARMS if a != "full"}
    vs0 = {a: paired_vs(rows, a, kappa="zero") for a in ["single_step", "static", "no_reuse"]}
    nr = {"kappa_high": {a: paired_vs(rows, a, ref="no_reuse") for a in NR_ARMS},
          "kappa_zero": {"nr_single_step": paired_vs(rows, "nr_single_step", ref="no_reuse", kappa="zero")}}
    H = per["kappa_high"]
    Z = per["kappa_zero"]
    # pass criteria (task_plan pilot)
    st = H["static"]
    comp_drop = H["full"]["completion"] - st["completion"]
    c_static = bool(st["fcr"] > 2 * DELTA or comp_drop >= 0.20)
    # multi-step must beat single-step under BOTH protocols (reuse stream: most problems are zero-cost, so its
    # point estimate alone is dominated by a few problems; fresh-start is the sensitive comparison)
    c_multi_reuse = bool(vs["single_step"]["rmst_ratio_boot"][0] > 1.0 and vs["single_step"]["mean_delta_steps"] > 0)
    c_multi_fresh = bool(nr["kappa_high"]["nr_single_step"]["rmst_ratio_boot"][0] > 1.0)
    c_multi = c_multi_reuse and c_multi_fresh
    c_dual = bool(H["dual_bnb"]["dual"]["soundness_violations"] == 0 and H["dual_bnb"]["false_cert"] == 0)
    gap0 = vs0["single_step"]["rmst_ratio_boot"][0]
    gaph = vs["single_step"]["rmst_ratio_boot"][0]
    candc = {"multi_step_beats_single_step_kappa_high": c_multi, "multi_step_point_estimate_reuse_stream": c_multi_reuse,
             "multi_step_point_estimate_fresh_start": c_multi_fresh, "single_over_full_rmst_ratio_kappa_high": gaph,
             "single_over_full_rmst_ratio_kappa_zero": gap0, "static_gap_lt_1.2x": bool(gap0 < 1.2),
             "csteer_matters": bool(vs["no_csteer"]["rmst_ratio_boot"][0] > 1.0),
             "fresh_start": {"single_over_full_rmst_ratio_kappa_high": nr["kappa_high"]["nr_single_step"]["rmst_ratio_boot"],
                             "single_over_full_rmst_ratio_kappa_zero": nr["kappa_zero"]["nr_single_step"]["rmst_ratio_boot"],
                             "no_csteer_over_full_rmst_ratio_kappa_high": nr["kappa_high"]["nr_no_csteer"]["rmst_ratio_boot"],
                             "top_m1_over_full_rmst_ratio_kappa_high": nr["kappa_high"]["nr_top_m1"]["rmst_ratio_boot"]}}
    h4 = {"static_fcr_kappa_high": st["fcr"], "static_fcr_cp95_kappa_high": st["fcr_cp95"],
          "full_completion_kappa_high": H["full"]["completion"], "static_completion_kappa_high": st["completion"],
          "completion_drop_pp": 100 * comp_drop, "supports_dynamics_necessary": c_static,
          "static_fcr_kappa_zero": Z["static"]["fcr"] if "static" in Z else None,
          "static_completion_kappa_zero": Z["static"]["completion"] if "static" in Z else None,
          "static_model_conflict_kappa_high": st["status_counts"]["MODEL_CONFLICT"]}
    flags = []
    for a, v in vs.items():
        if v["rmst_ratio_boot"][0] > 5 or v["rmst_ratio_boot"][0] < 0.2:
            flags.append(f"{a}: RMST ratio vs full {v['rmst_ratio_boot'][0]:.2f} (>5x / <0.2x flag)")
    for a, v in H.items():
        if a != "static" and v["fcr"] > 0:
            flags.append(f"{a}: {v['false_cert']} false certificates at eta=0")
    for a, v in H.items():
        if v["theta_star_in_set_rate"] is not None and v["theta_star_in_set_rate"] < 1.0:
            flags.append(f"{a}: theta* excluded from Theta_t in {(1 - v['theta_star_in_set_rate']):.0%} of problems")
    go = "GO" if (c_static and c_multi and c_dual) else "NO_GO"
    return {"task_id": TASK, "mode": "pilot", "candidate_id": "cand_c", "env": "E1-NL-S", "eta": 0, "eps": EPS,
            "delta": DELTA, "t_max": T_MAX, "noise_seed": NOISE_SEED, "d_steer_max": D_STEER, "n_audit": N_AUDIT,
            "arms": per, "paired_vs_full_kappa_high": vs, "paired_vs_full_kappa_zero": vs0,
            "paired_vs_no_reuse_fresh_start": nr,
            "H4_static_model": h4, "cand_c": candc,
            "pass_criteria": {"static_FCR_gt_2delta_or_completion_drop_ge_20pp": c_static,
                              "multi_step_fewer_interactions_than_single_step": c_multi,
                              "dual_zero_soundness_violations": c_dual},
            "go_no_go": go, "suspicious_flags": flags, "timing": timing}


def table_md(s):
    lines = ["| Variant | kappa | RMST new steps | Delta vs full (mean steps) | RMST ratio vs full [95% CI] | Completion | "
             "FCR (CP95 up) | Wall-clock / problem (s) | Notes |", "|---|---|---|---|---|---|---|---|---|"]
    for key, kap, vs in (("kappa_high", "high", s["paired_vs_full_kappa_high"]),
                         ("kappa_zero", "0", s["paired_vs_full_kappa_zero"])):
        for arm, a in s["arms"][key].items():
            v = vs.get(arm)
            d = "—" if v is None else f"{v['mean_delta_steps']:+.1f}"
            rr = "1 (ref)" if v is None else f"{v['rmst_ratio_boot'][0]:.2f} [{v['rmst_ratio_boot'][1]:.2f}, {v['rmst_ratio_boot'][2]:.2f}]"
            note = []
            if "dual" in a:
                note.append(f"dual UB-exact max {a['dual']['max_ub_minus_exact']:.1e}, viol {a['dual']['soundness_violations']}, "
                            f"dual {a['dual']['dual_s_median_per_call']:.3f}s/call vs exact {a['dual']['exact_s_median_per_call']:.1e}s")
            if "audit" in a:
                note.append(f"audit {a['audit']['mean_audit_steps']:.0f} steps/problem, rejections {a['audit']['n_rejections']}")
            if a["status_counts"]["MODEL_CONFLICT"]:
                note.append(f"MODEL_CONFLICT {a['status_counts']['MODEL_CONFLICT']}")
            if a["compute_unknown_rate"]:
                note.append(f"COMPUTE_UNKNOWN {a['compute_unknown_rate']:.0%}")
            lines.append(f"| {arm} | {kap} | {a['rmst']:.1f} | {d} | {rr} | {a['completion']:.2f} | "
                         f"{a['fcr']:.3f} ({a['fcr_cp95'][1]:.3f}) | {a['mean_wall_s']:.2f} | {'; '.join(note)} |")
    lines += ["", "Supplementary fresh-start protocol (reference = no_reuse, i.e. full JPC restarted from n0 each problem):", "",
              "| Variant | kappa | RMST new steps | Delta vs no_reuse (mean steps) | RMST ratio vs no_reuse [95% CI] | Completion | FCR (CP95 up) | Wall-clock / problem (s) |",
              "|---|---|---|---|---|---|---|---|"]
    for key, kap in (("kappa_high", "high"), ("kappa_zero", "0")):
        for arm, v in s["paired_vs_no_reuse_fresh_start"][key].items():
            a = s["arms"][key][arm]
            lines.append(f"| {arm} | {kap} | {a['rmst']:.1f} | {v['mean_delta_steps']:+.1f} | {v['rmst_ratio_boot'][0]:.2f} "
                         f"[{v['rmst_ratio_boot'][1]:.2f}, {v['rmst_ratio_boot'][2]:.2f}] | {a['completion']:.2f} | "
                         f"{a['fcr']:.3f} ({a['fcr_cp95'][1]:.3f}) | {a['mean_wall_s']:.2f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------- main
def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--problems", type=int, default=10)
    ap.add_argument("--append-arms", default="", help="comma list: only run these (instance, arm) streams and append "
                    "to the existing results.jsonl, then re-analyse everything")
    args = ap.parse_args()
    if args.mode != "pilot":
        raise SystemExit("full mode (32 eval instances x 15 problems x 3 noise seeds) is configured by a later task")
    out_dir = RES_ROOT / "pilots" / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t0 = time.perf_counter()
    seeds = list(range(args.instances))
    import torch
    progress(0, 3, "precompute")
    timing = {"precompute": precompute(seeds, args.problems, log)}
    timing["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
    log(f"precompute done {timing}")
    jobs = [{"seed": s, "arm": a, "kappa": "high", "n_prob": args.problems, "trace": s == 0} for a in ARMS for s in seeds]
    jobs += [{"seed": s, "arm": a, "kappa": "zero", "n_prob": args.problems, "trace": False} for a in KAPPA_ZERO_ARMS
             for s in seeds]
    prev = []
    if args.append_arms:
        keep = set(args.append_arms.split(","))
        prev = [json.loads(x) for x in open(out_dir / "results.jsonl")]
        have = {(r["arm"], r["kappa"]) for r in prev}
        jobs = [j for j in jobs if f"{j['arm']}:{j['kappa']}" in keep and (j["arm"], j["kappa"]) not in have]
        log(f"append mode: {len(prev)} existing rows, {len(jobs)} new jobs")
    # longest arms first for load balance
    jobs.sort(key=lambda j: (j["arm"] not in ("single_step", "static", "top_m1", "no_reuse"), j["seed"]))
    progress(1, 3, "ablation streams", {"jobs": len(jobs)})
    rows, errs, done_n = list(prev), [], 0
    with open(out_dir / "results.jsonl", "a" if prev else "w") as f:
        for r in Parallel(n_jobs=args.workers, return_as="generator_unordered")(delayed(run_job)(j) for j in jobs):
            done_n += 1
            if r["ok"]:
                rows += r["rows"]
                for x in r["rows"]:
                    f.write(json.dumps({k: v for k, v in x.items() if k != "trace"}, default=float) + "\n")
                f.flush()
            else:
                errs.append(r)
                log(f"ERROR {r['arm']} {r['kappa']} seed {r['seed']}: {r['traceback']}")
            if done_n % 5 == 0 or done_n == len(jobs):
                progress(1, 3, "ablation streams", {"jobs_done": done_n, "jobs_total": len(jobs), "errors": len(errs)})
                log(f"streams: {done_n}/{len(jobs)} (errors {len(errs)}), {time.perf_counter() - t0:.0f}s")
    progress(2, 3, "analysis")
    timing["streams_total_s"] = time.perf_counter() - t0
    timing["n_job_errors"] = len(errs)
    summary = analyse(rows, timing)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "ablation_table.md").write_text(table_md(summary) + "\n")
    samples = [r for r in rows if "trace" in r and r["pidx"] < 2]
    (out_dir / "samples" / ("traces_fresh_start.json" if prev else "traces.json")).write_text(json.dumps(samples, indent=1, default=float))
    gp = {"gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if torch.cuda.is_available() else 0,
          "max_batch_size": 13824, "batch_unit": "theta per J / contribution table pass (whole class, chunks of 4096)",
          "vram_used_mb": timing["gpu_peak_mb"], "utilization_pct": None,
          "note": "GPU only for static-class J tables and per-participant contribution tables (main process); the "
                  "sequential acquisition streams are CPU-bound (4 workers x 1 thread). Concurrent with nl_reuse_kappa_high."}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(gp, indent=1))
    log(f"GO/NO-GO {summary['go_no_go']} criteria {summary['pass_criteria']} total {time.perf_counter() - t0:.0f}s")
    progress(3, 3, "done", {"go_no_go": summary["go_no_go"]})
    mark_done("success", f"pilot {summary['go_no_go']}: {summary['pass_criteria']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # always leave a DONE marker
        traceback.print_exc()
        mark_done("failed", repr(e))
        raise
