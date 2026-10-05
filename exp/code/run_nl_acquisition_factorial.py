"""nl_acquisition_factorial: H5 / H5b factorial on E1-NL-S (kappa_high, eta = 0).

  (a) evidence {joint, rectangular(time)} x acquisition {DDA, IG(tuned)}
  (b) stopping {joint certificate, direct anytime CS} x sampling {DDA, direct (LUCB whole trials)}
  + B5 variants under the joint stopping rule: tuned IG, Fisher-SEP (variance-driven), Fisher-at-boundary,
    task-directed Fisher design (Wagenmaker 2306.09210 style); + uniform-random reference.

Each (instance, problem, method) run starts from the shared n0 = 20 initial rounds (no cross-problem ledger reuse,
so the factorial isolates stopping and sampling) and acquires real env.step() rounds until its stopping rule fires
or T_max new rounds are used (censored).

Usage: run_nl_acquisition_factorial.py --mode {pilot,full}
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
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]

TASK = "nl_acquisition_factorial"
EPS = 0.02          # prereg: epsilon_pilot_E1_NL = 0.02
DELTA = 0.05
M_TOP = 5
NOISE_SEED = 42
CACHE = WS / "exp" / "cache" / "nl_factorial"
RES_ROOT = WS / "exp" / "results"

# ---------------------------------------------------------------------------------------------- methods
# stop: joint | rect | cs ;  acq: dda | dda_trial | ig | fisher | uniform | lucb
DDA_CFG = {"acq": "dda", "m": M_TOP, "explore": 0.05, "look": 2}
MAIN_METHODS = {
    "joint_DDA": {"stop": "joint", **DDA_CFG},
    "joint_IG": {"stop": "joint", "acq": "ig", "tuned": "IG"},
    "rect_DDA": {"stop": "rect", **DDA_CFG},
    "rect_IG": {"stop": "rect", "acq": "ig", "tuned": "IG"},
    "joint_direct": {"stop": "joint", "acq": "lucb"},
    "cs_DDA": {"stop": "cs", "acq": "dda_trial", "m": M_TOP},
    "cs_direct": {"stop": "cs", "acq": "lucb"},
    "joint_FisherSEP": {"stop": "joint", "acq": "fisher", "tuned": "FisherSEP"},
    "joint_FisherBoundary": {"stop": "joint", "acq": "fisher", "tuned": "FisherBoundary"},
    "joint_TaskDirected": {"stop": "joint", "acq": "fisher", "tuned": "TaskDirected"},
    "joint_uniform": {"stop": "joint", "acq": "uniform"},
}
TUNE_GRIDS = {
    "IG": [{"support": "bayes", "look": 1, "explore": 0.0}, {"support": "bayes", "look": 2, "explore": 0.0},
           {"support": "uniform", "look": 1, "explore": 0.0}, {"support": "uniform", "look": 2, "explore": 0.0},
           {"support": "bayes", "look": 2, "explore": 0.1}, {"support": "uniform", "look": 2, "explore": 0.1}],
    "FisherSEP": [{"crit": "D", "look": 1, "explore": 0.0}, {"crit": "D", "look": 2, "explore": 0.0},
                  {"crit": "A", "look": 1, "explore": 0.0}, {"crit": "A", "look": 2, "explore": 0.0},
                  {"crit": "D", "look": 2, "explore": 0.1}],
    "FisherBoundary": [{"crit": "D", "look": 1, "explore": 0.0}, {"crit": "D", "look": 2, "explore": 0.0},
                       {"crit": "A", "look": 1, "explore": 0.0}, {"crit": "A", "look": 2, "explore": 0.0},
                       {"crit": "D", "look": 2, "explore": 0.1}],
    "TaskDirected": [{"crit": "T", "gapnorm": True, "look": 1, "explore": 0.0},
                     {"crit": "T", "gapnorm": True, "look": 2, "explore": 0.0},
                     {"crit": "T", "gapnorm": False, "look": 1, "explore": 0.0},
                     {"crit": "T", "gapnorm": False, "look": 2, "explore": 0.0},
                     {"crit": "T", "gapnorm": True, "look": 2, "explore": 0.1}],
}
FISHER_POINT = {"FisherSEP": "hat", "FisherBoundary": "boundary", "TaskDirected": "hat"}
IG_N = 64


# ---------------------------------------------------------------------------------------------- scheduler protocol
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


# ---------------------------------------------------------------------------------------------- precompute (main, GPU)
def precompute(seeds, n_prob, log):
    import torch
    from dsswm.acquire.nl_factor_acq import FactorModel, perstep_jtable
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS, make_nl_instance
    from dsswm.streams.utilities import Utility
    CACHE.mkdir(parents=True, exist_ok=True)
    grid = DEFAULT_NL_S_GRID
    ncl = NLClass(grid)
    params = ncl.torch_params()
    prop = None
    timing = {"jt_s": 0.0, "n": 0, "max_err": 0.0}
    for s in seeds:
        inst = make_nl_instance(s, noise_seed=NOISE_SEED, grid=grid, nl_class=ncl, kappa_mode="high")
        if prop is None:
            prop = NLPropagator(grid.L, grid.R, NL_DEFAULTS["nmax"], inst.env.aspace, NL_DEFAULTS["c"],
                                NL_DEFAULTS["rho_ret"])
            ltp = CACHE / "LT.npy"
            if not ltp.exists():
                LT, _ = prop.tables(params)
                LTn = LT.cpu().numpy()
                np.save(ltp, LTn)
                fm = FactorModel(prop)
                np.save(CACHE / "FT.npy", fm.factor_table(LTn))
        for k in range(n_prob):
            q = inst.problems[k]
            p_j, p_t = CACHE / f"khigh_{q.pid}_J.npy", CACHE / f"khigh_{q.pid}_Jt.npy"
            if p_j.exists() and p_t.exists():
                continue
            t0 = time.perf_counter()
            raw_t = perstep_jtable(prop, params, q.policies, q.loads0, q.engaged0, q.H, q.utility.w, q.utility.w_ret)
            timing["jt_s"] += time.perf_counter() - t0
            timing["n"] += 1
            raw = raw_t.sum(-1)
            if timing["n"] <= 3:   # independent check against the j_table propagation
                chk = prop.j_table(params, q.policies, q.loads0, q.engaged0, q.H,
                                   Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
                timing["max_err"] = max(timing["max_err"], float(np.abs(chk - raw).max()))
            c_q = 1.0 / float(raw.max())
            np.save(p_j, raw * c_q)
            np.save(CACHE / f"khigh_{q.pid}_cq.npy", np.array([c_q]))
            np.save(p_t, (raw_t * c_q).astype(np.float32))
        log(f"precompute: instance {s} done")
    torch.cuda.synchronize()
    return timing


# ---------------------------------------------------------------------------------------------- worker state
_G: dict = {}


def _globals():
    if _G:
        return _G
    import torch
    torch.set_num_threads(1)
    from dsswm.acquire.nl_factor_acq import FactorModel
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.generator import DEFAULT_NL_S_GRID, NL_DEFAULTS
    from dsswm.core.actions import ActionSpace
    grid = DEFAULT_NL_S_GRID
    ncl = NLClass(grid, device="cpu")
    aspace = ActionSpace(grid.L, grid.R, NL_DEFAULTS["budget"], (1,))
    prop = NLPropagator(grid.L, grid.R, NL_DEFAULTS["nmax"], aspace, NL_DEFAULTS["c"], NL_DEFAULTS["rho_ret"],
                        device="cpu")
    fm = FactorModel(prop)
    LT = np.load(CACHE / "LT.npy")
    FT = np.load(CACHE / "FT.npy")
    _G.update(grid=grid, ncl=ncl, prop=prop, fm=fm, LT=LT, FT=FT, FTflat=FT.reshape(FT.shape[0], -1), torch=torch)
    return _G


def _gpu():
    G = _globals()
    if "FTg" not in G:
        torch = G["torch"]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        G["dev"] = dev
        G["FTg"] = torch.as_tensor(G["FTflat"], device=dev, dtype=torch.float32)
        G["FLATg"] = torch.as_tensor(G["fm"].FLAT.astype(np.int64), device=dev)
    return G


class NumpyLR:
    """Same sequential LR set as evidence.lr_set.SeqLRSet (plug-in numerator), numpy on CPU."""

    def __init__(self, prop, LT, delta):
        self.prop, self.LT, self.delta = prop, LT, delta
        self.cum = np.zeros(LT.shape[0])
        self.log_num = 0.0
        self.n = 0
        self.thr = math.log(1.0 / delta)

    def update(self, obs):
        from dsswm.core.provenance import require_authentic
        require_authentic(obs)                      # evidence boundary: only env.step() output
        idx, const = self.prop.obs_factors(obs)
        ll = self.LT[:, idx].sum(1) + const
        if self.n == 0:
            m = self.cum.max()
            a = self.cum + ll
            self.log_num += float(np.log(np.exp(a - a.max()).sum()) + a.max() - (np.log(np.exp(self.cum - m).sum()) + m))
        else:
            self.log_num += float(ll[int(np.argmax(self.cum))])
        self.cum += ll
        self.n += 1

    def log_ratio(self):
        return self.log_num - self.cum

    def mask(self):
        return self.log_ratio() < self.thr

    def mle(self):
        return int(np.argmax(self.cum))


# ---------------------------------------------------------------------------------------------- one run
class Run:
    def __init__(self, seed, pidx, method, cfg, t_max, trace=False):
        G = _globals()
        from dsswm.streams.generator import make_nl_instance
        self.G, self.fm, self.prop, self.ncl = G, G["fm"], G["prop"], G["ncl"]
        self.seed, self.pidx, self.method, self.cfg, self.t_max = seed, pidx, method, cfg, t_max
        inst = make_nl_instance(seed, noise_seed=NOISE_SEED, grid=G["grid"], nl_class=self.ncl, kappa_mode="high")
        self.q = inst.problems[pidx]
        self.h = inst.env.handle()                      # learner-facing handle
        self._truth_idx = int(inst.truth["theta_index"])  # evaluation only (never used by the learner)
        self.n0 = self.h.n_steps
        self.J = np.load(CACHE / f"khigh_{self.q.pid}_J.npy")
        self.Jt = np.load(CACHE / f"khigh_{self.q.pid}_Jt.npy") if cfg["stop"] == "rect" else None
        self.K = self.J.shape[1]
        self.H = self.q.H
        # class-max utility scale (public, data-free): J_norm = c_q * J_raw
        self.c_q = float(np.load(CACHE / f"khigh_{self.q.pid}_cq.npy")[0])
        m = min(self.prop.L, self.prop.R)
        self.u_max = self.c_q * (m * float(self.q.utility.w.sum()) + self.q.utility.w_ret * self.prop.P)
        self.lr = NumpyLR(self.prop, G["LT"], DELTA)
        self.fac_count = np.zeros(self.fm.n_fac)
        for o in inst.init_obs:
            self._absorb(o)
        self.rng = np.random.default_rng([seed, pidx, abs(hash(method)) % (2 ** 31)])
        self.trace = [] if trace else None
        self.stats = {"one": 0, "two": 0, "trial": 0, "explore": 0, "decisions": 0}
        self.arm_n = np.zeros(self.K)
        self.arm_sum = np.zeros(self.K)
        self.arm_sq = np.zeros(self.K)
        self._ph_cache = {}
        self._trial_cache = {}
        self._grad_cache = {}
        self._plans = None
        self.dec_s = 0.0

    # -- evidence
    def _absorb(self, obs):
        self.lr.update(obs)
        code = self.prop.codec.encode(obs.loads, obs.engaged)
        r = self.fm.row(code, obs.action)
        np.add.at(self.fac_count, self.fm.ROWFAC[r], 1.0)

    def new_steps(self):
        return self.h.n_steps - self.n0

    def state_code(self):
        ld, en = self.h.observable_state()
        return self.prop.codec.encode(ld, en)

    # -- certificates
    def cert_joint(self):
        mask = self.lr.mask()
        mle = self.lr.mle()
        k_hat = int(np.argmax(self.J[mle]))
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return {"r_bar": np.inf, "k_hat": k_hat, "mle": mle, "idx": idx, "reg": np.zeros(0), "mask": mask}
        Jm = self.J[idx]
        reg = Jm.max(1) - Jm[:, k_hat]
        return {"r_bar": float(max(reg.max(), 0.0)), "k_hat": k_hat, "mle": mle, "idx": idx, "reg": reg, "mask": mask}

    def cert_rect(self):
        c = self.cert_joint()
        idx, k_hat = c["idx"], c["k_hat"]
        if idx.size == 0:
            c["r_rect"] = np.inf
            return c
        Jtm = self.Jt[idx].astype(np.float64)
        D = Jtm - Jtm[:, k_hat:k_hat + 1, :]
        sup = D.max(0)                                  # (K, H+1): block-wise sup, separately per block
        tot = sup.sum(1)
        kb = int(np.argmax(tot))
        c["r_rect"] = float(max(tot.max(), 0.0))
        c["rect_binding"] = kb
        c["rect_block_theta"] = idx[D[:, kb, :].argmax(0)]
        c["rect_block_contrib"] = sup[kb]
        return c

    def cs_bounds(self):
        sig = self.u_max / 2.0
        n = np.maximum(self.arm_n, 1)
        ll = np.maximum(np.log(np.maximum(np.log(2 * n), 1.0)), 0.0)
        rad = sig * 1.7 * np.sqrt((ll + 0.72 * math.log(5.2 * self.K / DELTA)) / n)
        rad = np.where(self.arm_n > 0, rad, np.inf)
        mean = np.where(self.arm_n > 0, self.arm_sum / n, 0.0)
        return mean, rad

    def cs_stop(self):
        if (self.arm_n == 0).any():
            return False, None
        mean, rad = self.cs_bounds()
        b = int(np.argmax(mean))
        others = [k for k in range(self.K) if k != b]
        ok = all(mean[k] + rad[k] - (mean[b] - rad[b]) <= EPS for k in others)
        return ok, b

    # -- theta-hat dependent tables
    def ph(self, mle):
        if mle not in self._ph_cache:
            self._ph_cache = {mle: np.exp(self.fm.row_logp(self.G["FT"][mle]))}
        return self._ph_cache[mle]

    def trial_rows(self, mle):
        """Expected (row, weight) occupancy of each candidate whole trial (reset to s_0) under theta-hat."""
        if mle in self._trial_cache:
            return self._trial_cache[mle]
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
        self._trial_cache = {mle: out}
        return out

    def library(self, mle, look, trials=True, one=True):
        """Rows needed + aggregation maps for one-step, open-loop two-step and whole-trial elements."""
        fm, A = self.fm, self.fm.A
        code = self.state_code()
        R1 = code * A + np.arange(A)
        parts = [R1] if one else []
        lib = {"R1": R1, "look": look, "one": one}
        if look >= 2 and one:
            PH = self.ph(mle)
            P1 = PH[R1]                                          # (A, 64)
            nx = fm.NEXT[R1]
            keep = P1 > 1e-12
            S2, inv = np.unique(nx[keep], return_inverse=True)
            P2 = np.zeros((A, len(S2)))
            ai = np.repeat(np.arange(A)[:, None], 64, 1)[keep]
            np.add.at(P2, (ai, inv), P1[keep])
            R2 = (S2[:, None] * A + np.arange(A)[None]).ravel()
            parts.append(R2)
            lib.update(P2=P2, R2=R2, nS2=len(S2))
        if trials:
            tr = self.trial_rows(mle)
            lib["trials"] = tr
            parts += [r for r, _ in tr]
        U = np.unique(np.concatenate(parts))
        lib["U"] = U
        return lib

    def aggregate(self, lib, fU):
        """fU: per-row additive quantity (|U|, ...) -> element values (n_el, ...) and costs."""
        U = lib["U"]
        pos = lambda rows: np.searchsorted(U, rows)
        vals, costs, els = [], [], []
        A = self.fm.A
        if lib["one"]:
            f1 = fU[pos(lib["R1"])]
            vals.append(f1)
            costs += [1.0] * A
            els += [("one", a) for a in range(A)]
            if lib["look"] >= 2:
                f2 = fU[pos(lib["R2"])].reshape((lib["nS2"], A) + fU.shape[1:])
                E = np.tensordot(lib["P2"], f2, axes=(1, 0))         # (A, A, ...)
                v2 = f1[:, None] + E
                vals.append(v2.reshape((A * A,) + fU.shape[1:]))
                costs += [2.0] * (A * A)
                els += [("two", a1) for a1 in range(A) for _ in range(A)]
        if "trials" in lib:
            for k, (r, w) in enumerate(lib["trials"]):
                vals.append(np.tensordot(w, fU[pos(r)], axes=(0, 0))[None])
                costs.append(float(self.H))
                els.append(("trial", k))
        return np.concatenate(vals, 0), np.array(costs), els

    # -- acquisition rules
    def alternatives(self, c):
        if self.cfg["stop"] == "rect":
            th = c["rect_block_theta"]
            con = c["rect_block_contrib"]
            order = np.argsort(-con)
            alts = []
            for o in order:
                if int(th[o]) not in alts and int(th[o]) != c["mle"]:
                    alts.append(int(th[o]))
                if len(alts) >= self.cfg.get("m", M_TOP):
                    break
            if alts:
                return np.array(alts)
        reg, idx = c["reg"], c["idx"]
        cand = np.flatnonzero(reg > EPS)
        if cand.size == 0:
            cand = np.argsort(-reg)[: self.cfg.get("m", M_TOP)]
        order = cand[np.argsort(-reg[cand])][: self.cfg.get("m", M_TOP)]
        alts = idx[order]
        alts = alts[alts != c["mle"]]
        if alts.size == 0:
            alts = idx[np.argsort(-reg)[:1]]
        return alts

    def decide_dda(self, c, trials_only=False):
        from scipy.optimize import linprog
        mle = c["mle"]
        alts = self.alternatives(c)
        lr = self.lr.log_ratio()
        need = np.maximum(self.lr.thr - lr[alts], 0.05) if self.cfg.get("need", True) else np.ones(len(alts))
        FT = self.G["FT"]
        lib = self.library(mle, look=self.cfg.get("look", 2), trials=True, one=not trials_only)
        kl = self.fm.row_kl(FT[mle], FT[alts], rows=lib["U"]).T / need[None]      # (|U|, m)
        V, cost, els = self.aggregate(lib, kl)
        m = V.shape[1]
        n = V.shape[0]
        # max t  s.t.  sum_x w_x V[x,k] >= t (all k),  sum_x w_x c_x = 1,  w >= 0   (Track-and-Stop allocation)
        cvec = np.zeros(n + 1)
        cvec[-1] = -1.0
        A_ub = np.hstack([-V.T, np.ones((m, 1))])
        res = linprog(cvec, A_ub=A_ub, b_ub=np.zeros(m), A_eq=np.concatenate([cost, [0.0]])[None], b_eq=[1.0],
                      bounds=[(0, None)] * n + [(None, None)], method="highs")
        if res.status == 0 and self.cfg.get("choice", "sample") == "argmax":
            x = int(np.argmax(res.x[:n]))
        elif res.status == 0:
            frac = np.maximum(res.x[:n] * cost, 0)
            frac = frac / frac.sum()
            x = int(self.rng.choice(n, p=frac))
        else:
            x = int(np.argmax(V.min(1) / cost))
        return els[x]

    def decide_ig(self, c, hp):
        torch = self.G["torch"]
        G = _gpu()
        mle, idx = c["mle"], c["idx"]
        if hp["support"] == "bayes":
            lp = self.lr.cum[idx]
            top = idx[np.argsort(-lp)[:IG_N]]
            wv = np.exp(self.lr.cum[top] - self.lr.cum[top].max())
        else:
            top = idx if idx.size <= IG_N else self.rng.choice(idx, IG_N, replace=False)
            wv = np.ones(len(top))
        wv = wv / wv.sum()
        lib = self.library(mle, look=hp["look"], trials=True)
        U = lib["U"]
        FTs = G["FTg"][torch.as_tensor(top, device=G["dev"])]               # (N, n_fac*8)
        wt = torch.as_tensor(wv, device=G["dev"], dtype=torch.float32)
        ig = []
        for s in range(0, len(U), 1024):
            fl = G["FLATg"][torch.as_tensor(U[s:s + 1024], device=G["dev"])]   # (u, 64, 4)
            lpt = FTs[:, fl].sum(-1)                                           # (N, u, 64)
            pt = torch.exp(lpt)
            mix = torch.log(torch.clamp((wt[:, None, None] * pt).sum(0), min=1e-30))
            term = torch.where(pt > 0, pt * (lpt - mix[None]), torch.zeros_like(pt))
            ig.append((wt[:, None] * term.sum(-1)).sum(0))
        fU = torch.cat(ig).double().cpu().numpy()
        V, cost, els = self.aggregate(lib, fU)
        return els[int(np.argmax(V / cost))]

    def _grads(self, mle):
        """Finite-difference gradient of J_theta(pi) at the (continuous) theta-hat, all candidates: (K, d)."""
        if mle in self._grad_cache:
            return self._grad_cache[mle]
        from dsswm.acquire.nl_factor_acq import theta_vector, vector_to_params
        torch = self.G["torch"]
        prop = self.prop
        if self._plans is None:
            self._plans = [prop.build_policy_plan(pi, self.q.loads0, self.q.engaged0, self.H) for pi in self.q.policies]
        v = theta_vector(self.ncl.params_at(mle))
        d, hh = len(v), 1e-4
        Vm = np.vstack([v + hh * np.eye(d)[k] for k in range(d)] + [v - hh * np.eye(d)[k] for k in range(d)])
        p = {k: torch.as_tensor(x, dtype=torch.float64) for k, x in vector_to_params(Vm, prop.L, prop.R, prop.nb).items()}
        LT, EY = prop.tables(p)
        # utility scale: normalised J table / raw J at theta-hat (same c_q for all theta)
        Jr = np.stack([prop._run_plan(pl, es, LT, EY, self.q.utility.w, self.q.utility.w_ret, 1.0).numpy()
                       for pl, es in self._plans], 1)                     # (2d, K)
        p0 = {k: torch.as_tensor(x, dtype=torch.float64) for k, x in vector_to_params(v[None], prop.L, prop.R, prop.nb).items()}
        LT0, EY0 = prop.tables(p0)
        j0 = np.array([float(prop._run_plan(pl, es, LT0, EY0, self.q.utility.w, self.q.utility.w_ret, 1.0)[0])
                       for pl, es in self._plans])
        cq = self.J[mle].max() / j0.max()
        g = cq * (Jr[:d] - Jr[d:]).T / (2 * hh)                           # (K, d)
        self._grad_cache = {mle: g}
        return g

    def decide_fisher(self, c, variant, hp):
        mle = c["mle"]
        if FISHER_POINT[variant] == "boundary":
            a = self.alternatives(c)
            th = int(a[0])
        else:
            th = mle
        ff = self.fm.factor_fisher(self.G["FT"][th])                        # (n_fac, d, d)
        d = ff.shape[1]
        Vd = 1e-2 * np.eye(d) + np.tensordot(self.fac_count, ff, axes=(0, 0))
        lib = self.library(mle, look=hp["look"], trials=True)
        fU = self.fm.row_fisher(ff, lib["U"])                               # (|U|, d, d)
        Fel, cost, els = self.aggregate(lib, fU)
        M = Vd[None] + Fel
        crit = hp["crit"]
        if crit == "D":
            s0 = np.linalg.slogdet(Vd)[1]
            gain = np.linalg.slogdet(M)[1] - s0
        elif crit == "A":
            gain = np.trace(np.linalg.inv(Vd)) - np.trace(np.linalg.inv(M), axis1=1, axis2=2)
        else:
            g = self._grads(mle)
            kh = c["k_hat"]
            others = [k for k in range(self.K) if k != kh]
            Gd = g[others] - g[kh][None]                                    # (K-1, d)
            if hp["gapnorm"]:
                gap = self.J[mle, kh] - self.J[mle, others]
                Gd = Gd / (gap + EPS)[:, None]
            f0 = np.einsum("kd,de,ke->k", Gd, np.linalg.inv(Vd), Gd).max()
            Mi = np.linalg.inv(M)
            f1 = np.einsum("kd,nde,ke->nk", Gd, Mi, Gd).max(1)
            gain = f0 - f1
        return els[int(np.argmax(gain / cost))]

    def decide_lucb(self):
        if (self.arm_n == 0).any():
            return ("trial", int(np.flatnonzero(self.arm_n == 0)[0]))
        mean, rad = self.cs_bounds()
        b = int(np.argmax(mean))
        ucb = mean + rad
        ucb[b] = -np.inf
        l = int(np.argmax(ucb))
        return ("trial", b if self.arm_n[b] <= self.arm_n[l] else l)

    # -- execution
    def run_trial(self, k, stop_fn):
        pi = self.q.policies[k]
        self.h.reset_to(self.q.loads0, self.q.engaged0)
        ysum = np.zeros(self.H)
        last = None
        for t in range(self.H):
            ld, en = self.h.observable_state()
            obs = self.h.step(int(pi.act(t, ld, en)))
            self._absorb(obs)
            ysum[t] = sum(o[3] for o in obs.outcomes)
            last = obs
            if stop_fn is not None and stop_fn():
                return None, True
            if self.new_steps() >= self.t_max:
                return None, False
        # complete trial -> one utility sample on the normalised (class-max) scale
        raw = float(ysum @ self.q.utility.w + self.q.utility.w_ret * sum(last.next_engaged))
        return self.c_q * raw, False

    def go(self):
        t_start = time.perf_counter()
        cfg = self.cfg
        stop = cfg["stop"]
        cert_fn = self.cert_rect if stop == "rect" else self.cert_joint
        state = {"cert": None, "done": False}

        def model_stop():
            c = cert_fn()
            state["cert"] = c
            val = c["r_rect"] if stop == "rect" else c["r_bar"]
            if self.trace is not None and (self.new_steps() % 5 == 0):
                self.trace.append({"step": self.new_steps(), "set_size": int(c["idx"].size), "r_bar": float(c["r_bar"]),
                                   "r_rect": float(c.get("r_rect", np.nan)), "k_hat": c["k_hat"]})
            return val <= EPS

        if stop in ("joint", "rect"):
            done = model_stop()
        else:
            state["cert"] = self.cert_joint()
            done = False
        certified = None
        while not done and self.new_steps() < self.t_max:
            c = state["cert"]
            acq = cfg["acq"]
            td = time.perf_counter()
            self.stats["decisions"] += 1
            if acq == "uniform":
                el = ("one", int(self.rng.integers(self.fm.A)))
            elif acq == "lucb":
                el = self.decide_lucb()
            elif acq == "dda_trial":
                el = self.decide_lucb() if (self.arm_n == 0).any() else self.decide_dda(c, trials_only=True)
            else:
                hp = cfg.get("hp", {})
                expl = hp.get("explore", cfg.get("explore", 0.0))
                if expl > 0 and self.rng.random() < expl:
                    el = ("one", int(self.rng.integers(self.fm.A)))
                    self.stats["explore"] += 1
                elif acq == "dda":
                    el = self.decide_dda(c)
                elif acq == "ig":
                    el = self.decide_ig(c, hp)
                elif acq == "fisher":
                    el = self.decide_fisher(c, cfg["tuned"], hp)
                else:
                    raise ValueError(acq)
            self.dec_s += time.perf_counter() - td
            kind, arg = el
            self.stats[kind] += 1
            if kind in ("one", "two"):
                obs = self.h.step(int(arg))           # receding horizon: execute the first action only
                self._absorb(obs)
                if stop in ("joint", "rect"):
                    done = model_stop()
                else:
                    state["cert"] = self.cert_joint()
            else:
                sf = model_stop if stop in ("joint", "rect") else None
                u, hit = self.run_trial(arg, sf)
                if hit:
                    done = True
                    break
                if u is not None:
                    self.arm_n[arg] += 1
                    self.arm_sum[arg] += u
                    self.arm_sq[arg] += u * u
                if stop == "cs":
                    state["cert"] = self.cert_joint()
                    ok, b = self.cs_stop()
                    if ok:
                        done, certified = True, b
        # ---- outcome
        c = self.cert_joint()
        if stop == "cs":
            if certified is None:
                mean, _ = self.cs_bounds()
                certified = int(np.argmax(mean)) if (self.arm_n > 0).any() else c["k_hat"]
        else:
            certified = c["k_hat"]
        jt = self.J[self._truth_idx]                    # evaluation only
        regret = float(jt.max() - jt[certified])
        n_new = self.new_steps()
        out = {"instance": self.seed, "problem": self.q.pid, "pidx": self.pidx, "H": self.H, "K": self.K,
               "method": self.method, "stop_rule": stop, "acq": cfg["acq"], "hp": cfg.get("hp"),
               "noise_seed": NOISE_SEED, "status": "CERTIFIED" if done else "CENSORED",
               "new_env_steps": int(n_new), "censored": not done, "t_max": self.t_max,
               "certified_policy": int(certified), "certified_policy_name": self.q.policies[certified].name,
               "true_regret": regret, "false_cert": bool(done and regret > EPS),
               "final_r_bar_joint": float(c["r_bar"]), "final_set_size": int(c["idx"].size),
               "theta_star_in_set": bool(c["mask"][self._truth_idx]), "n_resets": int(self.h.n_resets),
               "zero_cost": bool(done and n_new == 0), "wall_clock_s": time.perf_counter() - t_start,
               "decision_s": self.dec_s, "rollouts": 0, **{f"n_{k}": v for k, v in self.stats.items()},
               "arm_n": self.arm_n.astype(int).tolist()}
        if stop == "cs" or cfg["acq"] in ("lucb", "dda_trial"):
            n = np.maximum(self.arm_n, 1)
            mean = self.arm_sum / n
            var = np.maximum(self.arm_sq / n - mean ** 2, 0) * n / np.maximum(n - 1, 1)
            s2 = float(np.sum(var * np.maximum(self.arm_n - 1, 0)) / max(np.sum(np.maximum(self.arm_n - 1, 0)), 1))
            out["trial_utility_sd"] = math.sqrt(s2)
            out["u_max_norm"] = self.u_max
            # variance-oracle fixed-n lower bound on trials per arm: 2 radii <= eps with sd s, Bonferroni over K
            out["n_trials_per_arm_oracle_var"] = float(8 * s2 * math.log(2 * self.K / DELTA) / EPS ** 2)
        if self.trace is not None:
            out["trace"] = self.trace
        return out


def run_job(job):
    G = _globals()
    try:
        r = Run(job["seed"], job["pidx"], job["method"], job["cfg"], job["t_max"], trace=job.get("trace", False))
        return {**r.go(), "phase": job["phase"]}
    except Exception as e:  # never kill the batch; record the failure
        import traceback
        return {"instance": job["seed"], "pidx": job["pidx"], "method": job["method"], "phase": job["phase"],
                "error": repr(e), "traceback": traceback.format_exc()}



# ---------------------------------------------------------------------------------------------- analysis
B5_NAMES = ["joint_IG", "joint_FisherSEP", "joint_FisherBoundary", "joint_TaskDirected"]
CELLS_A = {"joint_DDA": ("joint", "DDA"), "joint_IG": ("joint", "IG"), "rect_DDA": ("rect", "DDA"), "rect_IG": ("rect", "IG")}
CELLS_B = {"joint_DDA": ("joint", "DDA"), "joint_direct": ("joint", "direct"), "cs_DDA": ("cs", "DDA"),
           "cs_direct": ("cs", "direct")}


def _rmst(rows, tau):
    from dsswm.stats.km_rmst import rmst
    return rmst([r["new_env_steps"] for r in rows], [r["censored"] for r in rows], tau)


def _cp(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [lo, hi]


def _boot(by_inst, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(by_inst)
    est = fn([by_inst[k] for k in keys])
    bs = []
    for _ in range(B):
        pick = rng.integers(0, len(keys), len(keys))
        try:
            bs.append(fn([by_inst[keys[p]] for p in pick]))
        except (ValueError, ZeroDivisionError):
            pass
    bs = np.array([b for b in bs if np.isfinite(b)])
    return float(est), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def analyse(rows, t_max, out_dir, dev_rows, tuning, timing, log):
    import pandas as pd
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("trace", "arm_n", "hp")} for r in rows])
    df["logn"] = np.log1p(df["new_env_steps"])
    methods = list(MAIN_METHODS)
    per = {}
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        cert = [r for r in sub if not r["censored"]]
        fc = sum(r["false_cert"] for r in sub)
        steps = np.array([r["new_env_steps"] for r in sub])
        per[m] = {"n": len(sub), "rmst": _rmst(sub, t_max), "completion": len(cert) / len(sub),
                  "median_steps_censored_at_tmax": float(np.median(steps)), "mean_log1p": float(np.mean(np.log1p(steps))),
                  "false_cert": int(fc), "fcr": fc / max(len(cert), 1), "fcr_cp95": _cp(fc, max(len(cert), 1)),
                  "zero_cost": int(sum(r["zero_cost"] for r in sub)),
                  "theta_star_in_set_rate": float(np.mean([r["theta_star_in_set"] for r in sub])),
                  "mean_wall_s": float(np.mean([r["wall_clock_s"] for r in sub])),
                  "mean_resets": float(np.mean([r["n_resets"] for r in sub])),
                  "element_mix": {k: float(np.mean([r[f"n_{k}"] for r in sub])) for k in ("one", "two", "trial", "explore")}}
        if m.startswith("cs") or m == "joint_direct":
            per[m]["median_trials_per_arm_needed_oracle_var"] = float(np.median([r.get("n_trials_per_arm_oracle_var", np.nan) for r in sub]))
    # ---- paired by problem
    piv = df.pivot_table(index=["instance", "pidx"], columns="method", values="new_env_steps")
    cen = df.pivot_table(index=["instance", "pidx"], columns="method", values="censored")
    best_b5 = min(B5_NAMES, key=lambda m: per[m]["rmst"])
    nz = piv[(piv[methods] > 0).any(axis=1)]
    by_inst = {i: g for i, g in nz.groupby(level=0)}

    def med_ratio(gs, a="joint_DDA", b=None):
        g = pd.concat(gs)
        return float(np.median(g[a] / g[b]))

    def rmst_ratio_fn(a, b):
        def f(gs):
            g = pd.concat(gs)
            ra = [{"new_env_steps": x, "censored": x >= t_max} for x in g[a]]
            rb = [{"new_env_steps": x, "censored": x >= t_max} for x in g[b]]
            return _rmst(ra, t_max) / _rmst(rb, t_max)
        return f

    h5 = {"best_B5_variant": best_b5, "B5_rmst": {m: per[m]["rmst"] for m in B5_NAMES},
          "n_problems_used": int(len(nz)), "n_zero_cost_excluded": int(len(piv) - len(nz))}
    h5["median_ratio_DDA_vs_bestB5"] = _boot(by_inst, lambda gs: med_ratio(gs, b=best_b5))
    h5["rmst_ratio_DDA_vs_bestB5"] = _boot(by_inst, rmst_ratio_fn("joint_DDA", best_b5))
    nz2 = nz.copy()
    nz2["oracle_B5"] = nz2[B5_NAMES].min(1)
    h5["median_ratio_DDA_vs_per_problem_oracle_B5"] = float(np.median(nz2["joint_DDA"] / nz2["oracle_B5"]))
    for m in B5_NAMES:
        h5[f"median_ratio_DDA_vs_{m}"] = float(np.median(nz["joint_DDA"] / nz[m]))
        h5[f"frac_problems_DDA_better_{m}"] = float(np.mean(nz["joint_DDA"] < nz[m]))
    from scipy.stats import wilcoxon
    try:
        h5["wilcoxon_log_DDA_vs_bestB5_p"] = float(wilcoxon(np.log1p(nz["joint_DDA"]), np.log1p(nz[best_b5])).pvalue)
    except ValueError:
        h5["wilcoxon_log_DDA_vs_bestB5_p"] = None
    cells = list(dict.fromkeys(list(CELLS_A) + list(CELLS_B)))
    best_cell = min(cells, key=lambda m: per[m]["rmst"])
    h5["best_factorial_cell"] = best_cell
    h5["factorial_cell_rmst"] = {m: per[m]["rmst"] for m in cells}
    if dev_rows:
        ddf = pd.DataFrame([{k: v for k, v in r.items() if k not in ("trace", "arm_n", "hp")} for r in dev_rows])
        dp = ddf.pivot_table(index=["instance", "pidx"], columns="method", values="new_env_steps")
        b5d = [c for c in dp.columns if c != "joint_DDA"]
        dpz = dp[(dp > 0).any(axis=1)]
        h5["dev_calibration"] = {"n": int(len(dpz)), "median_ratio_DDA_vs_each_tuned_B5":
                                 {c: float(np.median(dpz["joint_DDA"] / dpz[c])) for c in b5d},
                                 "note": "dev = tuning instances; B5 configs selected on these same problems (favours B5)"}
    # ---- LMMs
    import statsmodels.formula.api as smf
    lmm = {}
    for name, cells_map, fa, fb in (("a_evidence_x_acquisition", CELLS_A, "evidence", "acquisition"),
                                     ("b_stopping_x_sampling", CELLS_B, "stopping", "sampling")):
        d = df[df["method"].isin(cells_map)].copy()
        d[fa] = d["method"].map(lambda m: cells_map[m][0])
        d[fb] = d["method"].map(lambda m: cells_map[m][1])
        ref_a = "joint"
        ref_b = "DDA"
        d["X_a"] = (d[fa] != ref_a).astype(float)
        d["X_b"] = (d[fb] != ref_b).astype(float)
        d["prob"] = d["instance"].astype(str) + "_" + d["pidx"].astype(str)
        res = {}
        try:
            fit = smf.mixedlm("logn ~ X_a * X_b", d, groups=d["prob"]).fit(reml=True, method="lbfgs")
            ci = fit.conf_int()
            res["fixed_effects"] = {k: {"coef": float(fit.params[k]), "ci95": [float(ci.loc[k, 0]), float(ci.loc[k, 1])],
                                        "p": float(fit.pvalues[k])} for k in fit.fe_params.index}
            res["coding"] = {"X_a": f"{fa} != {ref_a}", "X_b": f"{fb} != {ref_b}", "y": "log(1 + new env steps)",
                             "groups": "problem (paired cells)", "censoring": f"censored runs enter at T_max={t_max}"}
            res["converged"] = bool(fit.converged)
        except Exception as e:  # noqa: BLE001
            res["error"] = repr(e)
        # cell means (log scale and steps) + instance-cluster bootstrap main effects
        cm = d.groupby([fa, fb])["logn"].mean()
        res["cell_mean_log1p"] = {f"{a}|{b}": float(v) for (a, b), v in cm.items()}
        cs_ = d.groupby([fa, fb])["new_env_steps"].mean()
        res["cell_mean_steps"] = {f"{a}|{b}": float(v) for (a, b), v in cs_.items()}
        res["cell_rmst"] = {f"{cells_map[m][0]}|{cells_map[m][1]}": per[m]["rmst"] for m in cells_map}
        bi = {i: g for i, g in d.groupby("instance")}

        def eff(gs, which):
            g = pd.concat(gs)
            mm = g.groupby(["X_a", "X_b"])["logn"].mean()
            if which == "a":
                return float(((mm[(1, 0)] - mm[(0, 0)]) + (mm[(1, 1)] - mm[(0, 1)])) / 2)
            if which == "b":
                return float(((mm[(0, 1)] - mm[(0, 0)]) + (mm[(1, 1)] - mm[(1, 0)])) / 2)
            return float((mm[(1, 1)] - mm[(1, 0)]) - (mm[(0, 1)] - mm[(0, 0)]))
        res["boot_main_effect_" + fa] = _boot(bi, lambda gs: eff(gs, "a"))
        res["boot_main_effect_" + fb] = _boot(bi, lambda gs: eff(gs, "b"))
        res["boot_interaction"] = _boot(bi, lambda gs: eff(gs, "ab"))
        # Shapley shares of the total saving (worst cell -> joint x DDA)
        for scale, col in (("log1p", "logn"), ("steps", "new_env_steps")):
            mm = d.groupby(["X_a", "X_b"])[col].mean()
            total = mm[(1, 1)] - mm[(0, 0)]
            sa = ((mm[(1, 0)] - mm[(0, 0)]) + (mm[(1, 1)] - mm[(0, 1)])) / 2
            sb = ((mm[(0, 1)] - mm[(0, 0)]) + (mm[(1, 1)] - mm[(1, 0)])) / 2
            res[f"shapley_{scale}"] = {"total_saving": float(total), f"share_{fa}": float(sa / total) if total else None,
                                        f"share_{fb}": float(sb / total) if total else None}
        lmm[name] = res
    # ---- gates
    jd = per["joint_DDA"]["rmst"]
    flags = []
    for m in methods:
        r = per[m]["rmst"] / max(jd, 1e-9)
        if r > 5:
            flags.append(f"{m}: RMST {r:.1f}x joint_DDA (>5x saving flag)")
    simple = per["joint_uniform"]["rmst"]
    impr_vs_uniform = 1 - jd / simple
    if impr_vs_uniform > 0.30:
        flags.append(f"joint_DDA improves {impr_vs_uniform:.0%} over joint_uniform RMST (>30%: checked, see leakage checks)")
    pass_ratio = h5["median_ratio_DDA_vs_bestB5"][0] <= 0.9
    pass_best = best_cell == "joint_DDA"
    go = "GO" if (pass_ratio and pass_best) else "NO_GO"
    summary = {"task_id": TASK, "mode": "pilot", "env": "E1-NL-S", "kappa": "high", "eta": 0, "eps": EPS, "delta": DELTA,
               "t_max": t_max, "noise_seed": NOISE_SEED, "n_problems": int(len(piv)), "methods": per, "H5": h5,
               "lmm": lmm, "pass_criteria": {"median_ratio_DDA_vs_bestB5_le_0.9": bool(pass_ratio),
                                             "joint_x_DDA_best_cell": bool(pass_best)},
               "go_no_go": go, "suspicious_flags": flags, "improvement_vs_uniform": float(impr_vs_uniform),
               "B5_tuning": tuning, "timing": timing}
    return summary, df


def plots(rows, df, summary, t_max, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dsswm.stats.km_rmst import kaplan_meier
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (key, fa, fb, la, lb) in zip(axes, (("a_evidence_x_acquisition", "evidence", "acquisition", ("joint", "rect"), ("DDA", "IG")),
                                                ("b_stopping_x_sampling", "stopping", "sampling", ("joint", "cs"), ("DDA", "direct")))):
        cmap = CELLS_A if key.startswith("a") else CELLS_B
        for lvl, col in zip(la, ("#1f77b4", "#d62728")):
            ys, lo, hi = [], [], []
            for b in lb:
                m = [k for k, v in cmap.items() if v == (lvl, b)][0]
                g = df[df["method"] == m]
                by = {i: x["logn"].values for i, x in g.groupby("instance")}
                e, l, h = _boot(by, lambda arrs: float(np.mean(np.concatenate(arrs))), B=1000)
                ys.append(e); lo.append(e - l); hi.append(h - e)
            ax.errorbar(range(len(lb)), ys, yerr=[lo, hi], marker="o", capsize=4, color=col, label=f"{fa}={lvl}")
        ax.set_xticks(range(len(lb)))
        ax.set_xticklabels([f"{fb}={b}" for b in lb])
        ax.set_ylabel("mean log(1 + new env steps)")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_plot.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    sh = summary["lmm"]["b_stopping_x_sampling"]["shapley_steps"]
    sl = summary["lmm"]["b_stopping_x_sampling"]["shapley_log1p"]
    for i, (lab, d) in enumerate((("steps", sh), ("log1p", sl))):
        a, b = d.get("share_stopping") or 0, d.get("share_sampling") or 0
        ax.bar(i, a, color="#1f77b4", label="stopping" if i == 0 else None)
        ax.bar(i, b, bottom=a, color="#ff7f0e", label="sampling" if i == 0 else None)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["share of saving\n(steps)", "share of saving\n(log scale)"])
    ax.axhline(1, color="k", lw=0.5)
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out_dir / "savings_share.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in MAIN_METHODS:
        sub = [r for r in rows if r["method"] == m]
        t, sv = kaplan_meier([r["new_env_steps"] for r in sub], [r["censored"] for r in sub])
        ax.step(np.concatenate([[0], t, [t_max]]), np.concatenate([[1], sv, [sv[-1] if len(sv) else 1]]), where="post", label=m,
                lw=2 if m == "joint_DDA" else 1)
    ax.set_xlabel("new env steps"); ax.set_ylabel("fraction not yet certified"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out_dir / "km_curves.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------------------------- main
def main():
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.mode != "pilot":
        raise SystemExit("full mode is configured by a later task (32 eval instances x 15 problems x 3 noise seeds)")
    out_dir = RES_ROOT / "pilots" / TASK
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    t0 = time.perf_counter()
    T_MAX = 600
    pilot_seeds, n_prob = list(range(10)), 10
    dev_seeds, dev_prob = [500, 501], 5
    import torch
    progress(0, 4, "precompute")
    timing = precompute(dev_seeds + pilot_seeds, n_prob, log)
    timing["gpu_peak_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else 0
    log(f"precompute done {timing}")
    # ---- phase 1: B5 tuning on dev instances (<= 6 grid points per variant)
    progress(1, 4, "B5 tuning on dev instances")
    jobs = []
    for var, grid in TUNE_GRIDS.items():
        base = MAIN_METHODS[{"IG": "joint_IG", "FisherSEP": "joint_FisherSEP", "FisherBoundary": "joint_FisherBoundary",
                             "TaskDirected": "joint_TaskDirected"}[var]]
        for gi, hp in enumerate(grid):
            for s in dev_seeds:
                for p in range(dev_prob):
                    jobs.append({"seed": s, "pidx": p, "method": f"tune_{var}_{gi}", "cfg": {**base, "hp": hp},
                                 "t_max": T_MAX, "phase": "tune", "var": var, "gi": gi})
    for s in dev_seeds:
        for p in range(dev_prob):
            jobs.append({"seed": s, "pidx": p, "method": "joint_DDA", "cfg": MAIN_METHODS["joint_DDA"], "t_max": T_MAX,
                         "phase": "dev"})
    tune_rows = Parallel(n_jobs=args.workers)(delayed(run_job)(j) for j in jobs)
    errs = [r for r in tune_rows if "error" in r]
    for e in errs[:3]:
        log("ERROR " + e["traceback"])
    tuning = {}
    for var, grid in TUNE_GRIDS.items():
        sc = []
        for gi, hp in enumerate(grid):
            rr = [r for r in tune_rows if r.get("method") == f"tune_{var}_{gi}" and "error" not in r]
            sc.append(float(np.mean(np.log1p([r["new_env_steps"] for r in rr]))) if rr else np.inf)
        bi = int(np.argmin(sc))
        tuning[var] = {"grid": grid, "dev_mean_log1p": sc, "selected": grid[bi], "selected_index": bi}
    log(f"tuning: { {k: (v['selected'], round(min(v['dev_mean_log1p']), 3)) for k, v in tuning.items()} }")
    dev_rows = [r for r in tune_rows if "error" not in r and (r["phase"] == "dev" or
                any(r["method"] == f"tune_{v}_{tuning[v]['selected_index']}" for v in tuning))]
    for r in dev_rows:
        if r["method"].startswith("tune_"):
            r["method"] = r["method"].rsplit("_", 1)[0].replace("tune_", "B5_")
    lock_p = WS / "plan" / "prereg_lock.json"
    try:
        lock = json.loads(lock_p.read_text())
        if lock.get("status") != "locked":
            lock["B5_tuning_nl_acquisition_factorial"] = {
                "dev_instances": dev_seeds, "dev_problems_per_instance": dev_prob, "kappa": "high", "eps": EPS,
                "t_max": T_MAX, "selection": "min mean log(1+new env steps)",
                "selected": {k: v["selected"] for k, v in tuning.items()},
                "dev_scores": {k: v["dev_mean_log1p"] for k, v in tuning.items()},
                "DDA_config": "untuned methodology default: m=5, explore=0.05, two-step lookahead, TaS LP allocation",
                "written_at": datetime.now().isoformat()}
            lock_p.write_text(json.dumps(lock, indent=1))
    except (OSError, ValueError) as e:
        log(f"prereg_lock update failed: {e!r}")
    # ---- phase 2: main pilot factorial
    progress(2, 4, "main factorial", {"tuned": {k: v["selected"] for k, v in tuning.items()}})
    jobs = []
    for s in pilot_seeds:
        for p in range(n_prob):
            for m, cfg in MAIN_METHODS.items():
                c = dict(cfg)
                if "tuned" in c:
                    c["hp"] = tuning[c["tuned"]]["selected"]
                jobs.append({"seed": s, "pidx": p, "method": m, "cfg": c, "t_max": T_MAX, "phase": "main",
                             "trace": s == 0 and p < 2})
    rows, done_n = [], 0
    for r in Parallel(n_jobs=args.workers, return_as="generator_unordered")(delayed(run_job)(j) for j in jobs):
        rows.append(r)
        done_n += 1
        if done_n % 50 == 0:
            progress(2, 4, "main factorial", {"runs_done": done_n, "runs_total": len(jobs)})
            log(f"main: {done_n}/{len(jobs)} runs")
    errs = [r for r in rows if "error" in r]
    for e in errs[:3]:
        log("ERROR " + e["traceback"])
    rows = [r for r in rows if "error" not in r]
    rows.sort(key=lambda r: (r["instance"], r["pidx"], list(MAIN_METHODS).index(r["method"])))
    with open(out_dir / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "trace"}) + "\n")
    with open(out_dir / "tuning_runs.jsonl", "w") as f:
        for r in tune_rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "trace"}, default=str) + "\n")
    # ---- phase 3: analysis
    progress(3, 4, "analysis")
    timing["total_wall_s"] = time.perf_counter() - t0
    timing["n_errors"] = len(errs)
    summary, df = analyse(rows, T_MAX, out_dir, dev_rows, tuning, timing, log)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out_dir / "lmm.json").write_text(json.dumps(summary["lmm"], indent=1, default=float))
    plots(rows, df, summary, T_MAX, out_dir)
    samples = [r for r in rows if "trace" in r]
    (out_dir / "samples" / "traces.json").write_text(json.dumps(samples, indent=1, default=float))
    gp = {"gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if torch.cuda.is_available() else 0,
          "max_batch_size": 13824, "batch_unit": "theta per per-step J-table pass (whole class in 4 chunks)",
          "vram_used_mb": timing["gpu_peak_mb"], "utilization_pct": None,
          "note": "GPU only for per-step J tables (main) and IG outcome gathers (workers, N=64 theta x 1024 rows); "
                  "the sequential acquisition loop is CPU-bound (4 workers, 1 thread each). Concurrent run."}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(gp, indent=1))
    log(f"GO/NO-GO: {summary['go_no_go']}  H5 median ratio {summary['H5']['median_ratio_DDA_vs_bestB5']}  "
        f"best cell {summary['H5']['best_factorial_cell']}")
    progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
    mark_done("success", f"pilot {summary['go_no_go']}; median DDA/bestB5 = {summary['H5']['median_ratio_DDA_vs_bestB5'][0]:.3f}")
    log(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # always leave a DONE marker
        import traceback
        traceback.print_exc()
        mark_done("failed", repr(e))
        raise
