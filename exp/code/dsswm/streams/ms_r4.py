"""Round-4 multi-step harness (methodology 1.6): MS-H, MS-S, MS-R3, MS-F (+ MS-H-d descriptive caliber).

HARNESS SIDE: builds ground truth, places problems, runs one stream per (instance, method) through the r3 reuse
switch (`full` arm) and scores it. Learners (dsswm.baselines.*) only receive dsswm.baselines.ms_common.MSPublic +
an EnvHandle; they never import this module.

All layers keep the r3 constants: eps = 0.02, delta = 0.05, T_max = 6000, tau = 3000 (primary truncation),
the ORIGINAL horizons H in {6, 8} (asserted, never shortened), n0 = 20 initial rounds (rng [s, 12]),
ONE stream per instance with noise seed 42 (stream index 0).

  MS-H   r3 R1-adv proposal distribution (run_r3_hazard.Placer / run_r3_p5 draw_theta: theta*_d uniform on the G_1
         hull, rng [s, 51, d]; pool j < 60 of the r3 gap-quota pool; mu_flip by M = 128 MC points, rng [s, 52, d0, j];
         draws scanned in batches of 10, exactly as r3) with the extra TRUTH-LABEL acceptance
             mu_flip >= 1/M  AND  eta_near / eps >= 1,
         eta_near = J*(best) - J*(argmax_pi J_{nearest grid point}(pi)) (design-free misspecification of the nearest
         grid point; the r3 FCR = 0.12 ante1/post2 caliber). Proposal = one (draw, pool problem) pair, at most 4000
         per instance. Stage 1 (<= 3000 proposals = 50 whole draws): the r3 scan unchanged; the instance uses the FIRST
         draw whose 60-pool supplies 15 accepted problems (first 15 in pool order). Stage 2 (only if stage 1 did not
         fill): the pool of the best-filled stage-1 draw is extended (j = 60, 61, ... from the same i.i.d. pool
         generator candidate_problem(s, j), same MC rng rule) until 15 are accepted or 4000 proposals are used; a
         remaining shortfall is recorded (underfilled = True). The per-pair proposal distribution is the r3 one.
  MS-S   r3 static twin at dose g = 1 (run_r3_static generator unchanged: gap-quota NL-R0 theta* uniform on G_1,
         5 tie / 5 near / 5 clear problems); r3 JPC learner class = parameter-matched static class ('static');
         the dynamic G_1 class is public too ('dyn_G1', always-dynamic JPC / FCC).
  MS-R3  r3 R1-adv ORIGINAL stratification (frozen v3-lock mu_flip ante-bin edges, 5/5/5, no eta filter).
  MS-F   held-out family m1r_2eps (run_r3_heldout_misspec: replica synergy pattern, strength calibrated to
         eta_min = 2 eps, truth J by the independent replica).
  MS-H-d descriptive: d = |Delta_true| / eps (top-1 minus top-2 true value among the problem's candidate policies);
         rows carry d and in_d_band = d in [0.3, 1.5].
Stream order (MS-H / MS-R3): placed problems sorted by pool index, permuted with rng [s, 45, 0] (r3 stream 0).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EPS, DELTA = 0.02, 0.05
TMAX = 6000
TAU = 3000
N0 = 20
NOISE = 42
STREAM = 0
Q = 15
MC = 128
N_POOL = 60
DRAW_BATCH = 10
MAX_PROPOSALS = 4000
STAGE1_PROPOSALS = 3000                  # stage-1 budget (whole draws of 60); the rest is the stage-2 reserve
TOP_M = 5
H_ALLOWED = (6, 8)
LAYERS = ("MS-H", "MS-S", "MS-R3", "MS-F")
D_BAND = (0.3, 1.5)

CODE = Path(__file__).resolve().parents[2]            # exp/code
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
WS = CODE.parents[1]
CACHE = WS / "exp" / "cache"
CACHE_DYN = CACHE / "jtables_r3"
CACHE_STA = CACHE / "jtables_r3_static"


def gap_layer(g):
    return "tie" if g < EPS else ("near" if g < 2 * EPS else "clear")


def top2(J):
    s = np.sort(np.asarray(J, float))
    return float(s[-1] - s[-2]) if len(s) > 1 else float("inf")


# ============================================================================================ placement (GPU, main)
class MSPlacer:
    """r3 hazard Placer + the MS-H truth-label filter. GPU only for class J tables (cached) and truth MC."""

    def __init__(self, edges_mu):
        import run_r3_hazard as RH
        RH.EPS = EPS
        self.pl = RH.Placer(edges_mu)
        self.edges_mu = list(edges_mu)

    def place_msh(self, s, caliber: str = "eta"):
        """caliber 'eta' (primary MS-H): mu_flip >= 1/M AND eta_near / eps >= 1. caliber 'muflip_bin2' (G-haz retry,
        methodology §3): the r3 mu_flip primary caliber, ex-ante bin 2 only (mu_flip >= frozen upper edge), no eta
        filter; same draws, pools, MC rng and two-stage scan."""
        if caliber not in ("eta", "muflip_bin2"):
            raise ValueError(caliber)
        mu_min = 1.0 / MC if caliber == "eta" else self.edges_mu[1]
        from dsswm.models.grid_ladder import cell_index_of
        from run_r3_p5_hazard_zone_sampler import draw_theta, in_r1_box
        pl = self.pl
        best, n_prop, n_mu, n_acc = None, 0, 0, 0
        done = False
        d0 = 0

        def accept(r, i):
            nonlocal n_prop, n_mu, n_acc
            n_prop += 1
            if float(r["mu_flip"][i]) < mu_min:
                return False
            n_mu += 1
            if caliber == "eta" and float(r["eta_arg_near"][i]) / EPS < 1.0:
                return False
            n_acc += 1
            return True

        def row(r, i):
            return {kk: (r[kk][i].tolist() if kk == "Jt" else float(r[kk][i])) for kk in r}
        # stage 1: the r3 R1-adv scan unchanged (draw batches of 10, pool j < 60)
        while not done and n_prop + N_POOL <= STAGE1_PROPOSALS:
            ds = list(range(d0, d0 + DRAW_BATCH))
            V = np.stack([draw_theta(s, d) for d in ds])
            star = cell_index_of(V, pl.ncl_cpu)
            res = [pl.truth_eval(s, j, V, star, np.random.default_rng([s, 52, d0, j])) for j in range(N_POOL)]
            for i, d in enumerate(ds):
                if n_prop + N_POOL > STAGE1_PROPOSALS:
                    break
                picked = [j for j in range(N_POOL) if accept(res[j], i)][:Q]
                if best is None or len(picked) > len(best["picked"]):
                    best = {"d": d, "d0": d0, "v": V[i], "star": int(star[i]), "picked": list(picked),
                            "rows": {j: row(res[j], i) for j in picked}}
                if len(picked) >= Q:
                    done = True
                    break
            d0 += DRAW_BATCH
        # stage 2: extend the pool of the best-filled draw (j = 60, 61, ...; same i.i.d. pool generator, same
        # MC rng rule [s, 52, d0, j]) until 15 accepted or MAX_PROPOSALS proposals in total
        stage2 = 0
        j = N_POOL
        V1 = best["v"][None]
        st1 = np.array([best["star"]])
        while len(best["picked"]) < Q and n_prop < MAX_PROPOSALS:
            r = pl.truth_eval(s, j, V1, st1, np.random.default_rng([s, 52, best["d0"], j]))
            stage2 += 1
            if accept(r, 0):
                best["picked"].append(j)
                best["rows"][j] = row(r, 0)
            j += 1
        chosen = sorted(best["picked"])
        per = [self._per(jj, best["rows"][jj]) for jj in chosen]
        return {"layer": "MS-H", "caliber": caliber, "draw": best["d"], "draws_scanned": d0, "fill": len(chosen),
                "underfilled": len(chosen) < Q, "theta": best["v"].tolist(), "star": best["star"],
                "in_r1_box": bool(in_r1_box(best["v"])), "problems": chosen, "per_problem": per,
                "proposals": n_prop, "proposals_stage2": stage2, "pool_max_j": int(max(chosen)) if chosen else None,
                "proposals_mu_ok": n_mu, "proposals_accepted": n_acc,
                "acceptance_rate": n_acc / n_prop if n_prop else None}

    def place_msr3(self, s):
        r = self.pl.place_r1adv(s)
        r["layer"] = "MS-R3"
        return r

    @staticmethod
    def _per(j, r):
        return {"pool_j": j, "true_gap": r["true_gap"], "gap_layer": gap_layer(r["true_gap"]), "mu_flip": r["mu_flip"],
                "mu_flip_half": r["mu_flip_half"], "mu_flip_arg": r["mu_flip_arg"],
                "eta_arg_near_exante": r["eta_arg_near"], "J_true": r["Jt"]}

    def drop_seed(self, s):
        self.pl.drop_seed(s)


def prepare_quota(seeds, need_static=True):
    """GPU: gap-quota selection (G_1 class tables, cached) + static-class tables of the selected problems (cached).
    Returns {seed: {"quota_fail": bool, ...}}."""
    import fcntl
    import torch
    from dsswm.exact.nl_propagate import NLPropagator
    from dsswm.models.nl_class import NLClass
    from dsswm.streams.gap_quota import QuotaBuilder
    from dsswm.streams.generator import DEFAULT_NL_S_GRID
    from dsswm.streams.utilities import Utility
    from run_r3_p4_t0_mechanism_gate import static_np_params
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ncl = NLClass(DEFAULT_NL_S_GRID, device=dev)
    b0 = QuotaBuilder(seeds[0], ncl, None, CACHE_DYN)
    propg = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device=dev)
    propc = NLPropagator(2, 2, 2, b0.aspace, 1.0, 0.3, device="cpu")
    Ps = {k: torch.as_tensor(v, device=dev, dtype=torch.float64) for k, v in static_np_params(ncl.np_params).items()}
    CACHE_STA.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in seeds:
        t0 = time.perf_counter()
        b = QuotaBuilder(s, ncl, propg, CACHE_DYN, prop_cpu=propc)
        sel = b.select("quota")
        out[s] = {"quota_fail": bool(sel["quota_fail"]), "n_draws": sel["n_draws"], "layer_counts": sel["layer_counts"]}
        if sel["quota_fail"] or not need_static:
            out[s]["sec"] = time.perf_counter() - t0
            continue
        with open(CACHE / ".r3_static_jt.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for q in b.stream(STREAM, "quota").problems[:Q]:
                path = CACHE_STA / f"{q.pid}_raw.npy"
                if path.exists():
                    continue
                raw = propg.j_table(Ps, q.policies, q.loads0, q.engaged0, q.H,
                                    Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
                tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npy")
                np.save(tmp, raw)
                os.replace(tmp, path)
            fcntl.flock(lk, fcntl.LOCK_UN)
        out[s]["sec"] = time.perf_counter() - t0
    del ncl, propg, Ps
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


# ============================================================================================ instances (CPU worker)
_W: dict = {}


def ctx():
    if "prop" not in _W:
        import torch
        torch.set_num_threads(1)
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import make_env, nl_r0_truth
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        from run_r3_p4_t0_mechanism_gate import StaticNLClass
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        aspace = make_env(750, nl_r0_truth(750), 42).aspace
        prop = NLPropagator(2, 2, 2, aspace, 1.0, 0.3, device="cpu")
        _W.update(ncl=ncl, aspace=aspace, prop=prop,
                  pub={"dyn_G1": PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA, top_m=TOP_M)},
                  static_cls=lambda: StaticNLClass(ncl))
    return _W


def class_pub(name):
    W = ctx()
    if name not in W["pub"]:
        from dsswm.baselines.switched_nl import PublicNL
        if name != "static":
            raise ValueError(name)
        W["pub"][name] = PublicNL(W["prop"], W["static_cls"](), [], [], 1.0, 2, EPS, DELTA, top_m=TOP_M)
    return W["pub"][name]


@dataclass
class MSInstance:
    layer: str
    seed: int
    problems: list
    J_true: list
    meta: list
    learner_tables: dict          # class name -> list of public class J tables
    default_class: str
    make_platform: object         # () -> (env, init_obs)   (CRN: identical platform + n0 for every method)
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        for q in self.problems:
            assert int(q.H) in H_ALLOWED, f"horizon {q.H} not in the r3 horizons {H_ALLOWED} (never shortened)"


def _hazard_instance(seed, plc, layer):
    from dsswm.streams.gap_quota import make_env
    from dsswm.streams.generator import _initial_data
    from run_r3_p5_hazard_zone_sampler import pool_problem, truth_dict
    W = ctx()
    order = list(np.random.default_rng([seed, 45, STREAM]).permutation(len(plc["problems"])))
    per = [plc["per_problem"][i] for i in order]
    pool = [pool_problem(seed, r["pool_j"], W["aspace"]) for r in per]
    problems, Js = [x[0] for x in pool], [x[1] for x in pool]
    v = np.asarray(plc["theta"], float)

    def platform():
        env = make_env(seed, truth_dict(v), NOISE)
        init = _initial_data(env, N0, np.random.default_rng([seed, 12]))
        return env, init
    meta = [{"pool_j": r["pool_j"], "mu_flip": r["mu_flip"], "eta_near": r["eta_arg_near_exante"],
             "eta_near_over_eps": r["eta_arg_near_exante"] / EPS} for r in per]
    info = {k: plc.get(k) for k in ("draw", "draws_scanned", "fill", "fill_per_bin", "underfilled", "star",
                                    "in_r1_box", "proposals", "proposals_accepted", "acceptance_rate")}
    return MSInstance(layer, seed, problems, [np.asarray(r["J_true"], float) for r in per], meta, {"dyn_G1": Js},
                      "dyn_G1", platform, info)


def build_msh(seed, plc):
    return _hazard_instance(seed, plc, "MS-H")


def build_msr3(seed, plc):
    return _hazard_instance(seed, plc, "MS-R3")


def _quota_stream(seed):
    from dsswm.streams.gap_quota import QuotaBuilder
    W = ctx()
    b = QuotaBuilder(seed, W["ncl"], None, CACHE_DYN, prop_cpu=W["prop"])
    return b, b.stream(STREAM, "quota", n_problems=Q)


def build_mss(seed):
    from dsswm.streams.gap_quota import make_env
    from dsswm.streams.generator import _initial_data
    _, st = _quota_stream(seed)
    assert st.noise_seed == NOISE, st.noise_seed
    truth = dict(st.truth)                                  # dose g = 1: NL-R0 itself (run_r3_static.dosed_truth)

    def platform():
        env = make_env(seed, truth, st.noise_seed)
        init = _initial_data(env, N0, np.random.default_rng([seed, 12]))
        return env, init
    J_sta = [np.load(CACHE_STA / f"{q.pid}_raw.npy") * q.utility.c_q for q in st.problems]
    J_true = [np.asarray(J[st.theta_index], float) for J in st.J]
    meta = [{"gap_layer_r0": h["gap_layer"]} for h in st.harness]
    return MSInstance("MS-S", seed, list(st.problems), J_true, meta, {"static": J_sta, "dyn_G1": list(st.J)},
                      "static", platform, {"theta_index": int(st.theta_index), "dose": 1.0})


def build_msf(seed, family="m1r_2eps"):
    from dsswm.streams.generator import NL_DEFAULTS, _initial_data
    import run_r3_heldout_misspec as HM
    HM.EPS = EPS
    W = ctx()
    b, st = _quota_stream(seed)
    fn, target, hi0, s_max, fmeta = HM.family_spec(family, seed, st.truth)
    calib, plans = HM.calibrate(W["prop"], st, fn, target, hi0, s_max)
    tp = fn(calib["strength"])
    J_rep = HM.replica_J(tp, st.problems, b.aspace)
    J_main = HM.true_J_main(W["prop"], st.problems, plans, tp)
    calib["replica_vs_main_max_abs_diff"] = max(float(np.max(np.abs(a - c))) for a, c in zip(J_main, J_rep))

    def platform():
        env = HM.make_platform(seed, NOISE, tp)
        init = _initial_data(env, NL_DEFAULTS["n0"], np.random.default_rng([seed, 12]))
        return env, init
    meta = [{"gap_layer_r0": h["gap_layer"]} for h in st.harness]
    return MSInstance("MS-F", seed, list(st.problems), [np.asarray(x, float) for x in J_rep], meta,
                      {"dyn_G1": list(st.J)}, "dyn_G1", platform, {"family": family, "calibration": calib})


# ============================================================================================ public view + run
def public_of(inst: MSInstance):
    from dsswm.baselines.ms_common import MSPublic
    classes = {name: class_pub(name).with_problems(inst.problems, J) for name, J in inst.learner_tables.items()}
    return MSPublic(inst.problems, classes, EPS, DELTA, rho_ret=0.3, c_known=1.0, nmax=2,
                    default_class=inst.default_class, layer=inst.layer)


def make_method(name: str, pub):
    from dsswm.baselines import dfsep_ledger, guard, mislid_ms, ms_common, perp_fact, policycert_tab
    if name == "JPC":
        return ms_common.JPC(pub)
    if name == "always-dynamic-JPC":
        return ms_common.AlwaysDynamicJPC(pub)
    if name.startswith("Guard_c"):
        return guard.Guard(pub, float(name[len("Guard_c"):]))
    if name == "MisLid-ms":
        return mislid_ms.MisLidMS(pub)
    if name == "MisLid-ms-k2":
        return mislid_ms.MisLidMS(pub, infl_mult=2.0)
    if name == "MisLid-ms-lcb":
        return mislid_ms.MisLidMS(pub, eta_mode="lcb")
    if name == "DF-sep-ledger":
        return dfsep_ledger.DFSepLedger(pub)
    if name == "PolicyCert-tab":
        return policycert_tab.PolicyCertTab(pub)
    if name == "PERP-fact":
        return perp_fact.PERPFact(pub)
    if name == "FCC":
        return ms_common.FCC(pub)
    raise ValueError(name)


METHODS = ("JPC", "Guard_c1.5", "Guard_c2", "Guard_c3", "MisLid-ms", "MisLid-ms-lcb", "DF-sep-ledger", "PolicyCert-tab",
           "PERP-fact", "always-dynamic-JPC", "FCC")
METHOD_CODE = {"JPC": 101, "always-dynamic-JPC": 101, "Guard_c1.5": 111, "Guard_c2": 112, "Guard_c3": 113,
               "MisLid-ms": 121, "MisLid-ms-lcb": 122, "MisLid-ms-k2": 123, "DF-sep-ledger": 131, "PolicyCert-tab": 141, "PERP-fact": 151, "FCC": 161}


class DecisionLog:
    """Learner write-ahead log: one fsync'ed line per problem, written BEFORE the harness reads any truth."""

    def __init__(self, path: Path | None):
        self.path, self.rows = path, []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)

    def write(self, rec):
        rec = {**rec, "logged_at": time.time()}
        self.rows.append(rec)
        if self.path is not None:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())


def run_stream(inst: MSInstance, method: str, tmax: int = TMAX, decision_path: Path | None = None,
               n_problems: int | None = None):
    """One (instance, method) stream through the `full` reuse arm. Returns (rows, stream_info)."""
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations, TAUS
    t_stream = time.perf_counter()
    pub = public_of(inst)
    m = make_method(method, pub)
    env, init = inst.make_platform()
    h = env.handle()
    sw = ReuseSwitch("full", m.make_set_factory(), init)
    m.begin_stream(init)
    code = 101 if method in ("JPC", "always-dynamic-JPC") else METHOD_CODE[method]
    rng = np.random.default_rng([int(inst.seed), NOISE, code])
    dlog = DecisionLog(decision_path)
    keep = []
    nP = len(inst.problems) if n_problems is None else min(n_problems, len(inst.problems))
    for k in range(nP):
        q = inst.problems[k]
        lr = sw.begin_problem(k, q.pid, context={"stream": STREAM, "kind": inst.layer})
        t0 = time.perf_counter()
        res = m.solve(k, sw, lr, h, rng, tmax)
        wall = time.perf_counter() - t0
        billing_ok, berr = True, None
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, berr = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        dlog.write({"instance": inst.seed, "layer": inst.layer, "method": method, "problem": k, "pid": q.pid,
                    "status": res["status"], "pi": res["pi"], "steps": int(res["steps"])})
        keep.append((k, res, wall, billing_ok, berr, int(acc.n_rounds_billed), int(env.n_steps)))
    # ------------------------------------------------------------------ harness scoring (after the stream)
    scored_at = time.time()
    rows = []
    for (k, res, wall, bok, berr, billed, ens), dec in zip(keep, dlog.rows):
        assert dec["logged_at"] <= scored_at and dec["problem"] == k and dec["pi"] == res["pi"]
        Jt = inst.J_true[k]
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        g = top2(Jt)
        d = g / EPS
        row = {"instance": inst.seed, "layer": inst.layer, "method": method, "problem": k, "pid": inst.problems[k].pid,
               "H": int(inst.problems[k].H), "n_policies": len(inst.problems[k].policies), "noise_seed": NOISE,
               "status": res["status"], "certified_policy": pi, "new_env_steps": int(res["steps"]),
               "censored": not cert, "zero_cost": bool(cert and res["steps"] == 0), "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS),
               "correct_cert": bool(cert and regret is not None and regret <= EPS),
               "true_gap": g, "gap_layer": gap_layer(g), "d_desc": d, "in_d_band": bool(D_BAND[0] <= d <= D_BAND[1]),
               "billing_ok": bok, "billing_error": berr, "n_rounds_billed": billed, "env_n_steps": ens,
               "decision_logged_at": dec["logged_at"], "scored_at": scored_at, "wall_clock_s": wall,
               "eps": EPS, "delta": DELTA, "tmax": tmax, **inst.meta[k]}
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        row.update({f"x_{a}": b for a, b in (res.get("extra") or {}).items()
                    if a != "wall_clock_s" and not isinstance(b, (list, dict))})
        rows.append(row)
    info = {"instance": inst.seed, "layer": inst.layer, "method": method, "n_problems": nP,
            "env_steps": int(env.n_steps), "env_resets": int(env.n_resets), "sec": time.perf_counter() - t_stream,
            "write_ahead_ok": all(r["decision_logged_at"] <= r["scored_at"] for r in rows)}
    return rows, info
