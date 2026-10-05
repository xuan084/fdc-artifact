"""r3_hazard_[a-b]: R1 / R1-adv decision-flip hazard zone (B-adv 3x3 table, Q1 pooled sample). Methodology 2.4 / 5.3.

Harness side (creates off-grid ground truth, places problems, scores); the learner (JPC / B3 through the reuse switch
`full` arm) only sees public objects: the G_1 class, the class-max normalised class J tables and its own ledger.

Layers (both: off-grid theta*, learner on G_1, UNCORRECTED JPC = SeqLRSet log 1/delta, minimax certificate, DDA,
no inflation, no fallback; B3 full = reference)
  R1      theta* = streams.offgrid.sample_r1_truth(s) (methodology 2.4: "R1 ... sample_r1_truth"); problems = gap quota
          5 tie + 5 near + 5 clear by the TRUE top-2 gap at theta* (exact propagator), first 5 per layer in pool order
          of the round-3 pool gap_quota.candidate_problem(s, j); cap 2000 pool draws per layer -> quota_fail -> next
          reserve seed (gap_quota.fill_instances).
  R1-adv  rejection sampler of r3_p5_hazard_zone_sampler with the FROZEN lock edges: theta*_d ~ uniform on the G_1 hull
          (grid_ladder.BOX), rng [s, 51, d]; pair (theta*_d, q_j), j < N_POOL, accepted iff mu_flip >= 1/M (M = 128 MC
          points in the clipped Voronoi cell of the nearest grid point, rng [s, 52] per draw batch as in P5);
          ex-ante bin = lock.frozen_items.r1adv_bin_edges.mu_flip (primary caliber, lock r1adv_primary_caliber);
          per instance the first draw (draw order) whose pool supplies 5 accepted problems in every bin; the first 5 per
          bin in pool order are placed (15 problems). If no draw fills within MAX_DRAWS, the best-filled draw is used
          and the shortfall is recorded.
  Stream order (both layers): the placed problems sorted by pool index, permuted with rng [s, 45, t]; noise seed
  STREAM_NOISE[t] = 42 / 123 / 456; n0 = 20 initial rounds with rng [s, 12] (identical to P5 for t = 0).
Recorded per (instance, stream, method, layer, problem)
  ex-ante: mu_flip (harness MC, both layers), Lambda_hat_perp on n0 (learner side, max over the two pre-registered
           pairs, with_rem=False; secondary caliber, bins = frozen lambda_n0_S1 edges), ex-ante bins
  ex-post (scoring only): eta_arg_gcirc (g° = KL projection of theta* onto G_1 over the realised design at the stopping
           time), eta_arg_near (nearest grid point, design-free), eta_arg_thetahat (learner's theta_hat argmax),
           post bins eta_arg/eps in [0,.5), [.5,1), [1,inf) for both flip calibers
  predictors.jsonl: learner-side Q-family record (run_r3_p4.learner_record, with_rem=True) at the certification time,
           fsync'ed BEFORE the harness scores the stream (results.jsonl written after the stream).
  JOIN KEY: (instance, stream, method, arm, problem) with method = <JPC|B3>_<R1|R1adv> (layer is part of the label).

Pilot (smoke + timing only, NO scientific readout): dev seeds 750-752, stream 0, both layers x {JPC, B3} (180 runs);
  projects the full wall-clock of one chunk for r3_prereg_lock.
Full: chunk a / b = eval 10400-10431 / 10432-10463 x 3 streams x 15 problems x {R1, R1-adv} x {JPC, B3}
  (requires the v3 lock: dsswm.stats.prereg.assert_locked()).

Usage: run_r3_hazard.py --chunk {a,b} --mode {pilot,full} [--workers 4]
Concurrent run (shares 20 cores + the RTX 4090 with 3 other round-3 tasks): timings are "concurrent" (并发运行).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = HERE.parents[1]
RES_ROOT = WS / "exp" / "results"
CACHE = WS / "exp" / "cache" / "jtables_r3"
EPS, DELTA = 0.02, 0.05
TOP_M = 5
TMAX = 6000
N0 = 20
MC = 128
N_POOL = 60
DRAW_BATCH = 10
MAX_DRAWS = 200
Q_PER_BIN = 5
QUOTA = 5
MAX_POOL_PER_LAYER = 2000
STREAM_NOISE = (42, 123, 456)
POST_EDGES = (0.0, 0.5, 1.0, math.inf)
POST_LABELS = ("[0,0.5)", "[0.5,1)", "[1,inf)")
LAYERS = ("R1", "R1adv")
METHODS = ("JPC", "B3")
CHUNKS = {"a": (10400, 10432), "b": (10432, 10464)}
RESERVE = {"a": 11100, "b": 11125, "pilot": 761}     # R1 quota_fail replacement (unallocated, disjoint per chunk;
#   moved from 10800/10825 by the v3 lock: 10800-10999 is the NL-R0 / controls replacement block)
LOCK_PROJECTION_MIN = 48.139044073804016
CODE_FILES = ["run_r3_hazard.py", "run_r3_p5_hazard_zone_sampler.py", "run_r3_p4_t0_mechanism_gate.py",
              "dsswm/mechanism/leverage.py", "dsswm/mechanism/predictor_log.py", "dsswm/baselines/switched_nl.py",
              "dsswm/evidence/reuse_switch.py", "dsswm/streams/gap_quota.py", "dsswm/streams/offgrid.py",
              "dsswm/certify/eta_loc.py"]
TASK = "r3_hazard_a"                                 # set in main()


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
        other = "failed" if key == "completed" else "completed"
        if TASK in d.get(other, []):
            d[other].remove(TASK)
        d.setdefault("running", {}).pop(TASK, None)
        d.setdefault("timings", {})[TASK] = {"planned_min": planned, "actual_min": int(round(wall_min)),
                                             "start_time": start_iso, "end_time": datetime.now().isoformat(),
                                             "config_snapshot": snapshot}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        fcntl.flock(lf, fcntl.LOCK_UN)


def update_shared_summary(entry, md):
    lockp = RES_ROOT / "pilot_summary.lock"
    with open(lockp, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            pj = RES_ROOT / "pilot_summary.json"
            d = json.loads(pj.read_text()) if pj.exists() else {"tasks": {}}
            d.setdefault("tasks", {})[TASK] = entry
            tmp = pj.with_name(pj.name + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=str))
            os.replace(tmp, pj)
            pm = RES_ROOT / "pilot_summary.md"
            txt = pm.read_text() if pm.exists() else ""
            head = f"## {TASK} "
            if head in txt:
                s = txt.index(head)
                e = txt.find("\n## ", s + 1)
                txt = txt[:s] + md.strip() + "\n" + (txt[e:] if e >= 0 else "")
            else:
                txt = txt.rstrip() + "\n\n" + md.strip() + "\n"
            pm.write_text(txt)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def code_sha():
    out = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in CODE_FILES}
    out["_combined"] = hashlib.sha256("".join(out[f] for f in CODE_FILES).encode()).hexdigest()
    return out


def vstar_of(truth: dict) -> np.ndarray:
    """Vector order of grid_ladder / P5: alpha(L), gamma(L), tauL(L), beta(R), tauR(R), psi, lam."""
    return np.concatenate([np.asarray(truth["alpha"], float), np.asarray(truth["gamma"], float),
                           np.asarray(truth["tauL"], float), np.asarray(truth["beta"], float),
                           np.asarray(truth["tauR"], float), [float(truth["psi"])], [float(truth["lam"])]])


def bin_of(x, edges):
    if x is None or not np.isfinite(x):
        return None
    return int(np.searchsorted(np.asarray(edges, float), x, side="right"))


def post_bin(r):
    for i in range(3):
        if POST_EDGES[i] <= r < POST_EDGES[i + 1]:
            return i
    return 2


def gap_layer(g):
    return "tie" if g < EPS else ("near" if g < 2 * EPS else "clear")


# ============================================================================================ placement (GPU, main)
class Placer:
    """Harness: pool tables (cached, truth-free), exact J at theta*, mu_flip MC; R1 quota and R1-adv rejection."""

    def __init__(self, edges_mu):
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.models.grid_ladder import cell_boxes
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import make_env, nl_r0_truth
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        if self.dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.ncl = NLClass(DEFAULT_NL_S_GRID, device=self.dev)
        self.ncl_cpu = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        self.aspace = make_env(750, nl_r0_truth(750), 42).aspace
        self.prop = NLPropagator(2, 2, 2, self.aspace, 1.0, 0.3, device=self.dev)
        self.params = None
        self.lo, self.hi = cell_boxes(self.ncl_cpu)
        self.edges_mu = edges_mu
        self.t_tab, self.n_tab = 0.0, 0
        CACHE.mkdir(parents=True, exist_ok=True)
        self._pool = {}

    def cand(self, s, j):
        """(problem, class J (class-max normalised), c_q, plans) of pool problem j of seed s (truth-free)."""
        key = (s, j)
        if key in self._pool:
            return self._pool[key]
        from dsswm.certify.eta_loc import build_plans
        from dsswm.streams.gap_quota import candidate_problem
        from dsswm.streams.utilities import Utility
        q = candidate_problem(s, j, self.aspace)
        path = CACHE / f"{q.pid}_raw.npy"
        if path.exists():
            raw = np.load(path)
        else:
            if self.params is None:
                self.params = self.ncl.torch_params()
            t0 = time.perf_counter()
            raw = self.prop.j_table(self.params, q.policies, q.loads0, q.engaged0, q.H,
                                    Utility(w=q.utility.w, w_ret=q.utility.w_ret, c_q=1.0))
            self.t_tab += time.perf_counter() - t0
            self.n_tab += 1
            tmp = path.with_name(path.name + f".tmp{os.getpid()}.npy")
            np.save(tmp, raw)
            os.replace(tmp, path)
        c = 1.0 / float(raw.max())
        out = (q, raw * c, c, build_plans(self.prop, q))
        self._pool[key] = out
        return out

    def drop_seed(self, s):
        self._pool = {k: v for k, v in self._pool.items() if k[0] != s}

    def truth_eval(self, s, j, V, star, mc_rng):
        """Exact J at the rows of V (n, 12) and mu_flip in the nearest cells star (n,)."""
        from dsswm.certify.eta_loc import j_values
        q, Jc, c, plans = self.cand(s, j)
        n = V.shape[0]
        lo, hi = self.lo[star], self.hi[star]
        Vmc = (lo[:, None, :] + (hi - lo)[:, None, :] * mc_rng.random((n, MC, V.shape[1]))).reshape(n * MC, -1)
        Jt = j_values(self.prop, plans, V, q.utility.w, q.utility.w_ret) * c
        Jm = (j_values(self.prop, plans, Vmc, q.utility.w, q.utility.w_ret) * c).reshape(n, MC, -1)
        pig = np.argmax(Jc[star], 1)
        regm = Jm.max(2) - np.take_along_axis(Jm, pig[:, None, None].repeat(MC, 1), 2)[..., 0]
        srt = np.sort(Jt, 1)
        return {"Jt": Jt, "mu_flip": (regm > EPS).mean(1), "mu_flip_half": (regm > EPS / 2).mean(1),
                "mu_flip_arg": (np.argmax(Jm, 2) != pig[:, None]).mean(1),
                "eta_arg_near": Jt.max(1) - Jt[np.arange(n), pig], "true_gap": srt[:, -1] - srt[:, -2]}

    # ---------------------------------------------------------------------------------------- R1
    def place_r1(self, s):
        from dsswm.models.grid_ladder import cell_index_of
        from dsswm.streams.offgrid import sample_r1_truth
        truth = sample_r1_truth(s)
        v = vstar_of(truth)
        star = cell_index_of(v[None], self.ncl_cpu)
        picked = {l: [] for l in ("tie", "near", "clear")}
        used = {l: 0 for l in picked}
        j, fail = 0, False
        recs = {}
        while any(len(p) < QUOTA for p in picked.values()):
            if any(len(picked[l]) < QUOTA and used[l] >= MAX_POOL_PER_LAYER for l in picked):
                fail = True
                break
            te = self.truth_eval(s, j, v[None], star, np.random.default_rng([s, 52, 1, j]))
            g = float(te["true_gap"][0])
            lay = gap_layer(g)
            for l in picked:
                if len(picked[l]) < QUOTA:
                    used[l] += 1
            if len(picked[lay]) < QUOTA:
                picked[lay].append(j)
                recs[j] = {"pool_j": j, "true_gap": g, "gap_layer": lay, "mu_flip": float(te["mu_flip"][0]),
                           "mu_flip_half": float(te["mu_flip_half"][0]), "mu_flip_arg": float(te["mu_flip_arg"][0]),
                           "eta_arg_near_exante": float(te["eta_arg_near"][0]), "J_true": te["Jt"][0].tolist()}
            j += 1
        chosen = sorted(x for l in picked for x in picked[l])
        return {"layer": "R1", "quota_fail": fail, "pool_draws": j, "theta": v.tolist(), "star": int(star[0]),
                "truth_source": "offgrid.sample_r1_truth (R1 sub-box)", "problems": chosen,
                "per_problem": [recs[x] for x in chosen],
                "layer_counts": {l: len(p) for l, p in picked.items()}}

    # ---------------------------------------------------------------------------------------- R1-adv
    def place_r1adv(self, s):
        from dsswm.models.grid_ladder import cell_index_of
        from run_r3_p5_hazard_zone_sampler import draw_theta, in_r1_box
        best, n_acc, n_pairs = None, 0, 0
        for d0 in range(0, MAX_DRAWS, DRAW_BATCH):
            ds = list(range(d0, d0 + DRAW_BATCH))
            V = np.stack([draw_theta(s, d) for d in ds])
            star = cell_index_of(V, self.ncl_cpu)
            res = [self.truth_eval(s, j, V, star, np.random.default_rng([s, 52, d0, j])) for j in range(N_POOL)]
            done = None
            for i, d in enumerate(ds):
                picked = {0: [], 1: [], 2: []}
                for j in range(N_POOL):
                    mu = float(res[j]["mu_flip"][i])
                    n_pairs += 1
                    if mu < 1.0 / MC:
                        continue
                    n_acc += 1
                    b = bin_of(mu, self.edges_mu)
                    if len(picked[b]) < Q_PER_BIN:
                        picked[b].append(j)
                fill = min(len(v) for v in picked.values())
                if best is None or fill > best["fill"]:
                    best = {"fill": fill, "d": d, "v": V[i], "star": int(star[i]), "picked": picked,
                            "rows": {j: {kk: (res[j][kk][i].tolist() if kk == "Jt" else float(res[j][kk][i]))
                                         for kk in res[j]} for b in picked for j in picked[b]}}
                if fill >= Q_PER_BIN:
                    done = True
                    break
            if done:
                break
        chosen = sorted(j for b in best["picked"] for j in best["picked"][b])
        per = []
        for j in chosen:
            r = best["rows"][j]
            per.append({"pool_j": j, "true_gap": r["true_gap"], "gap_layer": gap_layer(r["true_gap"]),
                        "mu_flip": r["mu_flip"], "mu_flip_half": r["mu_flip_half"], "mu_flip_arg": r["mu_flip_arg"],
                        "eta_arg_near_exante": r["eta_arg_near"], "J_true": r["Jt"]})
        return {"layer": "R1adv", "draw": best["d"], "draws_scanned": best["d"] + 1, "fill_per_bin": best["fill"],
                "underfilled": best["fill"] < Q_PER_BIN, "theta": best["v"].tolist(), "star": best["star"],
                "in_r1_box": bool(in_r1_box(best["v"])), "problems": chosen, "per_problem": per,
                "pairs_scanned": n_pairs, "pairs_accepted": n_acc,
                "acceptance_rate_scanned": n_acc / n_pairs if n_pairs else None}


def placements(seeds, edges_mu, out_dir, reserve):
    """GPU stage: per-instance placements for both layers (cached to parts/place_i<s>.json)."""
    from dsswm.streams.gap_quota import fill_instances
    pl = Placer(edges_mu)
    t0 = time.perf_counter()
    per_seed_sec = {}
    cache = {}

    def build(s):
        fn = out_dir / "parts" / f"place_i{s}.json"
        if fn.exists():
            cache[s] = json.loads(fn.read_text())
            return cache[s]
        t1 = time.perf_counter()
        r1 = pl.place_r1(s)
        ra = pl.place_r1adv(s) if not r1["quota_fail"] else None
        d = {"seed": s, "R1": r1, "R1adv": ra}
        per_seed_sec[s] = time.perf_counter() - t1
        tmp = fn.with_name(fn.name + ".tmp")
        tmp.write_text(json.dumps(d))
        os.replace(tmp, fn)
        pl.drop_seed(s)
        cache[s] = d
        log(f"placement i{s}: R1 draws={r1['pool_draws']} fail={r1['quota_fail']} "
            f"R1adv draw={ra and ra['draw']} fill={ra and ra['fill_per_bin']} ({per_seed_sec[s]:.1f}s)")
        return d

    used, repl = fill_instances(seeds, lambda s: not build(s)["R1"]["quota_fail"], reserve_start=reserve)
    vram = torch.cuda.max_memory_allocated() / 2 ** 20 if pl.dev == "cuda" else 0.0
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 20 if pl.dev == "cuda" else 0
    prof = {"gpu_name": torch.cuda.get_device_name(0) if pl.dev == "cuda" else "cpu", "vram_total_mb": tot,
            "max_batch_size": f"j_values chunk 16384 points (MC batch per pool problem = {DRAW_BATCH * MC} points)",
            "vram_used_mb": vram, "utilization_pct": (100.0 * vram / tot) if tot else None,
            "note": "GPU only for class J tables of the pools + exact J / mu_flip MC at theta*; JPC/B3 loop, predictors "
                    "and scoring on CPU (4 workers, OMP=1). Shared 4090 (per-task cap 5 GB) -> low utilisation by "
                    "design; 并发运行"}
    (RES_ROOT / f"{TASK}_gpu_profile.json").write_text(json.dumps(prof))
    info = {"sec": time.perf_counter() - t0, "per_seed_sec": per_seed_sec, "tables_computed": pl.n_tab,
            "table_sec": pl.t_tab, "device": pl.dev, "vram_peak_mb": vram}
    del pl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return used, repl, {s: cache[s] for s in used}, info


# ============================================================================================ worker
_W: dict = {}


def _ctx():
    if "pub" not in _W:
        torch.set_num_threads(1)
        from dsswm.acquire.nl_kl_dda import IncidenceIndex, class_prob_tables
        from dsswm.baselines.switched_nl import PublicNL
        from dsswm.exact.nl_propagate import NLPropagator
        from dsswm.mechanism.leverage import NLLeverage
        from dsswm.models.nl_class import NLClass
        from dsswm.streams.gap_quota import make_env, nl_r0_truth
        from dsswm.streams.generator import DEFAULT_NL_S_GRID
        ncl = NLClass(DEFAULT_NL_S_GRID, device="cpu")
        aspace = make_env(750, nl_r0_truth(750), 42).aspace
        prop = NLPropagator(2, 2, 2, aspace, 1.0, 0.3, device="cpu")
        TH = NLLeverage(prop).theta_matrix(ncl.np_params)
        lev = NLLeverage(prop, theta_grid=TH, eps=EPS)
        pub = PublicNL(prop, ncl, [], [], 1.0, 2, EPS, DELTA, top_m=TOP_M)
        py, pe = class_prob_tables(ncl.np_params, 1.0, 2, aspace.nb)
        _W.update(ncl=ncl, aspace=aspace, prop=prop, TH=TH, lev=lev, pub=pub, py=py, pe=pe,
                  inc=IncidenceIndex(prop.codec, aspace, 2))
    return _W


def jtag(j):
    seed, layer, method, stream = j
    return f"i{seed}_{layer}_{method}_s{stream}"


def job(seed, layer, method, stream, plc, edges_lam, out_dir):
    from dsswm.acquire.nl_kl_dda import kl_features, single_prob_tables
    from dsswm.baselines.switched_nl import solve
    from dsswm.certify.minimax_enum import certify_minimax
    from dsswm.evidence.lr_set import SeqLRSet
    from dsswm.evidence.reuse_switch import BillingError, ReuseSwitch, score_truncations, TAUS
    from dsswm.mechanism.leverage import select_pairs, theta_hat_index
    from dsswm.mechanism.predictor_log import PredictorLog, ResultLog
    from dsswm.streams.gap_quota import make_env
    from dsswm.streams.generator import _initial_data
    from dsswm.streams.r3_harness import METHOD_CODE
    from run_r3_p5_hazard_zone_sampler import pool_problem, truth_dict
    import run_r3_p4_t0_mechanism_gate as P4
    P4.EPS = EPS
    W = _ctx()
    tag = jtag((seed, layer, method, stream))
    mlabel = f"{method}_{layer}"
    parts = out_dir / "parts"
    pp, rp, jp = parts / f"{tag}_pred.jsonl", parts / f"{tag}_res.jsonl", parts / f"{tag}.json"
    for p in (pp, rp):
        p.unlink(missing_ok=True)
    plog, rlog = PredictorLog(pp, fsync=True), ResultLog(rp, fsync=False)
    t_job = time.perf_counter()
    P = plc[layer]
    order = list(np.random.default_rng([seed, 45, stream]).permutation(len(P["problems"])))
    per = [P["per_problem"][i] for i in order]
    js = [r["pool_j"] for r in per]
    pool = [pool_problem(seed, j, W["aspace"]) for j in js]
    problems, Js, cqs = [x[0] for x in pool], [x[1] for x in pool], [x[2] for x in pool]
    pub = W["pub"].with_problems(problems, Js)
    noise = STREAM_NOISE[stream]
    # ------------------------------------------------------------------ harness platform
    v = np.asarray(P["theta"], float)
    env = make_env(seed, truth_dict(v), noise)
    init = _initial_data(env, N0, np.random.default_rng([seed, 12]))
    h = env.handle()
    # ------------------------------------------------------------------ learner: ex-ante Lambda_hat_perp on n0
    t_l = time.perf_counter()
    lam_n0, errs = [], []
    try:
        E = SeqLRSet(W["prop"], pub.LT, DELTA)
        for o in init:
            E.update(o)
        mask0 = E.mask().numpy().astype(bool)
        k0 = theta_hat_index(E.cum.cpu().numpy(), mask0)
        for k, q in enumerate(problems):
            c0 = certify_minimax(pub.Reg[k], mask0, EPS, TOP_M)
            pr0 = W["lev"].predictors(q, list(init), W["TH"][k0], select_pairs(Js[k], mask0, k0, int(c0["pi"])),
                                      float(c0["r_bar"]), EPS, with_rem=False)
            lam_n0.append(float(pr0["S1"]))
    except Exception as e:  # noqa: BLE001
        errs.append({"stage": "lambda_n0", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        lam_n0 = [None] * len(problems)
    t_lam = time.perf_counter() - t_l
    sw = ReuseSwitch("full", pub.make_set_factory(method), init)
    rng = np.random.default_rng([int(seed), int(noise), METHOD_CODE[method]])
    keep = []
    for k, q in enumerate(problems):
        key = {"instance": seed, "stream": stream, "method": mlabel, "arm": "full", "problem": k}
        lr = sw.begin_problem(k, q.pid, context={"stream": stream, "kind": layer})
        t0 = time.perf_counter()
        try:
            res = solve(pub, method, k, sw, lr, h, rng, TMAX)
        except Exception as e:  # noqa: BLE001
            errs.append({**key, "stage": "solve", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            res = {"status": "CRASH", "pi": None, "steps": sw.new_steps, "extra": {}}
        wall = time.perf_counter() - t0
        t1 = time.perf_counter()
        pred_ok, k_hat, pr = False, None, {}
        try:                                                      # certification-time predictors (write-ahead)
            mask = lr.mask().numpy().astype(bool)
            k_hat = theta_hat_index(lr.inner.cum.cpu().numpy(), mask)
            pi_hat, pi_src = res["pi"], "certificate"
            if pi_hat is None and mask.any():
                pi_hat, pi_src = certify_minimax(pub.Reg[k], mask, EPS, TOP_M)["pi"], "minimax_on_mask"
            if pi_hat is None:
                pi_hat, pi_src = int(np.argmax(Js[k][k_hat])), "argmax_theta_hat"
            r_bar = float((res["extra"] or {}).get("r_bar_end", float("nan")))
            pr = P4.learner_record(W["lev"], W["TH"], q, Js[k], list(lr.obs), mask, k_hat, int(pi_hat), r_bar,
                                   with_rem=True)
            plog.write({**key, "base_method": method, "layer": layer, "source": f"{layer}_cert", "pid": q.pid,
                        "status": res["status"], "new_steps": int(res["steps"]), "theta_hat": int(k_hat),
                        "pi_hat": int(pi_hat), "pi_hat_source": pi_src, "set_size": int(mask.sum()),
                        "n_rows_in_set": len(lr.obs), "lambda_n0": lam_n0[k],
                        **{kk: vv for kk, vv in pr.items() if kk != "pairs"},
                        "pairs": [{kk: vv for kk, vv in p.items() if kk != "need_data"} |
                                  {"n_need_data": len(p.get("need_data", []))} for p in pr["pairs"]]})
            pred_ok = True
        except Exception as e:  # noqa: BLE001
            errs.append({**key, "stage": "cert_pred", "error": repr(e), "tb": traceback.format_exc()[-2000:]})
        t_pred = time.perf_counter() - t1
        billing_ok, berr = True, None
        try:
            acc = sw.end_problem(env.n_steps)
        except BillingError as e:
            billing_ok, berr = False, str(e)
            acc = sw.accounts[-1]
            sw._k = sw._pid = None
            sw._new = 0
        # the learner's evidence at the stopping time (kept for the g° projection)
        cnt_pairs = Counter((W["prop"].codec.encode(o.loads, o.engaged), int(o.action)) for o in lr.obs)
        alive_mask = lr.mask().numpy().astype(bool)
        keep.append({"k": k, "res": res, "wall": wall, "t_pred": t_pred, "pred_ok": pred_ok, "k_hat": k_hat,
                     "billing_ok": billing_ok, "billing_error": berr, "billed": int(acc.n_rounds_billed),
                     "replay": int(acc.replay_steps), "cnt": cnt_pairs, "alive_star": bool(alive_mask[P["star"]]),
                     "S1": pr.get("S1"), "S2": pr.get("S2")})
    # ------------------------------------------------------------------ harness scoring (after the stream)
    tpy, tpe = single_prob_tables(env.true_params(), 1.0, 2, W["aspace"].nb)
    Fkl = kl_features(tpy, tpe, W["py"], W["pe"])
    star = P["star"]
    samples = []
    for kr in keep:
        k, res = kr["k"], kr["res"]
        q, meta = problems[k], per[k]
        Jt = np.asarray(meta["J_true"], float)
        Jc = Js[k]
        cnt = np.zeros(W["inc"].nf)
        for (code, a), n in kr["cnt"].items():
            cnt += n * W["inc"].S[code * W["inc"].nA + a]
        gc = int(np.argmin(cnt @ Fkl))
        pi = res["pi"]
        cert = res["status"] == "CERTIFIED"
        regret = float(Jt.max() - Jt[pi]) if pi is not None else None
        e_g = float(Jt.max() - Jt[int(np.argmax(Jc[gc]))])
        e_n = float(Jt.max() - Jt[int(np.argmax(Jc[star]))])
        srt = np.sort(Jt)
        tg = float(srt[-1] - srt[-2])
        lam = lam_n0[k]
        row = {"task": TASK, "instance": seed, "stream": stream, "method": mlabel, "base_method": method,
               "arm": "full", "layer": layer, "problem": k, "pid": q.pid, "pool_j": meta["pool_j"],
               "noise_seed": noise, "theta_draw": P.get("draw"), "status": res["status"],
               "new_env_steps": int(res["steps"]), "replay_steps": kr["replay"], "n_rounds_billed": kr["billed"],
               "billing_ok": kr["billing_ok"], "billing_error": kr["billing_error"], "censored": not cert,
               "zero_cost": bool(cert and res["steps"] == 0), "certified_policy": pi, "true_regret": regret,
               "false_cert": bool(cert and regret is not None and regret > EPS), "true_gap": tg,
               "gap_layer": gap_layer(tg), "set_size": (res["extra"] or {}).get("set_size"),
               "eta_dec": float(np.abs(Jt - Jc[gc]).max()), "eta_dec_near": float(np.abs(Jt - Jc[star]).max()),
               "eta_arg": e_g, "eta_arg_gcirc": e_g, "eta_arg_near": e_n,
               "eta_arg_thetahat": (float(Jt.max() - Jt[int(np.argmax(Jc[kr['k_hat']]))])
                                    if kr["k_hat"] is not None else None),
               "post_bin_gcirc": post_bin(e_g / EPS), "post_bin_near": post_bin(e_n / EPS),
               "gcirc": gc, "nearest": star, "gcirc_eq_nearest": gc == star, "theta_cell_alive": kr["alive_star"],
               "mu_flip": meta["mu_flip"], "mu_flip_half": meta["mu_flip_half"], "mu_flip_arg": meta["mu_flip_arg"],
               "ante_bin_muflip": bin_of(meta["mu_flip"], EDGES["mu_flip"]),
               "accepted_hazard": meta["mu_flip"] >= 1.0 / MC,
               "lambda_n0": lam, "ante_bin_lambda": bin_of(lam, edges_lam) if lam is not None else None,
               "lambda_cert": kr["S1"], "S2_cert": kr["S2"], "predictor_logged": kr["pred_ok"],
               "orth_feasible": None, "cos_overlap": None, "c_q": cqs[k], "eps": EPS, "delta": DELTA, "tmax": TMAX,
               "wall_clock_s": kr["wall"], "predictor_sec": kr["t_pred"],
               "lambda_n0_sec": t_lam / max(len(problems), 1),
               "first_cert_step": (res["extra"] or {}).get("first_cert_step")}
        row.update(score_truncations(int(res["steps"]), cert, TAUS))
        rlog.write(row)
        if len(samples) < 2 or (row["false_cert"] and len(samples) < 5):
            samples.append({"tag": tag, "problem": q.public_dict() if hasattr(q, "public_dict") else q.pid,
                            "J_true": np.round(Jt, 5).tolist(), "J_nearest": np.round(Jc[star], 5).tolist(),
                            "J_gcirc": np.round(Jc[gc], 5).tolist(),
                            "row": {kk: row[kk] for kk in ("status", "new_env_steps", "certified_policy",
                                                           "true_regret", "false_cert", "gap_layer", "mu_flip",
                                                           "lambda_n0", "lambda_cert", "eta_arg_gcirc",
                                                           "eta_arg_near")}})
    out = {"tag": tag, "seed": seed, "layer": layer, "method": method, "stream": stream, "n_rows": len(keep),
           "errors": errs, "samples": samples, "job_sec": time.perf_counter() - t_job, "lambda_n0_sec": t_lam,
           "env_steps": int(env.n_steps), "worker_pid": os.getpid()}
    tmp = jp.with_name(jp.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, jp)
    return {"tag": tag, "n_rows": len(keep), "n_err": len(errs), "sec": out["job_sec"]}


EDGES: dict = {}


def _job_entry(seed, layer, method, stream, plc, edges, out_dir):
    EDGES.update(edges)
    return job(seed, layer, method, stream, plc, edges["lambda_n0_S1"], out_dir)


def run_pool(jobs, plcs, edges, out_dir, workers):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [j for j in jobs if not (out_dir / "parts" / f"{jtag(j)}.json").exists()]
    done = len(jobs) - len(todo)
    progress(done, len(jobs), "runs", {"resumed_jobs": done})
    outs = []
    # JPC jobs first (longest), then B3
    todo.sort(key=lambda j: (j[2] != "JPC", j[0], j[1], j[3]))
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_job_entry, *j, plcs[j[0]], edges, out_dir): j for j in todo}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"tag": jtag(futs[f]), "fatal": repr(e), "tb": traceback.format_exc()[-2000:]}
            outs.append(r)
            done += 1
            progress(done, len(jobs), "runs", {"last": r.get("tag")})
            log(f"job {done}/{len(jobs)} {r}")
    return outs


def merge(jobs, out_dir):
    from dsswm.mechanism.predictor_log import join, load_jsonl
    parts = out_dir / "parts"
    P, R, errs, meta, samples, missing = [], [], [], [], [], []
    for j in jobs:
        t = jtag(j)
        if not (parts / f"{t}.json").exists():
            missing.append(t)
            continue
        d = json.loads((parts / f"{t}.json").read_text())
        errs += d["errors"]
        samples += d["samples"]
        meta.append({kk: v for kk, v in d.items() if kk not in ("errors", "samples")})
        P += load_jsonl(parts / f"{t}_pred.jsonl")
        R += load_jsonl(parts / f"{t}_res.jsonl")
    for name, rows in (("predictors.jsonl", P), ("results.jsonl", R)):
        with open(out_dir / name, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
    try:
        Jn = join(out_dir / "predictors.jsonl", out_dir / "results.jsonl", check_order=True)
        wa = {"write_ahead_ok": True, "n_joined": len(Jn)}
    except RuntimeError as e:
        wa = {"write_ahead_ok": False, "error": str(e)}
    pk = {}
    for r in P:
        pk.setdefault((r["instance"], r["stream"], r["method"], r["arm"], r["problem"]), r["logged_at"])
    cert = [r for r in R if r.get("status") == "CERTIFIED"]
    miss = [r for r in cert if (r["instance"], r["stream"], r["method"], r["arm"], r["problem"]) not in pk]
    late = [r for r in cert if (k := (r["instance"], r["stream"], r["method"], r["arm"], r["problem"])) in pk
            and pk[k] > r["scored_at"]]
    wa.update({"n_certified": len(cert), "n_cert_without_predictor": len(miss), "n_cert_predictor_late": len(late)})
    (out_dir / "samples").mkdir(exist_ok=True)
    (out_dir / "samples" / "trajectories.json").write_text(json.dumps(samples[:40], indent=1, default=str))
    if errs or missing:
        (out_dir / "errors.json").write_text(json.dumps({"errors": errs, "missing_parts": missing}, indent=1))
    return R, P, meta, errs, missing, wa


# ============================================================================================ analysis
def cp(k, n, a=0.05):
    from dsswm.stats.cp import clopper_pearson
    return list(clopper_pearson(k, n, a)) if n > 0 else [None, None]


def cell_table(rows, ante_key, post_key):
    out = []
    for a in range(3):
        for b in range(3):
            rr = [r for r in rows if r[ante_key] == a and r[post_key] == b]
            cc = [r for r in rr if r["status"] == "CERTIFIED"]
            k = sum(r["false_cert"] for r in cc)
            out.append({"ante_bin": a, "post_bin": POST_LABELS[b], "n_problems": len(rr), "n_cert": len(cc),
                        "n_false_cert": int(k), "fcr": k / len(cc) if cc else None, "cp95": cp(k, len(cc))})
    return out


def analyse(R, slots_per_cell):
    by = defaultdict(list)
    for r in R:
        by[(r["layer"], r["base_method"])].append(r)
    cells, proj = {}, []
    for (layer, m), rr in sorted(by.items()):
        cert = [r for r in rr if r["status"] == "CERTIFIED"]
        k = sum(r["false_cert"] for r in cert)
        nt = [r for r in cert if r["gap_layer"] != "tie"]
        sec = [r["wall_clock_s"] + r["predictor_sec"] + r["lambda_n0_sec"] for r in rr]
        cells[f"{layer}|{m}|full"] = {
            "n": len(rr), "status_counts": dict(Counter(r["status"] for r in rr)), "n_cert": len(cert),
            "n_false_cert": int(k), "fcr_descriptive": k / len(cert) if cert else None, "cp95": cp(k, len(cert)),
            "n_cert_nontie": len(nt), "n_false_cert_nontie": int(sum(r["false_cert"] for r in nt)),
            "zero_cost_rate": float(np.mean([r["zero_cost"] for r in rr])),
            "mean_new_steps": float(np.mean([r["new_env_steps"] for r in rr])),
            "median_new_steps": float(np.median([r["new_env_steps"] for r in rr])),
            "gap_layers": dict(Counter(r["gap_layer"] for r in rr)),
            "ante_bin_muflip_counts": dict(Counter(str(r["ante_bin_muflip"]) for r in rr)),
            "post_bin_gcirc_counts": dict(Counter(POST_LABELS[r["post_bin_gcirc"]] for r in rr)),
            "post_bin_near_counts": dict(Counter(POST_LABELS[r["post_bin_near"]] for r in rr)),
            "gcirc_eq_nearest_rate": float(np.mean([r["gcirc_eq_nearest"] for r in rr])),
            "accepted_hazard_rate": float(np.mean([r["accepted_hazard"] for r in rr])),
            "sec_per_problem": float(np.mean(sec)),
            "sec_per_problem_method": float(np.mean([r["wall_clock_s"] for r in rr])),
            "sec_per_problem_predictor": float(np.mean([r["predictor_sec"] for r in rr])),
            "sec_per_problem_lambda_n0": float(np.mean([r["lambda_n0_sec"] for r in rr])),
            "billing_mismatch": int(sum(not r["billing_ok"] for r in rr))}
        proj.append({"cell": f"{layer}|{m}|full", "sec_per_problem": cells[f"{layer}|{m}|full"]["sec_per_problem"],
                     "problem_slots": slots_per_cell, "safety": 1.2,
                     "cpu_s": cells[f"{layer}|{m}|full"]["sec_per_problem"] * slots_per_cell * 1.2,
                     "source": "this pilot (dev 750-752, stream 0, 15 problems; concurrent)"})
    return cells, proj


def make_plot(R, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    (out_dir / "figures").mkdir(exist_ok=True)
    rows = [r for r in R if r["method"] == "JPC_R1adv"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, (pk, title) in zip(axes, (("post_bin_gcirc", "g° caliber"), ("post_bin_near", "nearest-grid caliber"))):
        n = np.zeros((3, 3), int)
        f = np.zeros((3, 3), int)
        for r in rows:
            if r["ante_bin_muflip"] is None:
                continue
            n[r["ante_bin_muflip"], r[pk]] += r["status"] == "CERTIFIED"
            f[r["ante_bin_muflip"], r[pk]] += r["false_cert"]
        fcr = np.where(n > 0, f / np.maximum(n, 1), np.nan)
        im = ax.imshow(fcr.T, origin="lower", cmap="Reds", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(a, b, f"{f[a, b]}/{n[a, b]}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_yticks(range(3), list(POST_LABELS))
        ax.set_xlabel("ex-ante mu_flip bin (frozen edges)")
        ax.set_ylabel("eta_arg / eps bin")
        ax.set_title(f"JPC R1-adv FCR, {title}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{TASK} pilot (dev seeds, descriptive only): false/certified per cell", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "r1adv_fcr_heatmap.png", dpi=130)
    plt.close(fig)


def main():
    global TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", default="a", choices=sorted(CHUNKS))
    ap.add_argument("--mode", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    TASK = f"r3_hazard_{a.chunk}"
    (RES_ROOT / f"{TASK}.pid").write_text(str(os.getpid()))
    t_all = time.perf_counter()
    start = datetime.now()
    out_dir = RES_ROOT / ("pilots" if a.mode == "pilot" else "full") / TASK
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    status, snapshot = "failed", {"mode": a.mode, "chunk": a.chunk, "workers": a.workers, "tmax": TMAX,
                                  "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                                  "gpu_count": 1, "concurrent": True}
    try:
        lock = json.loads((WS / "plan" / "prereg_lock.json").read_text())
        if a.mode == "full":
            from dsswm.stats.prereg import assert_locked
            lock = assert_locked()
            lo, hi = CHUNKS[a.chunk]
            seeds, streams, reserve = list(range(lo, hi)), [0, 1, 2], RESERVE[a.chunk]
        else:
            seeds, streams, reserve = [750, 751, 752], [0], RESERVE["pilot"]
        edges = {k: lock["frozen_items"]["r1adv_bin_edges"][k] for k in ("mu_flip", "lambda_n0_S1")}
        EDGES.update(edges)
        snapshot.update({"seeds": f"{seeds[0]}-{seeds[-1]}", "streams": streams, "n_problems": 15,
                         "layers": list(LAYERS), "methods": list(METHODS)})
        progress(0, 1, "placements")
        used, repl, plcs, pinfo = placements(seeds, edges["mu_flip"], out_dir, reserve)
        (out_dir / "placements.json").write_text(json.dumps({str(s): {l: {kk: v for kk, v in plcs[s][l].items()
                                                                           if kk != "per_problem"} | {
            "per_problem": [{kk: v for kk, v in r.items() if kk != "J_true"} for r in plcs[s][l]["per_problem"]]}
            for l in LAYERS} for s in used}, indent=1))
        log(f"placements done {pinfo['sec']:.1f}s tables={pinfo['tables_computed']} repl={repl}")
        jobs = [(s, l, m, t) for s in used for l in LAYERS for m in METHODS for t in streams]
        t_runs = time.perf_counter()
        outs = run_pool(jobs, plcs, edges, out_dir, a.workers)
        runs_sec = time.perf_counter() - t_runs
        R, P, meta, errs, missing, wa = merge(jobs, out_dir)
        fatal = [o for o in outs if "fatal" in o]
        slots = 32 * 3 * 15
        cells, proj_cells = analyse(R, slots)
        place_sec = [v for v in pinfo["per_seed_sec"].values()]
        place_per_seed = float(np.mean(place_sec)) if place_sec else 0.0
        allowance_min = place_per_seed * 32 / 60 + 2.0
        cpu_s = sum(c["cpu_s"] for c in proj_cells)
        proj_min = cpu_s / (4 * 60) + allowance_min
        crashes = len(fatal) + len(missing) + len(errs) + sum(r["status"] == "CRASH" for r in R)
        bill_mm = sum(not r["billing_ok"] for r in R)
        flags = {"zero_crashes": crashes == 0, "billing_zero_mismatch": bill_mm == 0,
                 "write_ahead_every_certification": bool(wa.get("write_ahead_ok") and
                                                          wa["n_cert_without_predictor"] == 0 and
                                                          wa["n_cert_predictor_late"] == 0),
                 "projected_full_le_55min": proj_min <= 55.0, "n_runs_ge_100": len(R) >= 100}
        go = all(flags.values())
        downscale = None
        if proj_min > 45.0:
            b3 = sum(c["cpu_s"] for c in proj_cells if "|B3|" in c["cell"])
            downscale = {"option": "B3 reference on stream 0 only (lock contingency, not a methodology preset)",
                         "projected_min": (cpu_s - b3 * 2 / 3) / 240 + allowance_min}
        R1adv_rows = [r for r in R if r["layer"] == "R1adv"]
        fcr_cells = {f"{m}_{ak}_x_{pk}": cell_table([r for r in R1adv_rows if r["base_method"] == m], ak, pk)
                     for m in METHODS for ak in ("ante_bin_muflip", "ante_bin_lambda")
                     for pk in ("post_bin_gcirc", "post_bin_near")}
        try:
            make_plot(R, out_dir)
        except Exception as e:  # noqa: BLE001
            log(f"plot failed {e!r}")
        place_sum = {str(s): {"R1": {kk: plcs[s]["R1"][kk] for kk in ("pool_draws", "quota_fail", "layer_counts",
                                                                        "star")},
                              "R1adv": {kk: plcs[s]["R1adv"][kk] for kk in ("draw", "draws_scanned", "fill_per_bin",
                                                                            "underfilled", "in_r1_box",
                                                                            "pairs_scanned", "pairs_accepted",
                                                                            "acceptance_rate_scanned")},
                              "R1_mu_flip_accept_share": float(np.mean([r["mu_flip"] >= 1 / MC for r in
                                                                         plcs[s]["R1"]["per_problem"]]))}
                     for s in used}
        summary = {
            "task": TASK, "mode": a.mode, "seeds": used, "streams": streams, "replacements": repl,
            "design": {"layers": {"R1": "theta* = offgrid.sample_r1_truth (R1 sub-box), 5/5/5 true-gap quota over "
                                        "the round-3 pool", "R1adv": "P5 rejection sampler, hull-uniform proposal, "
                                                                     "frozen mu_flip edges, 5 per ex-ante bin"},
                       "methods": {"JPC": "uncorrected, full arm", "B3": "full arm (reference)"},
                       "frozen_edges": edges, "mc": MC, "n_pool_r1adv": N_POOL, "tmax": TMAX, "eps": EPS,
                       "delta": DELTA},
            "placement": place_sum, "placement_gpu": pinfo, "n_jobs": len(jobs), "n_runs": len(R),
            "n_predictor_lines": len(P), "crashes": {"fatal_jobs": fatal, "missing_parts": missing,
                                                     "n_errors": len(errs), "errors_head": errs[:5]},
            "billing_mismatch": bill_mm, "write_ahead": wa, "cells": cells,
            "fcr_by_cell_descriptive": fcr_cells,
            "runs_wall_sec": runs_sec, "total_wall_sec": time.perf_counter() - t_all,
            "timing_projection": {"rule": "sum(sec/problem x 1440 slots x 1.2) / (4 workers x 60) + allowance "
                                          "(placement GPU sec/instance x 32 + 2 min)",
                                  "cells": proj_cells, "cpu_s": cpu_s, "allowance_min": allowance_min,
                                  "projected_min_full_chunk": proj_min, "lock_projection_min": LOCK_PROJECTION_MIN,
                                  "downscale_option": downscale,
                                  "caveat": "pilot = stream 0 only, 3 dev instances; full streams 1-2 use other noise "
                                            "seeds; 并发运行（与其他 3 个任务共享 20 核与 4090），计时偏高"},
            "pass_criteria": flags, "go_no_go": "GO" if go else "NO_GO",
            "scientific_readout": "none (pilot = smoke + timing only; dev seeds; FCR cells are descriptive)",
            "lock_status_at_run": lock.get("status"), "code_sha256": code_sha(),
            "start_time": start.isoformat(), "end_time": datetime.now().isoformat()}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        log(f"summary: go={go} flags={flags} proj={proj_min:.1f} min runs={len(R)}")
        if a.mode == "pilot":
            jc = cells.get("R1adv|JPC|full", {})
            md = (f"## {TASK} (pilot, smoke + timing)\n"
                  f"- GO/NO-GO: **{'GO' if go else 'NO_GO'}**; flags: {flags}\n"
                  f"- runs: {len(R)} (dev 750-752, stream 0, {{R1, R1-adv}} x {{JPC, B3}} x 15)；崩溃 {crashes}；"
                  f"计费不一致 {bill_mm}；write-ahead {wa}\n"
                  f"- 投影 full 单块: {proj_min:.1f} min（lock 外推 {LOCK_PROJECTION_MIN:.1f}；并发运行）；"
                  f"降规模备选: {downscale}\n"
                  f"- R1-adv JPC 描述性: n_cert={jc.get('n_cert')} false={jc.get('n_false_cert')}（开发种子，非证据）\n")
            update_shared_summary({"go_no_go": "GO" if go else "NO_GO", "pass_criteria": flags,
                                   "projected_full_min": proj_min, "n_runs": len(R), "downscale_option": downscale,
                                   "summary_path": str(out_dir / "summary.json")}, md)
        status = "success" if flags["zero_crashes"] else "failed"
        snapshot["projected_full_min"] = round(proj_min, 1)
        mark_done(status, f"{TASK} {a.mode}: GO={go} runs={len(R)} proj={proj_min:.1f}min")
        return summary
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log(tb)
        (out_dir / "fatal.txt").write_text(tb)
        mark_done("failed", f"{TASK} {a.mode} crashed: {e!r}")
        raise
    finally:
        update_gpu_progress(status, start.isoformat(), (time.perf_counter() - t_all) / 60, snapshot,
                            40 if a.mode == "full" else 8)


if __name__ == "__main__":
    main()
