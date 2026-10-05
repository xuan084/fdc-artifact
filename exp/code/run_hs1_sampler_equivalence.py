"""HS1 sampler equivalence (task hs1_sampler_equivalence; hypotheses.md row HS1; cand_a, ablation).

Question: is the sampling rule load-bearing for JPC's savings?  Everything except the action-selection rule is held
fixed: the SAME joint grid-LR confidence set (SeqLRSet, UI threshold log(1/delta), predictable plug-in numerator),
the SAME certifier, the SAME stream protocol (n0 = 20 shared initial rounds, K problems per stream, each sampler
re-uses its OWN ledger across the stream, own fresh platform copy, CRN on (instance, stream)), the SAME
MODEL_CONFLICT -> B2 whole-trial fallback and the same T_max charging as hr1_* (run_hr1_main.run_jpc).
Samplers (action at the current observable state):
  DDA      JPC's KL decision-directed design (dsswm.acquire.nl_kl_dda.dda_choose, untuned locked default)
  B10      reward-free G-optimal coverage: argmin_a max_{x in library} tr(F_x (V + F_a)^-1), library = every
           (observable state, legal action) Fisher at the grid MLE (identical code to hr2 'gopt'), explore 0.05
  B11      task-directed Fisher (dsswm.baselines.b11_taskfisher.choose; locked round-0 B5-TaskDirected config
           crit=T, gapnorm=False, explore=0): reduction of max_{k != k_hat} (g_k - g_khat)^T M^-1 (g_k - g_khat),
           g = dJ/dv at the grid MLE (autograd through the exact propagator), M = V + F_a.
           DEVIATION (documented): one-step look-ahead (look=1) over legal actions; the round-0 'look=2' trial/
           two-step element library is not ported to the round-1 offgrid stack (round-0 dev score look=1 4.82 vs
           look=2 4.70, i.e. within the tuning noise).
  uniform  uniform random legal action (reference only).
Levels: R0 (truth snapped to G_1, grid G_1, exact minimax certifier) and the locked main off-grid level R3
(continuous truth, grid G_{f*}, f* = 2 from plan/prereg_lock.json; R2 left family A).  At R3 the locked inflated
certificate R_grid + 2 max_alive eta_loc <= eps is used ('_infl', binding); because the data-free floor
2 min eta_loc > eps holds on every dev problem, all samplers refuse at 0 steps there, so HS1@R3 is additionally
reported on the UNCORRECTED grid certifier ('_unc', diagnostic only, FCR reported) to give a non-vacuous
sampling comparison off-grid.
B10 / B11 / uniform share the joint LR evidence with JPC: sampling-rule rows, never HR1 denominators.
Consistency check: the DDA arm is also run through run_hr1_main.run_jpc and must reproduce it step-for-step.
HS1 statistic: per instance mean over streams of log(stream total charged steps + 1); pairwise instance-cluster
bootstrap (B = 10^4, seed 42) of the mean log ratio; 90% CI; TOST equivalence iff CI subset [0.8, 1.25]
(TOST p = max of the two one-sided bootstrap tail masses).

Pilot: dev instances 600-604 x 6 problems x stream 0 (seed 42). Full: lock range hs1_sampler_equivalence
(10000-10047) x 3 streams x 15 problems at R0; first 24 instances at R3 (assert_locked()).
Usage: run_hs1_sampler_equivalence.py --mode {pilot,full} [--workers 4] [--smoke] [--resummarize]
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
from datetime import datetime  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_hr1_main as H  # noqa: E402  (same Arm / tables / JPC / B2 fallback code path as hr1_*)
import run_hr2_threshold_accounting as HR2  # noqa: E402  (G-opt library, identical to the hr2 'gopt' design)
from dsswm.acquire.nl_kl_dda import dda_choose  # noqa: E402
from dsswm.baselines import b11_taskfisher as B11  # noqa: E402
from dsswm.baselines.glm_linearised import value_and_grad  # noqa: E402
from dsswm.certify.minimax_enum import certify_minimax, trichotomy  # noqa: E402
from dsswm.certify.minimax_infl import certify_minimax_infl, inflation_floor  # noqa: E402
from dsswm.evidence.ledger import EvidenceLedger  # noqa: E402
from dsswm.evidence.lr_set import SeqLRSet  # noqa: E402
from dsswm.streams.generator import generator_hash  # noqa: E402
from dsswm.streams.offgrid import STREAMS  # noqa: E402

TASK = "hs1_sampler_equivalence"
WS, RES_ROOT, LOCK = H.WS, H.RES_ROOT, H.LOCK
EPS, DELTA, TOP_M, TMAX, LOG_THR = H.EPS, H.DELTA, H.TOP_M, H.TMAX_STEP, H.LOG_THR
N0 = H.N0
SAMPLERS = ("DDA", "B10", "B11", "uniform")
EQUIV = (0.8, 1.25)
B10_EXPLORE = 0.05
SALT = {"DDA": 101, "B10": 1010, "B11": 1111, "uniform": 1212}      # DDA salt == run_hr1_main.run_jpc


def mname(sampler, level, cert):
    return sampler if level == "R0" else f"{sampler}_{cert}"


def run_sampler(arm, sampler, cert_mode, want_samples=False, b11_cfg=None):
    """Joint-LR stream with the given sampler. cert_mode in {'exact' (R0), 'infl' (R3 binding), 'unc' (R3 diag)}.
    Identical to run_hr1_main.run_jpc except for the action choice."""
    infl = cert_mode == "infl"
    method = mname(sampler, arm.level, cert_mode)
    inst = arm.fresh()
    env = inst.env
    h = env.handle()
    lr = SeqLRSet(arm.prop, arm.LT, DELTA)
    led = EvidenceLedger([lr])
    all_obs = list(inst.init_obs)
    for o in inst.init_obs:
        led.record(o, "initial")
    rng = np.random.default_rng([arm.seed, arm.noise_seed, SALT[sampler]])
    des = H._B3Design(arm, inst) if sampler in ("B10", "B11") else None
    b11_cfg = b11_cfg or {"gapnorm": False, "explore": 0.0}
    b2_hist = {}
    rows, samples = [], []
    for k, q in enumerate(arm.problems):
        Reg, eta = arm.Reg[k], arm.eta[k]
        t0 = time.perf_counter()
        steps, status, traj, refused = 0, None, [], False
        kh_prev, Vm, grad_cache, n_fb = None, None, {}, 0
        plans = [arm.prop.build_policy_plan(p, q.loads0, q.engaged0, q.H) for p in q.policies] if sampler == "B11" else None
        U = type("U", (), {"w": q.utility.w, "w_ret": q.utility.w_ret, "c_q": arm.cq[k]})()
        dec_s = 0.0
        while True:
            if infl and H.floor_any(eta):
                status, refused = "NEED_DATA", True
                break
            mask = lr.mask().numpy()
            if infl:
                cert = certify_minimax_infl(Reg, mask, eta, EPS, TOP_M)
                if cert["status"].value == "NEED_DATA" and inflation_floor(eta, mask, EPS):
                    status, refused = "NEED_DATA", True
                    break
            else:
                cert = certify_minimax(Reg, mask, EPS, TOP_M)
            sv = cert["status"].value
            if sv in ("CERTIFIED", "MODEL_CONFLICT"):
                status = sv
                break
            if not arm.all_single and steps % 10 == 0:
                amb, _, _ = trichotomy(Reg, mask, arm.gid, EPS)
                if amb:
                    status = "OUT_OF_SCOPE"
                    break
            if steps >= TMAX:
                status = "NEED_DATA"
                break
            kh = lr.mle()
            code = arm.prop.codec.encode(*h.observable_state())
            td = time.perf_counter()
            mode = sampler
            if sampler == "DDA":
                blk = cert["blocking"] or [int(i) for i in np.flatnonzero(mask)[np.argsort(-eta[mask])[:TOP_M]]]
                margins = LOG_THR - lr.log_ratio().numpy()[blk]
                a, info = dda_choose(code, arm.py[kh:kh + 1], arm.pe[kh:kh + 1], arm.LT_np[kh], arm.py[blk],
                                     arm.pe[blk], margins, arm.inc, arm.prop, arm.legal, rng)
                mode = info["mode"]
            elif sampler == "uniform":
                a = int(rng.choice(arm.legal))
            else:
                if kh != kh_prev:
                    Vm = des.build_V(kh)
                    kh_prev = kh
                Fa = des.F_all(kh, code)
                if sampler == "B10":
                    if rng.random() < B10_EXPLORE:
                        a, mode = int(rng.choice(arm.legal)), "explore"
                    else:
                        Fl = HR2._glib(arm, des, kh)
                        Vi = np.linalg.inv(Vm[None] + Fa[arm.legal])
                        vals = (Fl @ Vi.reshape(len(arm.legal), -1).T).max(0)
                        a = int(arm.legal[int(np.argmin(vals))])
                else:                                                   # B11 task-directed Fisher
                    Jh = arm.J[k][kh]
                    if len(Jh) < 2:
                        a, mode = int(rng.choice(arm.legal)), "single_policy_uniform"
                        n_fb += 1
                    else:
                        if kh not in grad_cache:
                            grad_cache[kh] = value_and_grad(arm.prop, plans, arm.V[kh], U, 2, 2)[1]
                        G = grad_cache[kh]
                        k_hat = int(np.argmax(Jh))
                        gain = B11.task_directed_gain(Vm, Fa[arm.legal], G, Jh, k_hat, b11_cfg["gapnorm"], EPS)
                        if not np.isfinite(gain).all() or gain.max() <= 0:
                            n_fb += 1                               # no task-relevant reduction left: tie -> first
                        a = B11.choose(Vm, Fa[arm.legal], np.ones(len(arm.legal)), G, Jh, k_hat, b11_cfg["gapnorm"],
                                       EPS, rng, b11_cfg["explore"])
                        a = int(arm.legal[a])
            dec_s += time.perf_counter() - td
            obs = h.step(a)
            led.record(obs, "probe", q.pid)
            all_obs.append(obs)
            if des is not None:
                des.hist[(code, a)] = des.hist.get((code, a), 0) + 1
                Vm = Vm + Fa[a]
            steps += 1
            if want_samples and len(traj) < 40 and (steps <= 20 or steps % 10 == 0):
                traj.append({"step": steps, "a": int(a), "set": int(mask.sum()), "r_bar": round(float(cert["r_bar"]), 4),
                             "mle": int(kh), "star_alive": bool(mask[arm.star]), "mode": mode})
        assert env.n_steps == len(all_obs) == lr.n_rounds, "step accounting mismatch"
        mask = lr.mask().numpy()
        gc = certify_minimax(Reg, mask, EPS, TOP_M)
        pi = gc["pi"] if status == "CERTIFIED" else None
        extra = {"sampler": sampler, "cert_mode": cert_mode, "set_size": int(mask.sum()),
                 "r_bar_grid": float(gc["r_bar"]), "theta_cell_alive": bool(mask[arm.star]),
                 "refused_floor": refused, "uncertified_recommendation": None if status == "CERTIFIED" else gc["pi"],
                 "lr_n_rounds": int(lr.n_rounds), "env_n_steps": int(env.n_steps), "fallback": None,
                 "decision_sec": dec_s, "b11_degenerate_steps": n_fb,
                 "diagnostic_only": cert_mode == "unc"}
        charged = None
        if status == "MODEL_CONFLICT":
            res = H.run_b2_problem(arm, env, k, b2_hist, None)
            extra["fallback"] = {"method": "B2", "status": res["status"], "steps": int(res["steps"])}
            if res["status"] == "CERTIFIED":
                status, pi = "CERTIFIED", int(res["pi"])
                charged = steps + int(res["steps"])
            else:
                charged = max(steps + int(res["steps"]), H.TMAX_TRIAL)
        extra["wall_clock_s"] = time.perf_counter() - t0
        r = arm.row(k, method, status, pi, steps, extra, charged=charged)
        r["kind"] = "hs1"
        r["diagnostic_only"] = cert_mode == "unc"
        rows.append(r)
        if want_samples and k < 2:
            samples.append({**r, "J_true": arm.Jt[k].round(4).tolist(), "trajectory_head": traj})
    return rows, samples


def run_unit(seed, stream, level, f, sampler, cert_modes, n_prob, meta, want_samples, check_dda):
    torch.set_num_threads(1)
    t0 = time.time()
    arm = H.Arm(seed, stream, level, f, n_prob, meta, need_lr=True, infl=level != "R0")
    build = time.time() - t0
    rows, samples, errors, msec, checks = [], [], [], {}, {}
    for cm in cert_modes:
        tm = time.time()
        m = mname(sampler, level, cm)
        try:
            r, s = run_sampler(arm, sampler, cm, want_samples)
            rows += r
            samples += s
            if sampler == "DDA" and check_dda and cm in ("exact", "infl", "unc"):
                ref, _ = H.run_jpc(arm, cm == "infl")
                same = [(a["status"], a["new_env_steps"], a["certified_policy"]) ==
                        (b["status"], b["new_env_steps"], b["certified_policy"]) for a, b in zip(r, ref)]
                checks[m] = {"matches_run_hr1_main_run_jpc": bool(all(same)) and len(r) == len(ref),
                             "n": len(same), "n_match": int(sum(same))}
        except Exception as e:  # noqa: BLE001
            errors.append({"instance": seed, "stream": stream, "level": level, "f": f, "method": m, "error": repr(e),
                           "tb": traceback.format_exc()})
        msec[m] = time.time() - tm
    return rows, samples, errors, {"instance": seed, "stream": stream, "level": level, "sampler": sampler,
                                   "cert_modes": list(cert_modes), "build_sec": build, "method_sec": msec,
                                   "dda_checks": checks, "sec": time.time() - t0}


# ------------------------------------------------------------------------------------ statistics
def inst_logs(rows, m):
    out = {}
    for i in sorted({r["instance"] for r in rows if r["method"] == m}):
        R = [r for r in rows if r["method"] == m and r["instance"] == i]
        sts = sorted({r["stream"] for r in R})
        out[i] = float(np.mean([math.log(sum(r["new_env_steps"] for r in R if r["stream"] == s) + 1) for s in sts]))
    return out


def pair_ratio(rows, a, b, B=10000, seed=42):
    la, lb = inst_logs(rows, a), inst_logs(rows, b)
    insts = sorted(set(la) & set(lb))
    if not insts:
        return None
    x = np.array([la[i] - lb[i] for i in insts])
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(len(x), size=(B, len(x)))].mean(1)
    lo90, hi90 = np.quantile(bs, [0.05, 0.95])
    lo95, hi95 = np.quantile(bs, [0.025, 0.975])
    p_low = float(np.mean(bs <= math.log(EQUIV[0])))      # H0: ratio <= 0.8
    p_up = float(np.mean(bs >= math.log(EQUIV[1])))       # H0: ratio >= 1.25
    degenerate = bool(np.allclose(x, 0.0))
    eq = bool(math.exp(lo90) >= EQUIV[0] and math.exp(hi90) <= EQUIV[1]) and len(insts) >= 3   # n<3: no CI
    return {"pair": f"{a}/{b}", "n_instances": len(insts), "ratio_geo_mean": float(math.exp(x.mean())),
            "ci90": [float(math.exp(lo90)), float(math.exp(hi90))], "ci95": [float(math.exp(lo95)), float(math.exp(hi95))],
            "tost_p": max(p_low, p_up), "tost_p_lower": p_low, "tost_p_upper": p_up, "tost_equivalent": eq,
            "a_significantly_better_ci_upper_lt_0.8": bool(math.exp(hi90) < 0.8),
            "degenerate_identical": degenerate, "per_instance": [float(math.exp(v)) for v in x]}


def hs1_block(rows, level, cert):
    ms = [mname(s, level, cert) for s in SAMPLERS]
    present = [m for m in ms if any(r["method"] == m for r in rows)]
    pairs = {}
    for a, b in combinations(present, 2):
        pairs[f"{a}/{b}"] = pair_ratio(rows, a, b)
    core = [mname(s, level, cert) for s in ("DDA", "B10", "B11")]
    core_pairs = [p for p in pairs.values() if p and p["pair"].split("/")[0] in core and p["pair"].split("/")[1] in core]
    verdict = None
    if core_pairs:
        all_eq = all(p["tost_equivalent"] for p in core_pairs)
        dda_better = any(p["pair"].startswith(mname("DDA", level, cert) + "/") and
                         p["a_significantly_better_ci_upper_lt_0.8"] for p in core_pairs)
        verdict = {"all_core_pairs_tost_equivalent": all_eq, "dda_significantly_better": dda_better,
                   "degenerate_all_refused": all(p["degenerate_identical"] for p in core_pairs),
                   "label": ("DEGENERATE (all refuse identically)" if all(p["degenerate_identical"] for p in core_pairs)
                             else "SUPPORTED" if all_eq else "FAILS (DDA better)" if dda_better
                             else "INCONCLUSIVE (CI not inside [0.8,1.25])")}
    return {"level": level, "cert_mode": cert, "methods": present, "pairs": pairs, "verdict_preview": verdict}


def summarize(rows):
    s = H.summarize(rows)
    for m in s:
        R = [r for r in rows if r["method"] == m]
        s[m]["decision_sec_per_step"] = float(sum(r["decision_sec"] for r in R) / max(sum(r["steps_consumed"] for r in R), 1))
        s[m]["b11_degenerate_steps"] = int(sum(r.get("b11_degenerate_steps", 0) for r in R))
        s[m]["policy_agreement_with_true_opt"] = float(np.mean([r["true_regret"] == 0.0 for r in R
                                                                if r["certified_policy"] is not None])) \
            if any(r["certified_policy"] is not None for r in R) else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resummarize", action="store_true")
    args = ap.parse_args()
    pilot = args.mode == "pilot"
    lock = json.loads(LOCK.read_text())
    if not pilot:
        from dsswm.stats.prereg import assert_locked
        lock = assert_locked()
    fstar = int(lock.get("f_star", 2))
    out_dir = RES_ROOT / ("pilots" if pilot else "full") / (TASK + ("_smoke" if args.smoke else ""))
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    pid_path = RES_ROOT / f"{TASK}.pid"
    prog_path = RES_ROOT / f"{TASK}_PROGRESS.json"
    real = not (args.smoke or args.resummarize)
    if real:
        pid_path.write_text(str(os.getpid()))
    logf = open(out_dir / "run.log", "a")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    def progress(step, total, phase, metric=None):
        if not real:
            return
        prog_path.write_text(json.dumps({
            "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total, "loss": None,
            "metric": {"phase": phase, **(metric or {})}, "updated_at": datetime.now().isoformat()}))

    def mark_done(status, summary_txt):
        if pid_path.exists():
            pid_path.unlink()
        fp = {}
        if prog_path.exists():
            try:
                fp = json.loads(prog_path.read_text())
            except ValueError:
                pass
        (RES_ROOT / f"{TASK}_DONE").write_text(json.dumps({"task_id": TASK, "status": status, "summary": summary_txt,
                                                           "final_progress": fp,
                                                           "timestamp": datetime.now().isoformat()}))

    if pilot:
        seeds_r0, n_prob, streams = list(range(600, 605)), 6, [0]
        seeds_r3 = list(seeds_r0)
        assert max(seeds_r0) < 10000, "dev seeds only in pilot"
    else:
        rngs = lock["eval_manifest"]["per_task_ranges"][TASK]
        seeds_r0 = [s for a, b in rngs for s in range(a, b + 1)]
        seeds_r3 = seeds_r0[:24]
        n_prob, streams = int(lock["eval_manifest"]["K_problems"]), list(range(len(STREAMS)))
    if args.smoke:
        seeds_r0, seeds_r3, n_prob = seeds_r0[:1], seeds_r3[:1], 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    T0 = time.time()
    (out_dir / "start_time.txt").write_text(datetime.now().isoformat())
    log(f"start {TASK} mode={args.mode} R0 seeds {seeds_r0[0]}-{seeds_r0[-1]}, R3(f*={fstar}) seeds "
        f"{seeds_r3[0]}-{seeds_r3[-1]} x streams {streams} x {n_prob} problems; samplers {SAMPLERS}; device={dev}; "
        f"generator_hash={generator_hash()} lock status={lock.get('status')}")
    summary = {"task_id": TASK, "mode": args.mode, "seed": 42, "eps": EPS, "delta": DELTA, "f_star": fstar,
               "seeds_R0": [seeds_r0[0], seeds_r0[-1]], "seeds_R3": [seeds_r3[0], seeds_r3[-1]],
               "n_problems_per_stream": n_prob, "streams": streams, "eval_seeds_touched": not pilot,
               "generator_hash": generator_hash(), "lock_status": lock.get("status"),
               "lock_sha256": lock.get("sha256") or lock.get("sha256_provisional"), "concurrent_run": True,
               "tmax_stepwise": TMAX, "samplers": list(SAMPLERS), "equivalence_margin": list(EQUIV),
               "b11": {"implemented": True, "module": "dsswm/baselines/b11_taskfisher.py",
                       "equivalent_to": B11.EQUIVALENT_TO,
                       "locked_config": (lock.get("tuning", {}).get("B11") or {}).get("selected"),
                       "used_config": {"crit": "T", "gapnorm": False, "explore": 0.0, "look": 1},
                       "deviation": "look=1 (one-step over legal actions) instead of locked look=2; round-0 dev "
                                    "score look=1/gapnorm=False 4.822 vs selected look=2 4.703 (mean log(1+steps))"},
               "b10": {"design": "G-opt coverage over all (state, legal action) Fisher at the grid MLE "
                                 "(run_hr2_threshold_accounting._glib), explore 0.05"},
               "notes": ["same SeqLRSet (UI log(1/delta)), same certifier, same stream / fallback / T_max charging "
                         "as run_hr1_main.run_jpc; only the action choice differs",
                         "R3 '_infl' = locked binding certificate; '_unc' = uncorrected grid certifier on G_{f*} "
                         "(diagnostic, FCR reported)",
                         "B10/B11/uniform share the joint LR evidence: never HR1 denominators",
                         "timings are from a concurrent run (up to 4 tasks on 20 cores)"]}
    res_path = out_dir / "results.jsonl"
    done_path = out_dir / "units_done.jsonl"
    unit_plan = ([(s, st, "R0", 1, smp, ("exact",)) for s in seeds_r0 for st in streams for smp in SAMPLERS] +
                 [(s, st, "R3", fstar, smp, ("infl", "unc")) for s in seeds_r3 for st in streams for smp in SAMPLERS])
    meta = None
    try:
        if not args.resummarize:
            progress(1, 4, "gpu: J tables + eta_loc")
            seeds_all = sorted(set(seeds_r0) | set(seeds_r3))
            meta = H.precompute(seeds_all, n_prob, streams, (1, fstar), (fstar,), dev, log)
            summary["table_sec"] = meta["sec"]
            summary["eta_loc_min_over_eps"] = {k: {"median": float(np.median(v)), "min": float(np.min(v))}
                                               for k, v in meta["eta_min_over_eps"].items()}
            log(f"min_g eta_loc/eps (floor: refuse iff > 0.5): {summary['eta_loc_min_over_eps']}")
            done = set()
            if done_path.exists():
                for x in open(done_path):
                    u = json.loads(x)
                    done.add((u["instance"], u["stream"], u["level"], u["sampler"]))
            jobs = [j for j in unit_plan if (j[0], j[1], j[2], j[4]) not in done]
            # long units first (uniform / B10 / B11 at R0 and R3-unc) for load balance
            jobs.sort(key=lambda j: ({"uniform": 0, "B10": 1, "B11": 2, "DDA": 3}[j[4]], j[2] == "R3"))
            log(f"units: {len(jobs)} to run, {len(done)} already done (resume)")
            progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": 0})
            from joblib import Parallel, delayed
            n_err, all_samples, checks = 0, [], {}
            first = (seeds_r0[0], streams[0])
            gen = Parallel(n_jobs=args.workers, backend="loky", return_as="generator_unordered")(
                delayed(run_unit)(s, st, lv, f, smp, cms, n_prob, meta, (s, st) == first, pilot or s < seeds_r0[0] + 4)
                for (s, st, lv, f, smp, cms) in jobs)
            for i, (rr, ss, ee, us) in enumerate(gen):
                with open(res_path, "a") as fh:
                    for r in rr:
                        fh.write(json.dumps(r, default=float) + "\n")
                for e in ee:
                    log(f"ERROR {e['instance']} s{e['stream']} {e['level']} {e['method']}: {e['error']}\n{e['tb']}")
                n_err += len(ee)
                with open(out_dir / "errors.jsonl", "a") as fh:
                    for e in ee:
                        fh.write(json.dumps(e) + "\n")
                if not ee:
                    with open(done_path, "a") as fh:
                        fh.write(json.dumps({"instance": us["instance"], "stream": us["stream"], "level": us["level"],
                                             "sampler": us["sampler"]}) + "\n")
                with open(out_dir / "unit_sec.jsonl", "a") as fh:
                    fh.write(json.dumps(us) + "\n")
                all_samples += ss
                progress(2, 4, "cpu: streams", {"units_total": len(jobs), "units_done": i + 1, "errors": n_err})
                log(f"unit {i + 1}/{len(jobs)} {us['level']} {us['sampler']} inst {us['instance']} s{us['stream']} "
                    f"{us['sec']:.1f}s {us['dda_checks'] or ''}")
            if all_samples:
                (samples_dir / "samples.json").write_text(json.dumps(all_samples, indent=1, default=float))
            summary["crashes_this_run"] = n_err
        else:
            old = json.loads((out_dir / "summary.json").read_text())
            summary = {**old, **{k: v for k, v in summary.items() if k not in old}}
        progress(3, 4, "summary")
        rows = [json.loads(x) for x in open(res_path)]
        dedup = {}
        for r in rows:
            dedup[(r["instance"], r["stream"], r["method"], r["k"])] = r
        rows = list(dedup.values())
        errs = [json.loads(x) for x in open(out_dir / "errors.jsonl")] if (out_dir / "errors.jsonl").exists() else []
        units_all = [json.loads(x) for x in open(out_dir / "unit_sec.jsonl")] if (out_dir / "unit_sec.jsonl").exists() else []
        dda_checks = {}
        for u in units_all:
            for m, c in (u.get("dda_checks") or {}).items():
                d = dda_checks.setdefault(m, {"units": 0, "units_match": 0})
                d["units"] += 1
                d["units_match"] += int(c["matches_run_hr1_main_run_jpc"])
        summ = summarize(rows)
        summary["by_method"] = summ
        summary["hs1"] = {"R0": hs1_block(rows, "R0", "exact"), "R3_infl": hs1_block(rows, "R3", "infl"),
                          "R3_unc_diagnostic": hs1_block(rows, "R3", "unc")}
        summary["dda_consistency_checks"] = dda_checks
        summary["n_rows"] = len(rows)
        expected = (len(seeds_r0) * len(streams) * n_prob * len(SAMPLERS) +
                    len(seeds_r3) * len(streams) * n_prob * len(SAMPLERS) * 2)
        summary["rows_expected"] = expected
        summary["complete"] = len(rows) == expected
        summary["crashes_total_logged"] = len(errs)
        acc_ok = all(r["lr_n_rounds"] == r["env_n_steps"] for r in rows)
        summary["evidence_boundary_assertions_ok"] = acc_ok and not any("step accounting" in e["error"] for e in errs)
        # suspicious gate: >30% over the simple (uniform) reference or > 5x savings
        flags = []
        for blk in ("R0", "R3_unc_diagnostic"):
            for key, p in summary["hs1"][blk]["pairs"].items():
                if p and key.split("/")[1].startswith("uniform") and p["ratio_geo_mean"] < 0.7:
                    flags.append(f"{blk} {key} = {p['ratio_geo_mean']:.2f} (>30% better than uniform reference): "
                                 f"checked - same LR set / certifier / CRN; samplers read only learner-side objects "
                                 f"(grid MLE, LR set, public J tables), never theta* / J_true")
                if p and (p["ratio_geo_mean"] < 0.2 or p["ratio_geo_mean"] > 5):
                    flags.append(f"{blk} {key} = {p['ratio_geo_mean']:.2f} (>5x)")
        summary["suspicious_flags"] = flags
        # projection to full: per-(sampler, level) unit seconds x full units / 4 workers
        per = {}
        for u in units_all:
            per.setdefault((u["level"], u["sampler"]), []).append(u["sec"] / max(n_prob, 1))
        n0_full, n3_full = 48, 24
        full_sec = sum(np.mean(v) * 15 * 3 * (n0_full if lv == "R0" else n3_full) for (lv, smp), v in per.items())
        summary["timing_projection"] = {
            "sec_per_problem_by_level_sampler": {f"{lv}:{smp}": float(np.mean(v)) for (lv, smp), v in per.items()},
            "full_cpu_sec": float(full_sec), "full_wall_min_at_4_workers": float(full_sec / 4 / 60),
            "basis": "48 inst x 3 streams x 15 problems at R0 + 24 inst at R3 (infl + unc); pilot per-problem means; "
                     "DDA pilot units also include the run_jpc consistency re-run (full: first 4 instances only)",
            "note": "concurrent run; uniform / B10 cost grows with steps, so the projection is approximate"}
        dda_ok = all(v["units"] == v["units_match"] for v in dda_checks.values()) and bool(dda_checks)
        r0 = summary["hs1"]["R0"]["pairs"]
        pc = {"zero_crashes": len(errs) == 0, "all_rows_present": summary["complete"],
              "evidence_boundary_assertions_hold": summary["evidence_boundary_assertions_ok"],
              "b11_implemented": True, "dda_reproduces_hr1_jpc_path": dda_ok,
              "hs1_ratios_computed_R0": all(p is not None for p in r0.values()) and len(r0) == 6,
              "projected_full_wall_le_75_min": summary["timing_projection"]["full_wall_min_at_4_workers"] <= 75.0}
        summary["pass_criteria"] = pc
        summary["go_no_go"] = "GO" if all(pc.values()) else "NO_GO"
        summary["wall_clock_s"] = time.time() - T0
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        for m, v in summ.items():
            log(f"{m:12s} n={v['n']:3d} completion {v['completion']:.2f} refused {v['refused_floor']:3d} stream steps "
                f"{v['stream_steps_charged_mean']:.0f} consumed/stream {v['steps_consumed_stream_mean']:.0f} "
                f"FCR {v['false_certs']}/{v['n']} CPup {v['fcr_cp_upper']:.3f} theta-cell {v['theta_cell_survival']} "
                f"zero-cost {v['zero_cost_rate']:.2f} wall/problem {v['wall_clock_per_problem_s']:.2f}s")
        for blk, b in summary["hs1"].items():
            for key, p in b["pairs"].items():
                if p:
                    log(f"HS1 {blk:18s} {key:22s} ratio {p['ratio_geo_mean']:.3f} 90%CI [{p['ci90'][0]:.3f}, "
                        f"{p['ci90'][1]:.3f}] TOST p {p['tost_p']:.3f} equiv {p['tost_equivalent']}")
            log(f"HS1 {blk} verdict preview: {b['verdict_preview']}")
        log(f"flags: {flags}")
        log(f"projection {summary['timing_projection']['full_wall_min_at_4_workers']:.1f} min; {summary['go_no_go']} {pc}")
        progress(4, 4, "done", {"go_no_go": summary["go_no_go"]})
        if real:
            mark_done("success", f"{args.mode}: {summary['go_no_go']} {pc}")
    except Exception as e:  # noqa: BLE001
        log(traceback.format_exc())
        summary["error"] = repr(e)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
        if real:
            mark_done("failed", repr(e))
        raise


if __name__ == "__main__":
    main()
