"""r5_supp_p0: POST-HOC / EXPLORATORY zero-cost re-analysis of the round-5 confirmatory blocks.

NOT PART OF THE LOCK v5 VERDICT. Reads existing eval-block raw rows under exp/results/full/ only (no rerun, no stream
exclusion, no change to any block file, the lock, frozen dsswm code or r5_analysis_aggregate outputs). Writes only to
exp/results/full/supp_r5_p0/.

  1. design x certificate 2x2 (pool/half x rect/QFC) with paired geomean ratios, bootstrap CIs and the
     log-interaction I; code-level and schedule-digest verification of the cell identities.
  2. per-budget (15 budgets) certification-time ratios FDC / rival, per-budget completion at stop, and a structural
     explanation of the B = 0.25 / 0.30 behaviour.
  3. N100: what the existing rows identify (rival 15/15 at stop), conservative upper bounds, continuation spec + cost.
  4. C1 ratios with two-sided CIs, one-sided UB, Bonferroni and Holm step-down bounds; K = 60 and CR12 side by side.
  5. consolidated deviation table (json + md) -- written from a hand-curated list in this script.
  Statistics: paired percentile bootstrap over streams, B = 10^4, seed 42 (same resampling convention as the lock SAP).

  python run_r5_supp_p0.py
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
WS = CODE.parent.parent
FULL = WS / "exp" / "results" / "full"
OUT = FULL / "supp_r5_p0"
LABEL = "post-hoc / exploratory, not part of the lock v5 verdict"

from dsswm.baselines import b4_bal, fdc, frontier_common, hait_sw, uniform_rs  # noqa: E402
from dsswm.baselines.frontier_common import make_ctx  # noqa: E402
from dsswm.streams import frontier as fr  # noqa: E402

B_BOOT, SEED, ALPHA = 10_000, 42, 0.05
TAU = 6989799
CK = [int(x) for x in fr.checkpoints(50_000, TAU, 20)]
SEEDS = list(range(30000, 30200))
BUDGETS = [round(0.10 + 0.05 * i, 2) for i in range(15)]
R_SET = ["B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool"]


# ------------------------------------------------------------------------------------------------ loading
def g(r, k, default=None):
    rs = r.get("run_stream") or {}
    for key in (k, "rs_" + k):
        if key in r and r[key] is not None:
            return r[key]
    if k in rs and rs[k] is not None:
        return rs[k]
    return default


def load(paths):
    out = {}
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                out[(r["method"], int(r["seed"]))] = {
                    "N80_pen": float(r["N80_pen"]), "completed": bool(g(r, "completed")),
                    "fwer_event": bool(g(r, "fwer_event")), "cert_k": list(g(r, "cert_k")),
                    "k_stop": g(r, "k_stop"), "digest": g(r, "schedule_digest"), "sec": g(r, "sec"),
                    "N100_pen": r.get("N100_pen"), "completed_15": r.get("completed_15"),
                    "fwer_event_full": r.get("fwer_event_full"), "decided_pi": g(r, "decided_pi")}
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr(rows, m, seeds, key="N80_pen"):
    miss = [s for s in seeds if (m, s) not in rows]
    if miss:
        raise RuntimeError(f"missing {m}: {miss[:5]}")
    return np.array([rows[(m, s)][key] for s in seeds], dtype=float)


# ------------------------------------------------------------------------------------------------ stats
def boot_idx(n, B=B_BOOT, seed=SEED):
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def paired(a, b, idx=None, m_family=None):
    d = np.log(a) - np.log(b)
    if idx is None:
        idx = boot_idx(len(d))
    bt = d[idx].mean(1)
    q = lambda p: float(np.exp(np.quantile(bt, p)))  # noqa: E731
    out = {"n": int(len(d)), "ratio": float(np.exp(d.mean())), "ci95_two_sided": [q(0.025), q(0.975)],
           "ub95_one_sided": q(0.95), "frac_faster": float((d < 0).mean()), "frac_tied": float((d == 0).mean()),
           "frac_slower": float((d > 0).mean()), "mean_log": float(d.mean()),
           "p_boot_one_sided": float((1 + (bt >= 0).sum()) / (len(bt) + 1))}
    out["speedup"] = 1.0 / out["ratio"]
    out["speedup_ci95"] = [1.0 / out["ci95_two_sided"][1], 1.0 / out["ci95_two_sided"][0]]
    if m_family:
        out["ub_bonferroni_one_sided"] = q(1 - ALPHA / m_family)
        out["ci_bonferroni_two_sided"] = [q(ALPHA / (2 * m_family)), q(1 - ALPHA / (2 * m_family))]
    return out


def r3(x):
    return None if x is None else float(f"{x:.4g}")


# ------------------------------------------------------------------------------------------------ 1. 2x2
def code_identity_checks():
    """Verify the cell identities from source: allocation kind + certificate routine."""
    def cert_calls(cls):
        return inspect.getsource(cls.certify)
    b4, bb = uniform_rs.UniformRS(), b4_bal.B4Bal(0.5)
    f, qh = fdc.FDCMethod(), fdc.FDCAblation("half", "qstar", "r4")
    qp = frontier_common.QFCMethod()
    hs = hait_sw.HaitSW(p_min=0.02, gamma=2 / 3)
    rect_b4 = "rect_certificate(ctx, st)" in cert_calls(uniform_rs.UniformRS)
    rect_bb = "rect_certificate(ctx, st)" in cert_calls(b4_bal.B4Bal)
    rect_hs = "rect_certificate(ctx, st)" in cert_calls(hait_sw.HaitSW)
    qfc_inherit = (fdc.FDCMethod.certify is frontier_common.QFCMethod.certify
                   and fdc.FDCAblation.certify is frontier_common.QFCMethod.certify)
    return {
        "B4": {"class": "uniform_rs.UniformRS", "alloc_kind": b4.alloc_kind, "certificate": "frontier_common.rect_certificate",
               "certify_calls_rect_certificate": rect_b4},
        "B4-bal": {"class": "b4_bal.B4Bal(share=0.5)", "alloc_kind": bb.alloc_kind, "share": bb.share,
                   "certificate": "frontier_common.rect_certificate", "certify_calls_rect_certificate": rect_bb},
        "Hait-SW": {"class": "hait_sw.HaitSW(p_min=0.02, gamma=2/3)", "alloc_kind": hs.alloc_kind,
                    "certificate": "frontier_common.rect_certificate", "certify_calls_rect_certificate": rect_hs},
        "QFC-pool": {"class": "frontier_common.QFCMethod()", "alloc_kind": qp.alloc_kind,
                     "certificate": "QFCMethod.certify (quadknap.qfc_certificate_enum), r4 ledger (qstar, 0.04/0.01/48)"},
        "F[half,qstar,r4] (QFC-half)": {"class": "fdc.FDCAblation('half','qstar','r4')", "alloc_kind": qh.alloc_kind,
                                        "certificate": "QFCMethod.certify inherited, r4 ledger (qstar, 0.04/0.01/48)"},
        "FDC": {"class": "fdc.FDCMethod()", "alloc_kind": f.alloc_kind,
                "certificate": "QFCMethod.certify inherited, r5 ledger (feas union 3386, 0.045/0.005/S*A)"},
        "verdict": {
            "B4_and_B4bal_same_certificate_routine": bool(rect_b4 and rect_bb),
            "B4_pool_vs_B4bal_fixed": (b4.alloc_kind, bb.alloc_kind) == ("pool", "fixed"),
            "FDC_and_ablations_inherit_QFC_certify": bool(qfc_inherit),
            "same_per_cell_delta": "both rect rivals use StreamEngine(delta_cell = delta / (S A)) WSR20 WoR CS "
                                   "(frontier_runner.run_stream); identical by construction",
            "note": "B4 = pool design + rect certificate; B4-bal = frozen 50/50 design + the SAME rect certificate "
                    "routine; FDC / QFC-half / QFC-pool share QFCMethod.certify and differ only in design and ledger."}}


def digest_checks(main, fac):
    def same(m1, rows1, m2, rows2):
        n = sum(rows1[(m1, s)]["digest"] == rows2[(m2, s)]["digest"] for s in SEEDS)
        return {"pair": f"{m1} vs {m2}", "n_same": int(n), "n": len(SEEDS)}
    return [same("FDC", main, "B4-bal", main),
            same("FDC", main, "F[half,qstar,r4]", fac),
            same("FDC", main, "F[half,feas,r5]", fac),
            same("B4-bal", main, "B4-bal", fac),
            same("B4", main, "QFC-pool", main),
            same("B4", main, "F[pool,qstar,r4]", fac),
            same("QFC-pool", main, "F[pool,feas,r5]", fac),
            same("Hait-SW", main, "B4-bal", main)]


def interaction(Npr, Npq, Nhr, Nhq, idx):
    I = np.log(Nhq) - np.log(Npq) - np.log(Nhr) + np.log(Npr)
    bt = I[idx].mean(1)
    return {"I_mean_log": float(I.mean()), "ci95": [float(np.quantile(bt, 0.025)), float(np.quantile(bt, 0.975))],
            "exp_I": float(np.exp(I.mean())),
            "exp_I_ci95": [float(np.exp(np.quantile(bt, 0.025))), float(np.exp(np.quantile(bt, 0.975)))],
            "frac_streams_I_pos": float((I > 0).mean()),
            "definition": "I = log N(half,QFC) - log N(pool,QFC) - log N(half,rect) + log N(pool,rect), mean over "
                          "200 paired streams; I > 0 = joint gain smaller than the product of the two single gains"}


def two_by_two(main, fac, idx):
    G = lambda v: float(np.exp(np.log(v).mean()))  # noqa: E731
    N = {"B4": arr(main, "B4", SEEDS), "QFC-pool": arr(main, "QFC-pool", SEEDS), "B4-bal": arr(main, "B4-bal", SEEDS),
         "QFC-half": arr(fac, "F[half,qstar,r4]", SEEDS), "FDC": arr(main, "FDC", SEEDS),
         "F[pool,feas,r5]": arr(fac, "F[pool,feas,r5]", SEEDS),
         "F[pool,qstar,r4]": arr(fac, "F[pool,qstar,r4]", SEEDS), "B4-bal(fac)": arr(fac, "B4-bal", SEEDS),
         "F[half,feas,r5]": arr(fac, "F[half,feas,r5]", SEEDS)}
    ident = {"F[pool,qstar,r4]==QFC-pool": int((N["F[pool,qstar,r4]"] == N["QFC-pool"]).sum()),
             "B4-bal(factorial)==B4-bal(main)": int((N["B4-bal(fac)"] == N["B4-bal"]).sum()),
             "F[half,feas,r5]==FDC": int((N["F[half,feas,r5]"] == N["FDC"]).sum()), "n": len(SEEDS)}
    cells = {"pool x rect (B4)": G(N["B4"]), "pool x QFC-r4 (QFC-pool)": G(N["QFC-pool"]),
             "half x rect (B4-bal)": G(N["B4-bal"]), "half x QFC-r4 (QFC-half = F[half,qstar,r4])": G(N["QFC-half"]),
             "half x QFC-r5 (FDC = F[half,feas,r5])": G(N["FDC"]),
             "pool x QFC-r5 (F[pool,feas,r5])": G(N["F[pool,feas,r5]"])}
    P = lambda a, b: paired(N[a], N[b], idx)  # noqa: E731
    main_2x2 = {
        "design_effect_under_rect (B4-bal/B4)": P("B4-bal", "B4"),
        "design_effect_under_QFC (QFC-half/QFC-pool)": P("QFC-half", "QFC-pool"),
        "cert_effect_under_pool (QFC-pool/B4)": P("QFC-pool", "B4"),
        "cert_effect_under_half (QFC-half/B4-bal)": P("QFC-half", "B4-bal"),
        "joint (QFC-half/B4)": P("QFC-half", "B4"),
        "ledger+union step (FDC/QFC-half)": P("FDC", "QFC-half"),
        "FDC/B4": P("FDC", "B4"), "FDC/B4-bal": P("FDC", "B4-bal"), "FDC/QFC-pool": P("FDC", "QFC-pool"),
        "interaction_I (QFC-r4 certificate)": interaction(N["B4"], N["QFC-pool"], N["B4-bal"], N["QFC-half"], idx)}
    sens = {"design_effect_under_QFC-r5 (FDC/F[pool,feas,r5])": P("FDC", "F[pool,feas,r5]"),
            "cert_effect_under_pool_QFC-r5 (F[pool,feas,r5]/B4)": P("F[pool,feas,r5]", "B4"),
            "cert_effect_under_half_QFC-r5 (FDC/B4-bal)": P("FDC", "B4-bal"),
            "interaction_I (QFC-r5 certificate: pool cell F[pool,feas,r5], half cell FDC)":
                interaction(N["B4"], N["F[pool,feas,r5]"], N["B4-bal"], N["FDC"], idx)}
    return {"cells_geomean_N80_pen": cells, "identities_200": ident, "main": main_2x2, "sensitivity_QFC_r5": sens,
            "reviewer_reference": {"I": 0.1066, "ci95": [0.0715, 0.1430], "cells": [5389443, 2489827, 2467271, 1268060]}}


# ------------------------------------------------------------------------------------------------ 2. per budget
def structural_ctx():
    L = json.loads((WS / "plan" / "prereg_lock.json").read_text())
    sc = L["structural_ctx"]["CR9_eval"]
    N = np.array(sc["pool_sizes"], dtype=np.int64)
    w = N.sum(1) / N.sum()
    return make_ctx(w, N, fr.cr_problems("visit"), 0.001, 0.05, CK, 1, TAU)


def per_budget(main, audit):
    ctx = structural_ctx()
    ckarr = np.array(CK, dtype=float)
    fdc_stop = np.array([main[("FDC", s)]["cert_k"] for s in SEEDS])
    fdc_full = np.array([audit[("FDC", s)]["cert_k"] for s in SEEDS])
    prefix_ok = bool(np.all((fdc_stop < 0) | (fdc_stop == fdc_full)))
    T_full = ckarr[fdc_full]  # all >= 0 (200/200 reach 15/15)
    out = {"budgets": BUDGETS, "fdc_full_horizon_all_certified": bool((fdc_full >= 0).all()),
           "fdc_stop_prefix_consistent_with_full_horizon": prefix_ok, "completion_at_stop": {}, "rivals": {},
           "fdc_full_horizon_median_cert_time": [float(np.median(T_full[:, q])) for q in range(15)],
           "fdc_full_horizon_geomean_cert_time": [float(np.exp(np.log(T_full[:, q]).mean())) for q in range(15)]}
    for m in ["FDC"] + R_SET:
        CKm = np.array([main[(m, s)]["cert_k"] for s in SEEDS])
        out["completion_at_stop"][m] = (CKm >= 0).mean(0).round(4).tolist()
    for m in R_SET:
        CKm = np.array([main[(m, s)]["cert_k"] for s in SEEDS])
        kst = np.array([main[(m, s)]["k_stop"] for s in SEEDS])
        rb = {"both_certified_stop_window": [], "ub_fdc_full_vs_rival_lower": [], "rival_median_cert_time_stop": []}
        for q in range(15):
            both = (fdc_stop[:, q] >= 0) & (CKm[:, q] >= 0)
            n = int(both.sum())
            if n >= 2:
                a, b = ckarr[fdc_stop[both, q]], ckarr[CKm[both, q]]
                p = paired(a, b, boot_idx(n))
                rb["both_certified_stop_window"].append({"budget": BUDGETS[q], "n": n, "ratio": r3(p["ratio"]),
                                                         "ci95": [r3(x) for x in p["ci95_two_sided"]],
                                                         "frac_faster": r3(p["frac_faster"]),
                                                         "frac_tied": r3(p["frac_tied"])})
            else:
                rb["both_certified_stop_window"].append({"budget": BUDGETS[q], "n": n, "ratio": None})
            # conservative bound on every stream: FDC exact full-horizon time / rival lower bound
            # (rival cert time if certified by its stop, else its stop checkpoint, which is < its true time)
            riv = np.where(CKm[:, q] >= 0, ckarr[np.clip(CKm[:, q], 0, None)], ckarr[kst])
            p = paired(T_full[:, q], riv, boot_idx(200))
            rb["ub_fdc_full_vs_rival_lower"].append({"budget": BUDGETS[q], "n": 200,
                                                     "n_rival_uncertified_at_stop": int((CKm[:, q] < 0).sum()),
                                                     "ratio_upper_bound": r3(p["ratio"]),
                                                     "ci95": [r3(x) for x in p["ci95_two_sided"]],
                                                     "exact": bool((CKm[:, q] >= 0).all())})
            tt = ckarr[CKm[CKm[:, q] >= 0, q]]
            rb["rival_median_cert_time_stop"].append(float(np.median(tt)) if len(tt) else None)
        out["rivals"][m] = rb
    # structural explanation: feasible-class size and Hamming-1 feasible neighbours of FDC's decided answer
    pols = ctx.pols
    struct = []
    dec = np.array([audit[("FDC", s)]["decided_pi"] for s in SEEDS])
    for q in range(15):
        vals, cnt = np.unique(dec[:, q], return_counts=True)
        pi = int(vals[np.argmax(cnt)])
        ham = (pols != pols[pi][None, :]).sum(1)
        feas = ctx.feas[q]
        struct.append({"budget": BUDGETS[q], "n_feasible": int(feas.sum()), "modal_decided_pi": pi,
                       "modal_share": float(cnt.max() / len(SEEDS)), "n_distinct_answers": int(len(vals)),
                       "treated_share_of_modal": float((ctx.w * pols[pi]).sum()),
                       "feasible_hamming1_neighbours": int(((ham == 1) & feas).sum()),
                       "feasible_hamming2_neighbours": int(((ham == 2) & feas).sum()),
                       "budget_slack": float(BUDGETS[q] - (ctx.w * pols[pi]).sum())})
    out["structure"] = struct
    out["segment_weights"] = ctx.w.round(5).tolist()
    # dev-half eps-optimal set sizes (r4 guard table, dev half only; no new eval truth read)
    gt = json.loads((WS / "exp/results/pilots/r4_g_eps_select/guard_tables.json").read_text())["CR9|dev"]
    out["dev_eps_set_size_0.001"] = [int(r["eps_set_size"]["0.001"]) for r in gt]
    return out


# ------------------------------------------------------------------------------------------------ 3. N100
def n100(main, audit):
    fN100 = arr(audit, "FDC", SEEDS, "N100_pen")
    res = {"FDC_geomean_N100_pen": float(np.exp(np.log(fN100).mean())),
           "FDC_completed_15_of_15": int(sum(bool(audit[("FDC", s)]["completed_15"]) for s in SEEDS)),
           "FDC_false_streams_full": int(sum(bool(audit[("FDC", s)]["fwer_event_full"]) for s in SEEDS)),
           "rivals": {}, "continuation_needed": {}}
    for m in R_SET:
        rr = [main[(m, s)] for s in SEEDS]
        full15 = np.array([all(k >= 0 for k in r["cert_k"]) and not r["fwer_event"] for r in rr])
        rN100 = np.array([CK[max(r["cert_k"])] if f else np.nan for r, f in zip(rr, full15)], dtype=float)
        rN80 = np.array([r["N80_pen"] for r in rr])
        d = {"n_15of15_at_stop": int(full15.sum()), "n_need_continuation": int((~full15).sum())}
        if full15.sum() >= 2:
            p = paired(fN100[full15], rN100[full15], boot_idx(int(full15.sum())))
            d["exact_subset"] = {"n": p["n"], "ratio": p["ratio"], "ci95": p["ci95_two_sided"],
                                 "note": "selection: only streams where the rival already had 15/15 at its own stop"}
        pb = paired(fN100, rN80, boot_idx(200))
        d["ub_vs_rival_N80"] = {"ratio_upper_bound": pb["ratio"], "ci95": pb["ci95_two_sided"],
                                "ub95_one_sided": pb["ub95_one_sided"],
                                "note": "FDC N100 / rival N80_pen; rival N100 >= N80 so this bounds the N100 ratio above"}
        hyb = np.where(full15, rN100, rN80)
        ph = paired(fN100, hyb, boot_idx(200))
        d["ub_hybrid"] = {"ratio_upper_bound": ph["ratio"], "ci95": ph["ci95_two_sided"], "ub95_one_sided": ph["ub95_one_sided"],
                          "note": "exact rival N100 where 15/15 at stop, else rival N80 (lower bound on its N100)"}
        res["rivals"][m] = d
        res["continuation_needed"][m] = [s for s, f in zip(SEEDS, full15) if not f]
    return res


def continuation_spec(main, nres):
    """Exact re-run spec + CPU estimate (NOT executed)."""
    spec = {"status": "SPEC ONLY - not run", "label": LABEL,
            "runner": "new task r5_supp_n100_cont (copy of exp/code/run_r5_cr_fwer_audit.py with --methods R_SET), "
                      "same lock_gate pattern but tagged post_unblinding_exploratory",
            "setting": {"layer": "CR9", "half": "eval", "eps": 0.001, "K": 20, "delta": 0.05, "checkpoints": CK,
                        "stop_rule": "ctx = dataclasses.replace(ctx, stop_k = Q = 15); run to 15/15 or tau_R = 6,989,799",
                        "frozen_configs": {"B4-bal": {"share": 0.5}, "Hait-SW": {"p_min": 0.02, "gamma": 2 / 3},
                                           "B2-rect": {"explore_c": 1.0}},
                        "seeds": "same eval permutation seeds as main (30000-30199); the run is deterministic in the "
                                 "seed, so it is a continuation of the main stream, not a new draw"},
            "acceptance_checks": ["12/15 prefix: k12 from the full-horizon run == main-block N80_pen on every rerun stream "
                                  "(as r5_cr_fwer_audit did for FDC); any mismatch -> BLOCKED",
                                  "schedule_digest == main-block digest on every stream",
                                  "billing_ok on every stream; false certificates reported (N100_pen = tau_R if any)"],
            "analysis": "paired geomean FDC N100_pen / rival N100_pen over all 200 streams (rival N100 = exact from main "
                        "where 15/15 at stop, else from continuation), B = 1e4 bootstrap, seed 42; report as exploratory",
            "per_method": {}}
    tot_min, tot_all = 0.0, 0.0
    for m in R_SET:
        need = nres["continuation_needed"][m]
        secs, secs_all = [], []
        for s in SEEDS:
            r = main[(m, s)]
            sec = float(r["sec"] or 0.0)
            # linear-in-arrivals upper estimate: run from 0 to tau_R
            full = sec * TAU / max(r["N80_pen"], 1)
            secs_all.append(full)
            if s in need:
                secs.append(full)
        spec["per_method"][m] = {"n_streams_needed": len(need), "seeds": need,
                                 "est_cpu_sec_needed": round(sum(secs), 1),
                                 "est_cpu_sec_all_200": round(sum(secs_all), 1),
                                 "main_block_mean_sec_per_stream": round(float(np.mean([main[(m, s)]["sec"] for s in SEEDS])), 2)}
        tot_min += sum(secs)
        tot_all += sum(secs_all)
    spec["cpu_estimate"] = {
        "method": "per stream: main-block wall sec x tau_R / N80_pen (linear in arrivals, run to tau_R = upper bound); "
                  "main-block sec were measured with 4 concurrent workers and are biased up",
        "needed_streams_cpu_hours": round(tot_min / 3600, 2),
        "needed_streams_wall_hours_4_workers": round(tot_min / 3600 / 4, 2),
        "all_200x9_cpu_hours": round(tot_all / 3600, 2),
        "all_200x9_wall_hours_4_workers": round(tot_all / 3600 / 4, 2),
        "recommendation": "run all 200 streams for every rival with nonzero need (gives a prefix replica check on the "
                          "already-complete streams for free); Peace-rect dominates the cost"}
    return spec


# ------------------------------------------------------------------------------------------------ 4. C1 bounds
def c1_bounds(main, k60, cr12):
    idx = boot_idx(200)
    F = arr(main, "FDC", SEEDS)
    rows = {}
    for r in R_SET:
        rows[r] = paired(F, arr(main, r, SEEDS), idx, m_family=len(R_SET))
    # Holm step-down levels: order by p (ties broken by ratio, largest ratio = weakest = last)
    order = sorted(R_SET, key=lambda r: (rows[r]["p_boot_one_sided"], rows[r]["ratio"]))
    bt_cache = {}
    for i, r in enumerate(order):
        d = np.log(F) - np.log(arr(main, r, SEEDS))
        bt = d[idx].mean(1)
        lvl = 1 - ALPHA / (len(R_SET) - i)
        rows[r]["holm_step"] = i + 1
        rows[r]["holm_level_one_sided"] = lvl
        rows[r]["ub_holm_step_one_sided"] = float(np.exp(np.quantile(bt, lvl)))
        bt_cache[r] = bt
    s60 = [s for s in SEEDS if ("FDC", s) in k60]
    k60r = {}
    for r in ["B4-bal", "B2-rect", "Peace-rect", "QFC-pool"]:
        k60r[r] = paired(arr(k60, "FDC", s60), arr(k60, r, s60), boot_idx(len(s60)), m_family=4)
    s12 = [s for s in SEEDS if ("FDC", s) in cr12]
    c12r = {}
    for r in ["B1", "B4", "B4-bal", "Hait-SW"]:
        c12r[r] = paired(arr(cr12, "FDC", s12), arr(cr12, r, s12), boot_idx(len(s12)), m_family=4)
    hb = rows["B4-bal"]
    head = {"K20_CR9_B4-bal": {"ratio": hb["ratio"], "speedup": hb["speedup"], "ci95": hb["ci95_two_sided"],
                               "ub_bonf": hb["ub_bonferroni_one_sided"]},
            "K60_CR9_B4-bal": {"ratio": k60r["B4-bal"]["ratio"], "speedup": k60r["B4-bal"]["speedup"],
                               "ci95": k60r["B4-bal"]["ci95_two_sided"]},
            "CR12_B4-bal": {"ratio": c12r["B4-bal"]["ratio"], "speedup": c12r["B4-bal"]["speedup"],
                            "ci95": c12r["B4-bal"]["ci95_two_sided"]}}
    sp = [head[k]["speedup"] for k in head]
    head["speedup_range_point"] = [min(sp), max(sp)]
    lo_ci = min(1 / head[k]["ci95"][1] for k in ["K20_CR9_B4-bal", "K60_CR9_B4-bal", "CR12_B4-bal"])
    hi_ci = max(1 / head[k]["ci95"][0] for k in ["K20_CR9_B4-bal", "K60_CR9_B4-bal", "CR12_B4-bal"])
    head["speedup_range_ci_envelope"] = [lo_ci, hi_ci]
    hardest = max(R_SET, key=lambda r: rows[r]["ub95_one_sided"])
    return {"K20_CR9_main": rows, "K60_CR9_100streams": k60r, "CR12_50streams": c12r, "headline_B4bal": head,
            "hardest_rival_by_ub": hardest,
            "note": "Bonferroni family m = 9 (K20) or 4 (K60, CR12 rivals run there); Holm step levels are the "
                    "step-down levels at which each rival is tested (ordering by bootstrap p, all p at the 1/(B+1) "
                    "floor, ties broken by ratio). p_boot is not null-centred (approximate)."}


# ------------------------------------------------------------------------------------------------ 5. deviations
def deviations():
    D = []

    def add(**k):
        D.append(k)
    add(id="DEV-R5-01", source="decision_log_r5 DL-01; lock declarations[0]; MEMORY.md 2026-10-03",
        when="2026-10-03, between 03:28:13 and 12:56:34 (+10:00); first in git 47a13eef 13:09:26",
        what="Headline narrowed from 'faster than published plug-in configurations (B2-fav, Peace-fav)' to 'faster than "
             "every listed rigorous competitor in R'; plug-in / asymptotic methods (B2-fav, Peace-fav, B3-fav, B5, H8) "
             "reclassified as descriptive comparators.",
        why="Round-4 dev block: QFC 1.67x slower than B2-fav but 0.39-0.77x every rigorous rival; synthesizer escalation "
            "that no candidate beats tuned plug-ins ported with balanced reads.",
        decided_by="user (explicit ruling)", evidence="plan/decision_log_r5.md#DL-01; exp/results/r4_gate_decision.json "
        "(variant_ratios_QFC_over_X); idea/r5_offline/contrarian/ci_out.txt",
        effect_on_verdict="Defines C1's comparison set; verdict would differ under the old H1 (FDC/B2-fav-tight = 1.084, "
                          "FIX-bal-fav = 1.098 > 1).",
        disclosure="Deviation paragraph opens with it; Table 1 lists plug-ins with 'faster but without guarantee'.")
    add(id="DEV-R5-02", source="DL-03; lock declarations[1]",
        when="2026-10-03, after DL-01 dev data, before lock v5 (16:20:44)",
        what="Effect-size gate changed from ratio <= 0.67 (r4 G1-shape / innovator) via 'UB < 1 and point <= 0.90' to "
             "the IUT rule 'one-sided 95% UB < 1.0 per rival'; 0.90 kept only as a wording gate ('faster by >= 10%').",
        why="0.67 judged unreachable under the rigorous framing by r5 theory analysis; 0.90 hard gate unresolved.",
        decided_by="planner/methodology under the user ruling", evidence="plan/decision_log_r5.md#DL-03; plan/methodology.md:118",
        effect_on_verdict="None realised: every rival ratio <= 0.487, i.e. would also pass 0.67 and 0.90 (UB <= 0.499).",
        disclosure="State both the relaxed gate and that the observed ratios also clear the original 0.67 gate.")
    add(id="DEV-R5-03", source="DL-04; lock declarations[1]",
        when="2026-10-03, planning (before lock v5)",
        what="C3 (non-inferiority vs tuned plug-ins, UB <= 1.15) demoted to the report-only E-cost estimate (no pass line).",
        why="Follows from DEV-R5-01 (plug-ins descriptive); dev ratio of an FDC proxy vs B2-tight was 1.110 (close to 1.15).",
        decided_by="planner under the user ruling", evidence="plan/decision_log_r5.md#DL-04",
        effect_on_verdict="None on the verdict; observed E-cost 1.084 / 1.098 (UB 1.110 / 1.123) would have passed 1.15.",
        disclosure="Report E-cost with two-sided CI and say it was demoted from a gate after dev data.")
    add(id="DEV-R5-04", source="aggregate claims.json C4; lock c4.prediction_model",
        when="2026-10-03 (eval aggregate, after unblinding)",
        what="C4 mechanism wording downgraded from 'predictive' (c4_mode at gate) to 'descriptive decomposition only': "
             "frozen-prediction max |log err| = 0.264 > 0.095.",
        why="Pre-registered rule triggered (not a discretionary change).", decided_by="pre-registered rule",
        evidence="exp/results/full/r5_analysis_aggregate/claims.json#C4; exp/results/r5_gates/c4_model.json",
        effect_on_verdict="None (C4 decides wording only).",
        disclosure="Say the frozen prediction model failed its error limit; report design share 0.924 with its QFC-pool "
                   "reference frame and the 2x2 with interaction I.")
    add(id="DEV-R5-05", source="DL-05; lock declarations[2]",
        when="2026-10-02 23:22:39 (r4_g_eps_select)",
        what="eps* = 0.001 chosen by a guard that loaded eval-half row-level outcomes to compute aggregate truth "
             "(per-cell means -> J_star, second_gap, eps_set_size, n_nontrivial_eval); no eval permutation seed or eval "
             "stream was used.",
        why="eps guard (non-trivial-problem count) was specified on both halves in r4.", decided_by="r4 protocol",
        evidence="plan/decision_log_r5.md#DL-05; exp/code/run_r4_g_eps_select.py:285,298; "
                 "exp/results/pilots/r4_g_eps_select/guard_tables.json; exp/results/r4_gates/eps.json",
        effect_on_verdict="Not binding on CR9 (15/15 non-trivial at all four eps), so eps* unchanged without the guard; "
                          "but the eval half is not an untouched test set.",
        disclosure="Never write 'evaluation data were never read'; write 'evaluation seeds and evaluation-stream results "
                   "were never used for any method or gate comparison; aggregate truth of the evaluation half was read "
                   "once for an eps guard that did not bind'.")
    add(id="DEV-R5-06", source="DL-06; lock declarations[3]",
        when="2026-10-03 02:36:16 (offline_ctr.py mtime), after r4 results (02:12:14), concurrent with first r5 dev runs",
        what="C_var 48 -> S*A = 18 and delta split 0.04/0.01 -> 0.045/0.005; union Q*|Pi| = 7680 -> sum_q|Pi_Bq| = 3386.",
        why="Exact value replaces a conservative constant; the split was never searched.",
        decided_by="research team (exploratory code, then methodology)", evidence="plan/decision_log_r5.md#DL-06",
        effect_on_verdict="Small: ledger+union step FDC/QFC-half ~ 0.95 (see 2x2); lock wording 'before dev results' "
                          "is only true relative to r5 gates, not to r4 dev results.",
        disclosure="Correct the lock wording: chosen after r4 dev results, no run varied it.")
    add(id="DEV-R5-07", source="DL-07; lock declarations[4]", when="2026-10-03 02:33:51",
        what="50/50 design chosen after seeing r4 results (dev shares 0.30-0.60 explored); FDC locks 0.50 though 0.40 was "
             "slightly faster on dev.", why="Motivated by the r4 failure mode (pool proportions starve control).",
        decided_by="research team", evidence="plan/decision_log_r5.md#DL-02, DL-07",
        effect_on_verdict="Forking-path risk; not tuned toward the faster dev value.",
        disclosure="Forking-path paragraph with the DL-02 exploration table.")
    add(id="DEV-R5-08", source="DL-02, DL-09", when="2026-10-03 02:30-12:56 (dev seeds 900-999 only)",
        what="Exploration of share {0.30..0.60}, beta {8..14.1, default}, K {20, 60}, rig/fav/tight variants; beta = 14.1 "
             "results (e.g. 0.856) downgraded (union bound 0.05595 > delta).",
        why="Exploratory design search.", decided_by="research team", evidence="plan/decision_log_r5.md#DL-02, DL-09",
        effect_on_verdict="None (dev only, excluded from confirmatory evidence).",
        disclosure="Full exploration table in the appendix.")
    add(id="DEV-R5-09", source="DL-08", when="2026-10-03 planning",
        what="Two-stage read schedule and the A = 2 Cauchy-Schwarz tight-accounting module deleted (the proposal's "
             "'cut if unproven by 10-06' rule was not the actual reason; removed proactively on 10-03).",
        why="Reduce forking paths.", decided_by="planner", evidence="plan/decision_log_r5.md#DL-08",
        effect_on_verdict="None (removes options).", disclosure="One sentence in the deviation paragraph.")
    add(id="DEV-R5-10", source="lock rival_configs.disclosures; DL-10; external reviewer review s2 (3rd point)",
        when="2026-10-03 13:52 (rival tuning freeze) / eval aggregate",
        what="Hait-SW tuned config (p_min = 0.02, gamma = 2/3, adaptive) gives N80 identical to B4-bal on 200/200 eval "
             "streams although its schedule digests differ on 200/200 streams: N80 coincidence on the coarse K = 20 "
             "grid, not the same algorithm. The aggregate table note 'reduces to the B4-bal design' is wrong.",
        why="Coarse checkpoint lattice + near-balanced Hait shares.", decided_by="observation (no decision)",
        evidence="exp/results/full/supp_r5_p0/two_by_two.json#digests; exp/results/full/r5_analysis_aggregate/table1.md",
        effect_on_verdict="IUT over 9 names has 8 distinct N80 profiles; verdict unchanged.",
        disclosure="Write 'N80 coincides with B4-bal on every stream (different adaptive schedule)'; merge rows or footnote.")
    add(id="DEV-R5-11", source="lock rival_configs (untuned_rivals); external reviewer review s2 (2nd point)",
        when="2026-10-03 (rival tuning, seeds 900-949)",
        what="Tuning-budget asymmetry: FDC's development explored share, threshold, checkpoint grid and ledger; rivals got "
             "B4-bal share {0.40,0.45,0.50}, Hait 6 configs, B2-rect 2 configs; B1, B3-rect, Peace-rect, Molitor, "
             "QFC-pool and the shared WSR20 rectangle certificate untuned.",
        why="Rival grids pre-declared in methodology s4.4; the common certificate kept at published constants.",
        decided_by="planner", evidence="plan/prereg_lock.json#rival_configs; plan/decision_log_r5.md#DL-02",
        effect_on_verdict="Unknown share of the certificate advantage could come from tuning asymmetry; verdict as locked.",
        disclosure="Table of tuning budgets per method; P1 stress test with a tightened checkpoint rectangle rival.")
    add(id="DEV-R5-12", source="plan/theory/fdc_theorem.md; external reviewer review s2 (1st point)",
        when="by construction (method spec)",
        what="Guarantee scope differs: FDC is valid at the K = 20 pre-specified checkpoints (union over k), the rectangle "
             "rivals' per-cell WSR20 CS are time-uniform (anytime-valid in each cell's local time).",
        why="Design of FDC-1.", decided_by="method specification",
        evidence="plan/theory/fdc_theorem.md; exp/code/dsswm/baselines/frontier_common.py (rect_certificate)",
        effect_on_verdict="None on the locked verdict; the ~2x certificate effect mixes certificate shape and guarantee "
                          "strength.",
        disclosure="Write 'valid at K pre-specified checkpoints', never 'anytime-valid'; state the confound.")
    add(id="DEV-R5-FULL-01", source="exp/results/full/deviations_r5_full.json",
        when="2026-10-03T22:35:15",
        what="r5_replica_check runner v1 -> v2 after v1 BLOCKED on factorial runner hash 91866273 (= commit 262ce6c0); "
             "diff adds a _digest() helper in the summary-stage identity check only.",
        why="Checker false positive on a summary-only change.", decided_by="authors (diff review)",
        evidence="exp/results/full/deviations_r5_full.json; full/r5_replica_check/summary_v1_blocked.json",
        effect_on_verdict="None (580/580 v1 field matches; 180/180 v2 factorial reruns match).",
        disclosure="Footnote in reproducibility appendix.")
    add(id="DEV-R5-13", source="external reviewer review s1; aggregate table note", when="eval aggregate / result debate 2026-10-03",
        what="Interpretation correction: B4 (20/200), B3-rect (52/200), Molitor-WoR (200/200) reach 12/15 exactly at "
             "t_K = tau_R and are completed under lock v5 (0/200 censored); 0.223/0.369/0.172 are endpoint ratios, "
             "not lower bounds. Debate synthesis text calling them censored is withdrawn.",
        why="Lock endpoint definition (prereg_lock.json endpoints.N80_pen).", decided_by="external reviewer review / authors",
        evidence="plan/prereg_lock.json:382; exp/results/full/r5_analysis_aggregate/table1.csv (At tau_R)",
        effect_on_verdict="None.", disclosure="Report 'reached 12/15 only at full-log exhaustion' counts; don't call them censored.")
    add(id="DEV-R5-14", source="external reviewer review s4; supervisor appendix item 8", when="2026-10-03",
        what="Practical-significance rows/MB saved are logical (rows billed); the replay engine preloads the full pool, "
             "so no measured I/O saving.", why="Implementation of frontier_runner.", decided_by="observation",
        evidence="exp/code/dsswm/streams/frontier_runner.py:45", effect_on_verdict="None.",
        disclosure="Say 'logged rows that need not be read'; no system speed-up claim.")
    return D


def dev_md(D):
    L = [f"# Consolidated deviation table, round 5 (cand_fdc)\n", f"> {LABEL}. Merges DEV-R5-FULL-01 "
         "(deviations_r5_full.json), the 8 lock v5 declarations, decision_log_r5.md DL-01..DL-10 and the external reviewer result-"
         "debate corrections. Generated by exp/code/run_r5_supp_p0.py.\n",
         "| id | when | what | why | who decided | evidence | effect on verdict | disclosure in paper |",
         "|---|---|---|---|---|---|---|---|"]
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    for d in D:
        L.append("| " + " | ".join(esc(d[k]) for k in ("id", "when", "what", "why", "decided_by", "evidence",
                                                          "effect_on_verdict", "disclosure")) + " |")
    L.append("\nNo deviation changes the locked verdict `positive_result_achieved`; DEV-R5-01/02/03/05/06/07/11/12 "
             "bound how the result may be worded.\n")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    srcs = {"main": [FULL / "r5_cr_main_a/results.jsonl", FULL / "r5_cr_main_b/results.jsonl"],
            "factorial": [FULL / "r5_cr_factorial/results.jsonl"], "audit": [FULL / "r5_cr_fwer_audit/results.jsonl"],
            "k60": [FULL / "r5_cr_k60_sens/results.jsonl"], "cr12": [FULL / "r5_cr12_scale/results.jsonl"]}
    before = {str(p.relative_to(WS)): sha(p) for v in srcs.values() for p in v}
    main_, fac, audit = load(srcs["main"]), load(srcs["factorial"]), load(srcs["audit"])
    k60, cr12 = load(srcs["k60"]), load(srcs["cr12"])
    idx = boot_idx(200)
    meta = {"label": LABEL, "written_at": datetime.now().isoformat(), "inputs_sha256": before,
            "bootstrap": {"B": B_BOOT, "seed": SEED, "type": "paired percentile, resample streams"},
            "script": "exp/code/run_r5_supp_p0.py"}

    tbt = two_by_two(main_, fac, idx)
    tbt["code_checks"] = code_identity_checks()
    tbt["digests"] = digest_checks(main_, fac)
    tbt["meta"] = meta
    (OUT / "two_by_two.json").write_text(json.dumps(tbt, indent=1))

    pb = per_budget(main_, audit)
    pb["meta"] = meta
    (OUT / "per_budget.json").write_text(json.dumps(pb, indent=1))

    nr = n100(main_, audit)
    spec = continuation_spec(main_, nr)
    nr["meta"] = meta
    (OUT / "n100.json").write_text(json.dumps(nr, indent=1))
    (OUT / "n100_continuation_spec.json").write_text(json.dumps(spec, indent=1))

    cb = c1_bounds(main_, k60, cr12)
    cb["meta"] = meta
    (OUT / "c1_bounds.json").write_text(json.dumps(cb, indent=1))

    D = deviations()
    (OUT / "deviations_complete.json").write_text(json.dumps({"label": LABEL, "deviations": D}, indent=1))
    (OUT / "deviations_complete.md").write_text(dev_md(D))

    after = {str(p.relative_to(WS)): sha(p) for v in srcs.values() for p in v}
    assert before == after, "input files changed during the run"
    print(json.dumps({"two_by_two_cells": tbt["cells_geomean_N80_pen"],
                      "I": tbt["main"]["interaction_I (QFC-r4 certificate)"],
                      "I_r5": tbt["sensitivity_QFC_r5"]["interaction_I (QFC-r5 certificate: pool cell F[pool,feas,r5], half cell FDC)"],
                      "digests": tbt["digests"], "ident": tbt["identities_200"], "code": tbt["code_checks"]["verdict"],
                      "cpu": spec["cpu_estimate"], "head": cb["headline_B4bal"]}, indent=1))


if __name__ == "__main__":
    main()
