#!/usr/bin/env python3
"""r9c: development-half decision difficulty at the block-G1 tolerances (critic r9, P1-B / R9c-B).

POST HOC, DESCRIPTIVE, DEVELOPMENT HALF.  Analysis of existing data only; no stream is run and no evaluation-half
outcome is read.

  DEV  -- for every (log, S, eps) of the block-G1 grid (run_v10_posthoc_G.EPS_GRID; S in {16, 32, 64}) the same
          development-half diagnostics as writing/scripts/difficulty_r9.py, computed with ITS functions (dev_seg:
          frozen lock-v10 design loader SegEnvV10Frozen(data, S, 'dev', frozen_sha256=<lock v10>), the exact counters
          enumeration / meet-in-the-middle / branch-and-bound with the rigorous rounding-DP bracket, TOL 1e-12):
          pooled share, mean share, all-control count, eps / dev base rate, eps / dev headroom (J*_15 - J(all-control)).
          Block G1's development streams use the same frozen design (run_v10_posthoc_G.init_env: SegEnvV10Frozen(data,
          S, 'dev'); its summary.json 'checks' show G1 dev reproduces the lock-v10 pilot rows 150/150 per cell).
  EVAL -- COPIED, not recomputed, from exp/results/full/v10_posthoc_G/summary.json (G1-eval-<log>-<S>, 200 streams,
          seeds 38400-38599), the file the supplement's Table G1 (writing/supplement/r6_tables/blockG_g1.tex, S19) is
          generated from: geometric-mean N80/tau of TU-FDC-DP(b) and RECT-BF-DP-TU, the number of streams that reached
          12 of 15 strictly before tau_R (n80_lt_tau), 'pinned', and the paired geometric-mean N_pen ratio FDC /
          RECT-BF-DP-TU with its two-sided 95% CI.

Row flags (all descriptive):
  rect_missed_80  RECT-BF-DP-TU reached 12 of 15 before tau_R on fewer than 80% of the 200 eval streams (Table G1
                  'Cens.' above 0.2), i.e. the rectangle failed the lock-v10 80% pre-horizon criterion at this eps
  rect_pinned     Table G1 'Pinned' (>= 95% of RECT-BF-DP-TU streams reach 12 of 15 at the same checkpoint)
  fdc_all_before  TU-FDC-DP(b) reached 12 of 15 before tau_R on all 200 eval streams
  selected        fdc_all_before and (rect_missed_80 or rect_pinned)   -- the rows the r9 critic asked about
  demanding       mean share < 0.5 and all-control eps-optimal at no budget (n_allctrl == 0) on the DEV half
                  (the md also reports a relaxed reading: mean share < 0.5 and all-control eps-optimal at <= 2 budgets)

Reproduction check: at each cell's lock-v10 eps the DEV numbers must equal writing/r9_difficulty.json exactly.

Usage (cwd anywhere):  .venv/bin/python3 iter_001/writing/scripts/difficulty_blockG_r9c.py [--workers 4] [--tex-only]
Outputs: writing/r9c_blockG_difficulty.json, writing/r9c_blockG_difficulty.md,
         writing/supplement/r9_tables/blockG_difficulty.tex  (tabular only, no float)
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import difficulty_r9 as D  # noqa: E402

IT = D.IT
GSUM = "exp/results/full/v10_posthoc_G/summary.json"
GTEX = "writing/supplement/r6_tables/blockG_g1.tex"
# block-G1 grid, as registered in exp/code/run_v10_posthoc_G.py (EPS_GRID); S in {16, 32, 64}
EPS_GRID = {"x5": (0.02, 0.03, 0.04, 0.05, 0.06, 0.08), "lenta": (0.003, 0.004, 0.006, 0.008, 0.010, 0.012)}
SS = (16, 32, 64)
NAME = {"x5": "X5", "lenta": "Lenta"}
R9_ID = {("x5", 16): "X5-16", ("x5", 32): "X5-32", ("x5", 64): "X5-64",
         ("lenta", 16): "LE-16", ("lenta", 32): "LE-32", ("lenta", 64): "LE-64"}
FDC, RECT = "TU-FDC-DP(b)", "RECT-BF-DP-TU"


def dev_job(args):
    data, S, eps = args
    t0 = time.time()
    d = D.dev_seg(dict(kind="v10", data=data, S=S, eps=eps))
    shares = [p["share"] for p in d["per_problem"]]
    out = dict(data=data, S=S, eps=eps, n_problems=len(shares), n_allctrl=d["n_allctrl"],
               eps_rel_base=eps / d["dev_base_rate"], eps_rel_headroom=eps / (d["Jstar"][-1] - d["J0"]),
               dev_base_rate=d["dev_base_rate"], headroom=d["Jstar"][-1] - d["J0"], J0=d["J0"],
               exact=d["exact"], method=d["method"], loader=d["loader"], notes=d["notes"],
               pooled_n_feasible=d["pooled_n_feasible"],
               pooled_n_eps_opt_lo=d["pooled_n_eps_opt_lo"], pooled_n_eps_opt_hi=d["pooled_n_eps_opt_hi"],
               per_problem_share=[None if s is None else float(s) for s in shares],
               per_problem_share_lo=[p["share_lo"] for p in d["per_problem"]],
               per_problem_share_hi=[p["share_hi"] for p in d["per_problem"]],
               allctrl_problems=[q for q in range(len(shares)) if d["J0"] >= d["Jstar"][q] - eps - D.TOL],
               budgets=d["budgets"], max_treated=d["max_treated"])
    import numpy as np
    if any(s is None for s in shares):
        out["mean_share"] = None
        out["mean_share_lo"] = float(np.mean(out["per_problem_share_lo"]))
        out["mean_share_hi"] = float(np.mean(out["per_problem_share_hi"]))
        out["pooled_share"] = None
        out["pooled_share_lo"] = d["pooled_n_eps_opt_lo"] / d["pooled_n_feasible"]
        out["pooled_share_hi"] = d["pooled_n_eps_opt_hi"] / d["pooled_n_feasible"]
    else:
        out["mean_share"] = float(np.mean(shares))
        out["pooled_share"] = d["pooled_n_eps_opt"] / d["pooled_n_feasible"]
    out["min_problem_share"] = min(out["per_problem_share_lo"])
    out["seconds"] = round(time.time() - t0, 1)
    return out


def eval_copy(G, data, S, eps):
    """Copy the published block-G1 eval numbers of one (log, S, eps) from summary.json (no recomputation)."""
    g = G["G1"][f"G1-eval-{data}-{S}"]
    key = [k for k in g["by_eps"] if abs(float(k) - eps) < 1e-12]
    assert len(key) == 1, (data, S, eps)
    b = g["by_eps"][key[0]]
    m, r = b["methods"], b["ratios"]["FDC/RECT-BF-DP-TU"]
    n = g["n"]
    return dict(source=f"{GSUM} G1['G1-eval-{data}-{S}']['by_eps']['{key[0]}'] (rendered in {GTEX})",
                n_streams=n, lock_eps=bool(b["lock_eps"]),
                fdc_N80_over_tau=m[FDC]["N80_over_tau"], rect_N80_over_tau=m[RECT]["N80_over_tau"],
                fdc_n80_lt_tau=m[FDC]["n80_lt_tau"], rect_n80_lt_tau=m[RECT]["n80_lt_tau"],
                rect_pinned=bool(m[RECT]["pinned"]), fdc_false_streams=m[FDC]["false_streams"],
                rect_false_streams=m[RECT]["false_streams"],
                ratio_fdc_rect=r["geomean_ratio"], ratio_ci95=r["ci95_two_sided"], tau_R=m[FDC]["tau_R"],
                fdc_exh_ctrl=m[FDC]["exhaustion_at_k80"]["ctrl"], fdc_exh_all=m[FDC]["exhaustion_at_k80"]["all"],
                fdc_sd_log_N80=m[FDC]["sd_log_N80"], rect_exh_ctrl=m[RECT]["exhaustion_at_k80"]["ctrl"])


def main(workers):
    t0 = time.time()
    G = json.loads((IT / GSUM).read_text())
    r9 = {b["id"]: b for b in json.loads((IT / "writing/r9_difficulty.json").read_text())["blocks"]}
    jobs = [(d, S, e) for d in EPS_GRID for S in SS for e in EPS_GRID[d]]
    # S = 64 cells first (slowest: branch-and-bound up to its node limit)
    order = sorted(jobs, key=lambda j: -j[1])
    with get_context("fork").Pool(workers) as pool:
        devs = {(o["data"], o["S"], o["eps"]): o for o in pool.imap_unordered(dev_job, order)}
    rows, checks = [], []
    for j in jobs:
        dv, ev = devs[j], eval_copy(G, *j)
        fl = dict(rect_missed_80=ev["rect_n80_lt_tau"] < 0.8 * ev["n_streams"], rect_pinned=ev["rect_pinned"],
                  fdc_all_before=ev["fdc_n80_lt_tau"] == ev["n_streams"])
        fl["selected"] = fl["fdc_all_before"] and (fl["rect_missed_80"] or fl["rect_pinned"])
        ms = dv["mean_share"] if dv["mean_share"] is not None else dv["mean_share_hi"]
        fl["demanding"] = bool(ms < 0.5 and dv["n_allctrl"] == 0)
        rows.append(dict(log=NAME[j[0]], S=j[1], eps=j[2], dev=dv, eval=ev, flags=fl))
        # reproduction of the r9 census at the lock-v10 eps of this cell
        b = r9[R9_ID[(j[0], j[1])]]
        if abs(b["eps"] - j[2]) < 1e-15:
            bd = b["dev"]
            for k in ("mean_share", "pooled_share", "mean_share_lo", "mean_share_hi", "pooled_share_lo",
                      "pooled_share_hi", "n_allctrl", "eps_rel_base", "eps_rel_headroom", "pooled_n_feasible"):
                if k in bd or dv.get(k) is not None:
                    checks.append(dict(cell=R9_ID[(j[0], j[1])], eps=j[2], quantity=k, r9=bd.get(k),
                                       r9c=dv.get(k), match=bd.get(k) == dv.get(k)))
            if not ev["lock_eps"]:
                checks.append(dict(cell=R9_ID[(j[0], j[1])], eps=j[2], quantity="G1 lock_eps flag", r9=True,
                                   r9c=False, match=False))
    res = {"what": "r9c: development-half decision difficulty at the block-G1 tolerances (critic r9 P1-B / R9c-B)",
           "label": "POST HOC, DESCRIPTIVE, DEVELOPMENT HALF (difficulty); evaluation numbers copied from block G1",
           "definitions": {
               "dev": "computed by writing/scripts/difficulty_r9.py dev_seg on the DEV half (lock-v10 frozen design); "
                      "see that script's definitions (a) pooled share, (b) mean share, (c) all-control",
               "eval": f"copied from {GSUM} (Table G1, {GTEX}); not recomputed",
               "rect_missed_80": "RECT-BF-DP-TU n80_lt_tau < 0.8 * 200 (Table G1 Cens. > 0.2)",
               "rect_pinned": "Table G1 Pinned",
               "fdc_all_before": "TU-FDC-DP(b) n80_lt_tau == 200",
               "selected": "fdc_all_before and (rect_missed_80 or rect_pinned)",
               "demanding": "DEV mean share < 0.5 (upper bracket if not exact) and n_allctrl == 0"},
           "rows": rows, "reproduction_checks": checks,
           "all_reproduced": bool(checks) and all(c["match"] for c in checks),
           "seconds_total": round(time.time() - t0, 1)}
    (IT / "writing/r9c_blockG_difficulty.json").write_text(json.dumps(res, indent=1))
    render(res)
    print("all_reproduced", res["all_reproduced"], "checks", len(checks))


# ------------------------------------------------------------------------------------------------- rendering
def sh(dv, key, nd=3):
    return D.share_str(dv, key, nd)


def render(res):
    rows = res["rows"]
    # ---- markdown (every grid row)
    L = ["# r9c: decision difficulty at the block-G1 tolerances", "",
         "**Post hoc, descriptive, development half.** Generated by `writing/scripts/difficulty_blockG_r9c.py`; do not "
         "edit by hand. Difficulty columns (Pooled, Mean, Min, Ctrl, eps/base, eps/head) are computed on the "
         "DEVELOPMENT half of each lock-v10 cell with the functions of `writing/scripts/difficulty_r9.py` (same "
         "loader, same exact counting). Evaluation columns (FDC and RECT N80/tau, streams before tau_R, Pinned, "
         f"ratio) are COPIED from `{GSUM}` (block G1, Table G1 in `{GTEX}`, 200 eval streams, seeds 38400-38599); "
         "nothing on the evaluation half is recomputed.", "",
         "Columns: Pooled / Mean / Min = pooled, mean and minimum over the 15 problems of the share of feasible "
         "policies that are eps-optimal (dev truth); `*` = rigorous bracket that rounds to one value; Ctrl = problems "
         "where all-control is eps-optimal; eps/head = eps / (J*_15 - J(all-control)); Before tau_R = streams of 200 "
         "reaching 12 of 15 strictly before tau_R (FDC/RECT); Sel. = FDC 200/200 before tau_R and the rectangle "
         "missed the 80% pre-horizon criterion (M) or was pinned (P); Dem. = mean share < 0.5 and Ctrl = 0.", "",
         "| Log | S | eps | eps/base | eps/head | Pooled | Mean | Min | Ctrl | FDC N80/tau | RECT N80/tau | "
         "Before tau_R F/R | Pinned | FDC/RECT [95% CI] | Sel. | Dem. |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dv, ev, fl = r["dev"], r["eval"], r["flags"]
        sel = ("M" if fl["rect_missed_80"] else "") + ("P" if fl["rect_pinned"] else "") if fl["selected"] else ""
        L.append(f"| {r['log']} | {r['S']} | {r['eps']:g}{' (lock)' if ev['lock_eps'] else ''} | "
                 f"{100 * dv['eps_rel_base']:.1f}% | {dv['eps_rel_headroom']:.2f} | {sh(dv, 'pooled_share')} | "
                 f"{sh(dv, 'mean_share')} | {dv['min_problem_share']:.3f} | {dv['n_allctrl']} | "
                 f"{ev['fdc_N80_over_tau']:.3f} | {ev['rect_N80_over_tau']:.3f} | "
                 f"{ev['fdc_n80_lt_tau']}/{ev['rect_n80_lt_tau']} | {'yes' if ev['rect_pinned'] else 'no'} | "
                 f"{ev['ratio_fdc_rect']:.3f} [{ev['ratio_ci95'][0]:.3f}, {ev['ratio_ci95'][1]:.3f}] | {sel} | "
                 f"{'yes' if fl['demanding'] else 'no'} |")
    sel = [r for r in rows if r["flags"]["selected"]]
    miss = [r for r in rows if r["flags"]["selected"] and r["flags"]["rect_missed_80"]]
    dem = [r for r in rows if r["flags"]["demanding"] and r["flags"]["fdc_all_before"]]
    L += ["", "## Reading (post hoc, descriptive)", ""]
    L.append(f"- Rows where FDC-DP reached 12 of 15 before tau_R on all 200 streams while the rectangle missed the 80% "
             f"criterion or was pinned: {len(sel)} of {len(rows)}; of these, the rectangle missed the 80% criterion in "
             f"{len(miss)}.")
    for r in miss:
        dv, ev = r["dev"], r["eval"]
        L.append(f"  - {r['log']} S={r['S']} eps={r['eps']:g}: mean share {sh(dv, 'mean_share')}, pooled "
                 f"{sh(dv, 'pooled_share')}, min {dv['min_problem_share']:.3f}, Ctrl {dv['n_allctrl']}/15, eps/head "
                 f"{dv['eps_rel_headroom']:.2f}; FDC {ev['fdc_N80_over_tau']:.3f} tau, RECT before tau_R "
                 f"{ev['rect_n80_lt_tau']}/200, ratio {ev['ratio_fdc_rect']:.3f}.")
    L.append(f"- Demanding rows (mean share < 0.5 and all-control eps-optimal at no budget) at which FDC-DP still "
             f"reached 12 of 15 before tau_R on all streams: {len(dem)}"
             + (": " + "; ".join(f"{r['log']} S={r['S']} eps={r['eps']:g}" for r in dem) if dem else "") + ".")
    near = [r for r in rows if r["flags"]["fdc_all_before"] and r["dev"]["n_allctrl"] <= 2
            and (r["dev"]["mean_share"] if r["dev"]["mean_share"] is not None else r["dev"]["mean_share_hi"]) < 0.5]
    L.append(f"- Relaxed reading (mean share < 0.5 and all-control eps-optimal at no more than 2 of 15 budgets) with "
             f"FDC-DP 200/200 before tau_R: {len(near)}"
             + (": " + "; ".join(
                 f"{r['log']} S={r['S']} eps={r['eps']:g} (mean {sh(r['dev'], 'mean_share')}, pooled "
                 f"{sh(r['dev'], 'pooled_share')}, all-control eps-optimal at problem(s) "
                 f"{', '.join('q%d (at most %d treated)' % (q + 1, r['dev']['max_treated'][q]) for q in r['dev']['allctrl_problems'])}; "
                 f"RECT {r['eval']['rect_n80_lt_tau']}/200 before tau_R, FDC {r['eval']['fdc_N80_over_tau']:.3f} tau)"
                 for r in near) if near else "") + ". In these rows the rectangle's N_pen is censored at tau_R, so "
             "the FDC/RECT ratio equals FDC's N80/tau and is horizon-sensitive.")
    lo = min(rows, key=lambda r: (r["dev"]["mean_share"] if r["dev"]["mean_share"] is not None
                                  else r["dev"]["mean_share_lo"]))
    L.append(f"- Lowest mean share on the grid: {lo['log']} S={lo['S']} eps={lo['eps']:g}, "
             f"{sh(lo['dev'], 'mean_share')} (Ctrl {lo['dev']['n_allctrl']}/15, eps/head "
             f"{lo['dev']['eps_rel_headroom']:.2f}).")
    L += ["", f"Reproduction of writing/r9_difficulty.json at the lock-v10 eps of each cell: "
          f"{sum(c['match'] for c in res['reproduction_checks'])} of {len(res['reproduction_checks'])} quantities "
          f"identical (all_reproduced = {res['all_reproduced']}).", ""]
    (IT / "writing/r9c_blockG_difficulty.md").write_text("\n".join(L))
    # ---- LaTeX tabular (no float): the selected rows
    T = [r"% generated by writing/scripts/difficulty_blockG_r9c.py -- do not edit by hand",
         r"% POST HOC, DESCRIPTIVE. Dev = development half (difficulty_r9.py functions); eval columns copied from",
         r"% exp/results/full/v10_posthoc_G/summary.json (Table G1). Rows: FDC-DP 200/200 before tau_R and RECT missed",
         r"% the 80% pre-horizon criterion (M) or was pinned (P).",
         r"\small", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lrrrrrrrrrl@{}}", r"\toprule",
         r" & & & \multicolumn{3}{c}{\emph{Development half}} & \multicolumn{5}{c}{\emph{Block G1 eval (copied)}} \\",
         r"\cmidrule(lr){4-6}\cmidrule(l){7-11}",
         r"Log & $S$ & $\eps$ & $\eps$/head & Mean & Ctrl & FDC $N_{80}/\tau$ & FDC ctrl empty & RECT before $\tau_R$ & "
         r"FDC/RECT & \\", r"\midrule"]
    prev = None
    for r in sel:
        dv, ev, fl = r["dev"], r["eval"], r["flags"]
        if prev is not None and (r["log"], r["S"]) != prev:
            T.append(r"\addlinespace")
        prev = (r["log"], r["S"])
        tag = ("M" if fl["rect_missed_80"] else "") + ("P" if fl["rect_pinned"] else "")
        T.append(f"{r['log']} & {r['S']} & {r['eps']:g}{r'$^\star$' if ev['lock_eps'] else ''} & "
                 f"{dv['eps_rel_headroom']:.2f} & {sh(dv, 'mean_share')} & {dv['n_allctrl']} & "
                 f"{ev['fdc_N80_over_tau']:.3f} & {ev['fdc_exh_ctrl']:.2f} & {ev['rect_n80_lt_tau']}/{ev['n_streams']} & "
                 f"{ev['ratio_fdc_rect']:.3f} & {tag} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    p = IT / "writing/supplement/r9_tables/blockG_difficulty.tex"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(T) + "\n")


if __name__ == "__main__":
    if "--recopy-eval" in sys.argv:   # re-copy the published eval columns into the record (no dev recomputation)
        res = json.loads((IT / "writing/r9c_blockG_difficulty.json").read_text())
        G = json.loads((IT / GSUM).read_text())
        for r in res["rows"]:
            r["eval"] = eval_copy(G, r["dev"]["data"], r["S"], r["eps"])
        (IT / "writing/r9c_blockG_difficulty.json").write_text(json.dumps(res, indent=1))
        render(res)
    elif "--tex-only" in sys.argv:
        render(json.loads((IT / "writing/r9c_blockG_difficulty.json").read_text()))
    else:
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
        main(w)
