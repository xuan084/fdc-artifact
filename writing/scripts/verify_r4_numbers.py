"""Recompute every number added or changed in paper revision r4 (part 1) from result files (read-only).

Extends verify_r3_numbers.py (which is run first, unchanged). Each check prints the computed value, the value the
paper prints, and OK / MISMATCH; the script exits with status 1 if any check fails.

Sources (all under iter_001/):
  exp/results/full/v8_analysis_A.json             observed RECT-ck-BF stopping fractions (R3-M1)
  exp/results/pilots/width_prop/prediction.json   common-horizon model, sandwich bounds, CR9 +box ratio (R3-M1/M2)
  exp/results/pilots/v9_candidates/analysis_dev.json   TU-FDC / FDC-LOC dev block (time-uniform remark, ablation)
  exp/results/pilots/fdc_hg/analysis_dev.json     FDC-HG dev block (ablation)
  exp/results/full/v8_posthoc_E/summary.json      sensitivity block
  exp/results/full/v8_posthoc_F/summary.json      adaptive-regime block (semi-synthetic)
  exp/results/pilots/hillstrom_v9/summary_b.json  Hillstrom design B dev block
  exp/results/full/v8_posthoc_D/summary.json      selected per-segment-menu member (ledger log(L/K))
  exp/code/dsswm/tests/test_fdc_bet.py            Monte Carlo test level (beta = log 8)
  plan/prereg_lock_v8_addendum.json, exp/results/full/v7_analysis_A.json   erratum (0.761 vs 0.670)
  plan/prereg_lock_v8_addendum_DRAFT.md           development basis of the 1.05 margin (R3-M3)
  writing/sourced_prices.md                       unit prices (Table 2)
The near-tie census (FDC-LOC paragraph) is recomputed from the development halves with the frozen v8 environment code.
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r4_numbers.py [--skip-census]
"""
import json
import math
import re
import runpy
import sys
from pathlib import Path

import numpy as np

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
FAIL = []


def check(label, computed, printed, tol):
    ok = abs(computed - printed) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: computed {computed:.6g}, paper {printed}")
    if not ok:
        FAIL.append(label)


def J(p):
    return json.load(open(p))


def section(t):
    print(f"\n== {t}")


def r3_numbers():
    section("r3 numbers (verify_r3_numbers.py, unchanged)")
    runpy.run_path(str(Path(__file__).with_name("verify_r3_numbers.py")), run_name="__main__")


def r3_m1_and_m2():
    section("R3-M1: observed vs model stopping fraction of the matched rectangle (X5 test); R3-M2: CR9 +box")
    a = J(RES / "full/v8_analysis_A.json")
    pr = J(RES / "pilots/width_prop/prediction.json")
    printed = {0.015: (0.817, 0.81, 0.439, 0.426), 0.02: (0.813, 0.73, 0.414, 0.299), 0.03: (0.660, 0.59, 0.266, 0.202)}
    obs_int = {0.02: (0.333, 0.650), 0.03: (0.216, 0.506), 0.015: (0.369, 0.656)}
    for eps, (x_obs_p, x_mod_p, pred_p, obs_ratio_p) in printed.items():
        x_obs = a["describe"][f"RECT-ck-BF@{eps}"]["geomean_N80_pen_over_tau"]
        lg = pr["logs"][f"X9@{eps}"]["corollary2"]
        x_mod = lg["common_horizon_model"]["x_R"]
        # r from the model: invert N80_ratio = r / (1 - x (1 - r)) at the model's own x
        q, x = lg["common_horizon_model"]["N80_ratio"], x_mod
        r = q * (1 - x) / (1 - q * x)
        f = lambda rr, xx: rr / (1 - xx * (1 - rr))  # noqa: E731
        lo, hi = lg["sandwich_all_q"]
        check(f"eps {eps}: observed x (RECT-ck-BF N80/tau)", x_obs, x_obs_p, 6e-4)
        check(f"eps {eps}: model x", x_mod, x_mod_p, 6e-3)
        check(f"eps {eps}: prediction at observed x", f(r, x_obs), pred_p, 1.5e-3)
        lo_o, hi_o = f(lo, x_obs), f(hi, x_obs)
        check(f"eps {eps}: interval at observed x, lower", lo_o, obs_int[eps][0], 1.5e-3)
        check(f"eps {eps}: interval at observed x, upper", hi_o, obs_int[eps][1], 1.5e-3)
        o = pr["observed"][f"X9@{eps}"]["RECT-ck-BF"]
        check(f"eps {eps}: observed FDC-BF/RECT-ck-BF", o["ratio"] if "ratio" in o else o["geomean_ratio"], obs_ratio_p, 6e-4)
        print(f"     observed ratio inside interval at observed x: {lo_o <= obs_ratio_p <= hi_o}")
    o = pr["observed"]["CR9@0.001"]["RECT-ck-BF+box"]
    check("CR9 FDC-BF/RECT-ck-BF+box", o.get("ratio", o.get("geomean_ratio", float("nan"))), 0.532, 6e-4)
    check("CR9 +box CI lower", o["ci95"][0], 0.517, 6e-4)
    check("CR9 +box CI upper", o["ci95"][1], 0.548, 6e-4)
    xs = [a["describe"][f"RECT-ck-BF@{e}"]["geomean_N80_pen_over_tau"] for e in (0.015, 0.02, 0.03)]
    print(f"     observed stopping range {min(xs):.3f}-{max(xs):.3f} (paper: 66-82%)")


def tu_and_loc():
    section("Time-uniform remark (TU-FDC) and FDC-LOC ablation, development halves, seeds 950-999")
    d = J(RES / "pilots/v9_candidates/analysis_dev.json")
    cr = d["cr9"][0]["ratios"]
    x5 = {b["eps"]: b["ratios"] for b in d["x5"]}
    crd = d["cr9d"][0]["ratios"]
    x5d = d["x5d"][0]["ratios"]
    check("CR9 FDC-BF/HC-WoR* (K20 grid)", cr["FDC-BF/HC-WoR"]["geomean_ratio"], 0.663, 6e-4)
    check("CR9 FDC-BF/HC-WoR* UB95", cr["FDC-BF/HC-WoR"]["ub95_one_sided"], 0.695, 6e-4)
    check("X5 0.02 FDC-BF/HC-WoR*", x5[0.02]["FDC-BF/HC-WoR"]["geomean_ratio"], 0.329, 6e-4)
    check("X5 0.02 FDC-BF/HC-WoR* UB95", x5[0.02]["FDC-BF/HC-WoR"]["ub95_one_sided"], 0.342, 6e-4)
    k = [k for k in crd if k.startswith("TU-FDC[K20]/HC-WoR")][0]
    check("CR9 dense TU-FDC[K20]/HC-WoR", crd[k]["geomean_ratio"], 0.708, 6e-4)
    check("CR9 dense UB95", crd[k]["ub95_one_sided"], 0.737, 6e-4)
    k = [k for k in x5d if k.startswith("TU-FDC[K20]/HC-WoR")][0]
    check("X5 dense TU-FDC[K20]/HC-WoR", x5d[k]["geomean_ratio"], 0.356, 6e-4)
    check("X5 dense UB95", x5d[k]["ub95_one_sided"], 0.365, 6e-4)
    td = d["toy"]["toyd"]
    check("dense toy TU-FDC false streams", td["TU-FDC[K20]"]["false_streams"], 0, 0)
    check("dense toy invalid NAIVE-joint false streams", td["NAIVE-joint"]["false_streams"], 6, 0)
    print("     dense-grid keys:", sorted(crd)[:12])
    check("FDC-LOC/FDC-BF CR9", cr["FDC-LOC/FDC-BF"]["geomean_ratio"], 0.925, 6e-4)
    check("FDC-LOC/FDC-BF CR9 UB95", cr["FDC-LOC/FDC-BF"]["ub95_one_sided"], 0.949, 6e-4)
    check("FDC-LOC/FDC-BF X5 0.02", x5[0.02]["FDC-LOC/FDC-BF"]["geomean_ratio"], 1.000, 6e-4)
    check("ORACLE-LOC/FDC-BF CR9", cr["ORACLE-LOC/FDC-BF"]["geomean_ratio"], 0.812, 6e-4)
    check("ORACLE-LOC/FDC-BF X5 0.02", x5[0.02]["ORACLE-LOC/FDC-BF"]["geomean_ratio"], 0.807, 6e-4)
    m = d["cr9"][0]["methods"]["FDC-LOC"]
    check("CR9 beta_hat/beta_J at stop", m["beta_ratio_at_stop_mean"], 0.944, 6e-4)


def census(skip):
    section("Near-tie census (true gaps on the development halves)")
    if skip:
        print("  skipped (--skip-census)")
        return
    sys.path.insert(0, str(IT / "exp/code"))
    import run_r5s_v8 as R8  # noqa: E402
    printed = {"CR9": (0.001, 3386, 437, 1531), "X9": (0.02, 3347, 3076, 3347)}
    for layer, (eps, Mp, goodp, w3p) in printed.items():
        R8.init_env(layer, "dev", (eps,))
        E = R8._ENV
        ctx, Jv, Js = E["ctxs"][eps], np.asarray(E["J"]), np.asarray(E["Js"])
        M = good = w3 = 0
        for q in range(ctx.Q):
            f = np.asarray(ctx.feas[q], bool)
            gap = Js[q] - Jv[f]
            M += int(f.sum())
            good += int((gap <= eps + 1e-12).sum())
            w3 += int((gap <= 3 * eps + 1e-12).sum())
        check(f"{layer} dev: sum_q |Pi_q|", M, Mp, 0)
        check(f"{layer} dev: eps-optimal (q, pi) pairs", good, goodp, 0)
        check(f"{layer} dev: pairs within 3 eps", w3, w3p, 0)


def fdc_hg():
    section("FDC-HG ablation (development, seeds 950-999)")
    d = J(RES / "pilots/fdc_hg/analysis_dev.json")
    r = d["cr9"][0]["ratios"]["FDC-HG/FDC-BF"]
    check("CR9 FDC-HG/FDC-BF", r["geomean_ratio"], 0.969, 6e-4)
    check("CR9 FDC-HG/FDC-BF UB95", r["ub95_one_sided"], 0.990, 6e-4)
    xs = [b["ratios"]["FDC-HG/FDC-BF"]["geomean_ratio"] for b in d["x5"]]
    check("X5 FDC-HG/FDC-BF min over eps", min(xs), 0.944, 6e-4)
    check("X5 FDC-HG/FDC-BF max over eps", max(xs), 0.976, 6e-4)
    check("max certificate seconds per checkpoint", d["go"]["c3_detail"]["max_cert_sec_per_checkpoint"], 7.1, 0.05)


def sensitivity():
    section("Sensitivity block (post hoc, development halves)")
    E = J(RES / "full/v8_posthoc_E/summary.json")
    rb = E["robustness"]
    check("CR9 worst PJC-local CI upper", rb["CR9"]["pjc_max_ci_upper"]["ci_upper"], 1.037, 6e-4)
    check("X5 worst PJC CI upper", rb["X9"]["pjc_max_ci_upper"]["ci_upper"], 1.029, 6e-4)
    check("CR9 worst rectangle CI upper", rb["CR9"]["rect_worst"]["ci_upper"], 0.723, 6e-4)
    check("X5 worst rectangle CI upper", rb["X9"]["rect_worst"]["ci_upper"], 0.369, 6e-4)
    for lay, lo_p, hi_p in [("X9", 0.27, 0.36), ("CR9", 0.50, 0.69)]:
        vals, pj = [], []
        for st in E["layers"][lay]["settings"].values():
            for rv, row in st["rivals"].items():
                if rv in ("RECT-ck-HG", "HC-WoR", "RECT-ck-BF"):
                    vals.append(row["FDCBF_over_rival"])
                if rv == "PJC-local":
                    pj.append(row["FDCBF_over_rival"])
        check(f"{lay} rectangle ratios min", min(vals), lo_p, 6e-3)
        check(f"{lay} rectangle ratios max", max(vals), hi_p, 6e-3)
        print(f"     {lay} PJC-local ratio range {min(pj):.3f}-{max(pj):.3f}")
    nfalse = sum(rb[l]["false_streams_total"] for l in rb)
    check("false streams over the block", nfalse, 0, 0)


def regime():
    section("Adaptive regime (semi-synthetic, post hoc)")
    F = J(RES / "full/v8_posthoc_F/summary.json")
    m = F["envs"]["CR9-asym"]["methods"]
    picks = {k: v for k, v in m.items() if k.startswith("PJC-A") or k == "PJC-menu*" or k.startswith("PJC-menu")}
    sp = {k: v["FDCBF_over_method_N80"] for k, v in picks.items() if "FDCBF_over_method_N80" in v}
    print("     CR9-asym FDC-BF/adaptive:", {k: round(v, 3) for k, v in sp.items()})
    faster = [v for k, v in sp.items() if "segmenu" not in k]
    check("Criteo-asym: smallest adaptive gain over frozen 50/50 (FDC-BF/method)", min(faster), 1.087, 6e-4)
    check("Criteo-asym: largest adaptive gain", max(faster), 1.127, 6e-4)
    check("Criteo-asym: frozen pilot-Neyman gain", m["FDC-BF[ney-pilot]"]["FDCBF_over_method_N80"], 1.180, 6e-4)
    nb = [F["envs"][e]["neypilot_vs_best_adaptive_pick"]["neypilot_over_adaptive_N80"] for e in F["envs"]]
    check("pilot-Neyman / best adaptive, min over regimes", min(nb), 0.938, 6e-4)
    check("pilot-Neyman / best adaptive, max over regimes", max(nb), 0.973, 6e-4)
    fs = sum(v.get("false_streams", 0) for e in F["envs"].values() for v in e["methods"].values())
    check("false streams over all regimes", fs, 0, 0)


def hillstrom():
    section("Hillstrom design B (development half, seeds 950-999, eps 0.0125)")
    h = J(RES / "pilots/hillstrom_v9/summary_b.json")
    e = h["by_eps"]["0.0125"]
    r = e["ratios_FDC_BF_over"]
    for k, p, ubp in [("RECT-ck-HG[0.2]", 0.842, 0.858), ("HC-WoR{nstar,0.75,1}[0.2]", 0.833, 0.848),
                      ("PJC-BF[local,b=4,neyman]", 0.993, 1.011)]:
        check(f"FDC-BF/{k}", r[k]["ratio"], p, 6e-4)
        check(f"FDC-BF/{k} UB95", r[k]["ub95_one_sided"], ubp, 6e-4)
    u = e["u12_ratio_FDC_BF_over"]
    check("width ratio vs RECT-ck-HG* at k = 18", u["RECT-ck-HG[0.2]"]["ratio"], 0.664, 6e-4)
    check("width ratio vs RECT-ck-BF at k = 18", u["RECT-ck-BF"]["ratio"], 0.326, 6e-4)
    check("FDC-BF N80/tau", e["methods"]["FDC-BF"]["geomean_N80_over_tau"], 0.709, 6e-4)
    check("FDC-BF false streams", e["methods"]["FDC-BF"]["false_streams"], 0, 0)


def reviewer_items():
    section("external reviewer r3 wrong statements")
    D = J(RES / "full/v8_posthoc_D/summary.json")
    pick = D["layers"]["X9"]["picks"]["segmenu"]["pick"]
    b = int(re.search(r"b=(\d+)", pick).group(1))
    M = len(re.search(r"menu=([0-9./]+)", pick).group(1).split("/"))
    S, K = 9, 20
    L = sum((M ** S) ** sum(1 for x in [b] if x < k) for k in range(K))
    check(f"log(L/K) for {pick}", math.log(L / K), 9.290, 6e-4)
    check("S ln M", S * math.log(M), 9.888, 6e-4)
    check("ln(37/20)", math.log(37 / 20), 0.615, 6e-4)
    src = open(IT / "exp/code/dsswm/tests/test_fdc_bet.py").read()
    assert "beta = math.log(8.0)" in src
    check("Monte Carlo test level e^-beta, beta = log 8", math.exp(-math.log(8.0)), 0.125, 1e-12)
    beta_j = math.log(3386 * 20 / 0.045)
    check("operational e^-beta_J", math.exp(-beta_j), 6.65e-7, 1e-9)
    t = open(IT / "plan/prereg_lock_v8_addendum_DRAFT.md").read()
    assert "1.042" in t and "0.97–0.98" in t
    print("  [OK] margin basis: worst dev UB 1.042 at eps 0.01 on 20 seeds; 0.97-0.98 at eps 0.02 (DRAFT text)")


def errata():
    section("Errata index")
    v8 = open(IT / "plan/prereg_lock_v8_addendum.json").read()
    assert "UB* = 0.761" in v8
    v7 = J(RES / "full/v7_analysis_A.json")
    check("v7 governing UB95 (decision.UB_star in v7_analysis_A)", v7["decision"]["UB_star"], 0.670, 6e-4)
    print(f"     governing rival: {v7['decision']['governing_rival']}")
    v6 = open(RES / "full/v6_summary.md").read()
    assert "Erratum (2026-10-04)" in v6
    print("  [OK] v8 lock JSON records 0.761; v6_summary.md carries the dated erratum line")


def prices():
    section("Table 2 at sourced unit prices")
    sp = open(IT / "writing/sourced_prices.md").read()
    for s in ["$0.50 per 1,000 records", "$2.00 per CRPU-hour", "$0.26 * 1000", "ε=19.61", "ε = 4"]:
        assert s in sp, s
    saved = {"Criteo": 505304, "X5": 42708}
    check("Criteo matching at $0.50/1k", saved["Criteo"] * 0.0005, 253, 0.5)
    check("X5 matching at $0.50/1k", saved["X5"] * 0.0005, 21, 0.5)
    check("Criteo labels at $0.26", saved["Criteo"] * 0.26 / 1000, 131.4, 0.05)
    check("X5 labels at $0.26", saved["X5"] * 0.26 / 1000, 11.1, 0.05)
    check("Clean Rooms minimum query: 32 CRPU x $2/h x 60 s", 32 * 2 / 60, 1.07, 0.005)


def tu_single_cell_mc():
    section("Remark (time-uniform): single-cell Monte Carlo of the TU lemma (test_v9_candidates.py, seed 12345)")
    import inspect
    sys.path.insert(0, str(IT / "exp/code"))
    from dsswm.tests import test_v9_candidates as T
    src = inspect.getsource(T.test_lemma_tu_single_cell_monte_carlo_with_invalid_control)
    src = src.replace("    assert f_tu <= a + 3 * se, (f_tu, a)\n    assert f_nv > a + 3 * se, (f_nv, a)\n",
                      "    return f_tu, f_nv\n")
    ns = dict(vars(T))
    exec(src, ns)
    f_tu, f_nv = ns["test_lemma_tu_single_cell_monte_carlo_with_invalid_control"]()
    check("TU crossing frequency (level 0.10)", f_tu, 0.036, 6e-4)
    check("invalid refreshed-quantile control", f_nv, 0.553, 6e-4)


def sensitivity_details():
    section("Sensitivity details")
    E = J(RES / "full/v8_posthoc_E/summary.json")
    L = E["layers"]
    hg = lambda lay, st: L[lay]["settings"][st]["rivals"]["RECT-ck-HG"]["FDCBF_over_rival"]  # noqa: E731
    check("CR9 FDC-BF/RECT-ck-HG* baseline", hg("CR9", "base"), 0.649, 6e-4)
    sk = [k for k in L["CR9"]["settings"] if k.startswith("dtot=0.01")][0]
    check("CR9 FDC-BF/RECT-ck-HG* at delta 0.01", hg("CR9", sk), 0.613, 6e-4)
    base = hg("X9", "base")
    shifts = [abs(hg("X9", k) - base) for k in L["X9"]["settings"] if k.startswith("dtot=")]
    check("X5 largest |shift| of FDC-BF/RECT-ck-HG* under delta changes (< 0.02)", max(shifts), 0.013, 2e-3)
    n = sum(len(v["methods_run"]) for v in E["settings"].values()) * len(L) * 50
    check("method-setting-stream runs (methods actually run x 2 layers x 50 streams)", n, 5500, 0)


def misc():
    section("Other new body numbers")
    d = J(RES / "pilots/fdc_hg/analysis_dev.json")
    rs = [d["cr9"][0]["ratios"]["FDC-HG/FDC-BF"]["geomean_ratio"]] + [b["ratios"]["FDC-HG/FDC-BF"]["geomean_ratio"] for b in d["x5"]]
    check("FDC-HG saving, smallest (intro '2-6%')", 1 - max(rs), 0.024, 6e-3)
    check("FDC-HG saving, largest", 1 - min(rs), 0.056, 6e-3)
    v8c = J(RES / "full/v8_analysis_C.json")
    sv = json.dumps(v8c)
    for k, p in [("RECT-ck-HG[Neyman]", 0.633), ("HC-WoR[Neyman]", 0.645)]:
        m = re.search(r'"%s[^"]*": \{[^{}]*?"geomean_ratio": ([0-9.]+)' % re.escape(k), sv)
        if m:
            check(f"Criteo FDC-BF/{k}", float(m.group(1)), p, 6e-4)
    print("     Neyman vs 50/50 change: 0.633/0.643 -> %.3f, 0.645/0.655 -> %.3f (paper: under 2%%)" % (1 - 0.633 / 0.643, 1 - 0.645 / 0.655))


def main():
    skip = "--skip-census" in sys.argv
    r3_numbers()
    r3_m1_and_m2()
    tu_and_loc()
    census(skip)
    fdc_hg()
    sensitivity()
    regime()
    hillstrom()
    reviewer_items()
    errata()
    prices()
    tu_single_cell_mc()
    sensitivity_details()
    misc()
    print(f"\n{len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
