"""Recompute every number added or changed in paper revision r4 part 2 (lock-v9 integration) from result files.

Runs verify_r4_numbers.py first (which runs verify_r3_numbers.py), unchanged, then checks the lock-v9 numbers printed in
main.tex and supplement.tex. Each check prints the computed value, the printed value and OK / MISMATCH; the script exits
with status 1 if any check in either script fails. Read-only.

Sources (all under iter_001/):
  exp/results/full/v9_analysis_{A,B,C}.json     lock-v9 analyses (authoritative)
  exp/results/full/v9_summary.md                lock-v9 summary (cross-checked against the JSON)
  exp/results/pilots/v9_analysis_{A,B}_dev.json lock-v9 development runner check (threshold sizing, NI headroom)
  exp/results/full/v7_analysis_A.json, v8_analysis_A.json   checkpoint-strength values used in comparisons and moved rows
  writing/latex_acm/main.tex, writing/supplement/supplement.tex   printed-text presence checks for every v9 number
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r4b_numbers.py [--skip-census]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
MAIN = (IT / "writing/latex_acm/main.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
SUPP_TABLES = "".join(p.read_text() for p in sorted((IT / "writing/supplement/r4_tables").glob("v9_*.tex")))
FAIL = []


def check(label, computed, printed, tol):
    ok = abs(computed - printed) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: computed {computed:.6g}, paper {printed}")
    if not ok:
        FAIL.append(label)


def present(label, needle, where):
    txt = {"main": MAIN, "supp": SUPP, "tables": SUPP_TABLES}[where]
    ok = needle in txt
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {needle!r} printed in {where}")
    if not ok:
        FAIL.append(f"{label} (text)")


def J(p):
    return json.load(open(p))


def section(t):
    print(f"\n== {t}")


A = J(RES / "full/v9_analysis_A.json")
B = J(RES / "full/v9_analysis_B.json")
C = J(RES / "full/v9_analysis_C.json")
H3 = 6e-4  # tolerance for 3-decimal printed values


def r3(x):
    return f"{x:.3f}"


def table1_panel_a():
    section("Table 1A (lock v9, 77 monitoring times): ratio, CI, UB95, F/T/S, x12, N100, rows")
    rows = [("A", "HC-WoR@D", r"\textbf{HC-WoR*} (betting), Criteo", 1e6, "M", 2),
            ("A", "RECT-ck-BF-TU@D", r"\textbf{RECT-ck-BF-TU} (matched), Criteo", 1e6, "M", 2),
            ("B", "HC-WoR@D", r"\textbf{HC-WoR*} (Neyman), X5", 1e3, "k", 1),
            ("B", "RECT-ck-BF-TU@D", r"\textbf{RECT-ck-BF-TU} (matched), X5", 1e3, "k", 1),
            ("B", "PJC-local@D", r"\textbf{PJC-local*} (77-time ledger), X5", 1e3, "k", 1),
            ("B", "TU-PJC@D", r"\textbf{TU-PJC}, X5", 1e3, "k", 1)]
    for blk, key, label, unit, suf, nd in rows:
        d = A if blk == "A" else B
        line = [l for l in MAIN.splitlines() if l.startswith(label + " &")]
        if len(line) != 1:
            FAIL.append(f"Table 1A row {label} not found")
            print(f"  [MISMATCH] row not found: {label}")
            continue
        cells = [c.strip() for c in line[0].rstrip("\\ ").split("&")]
        n80 = d["comparisons"]["TU-FDC@D|N80_pen"][key]
        x12 = d["comparisons"]["TU-FDC@D|x12"][key]
        n100 = d["comparisons"]["TU-FDC@D|N100_pen"][key]
        tau = d["describe"]["TU-FDC@D"]["tau_R"]
        rows_v = d["describe"][key]["geomean_N80_pen_over_tau"] * tau / unit
        num = lambda s: [float(x) for x in re.findall(r"\d+\.\d+", s)]  # noqa: E731
        check(f"{blk} {key} rows ({suf})", rows_v, num(cells[3])[0], 0.5 * 10 ** -nd + 1e-9)
        v = num(cells[4])
        check(f"{blk} {key} ratio", n80["geomean_ratio"], v[0], H3)
        check(f"{blk} {key} CI low", n80["ci95_two_sided"][0], v[1], H3)
        check(f"{blk} {key} CI high", n80["ci95_two_sided"][1], v[2], H3)
        check(f"{blk} {key} UB95", n80["ub95_one_sided"], num(cells[5])[0], H3)
        f = [float(x) for x in re.findall(r"[\d.]+", cells[7])]
        for lab, comp, pv in zip(("faster", "tied", "slower"), ("frac_faster", "frac_tied", "frac_slower"), f):
            check(f"{blk} {key} % {lab}", 100 * n80[comp], pv, 0.05)
        v = num(cells[8])
        check(f"{blk} {key} x12 ratio", x12["geomean_ratio"], v[0], H3)
        check(f"{blk} {key} x12 UB95", x12["ub95_one_sided"], v[1], H3)
        v = num(cells[9])
        check(f"{blk} {key} N100 ratio", n100["geomean_ratio"], v[0], H3)
        check(f"{blk} {key} N100 UB95", n100["ub95_one_sided"], v[1], H3)
    check("Criteo TU-FDC rows (M)", A["describe"]["TU-FDC@D"]["geomean_N80_pen_over_tau"] * A["describe"]["TU-FDC@D"]["tau_R"] / 1e6, 0.87, 0.005)
    check("X5 TU-FDC rows (k)", B["describe"]["TU-FDC@D"]["geomean_N80_pen_over_tau"] * B["describe"]["TU-FDC@D"]["tau_R"] / 1e3, 23.7, 0.05)
    check("monitoring times (K_eval), A", A["describe"]["TU-FDC@D"]["K_eval"], 77, 0)
    check("monitoring times (K_eval), B", B["describe"]["TU-FDC@D"]["K_eval"], 77, 0)
    present("Table 1A seeds", "Criteo seeds 37000--37199", "main")
    present("Table 1A seeds", "X5 seeds 37200--37399", "main")


def verdicts_and_validity():
    section("Verdicts, false streams, replicas (C1, C2)")
    for name, d, thr, ubp in [("A", A, 0.85, 0.723), ("B", B, 0.50, 0.379)]:
        dec = d["decision"]
        ok = dec["verdict"] == "positive_result_achieved" and dec["replica_status"] == "pass"
        print(f"  [{'OK' if ok else 'MISMATCH'}] block {name}: verdict {dec['verdict']}, replica {dec['replica_status']}")
        if not ok:
            FAIL.append(f"block {name} verdict")
        check(f"block {name} governing UB95", dec["UB_star"], ubp, H3)
        check(f"block {name} governing UB95 below threshold {thr}", float(dec["UB_star"] < thr), 1, 0)
        check(f"block {name} TU-FDC false streams", d["false_streams"]["TU-FDC@D"]["false_streams"], 0, 0)
        check(f"block {name} false streams, all rigorous methods", d["all_rigorous_false_streams_total"], 0, 0)
        check(f"block {name} streams", d["n_streams"], 200, 0)
        check(f"block {name} combined replica (incl. R2b, R2c) pass", float(d["replica_status_combined"]["status"] == "pass"
              and d["replica_status_combined"]["cross_grid_R2c"]["pass"] and d["replica_status_combined"]["cross_task_R2b"]["pass"]), 1, 0)
    check("block B NI governing UB95 (TU-PJC)", B["decision"]["UB_star_NI"], 1.009, H3)
    check("block B NI UB95 vs PJC-local*", B["decision"]["per_rival_ub"]["PJC-local@D"], 0.992, H3)
    check("block B NI below margin 1.05", float(B["decision"]["UB_star_NI"] < 1.05), 1, 0)
    check("Clopper-Pearson UB for 0/200", A["false_streams"]["TU-FDC@D"]["cp_ub95"], 0.0149, 1e-4)


def c2_text():
    section("C2 text: stopping fractions, strength cost, dense-monitoring cuts, endpoint ranges, abstract and intro")
    dA, dB = A["describe"], B["describe"]
    check("Criteo TU-FDC N80/tau", dA["TU-FDC@D"]["geomean_N80_pen_over_tau"], 0.125, 6e-4)
    check("X5 TU-FDC N80/tau", dB["TU-FDC@D"]["geomean_N80_pen_over_tau"], 0.236, 6e-4)
    for d, lab in [(A, "Criteo"), (B, "X5")]:
        for r in ("HC-WoR@D", "RECT-ck-BF-TU@D"):
            check(f"{lab}: TU-FDC earlier on every stream vs {r}", d["comparisons"]["TU-FDC@D|N80_pen"][r]["frac_faster"], 1.0, 0)
    v7 = J(RES / "full/v7_analysis_A.json")
    pe7 = v7["per_eps"][list(v7["per_eps"])[0]]
    v7hc = pe7["FDC-BF|N80_pen"]["HC-WoR"]["geomean_ratio"]
    check("v7 Criteo FDC-BF/HC-WoR (checkpoint)", v7hc, 0.655, H3)
    check("v7 Criteo FDC-BF/RECT-ck-HG (checkpoint)", pe7["FDC-BF|N80_pen"]["RECT-ck-HG"]["geomean_ratio"], 0.643, H3)
    v8 = J(RES / "full/v8_analysis_A.json")
    v8hc = v8["per_eps"]["0.02"]["FDC-BF|N80_pen"]["HC-WoR"]["geomean_ratio"]
    check("v8 X5 FDC-BF/HC-WoR* (checkpoint)", v8hc, 0.362, H3)
    hcA = A["comparisons"]["TU-FDC@D|N80_pen"]["HC-WoR@D"]["geomean_ratio"]
    hcB = B["comparisons"]["TU-FDC@D|N80_pen"]["HC-WoR@D"]["geomean_ratio"]
    check("strength cost on Criteo, points (0.655 -> 0.708)", 100 * (hcA - v7hc), 5, 0.6)
    check("strength cost on X5, points (0.362 -> 0.374)", 100 * (hcB - v8hc), 1, 0.6)
    # summary.md comparison table values
    sm = (RES / "full/v9_summary.md").read_text()
    for s_ in ["| v7 A (K = 20, checkpoint guarantee) | CR9 | FDC-BF / HC-WoR* | 0.655 | 0.670 |",
               "| v8 A | X5 | FDC-BF / HC-WoR* | 0.362 | 0.367 |"]:
        ok = s_ in sm
        print(f"  [{'OK' if ok else 'MISMATCH'}] v9_summary.md row: {s_}")
        if not ok:
            FAIL.append("summary comparison row")
    cut = lambda d, m: 1 - d["describe"][f"{m}@D"]["geomean_N80_pen_over_tau"] / d["describe"][f"{m}@K"]["geomean_N80_pen_over_tau"]  # noqa: E731
    check("dense cut of HC-WoR* rows, Criteo (11%)", 100 * cut(A, "HC-WoR"), 11, 0.5)
    check("dense cut of HC-WoR* rows, X5 (6%)", 100 * cut(B, "HC-WoR"), 6, 0.5)
    tcA = 1 - dA["TU-FDC@D"]["geomean_N80_pen_over_tau"] / dA["FDC-BF@K"]["geomean_N80_pen_over_tau"]
    tcB = 1 - dB["TU-FDC@D"]["geomean_N80_pen_over_tau"] / dB["FDC-BF@K"]["geomean_N80_pen_over_tau"]
    check("dense cut of TU-FDC rows, Criteo (4%)", 100 * tcA, 4, 0.5)
    check("dense cut of TU-FDC rows, X5 (3.5%)", 100 * tcB, 3.5, 0.05)
    check("same as paired ratio TU-FDC@D / FDC-BF@K, Criteo", A["comparisons"]["TU-FDC@D|N80_pen"]["FDC-BF@K"]["geomean_ratio"], 1 - tcA, 1e-9)
    check("Supp S12: HC-WoR* dense N80/tau, Criteo", dA["HC-WoR@D"]["geomean_N80_pen_over_tau"], 0.176, H3)
    check("Supp S12: HC-WoR* 20-checkpoint N80/tau, Criteo", dA["HC-WoR@K"]["geomean_N80_pen_over_tau"], 0.197, H3)
    check("Supp S12: FDC-BF 20-checkpoint N80/tau, Criteo", dA["FDC-BF@K"]["geomean_N80_pen_over_tau"], 0.130, H3)
    for d, lab, lo_p, hi_p in [(A, "Criteo", 0.530, 0.770), (B, "X5", 0.307, 0.398)]:
        vals = [d["comparisons"][f"TU-FDC@D|{k}"][r]["geomean_ratio"] for k in ("x12", "N100_pen") for r in ("HC-WoR@D", "RECT-ck-BF-TU@D")]
        check(f"{lab} x12/N100 time-uniform ratio range low", min(vals), lo_p, H3)
        check(f"{lab} x12/N100 time-uniform ratio range high", max(vals), hi_p, H3)
    check("Criteo exact rectangle (77-time ledger) needs x TU-FDC rows", 1 / A["comparisons"]["TU-FDC@D|N80_pen"]["RECT-ck-HG@D"]["geomean_ratio"], 1.6, 0.05)
    check("X5 exact rectangle (77-time ledger) needs x TU-FDC rows", 1 / B["comparisons"]["TU-FDC@D|N80_pen"]["RECT-ck-HG@D"]["geomean_ratio"], 2.8, 0.05)
    # abstract / intro / conclusion
    for d, lab, r, p in [(A, "Criteo", "HC-WoR@D", 0.71), (A, "Criteo", "RECT-ck-BF-TU@D", 0.53), (B, "X5", "HC-WoR@D", 0.37), (B, "X5", "RECT-ck-BF-TU@D", 0.30)]:
        check(f"abstract: {lab} TU-FDC/{r} (2 d.p.)", d["comparisons"]["TU-FDC@D|N80_pen"][r]["geomean_ratio"], p, 0.005)
    tu = [d["comparisons"]["TU-FDC@D|N80_pen"][r]["geomean_ratio"] for d in (A, B) for r in ("HC-WoR@D", "RECT-ck-BF-TU@D")]
    cp = [0.643, 0.655, 0.364, 0.362]  # v7/v8 primary rectangle ratios, verified by verify_r3_numbers.py
    check("thesis range low (0.30)", min(tu + cp), 0.30, 0.005)
    check("thesis range high (0.71)", max(tu + cp), 0.71, 0.005)
    check("matched rectangle needs 1/0.530 = 1.9x (Criteo)", 1 / A["comparisons"]["TU-FDC@D|N80_pen"]["RECT-ck-BF-TU@D"]["geomean_ratio"], 1.9, 0.05)
    for blk, d, key, unit, p, tol in [("A", A, "HC-WoR@D", 1e6, 1.23, 0.005), ("A", A, "RECT-ck-BF-TU@D", 1e6, 1.64, 0.005),
                                      ("B", B, "HC-WoR@D", 1e3, 63.4, 0.05), ("B", B, "RECT-ck-BF-TU@D", 1e3, 78.0, 0.05)]:
        check(f"intro rows {blk} {key}", d["describe"][key]["geomean_N80_pen_over_tau"] * d["describe"][key]["tau_R"] / unit, p, tol)
    for s_ in ["0.71 and 0.53 times", "(one-sided 95\\% bounds 0.723, 0.542)", "0.37 and 0.30 times", "(0.379, 0.308)",
               "0.87M rows against 1.23M", "1.64M", "23.7k rows against 63.4k and 78.0k", "0.30--0.71 times",
               "0.708 (UB95 0.723) times HC-WoR's rows and 0.530 (UB95 0.542)", "ratios 0.374 (UB95 0.379) and 0.304 (UB95 0.308)",
               "0.655 to 0.708 against HC-WoR", "(0.362 to 0.374)", "by 11\\% on Criteo and 6\\% on X5", "by only 4\\% and 3.5\\%",
               "0.530--0.770 on Criteo and 0.307--0.398 on X5", "1.6 and 2.8 times", "0.125\\,$\\tau_R$", "0.236\\,$\\tau_R$"]:
        present("main text", s_, "main")


def c3_text():
    section("C3 text: non-inferiority at 77 monitoring times; dev headroom")
    nb = B["comparisons"]["TU-FDC@D|N80_pen"]
    na = A["comparisons"]["TU-FDC@D|N80_pen"]
    for k, p in [("PJC-local@D", (0.985, 0.977, 0.994, 0.992)), ("TU-PJC@D", (1.004, 0.997, 1.010, 1.009))]:
        r = nb[k]
        for lab, comp, pv in zip(("ratio", "CI low", "CI high", "UB95"),
                                 (r["geomean_ratio"], r["ci95_two_sided"][0], r["ci95_two_sided"][1], r["ub95_one_sided"]), p):
            check(f"X5 TU-FDC/{k} {lab}", comp, pv, H3)
    check("Criteo TU-FDC/PJC-local*@D (descriptive)", na["PJC-local@D"]["geomean_ratio"], 0.997, H3)
    check("Criteo TU-FDC/TU-PJC (descriptive)", na["TU-PJC@D"]["geomean_ratio"], 1.000, H3)
    dev = J(RES / "pilots/v9_analysis_B_dev.json")
    check("dev headroom of the 1.05 margin against TU-PJC (0.034)", 1.05 - dev["decision"]["per_rival_ub"]["TU-PJC@D"], 0.034, 6e-4)
    devA = J(RES / "pilots/v9_analysis_A_dev.json")
    check("dev threshold basis, Criteo TU-FDC/HC-WoR", devA["comparisons"]["TU-FDC@D|N80_pen"]["HC-WoR@D"]["geomean_ratio"], 0.708, H3)
    check("dev threshold basis, Criteo UB95", devA["decision"]["per_rival_ub"]["HC-WoR@D"], 0.737, H3)
    check("dev threshold basis, X5 TU-FDC/HC-WoR", dev["comparisons"]["TU-FDC@D|N80_pen"]["HC-WoR@D"]["geomean_ratio"], 0.356, H3)
    check("dev threshold basis, X5 UB95", dev["decision"]["per_rival_ub"]["HC-WoR@D"], 0.365, H3)
    for s_ in ["0.985 [0.977, 0.994] (UB95 0.992)", "1.004 [0.997, 1.010] (UB95 1.009)", "only 0.034 of development headroom",
               "0.997 and 1.000 (supplement S12)", "development ratios of 0.708 and 0.356"]:
        present("main text", s_, "main")


def c4_ablation_eval():
    section("C4: matched time-uniform rectangle; ablation rows on the lock-v9 evaluation streams")
    check("Criteo TU-FDC/RECT-ck-BF-TU", A["comparisons"]["TU-FDC@D|N80_pen"]["RECT-ck-BF-TU@D"]["geomean_ratio"], 0.530, H3)
    check("X5 TU-FDC/RECT-ck-BF-TU", B["comparisons"]["TU-FDC@D|N80_pen"]["RECT-ck-BF-TU@D"]["geomean_ratio"], 0.304, H3)
    for d, lab, hg_p, loc_p, hg_pct, loc_pct in [(A, "Criteo", 0.98, 0.94, 2, 6), (B, "X5", 0.98, 1.00, 2, 0)]:
        k = d["comparisons"]["FDC-BF@K|N80_pen"]
        hg, loc = 1 / k["FDC-HG@K"]["geomean_ratio"], 1 / k["FDC-LOC@K"]["geomean_ratio"]
        check(f"{lab} FDC-HG/FDC-BF (eval, 2 d.p.)", hg, hg_p, 0.005)
        check(f"{lab} FDC-LOC/FDC-BF (eval, 2 d.p.)", loc, loc_p, 0.005)
        check(f"{lab} FDC-HG fewer rows, % (supp S12)", 100 * (1 - hg), hg_pct, 0.5)
        check(f"{lab} FDC-LOC fewer rows, % (supp S12)", 100 * (1 - loc), loc_pct, 0.5)
    present("main text", "FDC-HG 0.98 on both logs; FDC-LOC 0.94 and 1.00", "main")
    present("main text", "TU-FDC needs 0.530 and 0.304 times those of its time-uniform form", "main")


def hillstrom():
    section("Hillstrom, lock-v9 block C (evaluation half)")
    tb, ta = C["per_task"]["v9c_full_b"], C["per_task"]["v9c_full_a"]
    pe = tb["per_eps"]["0.0125"]
    r = pe["FDC-BF|N80_pen"]
    for k, p, lo, hi, ub in [("RECT-ck-HG[0.2]", 0.807, 0.798, 0.815, 0.814), ("HC-WoR{nstar,0.75,1}[0.2]", 0.791, 0.782, 0.801, 0.799)]:
        check(f"design B FDC-BF/{k}", r[k]["geomean_ratio"], p, H3)
        check(f"design B FDC-BF/{k} CI low", r[k]["ci95_two_sided"][0], lo, H3)
        check(f"design B FDC-BF/{k} CI high", r[k]["ci95_two_sided"][1], hi, H3)
        check(f"design B FDC-BF/{k} UB95", r[k]["ub95_one_sided"], ub, H3)
    check("design B FDC-BF/PJC-local*", r["PJC-BF[local,b=4,neyman]"]["geomean_ratio"], 0.993, H3)
    check("design B FDC-BF/PJC-local* UB95", r["PJC-BF[local,b=4,neyman]"]["ub95_one_sided"], 0.997, H3)
    check("design B tau_R (rows)", pe["describe"]["FDC-BF"]["tau_R"], 31881, 0)
    check("design B RECT-ck-HG* share at tau_R (%)", 100 * pe["describe"]["RECT-ck-HG[0.2]"]["share_N80_at_tau"], 20.5, 0.05)
    check("design B HC-WoR* share at tau_R (%)", 100 * pe["describe"]["HC-WoR{nstar,0.75,1}[0.2]"]["share_N80_at_tau"], 31, 0.05)
    rb = pe["running_bound_ratio_k18_FDC-BF_over"]
    check("design B running-bound ratio vs RECT-ck-HG* (0.577 / 0.58)", rb["RECT-ck-HG[0.2]"]["ratio"], 0.577, H3)
    check("design B running-bound ratio vs HC-WoR* (0.557 / 0.56)", rb["HC-WoR{nstar,0.75,1}[0.2]"]["ratio"], 0.557, H3)
    check("design B running-bound eligible pairs", rb["RECT-ck-HG[0.2]"]["eligible"], 162, 0)
    check("running-bound checkpoint index = K - 2", rb["RECT-ck-HG[0.2]"]["checkpoint_index"], 18, 0)
    fs = sum(v["false_streams"] for t in (tb, ta) for e in t["per_eps"].values() for v in e["false_streams"].values())
    check("Hillstrom false streams, every method, every eps, both designs", fs, 0, 0)
    ok = tb["wording"]["wording"] == "faster_descriptive" and ta["wording"]["wording"] == "faster_descriptive" \
        and tb["replica_status"] == "pass" and ta["replica_status"] == "pass"
    print(f"  [{'OK' if ok else 'MISMATCH'}] wording rule: B {tb['wording']['wording']}, A {ta['wording']['wording']}; replicas pass")
    if not ok:
        FAIL.append("Hillstrom wording/replica")
    pa = ta["per_eps"]["0.0075"]["FDC-BF|N80_pen"]
    for k in ("RECT-ck-HG[bal]", "HC-WoR{prpl,0.5,-}[bal]", "RECT-ck-BF"):
        check(f"design A FDC-BF/{k}", pa[k]["geomean_ratio"], 0.857, H3)
        check(f"design A FDC-BF/{k} UB95", pa[k]["ub95_one_sided"], 0.864, H3)
        check(f"design A {k} share at tau_R", ta["per_eps"]["0.0075"]["describe"][k]["share_N80_at_tau"], 1.0, 0)
    check("design A FDC-BF/PJC-local*", pa["PJC-BF[local,b=,bal]"]["geomean_ratio"], 1.001, H3)
    for s_ in ["0.807 (UB95 0.814) and 0.791 (UB95 0.799)", "0.993 (UB95 0.997)", "20.5\\% and 31\\%", "0.58 and 0.56 times theirs (162 of 200",
               "31,881 rows"]:
        present("main text", s_, "main")
    for s_ in ["0.807 [0.798, 0.815] (UB95 0.814) and 0.791 [0.782, 0.801] (UB95 0.799)", "0.577 and 0.557",
               "0.857 (UB95 0.864) against every rectangle and 1.001"]:
        present("supplement S13", s_, "supp")


def summary_crosscheck():
    section("v9_summary.md agrees with the analysis JSON (every ratio / UB pair in its tables)")
    sm = (RES / "full/v9_summary.md").read_text()
    n = 0
    for blk, d in (("A", A), ("B", B)):
        part = sm.split(f"## Block {blk}")[1].split("## Block")[0]
        dense, grid = part.split("On the K = 20 grid")
        for line in dense.splitlines():
            m = re.match(r"\| ([^|]+?)@D \| [^|]+ \| ([\d.]+) \| ([\d.]+) \|", line)
            if m:
                key = m.group(1).replace("*", "") + "@D"
                r = d["comparisons"]["TU-FDC@D|N80_pen"][key]
                check(f"summary {blk} TU-FDC/{key}", r["geomean_ratio"], float(m.group(2)), H3)
                check(f"summary {blk} TU-FDC/{key} UB95", r["ub95_one_sided"], float(m.group(3)), H3)
                n += 1
        for line in grid.splitlines():
            m = re.match(r"\| ([^|]+?)@K \| ([\d.]+) \| ([\d.]+) \|", line)
            if m:
                key = m.group(1).replace("*", "") + "@K"
                r = d["comparisons"]["FDC-BF@K|N80_pen"][key]
                check(f"summary {blk} FDC-BF/{key}", r["geomean_ratio"], float(m.group(2)), H3)
                check(f"summary {blk} FDC-BF/{key} UB95", r["ub95_one_sided"], float(m.group(3)), H3)
                n += 1
    check("summary rows cross-checked (dense + K=20 with UB)", n, 22, 0)


def supplement_text():
    section("Supplement S2, S9, S12, S16 and the moved X5 rows (S6)")
    for s_ in ["0.708, UB95 0.723, and 0.374, UB95 0.379", "0.530, UB95 0.542, and 0.304, UB95 0.308",
               "non-inferiority UB95 0.992 against PJC-local* and 1.009 against TU-PJC"]:
        present("S9", s_, "supp")
    for s_ in ["0.708 (UB95 0.723) and TU-FDC / RECT-ck-BF-TU 0.530 (UB95 0.542)", "0.374 (UB95 0.379) and 0.304 (UB95 0.308)",
               "(0.985, UB95 0.992)", "(1.004, UB95 1.009)", "0.197 to 0.176 against 0.130 to 0.125", "from the lock-v7 value 0.655 to 0.708",
               "from the lock-v8 value 0.362 to 0.374", "2\\% and 6\\% fewer rows than FDC-BF on Criteo and 2\\% and 0\\% on X5"]:
        present("S12", s_, "supp")
    for s_ in ["0.708 (UB95 0.737) and 0.356 (UB95 0.365)", "0.034 of development headroom", "observed UB95 0.723, 0.379 and non-inferiority 1.009"]:
        present("S2", s_, "supp")
    for d, lab, r, p in [(A, "Criteo", "HC-WoR@D", 29), (A, "Criteo", "RECT-ck-BF-TU@D", 47), (B, "X5", "HC-WoR@D", 63), (B, "X5", "RECT-ck-BF-TU@D", 70)]:
        check(f"S16: {lab} fewer rows than {r}, %", 100 * (1 - d["comparisons"]["TU-FDC@D|N80_pen"][r]["geomean_ratio"]), p, 0.5)
    present("S16", "29\\% and 47\\% fewer rows on Criteo and 63\\% and 70\\% fewer on X5", "supp")
    v8 = J(RES / "full/v8_analysis_A.json")["per_eps"]["0.02"]
    for meth, p in [("FDC", (0.785, 0.772, 0.798, 0.795, 0.779, 0.786, 0.742, 0.753, 91.5, 8.5, 0)),
                    ("FDC-MR[front3]", (0.852, 0.838, 0.867, 0.864, 0.851, 0.856, 0.838, 0.850, 70, 30, 0))]:
        n80, x12, n100 = (v8[f"FDC-BF|{k}"][meth] for k in ("N80_pen", "x12", "N100_pen"))
        comp = (n80["geomean_ratio"], n80["ci95_two_sided"][0], n80["ci95_two_sided"][1], n80["ub95_one_sided"],
                x12["geomean_ratio"], x12["ub95_one_sided"], n100["geomean_ratio"], n100["ub95_one_sided"],
                100 * n80["frac_faster"], 100 * n80["frac_tied"], 100 * n80["frac_slower"])
        for i, (c_, p_) in enumerate(zip(comp, p)):
            check(f"S6 moved X5 row {meth} [{i}]", c_, p_, H3 if i < 8 else 0.05)
    d8 = J(RES / "full/v8_analysis_A.json")["describe"]
    for meth, p in [("FDC", 31.1), ("FDC-MR[front3]", 28.6)]:
        check(f"S6 moved X5 row {meth} rows (k)", d8[f"{meth}@0.02"]["geomean_N80_pen_over_tau"] * 100393 / 1e3, p, 0.05)


def no_placeholders():
    section("No lock-v9 placeholders remain")
    for name, txt in (("main", MAIN), ("supp", SUPP)):
        n = txt.count("vninePending")
        check(f"{name}: vninePending occurrences", n, 0, 0)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    rc = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_r4_numbers.py")), *args]).returncode
    print(f"\n== verify_r4_numbers.py exit status {rc}")
    table1_panel_a()
    verdicts_and_validity()
    c2_text()
    c3_text()
    c4_ablation_eval()
    hillstrom()
    summary_crosscheck()
    supplement_text()
    no_placeholders()
    print(f"\nr4b: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if (FAIL or rc) else 0)


if __name__ == "__main__":
    main()
