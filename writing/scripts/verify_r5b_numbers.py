"""Recompute every number added in paper revision r5 part 2 (lock v10, FDC-DP) from result files (read-only).

1. Runs verify_r5_numbers.py (which runs the r4b -> r4 -> r3 chain) unchanged; it must report 0 mismatches.
2. Checks the lock-v10 numbers printed in main.tex (abstract, Sec. 4, C2, Table 2, Sec. 6.5, conclusion) and in the
   supplement (S17 text, ledger table, generated tables, S2 rows) against
     exp/results/full/v10_analysis_{A,B,D}.json       lock-v10 evaluation analyses
     exp/results/pilots/fdc_dp_v2/analysis.json        v2 development record (epsilon rule, runtime, branch-only)
     plan/prereg_lock_v10_addendum.json                lock hash
   and recomputes the ledger (M, beta_J, beta_C) by exact counting.
3. Checks that every v10 placeholder macro is gone and that the generated supplement tables match the generator.
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r5b_numbers.py [--skip-census]
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from math import comb, log
from pathlib import Path

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
MAIN = (IT / "writing/latex_acm/main.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
TABDIR = IT / "writing/supplement/r5b_tables"
TABS = "".join(p.read_text() for p in sorted(TABDIR.glob("*.tex")))
FAIL = []
H3 = 6e-4


def check(label, computed, printed, tol):
    ok = abs(computed - printed) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: computed {computed:.6g}, paper {printed}")
    if not ok:
        FAIL.append(label)


def present(label, needle, where):
    txt = {"main": MAIN, "supp": SUPP, "tabs": TABS}[where]
    ok = needle in txt
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {needle!r} printed in {where}")
    if not ok:
        FAIL.append(f"{label} (text)")


def absent(label, needle, where):
    txt = {"main": MAIN, "supp": SUPP}[where]
    ok = needle not in txt
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {needle!r} absent from {where}")
    if not ok:
        FAIL.append(f"{label} (stale text)")


def section(t):
    print(f"\n== {t}")


AN = {b: json.load(open(RES / f"full/v10_analysis_{b}.json")) for b in "ABD"}
DEV = json.load(open(RES / "pilots/fdc_dp_v2/analysis.json"))
CONF = [("A", "v10a_full_s16", "X5", 16), ("A", "v10a_full_s32", "X5", 32), ("B", "v10b_full_s16", "Lenta", 16),
        ("B", "v10b_full_s32", "Lenta", 32), ("B", "v10b_full_s64", "Lenta", 64)]
DESC = [("D", "v10d_full_x5s64", "X5", 64)]
P, R, HC, SA = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP", "FDC-DP(a)"


def cell(b, t):
    return AN[b]["cells"][t]


def cmp_(b, t, key=f"{P}/{R}"):
    return cell(b, t)["comparisons"][key]


def meth(b, t, m):
    return cell(b, t)["methods"][m]


def r5_chain(args):
    section("r5 chain (verify_r5 -> r4b -> r4 -> r3), unchanged; must report 0 mismatches")
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_r5_numbers.py")), *args],
                       capture_output=True, text=True)
    out = p.stdout.splitlines()
    n_ok = sum("[OK]" in l for l in out)
    bad = [l.strip() for l in out if "[MISMATCH]" in l and "[SUPERSEDED]" not in l]
    last = [l for l in out if l.startswith("r5:")]
    print(f"  chain: {n_ok} OK lines; final line: {last[-1] if last else '(none)'}; exit {p.returncode}")
    for l in out:
        if "body sentences" in l or "thesis words" in l:
            print("  " + l.strip())
    if p.returncode != 0:
        for l in bad:
            print("  " + l)
        FAIL.append("r5 chain")


def verdicts():
    section("Verdicts, replicas, false streams (lock v10)")
    for b in "AB":
        d = AN[b]["decision"]
        ok = d["verdict"] == "positive_result_achieved" and d["replica_status"] == "pass"
        print(f"  [{'OK' if ok else 'MISMATCH'}] block {b}: verdict {d['verdict']}, replica {d['replica_status']}")
        if not ok:
            FAIL.append(f"block {b} verdict")
    check("block D replica status pass", float(AN["D"]["replica_status_combined"]["status"] == "pass"), 1, 0)
    check("block A governing UB (0.193)", AN["A"]["decision"]["UB_star"], 0.193, H3)
    check("block B governing UB (0.636)", AN["B"]["decision"]["UB_star"], 0.636, H3)
    for b, t, *_ in CONF:
        check(f"{t}: UB95 < 0.80 (C2 claim)", float(cmp_(b, t)["ub95_one_sided"] < 0.80), 1, 0)
    fs = sum(m["false_streams"] for b in "ABD" for c in AN[b]["cells"].values() for m in c["methods"].values())
    check("false streams, every method and cell, whole run", fs, 0, 0)
    ns = {c["n_streams"] for b in "ABD" for c in AN[b]["cells"].values()}
    check("200 streams per cell", float(ns == {200}), 1, 0)
    cp = {round(m["cp_ub95"], 4) for b in "ABD" for c in AN[b]["cells"].values() for m in c["methods"].values()}
    check("Clopper-Pearson bound per cell (0.0149)", max(cp), 0.0149, 1e-4)
    present("supp S17", "one-sided Clopper--Pearson bound 0.0149 per cell", "supp")
    present("supp S17", "the governing bounds are 0.193 (block A, X5 $S = 16$) and 0.636 (block B, Lenta $S = 16$)", "supp")
    present("main 6.5", "Both blocks passed (Table~\\ref{tab:scale}), with no false stream for any method and passing replicas", "main")


def fmt_ctrl(x):
    if x == 0:
        return "0"
    if x == 1:
        return "1"
    return f"{x:.2f}"


def table2():
    section("Main Table 2 (tab:scale), row by row from the JSON")
    for b, t, log_, S in CONF + DESC:
        c = cell(b, t)
        r = cmp_(b, t)
        lo, hi = r["ci95_two_sided"]
        nf, nr = meth(b, t, P)["geomean_N80_over_tau"], meth(b, t, R)["geomean_N80_over_tau"]
        ef, er = meth(b, t, P)["exhaustion_at_k80_mean"]["ctrl"], meth(b, t, R)["exhaustion_at_k80_mean"]["ctrl"]
        lab = f"{log_}, {S}" + ("$^\\dagger$" if b == "D" else "")
        row = (f"{lab} & {c['eps']:g} & {r['geomean_ratio']:.3f} [{lo:.3f}, {hi:.3f}] & {r['ub95_one_sided']:.3f} & "
               f"{nf:.3f} / {nr:.3f} & {fmt_ctrl(ef)} / {fmt_ctrl(er)} \\\\")
        present(f"Table 2 row {t}", row, "main")


def text_main():
    section("Main text numbers (abstract, intro, C2, Sec. 4, Sec. 6.5, conclusion)")
    rat = [cmp_(b, t)["geomean_ratio"] for b, t, *_ in CONF]
    check("confirmatory ratio range low (0.17)", min(rat), 0.17, 0.005)
    check("confirmatory ratio range high (0.63)", max(rat), 0.63, 0.005)
    for s_ in ["it needed 0.17--0.63 times the reads of the matched time-uniform rectangle on X5 and Lenta",
               "A knapsack dynamic program carries this to 16--64 segments, with 0.17--0.63 times the matched rectangle's rows",
               "FDC-DP's bound against the matched time-uniform rectangle is below 0.80 in each of five cells on X5 and Lenta"]:
        present("abstract / conclusion / C2", s_, "main")
    vals = {t: cmp_(b, t)["geomean_ratio"] for b, t, *_ in CONF}
    for t, p in [("v10a_full_s16", 0.191), ("v10a_full_s32", 0.169), ("v10b_full_s16", 0.632),
                 ("v10b_full_s32", 0.516), ("v10b_full_s64", 0.513)]:
        check(f"6.5 ratio {t}", vals[t], p, H3)
    present("6.5 ratios", "TU-FDC-DP needed 0.191 and 0.169 times the rectangle's rows on X5 ($S$ = 16, 32) and 0.632, 0.516 and 0.513 times on Lenta", "main")
    sd64 = cmp_("B", "v10b_full_s64")["sd_log_ratio"]
    check("Lenta S = 64 zero spread (sd of log ratio)", sd64, 0, 0)
    check("Lenta S = 64 UB equals ratio", cmp_("B", "v10b_full_s64")["ub95_one_sided"] - cmp_("B", "v10b_full_s64")["geomean_ratio"], 0, 1e-12)
    present("6.5 quantisation", "Lenta's zero-width interval at $S = 64$ is checkpoint quantisation", "main")
    # exhaustion
    check("Lenta S16 primary control exhaustion (90%)", 100 * meth("B", "v10b_full_s16", P)["exhaustion_at_k80_mean"]["ctrl"], 90, 0.5)
    ctl = [meth("B", t, P)["exhaustion_at_k80_mean"]["ctrl"] for t in ("v10b_full_s32", "v10b_full_s64")]
    check("Lenta S32/64 primary control exhaustion at most 2% (max)", 100 * max(ctl), 2, 0.5)
    check("S17: at most 2.1%", 100 * max(ctl), 2.1, 0.05)
    rc = [meth("B", t, R)["exhaustion_at_k80_mean"]["ctrl"] for t in ("v10b_full_s16", "v10b_full_s32", "v10b_full_s64")]
    check("Lenta rectangle needs every control pool exhausted (min ctrl fraction)", min(rc), 1.0, 0)
    x5f = [meth(b, t, P)["geomean_N80_over_tau"] for b, t in (("A", "v10a_full_s16"), ("A", "v10a_full_s32"), ("D", "v10d_full_x5s64"))]
    check("X5 primary reads low (14%)", 100 * min(x5f), 14, 0.5)
    check("X5 primary reads high (16%)", 100 * max(x5f), 16, 0.5)
    ex = [meth(b, t, P)["exhaustion_at_k80_mean"]["all"] for b, t in (("A", "v10a_full_s16"), ("A", "v10a_full_s32"), ("D", "v10d_full_x5s64"))]
    check("X5 primary: no pool exhausted at N80", max(ex), 0, 0)
    x5r = [meth(b, t, R)["geomean_N80_over_tau"] for b, t in (("A", "v10a_full_s16"), ("A", "v10a_full_s32"))]
    check("X5 rectangle reads about 81% (confirmatory cells)", 100 * max(x5r), 81, 0.5)
    x5rd = x5r + [meth("D", "v10d_full_x5s64", R)["geomean_N80_over_tau"]]
    check("S17: rectangles need 81--83% (high)", 100 * max(x5rd), 83, 0.5)
    for s_ in ["At $S = 16$ TU-FDC-DP also waits for 90\\% of them, so that gain is exhaustion-assisted; at $S = 32$ and 64 it certifies with at most 2\\% empty",
               "On X5 it certifies after 14--16\\% of the table with no pool empty, against about 81\\% for the rectangle",
               "the cells are not a controlled scaling experiment",
               "Exhaustion qualifies Lenta, which is about 25\\% control"]:
        present("6.5 exhaustion", s_, "main")
    # Sec. 4 numbers
    s64 = [meth(b, t, P)["sec_cert_per_ck_max"] for b, t in (("B", "v10b_full_s64"), ("D", "v10d_full_x5s64"))]
    check("Sec. 4: at most 0.42 s per checkpoint at S = 64", max(s64), 0.42, 0.005)
    present("Sec. 4 runtime", "At $S = 64$ all 15 certificates took at most 0.42\\,s per checkpoint", "main")
    bnb = sum(meth(b, t, P)["bnb_calls"] for b, t, *_ in CONF + DESC)
    cap = sum(meth(b, t, P)["node_limit_hits"] for b, t, *_ in CONF + DESC)
    bo = sum(meth(b, t, P)["b_only_certs"] for b, t, *_ in CONF + DESC)
    check("B&B calls over the six tasks (215,541)", bnb, 215541, 0)
    check("node-cap hits (0)", cap, 0, 0)
    check("branch-only certificates (19)", bo, 19, 0)
    present("supp S17", "found 19 certificates that scheme (a) missed, over 215,541 branch-and-bound calls without a cap hit", "supp")
    tied = [cmp_(b, t, f"{P}/{SA}")["frac_tied"] for b, t, *_ in CONF + DESC]
    check("(b) = (a) on N80: fewest tied streams (199)", 200 * min(tied), 199, 0)
    check("(b) = (a) on N80: most tied streams (200)", 200 * max(tied), 200, 0)
    present("supp S17", "Scheme (b) equals scheme (a) on $N_{80}$ on 199 or 200 of 200 streams per cell", "supp")
    # HC-WoR-DP on Lenta, cell D shares
    hc = sum(round(200 * meth("B", t, HC)["share_N80_lt_tau"]) for t in ("v10b_full_s16", "v10b_full_s32", "v10b_full_s64"))
    check("HC-WoR-DP reaches 12/15 before tau on Lenta (1 of 600)", hc, 1, 0)
    present("supp S17", "HC-WoR-DP reaches 12 of 15 before $\\tau_R$ on 1 of 600 Lenta streams", "supp")
    check("cell D rectangle share before tau (91%)", 100 * meth("D", "v10d_full_x5s64", R)["share_N80_lt_tau"], 91, 0.05)
    check("cell D HC share before tau (65.5%)", 100 * meth("D", "v10d_full_x5s64", HC)["share_N80_lt_tau"], 65.5, 0.05)
    check("cell D ratio (0.182)", cmp_("D", "v10d_full_x5s64")["geomean_ratio"], 0.182, H3)
    present("supp S17", "(91\\% and 65.5\\% of streams before $\\tau_R$), so its ratio of 0.182 understates", "supp")
    # epsilon per cell in Table 2 already; rule wording
    present("6.5 rule", "the smallest grid value at which the matched rectangle reaches 12 of 15 before $\\tau_R$ on at least 40 of 50 streams", "main")
    for s_ in ["Both halves are reused (X5 for the third time; Lenta for its third fresh-stream use, first confirmatory, outcome-exposed), and a weighted-centre bug was fixed before the lock",
               "the untuned betting rectangle HC-WoR-DP are descriptive, so the claim covers the named rectangles on these tables",
               "The certificate and its proof are FDC-BF's (App.~\\ref{app:proofs}, supplement S17), so the contribution is computational",
               "decides $U_q(k) \\le \\eps$ exactly over one global $\\lambda$ grid, and abstains at $2 \\cdot 10^5$ nodes",
               "Classes beyond 4,096 policies were tested only with cardinality budgets, on X5 and an outcome-exposed Lenta half"]:
        present("main disclosures / method", s_, "main")


def ledger():
    section("Ledger by exact counting (Sec. 4: beta_J = 52.34 at S = 64; supplement Table S-ledger)")
    vals = {}
    for S in (9, 16, 32, 64):
        M = 0
        for i in range(15):
            B = (Fraction(10 + 5 * i, 100) * 8 * S).__floor__()
            M += sum(comb(S, j) for j in range(B // 8 + 1))
        bJ, bC = log(M * 20 / 0.045), log(2 * S * 2 * 20 / 0.045)
        vals[S] = (M, bJ, bC)
        mant, ex = f"{M:.1e}".split("e")
        row = f"{S} & ${mant} \\cdot 10^{{{int(ex)}}}$ & {bJ:.2f} & {bC:.2f} & {bJ / bC:.2f} \\\\"
        present(f"S17 ledger row S = {S}", row, "supp")
    check("beta_J at S = 64 (52.34)", vals[64][1], 52.34, 0.005)
    check("M at S = 64 (1.2e20)", vals[64][0] / 1e20, 1.2, 0.05)
    check("M at S = 9 equals the enumerable ledger 3,339 (CR9 family)", vals[9][0], 3339, 0)
    present("Sec. 4", "so $\\beta_J$ grows linearly in $S$ (52.34 at $S = 64$)", "main")
    present("S17", "at $S = 64$ they hold $1.2 \\cdot 10^{20}$ policies in total", "supp")


def dev_record():
    section("Development record (v2) and the epsilon rule")
    rule = DEV["eps_rule"]
    for k, e in [("x5_16", 0.03), ("x5_32", 0.04), ("lenta16", 0.004), ("lenta32", 0.006), ("lenta64", 0.008)]:
        check(f"rule-selected eps {k}", rule[k]["selected_eps"], e, 0)
    for b, t, *_ in CONF + DESC:
        pass
    check("X5 S = 64: no qualifying eps (descriptive)", float(rule["x5_64"]["selected_eps"] is None), 1, 0)
    ev = {t: cell(b, t)["eps"] for b, t, *_ in CONF + DESC}
    for t, k in [("v10a_full_s16", "x5_16"), ("v10a_full_s32", "x5_32"), ("v10b_full_s16", "lenta16"),
                 ("v10b_full_s32", "lenta32"), ("v10b_full_s64", "lenta64")]:
        check(f"eval eps {t} equals the dev rule's choice", ev[t], rule[k]["selected_eps"], 0)
    check("eval eps cell D (0.05)", ev["v10d_full_x5s64"], 0.05, 0)
    g3 = DEV["go_criterion_3"]["cells"]
    check("dev runtime S = 64 median low (0.29)", min(c["b_median"] for c in g3), 0.29, 0.005)
    check("dev runtime S = 64 median high (0.33)", max(c["b_median"] for c in g3), 0.33, 0.005)
    check("dev runtime S = 64 max (0.46)", max(c["b_max"] for c in g3), 0.46, 0.005)
    check("dev node-cap hits (0)", sum(c["node_limit_hits"] for c in g3), 0, 0)
    bo = sum(v["b_only_certs"] for t in DEV["b_vs_a"].values() for v in t.values())
    check("dev branch-only certificates (25)", bo, 25, 0)
    present("S17 dev", "runtime (median 0.29--0.33\\,s and maximum 0.46\\,s per checkpoint at $S = 64$, no cap hit)", "supp")
    present("S17 dev", "branch-only counts (25 rather than 27)", "supp")
    lock = json.loads((IT / "plan/prereg_lock_v10_addendum.json").read_text())
    sha = AN["A"]["addendum_sha256"]
    ok = sha.startswith("ecaf78e1") and sha.endswith("69d82d8") and all(AN[b]["addendum_sha256"] == sha for b in "ABD")
    print(f"  [{'OK' if ok else 'MISMATCH'}] lock sha256 {sha[:8]}...{sha[-7:]} in all three analyses")
    if not ok:
        FAIL.append("lock sha")
    raw = hashlib.sha256((IT / "plan/prereg_lock_v10_addendum.json").read_bytes()).hexdigest()
    print(f"     (lock JSON file sha256 {raw[:8]}..., analyses bind {sha[:8]}...; keys: {len(lock)})")
    present("S17 lock hash", "sha256 \\texttt{ecaf78e1\\dots69d82d8}", "supp")


def generated_tables():
    section("Generated supplement tables match the generator output (no hand edits)")
    spec = importlib.util.spec_from_file_location("gen", Path(__file__).with_name("gen_r5b_supp_tables.py"))
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    an = g.load()
    for name, txt in [("v10_ratios.tex", g.ratios_table(an)), ("v10_methods.tex", g.methods_table(an)),
                      ("v10_eps_rule.tex", g.eps_table())]:
        ok = (TABDIR / name).read_text() == txt
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name} equals generator output")
        if not ok:
            FAIL.append(name)
        present(f"supplement inputs {name}", f"\\input{{r5b_tables/{name}}}", "supp")
    # spot checks of generated cells against the JSON
    hc16 = cmp_("A", "v10a_full_s16", f"{P}/{HC}")
    present("ratios table: X5 S16 vs HC-WoR-DP", f"{hc16['geomean_ratio']:.3f} [{hc16['ci95_two_sided'][0]:.3f}, {hc16['ci95_two_sided'][1]:.3f}] & {hc16['ub95_one_sided']:.3f}", "tabs")
    present("dev eps table: Lenta 16 trace", "0.003: 28 / 0; 0.004: 50 / 0", "tabs")


def placeholders_and_structure():
    section("Placeholders removed; S2 rows; glossary; claim count")
    for s_ in ["vtenPending", "vtenReserve", "v10 pending", "\\color{red}"]:
        absent("placeholder", s_, "main")
        absent("placeholder", s_, "supp")
    for c in ["C1", "C2", "C3", "C4"]:
        present("claims", f"\\item[\\textbf{{{c}}}]", "main")
    check("at most four claims", MAIN.count("\\item[\\textbf{C"), 4, 0)
    present("contribution", "FDC-DP, an exact knapsack form of the certificate that scales it to implicit policy classes, extends C1 and C2", "main")
    present("glossary", "(TU-)FDC-DP & FDC-BF's certificate over implicit classes", "main")
    present("App. A", "\\textbf{FDC-DP (Theorem DP-1).}", "main")
    present("S2 caption", "(lock v5 through lock v10)", "supp")
    check("S2 rows for v10", SUPP.count("2026-10-05 (v10) &"), 2, 0)
    present("S2 r5 row", "lock-v10 results integrated: FDC-DP added as a contribution under C1 (validity) and C2", "supp")
    present("S17 label", "\\label{app:fdcdp}", "supp")
    for s_ in ["Proposition S17.3 (exactness, safe pruning)", "Lemma S17.4 (lossless dominance)", "\\caption{Scheme (a): knapsack bound",
               "\\caption{Scheme (b): exact decision", "\\textbf{Ledger by counting.}", "\\textbf{Development record and the $\\eps$ rule.}"]:
        present("S17 contents", s_, "supp")


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r5_chain(args)
    verdicts()
    table2()
    text_main()
    ledger()
    dev_record()
    generated_tables()
    placeholders_and_structure()
    print(f"\nr5b: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
