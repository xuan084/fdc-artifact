"""Recompute every number added or reworded in paper revision r6 (parts 1, 2) and r6c from result files (read-only).

r6d (after the r6c reviews, critic writing/review_r6c_critic.md and external reviewer reviews/paper_r6c_review.md): r6d_checks()
recomputes supplement S20 (localised union ledgers, development only) from the v11 pilot rows against the block-G1
development rows, and the pricing basis of the priced-access workflow (per-record share, one-time fee per collaboration).

r6c (after the r6 reviews, external reviewer reviews/paper_r6_review.md and critic writing/review_r6_critic.md): r6c_checks() adds the
corrected G2 cost range (1-24) and branch-only denominator, the G1/G3 qualifiers (pre-horizon criterion, unpinned,
CP/TU labels, horizon-sensitive caption), the development-only G2 wording, the Sec. 7 block-G disclosure, the
priced-access workflow numbers, the strongest-rival clause, the method-name budget, the Algorithm S2 guard and the
errata entries; table1_panel_a_r6c() re-checks Table 1A under its role labels.

1. Runs verify_r5b_numbers.py (which runs the whole r5 -> r4b -> r4 -> r3 chain) unchanged. Every value check of
   that chain must still pass. A text-presence check may fail only if it is listed in SUPERSEDED below, i.e. the
   earlier wording was deliberately replaced in r6 after the r5 reviews (external reviewer reviews/paper_r5_review.md, critic
   writing/review_r5_critic.md); each superseded string has a replacement that this script checks.
2. Checks the r6 numbers and wording:
   - lock-v10 rival pinning, per stream, from exp/results/full/v10*_full_*/results.jsonl (critic W2);
   - union-cost trend beta_J / beta_C by exact counting and the (a) = (b) N80 ties (critic W1);
   - (budget, policy) memberships vs distinct policies (external reviewer 2.1);
   - exhaustion means and the withdrawn Lenta pre-exhaustion claim (external reviewer 4, v10 lock condition);
   - the epsilon rule as locked (better of two named rectangles; the matched one governed every cell);
   - block F2 equal-information vs no-pilot comparisons, the one Criteo CI that includes 1, break-even J and whole
     runs, and read-saving percentages with 1 - 1/r and 1/r - 1 (external reviewer 3.2, 4);
   - Table 1B summary row against the lock-v7 / lock-v8 analyses, and the moved panels in the supplement;
   - DP exposition fixes D1-D4 in the supplement, stale-remnant removals, abstract length, block-G placeholders.
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r6_numbers.py [--skip-census]
"""
import json
import math
import re
import subprocess
import sys
from collections import Counter
from fractions import Fraction
from math import comb, log
from pathlib import Path

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
MAIN = (IT / "writing/latex_acm/main.tex").read_text()
PRE = (IT / "writing/latex_acm/main_pre_r6.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
TABS = (IT / "writing/supplement/r5b_tables/v10_ratios.tex").read_text()
FAIL = []
H3 = 6e-4
# exp/results/full/v10_summary.md before the r6c note: original file (git HEAD 9af39b2c) and the scrubbed anonymous copy
V10_SUMMARY_PRE_SHA = ("3292fe60101cc7ce300222adf23e73c9b900a39afc8625f92bb3c5eb33f020d6",
                       "1bf87d9d1081e99372182fe35532c9b751f4070d8bcf7173685bcad36133b2d1")

# Text checks of the r5b chain deliberately replaced in r6 -> reason / replacement checked below.
SUPERSEDED = {
    "main text: 'FDC-HG 0.98 on both logs; FDC-LOC 0.94 and 1.00' printed in main":
        "ablation names moved to the glossary (critic W5): 'envelope 0.98 on both logs; localisation 0.94 and 1.00'",
    "main C3: 'needed 0.90--0.98 times the rows of each tested reset and menu design in all four regimes, with every 95\\\\% interval below 1' printed in main":
        "equal-information sentence no longer ends on a number (critic P2-1); 128 comparisons checked below",
    "main C3: 'still needs 0.89--0.99 times the rows of the no-pilot adaptive designs' printed in main":
        "no-pilot comparison separated, point estimates 0.89--0.99, one CI includes 1 (external reviewer 5.3)",
    "main C3: 'a pilot of 4.5\\\\% or 18\\\\% of $\\\\tau_R$ reverses the comparison (1.08--1.67) unless one pilot serves about 2 to 26 certification runs' printed in main":
        "point-estimate break-even 2.1--25.5 runs, 3--26 whole runs (external reviewer 5.3)",
    "main C3: 'by 9--13\\\\% on the Criteo-structured asymmetric regime' printed in main":
        "read reduction 1 - 1/r = 8--11% (external reviewer 3.2)",
    "supp S11 (F2): 'the ratios lie between 0.90 and 0.98, each with a 95\\\\% CI below 1' printed in supp":
        "'the 128 ratios lie between 0.90 and 0.98, each with a 95\\% CI below 1 (largest upper limit 0.995)'",
    "supp S11 (F2): 'still needs 0.89--0.99 times the rows of the no-pilot picks' printed in supp":
        "no-pilot comparison separated with the one CI including 1 (external reviewer 3.2)",
    "supp S11 (F2): '$J \\\\ge 2.1$--25.5 runs' printed in supp": "'$J$ = 2.1--25.5, that is 3 to 26 whole runs'",
    "intro scoping: \"Hoeffding's convex-order theorem~\\\\citep{hoeffding1963probability} carries any such moment bound over to sampling without replacement\" printed in main":
        "restricted to fixed-design MGF bounds (external reviewer S2)",
    "6.5 exhaustion: 'At $S = 16$ TU-FDC-DP also waits for 90\\\\% of them, so that gain is exhaustion-assisted; at $S = 32$ and 64 it certifies with at most 2\\\\% empty' printed in main":
        "exhaustion fractions labelled as means 89.8%, 2.1%, 0% (external reviewer 3.2)",
    "6.5 exhaustion: 'On X5 it certifies after 14--16\\\\% of the table with no pool empty, against about 81\\\\% for the rectangle' printed in main":
        "'On X5 TU-FDC-DP certifies after 14--16\\% ...'",
    "6.5 rule: 'the smallest grid value at which the matched rectangle reaches 12 of 15 before $\\\\tau_R$ on at least 40 of 50 streams' printed in main":
        "rule as locked: better of the two named rectangles, strictly before tau (external reviewer 4)",
    "main disclosures / method: \"The certificate and its proof are FDC-BF's (App.~\\\\ref{app:proofs}, supplement S17), so the contribution is computational\" printed in main":
        "same certificate family on a global lambda grid (critic W6a)",
    "main disclosures / method: 'decides $U_q(k) \\\\le \\\\eps$ exactly over one global $\\\\lambda$ grid, and abstains at $2 \\\\cdot 10^5$ nodes' printed in main":
        "exactness qualified: completed search, floating point, exponential worst case (external reviewer D4)",
    "S17: 'at $S = 64$ they hold $1.2 \\\\cdot 10^{20}$ policies in total' printed in supp":
        "(budget, policy) memberships vs at most 2^64 distinct policies (external reviewer 2.1)",
    "contribution: 'FDC-DP, an exact knapsack form of the certificate that scales it to implicit policy classes, extends C1 and C2' printed in main":
        "'a knapsack evaluation of the same certificate family on a global lambda grid' (external reviewer D4, critic W6a)",
    "glossary: \"(TU-)FDC-DP & FDC-BF's certificate over implicit classes\" printed in main":
        "'(TU-)FDC-DP & FDC-BF's certificate family on a global lambda grid over implicit classes'",
}


# r6c (2026-10-05): chain text checks deliberately replaced after the r6 reviews (external reviewer reviews/paper_r6_review.md,
# critic writing/review_r6_critic.md); each replacement is checked in r6c_checks() / table1_panel_a_r6c().
SUPERSEDED.update({
    'row not found: \\textbf{HC-WoR*} (betting), Criteo':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    'row not found: \\textbf{RECT-ck-BF-TU} (matched), Criteo':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    'row not found: \\textbf{HC-WoR*} (Neyman), X5':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    'row not found: \\textbf{RECT-ck-BF-TU} (matched), X5':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    'row not found: \\textbf{PJC-local*} (77-time ledger), X5':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    'row not found: \\textbf{TU-PJC}, X5':
        'Table 1A rows relabelled by role (critic r6 P1-4); every cell re-checked by table1_panel_a_r6c() under the new labels',
    "main text: 'by 11\\\\% on Criteo and 6\\\\% on X5' printed in main":
        'dense-monitoring percentages left the body for space (P0-2 paid by P1-4); within-v9 comparison 0.658 -> 0.708 / 0.364 -> 0.374 kept and checked (r6c)',
    "main text: 'by only 4\\\\% and 3.5\\\\%' printed in main":
        'as above; detail in supplement S12',
    "main text: '1.6 and 2.8 times' printed in main":
        '77-time exact-rectangle ratio left the body for space; descriptive row stays in supplement Table S12',
    "main text: '0.58 and 0.56 times theirs (162 of 200' printed in main":
        "censoring-based 'understates' inference removed (same objection as external reviewer r6 on Figure 2); width comparison kept in S13",
    "main text: '31,881 rows' printed in main":
        'Hillstrom size left the body for space; stays in S13',
    "main C2: 'Against HC-WoR the ratio is 0.658 at 20 checkpoints and 0.708 at 77 times on Criteo, and 0.364 and 0.374 on X5' printed in main":
        "reworded: 'moves the ratio against HC-WoR from 0.658 to 0.708 on Criteo and from 0.364 to 0.374 on X5' (checked r6c)",
    "main C4: 'TU-FDC needs 0.530 and 0.304 times the rows of RECT-ck-BF-TU, the time-uniform form of the +box variant' printed in main":
        "role name (critic P1-4): 'the time-uniform matched rectangle' (checked r6c)",
    "main C4: 'FDC-BF/RECT-ck-BF+box is 0.533 and 0.309' printed in main":
        "reworded: 'FDC-BF needs 0.533 and 0.309 times the rows of its +box variant' (checked r6c)",
    'main Sec. 4: "RECT-ck-BF-TU, the time-uniform form of the matched rectangle\'s +box variant" printed in main':
        "role name (critic P1-4): 'the time-uniform matched rectangle, its +box variant with radii frozen the same way'",
    "Table 1 caption: 'RECT-ck-BF-TU is the time-uniform form of RECT-ck-BF+box' printed in main":
        "Table 1 caption: 'Matched, TU: the matched rectangle's +box variant with radii frozen at its last checkpoint' (checked r6c)",
    "main C1: 'The blocks are lock v7 on Criteo and lock v8 on X5 at 20 checkpoints, and lock v9 on both logs at 77 monitoring times, each with a one-sided Clopper--Pearson bound of 0.0149' printed in main":
        "C1 shortened: 'The blocks are locks v7 (Criteo) and v8 (X5) at 20 checkpoints and lock v9 on both logs at 77 times' (checked r6c)",
    "related work: 'our Bernstein variant FDC is exactly that construction' printed in main":
        "FDC named only in the glossary (critic P1-4): 'our Bernstein variant is exactly that construction'",
    "T4: 'We analysed every adaptive variant at checkpoint strength only' printed in main":
        "merged: 'but we built no time-uniform version of any adaptive variant' (checked r6c)",
    "S5 price conversion: 'so they are not expected per-deployment savings' printed in main":
        "workflow paragraph (critic P0-2): 'These figures price geometric-mean reads on the tested tables, not a measured bill' (checked r6c)",
    'abstract / conclusion / C2: "A knapsack dynamic program carries this to 16--64 segments, with 0.17--0.63 times the matched rectangle\'s rows" printed in main':
        "conclusion: 'FDC-DP carries this to 16--64 segments, with 0.17--0.63 ...' (checked r6c)",
    '6.5 ratios: "TU-FDC-DP needed 0.191 and 0.169 times the rectangle\'s rows on X5 ($S$ = 16, 32) and 0.632, 0.516 and 0.513 times on Lenta" printed in main':
        "role name: 'FDC-DP needed 0.191 and 0.169 ...' (checked r6c)",
    "6.5 exhaustion: 'Exhaustion qualifies Lenta, which is about 25\\\\% control' printed in main":
        "shortened: 'Exhaustion qualifies Lenta, about 25\\% control' (checked r6c)",
    "supp S17: '(91\\\\% and 65.5\\\\% of streams before $\\\\tau_R$), so its ratio of 0.182 understates' printed in supp":
        "censoring-bias wording replaced by 'horizon-sensitive endpoint ratio' (external reviewer r6 2, Figure 2 logic) (checked r6c)",
    "main disclosures / method: 'Both halves are reused (X5 for the third time; Lenta for its third fresh-stream use, first confirmatory, outcome-exposed), and a weighted-centre bug was fixed before the lock' printed in main":
        "shortened: 'Both halves are reused (X5 a third time; Lenta a third fresh-stream use, ...)' (checked r6c)",
    "main disclosures / method: 'the untuned betting rectangle HC-WoR-DP are descriptive, so the claim covers the named rectangles on these tables' printed in main":
        "role name: 'the untuned betting rectangle are descriptive' (checked r6c)",
    "main disclosures / method: 'Classes beyond 4,096 policies were tested only with cardinality budgets, on X5 and an outcome-exposed Lenta half' printed in main":
        "external reviewer r6 3: 'Confirmatory scaling tests used cardinality budgets ...; heterogeneous-cost and robustness tests were descriptive and used development halves' (checked r6c)",
})


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


def J(p):
    return json.load(open(p))


AN = {b: J(RES / f"full/v10_analysis_{b}.json") for b in "ABD"}
CELLS = [("A", "v10a_full_s16"), ("A", "v10a_full_s32"), ("B", "v10b_full_s16"), ("B", "v10b_full_s32"),
         ("B", "v10b_full_s64"), ("D", "v10d_full_x5s64")]
P, R, SA = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "FDC-DP(a)"


def r5b_chain(args):
    section("r5b chain (verify_r5b -> r5 -> r4b -> r4 -> r3), unchanged; only listed text checks may fail")
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_r5b_numbers.py")), *args],
                       capture_output=True, text=True)
    out = p.stdout.splitlines()
    n_ok = sum("[OK]" in l for l in out)
    seen = set()
    for l in out:
        if "[MISMATCH]" not in l:
            continue
        key = l.strip().replace("[MISMATCH] ", "", 1)
        key = key[len("chain: "):] if key.startswith("chain: ") else key
        if key in SUPERSEDED:
            if key not in seen:
                print(f"  [SUPERSEDED] {key[:110]} -> {SUPERSEDED[key]}")
            seen.add(key)
        else:
            print(f"  [MISMATCH] chain: {key}")
            FAIL.append(f"chain: {key[:80]}")
    for k in SUPERSEDED:
        if k not in seen:
            print(f"  [note] superseded item not failing (string still present?): {k[:100]}")
    last = [l for l in out if l.startswith("r5b:")]
    print(f"  chain: {n_ok} OK lines; {len(seen)} superseded text checks; r5b final line: {last[-1][:60] if last else '(none)'}")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])
        FAIL.append("chain stderr")


def rival_pinning():
    section("Lock v10: the epsilon rule pins the rival (per stream, results.jsonl; critic W2)")
    k = {}
    for b, t in CELLS:
        rows = [json.loads(l) for l in open(RES / f"full/{t}/results.jsonl")]
        r = [x for x in rows if x["method"] == R]
        check(f"{t}: rival streams", len(r), 200, 0)
        check(f"{t}: K_eval = 20", float(all(x["K_eval"] == 20 for x in r)), 1, 0)
        k[t] = Counter((x["k80"], x["N80_pen"], x["N80_pen"] == x["tau_R"]) for x in r)
    for t, rows in (("v10a_full_s16", 81695), ("v10a_full_s32", 81695), ("v10b_full_s32", 274954), ("v10b_full_s64", 274954)):
        check(f"{t}: rival at checkpoint 19 of 20 (index 18) on every stream, {rows} rows", k[t][(18, rows, False)], 200, 0)
    check("Lenta S16: rival at checkpoint 19 on 197 streams", k["v10b_full_s16"][(18, 274954, False)], 197, 0)
    check("Lenta S16: other 3 streams at tau_R", sum(v for (kk, n, tau), v in k["v10b_full_s16"].items() if tau), 3, 0)
    for t, b in (("v10a_full_s16", "A"), ("v10b_full_s32", "B")):
        check(f"{t}: JSON geomean equals the pinned checkpoint", AN[b]["cells"][t]["methods"][R]["geomean_N80_pen"],
              81695 if b == "A" else 274954, 1e-6)
    fr = [AN[b]["cells"][t]["methods"][R]["geomean_N80_over_tau"] for b, t in CELLS[:5]]
    check("0.80 x rival fraction, low end (64%)", 100 * 0.8 * min(fr), 64, 0.5)
    check("0.80 x rival fraction, high end (65%)", 100 * 0.8 * max(fr), 65, 0.5)
    for s_ in ["The matched rectangle reaches 12 of 15 at checkpoint 19 of 20 on every stream of four cells and on 197 of 200 of the fifth",
               "so each ratio is FDC-DP's stopping point over a fixed rival reading",
               "at tolerances that place the rival at its penultimate checkpoint"]:
        present("main 6.5 / conclusion", s_, "main")
    present("supp S17 (viii)", "on all 200 streams of X5 $S$ = 16 and 32 (81,695 rows) and Lenta $S$ = 32 and 64 (274,954 rows), and on 197 of 200 Lenta $S = 16$ streams", "supp")


def union_trend_and_bnb():
    section("Union-cost trend and (a) = (b) ties (critic W1(c), W1(d))")
    ratios = {}
    for S in (9, 16, 32, 64):
        M = sum(sum(comb(S, j) for j in range((Fraction(10 + 5 * i, 100) * 8 * S).__floor__() // 8 + 1)) for i in range(15))
        ratios[S] = log(M * 20 / 0.045) / log(2 * S * 2 * 20 / 0.045)
        if S == 64:
            check("memberships at S = 64 exceed distinct policies 2^64", float(M > 2 ** 64), 1, 0)
            check("2^64 (1.8e19)", 2 ** 64 / 1e19, 1.8, 0.05)
            check("memberships at S = 64 (1.2e20)", M / 1e20, 1.2, 0.05)
    for S, pr in ((9, 1.47), (16, 1.86), (32, 2.75), (64, 4.50)):
        check(f"beta_J/beta_C at S = {S}", ratios[S], pr, 0.005)
    present("main 6.5", "$\\beta_J/\\beta_C$ is 1.47, 1.86, 2.75 and 4.50 at $S$ = 9, 16, 32 and 64", "main")
    tied = [AN[b]["cells"][t]["comparisons"][f"{P}/{SA}"]["frac_tied"] for b, t in CELLS]
    check("(b) changed N80 on at most 1 of 200 streams per cell", 200 * (1 - min(tied)), 1, 1e-9)
    check("(b) changed N80 on at least 0 streams (max ties 200)", 200 * max(tied), 200, 1e-9)
    present("main 6.5", "in lock v10, where every class is a cardinality budget, it changed $N_{80}$ on 0--1 of 200 streams per cell", "main")
    present("supp S17", "$1.2 \\cdot 10^{20}$ (budget, policy) memberships in total, counting a policy once for every class that contains it, among at most $2^{64} \\approx 1.8 \\cdot 10^{19}$ distinct policies", "supp")
    present("supp S17 end", "implicit classes of up to $2^{64}$ distinct policies ($1.2 \\cdot 10^{20}$ (budget, policy) memberships per frontier)", "supp")


def exhaustion_and_rule():
    section("Exhaustion means, withdrawn pre-exhaustion claim, epsilon rule as locked")
    m = lambda b, t, meth: AN[b]["cells"][t]["methods"][meth]["exhaustion_at_k80_mean"]["ctrl"]
    check("Lenta S16 primary mean control exhaustion (89.8%)", 100 * m("B", "v10b_full_s16", P), 89.8, 0.05)
    check("Lenta S32 primary mean control exhaustion (2.1%, exactly 0.02125)", m("B", "v10b_full_s32", P), 0.02125, 1e-12)
    check("Lenta S64 primary mean control exhaustion (0%)", m("B", "v10b_full_s64", P), 0, 0)
    for t in ("v10b_full_s16", "v10b_full_s32", "v10b_full_s64"):
        check(f"{t}: rival mean control exhaustion 1", m("B", t, R), 1, 0)
    both0 = [t for b, t in CELLS if m(b, t, P) == 0 and m(b, t, R) == 0]
    check("only X5 S16 has both fractions zero", float(both0 == ["v10a_full_s16"]), 1, 0)
    present("main 6.5", "FDC-DP's mean fraction is about 89.8\\% at $S = 16$, so that gain is exhaustion-assisted, and 2.1\\% and 0\\% at $S$ = 32 and 64", "main")
    present("main 6.5", "these comparisons do not isolate a pre-exhaustion width advantage", "main")
    present("main Table 2 caption", "Ctrl.\\ empty: mean fraction of control pools exhausted at $N_{80}$", "main")
    present("supp S17", "On X5 only $S = 16$ meets that condition", "supp")
    absent("Lenta pre-exhaustion claim (lock v10 condition)", "a pre-exhaustion width advantage.", "supp")
    absent("per-stream exhaustion wording", "certifies with at most 2", "main")
    rule = J(RES / "pilots/fdc_dp_v2/analysis.json")["eps_rule"]
    gov = {k: v["rival"] for k, v in rule.items() if v["selected_eps"] is not None}
    check("matched rectangle governed every selected cell", float(set(gov.values()) == {"RECT-BF-DP"} and len(gov) == 5), 1, 0)
    present("main 6.5 rule", "the smallest grid value at which the better of the two named rectangles reaches 12 of 15 strictly before $\\tau_R$ on at least 40 of 50 streams. The matched rectangle governed every selected cell", "main")
    present("main v10 scope (abstract)", "With 16--64 segments and rule-selected tolerances, it needed 0.17--0.63 times the reads of the matched time-uniform rectangle on X5 and Lenta", "main")
    present("main v10 scope (conclusion)", "with 0.17--0.63 times the matched rectangle's rows at development-rule-selected tolerances on these reused tables", "main")
    present("main v10 scope (6.5)", "FDC-DP thus keeps the joint saving over the named rectangle, at these rule-selected tolerances, where enumeration is impossible", "main")
    present("supp (vi) v10 qualifier", "no evaluation outcome was used by any v10 computation before the lock", "supp")


def pilot_comparisons():
    section("Block F2: equal information vs no-pilot comparisons; read percentages (external reviewer 5.3, 3.2)")
    envs = J(RES / "full/v8_posthoc_F2/summary.json")["envs"]
    eq = [(h[j]["ratio"], h[j]["ci95"][1]) for v in envs.values() for k, h in v["head_to_head"].items()
          if "no pilot" not in k for j in ("J1", "J4", "J16", "Jinf")]
    check("equal information: 128 comparisons", len(eq), 128, 0)
    check("equal information incl. J = 1 (full charge): all upper CI < 1", float(all(u < 1 for _, u in eq)), 1, 0)
    check("equal information: largest upper CI (0.995)", max(u for _, u in eq), 0.995, 5e-4)
    np_cr = [h["J1"] for e in ("CR9-asym", "CR9-mixed") for k, h in envs[e]["head_to_head"].items() if "no pilot" in k]
    check("Criteo no-pilot comparisons (8)", len(np_cr), 8, 0)
    inc = [h for h in np_cr if h["ci95"][1] >= 1]
    check("Criteo no-pilot: exactly one CI includes 1", len(inc), 1, 0)
    h = envs["CR9-mixed"]["head_to_head"]["FDC-BF[ney-pilot,n=1000] / PJC-A-reset[no pilot]"]["J1"]
    check("that interval: ratio 0.990", h["ratio"], 0.990, H3)
    check("that interval: lower 0.975", h["ci95"][0], 0.975, H3)
    check("that interval: upper 1.005", h["ci95"][1], 1.005, H3)
    be = [h["breakeven_J"] for e in ("X9-asym", "X9-mixed") for k, h in envs[e]["head_to_head"].items() if "no pilot" in k]
    check("break-even J low (2.1)", min(be), 2.1, 0.05)
    check("break-even J high (25.5)", max(be), 25.5, 0.05)
    check("whole runs low (ceil = 3)", math.ceil(min(be)), 3, 0)
    check("whole runs high (ceil = 26)", math.ceil(max(be)), 26, 0)
    for s_ in ["All 128 such comparisons have 95\\% intervals below 1, also with the pilot charged in full to both sides",
               "the point estimates still favour the frozen plan (0.89--0.99), although one of eight intervals includes 1",
               "in point estimates it breaks even only when one pilot serves 2.1 to 25.5 runs, that is 3 to 26 whole runs",
               "The adaptive designs were compared at checkpoint strength and not retuned with a pilot",
               "the same independent pilot of 250 or 1,000 draws per cell, as from a prior campaign"]:  # r6d: shortened for space
        present("main C3", s_, "main")
    for s_ in ["the 128 ratios lie between 0.90 and 0.98, each with a 95\\% CI below 1 (largest upper limit 0.995)",
               "one of the eight intervals, 0.990 [0.975, 1.005] (CR9-mixed, $n$ = 1,000, reset), includes 1",
               "$J$ = 2.1--25.5, that is 3 to 26 whole runs; it is not a simultaneous inferential guarantee"]:
        present("supp S11", s_, "supp")
    F = J(RES / "full/v8_posthoc_F/summary.json")["envs"]
    sp = [v["FDCBF_over_method_N80"] for k, v in F["CR9-asym"]["methods"].items()
          if (k.startswith("PJC-A") or k.startswith("PJC-menu")) and "segmenu" not in k]
    check("adaptive read reduction low, 1 - 1/r (8.0%)", 100 * (1 - 1 / min(sp)), 8.0, 0.05)
    check("adaptive read reduction high, 1 - 1/r (11.3%)", 100 * (1 - 1 / max(sp)), 11.3, 0.05)
    pn = F["CR9-asym"]["methods"]["FDC-BF[ney-pilot]"]["FDCBF_over_method_N80"]
    check("pilot-Neyman read reduction vs 50/50 (15.2%)", 100 * (1 - 1 / pn), 15.2, 0.05)
    seg = {e: [v["FDCBF_over_method_N80"] for k, v in F[e]["methods"].items() if "segmenu" in k][0] for e in ("CR9-mixed", "X9-mixed")}
    check("segment menu extra reads, CR9-mixed, 1/r - 1 (25.7%)", 100 * (1 / seg["CR9-mixed"] - 1), 25.7, 0.05)
    check("segment menu extra reads, X9-mixed, 1/r - 1 (53.9%)", 100 * (1 / seg["X9-mixed"] - 1), 53.9, 0.05)
    present("main C3", "they read 8--11\\% fewer rows (FDC-BF/method 1.087--1.127)", "main")
    present("supp S11", "read 8.0--11.3\\% fewer rows than FDC-BF (FDC-BF/method 1.087--1.127, a read reduction of $1 - 1/r$)", "supp")
    present("supp S11", "reads 15.2\\% fewer rows than 50/50 (ratio 1.180)", "supp")
    present("supp S11", "reads 25.7\\% and 53.9\\% more rows than FDC-BF (ratios 0.795 and 0.650, extra reads $1/r - 1$)", "supp")
    for s_ in ["9--13\\%", "20--35\\%", "18\\% faster", "one CI touches 1"]:
        absent("stale percentage", s_, "supp")
        absent("stale percentage", s_, "main")


def table1_summary():
    section("Table 1B summary row against lock v7 / lock v8 analyses; moved panels in the supplement")
    v7 = J(RES / "full/v7_analysis_A.json")["per_eps"]["0.001"]["FDC-BF|N80_pen"]
    v8 = J(RES / "full/v8_analysis_A.json")["per_eps"]["0.02"]["FDC-BF|N80_pen"]
    for lab, d, k, r, u in [("v7 RECT-ck-HG", v7, "RECT-ck-HG", 0.643, 0.658), ("v7 HC-WoR", v7, "HC-WoR", 0.655, 0.670),
                            ("v8 RECT-ck-HG*", v8, "RECT-ck-HG", 0.364, 0.368), ("v8 HC-WoR*", v8, "HC-WoR", 0.362, 0.367),
                            ("v8 PJC-local*", v8, "PJC-local", 1.003, 1.011), ("v8 PJC-menu*", v8, "PJC-menu", 0.968, 0.978)]:
        check(f"{lab} ratio", d[k]["geomean_ratio"], r, H3)
        check(f"{lab} UB95", d[k]["ub95_one_sided"], u, H3)
    for s_ in ["\\textbf{RECT-ck-HG} 0.643 (0.658) and \\textbf{HC-WoR} 0.655 (0.670) [$<$0.80]",
               "\\textbf{RECT-ck-HG} (tuned) 0.364 (0.368) and \\textbf{HC-WoR} (tuned) 0.362 (0.367) [$<$0.60]",
               "residual-pool \\textbf{PJC} 1.003 (1.011) and menu \\textbf{PJC} 0.968 (0.978) [$<$1.05]"]:
        present("Table 1B summary", s_, "main")
    d7 = J(RES / "full/v7_analysis_A.json")["describe"]["FDC-BF@0.001"]["geomean_N80_pen_over_tau"] * 6989799
    d8 = J(RES / "full/v8_analysis_A.json")["describe"]["FDC-BF@0.02"]["geomean_N80_pen_over_tau"] * 100393
    check("v7 FDC-BF rows (0.91M)", d7 / 1e6, 0.91, 0.005)
    check("v8 FDC-BF rows (24.4k)", d8 / 1e3, 24.4, 0.05)
    # the moved panels are byte-identical to the pre-r6 Table 1 rows
    seg = PRE[PRE.index("\\emph{B. Criteo test (lock v7)"):]
    seg = seg[:seg.index("\\bottomrule")]
    pre_rows = [l for l in seg.split("\n")[1:] if l.strip() and not l.startswith("\\midrule") and not l.startswith("\\multicolumn")]
    check("pre-r6 panel B/C rows found (10)", len(pre_rows), 10, 0)
    check("every pre-r6 panel B/C row is in supplement Table tab:ckpanels", float(all(l in SUPP for l in pre_rows)), 1, 0)
    present("supp", "\\label{tab:ckpanels}", "supp")


def dp_exposition():
    section("DP exposition (external reviewer D1-D4) and stale remnants (external reviewer 5.4, critic P0-1 residue)")
    for s_ in ["U_q = \\max(0, \\max_{\\pi \\ne \\hat\\pi} \\min_j C_j(\\pi))",
               "Let $U_a := \\max(0, \\min_{j}",
               "for $\\eps = 0$ the second term is replaced by 1",
               "it need not contain the native grid points or $\\beta/\\eps$ itself",
               "\\textbf{skip} if $c' > B_Q$", "with $V_{s-1}[h][c] > -\\infty$",
               "base $F_S(j, r, 1) = 0$ and $F_S(j, r, 0) = -\\infty$ for $r \\ge 0$",
               "this is pseudo-polynomial in the integer capacity",
               "the search can be exponential in the worst case",
               "The decision is exact on that set up to floating point",
               "the exact column has $\\mathrm{const}_\\infty = 0$",
               "if $\\pi^*_q = \\hat\\pi$, the loss is $0 \\le U_q$",
               "scheme (b) certifies only if its completed search finds no violator, that is $U_q \\le \\eps$",
               "FDC-BF's zero-width rule",
               "regression tests for the weighted centre, the 15-of-15 horizon and the exactness contract were added after the v1 development run",
               "whose guarantee is stronger in that it holds under any predictable sampling rule",
               "lock v10 (S17) certifies implicit classes of up to $2^{64}$ policies",
               "robustness claims in Section 6.6 of the paper",
               "the reads saved in Table~\\ref{tab:cost}",
               "\\label{app:converse}"]:
        present("supp", s_, "supp")
    for s_ in ["implicit classes are untested", "which is stronger.", "(unit tests, before any development run)",
               "contains every direction's own FDC-BF range and the point", "because the grid contains $\\lambda \\ge \\beta/\\eps$"]:
        absent("supp stale", s_, "supp")
    present("tabs", "0 = a constant observed ratio", "tabs")
    for s_ in ["Let $U_q = \\max(0, \\max_{\\pi \\ne \\hat\\pi} \\min_j C_j(\\pi))$",
               "a completed search returns $[U_q \\le \\eps]$ exactly on $\\Lambda$, up to floating point",
               "pseudo-polynomial in the integer budget", "its worst case is exponential",
               "FDC-DP thus evaluates FDC-BF's certificate family on a different $\\lambda$ set, which leaves validity unaffected",
               "which evaluates the same certificate family on one global $\\lambda$ grid per checkpoint",
               "FDC-DP, a knapsack evaluation of the same certificate family on a global $\\lambda$ grid, extends C1 and C2",
               "Under the checkpoint protocol, certificates are issued only at $K = 20$ checkpoints",
               "All four preregistered tests passed",
               "A matched construction comparison isolates aggregation as far as the implementations allow",
               "but sums per-cell radii; its intervals are also clipped to pool bounds and intersected over checkpoints",
               "and on X5 it matched the two registered phased joint certificates within 5\\%",
               "X5 is the only log with a confirmatory non-inferiority test",
               "carries fixed-design moment-generating-function bounds over to sampling without replacement",
               "Beyond Hoeffding's transfer, we combine four pieces",
               "The ledger and the time-uniform step apply standard tools"]:
        present("main", s_, "main")
    for s_ in ["negligible when reads are priced", "changes only", "four pieces are new", "All three preregistered tests passed",
               "computes the same certificate", "as we noticed", "locates the new steps", "Certificates may be issued only at",
               "measures the saving from aggregation directly"]:
        absent("main stale", s_, "main")


def abstract_and_placeholders():
    section("Abstract length, process history, block G filled (r6 part 2)")
    a = MAIN[MAIN.index("\\begin{abstract}") + len("\\begin{abstract}"):MAIN.index("\\end{abstract}")]
    n = len(a.split())
    check("abstract words <= 220", float(n <= 220), 1, 0)
    print(f"     abstract words: {n}")
    for s_ in ["after our first tests", "fourth preregistered test"]:
        absent("abstract process history", s_, "main")
    for s_ in ["\\gPending", "\\color{red}"]:
        absent("no placeholder left", s_, "main")
    check("no block-G placeholder in the supplement", SUPP.count("gPending"), 0, 0)
    present("figure", "\\includegraphics[width=\\columnwidth]{figures/fig5_eps_curve.pdf}", "main")


def block_g():
    section("Block G (post hoc, descriptive): numbers printed in main 6.5 and supplement S19")
    G = J(RES / "full/v10_posthoc_G/summary.json")
    g1 = {k: v for k, v in G["G1"].items() if "-eval-" in k}
    def rat(b, r="FDC/RECT-BF-DP-TU"):
        return b["ratios"][r]["geomean_ratio"]
    def ok_unc(c, b):
        return b["methods"][R]["n80_lt_tau"] >= 0.8 * c["n"]
    fals = sum(m["false_streams"] for c in g1.values() for b in c["by_eps"].values() for m in b["methods"].values())
    check("G1: false streams (all methods, all eps)", fals, 0, 0)
    for d, lo, hi, ulo, uhi in (("x5", 0.08, 0.09, 0.09, 0.18), ("lenta", 0.29, 0.37, 0.34, 0.45)):
        cs = [c for k, c in g1.items() if f"-{d}-" in k]
        last = [rat(c["by_eps"][max(c["by_eps"], key=float)]) for c in cs]
        check(f"G1 {d}: min ratio at largest eps ({lo})", min(last), lo, 0.005)
        check(f"G1 {d}: max ratio at largest eps ({hi})", max(last), hi, 0.005)
        free = [rat(b) for c in cs for b in c["by_eps"].values() if ok_unc(c, b) and not b["methods"][R]["pinned"]]
        check(f"G1 {d}: uncensored unpinned min ({ulo})", min(free), ulo, 0.005)
        check(f"G1 {d}: uncensored unpinned max ({uhi})", max(free), uhi, 0.005)
    ub = max(b["ratios"]["FDC/RECT-BF-DP-TU"]["ci95_two_sided"][1] - rat(b) for c in g1.values() for b in c["by_eps"].values())
    lb = max(rat(b) - b["ratios"]["FDC/RECT-BF-DP-TU"]["ci95_two_sided"][0] for c in g1.values() for b in c["by_eps"].values())
    check("G1: every 95% CI within +-0.01 of its ratio", float(max(ub, lb) <= 0.01), 1, 0)
    ba, vr, bo = [], [], []
    for c in G["G2"].values():
        for b in c["by_eps"].values():
            m = b["methods"]
            ba.append(b["ratios"]["FDC(b)/FDC(a)"]["geomean_ratio"])
            if m["RECT-BF-DP"]["n80_lt_tau"] >= 0.8 * m["RECT-BF-DP"]["n"]:
                vr.append(b["ratios"]["FDC(b)/RECT-BF-DP"]["geomean_ratio"])
            check("G2: no node-limit hit", m["FDC-DP(b)"]["node_limit_hits"], 0, 0)
            check("G2: no false stream (b)", m["FDC-DP(b)"]["false_streams"], 0, 0)
    check("G2: settings (36)", len(ba), 36, 0)
    check("G2: (b)/(a) = 1.000 everywhere", max(abs(x - 1) for x in ba), 0, 0.0005)
    check("G2: vs RECT uncensored min (0.09)", min(vr), 0.09, 0.005)
    check("G2: vs RECT uncensored max (0.67)", max(vr), 0.67, 0.005)
    lock = {"x5-16": 0.191, "x5-32": 0.169, "x5-64": 0.182, "lenta-16": 0.632, "lenta-32": 0.516, "lenta-64": 0.513}
    shift, rep = 0.0, 0.0
    for k, c in G["G3"].items():
        if k == "hc_selection":
            continue
        key = k
        r = c["eval"]["ratios"]
        base = r["FDC/RECT-BF-DP-TU"]["geomean_ratio"]
        rep = max(rep, abs(base - lock[key]))
        for rv in ("FDC/RECT-HG-DP", "FDC/HC-WoR-DP[tuned]"):
            shift = max(shift, abs(r[rv]["geomean_ratio"] - base))
    check("G3: RECT-BF-DP-TU replicates lock v10 within 0.003", float(rep <= 0.0035), 1, 0)
    check("G3: stronger rivals shift ratios by at most 0.035", shift, 0.035, 0.0006)
    for s_ in ["the ratio generally falls as $\\eps$ grows, to 0.08--0.09 on X5 and 0.29--0.37 on Lenta at the largest $\\eps$, and is 0.09--0.18 and 0.34--0.45 where the rectangle passes that criterion and is unpinned",
               "FDC-DP needed 0.09--0.67 of the rectangle's rows wherever it passed that criterion",
               "Branch-and-bound left $N_{80}$ unchanged on every stream of all 36 knapsack settings",
               "moved the lock-cell ratios by about 0.035 at most"]:
        present("main 6.5 block G", s_, "main")
    for s_ in ["0.093--0.183 on X5 and 0.336--0.446 on Lenta", "replicate lock v10 within 0.003", "by about 0.035 at most (0.0353"]:
        present("supp S19", s_, "supp")


# ============================================================================================ r6c (after the r6 reviews)
def table1_panel_a_r6c():
    section("r6c: Table 1A rows under role labels (critic P1-4), every cell recomputed from the lock-v9 analyses")
    A9, B9 = J(RES / "full/v9_analysis_A.json"), J(RES / "full/v9_analysis_B.json")
    rows = [("A", "HC-WoR@D", r"\textbf{HC-WoR} (betting), Criteo", 1e6, 2),
            ("A", "RECT-ck-BF-TU@D", r"\textbf{Matched rectangle}, TU, Criteo", 1e6, 2),
            ("B", "HC-WoR@D", r"\textbf{HC-WoR} (tuned, Neyman), X5", 1e3, 1),
            ("B", "RECT-ck-BF-TU@D", r"\textbf{Matched rectangle}, TU, X5", 1e3, 1),
            ("B", "PJC-local@D", r"\textbf{PJC} (residual, 77-time ledger), X5", 1e3, 1),
            ("B", "TU-PJC@D", r"\textbf{PJC} (residual, TU), X5", 1e3, 1)]
    num = lambda s: [float(x) for x in re.findall(r"\d+\.\d+", s)]  # noqa: E731
    for blk, key, label, unit, nd in rows:
        d = A9 if blk == "A" else B9
        line = [l for l in MAIN.splitlines() if l.startswith(label + " &")]
        check(f"Table 1A row present: {label}", len(line), 1, 0)
        if len(line) != 1:
            continue
        cells = [c.strip() for c in line[0].rstrip("\\ ").split("&")]
        n80 = d["comparisons"]["TU-FDC@D|N80_pen"][key]
        x12 = d["comparisons"]["TU-FDC@D|x12"][key]
        n100 = d["comparisons"]["TU-FDC@D|N100_pen"][key]
        tau = d["describe"]["TU-FDC@D"]["tau_R"]
        check(f"{blk} {key} rows", d["describe"][key]["geomean_N80_pen_over_tau"] * tau / unit, num(cells[3])[0], 0.5 * 10 ** -nd + 1e-9)
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


def _g2_rows(cell):
    import gzip
    base = RES / "full/v10_posthoc_G/rows" / f"{cell}.jsonl"
    if base.exists():
        return [json.loads(l) for l in open(base)]
    gz = base.with_name(base.name + ".gz")
    if gz.exists():
        return [json.loads(l) for l in gzip.open(gz, "rt")]
    return None


def r6c_checks():
    section("r6c: block-G corrections and qualifiers (external reviewer r6 2-4), critic r6 P0-2, P1-1..P1-4, P2-*")
    G = J(RES / "full/v10_posthoc_G/summary.json")
    GS = (RES / "full/v10_posthoc_G/SUMMARY.md").read_text()
    G2T = (IT / "writing/supplement/r6_tables/blockG_g2.tex").read_text()
    G1T = (IT / "writing/supplement/r6_tables/blockG_g1.tex").read_text()
    G3T = (IT / "writing/supplement/r6_tables/blockG_g3.tex").read_text()
    # --- G2 cost range (external reviewer 2: 1-24, not 1-22), from summary.json and the row metadata
    costs = [c for v in G["G2"].values() for c in v["costs"]]
    check("G2 cost range: min (1)", min(costs), 1, 0)
    check("G2 cost range: max (24)", max(costs), 24, 0)
    x64 = G["G2"]["G2-dev-x5-64"]["costs"]
    check("G2: cost 24 at X5 S64 segment index 20", x64[20], 24, 0)
    meta = RES / "full/v10_posthoc_G/rows/G2-dev-x5-64.meta.json"
    if meta.exists():
        check("G2 row metadata: X5 S64 max cost 24", max(J(meta)["costs"]), 24, 0)
    present("supp S19 cost range", "an integer treatment cost from 1 to 24", "supp")
    absent("supp S19 old cost range", "from 1 to 22", "supp")
    ok = "κ ∈ [1, 24]" in GS and "κ ∈ [1, 22]" not in GS
    check("SUMMARY.md cost range [1, 24]", float(ok), 1, 0)
    # --- branch-only denominator: problem instances, 50 streams x 6 eps x 15 problems
    for k, v in G["G2"].items():
        tot = 0
        for e, b in v["by_eps"].items():
            m = b["methods"]["FDC-DP(b)"]
            check(f"{k} eps {e}: a_certs + b_only = 50 x 15", m["a_certs"] + m["b_only_certs"], 750, 0)
            tot += m["a_certs"] + m["b_only_certs"]
        check(f"{k}: problem instances per row (4,500)", tot, 4500, 0)
    present("supp S19 denominator", "0--8 branch-only certification events per row among 4,500 problem instances (50 streams $\\times$ 6 $\\eps$ $\\times$ 15 problems)", "supp")
    ok = "branch-only certification events (problems certified by (b) but not (a)) among 4,500 problem instances" in G2T
    check("G2 table caption: branch-only events among 4,500 problem instances", float(ok), 1, 0)
    check("G2 table caption: development halves only", float("development halves only" in G2T), 1, 0)
    for t, nm in ((SUPP, "supp"), (G2T, "G2 table"), (GS, "SUMMARY.md")):
        check(f"no 'problem-checkpoints' denominator in {nm}", float("problem-checkpoints" not in t), 1, 0)
    # --- every G2 paired endpoint equal under (a) and (b) (1,800 streams), when the rows are shipped
    n_pairs, n_eq, seen = 0, 0, True
    for k in G["G2"]:
        rows = _g2_rows(k)
        if rows is None:
            seen = False
            continue
        a = {(r["seed"], r["eps"]): r["N80_pen"] for r in rows if r["method"] == "FDC-DP(a)"}
        b = {(r["seed"], r["eps"]): r["N80_pen"] for r in rows if r["method"] == "FDC-DP(b)"}
        n_pairs += len(set(a) & set(b))
        n_eq += sum(a[x] == b[x] for x in set(a) & set(b))
    if seen:
        check("G2: paired (a)/(b) endpoints (1,800)", n_pairs, 1800, 0)
        check("G2: every paired endpoint equal", n_eq, 1800, 0)
    else:
        print("  [note] G2 rows not present; per-stream (a)=(b) check skipped (summary-level 1.000 still checked)")
    # --- runtime by S across logs (SUMMARY.md said 0.11 at S16; Lenta has 0.1737)
    mx = {S_: max(v["by_eps"][e]["methods"]["FDC-DP(b)"]["sec_per_ck_max"] for k, v in G["G2"].items() if v["S"] == S_
                  for e in v["by_eps"]) for S_ in (16, 32, 64)}
    for S_, pr in ((16, 0.17), (32, 0.45), (64, 1.8)):
        check(f"G2 max sec per checkpoint at S = {S_} across logs", mx[S_], pr, 0.005 if S_ < 64 else 0.05)
    check("SUMMARY.md runtime 0.17 / 0.45 / 1.8", float("0.17 / 0.45 / 1.8 s per checkpoint" in GS), 1, 0)
    check("SUMMARY.md errata note (r6c)", float("## Errata (paper revision r6c, 2026-10-05)" in GS), 1, 0)
    # --- G1: lock eps = largest ratio among eps passing the 80% pre-horizon criterion, in all six cells
    g1 = {k: v for k, v in G["G1"].items() if "-eval-" in k}
    n_ok, n_lt = 0, {}
    for k, c in g1.items():
        q = [(float(e), b["ratios"]["FDC/RECT-BF-DP-TU"]["geomean_ratio"], b["lock_eps"],
              b["methods"][R]["n80_lt_tau"], b["methods"][R]["pinned"]) for e, b in c["by_eps"].items()]
        passing = [x for x in q if x[3] >= 0.8 * c["n"]]
        lock = [x for x in q if x[2]][0]
        n_ok += max(passing, key=lambda x: x[1])[0] == lock[0]
        n_lt[k] = lock[3]
        if k == "G1-eval-x5-32":
            check("G1 X5 S32: no point passing the criterion and unpinned", len([x for x in passing if not x[4]]), 0, 0)
        ys = [x[1] for x in sorted(q)]
        rises = [(a, b) for a, b in zip(ys, ys[1:]) if b > a]
        check(f"{k}: increases of the ratio along eps (only Lenta S16 has one)", len(rises), 1 if k == "G1-eval-lenta-16" else 0, 0)
    check("G1: lock eps has the largest passing ratio in all six cells", n_ok, 6, 0)
    check("G1: circled X5 S64 streams before tau (183)", n_lt["G1-eval-x5-64"], 183, 0)
    check("G1: circled Lenta S16 streams before tau (192)", n_lt["G1-eval-lenta-16"], 192, 0)
    check("G1: other circled points 200 before tau", float(all(v == 200 for k, v in n_lt.items() if k not in ("G1-eval-x5-64", "G1-eval-lenta-16"))), 1, 0)
    d_rule = J(RES / "pilots/fdc_dp_v2/analysis.json")["eps_rule"]
    sel = {k: v["selected_eps"] for k, v in d_rule.items()}
    check("lock-v10 dev rule: exactly one cell without a qualifying eps", sum(v is None for v in sel.values()), 1, 0)
    for s_ in ["In all six cells the lock-v10 $\\eps$ has the largest point ratio among the tested $\\eps$ at which the rectangle passes the 80\\% pre-horizon criterion",
               "183 of 200 (X5 $S = 64$) and 192 of 200 (Lenta $S = 16$) rectangle streams succeed strictly before $\\tau_R$",
               "at X5 $S = 64$ no development $\\eps$ qualified, so that cell's $\\eps$ is descriptive",
               "X5 $S = 32$ has no such point", "the same 200 streams at every $\\eps$",
               "generally falls as $\\eps$ grows", "this holds on the tested grid only and does not show that pinning is causally irrelevant",
               "Block G addresses three objections", "G2 uses the development halves only",
               "further fresh-stream uses of the X5 and Lenta evaluation halves (X5's fourth)",
               "before any evaluation replay row was computed"]:
        present("supp S19 (r6c)", s_, "supp")
    for s_ in ["on the tested grid the rule-selected $\\eps$ gives the largest ratio among tolerances at which the rectangle passes the 80\\% pre-horizon criterion",
               "A post-hoc block G addresses three objections", "Circled: the lock-v10 $\\eps$ (descriptive at X5 $S = 64$)",
               "the same 200 fresh streams at every $\\eps$"]:
        present("main 6.5 / Figure 2 (r6c)", s_, "main")
    # --- horizon-sensitive captions instead of an asserted censoring bias
    present("Figure 2 caption", "Hollow: fewer than 80\\% of rectangle streams certify before the horizon; these endpoint ratios are horizon-sensitive", "main")
    check("G1 table caption: horizon-sensitive", float("horizon-sensitive endpoint ratios" in G1T), 1, 0)
    check("G1 table caption: same 200 streams at every eps", float("the same 200 streams at every $\\eps$" in G1T), 1, 0)
    for t, nm in ((MAIN, "main"), (SUPP, "supp"), (G1T, "G1 table")):
        for bad in ("overstates FDC-DP", "removes three objections", "not products of the pinned rival",
                    "correctness device", "correctness guarantee rather", "least favourable uncensored point",
                    "understates the width advantage", "understates any advantage"):
            check(f"absent from {nm}: {bad!r}", float(bad not in t), 1, 0)
    # --- G2 development-only and exact-decision refinement
    present("main 6.5 G2", "on development halves (50 streams per $\\eps$) with simulated or proxy heterogeneous costs", "main")
    present("main 6.5 G2", "so it is an exact-decision refinement rather than a source of speed", "main")
    present("supp S19 G2", "branch-and-bound is an exact-decision refinement", "supp")
    present("supp S19 G2", "G2 runs on the development halves (seeds 950--999, 50 streams per $\\eps$)", "supp")
    present("supp S19 G2", "neither cost is a measured campaign cost", "supp")
    # --- G3 CP / TU labels, 'about 0.035', HC not uniformly stronger, Lenta exhaustion of all rivals
    ok = ("RECT-HG-DP CP only" in G3T and "HC-WoR-DP TU" in G3T and "RECT-BF-DP-TU TU on the frozen schedule" in G3T)
    check("G3 table caption: CP/TU labels", float(ok), 1, 0)
    present("supp S19 G3", "is valid at the checkpoints only (CP)", "supp")
    present("supp S19 G3", "HC-WoR-DP is time-uniform (TU)", "supp")
    present("main 6.5 G3", "a checkpoint-valid exact-hypergeometric rectangle, weaker in guarantee than the time-uniform rivals", "main")
    g3 = {k: v for k, v in G["G3"].items() if k != "hc_selection"}
    sh = max(abs(c["eval"]["ratios"][rv]["geomean_ratio"] - c["eval"]["ratios"]["FDC/RECT-BF-DP-TU"]["geomean_ratio"])
             for c in g3.values() for rv in ("FDC/RECT-HG-DP", "FDC/HC-WoR-DP[tuned]"))
    check("G3: largest shift (0.0353)", sh, 0.0353, 5e-5)
    r64 = g3["x5-64"]["eval"]
    check("G3 X5 S64: FDC / tuned HC (0.152)", r64["ratios"]["FDC/HC-WoR-DP[tuned]"]["geomean_ratio"], 0.152, H3)
    check("G3 X5 S64: FDC / matched rectangle (0.182)", r64["ratios"]["FDC/RECT-BF-DP-TU"]["geomean_ratio"], 0.182, H3)
    check("G3 X5 S64: tuned HC before tau on no stream", r64["methods"]["HC-WoR-DP[tuned]"]["n80_lt_tau"], 0, 0)
    present("supp S19 G3", "at X5 $S = 64$ the tuned HC-WoR-DP gives 0.152 against 0.182, because it reaches 12 of 15 before $\\tau_R$ on no stream", "supp")
    ex = [c["eval"]["methods"][m]["exhaustion_at_k80"]["ctrl"] for k, c in g3.items() if k.startswith("lenta")
          for m in ("RECT-BF-DP-TU", "RECT-HG-DP", "HC-WoR-DP[tuned]")]
    check("G3 Lenta: every rival's mean exhausted-control fraction is 1", float(all(x == 1 for x in ex) and len(ex) == 9), 1, 0)
    present("supp S19 G3", "All three rivals have a mean exhausted-control fraction of 1 in every Lenta cell", "supp")
    # --- Sec. 7 block-G disclosure, conclusion scope, P2-1, P1-2, P1-3
    present("main Sec. 7 block G", "In block G (supplement S19), G1 and G3 reuse the X5 (fourth use) and Lenta evaluation halves, G2 used development halves, and a failed first launch logged access to six evaluation cells twice before any replay row was computed. Block G, the width predictions", "main")
    present("main conclusion", "Confirmatory scaling tests used cardinality budgets on X5 and an outcome-exposed Lenta half", "main")
    present("main conclusion", "heterogeneous-cost and robustness tests were descriptive and used development halves", "main")
    present("supp S9", "Heterogeneous knapsack costs were tested only descriptively, on development halves", "supp")
    present("main C2 (P2-1)", "both verdicts are positive, with passing replicas", "main")
    absent("main C2 (P2-1)", "\\emph{positive result achieved} with passing replicas", "main")
    present("main 6.5 (P1-2)", "they are exhaustion-sensitive stopping comparisons that count whole steps of the $K = 20$ grid", "main")
    st = {}
    for b, t in CELLS:
        c = AN[b]["cells"][t]["comparisons"]
        st[t] = max(c[f"{P}/{R}"]["geomean_ratio"], c[f"{P}/HC-WoR-DP"]["geomean_ratio"])
    conf = [st[t] for b, t in CELLS[:5]]
    check("strongest tested rival: X5 S16 is the betting rectangle (0.216)", st["v10a_full_s16"], 0.216, H3)
    check("strongest tested rival: range low (0.17)", min(conf), 0.17, 0.005)
    check("strongest tested rival: range high (0.63)", max(conf), 0.63, 0.005)
    present("main 6.5 (P1-3)", "Against the strongest rival tested in each cell, the untuned betting rectangle at X5 $S = 16$ (0.216) and the matched rectangle elsewhere, the range stays 0.17--0.63", "main")
    # --- P0-2 priced-access workflow, from the lock-v7 / lock-v8 analyses
    v7 = J(RES / "full/v7_analysis_A.json")["describe"]
    v8 = J(RES / "full/v8_analysis_A.json")["describe"]
    hg7, bf7 = (v7[f"{m}@0.001"]["geomean_N80_pen_over_tau"] * 6989799 for m in ("RECT-ck-HG", "FDC-BF"))
    hg8, bf8 = (v8[f"{m}@0.02"]["geomean_N80_pen_over_tau"] * 100393 for m in ("RECT-ck-HG", "FDC-BF"))
    check("workflow: Criteo exact-rectangle records (1,414,382)", hg7, 1414382, 0.5)
    check("workflow: Criteo FDC-BF records (909,078)", bf7, 909078, 0.5)
    check("workflow: Criteo saving (505,304)", hg7 - bf7, 505304, 0.5)
    check("workflow: Criteo saving share (36%)", 100 * (1 - bf7 / hg7), 36, 0.5)
    check("workflow: dollars per certification (252.65)", (hg7 - bf7) * 0.5 / 1000, 252.65, 0.005)
    check("workflow: 40 certifications a month (10,106)", 40 * round((hg7 - bf7) * 0.5 / 1000, 2), 10106, 0.5)
    check("workflow: X5 saving (42,708 of 67,098)", hg8 - bf8, 42708, 0.5)
    check("workflow: X5 rectangle records (67,098)", hg8, 67098, 0.5)
    check("workflow: X5 share (64%)", 100 * (1 - bf8 / hg8), 64, 0.5)
    check("workflow: X5 dollars (21.35)", (hg8 - bf8) * 0.5 / 1000, 21.35, 0.005)
    for s_ in ["\\textbf{A priced-access workflow.}", "The advertiser uploads (segment, arm) exposure keys",
               "the partner matches planned keys only until $N_{80}$, so no other key is matched or billed",
               # r6d (critic r6c W3): "bill" became "per-record matching charge" and the price basis got its own
               # sentence; the new wording is checked in r6d_checks()
               "Ten Criteo-sized campaigns re-certified four times a month save about \\$10k a month",
               "the saving goes to whichever collaborator pays for matching",
               "These figures price geometric-mean reads on the tested tables, not a measured bill"]:
        present("main 6.2 workflow (P0-2)", s_, "main")
    for s_ in ["1,414,382 matched records for the exact rectangle against 909,078 for FDC-BF on Criteo, a saving of 505,304 records or \\$252.65",
               "$40 \\times \\$252.65 \\approx \\$10{,}106$ a month", "saves 42,708 of 67,098 records (64\\%), \\$21.35 per certification"]:
        present("supp S16 workflow", s_, "supp")
    # --- P1-4 method-name budget in the body (Sections 1-9 with their tables and figures)
    body = MAIN[MAIN.index("\\begin{abstract}"):MAIN.index("\\bibliographystyle")]
    names = set(re.findall(r"(?<![\w-])((?:TU-)?(?:FDC|RECT|HC|PJC)[A-Za-z0-9\-+*]*)", body))
    print(f"     method names in the body: {sorted(names)}")
    check("body method names (FDC-BF, TU-FDC, FDC-DP, HC-WoR, RECT-ck-HG, RECT-ck-BF, PJC) <= 8", float(len(names) <= 8), 1, 0)
    for bad in ("TU-FDC-DP", "RECT-BF-DP-TU", "HC-WoR-DP", "RECT-ck-BF-TU", "PJC-local", "PJC-menu", "TU-PJC", "HC-WoR*", "RECT-ck-HG*"):
        check(f"body free of {bad}", float(bad not in body), 1, 0)
    # --- DP exposition (external reviewer 3) and cross-references
    for s_ in ["\\If{$F_i(\\cdot, B_q - u, h) = -\\infty$} \\textbf{continue} \\Comment{no feasible completion: discard}",
               "$s \\gets i+1$; push children $(s, u{+}\\kappa[s,a]",
               "is discarded before its bound is formed, so the sum below never meets $+\\infty + (-\\infty)$",
               "and, for $\\eps > 0$, has a point at least $\\beta/\\eps$",
               "dividing every cost by the greatest common divisor $g$ of the positive costs and replacing each budget $B$ by $\\lfloor B/g \\rfloor$",
               "FDC-DP evaluates the same certificate family as FDC-BF, on one global $\\lambda$ grid per checkpoint",
               "Table~\\ref{tab:ckpanels} in S1 holds the checkpoint-test rows",
               "the post-hoc block G with its tolerance sweep, knapsack costs and stronger rivals (S19)",
               "(checkpoint tests: Table 1B of the paper and Table~\\ref{tab:ckpanels})"]:
        present("supp DP / cross-references (r6c)", s_, "supp")
    absent("supp stale cross-reference", "Table~\\ref{tab:ckpanels} in S5", "supp")
    absent("supp stale (i)", "FDC-DP's certificate is FDC-BF's.", "supp")
    absent("supp tool name", "Co" + "dex", "supp")  # split so that the artifact scrub leaves the needle intact
    # --- legacy checkpoint API: guarded entry point and its contract (external reviewer 5.4)
    code = IT / "exp/code/dsswm"
    gm = code / "baselines/fdc_bet_guarded.py"
    check("guarded entry point exists", float(gm.exists()), 1, 0)
    if gm.exists():
        g = gm.read_text()
        check("guarded module: off-grid rejection directs to FDCTimeUniform",
              float("OffGridCertifyError" in g and "FDCTimeUniform" in g), 1, 0)
    check("guarded entry point test exists", float((code / "tests/test_fdc_bet_guarded.py").exists()), 1, 0)
    present("supp S4 usage contract", "arbitrary increasing times do not authorise fresh widths", "supp")
    present("supp S4 usage contract", "\\texttt{dsswm/baselines/fdc\\_bet\\_guarded.py}", "supp")
    # --- errata: v10_summary.md note (append-only) indexed in S14; SUMMARY.md corrected
    vs = (RES / "full/v10_summary.md").read_text()
    mark = "\n---\n**Note added 2026-10-05 (paper revision r6c; append-only, nothing above is edited).**"
    check("v10_summary.md: r6c note appended at the end", float(mark in vs and vs.rstrip().endswith("is indexed in the errata (S14).")), 1, 0)
    import hashlib
    pre = vs[:vs.index(mark)] if mark in vs else vs
    check("v10_summary.md: text above the note unchanged (sha256 of the pre-r6c file)",
          float(hashlib.sha256(pre.encode()).hexdigest() in V10_SUMMARY_PRE_SHA), 1, 0)
    present("supp S14", "\\texttt{exp/results/full/v10\\_summary.md} (lock-v10 result summary)", "supp")
    present("supp S14", "A dated note pointing to S17 was appended to the file in revision r6c", "supp")
    present("supp S14", "the G2 treatment costs range from 1 to 24, not 1 to 22", "supp")


def _lines(p):
    """Lines of a .jsonl file, or of its .jsonl.gz copy (anonymous package); [] if neither exists."""
    import gzip
    if p.exists():
        return p.read_text().splitlines()
    gz = p.with_name(p.name + ".gz")
    return gzip.open(gz, "rt").read().splitlines() if gz.exists() else []


def _exists(p):
    return p.exists() or p.with_name(p.name + ".gz").exists()


def _rows(p, meth):
    d = {}
    if True:
        for line in _lines(p):
            r = json.loads(line)
            if r.get("method") == meth and not r.get("error"):
                d[(r["eps"], r["seed"])] = r
    return d


def r6d_checks():
    """r6d: supplement S20 (localised union ledgers, development only) recomputed from the v11 pilot rows, and the
    pricing basis of the priced-access workflow (critic r6c W3, W4; external reviewer r6c sections 6-7)."""
    section("r6d: S20 localised ledgers (v11 dev rows) and the pricing basis (critic r6c W3/W4, external reviewer r6c 6-7)")
    pw = RES / "pilots/fdc_pw"
    base = RES / "full/v10_posthoc_G/rows"
    variants = {"FDC-PW": ("rows_v1_twocentre", "TU-FDC-PW(b)"), "FDC-LS": ("rows", "TU-FDC-LS(a)")}
    cells = ("x5-16", "x5-32", "x5-64", "lenta-16", "lenta-32", "lenta-64")
    have_rows = all(_exists(pw / sub / f"G1-dev-{c}.jsonl") for sub, _ in variants.values() for c in cells) and \
        all(_exists(base / f"G1-dev-{c}.jsonl") for c in cells)
    if have_rows:
        comp, stopped, n_false, part_g = {}, {}, 0, {}
        for name, (sub, meth) in variants.items():
            for cell in cells:
                v = _rows(pw / sub / f"G1-dev-{cell}.jsonl", meth)
                b = _rows(base / f"G1-dev-{cell}.jsonl", "TU-FDC-DP(b)")
                n_false += sum(r["n_false"] for r in v.values())
                for e in sorted({e for e, _ in v}):
                    prs = [(v[k], b[k]) for k in v if k[0] == e and k in b]
                    g = math.exp(sum(math.log(x["N80_pen"] / y["N80_pen"]) for x, y in prs) / len(prs))
                    if len(prs) == 50:
                        comp[(name, cell, e)] = g
                    else:
                        stopped[(name, cell, e)] = len(prs)
                        part_g[(name, cell)] = g
        check("S20: no false certificate in any v11 row", n_false, 0, 0)
        for name, lo, hi in (("FDC-PW", 1.000, 1.086), ("FDC-LS", 1.000, 1.046)):
            g = [x for (n, _, _), x in comp.items() if n == name]
            check(f"S20 {name}: complete-cell ratio min ({lo})", min(g), lo, 5e-4)
            check(f"S20 {name}: complete-cell ratio max ({hi})", max(g), hi, 5e-4)
        check("S20: overall 1.00 to 1.09 (min)", min(comp.values()), 1.00, 0.005)
        check("S20: overall 1.00 to 1.09 (max)", max(comp.values()), 1.09, 0.005)
        check("S20: no complete geometric mean below 1", float(min(comp.values()) >= 1.0), 1, 0)
        ncomp = Counter((n, c) for (n, c, _) in comp)
        for (n, c), k in {("FDC-PW", "x5-16"): 6, ("FDC-PW", "x5-32"): 6, ("FDC-PW", "lenta-16"): 6,
                          ("FDC-PW", "x5-64"): 2, ("FDC-PW", "lenta-32"): 1, ("FDC-PW", "lenta-64"): 1,
                          ("FDC-LS", "x5-16"): 6, ("FDC-LS", "x5-32"): 6, ("FDC-LS", "lenta-16"): 6,
                          ("FDC-LS", "lenta-32"): 6, ("FDC-LS", "x5-64"): 0, ("FDC-LS", "lenta-64"): 0}.items():
            check(f"S20 coverage: {n} {c} complete eps", ncomp.get((n, c), 0), k, 0)
        st = {(n, c): k for (n, c, e), k in stopped.items()}
        for (n, c), k in {("FDC-PW", "x5-64"): 33, ("FDC-PW", "lenta-32"): 14, ("FDC-PW", "lenta-64"): 46,
                          ("FDC-LS", "x5-64"): 3, ("FDC-LS", "lenta-64"): 3}.items():
            check(f"S20 stopped: {n} {c} streams", st.get((n, c), -1), k, 0)
        check("S20: exactly five stopped (variant, cell, eps) runs", len(stopped), 5, 0)
        for c, eps0 in (("x5-64", 0.02), ("lenta-64", 0.003)):
            check(f"S20 FDC-LS {c}: the 3 stopped streams have ratio 1.000", part_g[("FDC-LS", c)], 1.0, 5e-4)
            check(f"S20 FDC-LS {c}: stopped at the smallest eps",
                  float([e for (n, cc, e) in stopped if n == "FDC-LS" and cc == c] == [eps0]), 1, 0)
        tab = (IT / "writing/supplement/r6_tables/s20_ratios.tex").read_text()
        for n in variants:
            for c in ("x5-16", "x5-32", "lenta-16", "lenta-32"):
                gs = [g for (nn, cc, _), g in comp.items() if nn == n and cc == c]
                if len(gs) == 6:
                    lab = (f"{'X5' if c.startswith('x5') else 'Lenta'}, {c.split('-')[1]} & {n} & 6 of 6 & "
                           f"{min(gs):.3f}--{max(gs):.3f}")
                    check(f"S20 table row {n} {c} recomputed", float(lab in tab), 1, 0)
        r = json.loads(_lines(pw / "rows/G1-dev-x5-64.jsonl")[0])
        check("S20: beta_1 at S = 64 (15.01)", r["ls_beta_1"], 15.01, 0.005)
        check("S20: far shells pay beta_J + ln 2 (53.03)", r["ls_beta_far"], 53.03, 0.005)
        check("S20: far-shell exponent minus beta_J = ln 2", r["ls_beta_far"] - r["beta_J"], math.log(2), 1e-9)
        check("S20: beta_J at S = 64 (52.34)", r["beta_J"], 52.34, 0.005)
    else:
        print("  [note] v11 pilot rows or block-G1 development rows not present; S20 ratio checks skipped")
    check("S20: Gaussian E|xi| / sigma (0.80)", math.sqrt(2 / math.pi), 0.80, 0.005)
    check("S20: Gaussian E(xi)_+ / sigma (0.40)", 1 / math.sqrt(2 * math.pi), 0.40, 0.005)
    sj = pw / "s20_summary.json"
    if sj.exists() and "diag" in J(sj):
        dg = J(sj)["diag"]  # written by gen_r6d_s20.py --diag, which needs the development halves
        for c, hl, hh, gl, gh in (("x5-64", 10, 34, 0.004, 0.014), ("lenta-64", 6, 32, 0.001, 0.005)):
            check(f"S20 diag {c}: Hamming min ({hl})", min(dg[c]["hamming"]), hl, 0)
            check(f"S20 diag {c}: Hamming max ({hh})", max(dg[c]["hamming"]), hh, 0)
            check(f"S20 diag {c}: value gap min ({gl})", min(dg[c]["value_gap"]), gl, 5e-4)
            check(f"S20 diag {c}: value gap max ({gh})", max(dg[c]["value_gap"]), gh, 5e-4)
    else:
        print("  [note] s20_summary.json without diag; Hamming / value-gap checks skipped")
    tol = J(RES / "full/v10_posthoc_G/summary.json")["G1"]
    for d, lo, hi in (("x5", 0.02, 0.08), ("lenta", 0.003, 0.012)):
        es = sorted(float(e) for k, v in tol.items() if f"-dev-{d}-" in k for e in v["by_eps"])
        check(f"S20 tolerance range {d} low ({lo})", es[0], lo, 1e-9)
        check(f"S20 tolerance range {d} high ({hi})", es[-1], hi, 1e-9)
    for s_ in ["\\section{Localised union ledgers (development only, negative)}",
               "On every complete cell the localised ledgers read 1.00 to 1.09 times FDC-DP's rows, and no geometric mean fell below 1",
               "FDC-PW gave 1.000--1.086 and FDC-LS 1.000--1.046",
               "its runs at the next $\\eps$ were stopped after 33, 14 and 46 streams",
               "was stopped at $S = 64$ after 3 streams at the smallest $\\eps$ on each log, with ratio 1.000 on all of them",
               "Its nearest shell has $\\beta_1 = 15.01$ and its far shells pay $\\beta_J + \\ln 2 = 53.03$, against $\\beta_J = 52.34$ at $S = 64$",
               "10--34 segments on X5 and 6--32 on Lenta, with value gaps of 0.004--0.014 and 0.001--0.005 against tolerances of 0.02--0.08 and 0.003--0.012",
               "($0.80\\sigma$ against $0.40\\sigma$)",
               "It used development halves only, read no evaluation row and involved no lock",
               "It is not a lower bound", "These are tested constructions, not formal results of the paper",
               "the localised union ledgers tried on development halves (S20, negative)"]:
        present("supp S20", s_, "supp")
    absent("supp S20: no universal impossibility claim", "cannot remove the price", "supp")
    present("main 6.5 pointer to S20",
            "Three localised union ledgers read no fewer rows on development halves (supplement S20), so FDC-DP pays this cost in full", "main")
    # --- pricing basis (critic r6c W3): per-record share, one-time fee per collaboration, whole-bill shares
    v7 = J(RES / "full/v7_analysis_A.json")["describe"]
    v8 = J(RES / "full/v8_analysis_A.json")["describe"]
    hg7, bf7 = (v7[f"{m}@0.001"]["geomean_N80_pen_over_tau"] * 6989799 for m in ("RECT-ck-HG", "FDC-BF"))
    hg8, bf8 = (v8[f"{m}@0.02"]["geomean_N80_pen_over_tau"] * 100393 for m in ("RECT-ck-HG", "FDC-BF"))
    usd = lambda n: n * 0.5 / 1000  # noqa: E731
    check("pricing: Criteo rectangle per-record charge (707.19)", usd(hg7), 707.19, 0.005)
    check("pricing: Criteo FDC-BF per-record charge (454.54)", usd(bf7), 454.54, 0.005)
    check("pricing: X5 rectangle per-record charge (33.55)", usd(hg8), 33.55, 0.005)
    check("pricing: X5 FDC-BF per-record charge (12.20)", usd(bf8), 12.20, 0.005)
    check("pricing: Criteo bill with one $100 fee (807.19)", usd(hg7) + 100, 807.19, 0.005)
    check("pricing: X5 bill with one $100 fee (133.55)", usd(hg8) + 100, 133.55, 0.005)
    check("pricing: Criteo whole-bill share with the fee (31%)", 100 * (usd(hg7) - usd(bf7)) / (usd(hg7) + 100), 31, 0.5)
    check("pricing: X5 whole-bill share with the fee (16%)", 100 * (usd(hg8) - usd(bf8)) / (usd(hg8) + 100), 16, 0.5)
    check("pricing: per-record share Criteo (36%)", 100 * (1 - bf7 / hg7), 36, 0.5)
    check("pricing: per-record share X5 (64%)", 100 * (1 - bf8 / hg8), 64, 0.5)
    for s_ in ["rule-based matching in AWS Entity Resolution on AWS Clean Rooms at \\$0.50 per 1,000 records matched plus a one-time \\$100 matching fee per collaboration",
               "the 36\\% and 64\\% are shares of the per-record matching charge (\\$707.19 against \\$454.54 on Criteo, \\$33.55 against \\$12.20 on X5)",
               "the saving would still be 31\\% of the whole Criteo bill (\\$252.65 of \\$807.19) and 16\\% of the X5 bill (\\$21.35 of \\$133.55)",
               "The certifier needs only per-cell counts and outcome sums of the matched records at each checkpoint",
               "cited in the paper's Section 6.2", "Rule-based record matching, \\$0.50 per 1,000 &"]:
        present("supp S16 pricing basis (r6d)", s_, "supp")
    absent("supp S16: old base-fee wording", "plus a \\$100 base fee", "supp")
    absent("supp S16: two pricing pages", "AWS Clean Rooms and Entity Resolution pricing pages", "supp")
    for s_ in ["spares 36\\% of the exact rectangle's per-record matching charge on Criteo (505k records, \\$253) and 64\\% on X5",
               "The price is AWS Entity Resolution rule-based matching, \\$0.50 per 1,000 records matched, whose \\$100 fee is one-time per collaboration~\\citep{awscleanrooms2026pricing}",
               "The certifier queries only per-cell counts and sums"]:
        present("main 6.2 pricing basis (r6d)", s_, "main")
    bib = (IT / "writing/latex_acm/references.bib").read_text()
    check("bib: pricing entry names the line item and the fee basis",
          float("rule-based matching \\$0.50 per 1,000 records matched, one-time \\$100 matching fee per collaboration" in bib), 1, 0)
    # --- small consistency points (external reviewer r6c section 7, critic W4)
    fig = (IT / "writing/scripts/make_fig_epscurve.py").read_text()
    check("Figure 2 legend: 'before tau_R'", float("80% before" in fig and "80% by" not in fig), 1, 0)
    absent("supp speed figure caption: lower-bound reading", "(value is a lower bound)", "supp")
    present("supp speed figure caption", "so the value is a horizon-sensitive endpoint ratio", "supp")
    GS = (RES / "full/v10_posthoc_G/SUMMARY.md").read_text()
    check("SUMMARY.md: no 'uncensored' shorthand outside the errata note",
          float("uncensored" not in GS.split("## Errata")[0]), 1, 0)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r5b_chain(args)
    rival_pinning()
    union_trend_and_bnb()
    exhaustion_and_rule()
    pilot_comparisons()
    table1_summary()
    dp_exposition()
    abstract_and_placeholders()
    block_g()
    table1_panel_a_r6c()
    r6c_checks()
    r6d_checks()
    print(f"\nr6: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
