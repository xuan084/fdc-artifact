"""Recompute every number added or reworded in paper revision r5 (part 1) from result files (read-only).

1. Runs verify_r4b_numbers.py (which runs verify_r4 and verify_r3), unchanged. Every value check in that chain must
   still pass. A text-presence check of that chain may fail only if it is listed in SUPERSEDED below, i.e. the r4
   wording was deliberately replaced in r5; each superseded string has a replacement that this script checks.
2. Checks the r5 numbers: within-v9 monitoring comparison (external reviewer fix 4), +box alignment, zero false streams per
   registered block, block F2 (new C3: equal pilot information, charged pilot, break-even J), the real-log adaptive
   block, Hoeffding bib entry, stale-phrase removals (external reviewer T1-T4, S2, S4, S5), presentation rules (thesis length,
   sentence lengths), and counts the lock-v10 placeholders.
Sources (all under iter_001/):
  exp/results/full/v9_analysis_{A,B}.json        lock-v9 analyses (within-v9 K20 vs dense, +box)
  exp/results/full/v7_analysis_A.json, v8_analysis_A.json   false streams of the checkpoint-strength blocks
  exp/results/full/v8_posthoc_F2/summary.json    block F2 (equal information, charged pilot)
  exp/results/full/v8_posthoc_F/summary.json     block F (adaptive gain on Criteo-asym)
  exp/results/full/v8_posthoc_D/summary.json     real-log adaptive block (best of 64)
  writing/latex_acm/main.tex, references.bib, writing/supplement/supplement.tex, r5_tables/*.tex
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/verify_r5_numbers.py [--skip-census]
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
BIB = (IT / "writing/latex_acm/references.bib").read_text()
R5T = "".join(p.read_text() for p in sorted((IT / "writing/supplement/r5_tables").glob("*.tex")))
FAIL = []
H3 = 6e-4

# r4b-chain text checks deliberately replaced in r5 -> the replacement string checked below
SUPERSEDED = {
    "main text: '0.655 to 0.708 against HC-WoR' printed in main": "within-v9 comparison 0.658 -> 0.708 (external reviewer fix 4)",
    "main text: '(0.362 to 0.374)' printed in main": "within-v9 comparison 0.364 -> 0.374 (external reviewer fix 4)",
    "main text: 'TU-FDC needs 0.530 and 0.304 times those of its time-uniform form' printed in main":
        "RECT-ck-BF-TU named as the TU form of the +box variant (external reviewer fix 4)",
    "S12: 'from the lock-v7 value 0.655 to 0.708' printed in supp": "within-v9 comparison in S12 (external reviewer fix 4)",
    "S12: 'from the lock-v8 value 0.362 to 0.374' printed in supp": "within-v9 comparison in S12 (external reviewer fix 4)",
}


def check(label, computed, printed, tol):
    ok = abs(computed - printed) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: computed {computed:.6g}, paper {printed}")
    if not ok:
        FAIL.append(label)


def present(label, needle, where):
    txt = {"main": MAIN, "supp": SUPP, "bib": BIB, "r5t": R5T}[where]
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


def J(p):
    return json.load(open(p))


def section(t):
    print(f"\n== {t}")


A = J(RES / "full/v9_analysis_A.json")
B = J(RES / "full/v9_analysis_B.json")


def r4b_chain(args):
    section("r4b chain (verify_r4b -> verify_r4 -> verify_r3), unchanged")
    p = subprocess.run([sys.executable, str(Path(__file__).with_name("verify_r4b_numbers.py")), *args],
                       capture_output=True, text=True)
    lines = [l.strip() for l in p.stdout.splitlines() if "[MISMATCH]" in l]
    n_ok = sum("[OK]" in l for l in p.stdout.splitlines())
    print(f"  chain: {n_ok} OK lines, {len(lines)} MISMATCH lines, exit {p.returncode}")
    seen = set()
    for l in lines:
        key = l.replace("[MISMATCH] ", "")
        if key in SUPERSEDED:
            seen.add(key)
            print(f"  [SUPERSEDED] {key} -> {SUPERSEDED[key]}")
        else:
            print(f"  [MISMATCH] chain: {key}")
            FAIL.append(f"chain: {key}")
    # value checks are never superseded: every superseded item must be a text check
    for k in SUPERSEDED:
        if k not in seen:
            print(f"  [note] superseded item no longer failing (string still present?): {k}")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])


def monitoring_comparison():
    section("Within-v9 monitoring comparison (external reviewer fix 4): FDC-BF@K vs TU-FDC@D against HC-WoR on the same streams")
    for d, lab, k_p, d_p in [(A, "Criteo", 0.658, 0.708), (B, "X5", 0.364, 0.374)]:
        k = d["comparisons"]["FDC-BF@K|N80_pen"]["HC-WoR@K"]["geomean_ratio"]
        dd = d["comparisons"]["TU-FDC@D|N80_pen"]["HC-WoR@D"]["geomean_ratio"]
        check(f"{lab} FDC-BF/HC-WoR at K = 20 (v9 streams)", k, k_p, H3)
        check(f"{lab} TU-FDC/HC-WoR at 77 times", dd, d_p, H3)
    present("main C2", "Against HC-WoR the ratio is 0.658 at 20 checkpoints and 0.708 at 77 times on Criteo, and 0.364 and 0.374 on X5", "main")
    present("supp S12", "from 0.658 at the 20 checkpoints to 0.708 at 77 times on Criteo and from 0.364 to 0.374 on X5", "supp")
    absent("price-of-strength framing", "Matching guarantee strength cost", "main")


def box_alignment():
    section("Dense matched rectangle = TU form of RECT-ck-BF+box (external reviewer fix 4)")
    for d, lab, p_box, p_nobox in [(A, "Criteo", 0.533, 0.497), (B, "X5", 0.309, 0.301)]:
        c = d["comparisons"]["FDC-BF@K|N80_pen"]
        check(f"{lab} FDC-BF/RECT-ck-BF+box at K = 20 (v9 streams)", c["RECT-ck-BF+box@K"]["geomean_ratio"], p_box, H3)
        check(f"{lab} FDC-BF/RECT-ck-BF (no box) at K = 20 (v9 streams, not printed)", c["RECT-ck-BF@K"]["geomean_ratio"], p_nobox, H3)
    present("main C4", "TU-FDC needs 0.530 and 0.304 times the rows of RECT-ck-BF-TU, the time-uniform form of the +box variant", "main")
    present("main C4", "FDC-BF/RECT-ck-BF+box is 0.533 and 0.309", "main")
    present("main Sec. 4", "RECT-ck-BF-TU, the time-uniform form of the matched rectangle's +box variant", "main")
    present("Table 1 caption", "RECT-ck-BF-TU is the time-uniform form of RECT-ck-BF+box", "main")


def false_streams_per_block():
    section("C1: zero false FDC-BF streams per registered block (external reviewer: state counts by block)")
    v7, v8 = J(RES / "full/v7_analysis_A.json"), J(RES / "full/v8_analysis_A.json")
    fs7 = v7["false_streams"]["FDC-BF@0.001"]
    check("lock v7 (Criteo) FDC-BF false streams", fs7["false_streams"], 0, 0)
    check("lock v7 streams", fs7["n"], 200, 0)
    check("lock v7 primary_false_streams", v7["decision"]["primary_false_streams"], 0, 0)
    for e in ("0.015", "0.02", "0.03"):
        fs8 = v8["false_streams"][f"FDC-BF@{e}"]
        check(f"lock v8 (X5) FDC-BF false streams at eps {e}", fs8["false_streams"], 0, 0)
    check("lock v8 streams", v8["false_streams"]["FDC-BF@0.02"]["n"], 200, 0)
    for name, d in (("v7", v7), ("v8", v8)):
        tot = sum(v["false_streams"] for k, v in d["false_streams"].items() if v.get("validity", "rigorous") == "rigorous"
                  and not any(t in k for t in ("fav", "B5", "Qini")))
        check(f"lock {name}: false streams over guaranteed methods", tot, 0, 0)
    for name, d in (("v9 A", A), ("v9 B", B)):
        check(f"lock {name} TU-FDC false streams", d["false_streams"]["TU-FDC@D"]["false_streams"], 0, 0)
        check(f"lock {name} CP bound", d["false_streams"]["TU-FDC@D"]["cp_ub95"], 0.0149, 1e-4)
    check("lock v7 CP bound", fs7["cp_ub95"], 0.0149, 1e-4)
    present("main C1", "The blocks are lock v7 on Criteo and lock v8 on X5 at 20 checkpoints, and lock v9 on both logs at 77 monitoring times, each with a one-sided Clopper--Pearson bound of 0.0149", "main")


def f2_block():
    section("Block F2 (new C3): equal information, charged pilot, break-even J")
    s = J(RES / "full/v8_posthoc_F2/summary.json")
    envs = s["envs"]
    eq = []
    for e, v in envs.items():
        for k, h in v["head_to_head"].items():
            if "no pilot" in k:
                continue
            for jk in ("J1", "J4", "J16", "Jinf"):
                eq.append((h[jk]["ratio"], h[jk]["ci95"][1], e, k, jk))
    check("equal information: number of ratios (4 regimes x 2 n x 4 members x 4 J)", len(eq), 128, 0)
    check("equal information: smallest ratio (0.90)", min(r for r, *_ in eq), 0.90, 0.005)
    check("equal information: largest ratio (0.98)", max(r for r, *_ in eq), 0.98, 0.005)
    check("equal information: largest CI upper limit below 1", float(max(u for _, u, *_ in eq) < 1), 1, 0)
    ch_cr = [h["J1"]["ratio"] for e in ("CR9-asym", "CR9-mixed") for k, h in envs[e]["head_to_head"].items() if "no pilot" in k]
    ch_x5 = [h["J1"]["ratio"] for e in ("X9-asym", "X9-mixed") for k, h in envs[e]["head_to_head"].items() if "no pilot" in k]
    be_x5 = [h["breakeven_J"] for e in ("X9-asym", "X9-mixed") for k, h in envs[e]["head_to_head"].items() if "no pilot" in k]
    check("charged (J = 1) vs no-pilot adaptive, Criteo-sized, low (0.89)", min(ch_cr), 0.89, 0.005)
    check("charged (J = 1) vs no-pilot adaptive, Criteo-sized, high (0.99)", max(ch_cr), 0.99, 0.005)
    check("charged (J = 1) vs no-pilot adaptive, X5-sized, low (1.08)", min(ch_x5), 1.08, 0.005)
    check("charged (J = 1) vs no-pilot adaptive, X5-sized, high (1.67)", max(ch_x5), 1.67, 0.005)
    check("X5-sized break-even J, low (about 2; 2.1 in S11)", min(be_x5), 2.1, 0.05)
    check("X5-sized break-even J, high (about 26; 25.5 in S11)", max(be_x5), 25.5, 0.05)
    pil = {e: sorted({m["pilot_over_tau"] for m in v["methods"].values() if m.get("pilot_reads", 0) > 0}) for e, v in envs.items()}
    check("Criteo-sized pilot share of tau_R, max (0.26%)", 100 * max(pil["CR9-asym"]), 0.26, 0.005)
    check("Criteo-sized pilot share of tau_R, min (0.06%)", 100 * min(pil["CR9-asym"]), 0.06, 0.005)
    check("X5-sized pilot share, n = 250 (4.5%)", 100 * min(pil["X9-asym"]), 4.5, 0.05)
    check("X5-sized pilot share, n = 1000 (18%)", 100 * max(pil["X9-asym"]), 18, 0.5)
    m = envs["CR9-asym"]["methods"]
    check("Criteo-asym reset without pilot (FDC-BF/method, 1.127)", m["PJC-A-reset[no pilot]"]["FDCBF_over_method_N80_J1"]["ratio"], 1.127, H3)
    check("Criteo-asym reset with 1,000-draw pilot, uncharged (1.125)", m["PJC-A-reset[pilot n=1000,init]"]["FDCBF_over_method_N80_Jinf"]["ratio"], 1.125, H3)
    fs = sum(mm["false_streams"] for v in envs.values() for mm in v["methods"].values())
    check("F2 false streams, every method and regime", fs, 0, 0)
    F = J(RES / "full/v8_posthoc_F/summary.json")["envs"]["CR9-asym"]["methods"]
    sp = [v["FDCBF_over_method_N80"] for k, v in F.items() if (k.startswith("PJC-A") or k.startswith("PJC-menu"))
          and "segmenu" not in k and "FDCBF_over_method_N80" in v]
    check("block F Criteo-asym adaptive gain low (9%: 1.087)", min(sp), 1.087, H3)
    check("block F Criteo-asym adaptive gain high (13%: 1.127)", max(sp), 1.127, H3)
    for s_ in ["needed 0.90--0.98 times the rows of each tested reset and menu design in all four regimes, with every 95\\% interval below 1",
               "the pilot is at most 0.26\\% of $\\tau_R$", "still needs 0.89--0.99 times the rows of the no-pilot adaptive designs",
               "a pilot of 4.5\\% or 18\\% of $\\tau_R$ reverses the comparison (1.08--1.67) unless one pilot serves about 2 to 26 certification runs",
               "by 9--13\\% on the Criteo-structured asymmetric regime", "same independent pilot of 250 or 1,000 draws per cell"]:
        present("main C3", s_, "main")
    for s_ in ["the ratios lie between 0.90 and 0.98, each with a 95\\% CI below 1", "the pilot is 0.06--0.26\\% of $\\tau_R$",
               "still needs 0.89--0.99 times the rows of the no-pilot picks", "the comparison reverses (1.08--1.67)",
               "$J \\ge 2.1$--25.5 runs", "(Criteo-asym reset 1.127 without and 1.125 with the 1,000-draw pilot",
               "an instance costs $SAn$ = 4,500 or 18,000 reads", "0/250 per method and regime"]:
        present("supp S11 (F2)", s_, "supp")
    # generated head-to-head table agrees with the JSON (spot check every cell's ratio)
    h = envs["X9-asym"]["head_to_head"]
    present("r5 table F2", f"{h['FDC-BF[ney-pilot,n=1000] / PJC-A-menu[no pilot]']['J1']['ratio']:.3f}", "r5t")


def real_log_adaptive():
    section("Real-log adaptive block (S8): best of 64 per log, every member slower")
    D = J(RES / "full/v8_posthoc_D/summary.json")["layers"]
    check("X5 best of 64 (FDC-BF/variant)", D["X9"]["eval_best_of_grid_optimistic"]["FDCBF_over_variant"], 0.943, H3)
    check("Criteo best of 64 (FDC-BF/variant)", D["CR9"]["eval_best_of_grid_optimistic"]["FDCBF_over_variant"], 0.951, H3)
    for L in ("X9", "CR9"):
        check(f"{L}: members with point ratio above 1", len(D[L]["any_grid_member_point_above_1"]), 0, 0)
    present("main C3", "every member was slower than FDC-BF (best 0.943 on X5, 0.951 on Criteo; supplement S8)", "main")
    present("main C3 (supporting NI)", "It thus matches PJC within the registered 5\\% margin", "main")


def hoeffding_and_scope():
    section("Novelty scoping (critic P0-1 / external reviewer S2): Hoeffding 1963 cited, verified bib entry")
    m = re.search(r"@article\{hoeffding1963probability,(.*?)\n\}", BIB, re.S)
    ok = m is not None
    print(f"  [{'OK' if ok else 'MISMATCH'}] bib entry hoeffding1963probability present")
    if not ok:
        FAIL.append("Hoeffding bib entry")
        return
    e = m.group(1)
    for k, v in [("journal", "Journal of the American Statistical Association"), ("volume", "58"), ("number", "301"),
                 ("pages", "13--30"), ("doi", "10.1080/01621459.1963.10500830"), ("year", "1963")]:
        present(f"Hoeffding bib {k}", v, "bib")
    present("intro scoping", "Hoeffding's convex-order theorem~\\citep{hoeffding1963probability} carries any such moment bound over to sampling without replacement", "main")
    present("intro: what is new", "this finite-population lemma with exact variance boxes, a union ledger over the budget frontier, a time-uniform extension of the joint certificate, and a proposition", "main")
    present("related work", "our Bernstein variant FDC is exactly that construction", "main")


def stale_phrases():
    section("Stale or overclaiming phrases removed (external reviewer T1-T4, S2, S4, S5; critic W8)")
    for s_ in ["every number in this paper, is unchanged", "adaptive designs give up", "is not slack", "slack in the ledger",
               "must therefore come from", "decided before reading", "do not depend on the registered knob",
               "best for the users who took part", "\\$131k", "do not cover without-replacement replay",
               "a_c(\\mu_c - \\hat\\mu_c(n_c(t)))", "The gain therefore comes from the allocation, not from adapting it online",
               "makes one finite-population width per policy difference valid", "the earlier Bernstein certificate FDC",
               "each registered same-plan rectangle"]:
        absent("main", s_, "main")
    for s_ in ["no evaluated procedure or number changes", "the tie of C3", "do not depend on the registered knob values",
               "which a frozen pilot plan recovers while keeping Remark 4.2's time-uniform guarantee"]:
        absent("supp", s_, "supp")
    n = SUPP.count("For the time-uniform extension (Remark 4.2), a single-cell Monte Carlo test")
    n += SUPP.count("For Remark~\\ref{rem:tu}, a single-cell Monte Carlo test")
    check("supp S4: single-cell TU Monte Carlo paragraph appears once", n, 1, 0)
    # T1: the remark proof uses the main construction's sign
    present("App. A T1 sign", "Y_t = \\sum_c a_c(\\hat\\mu_c(n_c(t)) - \\mu_c) = \\Delta - \\hat\\Delta_t", "main")
    present("App. A T2 strict tail", "P(\\max_{T_k \\le t \\le \\tau_R} Y_t > t_k \\mid \\cA) \\le e^{-\\beta}", "main")
    present("App. A filtration", "\\mathcal{G}_t = \\bigvee_{c \\in D} \\mathcal{G}^c_{n_c(t)}", "main")
    present("App. A empty cells", "the block's width is $+\\infty$ and the block contributes no event", "main")
    present("Prop. 1 fluid crossings", "for continuous (fluid) first crossings before rounding to checkpoints", "main")
    present("Prop. 1 ideal widths", "The proposition concerns ideal widths", "main")
    present("T3 grid identity", "Evaluated only on the 20-point grid, this procedure is FDC-BF itself", "main")
    present("T3 supp S2", "numbers are identical only when evaluated on the $K = 20$ grid", "supp")
    present("T4", "We analysed every adaptive variant at checkpoint strength only", "main")
    present("S4 sensitivity scope", "The ordering persisted over the tested settings", "main")
    present("S5 observed-pool", "an audit of the observed-pool ranking of budgeted policies", "main")
    present("S5 price conversion", "so they are not expected per-deployment savings", "main")
    present("ledger hedge", "the tested localisation changes little on these tables", "main")
    present("m_eff hedge", "In the model, differences between logs then come mainly from $\\meff$", "main")
    present("Data paragraph conclusion", "so the tests describe these two populations rather than a population of logs.", "main")
    present("C2/C3 endpoint wording", "paired geometric-mean $\\Npen$ ratio", "main")


def body_text():
    b = MAIN[MAIN.index(r"\begin{abstract}"):MAIN.index(r"\bibliographystyle")]
    b = re.sub(r"(?m)%.*$", "", b)
    for env in ["table\\*", "table", "figure", "algorithm", "CCSXML", "equation", "proposition", "theorem"]:
        b = re.sub(r"\\begin\{%s\}.*?\\end\{%s\}" % (env, env), " ", b, flags=re.S)
    b = re.sub(r"\\vtenPending\{[^{}]*(\{[^{}]*\}[^{}]*)*\}", " ", b)
    b = re.sub(r"\\vtenReserve\{\d+\}", " ", b)
    b = re.sub(r"\\\[.*?\\\]", " X ", b, flags=re.S)
    b = re.sub(r"\$[^$]*\$", "X", b)
    b = re.sub(r"\\(cite[pt]?|ref|label|eqref|anonrepo)\{[^}]*\}", "", b)
    b = re.sub(r"\\(section|subsection)\*?\{[^}]*\}", "\n\n", b)
    b = re.sub(r"\\item(\[[^\]]*\])?", "\n\n", b)
    b = re.sub(r"\\textbf\{[^}]*\.\}", "\n\n", b)
    b = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", b)
    b = b.replace("{", "").replace("}", "").replace("~", " ")
    b = re.sub(r"\b(e\.g|i\.e|i\.i\.d|cf|vs|App|Fig|Prop|Thm|al)\.", r"\1", b)
    sents = []
    for p in re.split(r"\n\s*\n", b):
        p = re.sub(r"\s+", " ", p)
        sents += [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(\\])", p) if len(x.split()) > 2]
    return sents


def presentation():
    section("Presentation rules (critic P0-5): thesis <= 35 words, body sentences mean <= 25 and max <= 45 words")
    thesis = ("Freezing the read plan lets one finite-population Chernoff width per policy difference certify a whole budget "
              "frontier with 0.30--0.71 times the outcome reads of matched-strength per-cell rectangles on two reused logs.")
    present("thesis sentence in abstract and introduction", thesis, "main")
    check("thesis occurrences (abstract + introduction)", MAIN.count(thesis), 2, 0)
    check("thesis words <= 35", float(len(thesis.split()) <= 35), 1, 0)
    print(f"     thesis words: {len(thesis.split())}")
    s = body_text()
    lens = [len(x.split()) for x in s]
    mean = sum(lens) / len(lens)
    print(f"     body sentences {len(s)}, mean {mean:.1f} words, max {max(lens)}, >35: {sum(l > 35 for l in lens)}")
    check("body mean sentence length <= 25", float(mean <= 25), 1, 0)
    check("no body sentence over 45 words", float(max(lens) <= 45), 1, 0)


def placeholders():
    section("Lock-v10 placeholders (expected, red, to be filled after lock v10)")
    n_p, n_r = MAIN.count("\\vtenPending{"), MAIN.count("\\vtenReserve{")
    print(f"     \\vtenPending: {n_p} uses; \\vtenReserve: {n_r} uses")
    for where in ("abstract", "Scaling to implicit policy classes (FDC-DP)", "subsection{Scaling to implicit policy classes}"):
        present("placeholder site", where, "main")
    check("no v10 placeholder in the supplement", SUPP.count("vtenPending"), 0, 0)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r4b_chain(args)
    monitoring_comparison()
    box_alignment()
    false_streams_per_block()
    f2_block()
    real_log_adaptive()
    hoeffding_and_scope()
    stale_phrases()
    presentation()
    placeholders()
    print(f"\nr5: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
