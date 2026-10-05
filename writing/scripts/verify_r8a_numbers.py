"""Check paper revision r8a (critic r7 W1 bounded fix, W3/W4 re-check) against the r7b chain (read-only).

r8a changes wording only in main.tex; no number, verdict, table or supplement text changes.
1. Runs verify_r7b_numbers.py (which runs verify_r7 and the whole r6 -> ... -> r3 chain, the lock-v11 recomputation and
   the page check). Every check must still pass, except those listed in SUPERSEDED: their wording was deliberately
   changed in r8a to pay for the C1 foregrounding (critic r7 W1) and the Sec. 8 OPE clause (critic r7 W3c). Each listed
   item has a replacement checked here, and every cut fact is checked to remain verbatim in the supplement.
2. Checks the new C1 text, the Sec. 8 clause, the shortened block-G sentences and the convergence rules on the edited
   paragraphs (claim-first, no closing number, <= 45 words per sentence).
3. Repeats the page-budget check explicitly: 12 pages, conclusion on p. 8, references heading on p. 9 and no body text
   above it.
Usage (workspace root): .venv/bin/python3 writing/scripts/verify_r8a_numbers.py [--skip-census]
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IT = HERE.parents[1]
MAIN = (IT / "writing/latex_acm/main.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
FAIL = []

# verify_r7b output lines (after "[MISMATCH] ") deliberately changed in r8a -> replacement checked below.
SUPERSEDED = {
    "r7 chain: r6 chain: chain: T3 grid identity: 'Evaluated only on the 20-point grid, this procedure is FDC-BF itself'":
        "main: 'this procedure is FDC-BF;' (one word cut for space)",
    "r7 chain: r6 chain: main: 'but sums per-cell radii; its intervals are also clipped":
        "main: 'its intervals are clipped to pool bounds and intersected over checkpoints' (one word cut)",
    "r7 chain: r6 chain: main 6.5 block G: 'Branch-and-bound left $N_{80}$ unchanged":
        "main: 'Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per cell'; full sentence in S22 and S19",
    "r7 chain: r6 chain: main 6.5 G2: 'on development halves (50 streams per":
        "main: 'on development halves with simulated or proxy heterogeneous costs'; 50 streams per eps in S19 and S22",
    "r7 chain: main Sec. 7 / 6.6 disclosure: 'The fresh-log part of C2 therefore covers":
        "main: 'The fresh-log part of C2 thus covers the named rectangles on this one log'",
    "r7 chain: main shortened replacement: 'Branch-and-bound left $N_{80}$ unchanged":
        "main: 'Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per cell'",
    "r7 chain: Open Bandit Dataset cited in main (intro and 6.6): computed 3, paper 2":
        "third citation added in Sec. 8 for the OPE pedigree (critic r7 W3c); count 3 checked here",
    "main r7b: 'Its core is the exact-variance Bennett lemma (Lemma~\\\\ref{lem:bennett}).'":
        "C1 extended (critic r7 W1): lemma + Proposition 4.3 + 'union paid in full, repaid by m_eff'",
}


def ok(label, cond, detail=""):
    print(f"  [{'OK' if cond else 'MISMATCH'}] {label}{(': ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(label)


def present(label, needle, txt, where):
    ok(label, needle in txt, f"{needle[:110]!r} printed in {where}")


def absent(label, needle, txt, where):
    ok(label, needle not in txt, f"{needle[:110]!r} absent from {where}")


def section(t):
    print(f"\n== {t}")


def r7b_chain(args):
    section("r7b chain (verify_r7b -> r7 -> r6 -> ... -> r3); only listed checks may fail")
    p = subprocess.run([sys.executable, str(HERE / "verify_r7b_numbers.py"), *args], capture_output=True, text=True)
    out = p.stdout.splitlines()
    n_ok = sum("[OK]" in l for l in out)
    n_sup_inner = sum("[SUPERSEDED]" in l for l in out)
    used = set()
    for l in out:
        if "[MISMATCH]" not in l:
            continue
        key = l.strip().replace("[MISMATCH] ", "", 1)
        hit = [k for k in SUPERSEDED if key.startswith(k)]
        if hit:
            used.add(hit[0])
            print(f"  [SUPERSEDED] {key[:100]} -> {SUPERSEDED[hit[0]]}")
        else:
            print(f"  [MISMATCH] r7b chain: {key}")
            FAIL.append(f"r7b chain: {key[:80]}")
    for k in SUPERSEDED:
        if k not in used:
            print(f"  [note] superseded item not failing (string still present?): {k[:100]}")
    last = [l for l in out if l.startswith("r7b:")]
    print(f"  chain: {n_ok} OK lines; {n_sup_inner} inherited r7b supersessions; {len(used)} r8a supersessions; "
          f"r7b final line: {last[-1][:40] if last else '(none)'}")
    if not last:
        FAIL.append("r7b chain produced no final line")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])
        FAIL.append("chain stderr")


def replacements():
    section("r8a replacements for the superseded checks")
    for s_ in ["Evaluated only on the 20-point grid, this procedure is FDC-BF;",
               "but sums per-cell radii; its intervals are clipped to pool bounds and intersected over checkpoints",
               "Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per cell, so it is an exact-decision refinement rather than a source of speed",
               "Second, on development halves with simulated or proxy heterogeneous costs, FDC-DP needed 0.09--0.67 of the rectangle's rows wherever it passed that criterion",
               "and the ratio generally falls as $\\eps$ grows (Figure S3 in S19)",
               "The fresh-log part of C2 thus covers the named rectangles on this one log"]:
        present("main r8a", s_, MAIN, "main")
    ok("Open Bandit Dataset cited in main (intro, 6.6, Sec. 8)", MAIN.count("\\citep{saito2021open}") == 3,
       f"computed {MAIN.count(chr(92) + 'citep{saito2021open}')}, expected 3")
    section("Facts cut from Sec. 6.5 remain verbatim in the supplement (S19 / S22)")
    for s_ in ["Branch-and-bound left $N_{80}$ unchanged on every stream of all 36 knapsack settings",
               "it changed $N_{80}$ on 0--1 of 200 streams per cell",
               "on development halves (50 streams per $\\eps$) with simulated or proxy heterogeneous costs",
               "to 0.08--0.09 on X5 and 0.29--0.37 on Lenta at the largest $\\eps$",
               "G2 runs on the development halves (seeds 950--999, 50 streams per $\\eps$)"]:
        present("supp keeps", s_, SUPP, "supp")
    for s_ in ["0--1 of 200 streams per lock-v10 cell", "all 36 knapsack settings", "0.08--0.09 on X5"]:
        absent("main 6.5 trimmed", s_, MAIN, "main")


C1_NEW = ("Its core is the exact-variance Bennett lemma (Lemma~\\ref{lem:bennett}), a finite-population moment bound for each frozen cell. "
          "By Proposition~\\ref{prop:width} the frontier union is then paid in full and repaid, to leading order, once the effective "
          "number of cells $\\meff$ exceeds its cost, as Open Bandit's 0.817 again bears out.")


def w1_w3():
    section("Critic r7 W1 (C1 foregrounding) and W3c (Sec. 8 OPE clause)")
    present("C1 W1", C1_NEW, MAIN, "main")
    c1 = MAIN[MAIN.index("\\item[\\textbf{C1}]"):MAIN.index("\\item[\\textbf{C2}]")]
    ok("C1 still ends on its falsifier", "\\emph{Falsified by} a flaw in the proofs" in c1 and c1.rstrip().endswith("only gross violations."))
    # 0.817 is the registered v11 headline already printed in the abstract, Sec. 6.6 and the conclusion
    ok("C1 cites only a number already in the paper (0.817)", MAIN.count("0.817") >= 4, f"0.817 occurs {MAIN.count('0.817')} times")
    present("6.5 keeps the union-cost evidence", "so FDC-DP pays this cost in full", MAIN, "main")
    present("6.5 keeps the union-cost growth", "from $\\beta_J/\\beta_C = 1.47$ at $S = 9$ to 4.50 at $S = 64$", MAIN, "main")
    present("Sec. 8 W3c", "and benchmarks estimators on public logs such as the Open Bandit Dataset~\\citep{saito2021open}, which we replay", MAIN, "main")
    present("Sec. 8 closing sentence", "FDC-BF is the frontier-wide, finite-population counterpart of these tools.", MAIN, "main")
    present("6.6 keeps 'never replayed in this work' (W3c)", "an off-policy-evaluation benchmark never replayed in this work", MAIN, "main")
    absent("W3a", "last eighth", MAIN, "main")
    absent("W3b old wording", "not an artifact of reused tables", MAIN, "main")
    present("W3b", "so the saving survives outside the reused tables", MAIN, "main")


def rules():
    section("Convergence rules on the r8a paragraphs")
    sents = [s for s in re.split(r"(?<=[.;])\s+", C1_NEW) if s]
    ok("C1 new sentences <= 45 words", max(len(s.split()) for s in sents) <= 45, f"max {max(len(s.split()) for s in sents)}")
    for start in ("A post-hoc block G addresses", "\\textbf{The fresh-log test.}", "\\textbf{Finite populations and certified selection.}"):
        para = [x for x in MAIN.split("\n\n") if x.strip().startswith(start)][0].strip()
        ok(f"paragraph '{start[:30]}' ends on words", not re.search(r"(\d\)?|\})\.$", para))
        body = re.sub(r"~?\\cite[pt]?\{[^}]*\}", "", para)
        lens = [len(s.split()) for s in re.split(r"(?<=[.;])\s+", body) if s.strip()]
        ok(f"paragraph '{start[:30]}' sentences <= 45 words", max(lens) <= 45, f"max {max(lens)}")
    for hedge in ("may potentially", "might possibly", "could potentially"):
        absent("no hedging stack", hedge, MAIN, "main")


def page_budget():
    section("Page budget (main.pdf: 12 pages, body ends on page 8, references begin on page 9)")
    try:
        import pymupdf
    except ImportError:
        print("  [note] pymupdf not installed; page check skipped")
        return
    d = pymupdf.open(str(IT / "writing/latex_acm/main.pdf"))
    ref = concl = None
    for i, p in enumerate(d):
        for b in p.get_text("blocks"):
            s = b[4].replace("\n", " ").strip()
            if s.startswith("REFERENCES") and ref is None:
                ref = (i + 1, b[1])
            if "read plan first." in s:
                concl = (i + 1, b[3])
    ok("main.pdf has 12 pages", len(d) == 12, f"{len(d)}")
    ok("conclusion ends on page 8", bool(concl) and concl[0] == 8, f"p. {concl[0]} y = {concl[1]:.0f}" if concl else "not found")
    ok("references heading on page 9", bool(ref) and ref[0] == 9, f"p. {ref[0]} y = {ref[1]:.0f}" if ref else "not found")
    above = [b for b in d[8].get_text("blocks") if 40 < b[0] < 570 and 75 < b[1] < (ref[1] if ref else 0) - 1 and not b[4].strip().isdigit()]
    ok("no body text above the references heading on page 9", not above)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r7b_chain(args)
    replacements()
    w1_w3()
    rules()
    page_budget()
    print(f"\nr8a: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
