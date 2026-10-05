"""Check paper revision r8b (wording fixes from the r8 reviews) against the r8 chain (read-only).

r8b changes wording only: no number, verdict, table or locked file changes.
1. Runs verify_r8_numbers.py (which runs verify_r8a, verify_r7b and the whole chain down to verify_r3, the lock-v11 and
   lock-v12 recomputations and the page check). Every check must still pass, except those listed in SUPERSEDED: their
   wording was deliberately changed in r8b (codex/paper_r8_review.md P2 items and results-note residues;
   writing/review_r8_critic.md W3 and W5). Each listed item has a replacement checked here.
2. Rebuilds main.tex and supplement.tex from their *_pre_r8b copies by applying exactly the r8b replacements and
   requires byte equality, so no other text (and no number) changed; checks every new string and the absence of every
   replaced one.
3. Recomputes the facts behind the new wording from the sealed lock-v12 rows and the analyses: the eps tolerances as a
   share of the development base rate (15.5%, 19.9%; lock v11 8.6%), no false certificate over the whole run, no
   exhausted pool at N80, and which runs reach the full-table horizon at 15 of 15 (RECT-BF-DP-TU on 3 women streams,
   tuned HC-WoR-DP on 9 women and 1 men stream).
4. Checks the dated corrections in exp/results/full/v12_summary.md (old lines struck, new lines present, nothing else
   changed), the convergence rules on the edited paragraphs, and the page budget (pymupdf).
Usage (workspace root): .venv/bin/python3 writing/scripts/verify_r8b_numbers.py [--skip-census]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IT = HERE.parents[1]
RES = IT / "exp/results"
LA = IT / "writing/latex_acm"
MAIN = (LA / "main.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
FAIL = []

# verify_r8 output lines (after "[MISMATCH] ") deliberately changed in r8b -> replacement checked below.
SUPERSEDED = {
    "r8a chain: main r8a: 'Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per cell":
        "main: '... on at most 1 of 200 streams per lock-v10 cell' (codex r8 P2: the 200-stream population is lock v10's)",
    "r8a chain: C1 W1: 'Its core is the exact-variance Bennett lemma":
        "C1: 'bears out' -> 'is consistent with it' (critic r8 W5); 'then' cut for space",
    "main r8: 'These departures (S2; errata in S14) bound how C1--C4 may be read.'":
        "Sec. 7: 'Its departures (S2; errata to locked files in S14) bound how to read the claims.' (critic r8 section 4b)",
    "v12 dev eps-optimal shares: 'where 0.661 and 0.858 of feasible development policies are $\\\\eps$-optimal (gate 0.5)'":
        "'(gate 0.5)' dropped to pay for the tolerance sentence (critic r8 W3); shares still printed, gate 0.5 in S23",
    "no false / no exhausted: 'no method made a false certificate or exhausted a pool'":
        "'No method made a false certificate; none had an exhausted pool at $N_{80}$.' (codex r8 P2)",
    "descriptive, not confirmatory, not pooled: \"This is replication evidence, not a second confirmatory test, and is not pooled with lock v11's ratio.\"":
        "'not comparable with, or pooled with, lock v11's' + 'This is replication evidence, not a second confirmatory test.'",
    "gate fixed before data: 'A gate fixed before any of their data were read made both blocks descriptive'":
        "'A gate fixed before any of their outcomes were read made both blocks descriptive' (critic r8 W3)",
    "claims C1-C4 unchanged from r8a":
        "only C1's last sentence changed (checked here); C2-C4 compared with main_pre_r8b",
}

# Exact r8b replacements (old, new). main_pre_r8b + these == main.tex.
MAIN_REPL = [
    ("By Proposition~\\ref{prop:width} the frontier union is then paid in full and repaid, to leading order, once the effective "
     "number of cells $\\meff$ exceeds its cost, as Open Bandit's 0.817 again bears out.",
     "By Proposition~\\ref{prop:width} the frontier union is paid in full and repaid, to leading order, once the effective "
     "number of cells $\\meff$ exceeds its cost, and Open Bandit's 0.817 is consistent with it."),
    ("Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per cell, so",
     "Branch-and-bound changed $N_{80}$ on at most 1 of 200 streams per lock-v10 cell, so"),
    ("A gate fixed before any of their data were read made both blocks descriptive",
     "A gate fixed before any of their outcomes were read made both blocks descriptive"),
    ("where 0.661 and 0.858 of feasible development policies are $\\eps$-optimal (gate 0.5).",
     "where 0.661 and 0.858 of feasible development policies are $\\eps$-optimal. At 15.5\\% and 19.9\\% of the base rate "
     "(8.6\\% in v11) the frontiers are near-trivial, so the ratios are not comparable with, or pooled with, lock v11's."),
    ("faster on every stream, and no method made a false certificate or exhausted a pool. This is replication evidence, "
     "not a second confirmatory test, and is not pooled with lock v11's ratio.",
     "faster on every stream. No method made a false certificate; none had an exhausted pool at $N_{80}$. This is "
     "replication evidence, not a second confirmatory test."),
    ("The confirmatory tests are clean, but the path to them is not. These departures (S2; errata in S14) bound how C1--C4 may be read.",
     "The confirmatory tests are clean; the path is not. Its departures (S2; errata to locked files in S14) bound how to read the claims."),
    ("Every ingredient of FDC-BF has a precedent, and naming them locates what we add.",
     "Each ingredient of FDC-BF has a precedent; we say what is new."),
]
SUPP_REPL = [
    ("Both blocks are descriptive by a gate fixed in the plan before any of their data were read",
     "Both blocks are descriptive by a gate fixed in the plan before any of their outcomes were read"),
    ("The plan's block-status gate, fixed before any data of either campaign were read,",
     "The plan's block-status gate, fixed before any outcome of either campaign was read,"),
]
SUMM_REPL = [
    ("Each eval half was read once.",
     "~~Each eval half was read once.~~ Each eval half had one registered evaluation block, followed by replica and analysis [r8b]."),
    ("The gate fixed in the plan before any data was read makes",
     "The gate fixed in the plan before any ~~data was~~ outcome was [r8b] read makes"),
    ("**The ordering replicates on two independent campaigns.**",
     "**The ordering replicates on ~~two independent campaigns~~ two further campaigns [r8b].**"),
    ("- Every rival stopped strictly before the end of the table (share N80 < τ_R = 1.0), and no method exhausted any pool.",
     "- ~~Every rival stopped strictly before the end of the table (share N80 < τ_R = 1.0), and no method exhausted any pool.~~ "
     "Every rival reached N80 strictly before the end of the table (share N80 < τ_R = 1.0), and no method had an exhausted pool at N80 [r8b]."),
    ("- These readings meet the lock-v10 pre-exhaustion condition (both fractions zero) in both campaigns. Unlike v11, they also have no horizon penalty.",
     "- ~~These readings meet the lock-v10 pre-exhaustion condition (both fractions zero) in both campaigns. Unlike v11, they also have no horizon penalty.~~ "
     "At N80 these readings meet the lock-v10 pre-exhaustion condition (both fractions zero) in both campaigns, and, unlike v11, no N80 reading has a horizon penalty [r8b]."),
]


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


def r8_chain(args):
    section("r8 chain (verify_r8 -> r8a -> r7b -> ... -> r3); only listed checks may fail")
    p = subprocess.run([sys.executable, str(HERE / "verify_r8_numbers.py"), *args], capture_output=True, text=True)
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
            print(f"  [MISMATCH] r8 chain: {key}")
            FAIL.append(f"r8 chain: {key[:80]}")
    for k in SUPERSEDED:
        if k not in used:
            print(f"  [note] superseded item not failing (string still present?): {k[:100]}")
    last = [l for l in out if l.startswith("r8:")]
    print(f"  chain: {n_ok} OK lines; {n_sup_inner} inherited supersessions; {len(used)} r8b supersessions; "
          f"r8 final line: {last[-1][:40] if last else '(none)'}")
    if not last:
        FAIL.append("r8 chain produced no final line")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])
        FAIL.append("chain stderr")


def rebuild(label, pre_path, now, repl):
    pre = Path(pre_path).read_text()
    t = pre
    for a, b in repl:
        ok(f"{label}: old string occurs once in the pre-r8b copy", pre.count(a) == 1, f"{a[:80]!r}")
        t = t.replace(a, b)
    ok(f"{label}: pre-r8b copy + r8b replacements == current file (no other change)", t == now)
    for a, b in repl:
        present(f"{label} new", b, now, label)
        if a not in b:
            absent(f"{label} old", a, now, label)
    return pre


def texts():
    section("Wording-only rebuild of main.tex, supplement.tex and the v12 results note")
    pre = rebuild("main", LA / "main_pre_r8b.tex", MAIN, MAIN_REPL)
    rebuild("supp", IT / "writing/supplement/supplement_pre_r8b.tex", SUPP, SUPP_REPL)
    rebuild("v12_summary", RES / "full/v12_summary_pre_r8b.md",
            (RES / "full/v12_summary.md").read_text().split("\n\n## Corrections (r8 reviews")[0].rstrip("\n") + "\n", SUMM_REPL)
    num = re.compile(r"\d+(?:\.\d+)?")
    def nums(rs, k):
        return set().union(*(set(num.findall(r[k])) for r in rs))
    # per file: numbers in all replaced strings that appear in none of their replacements, and vice versa
    gone_m, add_m = nums(MAIN_REPL, 0) - nums(MAIN_REPL, 1), nums(MAIN_REPL, 1) - nums(MAIN_REPL, 0)
    ok("main: numbers removed = {0.5 (gate limit, still in S23), 4 (the 'C1--C4' label, now 'the claims')}",
       gone_m == {"0.5", "4"}, f"{sorted(gone_m)}")
    ok("main: numbers added are only 15.5, 19.9, 8.6 (tolerance shares) and 10 (lock-v10)",
       add_m <= {"15.5", "19.9", "8.6", "10"}, f"{sorted(add_m)}")
    ok("supp: no number removed or added", nums(SUPP_REPL, 0) == nums(SUPP_REPL, 1))
    ok("v12_summary: no number removed; only the '[r8b]' tags add one",
       nums(SUMM_REPL, 0) <= nums(SUMM_REPL, 1) and nums(SUMM_REPL, 1) - nums(SUMM_REPL, 0) <= {"8"})
    present("gate limit 0.5 kept in S23", "at most half of the feasible policies are $\\eps$-optimal on average", SUPP, "S23")
    for n_ in ("15.5\\%", "19.9\\%", "8.6\\%"):
        present(f"{n_} already printed in S23 / Sec. 6.6", n_, SUPP + MAIN, "supp or main")
    section("Absence of the replaced wording")
    for s_ in ["bears out", "(gate 0.5)", "exhausted a pool", "data were read", "errata in S14", "streams per cell,",
               "independent campaigns", "naming them locates"]:
        absent("main", s_, MAIN, "main")
    for s_ in ["before any of their data were read", "before any data of either campaign were read"]:
        absent("supp", s_, SUPP, "supp")
    summ = (RES / "full/v12_summary.md").read_text()
    for s_ in ["Each eval half was read once.", "two independent campaigns", "no method exhausted any pool.",
               "they also have no horizon penalty."]:
        live = re.sub(r"~~.*?~~", "", summ)
        absent("v12_summary (outside struck text)", s_, live, "v12_summary")
    grab = lambda t, a, b: t[t.index(a):t.index(b)]
    ok("abstract unchanged from r8", grab(MAIN, "\\begin{abstract}", "\\end{abstract}") == grab(pre, "\\begin{abstract}", "\\end{abstract}"))
    ok("claims C2-C4 unchanged from r8", grab(MAIN, "\\item[\\textbf{C2}]", "\\noindent Our contributions") == grab(pre, "\\item[\\textbf{C2}]", "\\noindent Our contributions"))
    ok("conclusion unchanged from r8", grab(MAIN, "\\section{Limitations and Conclusion}", "\\bibliographystyle") == grab(pre, "\\section{Limitations and Conclusion}", "\\bibliographystyle"))
    present("S23 keeps the N80-qualified exhaustion statement", "and no method had an exhausted pool at $N_{80}$", SUPP, "S23")
    present("S23 keeps 'near-trivial'", "a speed-up describes how fast a near-trivial frontier is certified", SUPP, "S23")


def facts():
    section("Facts behind the new wording (recomputed from sealed rows and analyses)")
    v11 = json.loads((RES / "full/v11_obd/v11_analysis.json").read_text())
    ok("lock v11 eps = 8.6% of the development base rate", f"{100 * v11['eps_units']['relative_to_dev_base']:.1f}" == "8.6",
       f"{100 * v11['eps_units']['relative_to_dev_base']:.3f}%")
    horizon = {}
    for c, pct in (("women", "15.5"), ("men", "19.9")):
        an = json.loads((RES / f"full/v12_obd/v12_{c}_analysis.json").read_text())
        rel = 100 * an["eps_units"]["relative_to_dev_base"]
        ok(f"{c} eps = {pct}% of the development base rate", f"{rel:.1f}" == pct, f"{rel:.3f}%")
        rows = [json.loads(l) for l in (RES / f"full/v12_obd/v12_{c}_full/results.jsonl").read_text().splitlines() if l.strip()]
        ok(f"{c}: 800 sealed rows", len(rows) == 800, f"{len(rows)}")
        ok(f"{c}: no false certificate over the whole run, any method", sum(r["n_false"] for r in rows) == 0
           and not any(r["fwer_event"] for r in rows))
        ok(f"{c}: no exhausted pool at N80, any method", all(max(r["exhaustion_at_k80"].values()) == 0 for r in rows))
        ok(f"{c}: every method reaches N80 before tau_R", all(r["n80_lt_tau"] for r in rows))
        h = {}
        for r in rows:
            if r["N_stop_pen"] >= r["tau_R"]:
                h.setdefault(r["method"], []).append(r["seed"])
        horizon[c] = h
    ok("full-table horizon at 15/15: RECT-BF-DP-TU on women seeds 39242, 39271, 39331",
       sorted(horizon["women"].get("RECT-BF-DP-TU", [])) == [39242, 39271, 39331])
    ok("full-table horizon at 15/15: tuned HC-WoR-DP on 9 women and 1 men stream",
       len(horizon["women"].get("HC-WoR-DP[tuned]", [])) == 9 and len(horizon["men"].get("HC-WoR-DP[tuned]", [])) == 1)
    ok("no other method reaches the horizon", set(horizon["women"]) | set(horizon["men"]) <= {"RECT-BF-DP-TU", "HC-WoR-DP[tuned]"})
    summ = (RES / "full/v12_summary.md").read_text()
    corr = summ.split("## Corrections (r8 reviews, added 2026-10-05 in paper revision r8b)")
    ok("v12_summary has the dated r8b corrections section", len(corr) == 2)
    corr = corr[-1]
    for s_ in ["RECT-BF-DP-TU on 3 women streams (seeds 39242, 39271, 39331), tuned HC-WoR-DP on 9 women streams and 1 men stream",
               "so they are two further campaigns, not statistically independent samples",
               "Each evaluation half had one registered evaluation block (the main run), followed by the prespecified ten-seed replica and the analysis",
               "No method had an exhausted pool at N80", "The no-false-certificate statement covers the whole run",
               "The gate was fixed before any outcome of either campaign was read"]:
        present("v12_summary correction", s_, corr, "v12_summary")


def rules():
    section("Convergence rules on the r8b paragraphs")
    for start in ("Lock v12 replicated", "The confirmatory tests are clean;", "Each ingredient of FDC-BF", "\\item[\\textbf{C1}]"):
        para = [x for x in MAIN.split("\n") if x.strip().startswith(start)]
        ok(f"one paragraph starts '{start[:30]}'", len(para) == 1)
        para = para[0].strip() if para else ""
        ok(f"paragraph '{start[:30]}' ends on words", bool(para) and not re.search(r"(\d\)?|\})\.$", para))
        body = re.sub(r"~?\\cite[pt]?\{[^}]*\}", "", para)
        lens = [len(s.split()) for s in re.split(r"(?<=[.;])\s+", body) if s.strip()]
        ok(f"paragraph '{start[:30]}' sentences <= 45 words", bool(lens) and max(lens) <= 45, f"max {max(lens) if lens else 0}")
    for hedge in ("may potentially", "might possibly", "could potentially"):
        absent("no hedging stack", hedge, MAIN, "main")


def page_budget():
    section("Page budget (main.pdf: 12 pages, body ends on page 8, references begin on page 9)")
    try:
        import pymupdf
    except ImportError:
        print("  [note] pymupdf not installed; page check skipped")
        return
    d = pymupdf.open(str(LA / "main.pdf"))
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
    txt = " ".join(p.get_text() for p in d).replace("\n", " ")
    for s_ in ("is consistent with it", "none had an exhausted pool at", "near-trivial", "errata to locked files"):
        ok(f"pdf compiled from this main.tex ('{s_}')", s_ in txt)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r8_chain(args)
    texts()
    facts()
    rules()
    page_budget()
    print(f"\nr8b: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
