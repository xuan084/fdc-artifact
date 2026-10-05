"""Check paper revision r7b (bounded repairs after the r7 reviews) against the sealed lock-v11 rows (read-only).

r7b corrects the lock-v11 reporting and narrows several scope sentences; it changes no verdict and no registered number.
1. Runs verify_r7_numbers.py (which runs the whole r6 -> r5b -> r5 -> r4b -> r4 -> r3 chain). Every value check must
   still pass. A check may fail only if it is listed in SUPERSEDED below: its wording was deliberately corrected in
   r7b, and each listed item has a replacement that this script checks.
   Note: verify_r7's check "last eighth of the table: both N80/tau > 7/8 - one step" passes numerically but tested the
   wrong quantity (a geometric mean, with a loosened bound for FDC-DP); it is replaced by per-stream checks here.
2. Recomputes from exp/results/full/v11_obd/v11_obd_full/results.jsonl (800 rows, paired by seed, bootstrap
   B = 10^4 with numpy default_rng(42), as in the registered analysis):
   the headline ratio / CI / UB95 (must equal v11_analysis.json); the per-stream N80 stopping points and fractions of
   the table; per-stream exhaustion and the lock-v10 pre-exhaustion condition (both fractions zero, NOT met at block
   level); the uncensored-subset ratio (188 streams, descriptive); the full-frontier (15 of 15) ratio (descriptive,
   secondary); and the corrected development ratio.
3. Checks the corrected and new text in main.tex, supplement.tex and exp/results/full/v11_summary.md, and asserts the
   absence of the removed false or overbroad phrases.
4. Checks the facts behind the three new disclosures (provenance timestamp if the record is present, the public
   verify_eval_bytes helper, the 'rigorous' label) and, if git history is available, the seal commit attribution.
Usage (package or workspace root): .venv/bin/python3 writing/scripts/verify_r7b_numbers.py [--skip-census]
"""
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IT = HERE.parents[1]
RES = IT / "exp/results"
MAIN = (IT / "writing/latex_acm/main.tex").read_text()
SUPP = (IT / "writing/supplement/supplement.tex").read_text()
SUMM = (RES / "full/v11_summary.md").read_text()
FAIL = []

# verify_r7 output lines (after "[MISMATCH] ") deliberately corrected in r7b -> replacement checked below.
# Keys are prefixes of the printed line (verify_r7 truncates long strings).
SUPERSEDED = {
    "r6 chain: supp: 'robustness claims in Section 6.6 of the paper'":
        "S9 now points to Section 6.7 (robustness); 6.6 is the fresh-log test",
    "r6 chain: main 6.5 block G: 'the ratio generally falls as $\\\\eps$ grows":
        "main keeps the falling-ratio endpoints; 'unpinned' ranges 0.09--0.18 / 0.34--0.45 cut for space, kept in S22 and S19",
    "r6 chain: main 6.5 block G: 'moved the lock-cell ratios by about 0.035 at most'":
        "main: 'moved the lock-cell ratios by at most about 0.035'; original sentence in S22",
    "r6 chain: main 6.5 G3: 'a checkpoint-valid exact-hypergeometric rectangle, weaker in guarantee":
        "main: 'a checkpoint-valid exact rectangle'; full wording in S22",
    "r6 chain: main conclusion: 'heterogeneous-cost and robustness tests were descriptive":
        "conclusion narrowed (review r7): 'only the nine-segment Open Bandit test had confirmatory knapsack costs'",
    "r6 chain: supp S9: 'Heterogeneous knapsack costs were tested only descriptively":
        "S9 narrowed (review r7): large-class costs descriptive; lock v11 is the one confirmatory knapsack test",
    "r6 chain: main 6.5 pointer to S20: 'Three localised union ledgers":
        "main: 'Two tested localised ledgers ... a Gaussian diagnostic rejected a third' (review r7)",
    "main v11: 'the Open Bandit Dataset~\\\\citep{saito2021open}, where the knapsack form FDC-DP":
        "intro ends 'so the saving survives outside the reused tables' (critic r7 W3b)",
    "main v11: 'It certified 12 of 15 problems after 73\\\\% of the table":
        "pre-exhaustion claim withdrawn (review r7); replacement sentence checked",
    "main Sec. 7 / 6.6 disclosure: '3.3\\\\% of development rows share a timestamp":
        "'weakly dependent' replaced by 'dependent to a degree the log cannot bound' (review r7)",
    "supp S21: 'a mean exhausted fraction of 0.06 for the governing rival'":
        "S21: 'which gives its mean exhausted fraction of 0.06' with per-stream counts",
    "main shortened replacement: 'unless 3 to 26 runs share it'":
        "main: 'unless, in point estimates, 3 to 26 runs share it' (review r7 qualifier)",
}


def check(label, computed, printed, tol):
    ok = abs(computed - printed) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: computed {computed:.6g}, paper {printed}")
    if not ok:
        FAIL.append(label)


def present(label, needle, txt, where):
    ok = needle in txt
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {needle[:110]!r} printed in {where}")
    if not ok:
        FAIL.append(label + " (text)")


def absent(label, needle, txt, where):
    ok = needle not in txt
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: {needle[:110]!r} absent from {where}")
    if not ok:
        FAIL.append(label + " (absent)")


def section(t):
    print(f"\n== {t}")


def r7_chain(args):
    section("r7 chain (verify_r7 -> r6 -> ... -> r3); only listed checks may fail")
    p = subprocess.run([sys.executable, str(HERE / "verify_r7_numbers.py"), *args], capture_output=True, text=True)
    out = p.stdout.splitlines()
    n_ok = sum("[OK]" in l for l in out)
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
            print(f"  [MISMATCH] r7 chain: {key}")
            FAIL.append(f"r7 chain: {key[:80]}")
    for k in SUPERSEDED:
        if k not in used:
            print(f"  [note] superseded item not failing (string still present?): {k[:100]}")
    last = [l for l in out if l.startswith("r7:")]
    print(f"  chain: {n_ok} OK lines; {len(used)} superseded checks; r7 final line: {last[-1][:40] if last else '(none)'}")
    if not last:
        FAIL.append("r7 chain produced no final line")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])
        FAIL.append("chain stderr")


AN = json.loads((RES / "full/v11_obd/v11_analysis.json").read_text())
DEV = json.loads((RES / "pilots/v11_dev/v11_analysis_dev.json").read_text())
P, R, HC, HG = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP[tuned]", "RECT-HG-DP"
ROWS = [json.loads(l) for l in (RES / "full/v11_obd/v11_obd_full/results.jsonl").read_text().splitlines() if l.strip()]
BY = {}
for r_ in ROWS:
    BY.setdefault(r_["method"], {})[r_["seed"]] = r_
SEEDS = sorted(BY[P])
TAU = AN["env"]["N"]


def boot(d):
    d = np.asarray(d, float)
    idx = np.random.default_rng(42).integers(0, len(d), size=(10000, len(d)))
    b = d[idx].mean(axis=1)
    return math.exp(d.mean()), math.exp(np.quantile(b, 0.95)), np.exp(np.quantile(b, [0.025, 0.975]))


def col(m, key):
    return np.array([BY[m][s][key] for s in SEEDS], float)


def recompute():
    section("Lock v11 rows: headline, stopping points, exhaustion (recomputed from results.jsonl)")
    check("800 rows, 4 methods x 200 seeds", len(ROWS) * 10 + len(BY), 8004, 0)
    check("seeds 39000-39199", float(SEEDS == list(range(39000, 39200)) and all(sorted(BY[m]) == SEEDS for m in BY)), 1, 0)
    p, r = col(P, "N80_pen"), col(R, "N80_pen")
    ratio, ub, ci = boot(np.log(p) - np.log(r))
    c = AN["cell"]["comparisons"][f"{P}/{R}"]
    check("headline ratio = analysis JSON", ratio, c["geomean_ratio"], 1e-12)
    check("headline UB95 = analysis JSON", ub, c["ub95_one_sided"], 1e-12)
    check("headline CI = analysis JSON", float(np.allclose(ci, c["ci95_two_sided"], atol=1e-12)), 1, 0)
    check("headline ratio 0.817 (unchanged)", ratio, 0.817, 5e-4)
    check("headline UB95 0.824 (unchanged)", ub, 0.824, 5e-4)
    # per-stream stopping points of the primary
    cnt = Counter(p.astype(int))
    want = {415272: 11, 471164: 87, 534578: 101, 606527: 1}
    check("primary N80 values and stream counts 415,272/471,164/534,578/606,527 on 11/87/101/1", float(dict(cnt) == want), 1, 0)
    for n_, pct in zip(sorted(want), (60.3, 68.5, 77.7, 88.1)):
        check(f"N80 {n_:,} = {pct}% of the table", 100 * n_ / TAU, pct, 0.05)
    kp = Counter(int(BY[P][s]["k80"]) for s in SEEDS)
    check("primary checkpoints 35-38 (1-based K=40 grid index)", float(set(kp) == {35, 36, 37, 38}), 1, 0)
    check("primary mostly at 68.5% or 77.7% (188 of 200)", cnt[471164] + cnt[534578], 188, 0)
    cr = Counter(r.astype(int))
    check("rival N80 606,527 on 188 streams and tau_R on 12", float(dict(cr) == {606527: 188, TAU: 12}), 1, 0)
    check("rival stopping point 88.1% of the table", 100 * 606527 / TAU, 88.1, 0.05)
    check("only the rival is in the last eighth (min primary fraction < 7/8 < rival fraction)",
          float(p.min() / TAU < 0.875 < 606527 / TAU and (p / TAU < 0.875).sum() == 199), 1, 0)
    check("one checkpoint step = factor 1.13", (TAU / 5000) ** (1 / 39), 1.13, 0.005)
    # exhaustion and the lock-v10 condition
    ep = np.array([BY[P][s]["exhaustion_at_k80"]["all"] for s in SEEDS])
    er = np.array([BY[R][s]["exhaustion_at_k80"]["all"] for s in SEEDS])
    check("FDC-DP: no exhausted pool on all 200 streams", int((ep == 0).sum()), 200, 0)
    both = (ep == 0) & (er == 0)
    check("both methods no exhausted pool on 188 streams", int(both.sum()), 188, 0)
    check("rival: every pool exhausted on the 12 horizon streams", float(((er == 1.0) == (r >= TAU)).all() and (er[r >= TAU] == 1).all()), 1, 0)
    check("rival mean exhausted fraction 0.06", er.mean(), 0.06, 1e-12)
    check("lock-v10 pre-exhaustion condition (both fractions zero on every stream) NOT met", float(not both.all()), 1, 0)
    v10 = (IT / "plan/prereg_lock_v10_addendum.md").read_text()
    check("lock-v10 addendum requires both methods at fraction 0", float("only where both methods have fraction 0" in v10), 1, 0)
    # uncensored subset (descriptive)
    lt = col(R, "n80_lt_tau").astype(bool)
    check("uncensored subset = rival before tau = both unexhausted", float((lt == both).all()), 1, 0)
    sr, su, _ = boot((np.log(p) - np.log(r))[both])
    check("uncensored-subset ratio 0.821 (188 streams)", sr, 0.821, 5e-4)
    check("uncensored-subset UB95 0.829", su, 0.829, 5e-4)
    # full frontier (secondary, descriptive)
    pf, rf = col(P, "N_stop_pen"), col(R, "N_stop_pen")
    check("all methods reach 15 of 15", float(all(BY[m][s]["stop_k"] == 15 for m in BY for s in SEEDS)), 1, 0)
    check("full-frontier geomean rows FDC-DP 572,301", math.exp(np.log(pf).mean()), 572301, 0.5)
    check("full-frontier geomean rows rectangle 670,998", math.exp(np.log(rf).mean()), 670998, 0.5)
    fr, fu, _ = boot(np.log(pf) - np.log(rf))
    check("full-frontier ratio 0.853", fr, 0.853, 5e-4)
    check("full-frontier UB95 0.859", fu, 0.859, 5e-4)
    check("rectangle reaches 15 of 15 only at tau_R on 160 of 200", int((rf >= TAU).sum()), 160, 0)
    check("FDC-DP full frontier before tau_R on every stream", int((pf < TAU).sum()), 200, 0)
    # development ratio
    dv = DEV["cell"]["comparisons"][f"{P}/{R}"]["geomean_ratio"]
    check("development ratio 0.8255 (summary correction)", dv, 0.8255, 5e-5)
    check("development ratio prints as 0.825", float(f"{dv:.3f}" == "0.825"), 1, 0)


def disclosure_facts():
    section("Facts behind the r7b disclosures and the seal attribution")
    src = (IT / "exp/code/dsswm/envs/obd_v11_eval.py").read_text()
    allblk = src[src.index("__all__"):src.index("]", src.index("__all__"))]
    check("verify_eval_bytes still public in the frozen module (__all__)", float('"verify_eval_bytes"' in allblk), 1, 0)
    m = re.search(r"def read_eval_rows\(.*?\n(.*?)\ndef ", src + "\ndef ", re.S)
    body = m.group(1) if m else ""
    gate_first = body.find("verify_eval_bytes") > max(body.find("eval_gate"), body.find("ACCESS_LOG"), body.find("fsync"))
    check("registered reader authorises/logs before verify_eval_bytes", float(m is not None and gate_first), 1, 0)
    meth = AN["cell"]["methods"]
    check("analysis JSON labels RECT-HG-DP 'rigorous' (paper: CP)", float(meth[HG].get("validity") == "rigorous"), 1, 0)
    log = [json.loads(l) for l in (RES / "full/v11_obd/eval_access_log.jsonl").read_text().splitlines() if l.strip()]
    check("access log: three records", len(log), 3, 0)
    check("access log: main run 15:12:18, then replica and analysis",
          float(log[0]["at"].startswith("2026-10-05T15:12:18") and log[1]["at"] < log[2]["at"]), 1, 0)
    src2 = (IT / "exp/code/dsswm/envs/obd_v11.py").read_text()
    mm = re.search(r'OBD_ROOT = "([^"]*)"', src2)
    prov = Path(mm.group(1)) / "PROVENANCE.json" if mm else None
    if prov is not None and prov.exists():
        pv = json.loads(prov.read_text())
        check("PROVENANCE split_created_utc = 03:26 UTC (record rewrite time)", float(pv["split_created_utc"].startswith("2026-10-05T03:26")), 1, 0)
        older = all((prov.parent / f).stat().st_mtime < prov.stat().st_mtime - 20 * 60 for f in ("dev.pkl", "eval_labels.pkl", "eval_outcome.npy"))
        check("split files about 25 minutes older than the provenance record", float(older), 1, 0)
    else:
        print("  [note] dataset provenance record not present (artifact copy); timestamp disclosure not rechecked")
    try:
        def files(c):
            o = subprocess.run(["git", "show", "--name-only", "--format=", c], capture_output=True, text=True, cwd=IT, check=True)
            return [Path(x).name for x in o.stdout.split()]
        f1, f2, f3 = files("81ba8691"), files("36c82389"), files("6972e18f")
        check("81ba8691 commits the seal alone", float(f1 == ["v11_obd_full.seal.json"]), 1, 0)
        check("36c82389 commits the seal reference", float(f2 == ["v11_obd_full.seal_ref.json"]), 1, 0)
        check("6972e18f commits the 800 rows and run record", float("results.jsonl" in f3 and "summary.json" in f3), 1, 0)
    except Exception:
        print("  [note] git history not available (artifact copy); seal commit attribution not rechecked")


def text():
    section("r7b text: corrected and new sentences present")
    for s_ in [
        "It certified 12 of 15 problems after 73\\% of the table against 89\\% for the rectangle.",
        "FDC-DP emptied no pool on any stream and neither method did on 188, but the rectangle reached the horizon on 12, so lock v10's pre-exhaustion condition fails.",
        "so the saving survives outside the reused tables",
        "read 8--11\\% fewer rows than frozen 50/50 FDC-BF in the Criteo-structured asymmetric regime",
        "unless, in point estimates, 3 to 26 runs share it",
        "Two tested localised ledgers read no fewer rows on development halves, and a Gaussian diagnostic rejected a third (supplement S20)",
        "an off-policy-evaluation benchmark never replayed in this work",
        "carries over, with the prespecified changes, to an unseen table",
        "so the halves are dependent to a degree the log cannot bound",
        "so C2 is a statement about registered rivals on the tested tables",
        "only the nine-segment Open Bandit test had confirmatory knapsack costs",
        "Its core is the exact-variance Bennett lemma (Lemma~\\ref{lem:bennett}).",
        "moved the lock-cell ratios by at most about 0.035",
        "needed 0.807 and 0.791 times the rows of the tuned exact and betting rectangles, ratios that are horizon-sensitive (S13)",
    ]:
        present("main r7b", s_, MAIN, "main")
    for s_ in [
        "415,272, 471,164, 534,578 or 606,527 rows (60.3\\%, 68.5\\%, 77.7\\% and 88.1\\% of the table, checkpoints 35--38) on 11, 87, 101 and 1 streams",
        "RECT-BF-DP-TU stops at 606,527 rows (88.1\\%) on 188 streams and reaches $\\tau_R$ on the other 12",
        "had no exhausted pool at its stopping point on all 200 streams; both methods had none on 188 streams",
        "which gives its mean exhausted fraction of 0.06",
        "requires both methods' exhausted fractions to be zero (S17), so it is not met at block level",
        "the ratio is 0.821 (UB95 0.829)",
        "the geometric means are 572,301 and 670,998 rows and the ratio is 0.853 (UB95 0.859)",
        "only at $\\tau_R$ on 160 of 200 streams, so that ratio is horizon-sensitive and is not the registered endpoint",
        "Because the checkpoints are log-spaced, one step changes a stream's ratio by a factor of about 1.13",
        "the halves are therefore not independent, and the log gives no quantitative bound on their dependence",
        "(ix) The provenance record's \\texttt{split\\_created\\_utc}",
        "(x) The frozen evaluation module still exports \\texttt{verify\\_eval\\_bytes}",
        "(xi) \\texttt{v11\\_analysis.json} labels every method, including RECT-HG-DP, \\texttt{validity: rigorous}",
        "the run was sealed alone (commit \\texttt{81ba8691}; rows and run record \\texttt{6972e18f})",
        "These gaps bound the scope of the robustness claims in Section 6.7 of the paper",
        "The one confirmatory knapsack test is lock v11 (S21), with nine segments, 512 policies, one tolerance",
        "Two localised or prior-weighted replacements for FDC-DP's uniform union were built and replayed",
        "For the examined thresholded-sum construction",
        "it replayed two constructions and examined a third analytically",
        "[r7b correction: only two of these ledgers were built and replayed",
        "and is 0.09--0.18 and 0.34--0.45 where the rectangle passes that criterion and is unpinned",
        "moved the lock-cell ratios by about 0.035 at most",
        "2026-10-05 (r7b) & lock-v11 reporting corrected after the r7 reviews",
        "\\item Paper r7, Section 6.6, and supplement r7, S21: said that lock v11 meets the lock-v10 pre-exhaustion condition",
        "\\item \\texttt{exp/results/full/v11\\_summary.md} (lock-v11 result summary, not a locked file)",
    ]:
        present("supp r7b", s_, SUPP, "supp")
    for s_ in ["## Corrections (r7 reviews, added 2026-10-05 in paper revision r7b)",
               "The condition is therefore **not met** at block level",
               "N80 = 415,272 / 471,164 / 534,578 / 606,527 rows (60.3 / 68.5 / 77.7 / 88.1% of the table, checkpoints 35–38) on 11 / 87 / 101 / 1 streams",
               "TU-FDC-DP(b) / RECT-BF-DP-TU = 0.821 (UB95 0.829)", "ratio 0.853 (UB95 0.859)", "on 160 of 200 streams",
               "**Development ratio.** 0.8255", "`81ba8691` committed the seal file alone, `36c82389` its reference, and `6972e18f`",
               "`split_created_utc`", "`verify_eval_bytes`", "`validity: \"rigorous\"`"]:
        present("summary corrections", s_, SUMM, "v11_summary.md")

    section("r7b text: removed false or overbroad phrases absent")
    for s_ in ["in the sense of lock v10", "pre-exhaustion in the sense", "Three localised union ledgers", "weakly dependent",
               "carries over unchanged", "on these two tables", "not an artifact of reused tables",
               "heterogeneous-cost and robustness tests were descriptive", "last eighth"]:
        absent("main r7b", s_, MAIN, "main")
    s22 = SUPP[SUPP.index("\\label{app:moved}"):]
    supp_wo_s22 = SUPP.replace(s22, "")
    for s_ in ["last eighth", "FDC-DP's exhausted fraction 0 and the rival's below 1", "therefore holds here, unlike on Lenta",
               "weakly dependent", "so no confirmatory test covers them", "robustness claims in Section 6.6",
               "tested three constructions", "a union-free event must also control", "Three localised or prior-weighted replacements",
               "Because both methods certify"]:
        absent("supp r7b", s_.replace("last eighth", "certify in the last eighth"), supp_wo_s22, "supp (S1-S21)")
    absent("supp S14 quotes the r7 'last eighth' wording only as an erratum", "certify in the last eighth of the table,", SUPP, "supp")
    i3 = s22.find("Three localised union ledgers")
    check("S22 keeps the r6d sentence only with the marked r7b correction right after it",
          float(i3 >= 0 and "[r7b correction:" in s22[i3:i3 + 260]), 1, 0)
    live = re.sub(r"~~.*?~~", "", SUMM, flags=re.S)
    live = live[:live.index("## Corrections (r7 reviews")]
    for s_ in ["whole 15-budget frontier", "meets the lock-v10 condition", "dev ratio was 0.826", "Sealed alone (6972e18f)",
               "weakly dependent", "of the 512 policies are"]:
        absent("summary (outside struck text)", s_, live, "v11_summary.md")


def rules():
    section("Convergence rules on the r7b paragraphs")
    for start in ("The test passed.", "\\textbf{The fresh-log test.}", "\\textbf{Limits of the rival set.}", "Beyond \\S\\ref{sec:deviations}",
                  "Both blocks passed", "A post-hoc block G addresses"):
        para = [x for x in MAIN.split("\n\n") if x.startswith(start)][0].strip()
        ok = not re.search(r"(\d\)?|\})\.$", para)
        check(f"main paragraph '{start[:26]}' ends on words", float(ok), 1, 0)
    for start in ("\\textbf{Where the rows go.}", "\\textbf{Disclosures.}"):
        para = [x.strip() for x in SUPP.split("\n\n") if x.strip().startswith(start)][0]
        check(f"supp paragraph '{start[8:26]}' ends on words", float(not re.search(r"(\d\)?|\})\.$", para)), 1, 0)
    sents = re.split(r"(?<=[.;])\s+", [x for x in MAIN.split("\n\n") if x.startswith("The test passed.")][0])
    check("Sec. 6.6 result paragraph: no sentence over 45 words", float(max(len(s.split()) for s in sents) <= 45), 1, 0)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r7_chain(args)
    recompute()
    disclosure_facts()
    text()
    rules()
    print(f"\nr7b: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
