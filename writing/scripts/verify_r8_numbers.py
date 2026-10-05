"""Check paper revision r8 (lock v12: descriptive replications on the Open Bandit women and men campaigns) (read-only).

1. Runs verify_r8a_numbers.py (which runs verify_r7b and the whole chain down to verify_r3, the lock-v11 recomputation
   and the page check). Every check must still pass, except those listed in SUPERSEDED: their wording was deliberately
   changed in r8 to pay for the lock-v12 paragraph (Sec. 6.6) and the Sec. 7 sentence. Each listed item has a
   replacement checked here, and every cut fact is checked to remain verbatim in the supplement (S16, S17, S22).
2. Recomputes lock v12 from the sealed rows exp/results/full/v12_obd/v12_{women,men}_full/results.jsonl (paired by seed,
   bootstrap B = 10^4 with numpy default_rng(42), as in the registered analysis): every comparison, the full-frontier
   and uncensored-subset readings, F/T/S shares, false streams, exhaustion, stopping points and checkpoint gains, and
   compares them with v12_{women,men}_analysis.json.
3. Checks every lock-v12 number printed in the paper (Sec. 6.6, Sec. 7) and in supplement S23 (and the new S2 rows)
   against the analyses, the development analyses, the v12 gates, the lock record, the plan, the development report,
   the summary and the two lock reviews; checks the S23 tables against gen_r8_v12_tables.py.
4. Convergence rules on the new paragraphs, unchanged thesis / abstract / claims, and the page budget (pymupdf).
Usage (workspace root): .venv/bin/python3 writing/scripts/verify_r8_numbers.py [--skip-census]
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
FAIL = []
CAMPS = ("women", "men")
P, R, HC, HG = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP[tuned]", "RECT-HG-DP"
METHODS = (P, R, HC, HG)

# verify_r8a output lines (after "[MISMATCH] ") deliberately changed in r8 -> replacement checked below.
SUPERSEDED = {
    "r7b chain: r7 chain: r6 chain: chain: main text: '0.997 and 1.000 (supplement S12)'":
        "main 6.3 ends 'X5 is the only log with a confirmatory non-inferiority test.'; Criteo 0.997 / 1.000 verbatim in S22",
    "r7b chain: r7 chain: r6 chain: main 6.2 workflow (P0-2): 'the saving goes to whichever collaborator pays for matching'":
        "main 6.2 keeps '$10k a month (supplement S16)'; 'The saving accrues to whichever collaborator pays for matching' in S16",
    "r7b chain: r7 chain: main shortened replacement: 'needed 0.807 and 0.791 times the rows of the tuned exact and betting rectangles":
        "main 6.7 keeps Hillstrom as descriptive context with a pointer to S13; 0.807 / 0.791 with UB95 verbatim in S22",
    "r7b chain: r7 chain: main shortened replacement: 'had 2 false Criteo streams of 200, so rules without a proof stay descriptive'":
        "main: 'had 2 false Criteo streams of 200, so unproven rules stay descriptive'; old wording verbatim in S22",
    "r7b chain: main r7b: 'needed 0.807 and 0.791 times the rows of the tuned exact and betting rectangles":
        "same Hillstrom cut as above (r7b copy of the check)",
    "r7b chain: r7 chain: supp S19 / S22 / S2: '(lock v5 through lock v11)'":
        "S2 caption now '(lock v5 through lock v12)' (v12 and r8 rows added); checked in supp_text()",
}


def ok(label, cond, detail=""):
    print(f"  [{'OK' if cond else 'MISMATCH'}] {label}{(': ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(label)


def check(label, computed, printed, tol):
    good = abs(computed - printed) <= tol
    print(f"  [{'OK' if good else 'MISMATCH'}] {label}: computed {computed:.6g}, printed {printed}")
    if not good:
        FAIL.append(label)


def present(label, needle, txt, where):
    ok(label, needle in txt, f"{needle[:110]!r} printed in {where}")


def absent(label, needle, txt, where):
    ok(label, needle not in txt, f"{needle[:110]!r} absent from {where}")


def section(t):
    print(f"\n== {t}")


def r8a_chain(args):
    section("r8a chain (verify_r8a -> r7b -> r7 -> r6 -> ... -> r3); only listed checks may fail")
    p = subprocess.run([sys.executable, str(HERE / "verify_r8a_numbers.py"), *args], capture_output=True, text=True)
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
            print(f"  [MISMATCH] r8a chain: {key}")
            FAIL.append(f"r8a chain: {key[:80]}")
    for k in SUPERSEDED:
        if k not in used:
            print(f"  [note] superseded item not failing (string still present?): {k[:100]}")
    for l in out:
        if l.strip().startswith("[note]"):
            print("  chain " + l.strip())
    last = [l for l in out if l.startswith("r8a:")]
    print(f"  chain: {n_ok} OK lines; {n_sup_inner} inherited supersessions; {len(used)} r8 supersessions; "
          f"r8a final line: {last[-1][:40] if last else '(none)'}")
    if not last:
        FAIL.append("r8a chain produced no final line")
    if p.stderr.strip():
        print("  chain stderr:", p.stderr.strip()[-400:])
        FAIL.append("chain stderr")


def replacements():
    section("r8 replacements for the superseded checks, and cut facts kept in the supplement")
    for s_ in ["It thus matches PJC within the registered 5\\% margin; X5 is the only log with a confirmatory non-inferiority test.",
               "Ten Criteo-sized campaigns re-certified four times a month save about \\$10k a month (supplement S16).",
               "It also held, descriptively, on Hillstrom's three-arm log~\\citep{hillstrom2008minethatdata}, whose cell means an earlier analysis had exposed, and on a 4,096-policy Criteo segmentation (S3, S6, S13).",
               "had 2 false Criteo streams of 200, so unproven rules stay descriptive.",
               "An exact hypergeometric cell envelope needed 0.969 (UB95 0.990) times FDC-BF's rows on Criteo, and a gap-weighted, still valid localisation of the union 0.925 (UB95 0.949).",
               "the menu member re-plans its control share at one boundary (supplement S7, S22)",
               "These departures (S2; errata in S14) bound how C1--C4 may be read."]:
        present("main r8", s_, MAIN, "main")
    for s_ in ["and on Criteo the descriptive ratios are 0.997 and 1.000 (supplement S12)",
               "The saving accrues to whichever collaborator pays for matching",
               "0.807 (UB95 0.814) and 0.791 (UB95 0.799) times the rows of the tuned exact and betting rectangles",
               "so these ratios are horizon-sensitive (supplement S13)",
               "so rules without a proof stay descriptive",
               "because at certification the two bounds differ only in third-order terms",
               "because most (problem, policy) pairs are near-ties",
               "keeps all data and chooses a control share of 0.4 or 0.5 at one boundary",
               "Its union over menu paths is charged in full (supplement S7)",
               "On X5, TU-FDC-DP certifies after 14--16\\% of the table with no pool exhausted, while the rectangles need 81--83\\%",
               "errata to locked files and earlier drafts are indexed in S14"]:
        present("supp keeps cut fact", s_, SUPP, "supp")
    for s_ in ["0.807 and 0.791", "whichever collaborator", "third-order terms", "14--16\\%", "0.997 and 1.000",
               "control share of 0.4 or 0.5"]:
        absent("main r8 cut", s_, MAIN, "main")
    # X5 stopping fractions also stand in Table 2 (N80/tau_R 0.155 / 0.814 and 0.137 / 0.814)
    present("Table 2 keeps X5 stopping fractions", "0.155 / 0.814", MAIN, "main")


def J(p):
    return json.loads(Path(p).read_text())


def boot(d):
    d = np.asarray(d, float)
    idx = np.random.default_rng(42).integers(0, len(d), size=(10000, len(d)))
    b = d[idx].mean(axis=1)
    return math.exp(d.mean()), math.exp(np.quantile(b, 0.95)), np.exp(np.quantile(b, [0.025, 0.975]))


def fts(d):
    d = np.asarray(d, float)
    return (d < 0).mean(), (d == 0).mean(), (d > 0).mean()


def load(c):
    an = J(RES / f"full/v12_obd/v12_{c}_analysis.json")
    rows = [json.loads(l) for l in (RES / f"full/v12_obd/v12_{c}_full/results.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for r_ in rows:
        by.setdefault(r_["method"], {})[r_["seed"]] = r_
    return {"an": an, "rows": rows, "by": by,
            "dev": J(RES / f"pilots/v12_dev/v12_{c}_analysis_dev.json"),
            "g": J(RES / f"v12_gates/v12_{c}_eps_hc.json"),
            "st": J(RES / f"v12_gates/v12_{c}_block_status.json"),
            "th": J(RES / f"v12_gates/v12_{c}_thresh.json"),
            "fr": J(RES / f"v12_gates/obd_v12_{c}_frozen.json"),
            "rep": J(RES / f"full/v12_obd/v12_{c}_full/replica_report.json")}


D = {c: load(c) for c in CAMPS}
SEEDS = {"women": list(range(39200, 39400)), "men": list(range(39400, 39600))}
V = {}  # recomputed values by campaign, for the text checks


def col(c, m, key):
    by = D[c]["by"]
    return np.array([by[m][s][key] for s in SEEDS[c]], float)


def recompute():
    for c in CAMPS:
        d = D[c]
        an, by = d["an"], d["by"]
        tau = an["env"]["N"]
        v = V[c] = {"tau": tau}
        section(f"Lock v12 {c}: recomputed from results.jsonl ({len(d['rows'])} rows)")
        check(f"{c}: 800 rows, 4 methods x 200 seeds", len(d["rows"]) * 10 + len(by), 8004, 0)
        ok(f"{c}: seeds {SEEDS[c][0]}-{SEEDS[c][-1]} for every method", all(sorted(by[m]) == SEEDS[c] for m in METHODS))
        ok(f"{c}: every row at the block eps", all(r_["eps"] == an["eps"] for r_ in d["rows"]))
        ok(f"{c}: tau_R in every row = env N", all(r_["tau_R"] == tau for r_ in d["rows"]))
        n80 = {m: col(c, m, "N80_pen") for m in METHODS}
        for k, cc in an["cell"]["comparisons"].items():
            a, b = k.split("/")
            dl = np.log(n80[a]) - np.log(n80[b])
            ratio, ub, ci = boot(dl)
            check(f"{c} {k} ratio = JSON", ratio, cc["geomean_ratio"], 1e-12)
            check(f"{c} {k} UB95 = JSON", ub, cc["ub95_one_sided"], 1e-12)
            ok(f"{c} {k} CI = JSON", bool(np.allclose(ci, cc["ci95_two_sided"], atol=1e-12)))
            f, t, s = fts(dl)
            ok(f"{c} {k} F/T/S = JSON", abs(f - cc["frac_faster"]) + abs(t - cc["frac_tied"]) + abs(s - cc["frac_slower"]) < 1e-12,
               f"{f:.3f}/{t:.3f}/{s:.3f}")
            v[k] = (ratio, ub, ci, f, t, s)
        # full frontier and uncensored subset (pre-stated descriptive)
        nsp, nsr = col(c, P, "N_stop_pen"), col(c, R, "N_stop_pen")
        fr_, fu, fci = boot(np.log(nsp) - np.log(nsr))
        dd = an["cell"]["descriptive_v12"]
        check(f"{c} full-frontier ratio = JSON", fr_, dd["full_frontier"]["geomean_ratio"], 1e-12)
        check(f"{c} full-frontier UB95 = JSON", fu, dd["full_frontier"]["ub95_one_sided"], 1e-12)
        v["ff"] = (fr_, fu)
        v["rival_15_at_tau"] = int((nsr >= tau).sum())
        check(f"{c} rival reaches 15/15 only at tau_R (count) = JSON", v["rival_15_at_tau"], dd["full_frontier"]["rival_reaches_15_only_at_tau"], 0)
        ep = np.array([by[P][s]["exhaustion_at_k80"]["all"] for s in SEEDS[c]])
        er = np.array([by[R][s]["exhaustion_at_k80"]["all"] for s in SEEDS[c]])
        both = (ep == 0) & (er == 0)
        check(f"{c} uncensored subset size", int(both.sum()), dd["uncensored_subset"]["n_streams_in_subset"], 0)
        ur, uu, _ = boot((np.log(n80[P]) - np.log(n80[R]))[both])
        check(f"{c} uncensored-subset ratio = JSON", ur, dd["uncensored_subset"]["geomean_ratio"], 1e-12)
        check(f"{c} uncensored-subset ratio = headline (all 200 streams)", ur, v[f"{P}/{R}"][0], 1e-12)
        # per-method facts
        allm = [r_ for r_ in d["rows"]]
        v["false_any"] = sum(bool(r_["fwer_event"]) or r_["n_false"] > 0 for r_ in allm)
        check(f"{c} false streams, all methods (rows)", v["false_any"], 0, 0)
        ok(f"{c} JSON false streams 0 for every method", all(an["cell"]["methods"][m]["false_streams"] == 0 for m in METHODS))
        v["exh_any"] = sum(any(r_["exhaustion_at_k80"][x] > 0 for x in ("all", "ctrl", "treat")) for r_ in allm)
        check(f"{c} exhausted pools at N80, all methods and streams (rows)", v["exh_any"], 0, 0)
        ok(f"{c} every method reaches 12/15 before tau_R and 15/15 on every stream (rows)",
           all(r_["n80_lt_tau"] and r_["stop_k"] == 15 for r_ in allm))
        for m in METHODS:
            x = an["cell"]["methods"][m]
            gm = math.exp(np.log(n80[m]).mean())
            check(f"{c} {m} geomean N80_pen = JSON", gm, x["geomean_N80_pen"], 1e-6)
            check(f"{c} {m} N80/tau = JSON", math.exp(np.log(n80[m] / tau).mean()), x["geomean_N80_over_tau"], 1e-12)
        # stopping points
        v["p_cnt"] = sorted(Counter(n80[P].astype(int)).items())
        v["r_cnt"] = sorted(Counter(n80[R].astype(int)).items())
        v["p_k"] = sorted(set(int(by[P][s]["k80"]) for s in SEEDS[c]))
        gain = Counter(int(by[R][s]["k80"]) - int(by[P][s]["k80"]) for s in SEEDS[c])
        v["gain"] = gain
        v["step"] = (tau / 5000) ** (1 / 39)
        print(f"  [info] {c}: primary N80 {v['p_cnt']}, rival N80 {v['r_cnt']}, checkpoint gain {sorted(gain.items())}")
        # replica
        ok(f"{c} replica report pass (R1 on 10 seeds x 4 methods, R2, R2b)",
           d["rep"]["status"] == "pass" and [x["rule"] for x in d["rep"]["checks"]] == ["R1", "R2", "R2b"]
           and all(x["pass"] for x in d["rep"]["checks"]) and d["rep"]["checks"][0]["n_compared"] == 40)
        ok(f"{c} analysis verdict descriptive, THRESH not applicable", an["decision"]["verdict"] == "descriptive"
           and an["decision"]["thresh_applicable"] is False and an["block_status"] == "descriptive")
        b = an["cell"]["methods"][P]
        v["bnb"] = (b["bnb_calls"], b["b_only_certs"], b["node_limit_hits"])


def fmt(x):
    return f"{x:.3f}"


def eps_tex(e):
    return {0.00075: "7.5 \\cdot 10^{-4}", 0.001: "10^{-3}"}[e]


def main_text():
    section("Main text: lock-v12 paragraph (Sec. 6.6) and Sec. 7 sentence")
    W, M = D["women"], D["men"]
    k = f"{P}/{R}"
    para = [x for x in MAIN.split("\n\n") if x.strip().startswith("Lock v12 replicated")]
    ok("one lock-v12 paragraph in Sec. 6.6", len(para) == 1)
    para = para[0].strip() if para else ""
    obd = MAIN[MAIN.index("\\subsection{A never-replayed log}"):MAIN.index("\\subsection{Robustness and descriptive context}")]
    ok("v12 paragraph sits inside Sec. 6.6 after the v11 result", para in obd and obd.index(para) > obd.index("The test passed."))
    present("v12 eps", f"moved $\\eps$ to ${eps_tex(W['an']['eps'])}$ and ${eps_tex(M['an']['eps'])}$", para, "6.6")
    sw, sm = W["st"]["mean_share_eps_optimal"], M["st"]["mean_share_eps_optimal"]
    present("v12 dev eps-optimal shares", f"where {fmt(sw)} and {fmt(sm)} of feasible development policies are $\\eps$-optimal (gate 0.5)", para, "6.6")
    ok("gate limit 0.5 = code SHARE_MAX", "SHARE_MAX = 0.5" in (IT / "exp/code/dsswm/stats/v12_analysis.py").read_text())
    ok("both shares above 0.5 and both gates descriptive", sw > 0.5 and sm > 0.5 and W["st"]["status"] == M["st"]["status"] == "descriptive")
    rw, rm = V["women"][k], V["men"][k]
    present("v12 headline ratios", f"FDC-DP needed {fmt(rw[0])} (UB95 {fmt(rw[1])}) and {fmt(rm[0])} (UB95 {fmt(rm[1])}) times the matched rectangle's reads", para, "6.6")
    check("women ratio 0.677", rw[0], 0.677, 5e-4)
    check("women UB95 0.683", rw[1], 0.683, 5e-4)
    check("men ratio 0.739", rm[0], 0.739, 5e-4)
    check("men UB95 0.744", rm[1], 0.744, 5e-4)
    ow = max(V["women"][f"{P}/{HC}"][0], V["women"][f"{P}/{HG}"][0])
    om = max(V["men"][f"{P}/{HC}"][0], V["men"][f"{P}/{HG}"][0])
    present("v12 other two rivals (max of the two ratios)", f"and at most {fmt(ow)} and {fmt(om)} times the other two rivals'", para, "6.6")
    ok("faster on every stream (both campaigns)", rw[3] == 1.0 and rm[3] == 1.0)
    present("faster on every stream", "faster on every stream", para, "6.6")
    ok("no false certificate and no exhausted pool for any method (rows)",
       V["women"]["false_any"] == V["men"]["false_any"] == 0 and V["women"]["exh_any"] == V["men"]["exh_any"] == 0)
    present("no false / no exhausted", "no method made a false certificate or exhausted a pool", para, "6.6")
    present("descriptive, not confirmatory, not pooled", "This is replication evidence, not a second confirmatory test, and is not pooled with lock v11's ratio.", para, "6.6")
    present("identical protocol, fresh splits, one replay", "under the identical frozen protocol, with fresh salted splits and each evaluation half replayed once", para, "6.6")
    present("two other uniform-random campaigns", "on the log's two other uniform-random campaigns, women and men", para, "6.6")
    present("gate fixed before data", "A gate fixed before any of their data were read made both blocks descriptive", para, "6.6")
    plan = (IT / "plan/v12_obd2_plan.md").read_text()
    ok("plan: gate pre-stated before any dev click is read", "## 6. Block-status gate (pre-stated here, before any dev click is read)" in plan
       and "before any row of `random/women` or `random/men` was split" in plan)
    ok("random/women and random/men are the two other uniform-random campaigns (README of OBD: random/all, men, women)",
       "random/{women,men}" in plan or "`random/women` or `random/men`" in plan)
    sec7 = [x for x in MAIN.split("\n\n") if x.strip().startswith("\\textbf{The fresh-log test.}")][0]
    present("Sec. 7 disclosure", "Lock v12 inherits these choices on two campaigns and is descriptive by a pre-stated gate (supplement S23).", sec7, "Sec. 7")
    present("Sec. 7 keeps the C2 scope sentence", "The fresh-log part of C2 thus covers the named rectangles on this one log.", sec7, "Sec. 7")
    # confirmatory claim stays v11 alone
    pre = (IT / "writing/latex_acm/main_pre_r8.tex").read_text()
    for s_ in ["0.677", "0.739", "0.683", "0.744", "0.731", "0.781", "0.661", "0.858"]:
        ok(f"{s_}: every new occurrence in main is in the Sec. 6.6 paragraph", MAIN.count(s_) - pre.count(s_) == para.count(s_),
           f"{MAIN.count(s_)} in main, {pre.count(s_)} before r8, {para.count(s_)} in the paragraph")
    grab = lambda t, a, b: t[t.index(a):t.index(b)]
    ok("abstract unchanged from r8a", grab(MAIN, "\\begin{abstract}", "\\end{abstract}") == grab(pre, "\\begin{abstract}", "\\end{abstract}"))
    ok("claims C1-C4 unchanged from r8a", grab(MAIN, "\\item[\\textbf{C1}]", "\\noindent Our contributions") == grab(pre, "\\item[\\textbf{C1}]", "\\noindent Our contributions"))
    ok("conclusion unchanged from r8a", grab(MAIN, "\\section{Limitations and Conclusion}", "\\bibliographystyle") == grab(pre, "\\section{Limitations and Conclusion}", "\\bibliographystyle"))
    thesis = ("Freezing the read plan lets one finite-population Chernoff width per policy difference certify a whole budget "
              "frontier with 0.30--0.82 times the outcome reads of matched-strength per-cell rectangles on two reused logs and a never-replayed one.")
    ok("thesis unchanged (abstract and intro)", MAIN.count(thesis) == 2)
    for s_ in ["v12", "women", "men"]:
        rx = re.compile(r"\b" + s_ + r"\b")
        ok(f"'{s_}' does not enter abstract or C2", not rx.search(grab(MAIN, "\\begin{abstract}", "\\end{abstract}"))
           and not rx.search(grab(MAIN, "\\item[\\textbf{C2}]", "\\item[\\textbf{C3}]")))
    for bad in ("confirms the", "positive", "pooled ratio"):
        absent(f"v12 paragraph avoids '{bad}'", bad, para, "6.6")


def supp_text():
    section("Supplement S23, S2 rows and S-list")
    i = SUPP.index("\\section{Lock v12: descriptive replications on two further Open Bandit campaigns}")
    s23 = SUPP[i:SUPP.index("\\clearpage\n\\end{document}")]
    ok("S23 is the 23rd section", SUPP[:i].count("\\section{") == 22, f"{SUPP[:i].count(chr(92) + 'section{') + 1}")
    ok("S23 is the last section", s23.count("\\section{") == 1)
    W, M = D["women"], D["men"]
    k = f"{P}/{R}"
    rw, rm = V["women"][k], V["men"][k]
    plan = (IT / "plan/v12_obd2_plan.md").read_text()
    rep = (IT / "plan/v12_dev_report.md").read_text()
    summ = (RES / "full/v12_summary.md").read_text()
    lock = J(IT / "plan/prereg_lock_v12_addendum.json")
    lock_txt = (IT / "plan/prereg_lock_v12_addendum.json").read_text()
    # review files: reviews/ in the artifact, the review tool's folder in the workspace
    rdir = next(d for d in (IT / "reviews", IT / "external reviewer") if (d / "v12_lock_review_r2.md").exists())
    r1, r2 = (rdir / "v12_lock_review_r1.md").read_text(), (rdir / "v12_lock_review_r2.md").read_text()
    # provenance and process
    ok("lock canonical sha256 9a99ef28...075c189", lock["sha256"].startswith("9a99ef28") and lock["sha256"].endswith("075c189"))
    present("S23 lock sha", "canonical sha256 \\texttt{9a99ef28\\ldots 075c189}", s23, "S23")
    ok("code freeze 20a5e314", lock["git_commit"].startswith("20a5e314"))
    present("S23 freeze / lock commit", "code freeze \\texttt{20a5e314}, lock commit \\texttt{be886f1c}", s23, "S23")
    ok("summary: lock commit be886f1c, seals 730f1d21 / d994784b, rows c161bec1",
       all(x in summ for x in ("be886f1c", "730f1d21", "d994784b", "c161bec1")))
    present("S23 seals", "each run was sealed alone (women \\texttt{730f1d21}, men \\texttt{d994784b}; rows and run records \\texttt{c161bec1})", s23, "S23")
    ok("plan commits 317619b4 then 5b3eb048", "317619b4" in plan and "317619b4, then 5b3eb048" in summ)
    present("S23 plan commits", "committed alone (\\texttt{317619b4}, renamed and extended in \\texttt{5b3eb048})", s23, "S23")
    ok("both analyses bind the lock sha", W["an"]["addendum_sha256"] == M["an"]["addendum_sha256"] == lock["sha256"])
    for c in CAMPS:
        log = [json.loads(l) for l in (RES / f"full/v12_obd/{c}_eval_access_log.jsonl").read_text().splitlines() if l.strip()]
        ok(f"{c} access log: three records, all v12_{c}_full, all bind the lock", len(log) == 3
           and all(x["task_id"] == f"v12_{c}_full" and x["v12_addendum_sha256"] == lock["sha256"] for x in log))
    present("S23 access log", "each access log has three records (the main run, then the prespecified ten-seed replica and the analysis)", s23, "S23")
    # data and split
    ok("items 46 / 34 (plan) and 23/23, 17/17 (frozen)", "women 46 items, men 34 items" in plan
       and len(W["fr"]["treat_items"]) == len(W["fr"]["control_items"]) == 23 and len(M["fr"]["treat_items"]) == len(M["fr"]["control_items"]) == 17)
    present("S23 items", "over its own item set (women 46 items, men 34)", s23, "S23")
    present("S23 arms", "(23 of 46 and 17 of 34)", s23, "S23")
    ok("salts in plan", "`dsswm-obd-women-2026-10-05` and `dsswm-obd-men-2026-10-05`" in plan)
    present("S23 salts", "(\\texttt{dsswm-obd-women-2026-10-05}, \\texttt{dsswm-obd-men-2026-10-05})", s23, "S23")
    ok("split names in plan", "`name = 'open_bandit_women'` / `'open_bandit_men'`" in plan)
    nd = {c: D[c]["st"]["dev_env"]["N"] for c in CAMPS}
    ne = {c: D[c]["an"]["env"]["N"] for c in CAMPS}
    check("women dev rows 432,009", nd["women"], 432009, 0)
    check("women eval rows 432,576", ne["women"], 432576, 0)
    check("men dev rows 226,819", nd["men"], 226819, 0)
    check("men eval rows 226,130", ne["men"], 226130, 0)
    present("S23 split sizes", "432,009 development and 432,576 evaluation rows (women) and 226,819 and 226,130 (men)", s23, "S23")
    # design
    ok("S = 8 / 10", W["fr"]["S"] == W["an"]["env"]["S"] == 8 and M["fr"]["S"] == M["an"]["env"]["S"] == 10)
    present("S23 S", "($S = 8$ for women, $S = 10$ for men; no evaluation row needed a fallback)", s23, "S23")
    ok("no eval fallback rows", all(D[c]["an"]["env"]["fallback_rows"] == {"pair": 0, "uf0": 0} for c in CAMPS))
    ok("K = 40, n_min 5,000", all(D[c]["an"]["frozen_configs_used"]["K"] == 40 and D[c]["fr"]["n_min"] == 5000 for c in CAMPS))
    # eps rule and gate
    for c, pct_, base in (("women", "15.5", "0.484"), ("men", "19.9", "0.501")):
        eu = D[c]["an"]["eps_units"]
        check(f"{c} eps / dev base {pct_}%", 100 * eu["relative_to_dev_base"], float(pct_), 0.05)
        check(f"{c} dev base CTR {base}%", 100 * eu["dev_base_ctr"], float(base), 5e-4)
    present("S23 eps rule", "selected $\\eps = 7.5 \\cdot 10^{-4}$ for women and $10^{-3}$ for men", s23, "S23")
    present("S23 eps units", "that is 15.5\\% and 19.9\\% of the development click rates of 0.484\\% and 0.501\\%, against 8.6\\% in lock v11", s23, "S23")
    ok("v11 eps 8.6% of base (S21)", "about 8.6\\% of the development click rate of 0.350\\%" in SUPP)
    mx = {}
    for c in CAMPS:
        tr = D[c]["g"]["eps_rule"]["trace"]
        mx[c] = max(max(t["per_rect"][R]["successes"], t["per_rect"][HC]["successes"]) for t in tr if not t["qualifies"])
        ok(f"{c} selected eps = block eps = first qualifying", D[c]["g"]["eps_rule"]["selected_eps"] == D[c]["an"]["eps"]
           == [t["eps"] for t in tr if t["qualifies"]][0] and tr[-1]["qualifies"])
    ok("non-selected eps: at most 1 of 50 successes (women 0, men 1)", mx["women"] == 0 and mx["men"] == 1)
    present("S23 rule failure", "neither rectangle reached 12 of 15 before $\\tau_R$ on more than 1 of 50 streams", s23, "S23")
    ok("women non-selected = 3e-4, 5e-4; men = 3e-4, 5e-4, 7.5e-4",
       [t["eps"] for t in W["g"]["eps_rule"]["trace"] if not t["qualifies"]] == [0.0003, 0.0005]
       and [t["eps"] for t in M["g"]["eps_rule"]["trace"] if not t["qualifies"]] == [0.0003, 0.0005, 0.00075])
    for c, ac, sh, npol in (("women", 3, "0.661", 256), ("men", 5, "0.858", 1024)):
        st = D[c]["st"]
        ok(f"{c} gate criteria (i) pass, (ii) pass, (iii) fail", st["criteria"] == {"i_rule_selects_eps": True,
           "ii_all_control_not_eps_optimal_in_some_problem": True, "iii_mean_eps_optimal_share_le_0.5": False})
        check(f"{c} all-control eps-optimal problems", st["n_problems_all_control_eps_optimal"], ac, 0)
        check(f"{c} dev eps-optimal share {sh}", st["mean_share_eps_optimal"], float(sh), 5e-4)
        check(f"{c} policies", st["n_policies"], npol, 0)
        check(f"{c} 2^S policies", st["n_policies"], 2 ** D[c]["fr"]["S"], 0)
    present("S23 gate counts", "(all-control $\\eps$-optimal in 3 and 5 of 15 problems)", s23, "S23")
    present("S23 gate shares", "the mean $\\eps$-optimal share was 0.661 over 256 policies (women) and 0.858 over 1,024 (men)", s23, "S23")
    ok("status file times 18:13:16 / 18:13:17", W["st"]["written_at"].startswith("2026-10-05T18:13:16") and M["st"]["written_at"].startswith("2026-10-05T18:13:17"))
    ok("dev report: dev cells started at 18:13:20", "the first\nFDC-DP dev rows) started at 18:13:20" in rep or "started at 18:13:20" in rep)
    present("S23 gate timing", "The status files were written at 18:13:16 and 18:13:17, before the registered development cells began at 18:13:20.", s23, "S23")
    check("women THRESH 0.83", W["th"]["thresh"], 0.83, 0)
    check("men THRESH 0.90", M["th"]["thresh"], 0.90, 0)
    ok("THRESH not applicable in both", W["th"]["applicable"] is False and M["th"]["applicable"] is False)
    present("S23 THRESH", "THRESH was computed by the lock-v11 rule (0.83 for women, 0.90 for men) and recorded as not applicable", s23, "S23")
    # result
    present("S23 women headline", f"{fmt(rw[0])} [{fmt(rw[2][0])}, {fmt(rw[2][1])}] (UB95 {fmt(rw[1])}) times the rows of RECT-BF-DP-TU on women", s23, "S23")
    present("S23 men headline", f"{fmt(rm[0])} [{fmt(rm[2][0])}, {fmt(rm[2][1])}] (UB95 {fmt(rm[1])}) on men", s23, "S23")
    vw, vm = V["women"], V["men"]
    present("S23 descriptive rivals", f"the ratios are {fmt(vw[f'{P}/{HC}'][0])} and {fmt(vw[f'{P}/{HG}'][0])} (women) and "
            f"{fmt(vm[f'{P}/{HC}'][0])} and {fmt(vm[f'{P}/{HG}'][0])} (men)", s23, "S23")
    present("S23 rival-vs-rival", f"({fmt(vw[f'{R}/{HC}'][0])} and {fmt(vw[f'{R}/{HG}'][0])} times the rows of HC-WoR-DP and RECT-HG-DP on women, "
            f"{fmt(vm[f'{R}/{HC}'][0])} and {fmt(vm[f'{R}/{HG}'][0])} on men)", s23, "S23")
    ok("RECT-BF-DP-TU slowest rectangle in both campaigns", all(V[c][f"{R}/{HC}"][0] > 1 and V[c][f"{R}/{HG}"][0] > 1 for c in CAMPS))
    ok("HC-WoR-DP is the strongest time-uniform rival (TU-FDC / HC smallest of TU rivals' ratios is the larger ratio)",
       all(V[c][f"{P}/{HC}"][0] > V[c][f"{P}/{R}"][0] for c in CAMPS))
    cp = W["an"]["cell"]["methods"][P]["cp_ub95"]
    check("Clopper-Pearson 0/200 bound 0.015", cp, 0.015, 5e-4)
    present("S23 CP", "No method made a false certificate (Clopper--Pearson bound 0.015 each)", s23, "S23")
    for c in CAMPS:
        dv = D[c]["dev"]["cell"]["comparisons"][f"{P}/{R}"]
        ok(f"{c} dev ratio > eval ratio", dv["geomean_ratio"] > V[c][f"{P}/{R}"][0])
    dw, dm = (D[c]["dev"]["cell"]["comparisons"][f"{P}/{R}"] for c in CAMPS)
    present("S23 dev estimates", f"The development estimates were {fmt(dw['geomean_ratio'])} (UB95 {fmt(dw['ub95_one_sided'])}) and "
            f"{fmt(dm['geomean_ratio'])} (UB95 {fmt(dm['ub95_one_sided'])})", s23, "S23")
    # where the rows go
    for c, want_p, want_r in (("women", [(217795, 10), (244184, 81), (273770, 109)], [(344131, 18), (385828, 182)]),
                              ("men", [(125801, 1), (138717, 38), (152959, 141), (168663, 20)], [(185980, 2), (205075, 198)])):
        ok(f"{c} primary N80 values and counts", V[c]["p_cnt"] == want_p)
        ok(f"{c} rival N80 values and counts", V[c]["r_cnt"] == want_r)
        tau = V[c]["tau"]
        pcts = ", ".join(f"{100 * n / tau:.1f}" for n, _ in want_p + want_r)
        print(f"  [info] {c} percentages {pcts}")
        for n, _ in want_p + want_r:
            present(f"{c} N80 {n:,} printed", f"{n:,}", s23, "S23")
            present(f"{c} {n:,} = {100 * n / tau:.1f}%", f"{100 * n / tau:.1f}\\%", s23, "S23")
    ok("women primary checkpoints 33-35, men 33-36", V["women"]["p_k"] == [33, 34, 35] and V["men"]["p_k"] == [33, 34, 35, 36])
    present("S23 women checkpoints", "checkpoints 33--35) on 10, 81 and 109 streams", s23, "S23")
    present("S23 men checkpoints", "checkpoints 33--36) on 1, 38, 141 and 20 streams", s23, "S23")
    g_all = V["women"]["gain"] + V["men"]["gain"]
    ok("checkpoint gain 2 to 5, mostly 3 or 4", min(g_all) == 2 and max(g_all) == 5 and g_all[3] + g_all[4] > 0.75 * sum(g_all.values()),
       str(sorted(g_all.items())))
    present("S23 gain", "a gain of 2 to 5 checkpoints per stream (mostly 3 or 4)", s23, "S23")
    check("women step factor 1.12", V["women"]["step"], 1.12, 0.005)
    check("men step factor 1.10", V["men"]["step"], 1.10, 0.005)
    present("S23 step", "a factor of about 1.12 (women) and 1.10 (men)", s23, "S23")
    present("S23 pre-exhaustion", "the lock-v10 pre-exhaustion condition (both fractions zero) holds on every stream of both campaigns, unlike lock v11", s23, "S23")
    present("S23 full frontier", f"is {fmt(V['women']['ff'][0])} (UB95 {fmt(V['women']['ff'][1])}) on women and {fmt(V['men']['ff'][0])} (UB95 {fmt(V['men']['ff'][1])}) on men", s23, "S23")
    ok("rival 15/15 only at tau_R: 3 women, 0 men", V["women"]["rival_15_at_tau"] == 3 and V["men"]["rival_15_at_tau"] == 0)
    present("S23 horizon", "reaches 15 of 15 only at $\\tau_R$ on 3 women streams and no men stream", s23, "S23")
    ew, em = W["an"]["eps_optimal_share"], M["an"]["eps_optimal_share"]
    present("S23 eval shares", f"On the evaluation truth {fmt(ew['mean_share_eps_optimal'])} (women) and {fmt(em['mean_share_eps_optimal'])} (men)", s23, "S23")
    ok("eval all-control eps-optimal 3 and 5", ew["n_problems_all_control_eps_optimal"] == 3 and em["n_problems_all_control_eps_optimal"] == 5)
    bw, bm = V["women"]["bnb"], V["men"]["bnb"]
    present("S23 B&B", f"Branch-and-bound ran {bw[0]:,} and {bm[0]:,} times, issued {bw[1]} and {bm[1]} certifications", s23, "S23")
    ok("node limit never hit", bw[2] == bm[2] == 0)
    # disclosures
    ok("smoke test facts (dev report)", "one dev stream per method, TU-FDC-DP(b) included, (seed 950, ε 5e-4)" in rep
       or ("seed 950, ε 5e-4" in rep and "TU-FDC-DP(b) included" in rep))
    present("S23 smoke", "an unsaved smoke test ran one development stream (seed 950, $\\eps = 5 \\cdot 10^{-4}$) of each of the four methods", s23, "S23")
    ok("test-suite stream at 1.5e-3 (dev report)", "one women dev stream per method at 1.5e-3, seed 950" in rep)
    ok("regeneration: 4,980 rows identical, 39dace28, d23b7aaa (dev report)", "**4,980** dev rows reproduce run 1 exactly" in rep
       and "39dace28" in rep and "d23b7aaa" in rep)
    present("S23 regeneration", "all 4,980 development rows reproduce the first run (\\texttt{d23b7aaa}) exactly", s23, "S23")
    decl = " ".join(lock["declarations"])
    for s_ in ("women rows vs random/all eval half 35 (0.004 %)", "men 29 (0.006 %)", "women vs men 27 rows", "women 3.27 %, men 3.04 %"):
        ok(f"lock declaration: {s_}", s_ in decl)
    present("S23 overlap", "35 rows (women, 0.004\\%) and 29 rows (men, 0.006\\%), women and men share 27 rows, and 3.27\\% (women) and 3.04\\% (men)", s23, "S23")
    ok("review r2: fix-then-lock, no P0/P1, two P2 residues", r2.startswith("Verdict: fix-then-lock") and "no new P0/P1 finding" in r2
       and "Two residual wording errors" in r2)
    ok("review r1: one P1, P2-1, P2-2", r1.startswith("Verdict: fix-then-lock") and "**P1-1" in r1 and "**P2-1" in r1 and "**P2-2" in r1 and "P1-2" not in r1)
    ok("lock records P2 fixes before lock", "P2" in lock_txt and ("residue" in lock_txt or "residual" in lock_txt))
    present("S23 review", "round 2 found no P0 or P1 and returned ``fix-then-lock'' for two residual P2 wording errors", s23, "S23")
    ok("v12_analysis: decide called, verdict discarded for descriptive", "d = decide(ratio, primary_false_streams, replica, thresh)"
       in (IT / "exp/code/dsswm/stats/v12_analysis.py").read_text())
    ok("eval seeds new (plan)", "block A (women) **39200–39399**; block B (men) **39400–39599**" in plan)
    present("S23 seeds", "the evaluation seeds 39200--39399 (women) and 39400--39599 (men) were never used before", s23, "S23")
    present("S23 closing", "the confirmatory Open Bandit claim of Section 6.6 rests on lock v11 alone", s23, "S23")
    present("S23 not pooled", "no ratio is pooled with lock v11 or across campaigns", s23, "S23")
    # S2 and S-list
    present("S2 caption", "(lock v5 through lock v12)", SUPP, "S2")
    present("S2 v12 row", "($\\eps$-optimal development share 0.661 and 0.858 $> 0.5$ at the rule-selected $\\eps = 7.5\\cdot 10^{-4}$ and $10^{-3}$)", SUPP, "S2")
    present("S2 v12 row regeneration", "the development block regenerated once (4,980 rows identical)", SUPP, "S2")
    present("S2 r8 row", "2026-10-05 (r8) & lock-v12 results integrated", SUPP, "S2")
    present("S-list", "and the lock-v12 descriptive replications on two further Open Bandit campaigns (S23).", SUPP, "intro")
    present("lock map", "and its two descriptive replications (Section 6.6, S23) are lock v12.", SUPP, "intro")
    # tables
    p = subprocess.run([sys.executable, str(HERE / "gen_r8_v12_tables.py"), "--check"], capture_output=True, text=True)
    ok("S23 tables equal gen_r8_v12_tables.py output", p.returncode == 0, p.stdout.strip())
    for lab in ("tab:v12ratios", "tab:v12methods", "tab:v12gate", "tab:v12design"):
        ok(f"{lab} input in S23 and referenced", f"\\label{{{lab}}}" in (IT / "writing/supplement/r8_tables").joinpath(
            {"tab:v12ratios": "v12_ratios.tex", "tab:v12methods": "v12_methods.tex", "tab:v12gate": "v12_rule_gate.tex",
             "tab:v12design": "v12_design.tex"}[lab]).read_text() and f"\\ref{{{lab}}}" in s23)


def rules():
    section("Convergence rules on the r8 paragraphs")
    for start in ("Lock v12 replicated", "\\textbf{The fresh-log test.}", "The ordering persisted over",
                  "In two ablations neither", "Lock v12 asks whether"):
        hits = [x for x in (MAIN + "\n\n" + SUPP).split("\n\n") if x.strip().startswith(start)]
        if not hits:
            ok(f"paragraph '{start[:30]}' found", False)
            continue
        para = hits[0].strip()
        ok(f"paragraph '{start[:30]}' ends on words", not re.search(r"(\d\)?|\})\.$", para))
        body = re.sub(r"~?\\cite[pt]?\{[^}]*\}", "", para)
        lens = [len(s.split()) for s in re.split(r"(?<=[.;])\s+", body) if s.strip()]
        if start.startswith("Lock v12 asks"):
            continue
        ok(f"paragraph '{start[:30]}' sentences <= 45 words", max(lens) <= 45, f"max {max(lens)}")
    i = SUPP.index("\\section{Lock v12")
    s23 = SUPP[i:SUPP.index("\\clearpage\n\\end{document}")]
    for p_ in [x.strip() for x in s23.split("\n\n") if x.strip().startswith("\\textbf{")]:
        ok(f"S23 paragraph '{p_[8:40]}' ends on words", not re.search(r"(\d\)?|\})\.$", p_))
    for hedge in ("may potentially", "might possibly", "could potentially"):
        absent("no hedging stack", hedge, MAIN + SUPP, "main+supp")


def page_budget():
    section("Page budget (main.pdf: 12 pages, body ends on page 8, references begin on page 9)")
    pdf = IT / "writing/latex_acm/main.pdf"
    try:
        import pymupdf
    except ImportError:
        print("  [note] pymupdf not installed; page check skipped")
        return
    if not pdf.exists():
        print("  [note] main.pdf not shipped; page check skipped")
        return
    d = pymupdf.open(str(pdf))
    ref = concl = app = None
    for i, p in enumerate(d):
        for b in p.get_text("blocks"):
            s = b[4].replace("\n", " ").strip()
            if s.startswith("REFERENCES") and ref is None:
                ref = (i + 1, b[1])
            if "read plan first." in s:
                concl = (i + 1, b[3])
            if s.startswith("A FULL PROOFS") and app is None:
                app = (i + 1, b[0], b[1])
    ok("main.pdf has 12 pages", len(d) == 12, f"{len(d)}")
    ok("conclusion ends on page 8", bool(concl) and concl[0] == 8, f"p. {concl[0]} y = {concl[1]:.0f}" if concl else "not found")
    ok("references heading on page 9", bool(ref) and ref[0] == 9, f"p. {ref[0]} y = {ref[1]:.0f}" if ref else "not found")
    above = [b for b in d[8].get_text("blocks") if 40 < b[0] < 570 and 75 < b[1] < (ref[1] if ref else 0) - 1 and not b[4].strip().isdigit()]
    ok("no body text above the references heading on page 9", not above)
    ok("appendix A starts on page 9", bool(app) and app[0] == 9, f"p. {app[0]} x = {app[1]:.0f} y = {app[2]:.0f}" if app else "not found")
    txt = "".join(p.get_text() for p in d)
    ok("pdf contains the lock-v12 paragraph (compiled from this main.tex)", "Lock v12 replicated the direction descriptively" in txt)


def main():
    args = [a for a in sys.argv[1:] if a == "--skip-census"]
    r8a_chain(args)
    replacements()
    recompute()
    main_text()
    supp_text()
    rules()
    page_budget()
    print(f"\nr8: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
