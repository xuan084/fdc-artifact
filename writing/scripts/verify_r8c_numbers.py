"""Check paper revision r8c (sourced motivation) against r8b (read-only).

r8c changes wording and citations only. This script checks that:
1. main_pre_r8c.tex / supplement_pre_r8c.tex equal the r8b files committed at HEAD~ (d3a34a1d), which verify_r8b passed
   with 0 mismatches;
2. no registered number changed: the multiset of numeric tokens of the r8c files differs from r8b only by the listed
   removed and added tokens (prices, sources, dates, and text moved within the paper or to the supplement);
3. the abstract and claims C1-C4 are unchanged;
4. removed unsourced statements are absent, new citations are present and defined in references.bib, and every new
   bib key is one verified in writing/motivation_sources.md;
5. the S16 dollar figures follow from the sealed reads saved (505,304 and 42,708 records);
6. the page budget (12 pages, body ends on page 8) and the sentence rules hold.
"""
import collections
import pathlib
import re
import subprocess
import sys

IT = pathlib.Path(__file__).resolve().parents[2]
LA = IT / "writing/latex_acm"
SUP = IT / "writing/supplement"
MAIN, PRE = (LA / "main.tex").read_text(), (LA / "main_pre_r8c.tex").read_text()
SUPP, SPRE = (SUP / "supplement.tex").read_text(), (SUP / "supplement_pre_r8c.tex").read_text()
BIB = (LA / "references.bib").read_text()
SRC = (IT / "writing/motivation_sources.md").read_text()
FAIL = []


def ok(label, cond, detail=""):
    print(f"  [{'OK' if cond else 'MISMATCH'}] {label}{(': ' + str(detail)) if detail else ''}")
    if not cond:
        FAIL.append(label)


# sha256 of the files committed at d3a34a1d in the authors' workspace (the anonymous artifact has its own history)
SHA = {"writing/latex_acm/main.tex": "ce91d87f901d523f4cb954b76aeb27c228d4eda3381ab81060ec14c63ec8909b",
       "writing/supplement/supplement.tex": "ca312320dcaaa3018ef178601aa96279f5398d2fbd542306541ac5963a42332b"}


def same_as_committed(text, path):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest() == SHA[path]


def nums(t):
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)  # citation keys carry years, not results
    return collections.Counter(re.findall(r"\d+(?:[.,{}]\d+)*", t))


def grab(t, a, b):
    return t[t.index(a):t.index(b, t.index(a))]


print("== 1. pre-r8c copies are the r8b files (verify_r8b: 0 mismatches at d3a34a1d)")
ok("main_pre_r8c.tex == r8b main.tex", same_as_committed(PRE, "writing/latex_acm/main.tex"))
ok("supplement_pre_r8c.tex == r8b supplement.tex", same_as_committed(SPRE, "writing/supplement/supplement.tex"))

print("== 2. numeric tokens: only listed removals and additions")
body_now = MAIN[:MAIN.index("\\bibliography{")]
body_pre = PRE[:PRE.index("\\bibliography{")]
# tables moved to the supplement verbatim: compare main + moved block
moved = PRE[PRE.index("\\section{Mechanism and method glossary}"):PRE.index("\\end{document}")]
d_main_removed = nums(body_pre) - nums(body_now)
d_main_added = nums(body_now) - nums(body_pre)
ALLOWED_MAIN_REMOVED = nums(moved) + collections.Counter({
    # C1: dropped the sentence naming the blocks (locks v7, v8 at 20 checkpoints, v9 at 77 times; Table 1 lists them)
    "7": 1, "8": 1, "9": 1, "20": 2, "77": 1,
    # protocol: worst-case PJC development UB95 1.042 at eps = 0.01 on 20 seeds (kept in S14)
    "1.042": 1, "0.01": 1,
    # 6.5: pointer "(Figure S3 in S19)" dropped; S19 is still cited in the same paragraph
    "3": 1, "19": 1,
    # 6.2: "Ten Criteo-sized campaigns ... \\$10k a month" (unsourced schedule, kept as hypothetical in S16)
    "10": 2,
    # conclusion merged ranges: 0.30--0.71 and 0.17--0.63 -> 0.17--0.71 (both ends already in Sec. 6), 16--64 -> 9--64
    "0.30": 1, "0.63": 1, "16": 1,
    # "X5" in the dropped C1 sentence; "\\ref{sec:c2}" in the replaced Sec. 2 clean-room clause
    "5": 1, "2": 1,
})
ALLOWED_MAIN_ADDED = collections.Counter({
    "0.50": 1, "1,000": 1,  # AWS Clean Rooms price, now also in Sec. 2
    "0.04": 1,              # SageMaker Ground Truth fee per object
    "2026": 1, "10": 1, "05": 1,  # access date 2026-10-05 in Sec. 6.2
    "80": 1, "50": 2,       # N_{80} and "50/50" in the replay-limitation sentence
    "24": 1, "14": 1,       # supplement S24 (moved tables) and S14 (threshold sizing)
    "9": 1, "64": 1,        # conclusion: 9--64 segments
    "0.82": 1,              # Figure 1 width 0.82\\columnwidth (layout, not a result)
})
extra_rm = d_main_removed - ALLOWED_MAIN_REMOVED
extra_add = d_main_added - ALLOWED_MAIN_ADDED
print(f"     main removed: {dict(d_main_removed)}")
print(f"     main added:   {dict(d_main_added)}")
ok("main: no unlisted number removed", not extra_rm, dict(extra_rm))
ok("main: no unlisted number added", not extra_add, dict(extra_add))
ok("main: every removed table number reappears in the supplement", not (nums(moved) - nums(SUPP)))
s_rm = nums(SPRE) - nums(SUPP)
s_add = nums(SUPP) - nums(SPRE) - nums(moved)
print(f"     supp removed: {dict(s_rm)}")
print(f"     supp added (beyond moved tables): {dict(s_add)}")
ALLOWED_SUPP_REMOVED = collections.Counter({"0.26": 2, "131": 1, "11.1": 1, "2019": 1, "2017": 0, "2021": 0, "1": 0, "3": 1})
ok("supp: only the $0.26 label row and the 2019 source removed", not (s_rm - ALLOWED_SUPP_REMOVED), dict(s_rm - ALLOWED_SUPP_REMOVED))

print("== 3. abstract and claims unchanged")
ok("abstract unchanged", grab(MAIN, "\\begin{abstract}", "\\end{abstract}") == grab(PRE, "\\begin{abstract}", "\\end{abstract}"))
ok("claims C1-C4 unchanged", grab(MAIN, "\\item[\\textbf{C1}]", "\\noindent Our contributions") ==
   grab(PRE, "\\item[\\textbf{C1}]", "\\noindent Our contributions"))

print("== 4. removed statements, new citations, bib keys")
for gone in ("an audit may require", "Ten Criteo-sized campaigns", "set only after the analysis of a large randomized log",
             "an unread record spends no budget"):
    ok(f"main: absent '{gone}'", gone not in MAIN)
for gone in ("Human labels, \\$0.26 each", "No public per-record price exists"):
    ok(f"supp: absent '{gone}'", gone not in SUPP)
ok("supp: $10k schedule labelled hypothetical", "In a hypothetical schedule (an assumption, not a sourced figure)" in SUPP)
ok("main: 50/50 replay is not a prefix of the logging order", "A 50/50 replay is not a prefix of the original logging order" in MAIN)
ok("main: 'a platform that wants to bound the risk'", "a platform that wants to bound the risk of that decision" in MAIN)
new_keys = ["schultzberg2024riskaware", "zhao2019uplift", "ai2022lbcf", "goldenberg2020freelunch", "ebadi2015personal",
            "awssagemaker2026pricing", "lindon2026anytime", "schultzberg2023sequential"]
for k in new_keys:
    ok(f"cited and defined: {k}", (k in MAIN) and (f"{{{k}," in BIB))
    ok(f"verified in motivation_sources.md: {k}", f"`{k}`" in SRC)
cited = set(x.strip() for c in re.findall(r"\\cite[pt]?\{([^}]*)\}", MAIN) for x in c.split(","))
undefined = sorted(k for k in cited if f"{{{k}," not in BIB)
ok("every cited key is defined", not undefined, undefined)

print("== 5. S16 arithmetic")
for recs, price, shown in ((505304, 0.50e-3, "\\$253"), (42708, 0.50e-3, "\\$21"), (505304, 0.25e-3, "\\$126"),
                           (42708, 0.25e-3, "\\$11"), (505304, 0.04, "\\$20.2k"), (42708, 0.04, "\\$1.7k")):
    v = recs * price
    txt = f"{v/1000:.1f}k" if v >= 1000 else f"{v:.0f}"
    ok(f"{recs} x {price} = {v:.2f} printed {shown}", shown.endswith(txt) and shown in SUPP)

print("== 6. page budget and sentence rules")
try:
    import pymupdf
    d = pymupdf.open(LA / "main.pdf")
    ok("main.pdf has 12 pages", len(d) == 12, len(d))
    p8 = d[7].get_text()
    ok("conclusion on page 8", "LIMITATIONS AND CONCLUSION" in p8 and "by fixing its read plan first." in p8.replace("\n", " "))
    lines9 = [l for l in d[8].get_text().split("\n") if len(l) > 3 and not l.isdigit()]
    ok("references head page 9", lines9[2] == "REFERENCES", lines9[2])
    ok("pdf compiled from this main.tex", "A 50/50 replay is not a prefix" in " ".join(" ".join(p.get_text().split()) for p in d[:3]))
except ImportError:
    FAIL.append("pymupdf missing")
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import verify_r5_numbers as v5  # noqa: E402
v5.MAIN = MAIN
s = v5.body_text()
lens = [len(x.split()) for x in s]
ok("no body sentence over 45 words", max(lens) <= 45, max(lens))
ok("body mean sentence length <= 25", sum(lens) / len(lens) <= 25, round(sum(lens) / len(lens), 1))

print(f"\nr8c: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
