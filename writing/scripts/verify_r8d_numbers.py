"""Check paper revision r8d (P2 wording fixes from the two r8c reviews) against r8c (read-only).

0. Re-runs the whole chain on the earlier texts in a temporary copy of the package: verify_r8b on the r8b state
   (main_pre_r8c.tex, supplement_pre_r8c.tex, main_r8b.pdf) and verify_r8c on the r8c state (main_pre_r8d.tex,
   supplement_pre_r8d.tex, main_r8c.pdf); each must end with 0 mismatches.
1. main_pre_r8d.tex / supplement_pre_r8d.tex must equal the r8c files committed at 25745ff4 (by sha256).
2. Rebuilds main.tex and supplement.tex from the *_pre_r8d copies by applying exactly the r8d replacements below and
   checks the result equals the current files, so nothing else changed.
3. Checks that the only numeric tokens added are the Ground Truth tier (50k--1M), the lock names v7--v9, and the
   section reference, and that none was removed.
4. Page budget (12 pages, body ends on page 8) and sentence rules.
"""
import collections
import pathlib
import re
import subprocess
import sys

IT = pathlib.Path(__file__).resolve().parents[2]
LA = IT / "writing/latex_acm"
SUP = IT / "writing/supplement"
FAIL = []


def ok(label, cond, detail=""):
    print(f"  [{'OK' if cond else 'MISMATCH'}] {label}{(': ' + str(detail)) if detail else ''}")
    if not cond:
        FAIL.append(label)


# sha256 of the files committed at 25745ff4 in the authors' workspace (the anonymous artifact has its own history)
SHA = {"writing/latex_acm/main.tex": "e05666aeb1c4e26fcf1dfa7243574dcb356e979000dbded555457a23efb40c67",
       "writing/supplement/supplement.tex": "68f79e627b6301803786f30ec66009e5325b17618f424aff1611979809a853a5"}


def same_as_committed(text, path):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest() == SHA[path]


MAIN_REPL = [
    ("must choose which budget-constrained policy to deploy, and a platform that wants to bound the risk of that decision, as experimentation platforms do for ship decisions~\\citep{schultzberg2024riskaware}, needs an error guarantee rather than a point ranking.",
     "must choose which budget-constrained policy to deploy and, if it wants to bound the risk of that decision, as experimentation platforms do for ship decisions~\\citep{schultzberg2024riskaware}, needs an error guarantee rather than a point ranking."),
    ("and the treated fraction may be fixed or recalibrated only after the analysis, as in Booking.com's ROI-constrained promotions~\\citep{goldenberg2020freelunch},",
     "and the treated fraction may be recalibrated after deployment, as Booking.com does online for ROI-constrained promotions~\\citep{goldenberg2020freelunch},"),
    ("under individual privacy accounting, where a record left unread by an outcome-independent plan incurs no privacy loss~\\citep{ebadi2015personal,feldman2021individual};",
     "under individual privacy accounting, which can make leaving outcomes unaccessed valuable~\\citep{ebadi2015personal,feldman2021individual};"),
    ("e.g.\\ \\$0.04 per object reviewed in SageMaker Ground Truth~\\citep{awssagemaker2026pricing}",
     "e.g.\\ a SageMaker Ground Truth fee of \\$0.04 per object (50k--1M tier, labour excluded)~\\citep{awssagemaker2026pricing}"),
    ("such as an audit of the observed-pool ranking of budgeted policies;", "such as a retrospective decision about the logged users;"),
    ("FDC-BF met its guarantee on replay. In each registered block,",
     "FDC-BF met its guarantee on replay. In each registered FDC-BF block (locks v7--v9),"),
    ("The advertiser uploads (segment, arm) exposure keys,",
     "The following workflow is our proposal, not run on a clean room. The advertiser uploads (segment, arm) exposure keys,"),
    ("faster on every stream; no method made a false certificate or had an exhausted pool at $N_{80}$.",
     "faster on every stream. No method made a false certificate; none had an exhausted pool at $N_{80}$."),
    ("costs are analogous list prices, and outcomes are binary.",
     "costs are analogous list prices, replay reads count outcomes revealed rather than users enrolled (\\S\\ref{sec:setting}), and outcomes are binary."),
]
SUPP_REPL = [
    ("and every unread record keeps its share: under individual privacy accounting a record left out of an analysis incurs no privacy loss when the choice to leave it out does not depend on its data (Ebadi, Sands and Schneider, POPL 2015; Feldman and Zrnic, NeurIPS 2021), which the outcome-free frozen plan satisfies.",
     "and outcomes that are never accessed can be worth keeping unaccessed under individual privacy accounting (Ebadi, Sands and Schneider, POPL 2015; Feldman and Zrnic, NeurIPS 2021). The certifier still reads every row's segment and arm, so this row counts outcomes never accessed, not whole records outside an analysis; a saving of privacy budget requires a privacy model in which only outcome accesses are charged, which we do not formalise."),
    ("Records with untouched privacy budget & +7.2\\% & +42.5\\% \\\\", "Additional share of outcomes never accessed & +7.2\\% & +42.5\\% \\\\"),
    ("Privacy row: extra share of the half never read.", "Last row: additional share of the half's outcomes never accessed."),
]


def nums(t):
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)
    return collections.Counter(re.findall(r"\d+(?:[.,{}]\d+)*", t))


def chain(script, main_src, supp_src, pdf_src, extra_args=()):
    """Run an earlier verify script on an earlier text state, in a temporary copy of the package."""
    import os
    import shutil
    import tempfile
    root = pathlib.Path(tempfile.mkdtemp(prefix="verify_r8d_"))
    try:
        for e in IT.iterdir():
            if e.name != "writing":
                os.symlink(e, root / e.name)
        (root / "writing").mkdir()
        for e in (IT / "writing").iterdir():
            if e.name not in ("latex_acm", "supplement", "scripts"):
                os.symlink(e, root / "writing" / e.name)
        shutil.copytree(IT / "writing/scripts", root / "writing/scripts", ignore=shutil.ignore_patterns("__pycache__"))
        for sub, overrides in (("latex_acm", {"main.tex": main_src, "main.pdf": pdf_src}),
                               ("supplement", {"supplement.tex": supp_src})):
            (root / "writing" / sub).mkdir()
            for e in (IT / "writing" / sub).iterdir():
                if e.name in overrides:
                    shutil.copy(overrides[e.name], root / "writing" / sub / e.name)
                else:
                    os.symlink(e, root / "writing" / sub / e.name)
        r = subprocess.run([sys.executable, str(root / "writing/scripts" / script), *extra_args], cwd=root,
                           capture_output=True, text=True)
        last = [l for l in r.stdout.strip().split("\n") if l.strip()][-1] if r.stdout.strip() else r.stderr[-300:]
        return last
    finally:
        shutil.rmtree(root, ignore_errors=True)


print("== 0. earlier revisions on their own texts")
last = chain("verify_r8b_numbers.py", LA / "main_pre_r8c.tex", SUP / "supplement_pre_r8c.tex", LA / "main_r8b.pdf",
             ("--skip-census",))
ok("verify_r8b on the r8b state", last.startswith("r8b: 0 mismatch"), last[:200])
last = chain("verify_r8c_numbers.py", LA / "main_pre_r8d.tex", SUP / "supplement_pre_r8d.tex", LA / "main_r8c.pdf")
ok("verify_r8c on the r8c state", last.startswith("r8c: 0 mismatch"), last[:200])

print("== 1. r8c state")
pre_m, pre_s = (LA / "main_pre_r8d.tex").read_text(), (SUP / "supplement_pre_r8d.tex").read_text()
ok("main_pre_r8d.tex == r8c main.tex (25745ff4)", same_as_committed(pre_m, "writing/latex_acm/main.tex"))
ok("supplement_pre_r8d.tex == r8c supplement.tex (25745ff4)", same_as_committed(pre_s, "writing/supplement/supplement.tex"))

print("== 2. pre-r8d + r8d replacements == current files")
for label, pre, path, repl in (("main", pre_m, LA / "main.tex", MAIN_REPL), ("supp", pre_s, SUP / "supplement.tex", SUPP_REPL)):
    t = pre
    for a, b in repl:
        ok(f"{label}: replacement source unique: {a[:50]!r}", t.count(a) == 1)
        t = t.replace(a, b)
    ok(f"{label}: rebuilt == current", t == path.read_text())

print("== 3. numeric tokens")
now_m = (LA / "main.tex").read_text()
now_s = (SUP / "supplement.tex").read_text()
rm, add = nums(pre_m) - nums(now_m), nums(now_m) - nums(pre_m)
ok("main: no number removed", not rm, dict(rm))
ok("main: only tier 50k--1M and locks v7--v9 added", not (add - collections.Counter({"50": 1, "1": 1, "7": 1, "9": 1})), dict(add))
rm_s, add_s = nums(pre_s) - nums(now_s), nums(now_s) - nums(pre_s)
ok("supp: no number removed or added", not rm_s and not add_s, (dict(rm_s), dict(add_s)))
ok("v12 sentence restored to the r8b wording", "No method made a false certificate; none had an exhausted pool at $N_{80}$." in now_m)

print("== 4. page budget and sentence rules")
import pymupdf  # noqa: E402
d = pymupdf.open(LA / "main.pdf")
ok("main.pdf has 12 pages", len(d) == 12, len(d))
ok("conclusion on page 8", "by fixing its read plan first." in d[7].get_text().replace("\n", " "))
lines9 = [l for l in d[8].get_text().split("\n") if len(l) > 3 and not l.isdigit()]
ok("references head page 9", lines9[2] == "REFERENCES", lines9[2])
ok("pdf compiled from this main.tex", "recalibrated after deployment" in " ".join(" ".join(p.get_text().split()) for p in d[:2]))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import verify_r5_numbers as v5  # noqa: E402
v5.MAIN = now_m
lens = [len(x.split()) for x in v5.body_text()]
ok("no body sentence over 45 words", max(lens) <= 45, max(lens))
ok("body mean sentence length <= 25", sum(lens) / len(lens) <= 25, round(sum(lens) / len(lens), 1))

print(f"\nr8d: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
