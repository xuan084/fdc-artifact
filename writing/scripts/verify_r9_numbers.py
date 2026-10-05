"""Check paper revision r9 (fixes after the independent r9 review of r8d) against r8d (read-only).

0. Re-runs verify_r8d on the r8d state (main_pre_r9.tex, supplement_pre_r9.tex, main_r8d.pdf) in a temporary copy of
   the package; it in turn re-runs verify_r8b and verify_r8c on their own texts. It must end with 0 mismatches.
1. main_pre_r9.tex / supplement_pre_r9.tex must equal the r8d files committed at ffaf65af (by sha256).
2. Numbers in the paper: every decimal removed from the main text still appears in the paper or the supplement (text
   moved, not changed); every decimal added appears in the r8d paper or supplement, or in the decision-difficulty
   fragment; the full-frontier and census numbers quoted in Sections 6.6-6.7 are those of S21 and S12.
3. Decision difficulty: the record r9_difficulty.json reproduced every published diagnostic, and both LaTeX fragments
   (paper Table 3, supplement S25) re-render byte-identically from it.
4. Supplement: only the listed numeric tokens were added (new S25, the r9 deviation row, the split table's column
   widths); the deviations table (now four floats) keeps every row of r8d verbatim.
5. Required r9 statements present, superseded statements absent.
6. PDFs: main 12 pages, conclusion on page 8, references from page 9; no text outside any page of either PDF;
   every deviation row is visible in the supplement PDF; sentence rules.
"""
import collections
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

IT = pathlib.Path(__file__).resolve().parents[2]
LA = IT / "writing/latex_acm"
SUP = IT / "writing/supplement"
FAIL = []


def ok(label, cond, detail=""):
    print(f"  [{'OK' if cond else 'MISMATCH'}] {label}{(': ' + str(detail)) if detail else ''}")
    if not cond:
        FAIL.append(label)


# sha256 of the r8d files committed at ffaf65af in the authors' workspace (the anonymous artifact has its own history)
SHA = {"main": "46f26db2e51c5f1b1d1d851be53d8568ef08b69ca2f7364d6e78840634fcab9f",
       "supp": "4c1f9e2ffec6484fc629a7e220390080dac4af7303449f88bd3e4db73116b975",
       "pdf": "03ee3355245c305ffa57a26de4b39f97f0646b198aaf9e3148125d02a213343e"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def nums(t):
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)
    return collections.Counter(re.findall(r"\d+(?:[.,{}]\d+)*", t))


def dec(c):
    return {k for k in c if "." in k}


def chain(script, main_src, supp_src, pdf_src, extra_args=()):
    root = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9_"))
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
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        return lines[-1] if lines else r.stderr[-300:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


print("== 0. r8d on its own text (chains r8b and r8c)")
last = chain("verify_r8d_numbers.py", LA / "main_pre_r9.tex", SUP / "supplement_pre_r9.tex", LA / "main_r8d.pdf")
ok("verify_r8d on the r8d state", last.startswith("r8d: 0 mismatch"), last[:200])

print("== 1. r8d state")
PM, PS = (LA / "main_pre_r9.tex").read_text(), (SUP / "supplement_pre_r9.tex").read_text()
ok("main_pre_r9.tex == r8d main.tex (ffaf65af)", sha(PM.encode()) == SHA["main"])
ok("supplement_pre_r9.tex == r8d supplement.tex (ffaf65af)", sha(PS.encode()) == SHA["supp"])
ok("main_r8d.pdf == r8d main.pdf (ffaf65af)", sha((LA / "main_r8d.pdf").read_bytes()) == SHA["pdf"])

M0 = (LA / "main.tex").read_text()
FRAG = (LA / "r9_difficulty_main.tex").read_text()
M = M0.replace("\\input{r9_difficulty_main.tex}", FRAG)
S0 = (SUP / "supplement.tex").read_text()
SFRAG = (SUP / "r9_tables/difficulty.tex").read_text()
S = S0.replace("\\input{r9_tables/difficulty.tex}", SFRAG)

print("== 2. numbers in the paper")
rm, add = nums(PM) - nums(M), nums(M) - nums(PM)
print(f"     main removed: {dict(rm)}")
print(f"     main added:   {dict(add)}")
miss = sorted(k for k in dec(rm) if k not in nums(M) and k not in nums(S))
ok("every removed decimal is still in the paper or the supplement", not miss, miss)
LAYOUT = {"0.72"}  # Figure 1 width 0.72\columnwidth (layout, not a result)
src = nums(PM) + nums(PS) + nums(FRAG)
new = sorted(k for k in dec(add) if k not in src and k not in LAYOUT)
ok("every added decimal comes from the r8d paper/supplement or the difficulty fragment", not new, new)
sentence = "At the full frontier (15 of 15, secondary) the ratio is 0.853 (UB95 0.859), but the rectangle reaches 15 of 15 only at $\\tau_R$ on 160 of 200 streams"
ok("6.6 full-frontier sentence present", sentence in M)
ok("0.853 / 0.859 / 160 of 200 match S21", "the ratio is 0.853 (UB95 0" in PS and "859), but the rectangle reaches 15 of 15 only at $\\tau_R$ on 160 of 200 streams" in PS)
ok("6.7 census 3,076 of 3,347 matches S12", "3,076 of 3,347" in PS and "3,076 of 3,347 (problem, policy) pairs are" in M)
for v, where in (("0.169", "0.169 [0.166, 0.171]"), ("0.191", "0.191 [0.188, 0.193]"), ("0.530", "\\textbf{0.530}"), ("0.708", "\\textbf{0.708}")):
    ok(f"6.7 quotes {v} as in the r8d tables", where in PM)

print("== 3. decision difficulty")
J = json.loads((IT / "writing/r9_difficulty.json").read_text())
ok("difficulty record: all published diagnostics reproduced", J["all_reproduced"] and all(r["match"] for r in J["reproduction_checks"]),
   [r["quantity"] for r in J["reproduction_checks"] if not r["match"]])
spec = importlib.util.spec_from_file_location("difficulty_r9", IT / "writing/scripts/difficulty_r9.py")
dmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dmod)
tmp = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9_tex_"))
try:
    (tmp / "writing/latex_acm").mkdir(parents=True)
    (tmp / "writing/supplement/r9_tables").mkdir(parents=True)
    dmod.IT = tmp
    dmod.write_tex(J)
    ok("paper Table 3 fragment re-renders from the json", (tmp / "writing/latex_acm/r9_difficulty_main.tex").read_text() == FRAG)
    ok("supplement S25 table re-renders from the json", (tmp / "writing/supplement/r9_tables/difficulty.tex").read_text() == SFRAG)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
by = {b["id"]: b for b in J["blocks"]}
ok("v12 shares in the record equal S23 (0.661 / 0.858)", round(by["OBD-W"]["dev"]["mean_share"], 3) == 0.661 and round(by["OBD-M"]["dev"]["mean_share"], 3) == 0.858)

print("== 4. supplement")
s_rm, s_add = nums(PS) - nums(S), nums(S) - nums(PS)
print(f"     supp removed: {dict(s_rm)}")
ok("supp: only one S20 cross-reference removed (S22 sentence replaced by its correction)", s_rm == collections.Counter({"20": 1}), dict(s_rm))
new_s = sorted(k for k in dec(s_add) if k not in nums(PS) + nums(M) + nums(SFRAG) and k not in {"6.6", "7.6", "0.99977", "1.01", "1.49", "0.142", "0.14", "0.66", "0.86", "0.85", "1.00"})
ok("supp: added decimals are S25 values (json), column widths, or already in the papers", not new_s, new_s)
for k, v in (("0.142", "OBD"), ("1.01", "X5-16"), ("1.49", "X5-64")):
    blk = by[v]["dev"]
    val = blk["mean_share"] if k == "0.142" else blk["eps_rel_headroom"]
    ok(f"S25 text {k} = record {v}", f"{val:.3f}".rstrip("0").rstrip(".") == k or f"{val:.2f}" == k, val)
rows_pre = [r.strip() for r in PS[PS.index("\\begin{table*}[t]\n\\caption{\\textbf{Dated deviations}"):].split("\\bottomrule")[0].split("\\midrule\n", 1)[1].split("\\\\\n") if r.strip()]
ok("every r8d deviation row is still in the supplement verbatim", all(r in S for r in rows_pre), sum(r not in S for r in rows_pre))
ok("deviations table is split into four floats", S.count("\\caption{\\textbf{Dated deviations}, continued (part") == 3)

print("== 5. required and superseded statements")
for need in ("It certifies 12 of the 15 problems with 0.30--0.82 times",
             "the stratified value an analyst reading every outcome of the half would compute",
             "so it does not bound causal deployment regret",
             "No false certificate was observed in these replay runs.",
             "It assumes that every row's segment and arm are known before matching",
             "\\label{tab:difficulty}"):
    ok(f"main has: {need[:60]}", need in M)
for gone in ("certify a whole budget frontier", "FDC-BF met its guarantee on replay", "such as a retrospective decision about the logged users"):
    ok(f"main lacks: {gone}", gone not in M)
ok("supp: S22 superseded sentence gone", "[r7b correction:" not in S and "Three localised union ledgers read no fewer rows on development halves" not in S)
ok("supp: workflow assumptions listed", "It assumes (a) that every row's segment and arm" in S)
ok("supp: S25 present", "\\label{app:difficulty}" in S)

print("== 6. PDFs and sentence rules")
import pymupdf  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from offpage_check import offpage  # noqa: E402
d = pymupdf.open(LA / "main.pdf")
ok("main.pdf has 12 pages", len(d) == 12, len(d))
ok("conclusion on page 8", "by fixing its read plan first." in " ".join(d[7].get_text().split()))
lines9 = [l for l in d[8].get_text().split("\n") if len(l) > 3 and not l.isdigit()]
ok("references head page 9", "REFERENCES" in lines9[:3], lines9[:3])
ok("main.pdf compiled from this main.tex", "No false certificate was observed in these replay runs." in " ".join(" ".join(p.get_text().split()) for p in d))
ok("main.pdf: no text outside the page", not offpage(LA / "main.pdf"), offpage(LA / "main.pdf"))
ok("supplement.pdf: no text outside the page", not offpage(SUP / "supplement.pdf"), offpage(SUP / "supplement.pdf"))
sp = " ".join(" ".join(p.get_text().split()) for p in pymupdf.open(SUP / "supplement.pdf"))
ok("supplement.pdf shows the last deviation rows and S25", all(k in sp for k in ("2026-10-06 (r9)", "by design", "Decision difficulty of every reported block")))
import verify_r5_numbers as v5  # noqa: E402
v5.MAIN = M0
lens = [len(x.split()) for x in v5.body_text()]
ok("no body sentence over 45 words", max(lens) <= 45, max(lens))
ok("body mean sentence length <= 25", sum(lens) / len(lens) <= 25, round(sum(lens) / len(lens), 1))

print(f"\nr9: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
