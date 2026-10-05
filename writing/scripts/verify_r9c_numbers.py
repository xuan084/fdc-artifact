"""Check paper revision r9c (fixes after the two r9 reviews) against r9 (read-only).

0. Re-runs verify_r9 on the r9 state (main_pre_r9c.tex, supplement_pre_r9c.tex, main_r9.pdf) in a temporary copy of
   the package; it in turn re-runs verify_r8d, verify_r8c and verify_r8b. It must end with 0 mismatches.
1. main_pre_r9c.tex / supplement_pre_r9c.tex / main_r9.pdf equal the r9 files committed at e1835f1f (by sha256).
2. Numbers in the paper: every decimal removed from the main text is still in the paper or the supplement (Table 1's
   checkpoint panel became one sentence in Section 6.2); every decimal added comes from the r9 paper or supplement,
   the census record, or the block-G census record.
3. The r9 census outputs are unchanged (sha256), its evaluation counts recompute from the row files present
   (difficulty_r9.py --check-eval, plain or gzipped rows), and the block-G census table and summary re-render
   byte-identically from their record, which reproduces the r9 census at each lock tolerance.
4. Supplement: only the new block-G paragraph and table, the checkpoint-grid sentence moved from the paper and the
   corrected headroom wording add numbers; nothing else is removed.
5. Required r9c statements present, superseded statements absent.
6. PDFs: main 12 pages, conclusion on page 8, no text outside any page of either PDF; sentence rules.
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


# sha256 of the r9 files committed at e1835f1f in the authors' workspace (the anonymous artifact has its own history)
SHA = {"main": "4d9a95f612481ff551721c6e21b9125dc1458923d024e07f284affa1126c3bd5",
       "supp": "53ceb5e7b52e4045076d2eabb470c19348166cebf3b69dfbe2decff48d39fac0",
       "pdf": "eb1a0c53d759fa0452e2d754b0fb0e742d3d6450648566a57383c57fcbdb8883"}
# sha256 prefixes of the r9 census outputs (unchanged by r9c)
CENSUS = {"writing/r9_difficulty.json": "af447731", "writing/supplement/r9_tables/difficulty.tex": "1d0a0ea9",
          "writing/latex_acm/r9_difficulty_main.tex": "0b0c2e69"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def nums(t):
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)
    return collections.Counter(re.findall(r"\d+(?:[.,{}]\d+)*", t))


def dec(c):
    return {k for k in c if "." in k}


def chain(script, main_src, supp_src, pdf_src):
    root = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9c_"))
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
        r = subprocess.run([sys.executable, str(root / "writing/scripts" / script)], cwd=root, capture_output=True, text=True)
        lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
        return lines[-1] if lines else r.stderr[-300:]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def load(name):
    spec = importlib.util.spec_from_file_location(name, IT / "writing/scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


print("== 0. r9 on its own text (chains r8d, r8c, r8b)")
last = chain("verify_r9_numbers.py", LA / "main_pre_r9c.tex", SUP / "supplement_pre_r9c.tex", LA / "main_r9.pdf")
ok("verify_r9 on the r9 state", last.startswith("r9: 0 mismatch"), last[:200])

print("== 1. r9 state")
PM, PS = (LA / "main_pre_r9c.tex").read_text(), (SUP / "supplement_pre_r9c.tex").read_text()
ok("main_pre_r9c.tex == r9 main.tex (e1835f1f)", sha(PM.encode()) == SHA["main"])
ok("supplement_pre_r9c.tex == r9 supplement.tex (e1835f1f)", sha(PS.encode()) == SHA["supp"])
ok("main_r9.pdf == r9 main.pdf (e1835f1f)", sha((LA / "main_r9.pdf").read_bytes()) == SHA["pdf"])

FRAG = (LA / "r9_difficulty_main.tex").read_text()
SFRAG = (SUP / "r9_tables/difficulty.tex").read_text()
GFRAG = (SUP / "r9_tables/blockG_difficulty.tex").read_text()
PM_full = PM.replace("\\input{r9_difficulty_main.tex}", FRAG)
M0 = (LA / "main.tex").read_text()
M = M0.replace("\\input{r9_difficulty_main.tex}", FRAG)
PS_full = PS.replace("\\input{r9_tables/difficulty.tex}", SFRAG)
S0 = (SUP / "supplement.tex").read_text()
S = S0.replace("\\input{r9_tables/difficulty.tex}", SFRAG).replace("\\input{r9_tables/blockG_difficulty.tex}", GFRAG)

print("== 2. numbers in the paper")
rm, add = nums(PM_full) - nums(M), nums(M) - nums(PM_full)
print(f"     main removed: {dict(rm)}")
print(f"     main added:   {dict(add)}")
LAYOUT = {"0.64", "0.72"}  # Figure 1 width 0.72 -> 0.64\columnwidth (layout, not a result)
miss = sorted(k for k in dec(rm) if k not in nums(M) and k not in nums(S) and k not in LAYOUT)
ok("every removed decimal is still in the paper or the supplement", not miss, miss)
src = nums(PM_full) + nums(PS_full) + nums(FRAG) + nums(GFRAG)
new = sorted(k for k in dec(add) if k not in src and k not in LAYOUT)
ok("every added decimal comes from the r9 texts or the census records", not new, new)
for k in ("0.643", "0.655", "0.658", "0.670", "0.364", "0.362", "0.368", "0.367", "1.003", "0.968", "1.011", "0.978"):
    ok(f"checkpoint-test value {k} kept in Section 6.2", k in M[M.index("The checkpoint tests (locks v7, v8"):M.index("The secondary endpoints give")])

print("== 3. census records")
for f, pre in CENSUS.items():
    ok(f"r9 census output unchanged: {f}", sha((IT / f).read_bytes()).startswith(pre))
r = subprocess.run([sys.executable, str(IT / "writing/scripts/difficulty_r9.py"), "--check-eval"], cwd=IT,
                   capture_output=True, text=True)
ok("difficulty_r9.py --check-eval: evaluation counts recompute from the row files", r.returncode == 0, (r.stdout + r.stderr).strip()[-200:])
G = load("difficulty_blockG_r9c")
GJ = json.loads((IT / "writing/r9c_blockG_difficulty.json").read_text())
tmp = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9c_tex_"))
try:
    (tmp / "writing/supplement/r9_tables").mkdir(parents=True)
    G.IT = tmp
    G.render(GJ)
    ok("block-G table re-renders from its record", (tmp / "writing/supplement/r9_tables/blockG_difficulty.tex").read_text() == GFRAG)
    ok("block-G summary re-renders from its record", (tmp / "writing/r9c_blockG_difficulty.md").read_text() == (IT / "writing/r9c_blockG_difficulty.md").read_text())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
for v in ("0.129", "0.314", "0.353", "0.798", "0.641"):
    ok(f"block-G value {v} in the table and the S25 text", v in GFRAG and (v in S0 or v[:-1] in S0 or f"{round(float(v)*100)}\\%" in S0))

print("== 4. supplement")
s_rm, s_add = nums(PS_full) - nums(S), nums(S) - nums(PS_full)
print(f"     supp removed: {dict(s_rm)}")
ok("supp: nothing removed", not s_rm, dict(s_rm))
new_s = sorted(k for k in dec(s_add) if k not in nums(PS_full) + nums(PM_full) + nums(GFRAG) + nums(M))
ok("supp: added decimals come from the papers or the block-G table", not new_s, new_s)
ok("supp: checkpoint-grid factors moved from the paper", "differ by a factor of 1.297 on Criteo and 1.229 on X5" in S and "differ by a factor of 1.297" not in M)
ok("supp: 6 of 434,393 matches the census record", json.loads((IT / "writing/r9_difficulty.json").read_text()) and "6 of 434,393 feasible pairs not $\\eps$-optimal at $S = 16$" in S)
J = json.loads((IT / "writing/r9_difficulty.json").read_text())
b16 = [b for b in J["blocks"] if b["id"] == "X5-16"][0]["dev"]
ok("record: X5 S=16 has 434,393 feasible and 434,387 eps-optimal pairs", (b16["pooled_n_feasible"], b16["pooled_n_eps_opt"]) == (434393, 434387))

print("== 5. required and superseded statements")
for need in ("It certifies 12 of the 15 problems with 0.53--0.82 times",
             "whose frontier is near-trivial at the registered tolerance",
             "so only the Criteo and Open Bandit parts test demanding decisions",
             "looser than an earlier round's 0.70, which the observed 0.670 also clears",
             "so this equal-information win needs no shared pilot",
             "unless, in point estimates, 3 to 26 runs share the pilot",
             "compared at checkpoint strength",
             "neither of the two replayed localised ledgers reduced reads",
             "No confirmatory FDC-DP test has a demanding frontier with 16 or more segments",
             "a checkpoint-valid exact rectangle gave 0.822 and 0.826",
             "so all-control is $\\eps$-optimal at every budget",
             "certify an $\\eps$-optimal policy for the observed-pool objective",
             "reuses the union over alternatives~\\citep{deheide2026union}."):
    ok(f"main has: {need[:60]}", need in M)
for gone in ("0.30--0.82", "no tested localised ledger reduced it", "The two other rivals gave", "reach a full-log analysis's decision",
             "\\citep{deheide2026union};"):
    ok(f"main lacks: {gone}", gone not in M)
ok("supp: false headroom implication gone", "above 1, every feasible policy is $\\eps$-optimal" not in S and "times the headroom, so every feasible policy is $\\eps$-optimal" not in S)

print("== 6. PDFs and sentence rules")
import pymupdf  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from offpage_check import offpage  # noqa: E402
d = pymupdf.open(LA / "main.pdf")
ok("main.pdf has 12 pages", len(d) == 12, len(d))
txt = [" ".join(p.get_text().split()) for p in d]
ok("conclusion on page 8", "by fixing its read plan first." in txt[7])
ok("main.pdf compiled from this main.tex", "so only the Criteo and Open Bandit parts test demanding decisions" in " ".join(txt))
ok("main.pdf: no text outside the page", not offpage(LA / "main.pdf"), offpage(LA / "main.pdf"))
ok("supplement.pdf: no text outside the page", not offpage(SUP / "supplement.pdf"), offpage(SUP / "supplement.pdf"))
sp = " ".join(" ".join(p.get_text().split()) for p in pymupdf.open(SUP / "supplement.pdf"))
ok("supplement.pdf shows the block-G census", "Decision difficulty at the block-G tolerances" in sp)
import verify_r5_numbers as v5  # noqa: E402
v5.MAIN = M0
lens = [len(x.split()) for x in v5.body_text()]
ok("no body sentence over 45 words", max(lens) <= 45, max(lens))
ok("body mean sentence length <= 25", sum(lens) / len(lens) <= 25, round(sum(lens) / len(lens), 1))

print(f"\nr9c: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
