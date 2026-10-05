"""Check paper revision r9d (wording fixes after the two r9c reviews) against r9c (read-only).

0. Re-runs verify_r9c on the r9c state (main_pre_r9d.tex, supplement_pre_r9d.tex, main_r9c.pdf) in a temporary copy
   of the package (it chains verify_r9, r8d, r8c and r8b). It must end with 0 mismatches.
1. main_pre_r9d.tex, supplement_pre_r9d.tex, main_r9c.pdf and the r9c block-G record and table equal the files
   committed at 6a0eff67 (by sha256).
2. Rebuilds main.tex and supplement.tex from the r9c copies by exactly the r9d replacements below; the results must
   equal the current files, so nothing else changed.
3. Block-G record: development columns and flags unchanged; every r9c evaluation field unchanged; the added fields
   (FDC-DP's exhausted control and overall fractions and log-N80 spread, the rectangle's exhausted control fraction)
   equal the sealed block-G summary; the table re-renders from the record and differs from r9c only by the new column.
4. Numbers: decimals added to the paper or supplement come from the block-G record or the r9c texts.
5. PDFs: main 12 pages, conclusion on page 8, no text outside any page of either PDF; sentence rules.
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


# sha256 of the r9c files committed at 6a0eff67 in the authors' workspace (the anonymous artifact has its own history)
SHA = {"writing/latex_acm/main_pre_r9d.tex": "41a46cf7e06348c872e614085ad1992658f959aba7cd46555215d02bc33c61eb",
       "writing/supplement/supplement_pre_r9d.tex": "2af41f52706c7c9214cbfd1bb3c92900b4ab2d398981a32e7b536c8fe98687b5",
       "writing/latex_acm/main_r9c.pdf": "da172575cf81a7a0a9db97fe2a93ff6da4a0136f5087847b1a9137099d6d52ec",
       "writing/r9c_blockG_difficulty_pre_r9d.json": "845d515cb9ef3af6fd4a1c14ce3731f06ca7b535498e3165b213a511e761b139",
       "writing/supplement/r9_tables/blockG_difficulty_pre_r9d.tex": "b7558c27fd850c4a4988c81df273c919a4fdc464a48f76a68a8727a808693963"}

MAIN_REPL = [
    ("because the rule-selected tolerances are 0.51--1.49 times the development headroom (S25).",
     "because the rule-selected tolerances are 0.51--1.23 times the development headroom (S25)."),
    ("Post hoc, at tighter block-G tolerances on Lenta, FDC-DP certified frontiers with 13--35\\% of feasible development policies $\\eps$-optimal that the rectangle never certified before $\\tau_R$ (S25).",
     "Post hoc (descriptive, reused halves), FDC-DP certified the three tighter block-G Lenta frontiers with 13--35\\% of feasible development policies $\\eps$-optimal after 0.64--0.80 of the table, when its control pools were empty, whereas the rectangle never did before $\\tau_R$ (S25): an exhaustion-time, not a width, advantage."),
    ("A platform paying per outcome read can therefore certify an $\\eps$-optimal policy for the observed-pool objective at every budget with fewer reads by fixing its read plan first.",
     "A platform paying per outcome read can therefore certify $\\eps$-optimal policies for the observed-pool objective, under a frontier-wide guarantee, with fewer reads by fixing its read plan first."),
    ("Ctrl.\\ empty: mean fraction of control pools exhausted at $N_{80}",
     "Lenta $S = 64$: zero-width interval from checkpoint quantisation. Ctrl.\\ empty: mean fraction of control pools exhausted at $N_{80}"),
]
SUPP_REPL = [
    ("\\caption{\\textbf{Decision difficulty at the block-G tolerances} (post hoc, descriptive, revision r9c). Development-half census as in Table~\\ref{tab:difficultyfull}; evaluation columns from the sealed block-G summary (S19). Rows: FDC-DP reached 12 of 15 before $\\tau_R$ on all 200 streams and the rectangle missed the 80\\% criterion (M) or was pinned (P). $^*$Exact bracket.}",
     "\\caption{\\textbf{Decision difficulty at the block-G tolerances} (post hoc, descriptive, revisions r9c--r9d). Development-half census as in Table~\\ref{tab:difficultyfull}; evaluation columns copied from the sealed block-G summary (S19). FDC ctrl empty: mean fraction of control pools exhausted at FDC-DP's $N_{80}$. Rows: FDC-DP reached 12 of 15 before $\\tau_R$ on all 200 streams and the rectangle missed the 80\\% criterion (M) or was pinned (P). $^\\star$Lock $\\eps$ of the cell. $^*$Exact bracket that rounds to the printed value.}"),
    ("which the paper's Table 1B now summarises in one row.", "which Section 6.2 of the paper now summarises in two sentences."),
    ("(moved from Table 1B and 1C of the paper in revision r6; the paper keeps a one-row summary)",
     "(moved from Table 1B and 1C of the paper in revision r6; since revision r9c the paper summarises them in Section 6.2)"),
    ("(checkpoint tests: Table 1B of the paper and Table~\\ref{tab:ckpanels})",
     "(checkpoint tests: Section 6.2 of the paper and Table~\\ref{tab:ckpanels})"),
]
# the S25 block-G paragraph is replaced as a whole (r9c wording claimed demanding frontiers; r9d states exhaustion)
PARA_START = "\\textbf{Block-G tolerances (revision"
PARA_END = "\\begin{table*}[t]\n\\caption{\\textbf{Decision difficulty at the block-G tolerances}"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def nums(t):
    t = re.sub(r"\\cite[pt]?\{[^}]*\}", "", t)
    return collections.Counter(re.findall(r"\d+(?:[.,{}]\d+)*", t))


def chain(script, main_src, supp_src, pdf_src):
    root = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9d_"))
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


print("== 0. r9c on its own text (chains r9, r8d, r8c, r8b)")
last = chain("verify_r9c_numbers.py", LA / "main_pre_r9d.tex", SUP / "supplement_pre_r9d.tex", LA / "main_r9c.pdf")
ok("verify_r9c on the r9c state", last.startswith("r9c: 0 mismatch"), last[:200])

print("== 1. r9c state")
for f, h in SHA.items():
    ok(f"{f} == r9c (6a0eff67)", sha((IT / f).read_bytes()) == h)

print("== 2. pre-r9d + r9d replacements == current files")
PM, PS = (LA / "main_pre_r9d.tex").read_text(), (SUP / "supplement_pre_r9d.tex").read_text()
M, S = (LA / "main.tex").read_text(), (SUP / "supplement.tex").read_text()
t = PM
for a, b in MAIN_REPL:
    ok(f"main: replacement source unique: {a[:50]!r}", t.count(a) == 1)
    t = t.replace(a, b)
ok("main: rebuilt == current", t == M)
u = PS
old_para = u[u.index(PARA_START):u.index(PARA_END)]
new_para = S[S.index(PARA_START):S.index(PARA_END)]
u = u.replace(old_para, new_para)
for a, b in SUPP_REPL:
    ok(f"supp: replacement source unique: {a[:50]!r}", u.count(a) == 1)
    u = u.replace(a, b)
ok("supp: rebuilt == current (block-G paragraph replaced as a whole)", u == S)
ok("supp: r9c over-claim gone", "descriptive evidence that FDC-DP certifies demanding frontiers" not in S)
for need in ("in each case once every control pool was empty (exhausted control fraction 1.00 on all 200 streams)",
             "not a width advantage, and are not evidence that FDC-DP certifies demanding frontiers before exhaustion",
             "At each cell's lock $\\eps$ its development columns reproduce"):
    ok(f"supp has: {need[:60]}", need in S)

print("== 3. block-G record")
old = json.loads((IT / "writing/r9c_blockG_difficulty_pre_r9d.json").read_text())
new = json.loads((IT / "writing/r9c_blockG_difficulty.json").read_text())
G = json.loads((IT / "exp/results/full/v10_posthoc_G/summary.json").read_text())["G1"]
ok("same rows", [(r["log"], r["S"], r["eps"]) for r in old["rows"]] == [(r["log"], r["S"], r["eps"]) for r in new["rows"]])
ok("development columns and flags unchanged", all(a["dev"] == b["dev"] and a["flags"] == b["flags"] for a, b in zip(old["rows"], new["rows"])))
ok("every r9c evaluation field unchanged", all(b["eval"][k] == v for a, b in zip(old["rows"], new["rows"]) for k, v in a["eval"].items()))
bad = []
for r in new["rows"]:
    g = G[f"G1-eval-{r['dev']['data']}-{r['S']}"]
    key = [k for k in g["by_eps"] if abs(float(k) - r["eps"]) < 1e-12][0]
    m = g["by_eps"][key]["methods"]
    f, R = m["TU-FDC-DP(b)"], m["RECT-BF-DP-TU"]
    e = r["eval"]
    if (e["fdc_exh_ctrl"], e["fdc_exh_all"], e["fdc_sd_log_N80"], e["rect_exh_ctrl"]) != (
            f["exhaustion_at_k80"]["ctrl"], f["exhaustion_at_k80"]["all"], f["sd_log_N80"], R["exhaustion_at_k80"]["ctrl"]):
        bad.append((r["log"], r["S"], r["eps"]))
ok("added evaluation fields equal the sealed block-G summary", not bad, bad)
cited = {(r["log"], r["S"], r["eps"]): r for r in new["rows"]}
for k in (("Lenta", 64, 0.003), ("Lenta", 64, 0.004), ("Lenta", 32, 0.003)):
    ok(f"{k}: FDC-DP control pools all empty at N80 (1.00), rectangle 0/200 before tau_R",
       cited[k]["eval"]["fdc_exh_ctrl"] == 1.0 and cited[k]["eval"]["rect_n80_lt_tau"] == 0)
x = cited[("X5", 64, 0.02)]["eval"]
ok("X5 S=64 eps=0.02: 0.496 of the table, control empty 0.01, rectangle 0/200",
   f"{x['fdc_N80_over_tau']:.3f}" == "0.496" and f"{x['fdc_exh_ctrl']:.2f}" == "0.01" and x["rect_n80_lt_tau"] == 0)
spec = importlib.util.spec_from_file_location("difficulty_blockG_r9c", IT / "writing/scripts/difficulty_blockG_r9c.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tmp = pathlib.Path(tempfile.mkdtemp(prefix="verify_r9d_tex_"))
GFRAG = (SUP / "r9_tables/blockG_difficulty.tex").read_text()
OLDG = (SUP / "r9_tables/blockG_difficulty_pre_r9d.tex").read_text()
try:
    (tmp / "writing/supplement/r9_tables").mkdir(parents=True)
    mod.IT = tmp
    mod.render(new)
    ok("block-G table re-renders from the record", (tmp / "writing/supplement/r9_tables/blockG_difficulty.tex").read_text() == GFRAG)
    ok("block-G summary re-renders from the record", (tmp / "writing/r9c_blockG_difficulty.md").read_text() == (IT / "writing/r9c_blockG_difficulty.md").read_text())
finally:
    shutil.rmtree(tmp, ignore_errors=True)
strip = lambda s: re.sub(r" & \d\.\d\d(?= & \d+/\d+ &)", "", s).replace(" & FDC ctrl empty", "").replace("{5}{c}", "{4}{c}").replace("7-11", "7-10").replace("lrrrrrrrrrl", "lrrrrrrrrl")
ok("block-G table differs from r9c only by the new column", strip(GFRAG) == OLDG)

print("== 4. numbers")
src = nums(PM) + nums(PS) + nums(GFRAG)
add_m = sorted(k for k in nums(M) - nums(PM) if "." in k and k not in src)
add_s = sorted(k for k in nums(S) - nums(PS) if "." in k and k not in src)
ok("main: added decimals come from the r9c texts or the block-G table", not add_m, add_m)
ok("supp: added decimals come from the r9c texts or the block-G table", not add_s, add_s)
ok("1.23 is the largest confirmatory eps/headroom (X5 S=32); 1.49 is the descriptive S=64 cell",
   "1.23" in (SUP / "r9_tables/difficulty.tex").read_text() and "1.49" in (SUP / "r9_tables/difficulty.tex").read_text())

print("== 5. PDFs and sentence rules")
import pymupdf  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from offpage_check import offpage  # noqa: E402
d = pymupdf.open(LA / "main.pdf")
txt = [" ".join(p.get_text().split()) for p in d]
ok("main.pdf has 12 pages", len(d) == 12, len(d))
ok("conclusion on page 8", "by fixing its read plan first." in txt[7])
ok("main.pdf compiled from this main.tex", "not a width, advantage" in " ".join(txt) and "1.23 times the development headroom" in " ".join(txt))
ok("main.pdf: no text outside the page", not offpage(LA / "main.pdf"), offpage(LA / "main.pdf"))
ok("supplement.pdf: no text outside the page", not offpage(SUP / "supplement.pdf"), offpage(SUP / "supplement.pdf"))
sp = " ".join(" ".join(p.get_text().split()) for p in pymupdf.open(SUP / "supplement.pdf"))
ok("supplement.pdf compiled from this supplement.tex", "not a width advantage" in sp and "FDC ctrl empty" in sp)
import verify_r5_numbers as v5  # noqa: E402
v5.MAIN = M
lens = [len(x.split()) for x in v5.body_text()]
ok("no body sentence over 45 words", max(lens) <= 45, max(lens))
ok("body mean sentence length <= 25", sum(lens) / len(lens) <= 25, round(sum(lens) / len(lens), 1))

print(f"\nr9d: {len(FAIL)} mismatch(es)" + (": " + "; ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
