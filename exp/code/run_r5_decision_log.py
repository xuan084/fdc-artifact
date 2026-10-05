"""r5_decision_log (deliverable 6): read-only cross-check of every decision-log entry against its source file.

No experiment is run. The script (1) checks each cited file / JSON field / source line / number,
(2) recomputes the exploratory ratios from the raw stream records, (3) records file mtimes and git
commit membership used for the timing claims, (4) checks every backticked path cited in
plan/decision_log_r5.md, and (5) writes plan/decision_log_r5.json plus pilot artefacts.
"""
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np

WS = Path(__file__).resolve().parents[2].resolve()   # = iter_001
REPO = WS.parent
GITP = WS.name  # path prefix inside the git repo
TASK = "r5_decision_log"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(WS / "exp/code"))


def progress(step, total, note=""):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": step, "total_epochs": total, "step": step, "total_steps": total,
        "loss": None, "metric": {"stage": note}, "updated_at": dt.datetime.now().isoformat()}))


CHECKS = []


def add(entry, kind, file, locator, expected, observed, ok, note=""):
    CHECKS.append(dict(id=f"c{len(CHECKS) + 1:03d}", entry=entry, kind=kind, file=file, locator=locator,
                       expected=expected, observed=observed, ok=bool(ok), note=note))


def jget(path, keys):
    o = json.loads((WS / path).read_text())
    for k in keys:
        o = o[k]
    return o


def chk_json(entry, path, keys, expected, tol=None, note=""):
    try:
        obs = jget(path, keys)
        if tol is None:
            ok = obs == expected
        else:
            ok = abs(float(obs) - float(expected)) <= tol
    except Exception as e:  # noqa: BLE001
        obs, ok = f"ERROR {e!r}", False
    add(entry, "json_field", path, ".".join(map(str, keys)), expected, obs, ok, note)
    return obs


def chk_line(entry, path, line, substr, note=""):
    try:
        lines = (WS / path).read_text().splitlines()
        txt = lines[line - 1]
        ok = substr in txt
        obs = txt.strip()[:200]
    except Exception as e:  # noqa: BLE001
        ok, obs = False, f"ERROR {e!r}"
    add(entry, "line_contains", path, f"L{line}", substr, obs, ok, note)
    return ok


def chk_num(entry, label, expected, observed, tol, file="(computed)", note=""):
    ok = abs(float(expected) - float(observed)) <= tol
    add(entry, "number", file, label, expected, observed, ok, note)


def mtime(path):
    p = WS / path if not str(path).startswith("/") else Path(path)
    return dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def sha(path):
    return hashlib.sha256((WS / path).read_bytes()).hexdigest()


def main():
    t0 = dt.datetime.now()
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    progress(0, 6, "start")

    # ------------------------------------------------------------------ (a) user verdict and timing
    E = "a_user_verdict"
    chk_line(E, "../.pipeline/project/MEMORY.md", 43, "主张收窄（用户 2026-10-03 裁决")
    chk_line(E, "../.pipeline/project/prompt_overlays/planner.md", 3, "该裁决晚于第 4 轮开发结果")
    for c, want in (("ee02cd9f", False), ("47a13eef", True)):
        txt = git("show", f"{c}:.pipeline/project/MEMORY.md")
        add(E, "git_content", ".pipeline/project/MEMORY.md", f"commit {c}", want, "主张收窄" in txt, ("主张收窄" in txt) == want,
            "verdict section absent at 02:30:19 commit, present at 13:09:26 commit")
    m_mem = mtime(REPO / ".pipeline/project/MEMORY.md")
    add(E, "mtime", ".pipeline/project/MEMORY.md", "mtime (upper bound of verdict time)", "2026-10-03T12:56:34", m_mem,
        m_mem == "2026-10-03T12:56:34")
    m_nov = mtime("idea/novelty_report.md")
    add(E, "mtime", "idea/novelty_report.md", "mtime (lower bound: still evaluates old H1 vs B2-fav)", "2026-10-03T03:28:13",
        m_nov, m_nov == "2026-10-03T03:28:13")
    chk_line(E, "idea/novelty_report.md", 56, "H1 的\"优效\"对象是发表配置的 B2-fav / Peace-fav")
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 46, "H1（主，优效）FDC 对 B2-fav 与 Peace-fav")
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 117, "截止风险上报")
    chk_json(E, "exp/results/r4_gate_decision.json", ["written_at"], "2026-10-03T02:12:14.402316")
    r4 = "exp/results/r4_gate_decision.json"
    for path, val in ((["B2", "B2-fav"], 1.669), (["Peace", "Peace-fav"], 0.987), (["B2", "B2-rect"], 0.769),
                      (["Peace", "Peace-rect"], 0.753), (["B3", "B3-rect"], 0.553), (["B1"], 0.582), (["B4"], 0.387),
                      (["B2", "B2-nominal"], 0.499)):
        chk_json(E, r4, ["variant_ratios_QFC_over_X", *path], val, tol=5e-4)
    chk_json(E, r4, ["gates", "G1-shape", "all"], False)
    chk_json(E, r4, ["real_pillar"], "dead")
    m_ci = mtime("idea/r5_offline/contrarian/ci_out.txt")
    add(E, "mtime", "idea/r5_offline/contrarian/ci_out.txt", "mtime (verdict also later than r5 exploratory CIs)",
        "2026-10-03T02:38:33", m_ci, m_ci == "2026-10-03T02:38:33")
    ev = [json.loads(l) for l in (REPO / "logs/events.jsonl").read_text().splitlines() if l.strip().startswith("{")]
    idb = [e for e in ev if e.get("stage") == "idea_debate" and e.get("event") in ("stage_start", "stage_end")
           and isinstance(e.get("ts"), (int, float)) and dt.datetime.fromtimestamp(e["ts"]).date() == dt.date(2026, 10, 3)]
    stamps = [(e["event"], dt.datetime.fromtimestamp(e["ts"]).isoformat(timespec="seconds")) for e in idb]
    ok = ("stage_start", "2026-10-03T02:30:19") in stamps and ("stage_end", "2026-10-03T13:09:26") in stamps
    add(E, "event_log", "../logs/events.jsonl", "idea_debate stage window", "02:30:19 -> 13:09:26", stamps, ok)
    chk_line(E, "plan/methodology.md", 3, "在列明并核验过的严格有效对手集中最快")
    progress(1, 6, "a done")

    # ------------------------------------------------------------------ (b) exploratory forking paths
    E = "b_exploration"
    r2 = json.loads((WS / "idea/r5_offline/contrarian/run2.json").read_text())
    r3 = json.loads((WS / "idea/r5_offline/contrarian/run3.json").read_text())
    add(E, "count", "idea/r5_offline/contrarian/run2.json", "rows", 360, len(r2), len(r2) == 360)
    add(E, "count", "idea/r5_offline/contrarian/run3.json", "rows", 700, len(r3), len(r3) == 700)
    exp_r2 = ["B2-fav", "FIX0.50-fav-tight", "FIX0.30-rig", "FIX0.40-rig", "FIX0.50-rig", "FIX0.60-rig",
              "FIX0.40-fav-b10", "FIX0.40-rig-b10", "FIX0.40-rig-b8"]
    exp_r3 = ["B2-fav", "B2fav-b14.1", "B2fav-b10", "FIX0.40-rig-b12", "FIX0.40-rig-b11", "FIX0.40-rig-b14.1",
              "FIX0.45-rig-b12"]
    for name, rows, want, seeds in (("run2", r2, exp_r2, (900, 939)), ("run3", r3, exp_r3, (900, 999))):
        ms = list(dict.fromkeys(x["method"] for x in rows))
        add(E, "set", f"idea/r5_offline/contrarian/{name}.json", "methods", want, ms, ms == want)
        sd = (min(x["seed"] for x in rows), max(x["seed"] for x in rows))
        add(E, "range", f"idea/r5_offline/contrarian/{name}.json", "seed range", list(seeds), list(sd), sd == seeds)

    def geo(rows, m):
        a = np.array([x["N80"] for x in rows if x["method"] == m], float)
        return float(np.exp(np.log(a).mean()))
    for m, g in (("B2-fav", 1132450), ("FIX0.30-rig", 1117823), ("FIX0.40-rig", 950162), ("FIX0.50-rig", 956358),
                 ("FIX0.60-rig", 987952), ("FIX0.50-fav-tight", 856307), ("FIX0.40-fav-b10", 572272),
                 ("FIX0.40-rig-b10", 682062), ("FIX0.40-rig-b8", 529329)):
        chk_num(E, f"run2 geomean N80 {m}", g, geo(r2, m), 1.0, "idea/r5_offline/contrarian/run2.json")
    for m, g in (("B2-fav", 1120733), ("B2fav-b14.1", 864135), ("B2fav-b10", 592716), ("FIX0.40-rig-b12", 831081),
                 ("FIX0.40-rig-b11", 747043), ("FIX0.40-rig-b14.1", 958848), ("FIX0.45-rig-b12", 809751)):
        chk_num(E, f"run3 geomean N80 {m}", g, geo(r3, m), 1.0, "idea/r5_offline/contrarian/run3.json")
    # exact replication of ci.py (rng seed 1, 4000 bootstrap draws) -> must reproduce ci_out.txt
    d = {}
    for x in r3:
        d.setdefault(x["method"], {})[x["seed"]] = x["N80"]
    ref = d["B2-fav"]
    rng = np.random.default_rng(1)
    rep = {}
    for m in d:
        if m == "B2-fav":
            continue
        lg = np.log([d[m][s] / ref[s] for s in sorted(ref)])
        b = [lg[rng.integers(0, 100, 100)].mean() for _ in range(4000)]
        lo, hi = np.exp(np.quantile(b, [.025, .975]))
        rep[m] = (float(np.exp(lg.mean())), float(lo), float(hi), float((lg < 0).mean()))
    ci_lines = (WS / "idea/r5_offline/contrarian/ci_out.txt").read_text().splitlines()
    for ln in ci_lines:
        mm = re.match(r"(\S+) ratio ([\d.]+) CI \[([\d.]+),([\d.]+)\] faster-share ([\d.]+)", ln)
        m = mm.group(1)
        vals = tuple(float(mm.group(i)) for i in range(2, 6))
        r = rep[m]
        ok = all(abs(a - b) <= 6e-4 for a, b in zip(vals[:3], r[:3])) and abs(vals[3] - r[3]) <= 5e-3
        add(E, "recompute", "idea/r5_offline/contrarian/ci_out.txt", f"{m} ratio/CI/faster-share", list(vals),
            [round(v, 4) for v in r], ok, "recomputed from run3.json with ci.py's exact bootstrap")
    for m, s in (("B2fav-b14.1", 973), ("B2fav-b10", 934)):
        bad = [(x["seed"], x["N80"], x["nfalse"]) for x in r3 if x["method"] == m and x["nfalse"] > 0]
        add(E, "records", "idea/r5_offline/contrarian/run3.json", f"{m} false streams", [[s, 6989793, 1]],
            [list(t) for t in bad], bad == [(s, 6989793, 1)], "error-penalised N80 = tau_R")
    for m in ("FIX0.40-rig-b12", "FIX0.40-rig-b11", "FIX0.40-rig-b14.1", "FIX0.45-rig-b12"):
        nf = sum(x["nfalse"] > 0 for x in r3 if x["method"] == m)
        add(E, "records", "idea/r5_offline/contrarian/run3.json", f"{m} false streams", 0, nf, nf == 0)
    # FDC/B2-tight point estimate quoted in proposal C3 (~1.11)
    lg = np.log([d["FIX0.40-rig-b14.1"][s] / d["B2fav-b14.1"][s] for s in sorted(ref)])
    chk_num(E, "paired geomean FIX0.40-rig-b14.1 / B2fav-b14.1 (proposal C3 '~1.11')", 1.110, float(np.exp(lg.mean())), 0.005,
            "idea/r5_offline/contrarian/run3.json")
    chk_line(E, "idea/proposal.md", 17, "FDC/B2-tight ≈ 1.11")
    # pragmatist
    qa, k60 = "idea/r5_offline/pragmatist/offline_qfc_alloc.json", "idea/r5_offline/pragmatist/offline_k60.json"
    chk_json(E, qa, ["QFC-half/B2-fav", "geomean_ratio"], 0.8896, tol=5e-5)
    chk_json(E, qa, ["QFC-half/B2-fav", "n"], 40)
    chk_json(E, qa, ["QFC-half/B2-rect", "geomean_ratio"], 0.4185, tol=5e-5)
    chk_json(E, qa, ["QFC-half/QFC", "geomean_ratio"], 0.5323, tol=5e-5)
    chk_json(E, qa, ["QFC-half/Peace-fav", "geomean_ratio"], 0.5393, tol=5e-5)
    chk_json(E, qa, ["QFC-ney", "median_N80"], 6989793.0)
    chk_json(E, qa, ["tau_R"], 6989793)
    chk_json(E, qa, ["QFC-ney/B2-fav", "geomean_ratio"], 6.172, tol=5e-4)
    chk_line(E, "idea/r5_offline/pragmatist/offline_qfc_alloc.py", 2, "A-Ney uses TRUTH")
    chk_line(E, "idea/r5_offline/pragmatist/offline_qfc_alloc.py", 28, "seeds=list(range(900,940))")
    chk_json(E, k60, ["K"], 60)
    chk_json(E, k60, ["QFC-half/B2-fav", "geomean"], 0.8869, tol=5e-5)
    chk_json(E, k60, ["QFC-half/B2-fav", "ci95"], [0.8505417193233769, 0.9222482611581883])
    chk_json(E, k60, ["QFC-half/B2-rect", "geomean"], 0.4451, tol=5e-5)
    chk_json(E, k60, ["QFC-half/Peace-fav", "geomean"], 0.5706, tol=5e-5)
    chk_line(E, "idea/r5_offline/pragmatist/offline_k60.py", 27, "seeds=list(range(900,930))")
    chk_line(E, "idea/history/round5_debate/perspectives/contrarian.md", 8, "β=ln(P²K/δ)=18.5 | 1.74")
    chk_line(E, "idea/history/round5_debate/perspectives/contrarian.md", 9, "β=18.5 | 0.97")
    raw97 = [p for p in (WS / "idea/r5_offline").rglob("*.json") if "0.97" in p.read_text() and "contrarian" in str(p)]
    add(E, "absence", "idea/r5_offline/contrarian/", "raw record of 0.97 / 1.74 (50/50 or POOL + plug-in GLR, beta 18.5)",
        "none (perspective text only)", [str(p.relative_to(WS)) for p in raw97], raw97 == [],
        "no surviving raw stream file; T1 must regenerate it")
    chk_line(E, "plan/methodology.md", 134, "0.887（QFC-half，r4 账本，K = 60）")
    chk_line(E, "plan/methodology.md", 32, "0.40 / 0.45 份额只作为探索性记录写进决策日志")
    progress(2, 6, "b done")

    # ------------------------------------------------------------------ (c) superiority object + effect size
    E = "c_superiority_and_effect_size"
    chk_line(E, "plan/history/r4_final/pilot_plan.json", 15, "QFC geometric-mean N80 ratio <= 0.67 vs EACH of B1, B2, B3, Peace, B4")
    chk_line(E, "idea/history/round5_debate/perspectives/innovator.md", 18, "门槛预注册为 ≤ 0.67")
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 46, "95% 上界 < 1.0，点估计 ≤ 0.90")
    chk_line(E, "idea/proposal.md", 15, "B1 CLUCB-joint, B4 uniform, B2-rect, B3-rect, Peace-rect")
    chk_line(E, "plan/methodology.md", 13, "单侧 95% 上界都 < 1.0")
    chk_line(E, "plan/methodology.md", 118, "点估计 ≤ 0.90 且上界 < 1.0")
    chk_line(E, "idea/history/round5_debate/perspectives/theoretical.md", 8, "没有找到能把 N80 比压到 ≤0.67 的纯理论机制")
    chk_line(E, "plan/methodology.md", 18, "E-cost 按用户裁决如实写出")
    chk_line(E, "plan/methodology.md", 101, "B2-fav、Peace-fav（发表配置）、B3-fav、B5 maq、H8-*")
    for ln, s in ((89, "Hait-SW"), (90, "Molitor-WoR"), (85, "B4-bal")):
        chk_line(E, "plan/methodology.md", ln, s)

    # ------------------------------------------------------------------ (d) C3 gate -> estimator
    E = "d_C3_to_estimator"
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 47, "比值 95% 上界 ≤ 1.15")
    chk_line(E, "idea/proposal.md", 17, "95% 上界 ≤ 1.15")
    chk_line(E, "idea/proposal.md", 86, "C3：最快 tuned 对手上界 ≤ 1.15")
    chk_line(E, "plan/methodology.md", 16, "不设通过线")
    chk_line(E, "plan/methodology.md", 9, "原 C3 降为估计量，原因见 §6.3")
    chk_line(E, "plan/methodology.md", 147, "### 6.3 C4 析因（机制）")
    progress(3, 6, "c,d done")

    # ------------------------------------------------------------------ (e) eval-half contact in r4_g_eps_select
    E = "e_eval_half_contact"
    eps, sm = "exp/results/r4_gates/eps.json", "exp/results/pilots/r4_g_eps_select/summary.json"
    chk_line(E, eps, 17, "AND eval guard tables (min-over-simple-policies regret > eps)")
    for e_ in ("0.001", "0.0015", "0.002", "0.003"):
        chk_json(E, eps, ["per_eps", e_, "n_nontrivial_eval"], 15)
        chk_json(E, sm, ["guard_counts", "CR9", "eval", e_], 15)
    for e_, n in (("0.01", 10), ("0.0125", 5), ("0.015", 5), ("0.0175", 5)):
        chk_json(E, sm, ["guard_counts", "HR8", "eval", e_], n)
    chk_json(E, eps, ["eps_star"], 0.001)
    chk_json(E, eps, ["integrity", "eval_seeds_touched"], False)
    chk_json(E, eps, ["dev_seeds"], "900-999")
    chk_json(E, sm, ["seeds"], [900, 999])
    gt = json.loads((WS / "exp/results/pilots/r4_g_eps_select/guard_tables.json").read_text())
    ev_rows = gt["CR9|eval"]
    keys = sorted(ev_rows[0].keys())
    add(E, "json_field", "exp/results/pilots/r4_g_eps_select/guard_tables.json", "CR9|eval rows / keys",
        [15, ["J_star", "eps_set_size", "half", "min_trivial_regret", "qid", "regret_pooled_uplift_sign",
              "regret_treat_by_w_desc", "second_gap"]], [len(ev_rows), keys], len(ev_rows) == 15 and "J_star" in keys)
    chk_line(E, "exp/code/run_r4_g_eps_select.py", 15, "the eval half is touched only through the truth-side trivial guard")
    chk_line(E, "exp/code/run_r4_g_eps_select.py", 284, 'for half in ("dev", "eval"):')
    chk_line(E, "exp/code/run_r4_g_eps_select.py", 285, "ENVS[(layer, half)] = PoolReplayEnv(layer, half)")
    chk_line(E, "exp/code/run_r4_g_eps_select.py", 298, "rows, cnt = guard_table(ENVS[(layer, half)], grid)")
    chk_line(E, "exp/code/run_r4_g_eps_select.py", 229, 'a["guard_ok"] = bool(g_dev >= NONTRIV_MIN and g_eval >= NONTRIV_MIN)')
    chk_line(E, "plan/methodology.md", 26, "评价半的汇总真值曾用于 ε 守卫")

    # ------------------------------------------------------------------ (f) C_var 48 -> 18, delta split
    E = "f_ledger"
    chk_line(E, "exp/code/dsswm/baselines/frontier_common.py", 204, "delta_main=0.04, delta_var=0.01, C_var=48")
    for k, v in (("L1", 15.16098292456391), ("x_v", 12.165250651009918), ("delta_main", 0.04), ("delta_var", 0.01),
                 ("C_var", 48), ("K", 20)):
        chk_json(E, sm, ["layer_info", "CR9", "params", k], v)
    chk_line(E, "plan/theory/qfc_lemma.md", 68, "The plan freezes C_var = 48")
    chk_num(E, "r4 L1 = 9 ln2 + ln(15*20/0.04)", 15.16098, 9 * math.log(2) + math.log(15 * 20 / 0.04), 1e-4)
    chk_num(E, "r4 x_v = ln(2*48*20/0.01)", 12.16525, math.log(2 * 48 * 20 / 0.01), 1e-4)
    chk_num(E, "FDC beta = ln(3386*20/0.045)", 14.2242, math.log(3386 * 20 / 0.045), 1e-4)
    chk_num(E, "FDC x_v = ln(2*18*20/0.005)", 11.8776, math.log(2 * 18 * 20 / 0.005), 1e-4)
    chk_num(E, "B2-fav-tight beta = ln(3386*20/0.05)", 14.12, math.log(3386 * 20 / 0.05), 5e-3)
    chk_num(E, "beta=14.1 union bound 3386*20*e^-14.1 + 0.005", 0.05595, 3386 * 20 * math.exp(-14.1) + 0.005, 5e-5)
    chk_num(E, "union size Q*|Pi| = 15*512", 7680, 15 * 512, 0)
    chk_line(E, "plan/methodology.md", 40, "β = ln(3386·20/0.045) = **14.2242**")
    chk_line(E, "plan/methodology.md", 41, "x_v = ln(720/0.005) = **11.8776**")
    chk_line(E, "plan/methodology.md", 44, "在开发结果出来之前就已写进提案和离线代码的默认值")
    chk_json(E, "exp/results/pilots/r5_fdc_spec_theory/summary.json", ["cr9_dev_ledger", "sigma_Pi_Bq"], 3386)
    # independent read-only recomputation of sum_q |Pi_{B_q}| from the CR9 dev context
    try:
        from dsswm.envs.pool_replay import PoolReplayEnv
        from dsswm.streams import frontier as fr
        from dsswm.streams.frontier_runner import build_ctx
        env = PoolReplayEnv("CR9", "dev")
        ctx = build_ctx(env, fr.cr_problems("visit"), 0.001)
        sp = int(ctx.feas.sum())
        add(E, "recompute", "exp/code/dsswm (CR9 dev ctx)", "sum_q |Pi_{B_q}|, Q, |Pi|, S*A, K",
            [3386, 15, 512, 18, 20], [sp, int(ctx.Q), int(ctx.P), int(ctx.S * ctx.A), len(ctx.checkpoints)],
            [sp, ctx.Q, ctx.P, ctx.S * ctx.A, len(ctx.checkpoints)] == [3386, 15, 512, 18, 20],
            "read-only context build, no stream replay")
    except Exception as e:  # noqa: BLE001
        add(E, "recompute", "exp/code/dsswm (CR9 dev ctx)", "sum_q |Pi_{B_q}|", 3386, f"ERROR {e!r}", False)
    oc = "idea/r5_offline/contrarian/offline_ctr.py"
    chk_line(E, oc, 29, "math.log(nfeas*K/0.045)", "external reviewer review cites offline_ctr.py:26; actual line is 29")
    chk_line(E, oc, 30, "math.log(ctx.S*ctx.A*K/0.005)")
    chk_line(E, oc, 26, "def setup(self,ctx):", "line 26 is the setup() header, not the delta split")
    chk_line(E, "reviews/idea_debate_review.md", 39, "offline_ctr.py:26")
    mt = {p: mtime(p) for p in ("exp/results/r4_gate_decision.json", "idea/r5_offline/contrarian/run2.json", oc,
                                "idea/r5_offline/contrarian/run3.json", "idea/r5_offline/contrarian/ci_out.txt")}
    order = list(mt.values())
    want = {"exp/results/r4_gate_decision.json": "2026-10-03T02:12:14", "idea/r5_offline/contrarian/run2.json": "2026-10-03T02:35:45",
            oc: "2026-10-03T02:36:16", "idea/r5_offline/contrarian/run3.json": "2026-10-03T02:37:51",
            "idea/r5_offline/contrarian/ci_out.txt": "2026-10-03T02:38:33"}
    add(E, "mtime", "(several)", "mtime order r4_gate_decision < run2 < offline_ctr.py < run3 < ci_out", want, mt,
        mt == want and order == sorted(order),
        "offline_ctr.py was last modified 31 s AFTER run2.json; mtime cannot prove the split predates run2")
    first = git("log", "--diff-filter=A", "--format=%h %ad", "--date=iso", "--", f"{GITP}/{oc}").strip().splitlines()
    add(E, "git_first_commit", oc, "first commit adding the file", "47a13eef 2026-10-03 13:09:26 +1000",
        first[-1] if first else None, bool(first) and first[-1].startswith("47a13eef 2026-10-03 13:09:26"),
        "committed only at end of idea_debate; git cannot order it against run2/run3")
    early = []
    for p in sorted((WS / "idea").rglob("*.md")) + sorted((WS / "plan/history").rglob("*.md")):
        if p.stat().st_mtime < dt.datetime(2026, 10, 3, 2, 36, 16).timestamp():
            if re.search(r"δ_?main\s*=\s*0\.045|delta_main\s*=\s*0\.045|0\.045\s*/\s*0\.005", p.read_text(errors="ignore")):
                early.append(str(p.relative_to(WS)))
    add(E, "absence", "idea/**, plan/history/**", "delta split 0.045/0.005 in any document older than offline_ctr.py",
        [], early, early == [], "earlier '0.045' hits are CP bounds / sigma^2, not the ledger")
    a = {x["seed"]: x["N80"] for x in r2 if x["method"] == "FIX0.40-rig"}
    b = {x["seed"]: x["N80"] for x in r3 if x["method"] == "FIX0.40-rig-b14.1"}
    ge, gt_ = sum(a[s] >= b[s] for s in a), sum(a[s] > b[s] for s in a)
    add(E, "consistency", "idea/r5_offline/contrarian/run2.json + run3.json",
        "default-beta FIX0.40-rig (run2) N80 >= FIX0.40-rig-b14.1 (run3) per seed", [40, 40, 1], [ge, len(a), gt_],
        ge == 40 and len(a) == 40 and gt_ == 1, "consistent with default beta > 14.1 (i.e. delta_main < 0.0504) at run2 time; not a proof")
    srch = sorted({x["method"] for x in r2 + r3})
    add(E, "absence", "idea/r5_offline/contrarian/run2.json + run3.json", "any run varying delta split",
        "none", [m for m in srch if "dv" in m or "dm" in m], True, "only shares and beta overrides were varied")
    progress(4, 6, "e,f done")

    # ------------------------------------------------------------------ (g) QFC-half 50/50 chosen after r4
    E = "g_qfc_half_post_r4"
    chk_line(E, "idea/r5_offline/pragmatist/offline_qfc_alloc.py", 1, "EXPLORATORY (r5 pragmatist)")
    m_qa = mtime("idea/r5_offline/pragmatist/offline_qfc_alloc.py")
    add(E, "mtime", "idea/r5_offline/pragmatist/offline_qfc_alloc.py", "mtime > r4_gate_decision written_at 02:12:14",
        "2026-10-03T02:33:51", m_qa, m_qa == "2026-10-03T02:33:51" and m_qa > "2026-10-03T02:12:14")
    chk_line(E, "idea/r5_offline/pragmatist/offline_qfc_alloc.py", 36, "r4_g1_shape_*/results.jsonl",
             "reference N80s are r4 dev-block stream results")
    chk_line(E, r4, 1, "{")
    chk_json(E, r4, ["primary_variant", "B2"], "B2-fav")
    chk_line(E, "idea/proposal.md", 47, "慢在池比例读法饿死对照池")
    chk_line(E, "plan/methodology.md", 168, "QFC-half 的 50/50 是看过 r4 结果之后才选的")

    # ------------------------------------------------------------------ (h) deletions
    E = "h_deletions"
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 54, "两阶段版本作为预注册备选")
    chk_line(E, "idea/history/convergence/proposal_20261003-1256_pre.md", 57, "10-06 前证不出即砍")
    chk_line(E, "idea/proposal.md", 63, "两阶段版本（前 5% 池比例读、估方差后冻结）作为预注册备选")
    chk_line(E, "idea/proposal.md", 66, "10-06 前证不出即砍")
    chk_line(E, "plan/methodology.md", 32, "**两阶段版本删除**")
    chk_line(E, "plan/methodology.md", 61, "10-06 前没有证明，按提案规则删除")
    m_meth = mtime("plan/methodology.md")
    add(E, "mtime", "plan/methodology.md", "deletion date vs the 10-06 deadline", "2026-10-03 (< 10-06)", m_meth,
        m_meth.startswith("2026-10-03"), "module was dropped pre-emptively on 10-03, not by the 10-06 rule firing")
    chk_line(E, "plan/methodology.md", 46, "β = 14.1 的旧结果")

    # ------------------------------------------------------------------ contribution statement sources
    E = "contribution"
    chk_line(E, "exp/code/dsswm/certify/quadknap.py", 21, "S ln A + ln(Q K / delta_main)")
    chk_line(E, "reviews/idea_debate_review.md", 102, "### P0-4")
    for cid in ("2609.37873", "2608.19903", "2405.19317", "2608.06512"):
        ok = cid in (WS / "idea/novelty_report.md").read_text()
        add(E, "citation", "idea/novelty_report.md", cid, True, ok, ok)
    chk_line(E, "idea/novelty_report.md", 39, "Garivier & Kaufmann 2016")
    chk_line(E, "idea/novelty_report.md", 39, "Peace")
    chk_line(E, "plan/methodology.md", 170, "冻结设计换来方向级证书的合法性")
    progress(5, 6, "entries done")

    # ------------------------------------------------------------------ MD citations
    md = WS / "plan/decision_log_r5.md"
    md_refs = []
    if md.exists():
        for ref in sorted(set(re.findall(r"`([A-Za-z0-9_./\-]+\.(?:json|md|py|txt|jsonl|log))(?::(\d+))?`", md.read_text()))):
            p, ln = ref
            base = REPO if p.startswith(".pipeline") or p.startswith("logs/") else WS
            fp = base / p
            ok = fp.exists() and (not ln or len(fp.read_text(errors="ignore").splitlines()) >= int(ln))
            md_refs.append({"path": p, "line": ln or None, "ok": ok})
            add("md_citations", "path_exists", p, f"L{ln}" if ln else "file", True, ok, ok)

    n_ok = sum(c["ok"] for c in CHECKS)
    n_num = sum(c["kind"] in ("json_field", "number", "recompute", "count", "records", "consistency", "range") for c in CHECKS)
    srcs = sorted({c["file"] for c in CHECKS if (WS / c["file"]).is_file()})
    summary = {"task_id": TASK, "mode": "pilot", "seed": 42, "started_at": t0.isoformat(timespec="seconds"),
               "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
               "n_checks": len(CHECKS), "n_checks_ok": n_ok, "n_checks_failed": len(CHECKS) - n_ok,
               "n_numeric_checks": n_num, "n_sources_verified": len(srcs), "n_md_citations": len(md_refs),
               "failed": [c for c in CHECKS if not c["ok"]],
               "pass": n_ok == len(CHECKS) and len(CHECKS) >= 100}
    (OUT / "checks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in CHECKS) + "\n")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    log = {"schema": "decision_log_r5/v1", "generated_at": summary["finished_at"], "generator": "exp/code/run_r5_decision_log.py",
           "companion_md": "plan/decision_log_r5.md",
           "source_sha256": {s: sha(s) for s in srcs},
           "verification": {k: summary[k] for k in ("n_checks", "n_checks_ok", "n_numeric_checks", "n_sources_verified", "pass")},
           "entries": ENTRIES, "checks": CHECKS}
    (WS / "plan/decision_log_r5.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failed"}, ensure_ascii=False))
    for c in summary["failed"]:
        print("FAIL", json.dumps(c, ensure_ascii=False)[:400])
    progress(6, 6, "done")
    return summary


ENTRIES = json.loads((Path(__file__).with_name("r5_decision_log_entries.json")).read_text())

if __name__ == "__main__":
    main()
