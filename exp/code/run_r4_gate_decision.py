"""r4_gate_decision: aggregate all round-4 development gates into a single decision file.

Reads ONLY the gate summary.json files under exp/results/pilots/ (no method is rerun,
no evaluation seed is touched). Every gate cell records the summary.json field it was
taken from. Rules follow plan/methodology.md section 3 as amended by
"Pilot 后锁前修订 (R1-R7)" and "FCC 锁前修订 (F1-F5)".

Outputs:
  exp/results/r4_gate_decision.json
  exp/results/pilots/r4_gate_decision/{summary.json,gate_table.md}
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
RES = WS / "exp" / "results"
PIL = RES / "pilots"
OUT_DIR = PIL / "r4_gate_decision"
TASK = "r4_gate_decision"
G1_THR = 0.67
GSIZE_THR = 0.7
GFAC_CERT_THR = 0.20


def load(task: str) -> dict:
    return json.loads((PIL / task / "summary.json").read_text())


def get(d: dict, path: str):
    cur = d
    for k in path.split("/"):
        cur = cur[k]
    return cur


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def progress(epoch: int, phase: str) -> None:
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": epoch, "total_epochs": 3, "step": epoch,
        "total_steps": 3, "loss": None, "metric": {"phase": phase},
        "updated_at": datetime.now().isoformat()}))


def main() -> int:
    t0 = datetime.now()
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    progress(0, "load")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = ["r4_lit_g0", "r4_theory_qfc", "r4_setup_quadknap", "r4_setup_criteo_replay",
             "r4_g_eps_select", "r4_g1_shape_a", "r4_g1_shape_b", "r4_g1_shape_c",
             "r4_g_haz", "r4_g_fac_a", "r4_g_fac_b"]
    S = {t: load(t) for t in tasks}
    src_sha = {f"pilots/{t}/summary.json": sha(PIL / t / "summary.json") for t in tasks}

    cells: list[dict] = []

    def cell(gate, pillar, statistic, value, threshold, verdict, source):
        cells.append({"gate": gate, "pillar": pillar, "statistic": statistic,
                      "value": value, "threshold": threshold,
                      "verdict": "PASS" if verdict else "FAIL", "pass": bool(verdict),
                      "source": source})
        return bool(verdict)

    progress(1, "judge")
    # ---------------- real-layer pillar (CR9) ----------------
    lit = S["r4_lit_g0"]
    g0 = cell("G0-lit", "real", "gate_pass (>=1 uncovered measurable capability); novelty",
              {"gate_pass": lit["gate_pass"], "novelty_estimate": lit["novelty_estimate"]},
              "gate_pass == true", lit["gate_pass"] is True,
              "pilots/r4_lit_g0/summary.json:gate_pass,novelty_estimate")

    th = S["r4_theory_qfc"]
    gth_ok = (th["gate"]["G_theory"] == "PASS" and th["gate"]["mc_max_violation_ratio_le_1p2"]
              and th["gate"]["proof_checklist_complete"])
    gth = cell("G-theory", "real", "MC max violation ratio vs e^-x; proof checklist",
               {"max_violation_ratio": th["partA"]["max_violation_ratio"],
                "full_certificate_max_violation_rate": th["partB"]["full_certificate_max_violation_rate"],
                "proof_checklist_complete": th["gate"]["proof_checklist_complete"]},
               "ratio <= 1.2 AND proof complete", gth_ok,
               "pilots/r4_theory_qfc/summary.json:gate.G_theory,partA.max_violation_ratio")

    qk = S["r4_setup_quadknap"]
    gdp = cell("G-dp", "real", "U_dp < U_enum violations; unit tests",
               {"violations": qk["pass_criteria"]["violations_U_dp_lt_U_enum"],
                "unit_tests_pass": qk["pass_criteria"]["unit_tests_pass"]},
               "0 violations", qk["pass_criteria"]["violations_U_dp_lt_U_enum"] == 0
               and qk["pass_criteria"]["pass"],
               "pilots/r4_setup_quadknap/summary.json:pass_criteria")

    cr = S["r4_setup_criteo_replay"]
    cell("setup-CR9 (prerequisite)", "real", "criteo replay pass criteria",
         cr["pass_criteria"], "all true", cr["pass"] is True,
         "pilots/r4_setup_criteo_replay/summary.json:pass,pass_criteria")

    ge = S["r4_g_eps_select"]
    eps_star = ge["eps_star"]["CR9"]
    eps_star_hr8 = ge["eps_star"]["HR8"]
    pe = ge["per_eps"]["CR9"][str(eps_star)] if eps_star is not None else None
    geps = cell("G-eps", "real", "eps* = min grid eps with uncensored>=0.7 and nontrivial>=5/15 (dev & eval)",
                {"eps_star_CR9": eps_star,
                 "uncensored_share": pe and pe["uncensored_share"],
                 "n_nontrivial_dev": pe and pe["n_nontrivial_dev"],
                 "n_nontrivial_eval": pe and pe["n_nontrivial_eval"]},
                "eps* exists in grid", eps_star is not None,
                "pilots/r4_g_eps_select/summary.json:eps_star.CR9,per_eps.CR9")

    # G1-shape: 5 opponents, primary variant rule fav -> nominal -> rect (first with S1 and S2)
    a = S["r4_g1_shape_a"]
    b = S["r4_g1_shape_b"]["gate_G1_shape_b"]
    c = S["r4_g1_shape_c"]["gate_G1_shape_c"]
    g1 = {}
    for X in ("B1", "B4"):
        r = a["gate"]["G1_shape_partA"]["by_opponent"][X]
        g1[X] = cell(f"G1-shape vs {X}", "real", f"paired geomean N80 ratio QFC/{X}",
                     {"ratio": r["geomean_ratio"], "ci95": r["ci95"], "variant": X},
                     f"<= {G1_THR}", r["geomean_ratio"] <= G1_THR,
                     f"pilots/r4_g1_shape_a/summary.json:gate.G1_shape_partA.by_opponent.{X}")
    primary_variant = {}
    variant_ratios = {}
    for X, blk, src in (("B2", b, "r4_g1_shape_b"), ("B3", b, "r4_g1_shape_b"),
                        ("Peace", c, "r4_g1_shape_c")):
        g = blk[X]
        prim = None
        for v in g["order"]:
            vs = g["variant_safety"][v]
            if vs["S1"] and vs["S2"]:
                prim = v
                break
        assert prim == g["dev_reference_primary"], (X, prim, g["dev_reference_primary"])
        primary_variant[X] = prim
        variant_ratios[X] = g["all_variants_ratio"]
        ratio = g["all_variants_ratio"][prim]
        g1[X] = cell(f"G1-shape vs {X}", "real",
                     f"paired geomean N80 ratio QFC/{prim} (primary variant, dev reference)",
                     {"ratio": ratio, "ci95": g["ci95"], "variant": prim,
                      "variant_safety": g["variant_safety"][prim],
                      "all_variants_ratio": g["all_variants_ratio"]},
                     f"<= {G1_THR}", ratio <= G1_THR,
                     f"pilots/{src}/summary.json:gate_G1_shape_{src[-1]}.{X}")
    g1_all = all(g1.values())

    gsz_val = a["gate"]["G_size"]["value"]
    gsize = cell("G-size", "real", "QFC strict uncensored share at eps*", gsz_val,
                 f">= {GSIZE_THR}", gsz_val >= GSIZE_THR,
                 "pilots/r4_g1_shape_a/summary.json:gate.G_size.value")

    real_alive = g0 and gth and gdp and geps and g1_all and gsize

    # ---------------- multi-step secondary claim (MS2) ----------------
    hz = S["r4_g_haz"]["g_haz"]
    haz_ok = {}
    for L in ("MS-H", "MS-S"):
        pl = hz["per_layer"][L]
        att = pl[pl["reproduced_on"]] if pl.get("reproduced_on") else pl["attempt_1"]
        rep = att["reproduction"]
        haz_ok[L] = cell(f"G-haz {L}", "ms2",
                         "JPC FWER CP lower > delta OR FCR_old cluster lower > delta",
                         {"fwer_cp_lower": rep["fwer_cp_lower"], "fcr_old_lower": rep["fcr_old_lower"],
                          "false_streams": att["false_streams"], "n_streams": att["n_streams"],
                          "attempt": pl.get("reproduced_on")},
                         "> 0.05 (either)", rep["reproduced"] is True,
                         f"pilots/r4_g_haz/summary.json:g_haz.per_layer.{L}")
    haz_all = all(haz_ok.values())

    fa = S["r4_g_fac_a"]
    fac_a_safe = cell("G-fac' a (MS-H) safety", "ms2", "FCC(PDL) false streams on dev 800-839",
                      {"false_streams": fa["fcc_false_streams"], "n": fa["n_instances"]},
                      "0", fa["fcc_false_streams"] == 0,
                      "pilots/r4_g_fac_a/summary.json:fcc_false_streams")
    fac_a_cert = cell("G-fac' a (MS-H) non-vacuity", "ms2", "cert rate at eps=0.02 by T_ext=2e5",
                      fa["cert_rate_eps0.02"], f">= {GFAC_CERT_THR}",
                      fa["cert_rate_eps0.02"] >= GFAC_CERT_THR,
                      "pilots/r4_g_fac_a/summary.json:cert_rate_eps0.02")
    fac_a_rt = cell("G-fac' a (MS-H) runtime", "ms2", "projected eval chunk minutes (p90)",
                    fa["runtime"]["proj_chunk_min_p90"], "<= 55", fa["gate"]["runtime_chunk_le_55min"],
                    "pilots/r4_g_fac_a/summary.json:gate.runtime_chunk_le_55min,runtime")
    fb = S["r4_g_fac_b"]
    ms_s = fb["per_layer"]["MS-S"]
    gb = fb["gate_G_fac_prime_part_b"]
    fac_b_safe = cell("G-fac' b (MS-S) safety", "ms2", "FCC(PDL) false streams on dev 860-889",
                      {"false_streams": ms_s["false_streams"], "n": ms_s["n_streams"]}, "0",
                      gb["false_streams_zero"], "pilots/r4_g_fac_b/summary.json:per_layer.MS-S.false_streams")
    fac_b_cert = cell("G-fac' b (MS-S) non-vacuity", "ms2", "cert rate at eps=0.02 by T_ext=2e5",
                      ms_s["cert_rate_eps0.02"], f">= {GFAC_CERT_THR}", gb["cert_rate_ge_0.20"],
                      "pilots/r4_g_fac_b/summary.json:per_layer.MS-S.cert_rate_eps0.02")
    fac_b_rt = cell("G-fac' b (MS-S) runtime", "ms2", "projected eval chunk minutes (worst)",
                    fb["runtime_projection"]["projected_chunk_min_worst"], "<= 55",
                    gb["projected_chunk_le_55min"],
                    "pilots/r4_g_fac_b/summary.json:runtime_projection")
    fac_all = all([fac_a_safe, fac_a_cert, fac_a_rt, fac_b_safe, fac_b_cert, fac_b_rt])

    if not haz_all:
        ms2_status = "dead_hazard_invalid"
    elif not fac_all:
        ms2_status = "dead_fac"
    else:
        ms2_status = "alive"
    ms_alive = ms2_status == "alive"

    # ---------------- pivot (FCC revision F4: only the real layer triggers a pivot) ----------------
    if real_alive:
        pivot = None
        pivot_alternatives = []
    else:
        pivot = "cand_rollout"           # candidates.json pivot_trigger: both qfc pillars fail AND cand_fcc not viable
        pivot_alternatives = ["cand_llm"]

    reasons = []
    if not g1_all:
        fails = [k for k, v in g1.items() if not v]
        reasons.append(
            "真实层判死：G1-shape 未过。QFC 对 " + "、".join(
                f"{k}（{primary_variant.get(k, k)}，比值 {variant_ratios[k][primary_variant[k]]:.3f}）"
                for k in fails if k in primary_variant)
            + f" 的配对几何平均 N80 比 > {G1_THR}。B2-fav 比 QFC 快约 1.67 倍（QFC 0/100 条流更快），"
              "Peace-fav 与 QFC 持平（0.987，42% 并列）。两者开发 FWER 0/100、FCR_old 0，按 fav→nominal→rect 规则为主比较变体。")
        reasons.append(
            "稳健性核对（不改变判定）：即便只和严格（rect）变体比，B2-rect 0.769、Peace-rect 0.753 也都 > 0.67；"
            "所以失败不依赖于'fav 是否合法'的选择。这与 methodology §1.5/R5 事先写明的风险一致：GLR 方向级阈值与 QFC 代数等价，plug-in 方差吃掉了方差自适应的增益。")
    if ms2_status != "alive":
        reasons.append(
            f"多步次要主张 MS2 判 {ms2_status}：G-haz 两层都复现了 r3 JPC 的失败（MS-H FWER 下界 0.204，MS-S 0.508），"
            f"但 G-fac′ part a（MS-H）的 ε=0.02 认证率只有 {fa['cert_rate_eps0.02']:.3f} < 0.20（600 题中 436 题为 tie，"
            f"27/40 个实例零认证），part b（MS-S）0.384 通过。两层安全性都是 0 错误流。")
    reasons.append(
        "pivot 规则（FCC 锁前修订 F4）：只由真实层触发；cand_fcc 已被 F1 诊断为 not_viable。"
        "建议控制面先评估 cand_rollout（其 pivot_trigger 恰为'两支柱都失败且 cand_fcc 不可行'），cand_llm 作为跨问题备选；本任务不自行转向。")
    reasons.append(
        "user_req2_partial：用户要求 2 中'在误设下把错误认证控制在 δ 内'的安全一半，FCC 在开发块上 0 错误流（MS-H 0/40、MS-S 0/30）；"
        "'新增交互成本可控'一半不满足——FCC 认证所需交互中位数约为 JPC 预算 6000 的 16.7–25 倍，且 MS-H 认证率 0.047。")

    anomalies = [
        "B3-fav、B3-nominal、Peace-nominal 的比值完全相同（0.2676）：三者在 τ_R 内 100/100 条流全部删失（N80 记为 τ_R），比值 = QFC N80 / τ_R。不是实现错误，是这些变体在 ε*=0.001 下无法停止。",
        "Peace-rect 与 A-Ney 的比值相同（0.7532）：N80 只能取 20 个对数检查点（相邻比 ≈1.297），离散化使不同方法常落在同一检查点；Peace-fav 的 N80 四分位都在 1,904,849（与 QFC 中位数同一检查点）。G1 门的统计量因此有约 30% 的格点分辨率。",
        "B2-fav 与 Peace-fav 标注 validity=none（plug-in 方差，无有限样本保证），但开发 100 条流上 FWER 0、FCR_old 0，按预注册规则就是主比较对手，不得替换。",
        "MS 开发块 FCC 安全：MS-H 0/40 错误流的单侧 CP 上界 0.072、MS-S 0/30 为 0.095，仅为开发参考，不是 E1-ms′ 判定。",
    ]

    unit_prices = {
        "CR9_sec_per_stream": dict(a["unit_prices_sec_per_stream"]["CR9"],
                                   **{k: v for k, v in S["r4_g1_shape_b"]["sec_per_stream"].items() if k != "QFC"},
                                   **{k: v for k, v in S["r4_g1_shape_c"]["sec_per_stream"].items() if k != "QFC"}),
        "CR12_sec_per_stream": a["unit_prices_sec_per_stream"]["CR12"],
        "HR8_sec_per_stream": a["unit_prices_sec_per_stream"]["HR8"],
        "MS_sec_per_stream_T6000": {k: v["mean"] for k, v in S["r4_g_haz"]["sec_per_stream"].items()},
        "MS_FCC_PDL_sec_per_instance_T2e5": {
            "MS-H_median": fa["runtime"]["sec_per_instance_median"],
            "MS-S_mean": fb["runtime_projection"]["fcc_sec_per_instance_mean"]},
        "note": "并发运行（4 槽），计时偏高",
    }

    hr8_status = {
        "role": "E1 (safety) + descriptive N80 only",
        "eps_star_hr8": eps_star_hr8,
        "status": "E1_only_no_eps_star" if eps_star_hr8 is None else "E1_plus_descriptive",
        "detail": "HR8 网格 {0.01,0.0125,0.015,0.0175} 上 QFC 未删失比例全部 0（τ_R=31,999），无 ε* 满足规则；开发 FWER 0。HR8 只能报告 E1，描述性 N80 全删失。",
        "source": "pilots/r4_g_eps_select/summary.json:eps_star.HR8,per_eps.HR8",
    }

    decision = {
        "task_id": TASK,
        "written_at": datetime.now().isoformat(),
        "mode": "pilot",
        "rules_source": "plan/methodology.md §3 + 'Pilot 后锁前修订 R1-R7' + 'FCC 锁前修订 F1-F5'; task_plan r4_gate_decision description",
        "real_pillar": "alive" if real_alive else "dead",
        "ms_pillar": "alive" if ms_alive else "dead",
        "ms2_status": ms2_status,
        "positive_result_reachable_this_round": bool(real_alive),
        "lr_in_E2": False,
        "eps_star": eps_star,
        "eps_star_lr": None,
        "eps_star_hr8": eps_star_hr8,
        "hr8_status": hr8_status,
        "primary_variant": primary_variant,
        "primary_variant_note": "开发流参考（fav→nominal→rect 中第一个满足 S1∧S2）；确认性应在评价块上重判，但本轮确认性主比较已因真实层判死而跳过",
        "variant_ratios_QFC_over_X": dict(variant_ratios,
                                          B1=a["gate"]["G1_shape_partA"]["by_opponent"]["B1"]["geomean_ratio"],
                                          B4=a["gate"]["G1_shape_partA"]["by_opponent"]["B4"]["geomean_ratio"]),
        "guard_c": S["r4_g_haz"]["guard"]["frozen_c"],
        "guard_c_flag": "no c in {1.5,2,3} passes FWER CP upper <= 0.05; c=3 frozen as fallback (descriptive comparator only)",
        "mislid_hparam": {"kappa": S["r4_g_haz"]["mislid"]["frozen_kappa"],
                          "note": "0 certifications for every kappa at T_max=6000 (structural); tie -> smaller kappa; post-hoc rule clarification 2026-10-03T01:55, no rerun"},
        "unit_prices_sec": unit_prices,
        "pivot": pivot,
        "pivot_alternatives": pivot_alternatives,
        "pivot_not_viable": ["cand_fcc"],
        "pivot_owner": "control plane (this plan does not pivot by itself)",
        "user_req2_partial": True,
        "user_req2_note": "多步'误设下错误认证 ≤ δ'的安全一半在开发块上成立（FCC 0 错误流），'新增交互成本可控'不满足（N_cert/6000 中位 16.7–25×，MS-H 认证率 0.047 < 0.20）。",
        "confirmatory_tasks": {
            "skipped_by_gate": ["r4_cr_main_a", "r4_cr_main_b", "r4_cr_main_c1", "r4_cr_main_c2",
                                "r4_cr_main_d1", "r4_cr_main_d2", "r4_cr_main_ext", "r4_cr12_scale",
                                "r4_cr_h8", "r4_ms_h_a", "r4_ms_h_b", "r4_ms_s_a", "r4_ms_s_b",
                                "r4_ms_r3_a", "r4_ms_r3_b", "r4_ms_f_a", "r4_ms_f_b", "r4_ms_desc"],
            "may_still_run_descriptive": ["r4_hr8_secondary", "r4_lr_e1", "r4_ms_h_c", "r4_ms_s_c"],
            "note": "真实层死 → CR 主比较全部 skipped_by_gate；MS2 死 → MS FCC 任务 skipped_by_gate，G-haz 通过所以描述对照 r4_ms_*_c 按计划可跑；r4_prereg_lock 写 status=locked_no_eval 留痕。是否仍跑安全层/描述任务由控制面在转向时决定（不影响正结果判定）。",
        },
        "gates": {
            "G0-lit": g0, "G-theory": gth, "G-dp": gdp, "G-eps": geps,
            "G1-shape": {"all": g1_all, **g1}, "G-size": gsize,
            "G-haz": {"all": haz_all, **haz_ok},
            "G-fac_prime": {"all": fac_all, "a_safety": fac_a_safe, "a_nonvacuity": fac_a_cert,
                            "a_runtime": fac_a_rt, "b_safety": fac_b_safe,
                            "b_nonvacuity": fac_b_cert, "b_runtime": fac_b_rt},
        },
        "gate_cells": cells,
        "reasons": reasons,
        "anomalies_checked": anomalies,
        "source_sha256": src_sha,
        "integrity": {"eval_seeds_touched": False, "methods_rerun": False,
                      "inputs": "only exp/results/pilots/<gate>/summary.json"},
    }

    progress(2, "write")
    (RES / "r4_gate_decision.json").write_text(json.dumps(decision, indent=1, ensure_ascii=False))

    # gate table (markdown)
    lines = ["# r4 gate table (development gates, zero evaluation seeds)", "",
             f"real_pillar = **{decision['real_pillar']}**, ms2_status = **{ms2_status}**, "
             f"pivot = **{pivot}** (alternatives: {', '.join(pivot_alternatives) or '-'})", "",
             "| Gate | Pillar | Statistic | Value | Threshold | Verdict | Source |",
             "|---|---|---|---|---|---|---|"]
    for cl in cells:
        v = cl["value"]
        if isinstance(v, dict):
            v = ", ".join(f"{k}={(round(x, 4) if isinstance(x, float) else x)}"
                          for k, x in v.items() if not isinstance(x, (dict, list)))
        elif isinstance(v, float):
            v = f"{v:.4f}"
        lines.append(f"| {cl['gate']} | {cl['pillar']} | {cl['statistic']} | {v} | "
                     f"{cl['threshold']} | {cl['verdict']} | `{cl['source']}` |")
    lines += ["", "## All variants (QFC / X, paired geometric-mean N80 ratio, CR9 dev 900-999, eps*=0.001)", "",
              "| Opponent | fav | nominal | rect |", "|---|---|---|---|"]
    for X in ("B2", "B3", "Peace"):
        r = variant_ratios[X]
        lines.append(f"| {X} | {r[X + '-fav']:.3f} | {r[X + '-nominal']:.3f} | {r[X + '-rect']:.3f} |")
    lines.append(f"| B1 | {decision['variant_ratios_QFC_over_X']['B1']:.3f} | - | - |")
    lines.append(f"| B4 | {decision['variant_ratios_QFC_over_X']['B4']:.3f} | - | - |")
    (OUT_DIR / "gate_table.md").write_text("\n".join(lines) + "\n")

    summary = {k: decision[k] for k in ("task_id", "mode", "real_pillar", "ms_pillar", "ms2_status",
                                        "positive_result_reachable_this_round", "eps_star",
                                        "eps_star_hr8", "primary_variant", "guard_c", "mislid_hparam",
                                        "pivot", "pivot_alternatives", "user_req2_partial", "reasons",
                                        "anomalies_checked")}
    summary.update({
        "go_no_go": "NO_GO",
        "candidate_id": "cand_qfc",
        "n_gate_cells": len(cells),
        "all_cells_sourced": all(cl["source"] for cl in cells),
        "pass_criteria": "decision file written with every gate cell sourced to a summary.json field",
        "pass": True,
        "decision_file": "exp/results/r4_gate_decision.json",
        "wall_sec": (datetime.now() - t0).total_seconds(),
        "code_sha256": sha(Path(__file__)),
    })
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    progress(3, "done")
    print(json.dumps({k: summary[k] for k in ("real_pillar", "ms2_status", "pivot", "n_gate_cells")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
