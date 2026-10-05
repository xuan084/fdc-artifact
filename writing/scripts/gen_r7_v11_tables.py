"""Generate the lock-v11 (Open Bandit Dataset) tables of supplement S21 from the result files (read-only).

Sources:
  exp/results/full/v11_obd/v11_analysis.json        eval analysis (all four methods, five comparisons, decision)
  exp/results/pilots/v11_dev/v11_analysis_dev.json  development cell (for the dev column of the ratio table)
  exp/results/v11_gates/v11_eps_hc.json             eps rule trace and HC-WoR-DP tuning per eps
  exp/results/v11_gates/obd_v11_frozen.json         frozen design (segment shares, costs, budgets, arms)
Writes writing/supplement/r7_tables/{v11_ratios,v11_methods,v11_eps_rule,v11_design}.tex, each a whole table
environment (an \\input of bare rows inside a tabular breaks \\bottomrule).
Usage (repo root): .venv/bin/python3 iter_001/writing/scripts/gen_r7_v11_tables.py [--check]
  --check: do not write; exit 1 if any shipped table differs from the generator output.
"""
import json
import sys
from pathlib import Path

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
OUT = IT / "writing/supplement/r7_tables"

P, R, HC, HG = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP[tuned]", "RECT-HG-DP"
LABEL = {P: "TU-FDC-DP(b)", R: "RECT-BF-DP-TU", HC: "HC-WoR-DP (tuned)", HG: "RECT-HG-DP"}
GUAR = {P: "TU fr.", R: "TU fr.", HC: "TU any", HG: "CP"}
ROLE = {f"{P}/{R}": "primary", f"{P}/{HC}": "descr.", f"{P}/{HG}": "descr.",
        f"{R}/{HC}": "descr.", f"{R}/{HG}": "descr."}


def J(p):
    return json.load(open(p))


def f3(x):
    return f"{x:.3f}"


def pct(x):
    s = f"{100 * x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def ratios(an, dev):
    rows = []
    for k, c in an["cell"]["comparisons"].items():
        a, b = k.split("/")
        d = dev["cell"]["comparisons"][k]
        lab = f"{LABEL[a]} / {LABEL[b]}"
        if ROLE[k] == "primary":
            lab = "\\textbf{" + lab + "}"
        rows.append(f"{lab} & {ROLE[k]} & {f3(c['geomean_ratio'])} [{f3(c['ci95_two_sided'][0])}, "
                    f"{f3(c['ci95_two_sided'][1])}] & {f3(c['ub95_one_sided'])} & "
                    f"{pct(c['frac_faster'])} / {pct(c['frac_tied'])} / {pct(c['frac_slower'])} & "
                    f"{f3(d['geomean_ratio'])} ({f3(d['ub95_one_sided'])}) \\\\")
    dec = an["decision"]
    return (
        "\\begin{table*}[t]\n"
        "\\caption{\\textbf{Lock v11, every comparison} (Open Bandit Dataset, \\texttt{random/all} evaluation half, "
        f"200 fresh paired streams, seeds 39000--39199, $\\eps = 3\\cdot 10^{{-4}}$, $K = {an['frozen_configs_used']['K']}$). "
        "Ratio = paired geometric mean of $\\Npen$ (first / second method) with two-sided 95\\% CI and one-sided UB95 "
        "(paired percentile bootstrap, $B = 10^4$, seed 42, one resample matrix). F / T / S = \\% of streams on which "
        "the first method stops earlier / at the same checkpoint / later. Dev = the same statistic on the 50 "
        "development streams of the selected $\\eps$ (seeds 950--999), UB95 in parentheses. Decision rule: UB95 of the "
        f"primary row $< {an['frozen_configs_used']['thresh']:.2f}$, no false TU-FDC-DP(b) stream over the whole run, "
        f"replica pass; verdict \\texttt{{{dec['verdict'].replace('_', chr(92) + '_')}}}.}}\n"
        "\\label{tab:v11ratios}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
        "Comparison & Role & Ratio [95\\% CI] & UB95 & F / T / S & Dev (UB95) \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def methods(an):
    rows = []
    for m in (P, R, HC, HG):
        x = an["cell"]["methods"][m]
        e = x["exhaustion_at_k80_mean"]
        bnb = (f"{x['bnb_calls']:,} / {x['b_only_certs']} / {x['node_limit_hits']}" if "bnb_calls" in x else "---")
        rows.append(f"{LABEL[m]} & {GUAR[m]} & {x['false_streams']} ({f3(x['cp_ub95'])[1:] if x['cp_ub95'] < 1 else f3(x['cp_ub95'])}) & "
                    f"{x['geomean_N80_pen'] / 1e3:,.1f}k & {f3(x['geomean_N80_over_tau'])} & "
                    f"{pct(x['share_N80_lt_tau'])} & {pct(x['share_reached_15'])} & "
                    f"{e['all']:.2f} / {e['ctrl']:.2f} / {e['treat']:.2f} & {bnb} \\\\")
    tau = an["cell"]["methods"][P]["tau_R"]
    return (
        "\\begin{table*}[t]\n"
        "\\caption{\\textbf{Lock v11, every method} (same 200 streams). Guar.: TU fr.\\ = time-uniform on the frozen "
        "schedule; TU any = time-uniform under any predictable sampling; CP = valid at the $K$ checkpoints only. False: "
        "streams with any false certificate over the whole run to 15 of 15 (one-sided 95\\% Clopper--Pearson bound). "
        f"Rows: geometric-mean $\\Npen$ ($\\tau_R$ = {tau:,} rows). $N_{{80}}/\\tau_R$: geometric mean. $<\\tau_R$: \\% of "
        "streams reaching 12 of 15 strictly before $\\tau_R$; 15/15: \\% reaching 15 of 15. Exhausted: mean fraction of "
        "all / control / treatment pools exhausted at $N_{80}$. B\\&B: branch-and-bound calls / branch-only "
        "certifications / node-limit hits (scheme (b)).}\n"
        "\\label{tab:v11methods}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}lllrrrrll@{}}\n\\toprule\n"
        "Method & Guar. & False (UB) & Rows & $N_{80}/\\tau_R$ & $<\\tau_R$ & 15/15 & Exhausted & B\\&B \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def eps_rule(g):
    tr = g["eps_rule"]["trace"]
    rows = []
    for t in tr:
        r, h = t["per_rect"][R], t["per_rect"][HC]
        cfg = g["hc_by_eps"][repr(t["eps"])] if repr(t["eps"]) in g["hc_by_eps"] else g["hc_by_eps"][str(t["eps"])]
        best = "HC-WoR-DP" if t["best"] == HC else "RECT-BF-DP-TU"
        rows.append(f"$3\\cdot 10^{{-4}}$ & {r['successes']} / {r['geomean_N80_pen']:,.0f} & "
                    f"({cfg['c']}, {cfg['tf']}) {h['successes']} / {h['geomean_N80_pen']:,.0f} & {best} & "
                    f"{'yes (selected)' if t['qualifies'] else 'no'} \\\\")
    hcrows = []
    for e in g["eps_grid"]:
        k = repr(e) if repr(e) in g["hc_by_eps"] else str(e)
        c = g["hc_by_eps"][k]
        best = c["grid"][c["name"]]
        es = {0.0003: "3", 0.0005: "5", 0.00075: "7.5", 0.001: "10", 0.0015: "15"}[e]
        hcrows.append(f"${es}\\cdot 10^{{-4}}$ & ({c['c']}, {c['tf']}) & {best['successes']} / 50 & "
                      f"{best['geomean_N80_pen']:,.0f} & {best['false_streams']} \\\\")
    return (
        "\\begin{table}[H]\n"
        "\\caption{\\textbf{Lock v11 $\\eps$ rule and rival tuning} (development half, 50 streams, seeds 950--999, "
        "$K = 40$). Top: the rule's trace. Each rectangle's entry is successes (12 of 15 strictly before $\\tau_R$) / "
        "geometric-mean $\\Npen$; the dev-best (lower geometric mean) must succeed on at least 40 of 50 streams, and "
        "the smallest qualifying $\\eps$ of the grid is selected, so the trace stops at its first entry. FDC-DP rows "
        "never enter the rule. Bottom: the tuned HC-WoR-DP configuration (schedule \\texttt{nstar}; $c$, target "
        "fraction) at each grid $\\eps$, its successes, geometric-mean $\\Npen$ and false streams.}\n"
        "\\label{tab:v11eps}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.5pt}\n"
        "\\begin{tabular}{@{}lllll@{}}\n\\toprule\n"
        "$\\eps$ & RECT-BF-DP-TU & HC-WoR-DP (tuned) & dev-best & qualifies \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\midrule\n"
        "$\\eps$ & HC config & successes & geo.\\ $\\Npen$ & false \\\\\n\\midrule\n"
        + "\n".join(hcrows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def design(an, fr):
    w_dev = fr["dev_segment_shares"]
    w_ev = an["env"]["w"]
    cost = [c[1] for c in fr["cost"]]
    rows = [f"{s + 1} & {w_dev[s]:.3f} & {w_ev[s]:.3f} & {cost[s]} \\\\" for s in range(fr["S"])]
    bud = ", ".join(str(b) for b in fr["budgets"])
    return (
        "\\begin{table}[H]\n"
        "\\caption{\\textbf{Lock v11 frozen design.} Segments (uf0 $\\times$ uf3 cells, cells under 2\\% of development "
        "rows merged into their uf0 ``other'' cell, ordered by development size), development and evaluation traffic "
        "shares $w_s$, and treatment cost $\\kappa_s = \\max(1, \\mathrm{round}(8 S w_s^{\\text{dev}}))$ (control costs "
        f"0). Budgets $\\lfloor b_q \\sum_s \\kappa_s \\rfloor$ for $b_q = 0.10, 0.15, \\dots, 0.80$: {bud}. "
        f"Arms: the {len(fr['treat_items'])} items with the highest development click rate (arm 1) against the other "
        f"{len(fr['control_items'])} (arm 0); an arm shows a uniformly random item of its group.}}\n"
        "\\label{tab:v11design}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}rrrr@{}}\n\\toprule\n"
        "Segment & $w_s$ (dev) & $w_s$ (eval) & $\\kappa_s$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def build():
    an = J(RES / "full/v11_obd/v11_analysis.json")
    dev = J(RES / "pilots/v11_dev/v11_analysis_dev.json")
    g = J(RES / "v11_gates/v11_eps_hc.json")
    fr = J(RES / "v11_gates/obd_v11_frozen.json")
    return {"v11_ratios.tex": ratios(an, dev), "v11_methods.tex": methods(an),
            "v11_eps_rule.tex": eps_rule(g), "v11_design.tex": design(an, fr)}


def main():
    tabs = build()
    if "--check" in sys.argv:
        bad = [n for n, s in tabs.items() if not (OUT / n).exists() or (OUT / n).read_text() != s]
        print("gen_r7_v11_tables --check:", "OK" if not bad else f"differ: {bad}")
        sys.exit(1 if bad else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, s in tabs.items():
        (OUT / n).write_text(s)
        print("wrote", OUT / n)


if __name__ == "__main__":
    main()
