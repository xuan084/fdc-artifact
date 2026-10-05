"""Generate the lock-v12 (Open Bandit women and men campaigns, DESCRIPTIVE) tables of supplement S23 (read-only).

Sources (per campaign c in {women, men}):
  exp/results/full/v12_obd/v12_{c}_analysis.json         eval analysis (four methods, five comparisons, descriptive
                                                         full-frontier and uncensored-subset readings, eps-optimal share)
  exp/results/pilots/v12_dev/v12_{c}_analysis_dev.json   development cell (dev column of the ratio table)
  exp/results/v12_gates/v12_{c}_eps_hc.json              eps rule trace and HC-WoR-DP tuning per eps
  exp/results/v12_gates/v12_{c}_block_status.json        pre-stated block-status gate (dev half, rule eps)
  exp/results/v12_gates/v12_{c}_thresh.json              THRESH computed by the rule (not applicable: descriptive block)
  exp/results/v12_gates/obd_v12_{c}_frozen.json          frozen design (segment shares, costs, budgets, arms)
Writes writing/supplement/r8_tables/{v12_ratios,v12_methods,v12_rule_gate,v12_design}.tex, each a whole table
environment.
Usage (workspace root): .venv/bin/python3 writing/scripts/gen_r8_v12_tables.py [--check]
  --check: do not write; exit 1 if any shipped table differs from the generator output.
"""
import json
import sys
from pathlib import Path

IT = Path(__file__).resolve().parents[2]
RES = IT / "exp/results"
OUT = IT / "writing/supplement/r8_tables"
CAMPS = ("women", "men")
SEEDS = {"women": "39200--39399", "men": "39400--39599"}
EPS_TEX = {0.0003: "3\\cdot 10^{-4}", 0.0005: "5\\cdot 10^{-4}", 0.00075: "7.5\\cdot 10^{-4}",
           0.001: "10^{-3}", 0.0015: "1.5\\cdot 10^{-3}"}

P, R, HC, HG = "TU-FDC-DP(b)", "RECT-BF-DP-TU", "HC-WoR-DP[tuned]", "RECT-HG-DP"
LABEL = {P: "TU-FDC-DP(b)", R: "RECT-BF-DP-TU", HC: "HC-WoR-DP (tuned)", HG: "RECT-HG-DP"}
GUAR = {P: "TU fr.", R: "TU fr.", HC: "TU any", HG: "CP"}
ROLE = {f"{P}/{R}": "headline", f"{P}/{HC}": "descr.", f"{P}/{HG}": "descr.",
        f"{R}/{HC}": "descr.", f"{R}/{HG}": "descr."}


def J(p):
    return json.loads(Path(p).read_text())


def load(c):
    return {
        "an": J(RES / f"full/v12_obd/v12_{c}_analysis.json"),
        "dev": J(RES / f"pilots/v12_dev/v12_{c}_analysis_dev.json"),
        "g": J(RES / f"v12_gates/v12_{c}_eps_hc.json"),
        "st": J(RES / f"v12_gates/v12_{c}_block_status.json"),
        "th": J(RES / f"v12_gates/v12_{c}_thresh.json"),
        "fr": J(RES / f"v12_gates/obd_v12_{c}_frozen.json"),
    }


def f3(x):
    return f"{x:.3f}"


def pct(x):
    s = f"{100 * x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def eps_tex(e):
    return EPS_TEX[round(float(e), 6)]


def ratio_cells(c, d=None):
    s = (f"{f3(c['geomean_ratio'])} [{f3(c['ci95_two_sided'][0])}, {f3(c['ci95_two_sided'][1])}] & "
         f"{f3(c['ub95_one_sided'])} & {pct(c['frac_faster'])} / {pct(c['frac_tied'])} / {pct(c['frac_slower'])}")
    if d is not None:
        s += f" & {f3(d['geomean_ratio'])} ({f3(d['ub95_one_sided'])})"
    return s


def ratios(D):
    body = []
    for c in CAMPS:
        an, dev = D[c]["an"], D[c]["dev"]
        eu = an["eps_units"]
        body.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{'A' if c == 'women' else 'B'}. {c} "
                    f"(seeds {SEEDS[c]}, $\\eps = {eps_tex(an['eps'])}$, {pct(eu['relative_to_dev_base'])}\\% of the "
                    f"development click rate; $\\tau_R$ = {an['env']['N']:,}; block status: "
                    f"{an['block_status']})}}}} \\\\")
        for k, cc in an["cell"]["comparisons"].items():
            a, b = k.split("/")
            lab = f"{LABEL[a]} / {LABEL[b]}"
            if ROLE[k] == "headline":
                lab = "\\textbf{" + lab + "}"
            body.append(f"{lab} & {ROLE[k]} & {ratio_cells(cc, dev['cell']['comparisons'][k])} \\\\")
        dd = an["cell"]["descriptive_v12"]
        ddv = dev["cell"]["descriptive_v12"]
        body.append(f"{LABEL[P]} / {LABEL[R]}, full frontier ($N^{{\\text{{pen}}}}_{{100}}$) & descr. & "
                    f"{ratio_cells(dd['full_frontier'], ddv['full_frontier'])} \\\\")
        body.append(f"{LABEL[P]} / {LABEL[R]}, uncensored subset (n = {dd['uncensored_subset']['n_streams_in_subset']}) "
                    f"& descr. & {ratio_cells(dd['uncensored_subset'], ddv['uncensored_subset'])} \\\\")
    return (
        "\\begin{table*}[t]\n"
        "\\caption{\\textbf{Lock v12, every comparison} (Open Bandit Dataset, \\texttt{random/women} and "
        "\\texttt{random/men} evaluation halves, 200 fresh paired streams each). Both blocks are \\emph{descriptive} by "
        "the block-status gate fixed in the plan before any of their data were read (Table~\\ref{tab:v12gate}); no "
        "row carries a verdict, THRESH is not applicable, and no ratio is pooled with lock v11 or across campaigns. "
        "Ratio = paired geometric mean of $\\Npen$ (first / second method) with two-sided 95\\% CI and one-sided UB95 "
        "(paired percentile bootstrap, $B = 10^4$, seed 42, one resample matrix per block). F / T / S = \\% of streams "
        "on which the first method stops earlier / at the same checkpoint / later. Full frontier: rows to 15 of 15 "
        "(horizon-sensitive). Uncensored subset: streams on which neither method had an exhausted pool at $N_{80}$ "
        "(pre-stated in the v12 plan). Dev = the same statistic on the 50 development streams of the selected "
        "$\\eps$ (seeds 950--999), UB95 in parentheses.}\n"
        "\\label{tab:v12ratios}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
        "Comparison & Role & Ratio [95\\% CI] & UB95 & F / T / S & Dev (UB95) \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def methods(D):
    body = []
    for c in CAMPS:
        an = D[c]["an"]
        body.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{'A' if c == 'women' else 'B'}. {c} "
                    f"($\\tau_R$ = {an['env']['N']:,} rows, $S = {an['env']['S']}$)}}}} \\\\")
        for m in (P, R, HC, HG):
            x = an["cell"]["methods"][m]
            e = x["exhaustion_at_k80_mean"]
            bnb = (f"{x['bnb_calls']:,} / {x['b_only_certs']} / {x['node_limit_hits']}" if "bnb_calls" in x else "---")
            cp = f3(x["cp_ub95"])
            body.append(f"{LABEL[m]} & {GUAR[m]} & {x['false_streams']} ({cp[1:] if x['cp_ub95'] < 1 else cp}) & "
                        f"{x['geomean_N80_pen'] / 1e3:,.1f}k & {f3(x['geomean_N80_over_tau'])} & "
                        f"{pct(x['share_N80_lt_tau'])} & {pct(x['share_reached_15'])} & "
                        f"{e['all']:.2f} / {e['ctrl']:.2f} / {e['treat']:.2f} & {bnb} \\\\")
    return (
        "\\begin{table*}[t]\n"
        "\\caption{\\textbf{Lock v12, every method} (same streams; descriptive blocks). Guar.: TU fr.\\ = time-uniform "
        "on the frozen schedule; TU any = time-uniform under any predictable sampling; CP = valid at the $K = 40$ "
        "checkpoints only (RECT-HG-DP rows carry the label \\texttt{checkpoint\\_strength\\_CP}). False: streams with "
        "any false certificate over the whole run to 15 of 15 (one-sided 95\\% Clopper--Pearson bound). Rows: "
        "geometric-mean $\\Npen$. $N_{80}/\\tau_R$: geometric mean. $<\\tau_R$: \\% of streams reaching 12 of 15 "
        "strictly before $\\tau_R$; 15/15: \\% reaching 15 of 15. Exhausted: mean fraction of all / control / "
        "treatment pools exhausted at $N_{80}$. B\\&B: branch-and-bound calls / branch-only certifications / "
        "node-limit hits (scheme (b)).}\n"
        "\\label{tab:v12methods}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}lllrrrrll@{}}\n\\toprule\n"
        "Method & Guar. & False (UB) & Rows & $N_{80}/\\tau_R$ & $<\\tau_R$ & 15/15 & Exhausted & B\\&B \\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def rule_gate(D):
    rows = []
    for c in CAMPS:
        g = D[c]["g"]
        for t in g["eps_rule"]["trace"]:
            r, h = t["per_rect"][R], t["per_rect"][HC]
            cfg = g["hc_by_eps"][str(t["eps"])]
            best = "HC" if t["best"] == HC else "RECT"
            rows.append(f"{c} & ${eps_tex(t['eps'])}$ & {r['successes']} / {r['geomean_N80_pen']:,.0f} & "
                        f"({cfg['c']}, {cfg['tf']}) {h['successes']} / {h['geomean_N80_pen']:,.0f} & {best} & "
                        f"{'yes (sel.)' if t['qualifies'] else 'no'} \\\\")
    grows = []
    for c in CAMPS:
        st, an, th = D[c]["st"], D[c]["an"], D[c]["th"]
        cr = st["criteria"]
        es = an["eps_optimal_share"]
        yn = lambda b: "pass" if b else "fail"
        grows.append(f"{c} & ${eps_tex(st['eps'])}$ & {pct(an['eps_units']['relative_to_dev_base'])}\\% & "
                     f"{yn(cr['i_rule_selects_eps'])} & {yn(cr['ii_all_control_not_eps_optimal_in_some_problem'])} "
                     f"({st['n_problems_all_control_eps_optimal']}/{st['n_problems']}) & "
                     f"{yn(cr['iii_mean_eps_optimal_share_le_0.5'])} ({f3(st['mean_share_eps_optimal'])}) & "
                     f"{f3(es['mean_share_eps_optimal'])} ({es['n_problems_all_control_eps_optimal']}/15) & "
                     f"{st['n_policies']:,} & {th['thresh']:.2f} (n/a) & {st['status']} \\\\")
    return (
        "\\begin{table*}[t]\n"
        "\\caption{\\textbf{Lock v12 $\\eps$ rule, rival tuning and block-status gate} (development halves, 50 "
        "streams per campaign, seeds 950--999, $K = 40$). Top: the rule's trace. Each entry is successes (12 of 15 "
        "strictly before $\\tau_R$) / geometric-mean $\\Npen$; HC-WoR-DP shows its tuned configuration "
        "(schedule \\texttt{nstar}; $c$, target fraction) at that $\\eps$. The dev-best rectangle (lower geometric "
        "mean, tie to RECT-BF-DP-TU) must succeed on at least 40 of 50 streams, the smallest qualifying grid $\\eps$ "
        "is selected, and FDC-DP rows never enter the rule. Bottom: the gate, fixed in the plan before any data of "
        "either campaign were read and computed on the development truth at the selected $\\eps$: a block is "
        "confirmatory iff (i) the rule selects an $\\eps$, (ii) all-control is not $\\eps$-optimal in at least one "
        "of the 15 problems (count of problems where it is, in parentheses) and (iii) the mean share of feasible "
        "policies that are $\\eps$-optimal is at most 0.5. Eval share: the same share on the evaluation truth, "
        "computed after the run. THRESH $= \\min(0.90, \\mathrm{UB95}_{\\text{dev}} + 0.10)$ was computed and "
        "recorded but is not applicable to a descriptive block.}\n"
        "\\label{tab:v12gate}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{@{}llllll@{}}\n\\toprule\n"
        "Campaign & $\\eps$ & RECT-BF-DP-TU & HC-WoR-DP (tuned) & dev-best & qualifies \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\n\\medskip\n"
        "\\begin{tabular}{@{}llllllllll@{}}\n\\toprule\n"
        "Campaign & $\\eps$ & $\\eps$ / dev CTR & (i) & (ii) & (iii) (dev share) & Eval share & Policies & THRESH & "
        "Status \\\\\n\\midrule\n"
        + "\n".join(grows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


def design(D):
    W, M = D["women"], D["men"]
    Smax = max(W["fr"]["S"], M["fr"]["S"])
    rows = []
    for s in range(Smax):
        cells = []
        for d in (W, M):
            fr, an = d["fr"], d["an"]
            if s < fr["S"]:
                cells.append(f"{fr['dev_segment_shares'][s]:.3f} & {an['env']['w'][s]:.3f} & {fr['cost'][s][1]}")
            else:
                cells.append(" & & ")
        rows.append(f"{s + 1} & " + " & ".join(cells) + " \\\\")
    bud = "; ".join(f"{c}: " + ", ".join(str(b) for b in D[c]["fr"]["budgets"]) for c in CAMPS)
    arms = "; ".join(f"{c}: {len(D[c]['fr']['treat_items'])} / {len(D[c]['fr']['control_items'])} items" for c in CAMPS)
    return (
        "\\begin{table}[tb]\n"
        "\\caption{\\textbf{Lock v12 frozen designs}, re-derived on each campaign's development half by the lock-v11 "
        "written rules (uf0 $\\times$ uf3 cells, cells under 2\\% of development rows merged into their uf0 "
        "``other'' cell, ordered by development size). Development and evaluation traffic shares $w_s$ and "
        "treatment cost $\\kappa_s = \\max(1, \\mathrm{round}(8 S w_s^{\\text{dev}}))$ (control costs 0). Budgets "
        f"$\\lfloor b_q \\sum_s \\kappa_s \\rfloor$, $b_q = 0.10, \\dots, 0.80$: {bud}. Arms (top half of the items "
        f"by development click rate against the rest): {arms}. No evaluation row needed a fallback segment.}}\n"
        "\\label{tab:v12design}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n"
        " & \\multicolumn{3}{c}{women ($S = " + str(W["fr"]["S"]) + "$)} & \\multicolumn{3}{c}{men ($S = "
        + str(M["fr"]["S"]) + "$)} \\\\\n"
        "Seg. & $w_s$ dev & $w_s$ eval & $\\kappa_s$ & $w_s$ dev & $w_s$ eval & $\\kappa_s$ \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def build():
    D = {c: load(c) for c in CAMPS}
    return {"v12_ratios.tex": ratios(D), "v12_methods.tex": methods(D), "v12_rule_gate.tex": rule_gate(D),
            "v12_design.tex": design(D)}


def main():
    out = build()
    if "--check" in sys.argv:
        bad = [n for n, s in out.items() if not (OUT / n).exists() or (OUT / n).read_text() != s]
        print("gen_r8_v12_tables --check:", "OK" if not bad else f"DIFFERS {bad}")
        sys.exit(1 if bad else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, s in out.items():
        (OUT / n).write_text(s)
        print("wrote", OUT / n)


if __name__ == "__main__":
    main()
