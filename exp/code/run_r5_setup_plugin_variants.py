"""r5_setup_plugin_variants (PILOT, dev seeds 900-909 identity only).

For each dev stream, run the promoted dsswm plug-in variants and their offline counterparts from
idea/r5_offline/contrarian/offline_ctr.py on the same CR9 dev ctx and compare checkpoint by checkpoint:
  B2-fav-tight  vs offline mk('B2-tightbeta')
  FIX-bal-fav   vs offline mk('FIX0.50-fav-tight')
  POOL (FixBalFav(share=None)) vs offline mk('POOL-fav-tight')   [extra GLR-path check, not registered]
  Peace-fav-bal -- no offline counterpart: determinism (two independent runs identical) + billing.
Only identity and timing are recorded; comparison results (ratios between methods) are deliberately not computed.
"""
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "exp/code"))
sys.path.insert(0, str(WS / "idea/r5_offline/contrarian"))

import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

TASK = "r5_setup_plugin_variants"
RES = WS / "exp/results"
OUT = RES / "pilots" / TASK
SEEDS = list(range(900, 910))
PAIRS = [("B2-fav-tight", "B2-tightbeta"), ("FIX-bal-fav", "FIX0.50-fav-tight"), ("POOL-fav-tight(dsswm)", "POOL-fav-tight"),
         ("Peace-fav-bal", None)]
ROW_KEYS = ("k", "t", "n_cert", "n_false", "new", "new_false", "billed", "served", "skipped", "reselected",
            "billing_ok", "n_cells", "U")
SUM_KEYS = ("N80", "censored", "reached_stop", "k_stop", "n_cert", "n_false", "cert_k", "decided_pi", "billed",
            "served", "skipped", "reselected", "billing_ok", "schedule_digest", "validity", "alloc_kind")


def progress(done, total, extra=None):
    (RES / f"{TASK}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK, "epoch": done, "total_epochs": total, "step": done, "total_steps": total, "loss": None,
        "metric": extra or {}, "updated_at": datetime.now().isoformat()}))


def init():
    import offline_ctr as oc
    oc.init()
    ctx = oc.G["ctx"]
    oc.TB = math.log(int(ctx.feas.sum()) * len(ctx.checkpoints) / ctx.delta)


def new_method(name):
    from dsswm.baselines.plugin_r5 import FixBalFav
    from dsswm.streams import r5_registry as reg
    if name == "POOL-fav-tight(dsswm)":
        return FixBalFav(share=None, name=name)
    return reg.make_method(name)


def job(a):
    import offline_ctr as oc
    from dsswm.streams.frontier_runner import run_stream
    new_name, off_name, seed = a
    G = oc.G
    ctx = G["ctx"]
    out = {"new": new_name, "off": off_name, "seed": seed}
    m = new_method(new_name)
    t0 = time.perf_counter()
    s1, r1 = run_stream(G["env"], m, seed, ctx.problems, oc.EPS, ctx=ctx, J_true=G["Jt"], J_star=G["Js"], keep_U=True)
    out["sec_new"] = time.perf_counter() - t0
    out["beta_new"] = getattr(m, "beta", None) or getattr(m, "beta_override", None)
    if off_name is not None:
        mo = oc.mk(off_name)
        t0 = time.perf_counter()
        s2, r2 = run_stream(G["env"], mo, seed, ctx.problems, oc.EPS, ctx=ctx, J_true=G["Jt"], J_star=G["Js"],
                            keep_U=True)
        out["sec_off"] = time.perf_counter() - t0
        out["validity_off"] = mo.validity
    else:   # determinism: a second independent instance
        m2 = new_method(new_name)
        t0 = time.perf_counter()
        s2, r2 = run_stream(G["env"], m2, seed, ctx.problems, oc.EPS, ctx=ctx, J_true=G["Jt"], J_star=G["Js"],
                            keep_U=True)
        out["sec_off"] = time.perf_counter() - t0
        out["peace_rounds"] = [{k: r.get(k) for k in ("k", "t_start", "N_k", "len", "t_end_actual", "done")}
                               for r in m.log]
    out["validity_new"] = m.validity
    out["n_ck_new"], out["n_ck_ref"] = len(r1), len(r2)
    ck_cmp, ck_match = 0, 0
    mism = []
    for x, y in zip(r1, r2):
        ck_cmp += 1
        bad = [k for k in ROW_KEYS if k != "U" and x.get(k) != y.get(k)]
        ux, uy = np.asarray(x["U"], float), np.asarray(y["U"], float)
        if not np.array_equal(ux, uy):
            bad.append("U")
        if bad:
            mism.append({"k": x["k"], "keys": bad})
        else:
            ck_match += 1
    sbad = [k for k in SUM_KEYS if k not in ("validity",) and s1.get(k) != s2.get(k)]
    out.update({"ck_compared": ck_cmp, "ck_identical": ck_match, "ck_len_equal": len(r1) == len(r2),
                "row_mismatch": mism[:5], "summary_mismatch": sbad,
                "identical": (not mism) and (not sbad) and len(r1) == len(r2),
                "N80_new": s1["N80"], "N80_ref": s2["N80"], "billing_ok": bool(s1["billing_ok"] and s2["billing_ok"]),
                "sec_plan": s1["sec_plan"], "sec_cert": s1["sec_cert"], "sec_total": s1["sec_total"],
                "fwer_event_new": s1["fwer_event"]})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RES / f"{TASK}.pid").write_text(str(os.getpid()))
    t_start = datetime.now()
    jobs = [(n, o, s) for (n, o) in PAIRS for s in SEEDS]
    progress(0, len(jobs))
    rows = []
    with ProcessPoolExecutor(4, initializer=init) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1)):
            rows.append(r)
            with open(OUT / "identity_rows.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
            progress(i + 1, len(jobs), {"identical_so_far": sum(x["identical"] for x in rows)})
            print(f"[{i+1}/{len(jobs)}] {r['new']} seed {r['seed']} identical={r['identical']} "
                  f"ck {r['ck_identical']}/{r['ck_compared']} sec_new={r['sec_new']:.1f}", flush=True)
    per = {}
    for n, o in PAIRS:
        rr = [x for x in rows if x["new"] == n]
        per[n] = {"offline_counterpart": o or "none (determinism check: two independent runs)",
                  "streams_identical": sum(x["identical"] for x in rr), "streams": len(rr),
                  "checkpoints_compared": sum(x["ck_compared"] for x in rr),
                  "checkpoints_identical": sum(x["ck_identical"] for x in rr),
                  "billing_ok_all": all(x["billing_ok"] for x in rr),
                  "validity": sorted({x["validity_new"] for x in rr}),
                  "beta": rr[0]["beta_new"] if rr else None,
                  "sec_per_stream_mean": float(np.mean([x["sec_new"] for x in rr])),
                  "sec_per_stream_max": float(np.max([x["sec_new"] for x in rr])),
                  "sec_cert_mean": float(np.mean([x["sec_cert"] for x in rr])),
                  "sec_plan_mean": float(np.mean([x["sec_plan"] for x in rr])),
                  "mismatches": [{"seed": x["seed"], "rows": x["row_mismatch"], "summary": x["summary_mismatch"]}
                                 for x in rr if not x["identical"]]}
    json.dump({"rows": rows, "per_variant": per}, open(OUT / "identity_raw.json", "w"), indent=1)
    return per, t_start


if __name__ == "__main__":
    per, t_start = main()
    print(json.dumps(per, indent=1))
