"""v11 development runs for FDC-PW (prior-weighted ledger), dev halves only (seeds 950-999).  Reuses the block-G
runner's environments, contexts and row format; comparators (TU-FDC-DP(b), RECT-BF-DP-TU) come from the existing G1 dev
rows, which were produced by identical library code (checked by code_sha of the shared files)."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v10_posthoc_G as G  # noqa: E402

OUT = G.RES / "pilots" / "fdc_pw"
ROWS = OUT / "rows"
REF: dict = {}


def make_pw(name, ck):
    from dsswm.baselines.fdc_pw import FDCPW, FDCPWTimeUniform
    ref = REF["ref"]
    if name == "TU-FDC-PW(b)":
        return FDCPWTimeUniform(ck, ref, scheme="b")
    if name == "TU-FDC-PW(b)[ref-only]":
        return FDCPWTimeUniform(ck, ref, centres="ref", scheme="b", name=name)
    if name == "TU-FDC-PW(b)[argmax-only]":
        return FDCPWTimeUniform(ck, ref, centres="argmax", scheme="b", name=name)
    if name == "FDC-PW(a)":
        return FDCPW(ref, scheme="a")
    if name == "TU-FDC-LS(a)":
        from dsswm.baselines.fdc_ls import FDCLSTimeUniform
        return FDCLSTimeUniform(ck)
    raise KeyError(name)


def job(a):
    name, seed, eps = a
    from dsswm.envs.seg_v10 import run_stream_seg
    E = G._ENV
    ctx = E["ctxs"][eps]
    try:
        m = make_pw(name, ctx.checkpoints.copy())
        t0 = time.perf_counter()
        s, _ = run_stream_seg(E["env"], m, seed, ctx, E["Js"], E["mu"], n80_k=G.N80_K)
        out = {"cell": E["cid"], "method": name, "seed": int(seed), "eps": float(eps), "label": "v11 dev (FDC-PW)"}
        out.update({k: s[k] for k in ("k80", "N80_pen", "N80_raw", "n80_lt_tau", "exhaustion_at_k80", "n_cert",
                                      "n_false", "fwer_event", "cert_k", "tau_R", "sec_cert_by_k")})
        cs = m.cert_stats
        out.update({"sec": round(time.perf_counter() - t0, 3), "beta_J": float(m.ledger["beta"]),
                    "sec_ck_max": float(max(x["sec"] for x in cs)) if cs else None, "error": None})
        for k in ("ref_certified", "argmax_certified", "ball_certified"):
            if cs and k in cs[0]:
                out[k] = int(sum(x.get(k, 0) for x in cs))
        for k in ("pw_ref_beta", "ls_D", "ls_beta_1", "ls_beta_far", "ls_beta_D"):
            if k in m.ledger:
                out[k] = m.ledger[k]
        return out
    except Exception:  # noqa: BLE001
        return {"method": name, "seed": int(seed), "eps": float(eps), "error": traceback.format_exc()}


def run(cid, methods, eps_list, workers):
    from dsswm.envs.pw_reference import pw_ref_vals
    C = G.CELLS[cid]
    if C["half"] != "dev":
        raise SystemExit("v11 dev runner: dev cells only")
    G.init_env(cid)
    REF["ref"], meta = pw_ref_vals(C["data"], C["S"])
    ROWS.mkdir(parents=True, exist_ok=True)
    f = ROWS / f"{cid}.jsonl"
    done = set()
    if f.exists():
        for l in f.read_text().splitlines():
            x = json.loads(l)
            if x.get("error") is None:
                done.add((x["method"], x["seed"], x["eps"]))
    jobs = [(m, s, float(e)) for m in methods for e in eps_list for s in C["seeds"] if (m, s, float(e)) not in done]
    print(f"[{cid}] {len(jobs)} jobs", flush=True)
    (ROWS / f"{cid}.ref.json").write_text(json.dumps(meta))
    with get_context("fork").Pool(workers) as pool, open(f, "a") as fo:
        for r in pool.imap_unordered(job, jobs, chunksize=1):
            if r.get("error"):
                print(r["error"][-600:], flush=True)
                continue
            fo.write(json.dumps(r) + "\n")
            fo.flush()
    print(f"[{cid}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--methods", default="TU-FDC-PW(b)")
    ap.add_argument("--eps", default="all")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for cid in a.cells.split(","):
        eps = G.CELLS[cid]["eps"] if a.eps == "all" else tuple(float(x) for x in a.eps.split(","))
        run(cid, a.methods.split(","), eps, a.workers)
