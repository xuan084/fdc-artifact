"""Outcome-free reference values for FDC-PW (NEW file, v11).  ref_vals[s, 1] = w_s * mean frozen v10 uplift score of
dev replay rows in segment s (covariates only, frozen score model and cut points), ref_vals[s, 0] = 0.  The reference
policy ref_q is the knapsack argmax of ref_vals: treat the segments with the highest predicted uplift per unit cost
that the budget allows, and never a segment whose predicted uplift is <= 0.  Never reads an outcome."""
from __future__ import annotations

import numpy as np


def pw_ref_vals(data, S):
    from dsswm.envs import seg_v10_eval as E
    fz = E.load_frozen_seg(None, None)
    models = E._load_models(fz, data, None)
    Xd, armd, yd, rowsd, msk = E._dev_replay(data)
    X = Xd[~msk]
    score = E._score(models, X)
    if E._sha_arr(score) != fz["data"][data]["dev_replay_score_sha256"]:
        raise RuntimeError("frozen score model does not reproduce the frozen dev scores")
    cuts = [float.fromhex(c) for c in fz["data"][data]["S"][str(int(S))]["cuts_hex"]]
    seg = E.assign_segments(score, cuts)
    w = np.bincount(seg, minlength=S) / len(seg)
    m = np.array([score[seg == s].mean() for s in range(S)])
    ref = np.zeros((S, 2))
    ref[:, 1] = w * m
    return ref, {"score_mean": m.tolist(), "w_dev": w.tolist(), "source": "frozen v10 score model, dev replay rows"}
