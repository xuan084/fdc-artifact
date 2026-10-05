"""Resolution ladder G_f over the E1-NL-S class box, Voronoi cells and the inflated set Theta_t^+ (learner side).

G_f: the E1-NL-S grid with only the alpha and psi step sizes multiplied by 1/f (methodology section 2.2):
  f = 0.5 -> alpha, psi in 2 points (box endpoints)  |Theta| = 4096
  f = 1   -> 3 points (the round-0 grid)              |Theta| = 13824
  f = 2   -> 5 points                                 |Theta| = 64000
  f = 4   -> 9 points                                 |Theta| = 373248
Class box (fixed for every f): alpha [-1.5, 1.5], gamma [0, 0.75], tau [0.5, 1.5], beta [-1, 1], psi [0, 1.5],
lam [0, 0.5].  The cell C(g) of a grid point is the Voronoi cell of the Cartesian grid clipped to the class box,
i.e. a box: per coordinate [midpoint to lower neighbour (or box lo), midpoint to upper neighbour (or box hi)].
Theta_t^+ = union of the cells of the grid points alive in Theta_t; it is represented by the alive mask itself.

Vector order (same as acquire.nl_factor_acq.theta_vector / baselines.glm_linearised.theta_vec):
  v = (alpha_1..L, gamma_1..L, tauL_1..L, beta_1..R, tauR_1..R, psi, lam)          d = 3L + 2R + 2
Nothing here reads ground truth; `cell_index_of` takes an arbitrary vector (the harness may pass theta*).
"""
from __future__ import annotations

import numpy as np

from .nl_class import NLClass, NLGrid

BASE = dict(L=2, R=2, alpha=(-1.5, 0.0, 1.5), gamma=(0.0, 0.75), tauL=(0.5, 1.5), beta=(-1.0, 1.0),
            tauR=(0.5, 1.5), psi=(0.0, 0.75, 1.5), lam=(0.0, 0.5))
BOX = {"alpha": (-1.5, 1.5), "gamma": (0.0, 0.75), "tauL": (0.5, 1.5), "beta": (-1.0, 1.0), "tauR": (0.5, 1.5),
       "psi": (0.0, 1.5), "lam": (0.0, 0.5)}
LADDER = (0.5, 1, 2, 4)


def _axis(lo: float, hi: float, f: float) -> tuple:
    n = int(round(2 * f)) + 1          # base step = (hi - lo) / 2 ; step / f
    return tuple(float(x) for x in np.round(np.linspace(lo, hi, n), 12))


def ladder_grid(f: float) -> NLGrid:
    g = dict(BASE)
    g["alpha"] = _axis(*BOX["alpha"], f)
    g["psi"] = _axis(*BOX["psi"], f)
    return NLGrid(**g)


def dose_grid() -> NLGrid:
    """HD1 dynamic reference arm: gamma in {0, .375, .75}, lam in {0, .25, .5}; |Theta| = 46656."""
    g = dict(BASE)
    g["gamma"] = (0.0, 0.375, 0.75)
    g["lam"] = (0.0, 0.25, 0.5)
    return NLGrid(**g)


def coord_names(L: int, R: int):
    return ([f"alpha{i}" for i in range(L)] + [f"gamma{i}" for i in range(L)] + [f"tauL{i}" for i in range(L)]
            + [f"beta{j}" for j in range(R)] + [f"tauR{j}" for j in range(R)] + ["psi", "lam"])


def _coord_axes(grid: NLGrid):
    L, R = grid.L, grid.R
    return ([grid.alpha] * L + [grid.gamma] * L + [grid.tauL] * L + [grid.beta] * R + [grid.tauR] * R
            + [grid.psi, grid.lam])


def _box_axes(grid: NLGrid):
    L, R = grid.L, grid.R
    bx = lambda k, vals: BOX.get(k, (min(vals), max(vals)))  # noqa: E731
    return ([bx("alpha", grid.alpha)] * L + [bx("gamma", grid.gamma)] * L + [bx("tauL", grid.tauL)] * L
            + [bx("beta", grid.beta)] * R + [bx("tauR", grid.tauR)] * R + [bx("psi", grid.psi), bx("lam", grid.lam)])


def axis_cells(values, lo: float, hi: float):
    """Per-value Voronoi interval on one axis, clipped to [lo, hi]."""
    v = np.asarray(values, float)
    mids = (v[1:] + v[:-1]) / 2
    return np.concatenate([[lo], mids]), np.concatenate([mids, [hi]])


def class_vectors(ncl: NLClass) -> np.ndarray:
    p = ncl.np_params
    return np.concatenate([p["alpha"], p["gamma"], p["tauL"], p["beta"], p["tauR"], p["psi_left"][:, 0, 1:2],
                           np.asarray(p["lam"])[:, None]], 1)


def cell_boxes(ncl: NLClass):
    """(lo, hi) arrays of shape (B, d): the clipped Voronoi box of every grid point."""
    V = class_vectors(ncl)
    lo, hi = np.empty_like(V), np.empty_like(V)
    for k, (vals, (blo, bhi)) in enumerate(zip(_coord_axes(ncl.grid), _box_axes(ncl.grid))):
        vals = np.asarray(sorted(vals), float)
        clo, chi = axis_cells(vals, blo, bhi)
        pos = np.searchsorted(vals, V[:, k])
        lo[:, k], hi[:, k] = clo[pos], chi[pos]
    return lo, hi


def half_widths(ncl: NLClass) -> np.ndarray:
    """(B, d) max distance from the grid point to its cell boundary per coordinate."""
    V = class_vectors(ncl)
    lo, hi = cell_boxes(ncl)
    return np.maximum(V - lo, hi - V)


def nearest_vector(v: np.ndarray, grid: NLGrid) -> np.ndarray:
    v = np.atleast_2d(np.asarray(v, float))
    out = np.empty_like(v)
    for k, vals in enumerate(_coord_axes(grid)):
        a = np.asarray(vals, float)
        out[:, k] = a[np.argmin(np.abs(v[:, k:k + 1] - a[None]), 1)]
    return out


def cell_index_of(v: np.ndarray, ncl: NLClass) -> np.ndarray:
    """Index of the grid point whose cell contains each vector (ties -> lower value)."""
    return _fast_cell_index(np.atleast_2d(np.asarray(v, float)), ncl)


def sample_in_cells(ncl: NLClass, idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    lo, hi = cell_boxes(ncl)
    lo, hi = lo[idx], hi[idx]
    return lo + (hi - lo) * rng.random(lo.shape)


def theta_plus_mask(mask: np.ndarray) -> np.ndarray:
    """Theta_t^+ is the union of the alive cells; as a set of cells it is indexed by the alive mask."""
    return np.asarray(mask, bool)


def lift_mask(mask_src: np.ndarray, ncl_src: NLClass, ncl_dst: NLClass) -> np.ndarray:
    """Cells of another grid whose grid point lies in an alive cell of `ncl_src` (diagnostic only)."""
    return np.asarray(mask_src, bool)[_fast_cell_index(class_vectors(ncl_dst), ncl_src)]


def _fast_cell_index(V: np.ndarray, ncl: NLClass) -> np.ndarray:
    """Vectorised cell_index_of using the mixed-radix layout of NLClass."""
    g = ncl.grid
    L, R = g.L, g.R
    ax = _coord_axes(g)
    dig = [np.argmin(np.abs(V[:, k:k + 1] - np.asarray(a, float)[None]), 1) for k, a in enumerate(ax)]
    nal, ng, nt = len(g.alpha), len(g.gamma), len(g.tauL)
    nb_, ntr = len(g.beta), len(g.tauR)
    nlam = len(g.lam)
    left = [dig[i] * ng * nt + dig[L + i] * nt + dig[2 * L + i] for i in range(L)]
    right = [dig[3 * L + j] * ntr + dig[3 * L + R + j] for j in range(R)]
    glob = dig[-2] * nlam + dig[-1]
    if g.psi2:
        raise NotImplementedError
    idx = np.zeros(V.shape[0], dtype=np.int64)
    for d_, r in zip(left + right + [glob], ncl._radices):
        idx = idx * r + d_
    return idx
