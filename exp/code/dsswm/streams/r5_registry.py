"""Round-5 method registry: one factory and one metadata table for every frontier method (r4 + r5), unified names.

r4 modules are frozen (6297e7f2); this module only imports them. New r5 rivals live in
``baselines/{b4_bal, hait_sw, molitor_wor}.py``. FDC itself is produced by task r5_fdc_impl; it is resolved lazily
from ``dsswm.baselines.fdc`` (class ``FDCMethod``) when that module exists. Other r5 tasks (e.g. the tuned plug-in
variants of r5_setup_plugin_variants) add their methods with ``register`` instead of editing this file.

Validity classes (methodology s4.1-4.3):
  rigorous    finite-sample, time-uniform guarantee that covers the WoR replay + adaptive counts + checkpoint
              stopping + one delta over the 15 problems (members of R are drawn from here);
  ablation    rigorous code path, but a component switch of FDC (QFC-half; C4 factorial) -- never a rival;
  nominal     guarantee proven only with replacement / i.i.d. (unproven here); never promoted, whatever its FWER;
  none        plug-in (no guarantee; E-cost descriptive comparators);
  asymptotic  asymptotic validity only (B5 maq bootstrap);
  model       validity only inside a parametric model class (H8-*).
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["MethodSpec", "REGISTRY", "RIGOROUS_SET_R", "TUNING_GRID", "make_method", "register", "spec",
           "names", "rigorous_set"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    validity_class: str            # rigorous / ablation / nominal / none / asymptotic / model / main
    role: str                      # main / rival_R / ablation / descriptive / excluded
    in_R: bool
    source: str
    module: str
    defaults: dict = field(default_factory=dict)
    note: str = ""


def _fdc(**kw):
    try:
        from ..baselines.fdc import FDCMethod  # produced by r5_fdc_impl
    except ImportError as e:  # pragma: no cover - depends on task order
        raise KeyError("FDC is not implemented yet (task r5_fdc_impl: dsswm/baselines/fdc.py:FDCMethod)") from e
    return FDCMethod(**kw)


def _qfc_pool(**kw):
    from ..baselines.frontier_common import QFCMethod
    return QFCMethod("QFC-pool", **kw)


def _qfc_half(**kw):
    import numpy as np

    from ..baselines.frontier_common import QFCMethod

    class _QFCHalf(QFCMethod):
        """50/50 frozen design + r4 ledger (C4 ablation; alloc matrix built from ctx)."""

        def setup(self, ctx):
            row = np.full(ctx.A, 0.5 / max(ctx.A - 1, 1))
            row[0] = 0.5
            self.alloc_p = np.tile(row, (ctx.S, 1))
            self.alloc_kind = "fixed"
            super().setup(ctx)

    m = _QFCHalf("QFC-half", alloc_p=np.full((1, 2), 0.5), **kw)
    m.validity = "rigorous"
    return m


def _b1(**kw):
    from ..baselines.clucb_joint import CLUCBJoint
    return CLUCBJoint()


def _b4(**kw):
    from ..baselines.uniform_rs import UniformRS
    return UniformRS()


def _b4_bal(share=0.5, **kw):
    from ..baselines.b4_bal import B4Bal
    return B4Bal(share=share)


def _b2(variant):
    def f(**kw):
        from ..baselines.combgame_joint import CombGameJoint
        return CombGameJoint(variant, **kw)
    return f


def _b3(variant):
    def f(**kw):
        from ..baselines.rage_frontier import RAGEFrontier
        return RAGEFrontier(variant, **kw)
    return f


def _peace(variant):
    def f(**kw):
        from ..baselines.peace_frontier import PeaceFrontier
        return PeaceFrontier(variant, **kw)
    return f


def _hait(p_min=0.05, gamma=2.0 / 3.0, lag=1, **kw):
    from ..baselines.hait_sw import HaitSW
    return HaitSW(p_min=p_min, gamma=gamma, lag=lag)


def _molitor(**kw):
    from ..baselines.molitor_wor import MolitorWoR
    return MolitorWoR()


def _b5(**kw):
    from ..baselines.maq_desc import MaqQini
    return MaqQini()


def _h8(kind):
    def f(env=None, **kw):
        from .frontier_runner import make_frontier_method
        return make_frontier_method(kind, env=env, **kw)
    return f


_FACT = {}
REGISTRY: dict = {}


def register(spec_: MethodSpec, factory):
    """Add (or replace) a method. Used by later r5 tasks; the name is the unified reporting name."""
    REGISTRY[spec_.name] = spec_
    _FACT[spec_.name] = factory


_RECT = "per-cell WSR20 WoR CS (Thm 4), delta/(S*A) per cell, radius-sum (qfc_lemma s6)"
for _s, _f in [
    (MethodSpec("FDC", "rigorous", "main", False, "Thm FDC-1 (plan/theory/fdc_theorem.md)", "baselines/fdc.py",
                note="no tuning; ledger from ctx"), _fdc),
    (MethodSpec("QFC-pool", "rigorous", "rival_R", True, "qfc_lemma Thm 1 (r4, signed)", "baselines/frontier_common.py",
                note="our r4 method"), _qfc_pool),
    (MethodSpec("B1", "rigorous", "rival_R", True, "Chen et al. 2014 CLUCB + " + _RECT, "baselines/clucb_joint.py"),
     _b1),
    (MethodSpec("B4", "rigorous", "rival_R", True, _RECT, "baselines/uniform_rs.py"), _b4),
    (MethodSpec("B4-bal", "rigorous", "rival_R", True, _RECT + "; frozen balanced design", "baselines/b4_bal.py",
                {"share": 0.5}), _b4_bal),
    (MethodSpec("B2-rect", "rigorous", "rival_R", True, "Jourdan et al. 2021 sampling + " + _RECT,
                "baselines/combgame_joint.py"), _b2("rect")),
    (MethodSpec("B3-rect", "rigorous", "rival_R", True, "Fiez et al. 2019 RAGE + " + _RECT,
                "baselines/rage_frontier.py"), _b3("rect")),
    (MethodSpec("Peace-rect", "rigorous", "rival_R", True, "Katz-Samuels et al. 2020 + " + _RECT,
                "baselines/peace_frontier.py"), _peace("rect")),
    (MethodSpec("Hait-SW", "rigorous", "rival_R", True, "Hait 2609.37873 s4.2/Thm 6.2/s7 + WSR20 WoR local CS",
                "baselines/hait_sw.py", {"p_min": 0.05, "gamma": 2.0 / 3.0, "lag": 1}), _hait),
    (MethodSpec("Molitor-WoR", "rigorous", "rival_R", True, "Molitor 2606.17515 Thm 1 (eps-relaxed) + WSR20 WoR",
                "baselines/molitor_wor.py", note="dominated by B4 by construction"), _molitor),
    (MethodSpec("QFC-half", "ablation", "ablation", False, "FDC code path with r4 ledger", "streams/r5_registry.py",
                note="C4 factorial cell; ratio to FDC is 1 under equal accounting"), _qfc_half),
    (MethodSpec("B2-nominal", "nominal", "excluded", False, "CombGame threshold; i.i.d. with replacement only",
                "baselines/combgame_joint.py"), _b2("nominal")),
    (MethodSpec("B3-nominal", "nominal", "excluded", False, "RAGE Gaussian/sub-G proxy", "baselines/rage_frontier.py"),
     _b3("nominal")),
    (MethodSpec("Peace-nominal", "nominal", "excluded", False, "Peace Alg. 1 sub-G proxy",
                "baselines/peace_frontier.py"), _peace("nominal")),
    (MethodSpec("B2-fav", "none", "descriptive", False, "plug-in GLR", "baselines/combgame_joint.py"), _b2("fav")),
    (MethodSpec("B3-fav", "none", "descriptive", False, "plug-in", "baselines/rage_frontier.py"), _b3("fav")),
    (MethodSpec("Peace-fav", "none", "descriptive", False, "Peace Alg. 2 plug-in (published config)",
                "baselines/peace_frontier.py"), _peace("fav")),
    (MethodSpec("B5", "asymptotic", "descriptive", False, "maq Qini bootstrap", "baselines/maq_desc.py"), _b5),
    (MethodSpec("H8-JPC1", "model", "descriptive", False, "in-model certifier", "baselines/h8_models.py"),
     _h8("H8-JPC1")),
    (MethodSpec("H8-Tboot", "model", "descriptive", False, "T-learner bootstrap", "baselines/h8_models.py"),
     _h8("H8-Tboot")),
    (MethodSpec("H8-MisLid", "model", "descriptive", False, "misspecification lid", "baselines/h8_models.py"),
     _h8("H8-MisLid")),
]:
    register(_s, _f)

#: rigorous rival set R (methodology s4.2), frozen before lock v5. C1 is an IUT over exactly these names.
RIGOROUS_SET_R = ("B1", "B4", "B4-bal", "B2-rect", "B3-rect", "Peace-rect", "Hait-SW", "Molitor-WoR", "QFC-pool")

#: pre-declared tuning grids (dev seeds 900-949; methodology s4.4 + qualification-table additions)
TUNING_GRID = {
    "B4-bal": [{"share": s} for s in (0.40, 0.45, 0.50)],
    "Hait-SW": [{"p_min": p, "gamma": g} for g in (2.0 / 3.0, 1.0) for p in (0.02, 0.05, 0.10)],
    "B2-rect": "forced-exploration constant in {sqrt(n_s), 0.5 sqrt(n_s)} (r5_rival_tuning; r4 module is frozen)",
    "B3-rect": "r4 frozen config (published constants, no tuning)",
    "Peace-rect": "r4 frozen config (published constants, no tuning)",
    "B1": "no free parameter", "B4": "no free parameter", "Molitor-WoR": "no free parameter",
    "QFC-pool": "no free parameter (r4 ledger)",
}


def names(role=None):
    return [n for n, s in REGISTRY.items() if role is None or s.role == role]


def rigorous_set():
    return list(RIGOROUS_SET_R)


def spec(name) -> MethodSpec:
    return REGISTRY[name]


def make_method(name, env=None, **kw):
    """Instantiate a method by unified name; ``kw`` overrides the spec defaults (tuning-grid points only)."""
    if name not in _FACT:
        raise KeyError(name)
    params = dict(REGISTRY[name].defaults)
    params.update(kw)
    if name.startswith("H8-"):
        return _FACT[name](env=env, **params)
    return _FACT[name](**params)


# ---------------------------------------------------------------------------------------------- r5 add-on modules
# Modules that call ``register`` at import time; imported here so that ``make_method`` resolves their names whenever
# the registry is imported (r5_setup_plugin_variants: B2-fav-tight / FIX-bal-fav / Peace-fav-bal).
from ..baselines import plugin_r5 as _plugin_r5  # noqa: E402,F401
# r5_rival_tuning: B2-rect factory with a tunable forced-exploration constant (explore_c=1.0 == r4 B2-rect).
from ..baselines import b2_rect_fe as _b2_rect_fe  # noqa: E402,F401
