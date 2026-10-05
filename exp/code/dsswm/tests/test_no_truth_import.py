"""(2) Learner code paths never import ground-truth modules or read true parameters."""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from dsswm.envs.base import TruthAccessError
from dsswm.streams.generator import make_lin_instance, make_nl_instance

PKG = Path(__file__).resolve().parents[1]
LEARNER_DIRS = ["core", "models", "exact", "evidence", "certify", "acquire", "baselines", "audit", "stats"]
LEARNER_FILES = ["streams/policies.py", "streams/utilities.py"]
FORBIDDEN = ("dsswm.envs", "dsswm.streams.generator", "envs", "streams.generator", "generator",
             "dsswm.streams.offgrid", "streams.offgrid", "offgrid", "dsswm.streams.lin_oos", "streams.lin_oos", "lin_oos")
TRUTH_SUFFIXES = ("generator", "offgrid", "lin_oos", "gap_quota", "r3_harness", "lin_stream")
# attribute names that only ground-truth objects expose; learner modules must never touch them
TRUTH_ATTRS = ("true_params", "true_theta_vector", "simulate_batch", "truth", "_alpha", "_beta", "_gamma", "_tau",
               "_lam", "_psi_left", "_syn", "env_truth")


def _imports(path: Path):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.append(("." * node.level) + mod)
            out += [("." * node.level) + mod + "." + a.name for a in node.names]
    return out


def _learner_files():
    files = [p for d in LEARNER_DIRS for p in (PKG / d).glob("*.py")]
    return files + [PKG / f for f in LEARNER_FILES]


@pytest.mark.parametrize("path", _learner_files(), ids=lambda p: str(p.relative_to(PKG)))
def test_static_no_truth_imports(path):
    for name in _imports(path):
        stripped = name.lstrip(".")
        bad = any(stripped == f or stripped.startswith(f + ".") for f in FORBIDDEN) or ".envs" in name or \
            any(stripped.endswith(x) for x in TRUTH_SUFFIXES)
        assert not bad, f"{path} imports truth module {name}"


@pytest.mark.parametrize("path", _learner_files(), ids=lambda p: str(p.relative_to(PKG)))
def test_static_no_truth_attribute_access(path):
    """Learner modules never read truth-only attributes (e.g. env.true_params(), inst.truth, env._alpha)."""
    tree = ast.parse(path.read_text())
    hits = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr in TRUTH_ATTRS
            and not (isinstance(n.value, ast.Name) and n.value.id in ("self", "cls"))]   # own members are fine
    assert not hits, f"{path} accesses truth attributes {hits}"


def _learner_module_names():
    out = []
    for p in _learner_files():
        rel = p.relative_to(PKG.parent).with_suffix("")
        if rel.name != "__init__":
            out.append(".".join(rel.parts))
    return sorted(out)


def test_runtime_learner_import_closure_excludes_envs():
    """Import EVERY learner module (round 0 + round 1, discovered by glob) and check no truth module is loaded."""
    mods = _learner_module_names()
    assert "dsswm.certify.eta_loc" in mods and "dsswm.models.grid_ladder" in mods and "dsswm.baselines.b12_tas" in mods
    code = ("import sys, importlib; sys.path.insert(0, %r)\n" % str(PKG.parent) +
            "for m in %r: importlib.import_module(m)\n" % mods +
            "bad=[m for m in sys.modules if m.startswith('dsswm.envs') or m in %r]\n" %
            (("dsswm.streams.generator", "dsswm.streams.offgrid", "dsswm.streams.lin_oos"),) +
            "assert not bad, bad\nprint('ok')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr


def test_handle_hides_truth():
    for inst in (make_lin_instance(0), make_nl_instance(0)):
        h = inst.env.handle()
        for attr in ("_alpha", "_beta", "_gamma", "_psi", "_sigma", "_tau", "_lam", "true_theta_vector",
                     "true_params", "simulate_batch", "rng", "_env", "_EnvHandle__env"):
            with pytest.raises(TruthAccessError):
                getattr(h, attr)
        with pytest.raises(TruthAccessError):
            h.loads = None
        obs = h.step(0)
        assert obs.t == inst.cfg["n0"]
