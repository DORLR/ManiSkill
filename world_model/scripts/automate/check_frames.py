"""Bit-identity of wm.sim_automate.frames.frame() with the existing frame builders where |a_z| < 0.9.

explore.frame() is taken from the source of wm/collect/explore.py (that module imports ManiSkill, so the function
is extracted with ast and executed alone); wm.sim.geometry.port_axes() is imported.
    python check_frames.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))
from wm.sim.geometry import port_axes  # noqa: E402
from wm.sim_automate.frames import frame, lateral_axes  # noqa: E402

tree = ast.parse((SRC / "wm/collect/explore.py").read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "frame")
ns = {"np": np}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "explore_frame", "exec"), ns)
explore_frame = ns["frame"]

rng = np.random.default_rng(0)
n_same_ex = n_same_pa = n = 0
for _ in range(200_000):
    a = rng.normal(size=3)
    if abs(a[2] / np.linalg.norm(a)) >= 0.9:
        continue
    n += 1
    F = frame(a)
    n_same_ex += np.array_equal(F, explore_frame(a))
    n_same_pa += np.array_equal(F, port_axes(a))
print(f"|a_z| < 0.9: {n} axes, bit-identical to explore.frame: {n_same_ex}, to geometry.port_axes: {n_same_pa}")
for a in (np.array([0.0, 0.0, -1.0]), np.array([1e-12, 0.0, -1.0]), np.array([0.02, -0.01, -1.0])):
    with np.errstate(all="ignore"):
        old = explore_frame(a.copy())
    F = frame(a)
    e1, e2 = lateral_axes(a)
    print("a", a, "explore.frame finite:", bool(np.isfinite(old).all()), "| new e1", np.round(e1, 4), "e2",
          np.round(e2, 4), "orthonormal:", bool(np.allclose(F.T @ F, np.eye(3))))
