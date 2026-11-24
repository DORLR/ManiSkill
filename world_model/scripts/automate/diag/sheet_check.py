"""Horizontal faces of the socket at the height where trimesh's sign went wrong: area facing up / down and the radial
extent of each, to show a zero-thickness double surface (coincident up- and down-facing faces).
    python -u sheet_check.py 00141:29.246,00175:0.001,00360:0.0
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402

for item in sys.argv[1].split(","):
    aid, z = item.split(":")
    z = float(z) / 1e3
    m = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    c = m.triangles_center
    n = m.face_normals
    at = (np.abs(m.triangles[:, :, 2] - z).max(axis=1) < 2e-6)
    for s, name in ((1, "up"), (-1, "down")):
        sel = at & (n[:, 2] * s > 0.99)
        r = np.hypot(m.triangles[sel][:, :, 0], m.triangles[sel][:, :, 1]) * 1e3 if sel.any() else np.zeros(1)
        rc = np.hypot(c[sel, 0], c[sel, 1]) * 1e3 if sel.any() else np.zeros(1)
        print(aid, f"z={z * 1e3:.3f} mm", name, "faces", int(sel.sum()), "area mm2", round(float(m.area_faces[sel].sum() * 1e6), 2),
              "vertex radius min/max mm", round(float(r.min()), 2), round(float(r.max()), 2), "centroid radius min mm", round(float(rc.min()), 2))
    # a vertical ray through the axis region: hits and their z (cast downwards from above the socket)
    o = np.array([[0.0005, 0.0003, m.bounds[1, 2] + 0.01]])
    loc, _, tri = m.ray.intersects_location(o, np.array([[0.0, 0.0, -1.0]]), multiple_hits=True)
    print("   ray down at r=0.6 mm: hits z mm", np.round(np.sort(loc[:, 2])[::-1] * 1e3, 3).tolist(), "normal z", np.round(n[tri, 2], 2).tolist())
