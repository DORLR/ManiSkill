"""Plug-socket overlap as a function of the plug's lift d out of the CAD seated pose (axis kept), from cross-sections on
a 0.05 mm grid (fit.py sections): A(d) = sum over planes of area(P(z - d) AND S(z)) * dz  [mm^3], and the largest
sliver thickness 2 area / perimeter over the planes. d_free = smallest d on the grid with A(d) = 0.
    python -u lift_overlap.py 01092,01026,00597,00855,00755 [max_lift_mm=3]
"""
import sys
from pathlib import Path

import numpy as np
import shapely
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import fit as F  # noqa: E402

DZ = 0.00005
max_lift = float(sys.argv[2]) / 1e3 if len(sys.argv) > 2 else 0.003
for aid in sys.argv[1].split(","):
    plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
    sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    z0 = max(float(plug.bounds[0, 2]), float(sock.bounds[0, 2])) + DZ / 3
    z1 = float(sock.bounds[1, 2])
    zs = np.arange(z0, z1, DZ)
    S = F._sections(sock, zs)
    P = F._sections(plug, zs)                    # plug at the seated pose; lifting by k steps pairs P[i - k] with S[i]
    rows = []
    for k in range(0, int(round(max_lift / DZ)) + 1):
        vol, thick, zmax = 0.0, 0.0, None
        for i in range(k, len(zs)):
            p, s = P[i - k], S[i]
            if p.is_empty or s.is_empty:
                continue
            inter = shapely.intersection(p, s)
            if inter.area > 1e-11:
                vol += inter.area * DZ
                t = 2 * inter.area / max(inter.length, 1e-12)
                if t > thick:
                    thick, zmax = t, zs[i]
        rows.append((k * DZ * 1e3, vol * 1e9, thick * 1e3, None if zmax is None else zmax * 1e3))
    free = next((d for d, v, _, _ in rows if v == 0.0), None)
    print(aid, "d_free_mm", free, "A(d) [d mm, overlap mm3, max thick mm, at z mm]:",
          [tuple(round(x, 3) if x is not None else None for x in r) for r in rows[::4]], flush=True)
