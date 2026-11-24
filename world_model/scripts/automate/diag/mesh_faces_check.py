"""Socket/plug mesh structure: back-to-back duplicate faces (same three vertices, opposite winding = a zero-thickness
sheet), their total area and height range; number of connected shells. Also, for each shell, its volume and z range.
    python -u mesh_faces_check.py 00141,01092,...
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402

for aid in sys.argv[1].split(","):
    for part in ("socket", "plug"):
        m = trimesh.load(A.fetch(f"{aid}/{part}.obj"), force="mesh")
        key = np.sort(m.faces, axis=1)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        dup = cnt[inv.ravel()] > 1
        shells = m.split(only_watertight=False)
        info = [(round(float(s.volume * 1e9), 1), round(float(s.bounds[0, 2] * 1e3), 2), round(float(s.bounds[1, 2] * 1e3), 2),
                 round(float(s.bounds[0, 0] * 1e3), 2), round(float(s.bounds[1, 0] * 1e3), 2)) for s in shells]
        z = m.triangles_center[dup][:, 2] * 1e3 if dup.any() else np.zeros(0)
        print(aid, part, "faces", len(m.faces), "faces in back-to-back pairs", int(dup.sum()),
              "area mm2", round(float(m.area_faces[dup].sum() * 1e6), 2),
              "z of those mm", (np.unique(np.round(z, 2))[:10]).tolist(), "shells (vol mm3, zmin, zmax, xmin, xmax)", info[:6],
              flush=True)
