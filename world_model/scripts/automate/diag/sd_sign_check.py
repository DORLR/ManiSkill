"""Why checks.penetration reports plug points inside the socket that are not inside: trimesh's signed_distance takes
the sign from the closest triangle's normal when the point projects into that triangle, else from ray containment.
For the flagged points this prints which branch set the sign and the ray-containment answer.
    python -u sd_sign_check.py 00141,00175,00360
"""
import inspect
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402

print("trimesh", trimesh.__version__)
print(inspect.getsource(trimesh.proximity.signed_distance)[-1800:])
for aid in sys.argv[1].split(","):
    plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
    sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    pts, _ = trimesh.sample.sample_surface_even(plug, 20000, seed=0)
    pts = np.vstack([pts, plug.vertices])
    sd = trimesh.proximity.signed_distance(sock, pts)
    bad = np.where(sd > 5e-5)[0]
    if not len(bad):
        print(aid, "no flagged points")
        continue
    p = pts[bad]
    closest, dist, tid = trimesh.proximity.closest_point(sock, p)
    nrm = sock.face_normals[tid]
    proj = p - (nrm.T * np.einsum("ij,ij->i", p - closest, nrm)).T
    bary = trimesh.triangles.points_to_barycentric(sock.triangles[tid], proj)
    ontri = ~((bary < -1e-8) | (bary > 1 + 1e-8)).any(axis=1)
    area = sock.area_faces[tid]
    print(aid, "flagged", len(bad), "sign from closest-triangle normal:", int(ontri.sum()), "from rays:", int((~ontri).sum()),
          "ray contains:", int(sock.contains(p).sum()), "closest-triangle area min/median mm2:",
          float(area.min() * 1e6), float(np.median(area) * 1e6),
          "n(.)(p - proj) sign:", np.unique(np.sign(np.einsum("ij,ij->i", nrm, p - proj))[ontri], return_counts=True),
          "nan bary:", int(np.isnan(bary).any(axis=1).sum()), "socket z max mm:", float(sock.bounds[1, 2] * 1e3),
          "flagged z max mm:", float(p[:, 2].max() * 1e3))
