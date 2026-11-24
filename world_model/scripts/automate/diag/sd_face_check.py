"""For the plug points that trimesh.signed_distance calls inside the socket while ray containment does not: the closest
socket triangle, its normal, and whether that normal points out of the material (ray containment just in front of /
behind the face).   python -u sd_face_check.py 00141,00175,00360
"""
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402

for aid in sys.argv[1].split(","):
    plug = trimesh.load(A.fetch(f"{aid}/plug.obj"), force="mesh")
    sock = trimesh.load(A.fetch(f"{aid}/socket.obj"), force="mesh")
    pts, _ = trimesh.sample.sample_surface_even(plug, 20000, seed=0)
    pts = np.vstack([pts, plug.vertices])
    sd = trimesh.proximity.signed_distance(sock, pts)
    bad = np.where(sd > 5e-5)[0]
    p = pts[bad]
    closest, dist, tid = trimesh.proximity.closest_point(sock, p)
    faces = np.unique(tid)
    n = sock.face_normals[faces]
    c = sock.triangles_center[faces]
    front = sock.contains(c + 2e-5 * n)
    back = sock.contains(c - 2e-5 * n)
    print(aid, "flagged points", len(bad), "closest faces", len(faces), "normal z of those faces", np.round(n[:, 2], 3)[:12],
          "face centre z mm", np.round(c[:, 2] * 1e3, 3)[:12], "inside just in front", int(front.sum()), "inside just behind",
          int(back.sum()))
    k = int(np.argmax(sd))
    j = int(np.where(bad == k)[0][0])
    print("   deepest point mm", np.round(pts[k] * 1e3, 3), "closest point mm", np.round(closest[j] * 1e3, 3), "dist mm",
          round(float(dist[j] * 1e3), 3), "triangle mm", np.round(sock.triangles[tid[j]] * 1e3, 3).tolist(), "normal",
          np.round(sock.face_normals[tid[j]], 3))
    # a second closest-point query on the single point (same mesh) and a brute-force check over all triangles
    tri = sock.triangles
    cp = trimesh.triangles.closest_point(tri, np.tile(pts[k], (len(tri), 1)))
    d_all = np.linalg.norm(cp - pts[k], axis=1)
    order = np.argsort(d_all)[:5]
    print("   brute force nearest triangles", order.tolist(), "dist mm", np.round(d_all[order] * 1e3, 4).tolist(),
          "normals z", np.round(sock.face_normals[order][:, 2], 3).tolist(), "closest_point() chose", int(tid[j]))
