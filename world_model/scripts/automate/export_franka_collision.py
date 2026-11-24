"""Export the collision meshes of panda_hand / panda_leftfinger / panda_rightfinger from AutoMate's franka_mimic.usd
(each in its own body frame) to <assets>/franka_mimic_collision/<link>.obj, for grasp.py's finger-socket check.
Also prints the collision approximation of every collision prim. Run on rs2 (needs the Isaac app to resolve the asset):
    python -u export_franka_collision.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.sim_automate.app import launch, shutdown  # noqa: E402

app = launch(headless=True, cameras=False)

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from isaaclab_tasks.direct.automate.assembly_tasks_cfg import ASSET_DIR  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from wm.sim_automate import assets as A  # noqa: E402

out = A.ASSET_DIR / "franka_mimic_collision"
out.mkdir(parents=True, exist_ok=True)
stage = Usd.Stage.Open(f"{ASSET_DIR}/franka_mimic.usd")
cache = UsdGeom.XformCache()
bodies = {p.GetName(): p for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
          if p.GetName() in ("panda_hand", "panda_leftfinger", "panda_rightfinger") and p.HasAPI(UsdPhysics.RigidBodyAPI)}
print("bodies", {k: str(v.GetPath()) for k, v in bodies.items()}, flush=True)
for name, body in bodies.items():
    Vs, Fs, n0 = [], [], 0
    Tb = np.array(cache.GetLocalToWorldTransform(body)).T                   # USD row-vector convention -> column
    for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not p.HasAPI(UsdPhysics.CollisionAPI) or not p.IsA(UsdGeom.Mesh):
            continue
        appr = p.GetAttribute("physics:approximation").Get() if p.HasAttribute("physics:approximation") else None
        m = UsdGeom.Mesh(p)
        pts = np.array(m.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.array(m.GetFaceVertexCountsAttr().Get())
        idx = np.array(m.GetFaceVertexIndicesAttr().Get())
        tris, k = [], 0
        for c in counts:                                                      # fan triangulation
            tris += [[idx[k], idx[k + j], idx[k + j + 1]] for j in range(1, c - 1)]
            k += c
        Tp = np.array(cache.GetLocalToWorldTransform(p)).T
        T = np.linalg.inv(Tb) @ Tp                                            # mesh -> body frame
        P = pts @ T[:3, :3].T + T[:3, 3]
        print(name, p.GetPath(), "approximation", appr, "mesh->body", np.round(T, 4).tolist(), "points", len(pts), "tris", len(tris),
              "bounds", np.round(P.min(0), 4).tolist(), np.round(P.max(0), 4).tolist(), flush=True)
        if appr == "boundingCube":                                            # box of the mesh in its own frame
            lo, hi = pts.min(0), pts.max(0)
            box = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
            box.apply_transform(T)
            P, tris = box.vertices, box.faces.tolist()
        elif appr in ("convexHull", "convexDecomposition", "boundingSphere"):
            hull = trimesh.Trimesh(P, np.array(tris)).convex_hull               # what PhysX collides with (upper
            P, tris = hull.vertices, hull.faces.tolist()                          # bound for a decomposition)
        Vs.append(P)
        Fs.append(np.array(tris) + n0)
        n0 += len(P)
    if Vs:
        trimesh.Trimesh(np.vstack(Vs), np.vstack(Fs), process=False).export(out / f"{name}.obj")
print("wrote", sorted(str(p) for p in out.glob("*.obj")), flush=True)
shutdown(0)
