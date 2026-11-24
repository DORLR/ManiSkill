"""Side sections for the milestone-1b diagnosis (no Isaac): socket, plug, finger and hand collision shapes cut by a plane
through the insertion axis, in the socket frame (z up = against the insertion direction; plug seated at the CAD pose =
identity). Two poses per figure:
  left   CAD seated pose with the nominal grasp (hand / fingers from grasp.FingerSocketCheck.frames at q_f)
  right  the simulator pose at the end of a diagnosis experiment (diag/sim/<id>.json, poses_socket_frame)
Plane: through the socket axis, spanned by z and a horizontal direction u: "hand_y" = the finger-closing axis of the
nominal grasp (both fingers in the cut), or "deepest" = towards the deepest plug point inside the socket (mesh_diag).
Finger / hand shapes are the PhysX collision shapes (assets/franka_mimic_collision, as grasp.py).
    python plot_sections.py --id 00681 --exp ii --plane hand_y --zoom 0,40,30 --out .../diag/figs
--zoom u0,z0,half: a second row zoomed on the square centred (u0, z0) mm with half-width `half` mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402


def segments(mesh, T, origin, normal, u, w):
    import trimesh
    m = mesh.copy()
    m.apply_transform(T)
    lines = trimesh.intersections.mesh_plane(m, normal, origin)
    if len(lines) == 0:
        return np.zeros((0, 2, 2))
    d = lines - origin
    return np.stack([d @ u, d @ w], -1) * 1e3                                      # mm


def polygons(mesh, T, origin, normal, u, w):
    """Filled section polygons in (u, w) [mm] from trimesh's closed section loops (None if the loops do not close)."""
    m = mesh.copy()
    m.apply_transform(T)
    path = m.section(plane_origin=origin, plane_normal=normal)
    if path is None:
        return []
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = u, w, normal, origin
    try:
        planar, _ = path.to_planar(to_2D=np.linalg.inv(F), check=False)
        return [p for p in planar.polygons_full if p is not None]
    except Exception:                                                              # noqa: BLE001
        return []


def fill(ax, polys, color, alpha):
    for p in polys:
        x, y = p.exterior.xy
        ax.fill(np.asarray(x) * 1e3, np.asarray(y) * 1e3, color=color, alpha=alpha, lw=0)
        for r in p.interiors:
            xi, yi = r.xy
            ax.fill(np.asarray(xi) * 1e3, np.asarray(yi) * 1e3, color="white", alpha=1.0, lw=0)


def draw(ax, bodies, origin, normal, u, w, title, lim=None):
    from matplotlib.collections import LineCollection
    style = {"socket": ("#7a7a7a", 0.35, 1.2), "plug": ("#2f6fb3", 0.35, 1.0), "leftfinger": ("#d9822b", 0.25, 1.0),
             "rightfinger": ("#d9822b", 0.25, 1.0), "hand": ("#3a9a5b", 0.15, 0.8)}
    for name, (mesh, T) in bodies.items():
        c, a, lw = style[name]
        s = segments(mesh, T, origin, normal, u, w)
        fill(ax, polygons(mesh, T, origin, normal, u, w), c, a)
        if len(s):
            ax.add_collection(LineCollection(s, colors=c, linewidths=lw))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("across the axis [mm]", fontsize=8)
    ax.set_ylabel("along the axis, socket frame z [mm]", fontsize=8)
    ax.tick_params(labelsize=7)
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    else:
        ax.autoscale_view()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--exp", default="ii")
    ap.add_argument("--plane", default="hand_y")
    ap.add_argument("--zoom", default="")
    ap.add_argument("--view", default="", help="u0,u1,z0,z1 limits of the full view [mm]")
    ap.add_argument("--sim", default="/mnt/storage3/qingxin/automate/m1b/diag/sim")
    ap.add_argument("--mesh", default="/mnt/storage3/qingxin/automate/m1b/diag/mesh")
    ap.add_argument("--out", default="/mnt/storage3/qingxin/automate/m1b/diag/figs")
    a = ap.parse_args()
    aid = a.id
    load = lambda p: trimesh.load(p, force="mesh")
    plug, sock = load(A.fetch(f"{aid}/plug.obj")), load(A.fetch(f"{aid}/socket.obj"))
    d = A.ASSET_DIR / "franka_mimic_collision"
    hand, left, right = load(d / "panda_hand.obj"), load(d / "panda_leftfinger.obj"), load(d / "panda_rightfinger.obj")
    chk = G.FingerSocketCheck(aid)
    q, _ = chk.closing([0.0, 0.0, 0.0])
    Th, Tl, Tr = chk.frames([0.0, 0.0, 0.0], q)
    cad = {"socket": (sock, np.eye(4)), "plug": (plug, np.eye(4)), "leftfinger": (left, Tl), "rightfinger": (right, Tr),
           "hand": (hand, Th)}
    w = np.array([0.0, 0.0, 1.0])
    if a.plane == "hand_y":
        y = Th[:3, 1]
    else:
        m = json.loads((Path(a.mesh) / f"{aid}.json").read_text())
        y = np.array(m["penetration"]["deepest_point_mm"], dtype=float)
    u = y - (y @ w) * w
    u /= np.linalg.norm(u)
    normal = np.cross(u, w)
    origin = np.zeros(3)
    panels = [("CAD seated pose, nominal grasp", cad)]
    if a.exp:
        s = json.loads((Path(a.sim) / f"{aid}.json").read_text())[a.exp]
        P = {k: np.array(v) for k, v in s["poses_socket_frame"].items()}
        sim = {"socket": (sock, np.eye(4)), "plug": (plug, P["plug"]), "leftfinger": (left, P["leftfinger"]),
               "rightfinger": (right, P["rightfinger"]), "hand": (hand, P["hand"])}
        panels.append((f"simulator, end of {a.exp}: depth left {s['depth_left_mm']:.2f} mm", sim))
    rows = 2 if a.zoom else 1
    fig, axs = plt.subplots(rows, len(panels), figsize=(5.2 * len(panels), 5.2 * rows), squeeze=False)
    view = [float(v) for v in a.view.split(",")] if a.view else None
    for j, (t, b) in enumerate(panels):
        draw(axs[0, j], b, origin, normal, u, w, t, view)
        if a.zoom:
            u0, z0, h = [float(v) for v in a.zoom.split(",")]
            draw(axs[1, j], b, origin, normal, u, w, "zoom", (u0 - h, u0 + h, z0 - h, z0 + h))
    fig.suptitle(f"{aid}, cut through the axis ({'finger-closing plane' if a.plane == 'hand_y' else 'plane through the deepest overlap'})\n"
                 "grey socket, blue plug, orange fingers, green hand", fontsize=9)
    fig.tight_layout()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    f = Path(a.out) / f"{aid}_{a.exp or 'cad'}_{a.plane}{'_zoom' if a.zoom else ''}.png"
    fig.savefig(f, dpi=110)
    print("wrote", f)


if __name__ == "__main__":
    main()
