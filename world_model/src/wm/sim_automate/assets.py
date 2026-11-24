"""AutoMate asset files and the plug/socket geometry, without Isaac (numpy, scipy, trimesh only).

Frames and units: metres; every mesh of an assembly is given in the ASSEMBLED frame, i.e. plug.obj and socket.obj
share one frame and the plug is fully seated (AutoMate's reset puts the held asset at the fixed asset's pose, then
lifts it along the socket's +z by the curriculum displacement, assembly_env.randomize_held_initial_state). So
    insertion direction (into the socket) = -z of the socket frame,
    seated plug pose = socket pose.

Insertion depth L (AssemblySpec.pin_length_m) = tight-fit depth from the lift sweep of fit.py:
    L = min { d >= 0 : g(d) > 2 * max(c, 0.1 mm) },  g(d) = lateral gap of the plug lifted by d (fit.py)
the entrance (start of the tight fit) is at z_ent = min_z(plug) + L. Also kept: L_top = max_z(socket) - min_z(plug)
and AutoMate's disassembly_dist.json d_dis (the lift at which its disassembly planner called the plug free).
Episode limit (decisions of 0.4 s): N_max = 150 + ceil(max(0, L - 20 mm) / 0.5 mm).

Radial clearance per side (comparable with ManiSkill's PortSpec.clearance_m = 0.5 mm, the gap on each side of a
centred pin): side points p of the plug (surface samples with outward normal n, |n_z| < 0.3, height between
min_z(plug) + 0.5 mm and max_z(socket) - 0.5 mm) are ray-cast against the socket along horizontal unit vectors u
(16 directions) when n.u > 0.5; play(u) = min over those points of the free distance along u. Then
    c(u)   = (play(u) + play(-u)) / 2      clearance per side along axis u (independent of where the plug sits)
    c_min  = min_u c(u),  off = max_u |play(u) - play(-u)| / 2   (how far the seated plug is from centred)
    d_min  = min over side points of the closest distance to the socket (one-sided worst gap at the seated pose)
A ray that never hits (an open side of a slot) gives play = inf.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# Isaac Sim 5.1 cloud asset root (carb setting /persistent/isaac/asset_root/cloud) + IsaacLab/AutoMate, as
# isaaclab.utils.assets.ISAACLAB_NUCLEUS_DIR resolves it in our environment.
ASSET_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/AutoMate"
ASSET_DIR = Path(os.environ.get("WM_AUTOMATE_ASSETS", "/mnt/storage3/qingxin/automate/assets"))
PALM_TO_FINGER_DIST = 0.1134       # assembly_tasks_cfg.AssemblyTask.palm_to_finger_dist (AutoMate)
ROBOT_TO_GRIPPER_QUAT = (0.0, 1.0, 0.0, 0.0)   # wxyz, assembly_env._init_tensors (pi about x)


def fetch(name: str) -> Path:
    """Local copy of <ASSET_URL>/<name> (downloaded once into ASSET_DIR)."""
    dst = ASSET_DIR / name
    if not dst.exists() or dst.stat().st_size == 0:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(f"{ASSET_URL}/{name}", timeout=120) as r, open(tmp, "wb") as f:
            f.write(r.read())
        tmp.rename(dst)
    return dst


def plug_grasps() -> dict:
    return json.loads(fetch("plug_grasps.json").read_text())


def disassembly_dists() -> dict:
    return json.loads(fetch("disassembly_dist.json").read_text())


def assembly_ids() -> list[str]:
    """The assemblies that have an optimised grasp (plug_grasps.json), sorted."""
    return sorted(k[len("asset_"):] for k in plug_grasps())


def _T(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def _R_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def grasp_fingertip_in_plug(assembly_id: str) -> np.ndarray:
    """AutoMate's optimised grasp as the pose of panda_fingertip_centered in the plug frame (4x4).

    Exactly assembly_env._move_gripper_to_grasp_pose: the json entry [px, py, pz, a, b, c, d] is read as position and
    torch.roll([a, b, c, d], -1) = [b, c, d, a] used as a wxyz quaternion (plug_grasp_quat_local), then
        T_plug_fc = T(q_grasp, p_grasp) * T(q_robot_to_gripper = Rx(pi), (0, 0, -palm_to_finger_dist)).
    """
    g = np.asarray(plug_grasps()[f"asset_{assembly_id}"], dtype=np.float64)
    q = np.roll(g[3:], -1)
    T_grasp = _T(_R_wxyz(q / np.linalg.norm(q)), g[:3])
    return T_grasp @ _T(_R_wxyz(ROBOT_TO_GRIPPER_QUAT), [0.0, 0.0, -PALM_TO_FINGER_DIST])


@dataclass(frozen=True)
class AssemblySpec:
    """What our code reads from wm.sim's PortSpec (pin_length_m, clearance_m), for one AutoMate assembly (metres)."""
    assembly_id: str
    pin_length_m: float            # L, tight-fit depth
    clearance_m: float             # c, radial clearance per side
    disassembly_dist_m: float      # AutoMate's d_dis
    entrance_z_m: float            # z_ent = min_z(plug) + L, assembled frame
    depth_top_m: float             # L_top
    socket_top_z_m: float          # max_z(socket)
    plug_bottom_z_m: float         # min_z(plug)
    plug_top_z_m: float
    seated_offcentre_m: float      # off
    min_gap_m: float               # d_min (rays) ; fit gap g(0) is in `fit`
    fingertip_z_m: float           # z of AutoMate's grasp fingertip centre in the plug frame
    plug_xy_m: tuple               # plug extent across the axis (side points of the seated band)
    symmetry_n: object             # int or "round" (fit.py)
    category: str                  # fit.py
    fit: dict                      # full fit.py output
    axis_xy_m: tuple               # (x_c, y_c) of the plug axis in the plug frame: centroid of the tight-fit band (fit.py)
    tip_centre_m: tuple            # plug tip centre in the plug frame (tip_centre): where depth_left / lateral are measured

    @property
    def max_decisions(self) -> int:
        """N_max = 150 + ceil(max(0, L - 20 mm) / 0.5 mm)."""
        return 150 + int(np.ceil(max(0.0, self.pin_length_m - 0.020) / 0.0005 - 1e-9))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["max_decisions"] = self.max_decisions
        return d


def _side_points(plug, z_lo: float, z_hi: float, n: int, seed: int = 0):
    import trimesh
    pts, fid = trimesh.sample.sample_surface_even(plug, n, seed=seed)
    nrm = plug.face_normals[fid]
    keep = (np.abs(nrm[:, 2]) < 0.3) & (pts[:, 2] > z_lo) & (pts[:, 2] < z_hi)
    return pts[keep], nrm[keep]


def _ray_geometry(assembly_id: str, plug, sock, n_samples: int = 20000, n_dirs: int = 16) -> dict:
    import trimesh
    top, bottom = float(sock.bounds[1, 2]), float(plug.bounds[0, 2])
    p, n = _side_points(plug, bottom + 5e-4, top - 5e-4, n_samples)
    inf = float("inf")
    play = np.full(n_dirs, inf)
    for k in range(n_dirs):
        a = 2 * np.pi * k / n_dirs
        u = np.array([np.cos(a), np.sin(a), 0.0])
        sel = n @ u > 0.5
        if not sel.any():
            continue
        o = p[sel] + 1e-6 * u
        loc, idx, _ = sock.ray.intersects_location(o, np.tile(u, (len(o), 1)), multiple_hits=False)
        if len(idx):
            play[k] = float(np.min(np.linalg.norm(loc - o[idx], axis=1)))
    h = n_dirs // 2
    c = (play[:h] + play[h:]) / 2.0
    off = np.abs(play[:h] - play[h:]) / 2.0
    off = float(np.max(off[np.isfinite(off)])) if np.isfinite(off).any() else inf
    if len(p):
        _, d, _ = trimesh.proximity.closest_point(sock, p)
        inside = sock.contains(p) if sock.is_watertight else np.zeros(len(p), bool)
        d_min = float(np.min(np.where(inside, -d, d)))
        xy = [float(v) for v in (p[:, :2].max(0) - p[:, :2].min(0))]
    else:
        d_min, xy = inf, [0.0, 0.0]
    return {"clearance_m": float(np.min(c)), "seated_offcentre_m": off, "min_gap_m": d_min, "plug_xy_m": xy}


SPEC_FILE = "geometry_v7.json"     # v7 (2026-11-23): v6 with symmetry_n "round" for SPIN_ROUND_2026_11_23 (below),
                                   # every other assembly byte-identical to v6 (scripts/automate/diag/spin_round_v7.py);
                                   # v6 (2026-11-22, round 2): plug axis = centroid of the band with interior rings
                                   # filled, bores in the round test (fit.py); v5: symmetry about the area centroid,
                                   # axis_xy_m, tip_centre_m. Older files are never overwritten.
# Spin symmetry override, the user's decision of 2026-11-23 (r4-decisions-06, option 2 "physical n: can the plug be
# inserted after rotation", applied ONLY to this named list): 15 assemblies whose plug is a 12-gon and 00863 (an 8-gon
# sleeve, included by name); for each the physical check plug-against-socket (fit.physical_symmetry, all assemblies,
# rs2 audit_am_r4/fix_env/out/bore_physical_n_all.json) says "round" with zero overlap at every angle. For them
#     symmetry_n = "round"   (spin box s = 15 deg, grasp.spin_limit_deg; spin information columns, checks.tilt_spin)
# while the shape category (split groups / quotas) stays the plug's own class (fit.py: polygonal, 00863 sleeve) and
# fit["n"] keeps the plug rule's n. Every other assembly keeps fit.py's n; the physical check is not applied to them
# (01036's 7 and the multi-pin / keyed results of that check are not used).
SPIN_ROUND_2026_11_23 = ("00320", "00863", "00030", "00103", "00110", "00117", "00143", "00187", "00514", "00537",
                         "00559", "00686", "00700", "00783", "00831", "01041")
SPIN_ROUND_RULE = "round by the user's decision 2026-11-23 (r4-decisions-06 option 2, named list)"


def tip_centre(plug) -> np.ndarray:
    """Centre of the plug's lowest 1 mm (plug frame): x, y = mean of the convex-hull vertices within 1 mm of the
    lowest point, z = the lowest point. The pivot of the expert (expert.PlugGeometry.c, the same rule) and the point
    at which the truth's depth_left and lateral are measured (checks.py)."""
    V = np.asarray(plug.convex_hull.vertices, np.float64)
    low = float(V[:, 2].min())
    ring = V[V[:, 2] < low + 0.001]
    return np.array([ring[:, 0].mean(), ring[:, 1].mean(), low])


def mesh_geometry(assembly_id: str) -> AssemblySpec:
    import trimesh
    from . import fit
    plug = trimesh.load(fetch(f"{assembly_id}/plug.obj"), force="mesh")
    sock = trimesh.load(fetch(f"{assembly_id}/socket.obj"), force="mesh")
    old = ASSET_DIR / assembly_id / "geometry_v3.json"          # same ray computation, computed earlier
    ray = json.loads(old.read_text()) if old.exists() else _ray_geometry(assembly_id, plug, sock)
    ray = {k: ray[k] for k in ("clearance_m", "seated_offcentre_m", "min_gap_m", "plug_xy_m")}
    f = fit.analyse(plug, sock, ray["clearance_m"])
    n_sym = f["n"]
    if assembly_id in SPIN_ROUND_2026_11_23:                     # the user's override of 2026-11-23 (above)
        n_sym = "round"
        f = {**f, "n_plug_rule": f["n"], "symmetry_rule": SPIN_ROUND_RULE}
    bottom = float(plug.bounds[0, 2])
    fc = grasp_fingertip_in_plug(assembly_id)
    return AssemblySpec(assembly_id=assembly_id, pin_length_m=f["L_tight_m"], clearance_m=ray["clearance_m"],
                        disassembly_dist_m=float(disassembly_dists()[f"asset_{assembly_id}"]),
                        entrance_z_m=f["entrance_z_m"], depth_top_m=float(sock.bounds[1, 2]) - bottom,
                        socket_top_z_m=float(sock.bounds[1, 2]), plug_bottom_z_m=bottom,
                        plug_top_z_m=float(plug.bounds[1, 2]), seated_offcentre_m=ray["seated_offcentre_m"],
                        min_gap_m=ray["min_gap_m"], fingertip_z_m=float(fc[2, 3]), plug_xy_m=tuple(ray["plug_xy_m"]),
                        symmetry_n=n_sym, category=f.get("category", "unknown"), fit=f,
                        axis_xy_m=tuple(float(v) for v in f.get("axis_xy_m", (0.0, 0.0))),
                        tip_centre_m=tuple(float(v) for v in tip_centre(plug)))


def load_spec(assembly_id: str, compute: bool = False) -> AssemblySpec:
    """mesh_geometry, cached as ASSET_DIR/<id>/SPEC_FILE. Read-only unless compute=True (analyse_assemblies.py):
    a missing file raises instead of being computed by a simulator or collection process."""
    f = ASSET_DIR / assembly_id / SPEC_FILE
    if f.exists():
        d = json.loads(f.read_text())
        d.pop("max_decisions", None)
        for k in ("plug_xy_m", "axis_xy_m", "tip_centre_m"):
            d[k] = tuple(d[k])
        if assembly_id in SPIN_ROUND_2026_11_23 and d.get("symmetry_n") != "round":
            raise ValueError(f"{f}: symmetry_n {d.get('symmetry_n')!r}, the user's decision 2026-11-23 says 'round' "
                             "(SPIN_ROUND_2026_11_23)")
        return AssemblySpec(**d)
    if not compute:
        raise FileNotFoundError(f"{f} is missing: written by scripts/automate/analyse_assemblies.py")
    spec = mesh_geometry(assembly_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(spec.as_dict(), default=float))
    return spec
