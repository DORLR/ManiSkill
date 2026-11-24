"""Truth-side helpers for checks and batch tests (simulator truth; never model inputs).

SUCCESS (approved by the user 2026-11-22: AutoMate's own rule, unchanged). Isaac Lab automate_algo_utils
check_plug_inserted_in_socket as called in assembly_env._get_rewards, with the Insertion task's values
(assembly_tasks_cfg: num_keypoints 4, keypoint_scale 0.15, close_error_thresh 0.015 m, fixed_success_pos_local 0):
    p, P   plug root position / rotation (AutoMate's held_pos, held_quat: the held asset's root),
    s, S   socket root position / rotation (fixed_pos, fixed_quat), d = disassembly_dist of the assembly
           (AutoMate's disassembly_dist.json, assets.AssemblySpec.disassembly_dist_m)
    height  h = p_z - s_z                            (AutoMate: world z; the socket is upright and our socket noise
                                                      is yaw only, so h = the plug's offset along the socket's z axis,
                                                      which is what is computed here, for every mount)
    keypoints  o_j = (0, 0, 0.15 (j / 3 - 1/2)), j = 0..3, in the plug frame and in the socket frame:
            k_dist = mean_j | (p + P o_j) - (s + S o_j) |
    success_now  =  (0 < h < d)  AND  (k_dist < 0.015 m)
    success of a trajectory = success_now after ANY control step (AutoMate: ep_succeeded |= curr_successes, computed in
           _get_rewards after every step, never on the state the reset produced). With eval_rel (T + 1, 7) of a
           trajectory (row 0 = the start state before the first control step, row k = after control step k):
               S_k = success_now(eval_rel[k]), k = 1..T;  success_any = OR_k S_k;  success_end = S_T (False if T = 0)
               first_success_step = min { k >= 1 : S_k } (None if none)
           eval_rel[0] is never counted (success_steps); whether the start state already satisfies the rule is
           recorded apart as start_success (information; start_box_success for a start of the start box).
In the goal frame used below (rel = plug in socket * R_G, charger axes): the plug in the socket frame is
M = R_G rel R_G^-1 = (R_s, t_s), so h = (t_s)_z and k_dist = mean_j |t_s + (R_s - I) o_j| (frame invariant).
"seated" is kept as a key with the same value as "success" (callers written before 2026-11-22 read it).

Recorded for information only (NOT the success criterion; ManiSkill-style numbers):
    rel        plug in the goal frame (charger axes: x = insertion direction a), as wm.data.recorder._truth_rel
               (Robot.truth_rel); rel = (t, q), q = (w, x, y, z), R its rotation; identity = assembled
    c          plug tip centre (assets.AssemblySpec.tip_centre_m) in charger axes: c = R_G^T c_plug. For multi-pin plugs
               it is the centre of the lowest 1 mm of the whole plug, which can lie off the plug axis (00444 26.6 mm,
               00308 10.7 mm, 00470 7.2 mm): a spin about the axis then also moves c sideways, by 2 r sin(spin / 2)
               (r = |c_xy - axis_xy|; 4 deg at r = 26.6 mm: 1.86 mm); scripts/automate/diag/axis_offsets.py lists r.
    tip        d = R c + t - c;  depth_left = -d_x [mm], lateral = |(d_y, d_z)| [mm]
    tilt_deg   = arccos(R_xx): angle between the plug axis and the insertion axis
    spin_deg   = |wrap_P(psi)|, psi = 2 atan2(q_x, q_w) the rotation about the axis, wrap_P(x) = ((x + P/2) mod P) - P/2,
                 P = 360 / n (assets.AssemblySpec.symmetry_n; n = 1: P = 360); 0 for a round plug, and for the
                 16 assemblies of assets.SPIN_ROUND_2026_11_23 (symmetry "round" by the user's decision 2026-11-23)
    angle_deg  = 2 arccos|w|: the full rotation angle (ManiSkill's angle, wm.data.outcome)
    tilt_symmetric_deg = min over k of angle(R_x(k P)^T R): the rotation to the nearest symmetric assembled pose
    penetration: plug and socket meshes placed at the simulator poses of env r.env_id; plug surface samples (+ vertices)
                 tested against the socket and socket surface samples (+ vertices) against the plug. A point is
                 inside a watertight mesh when a ray test says so (trimesh contains, ray parity); for a mesh that is
                 not watertight (22 of the 200 AutoMate meshes, e.g. 00014, 00117, 00444) ray parity is unreliable and
                 the generalised winding number is used instead: w(p) = sum over triangles of the signed solid angle
                 / (4 pi), inside when w > 0.5 (Jacobson et al. 2013; exact 1 inside / 0 outside for a closed mesh,
                 robust to holes and cracks). Depth of an inside point = its unsigned distance to the other mesh's
                 surface (closest point). max_penetration = largest depth of an inside point, 0 if none. Not
                 trimesh.proximity.signed_distance: that takes the sign from the closest triangle's normal when the
                 point projects into the triangle, and called points outside the socket inside (2026-11-22,
                 scripts/automate/diag/sd_sign_check.py). Evaluated in chunks (bounded memory); the result records
                 whether each mesh is watertight and which inside test was used. The winding number assumes
                 consistently oriented triangles (a flipped triangle subtracts its solid angle); a mesh whose winding
                 is not consistent (trimesh is_winding_consistent False, e.g. 00192's socket) is used through a cached
                 copy repaired by trimesh.repair.fix_winding (orientation made consistent; inside = |w| > 0.5, so a
                 globally inverted mesh counts the same), recorded as winding_fixed.
    contact force (the expert's guards; scene.AutomateScene.socket_contacts): Chongyu's socket_contact_force_n
                 (wminsertion_v2/scripted.py): over every contact pair that involves the receptacle, the norm of the
                 sum of that pair's contact-point impulses times the simulation frequency, summed over the pairs:
                     F = sum over shape pairs p of | sum over points i of p of J_i | / dt_physics
                 A "pair" is a pair of collision shapes, as SAPIEN's scene.get_contacts() returns them; scene.py reads
                 the same per-shape-pair, per-point impulses from PhysX's contact report (pair_force_sum below).
                 penetration guard (Chongyu's socket_penetration_mm): max over those points of max(0, -separation).
"""

from __future__ import annotations

import numpy as np

from ..sim.geometry import interpolate_T, pose_distance
from . import assets as A
from . import config as C

# AutoMate's Insertion task (assembly_tasks_cfg.Insertion / AssemblyTask), unchanged
NUM_KEYPOINTS, KEYPOINT_SCALE, CLOSE_ERROR_THRESH_M = 4, 0.15, 0.015
KEYPOINT_OFFSETS = np.zeros((NUM_KEYPOINTS, 3))
KEYPOINT_OFFSETS[:, 2] = (np.linspace(0.0, 1.0, NUM_KEYPOINTS) - 0.5) * KEYPOINT_SCALE   # _get_keypoint_offsets * scale
# per-step truth columns (eval_tip of the collector): AutoMate's quantities first, then the information numbers
TRUTH_COLUMNS = ("success", "height_mm", "keypoint_dist_mm", "depth_left_mm", "lateral_mm", "tilt_deg", "spin_deg",
                 "angle_deg", "tilt_symmetric_deg")


def period_deg(n) -> float | None:
    """Symmetry period P = 360 / n [deg]; None for a round plug; n = None (unknown) counts as n = 1."""
    if n == "round":
        return None
    return 360.0 / (1 if n is None else int(n))


def tilt_spin(q, n) -> dict:
    """q (..., 4) = (w, x, y, z): rotation of the plug relative to assembled, x = the insertion axis -> tilt_deg
    (axis), spin_deg, angle_deg (full), tilt_symmetric_deg [deg] (module docstring; information only)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = (np.asarray(v) for v in np.moveaxis(q, -1, 0))
    full = np.degrees(2.0 * np.arccos(np.clip(np.abs(w), 0.0, 1.0)))
    Rxx = 1.0 - 2.0 * (y * y + z * z)
    axis_tilt = np.degrees(np.arccos(np.clip(Rxx, -1.0, 1.0)))
    psi = np.degrees(2.0 * np.arctan2(x, w))
    P = period_deg(n)
    if P is None:
        spin, sym = np.zeros_like(axis_tilt), axis_tilt
    else:
        spin = np.abs((psi + P / 2.0) % P - P / 2.0)
        k = np.arange(int(round(360.0 / P)))
        a = np.radians(k * P / 2.0)                                       # half angles of R_x(k P)
        wk = np.abs(np.cos(a) * w[..., None] + np.sin(a) * x[..., None])  # |w| of R_x(k P)^T R
        sym = np.degrees(2.0 * np.arccos(np.clip(wk.max(-1), 0.0, 1.0)))
    return {"tilt_deg": axis_tilt, "spin_deg": spin, "angle_deg": full, "tilt_symmetric_deg": sym}


def automate_success(t_s, R_s, disassembly_dist_m: float) -> dict:
    """AutoMate's check_plug_inserted_in_socket (module docstring) for the plug pose in the socket frame:
    t_s (..., 3), R_s (..., 3, 3) -> success (bool), height_mm, keypoint_dist_mm."""
    t_s, R_s = np.asarray(t_s, np.float64), np.asarray(R_s, np.float64)
    h = t_s[..., 2]
    kp = t_s[..., None, :] + np.einsum("...ij,kj->...ki", R_s - np.eye(3), KEYPOINT_OFFSETS)   # (..., 4, 3)
    kd = np.linalg.norm(kp, axis=-1).mean(-1)
    ok = (h > 0.0) & (h < float(disassembly_dist_m)) & (kd < CLOSE_ERROR_THRESH_M)
    return {"success": ok, "height_mm": h * 1000.0, "keypoint_dist_mm": kd * 1000.0}


def truth_from_rel(rel7, tip_plug, n, disassembly_dist_m: float) -> dict:
    """rel7 (..., 7) plug in the goal frame (eval_rel) -> success, height_mm, keypoint_dist_mm (AutoMate's rule, per
    step; a trajectory succeeded if any step has success) and the information numbers depth_left_mm, lateral_mm (at
    the tip centre tip_plug, assets.AssemblySpec.tip_centre_m), tilt_deg, spin_deg, angle_deg, tilt_symmetric_deg;
    seated = success. The one AutoMate truth function: the collector, check scripts and AutoMate policy evaluations
    all use it (truth_from_rel_spec passes the AssemblySpec values)."""
    from scipy.spatial.transform import Rotation
    rel7 = np.asarray(rel7, np.float64)
    G3 = C.R_G[:3, :3]
    c = G3.T @ np.asarray(tip_plug, np.float64)
    q = rel7[..., 3:7]
    R = Rotation.from_quat(np.concatenate([q[..., 1:], q[..., :1]], -1).reshape(-1, 4)).as_matrix()
    R = R.reshape(rel7.shape[:-1] + (3, 3))
    d = np.einsum("...ij,j->...i", R, c) + rel7[..., :3] - c
    t_s = np.einsum("ij,...j->...i", G3, rel7[..., :3])               # plug in the socket frame: M = R_G rel R_G^-1
    R_s = np.einsum("ij,...jk,lk->...il", G3, R, G3)
    out = automate_success(t_s, R_s, disassembly_dist_m)
    out.update(depth_left_mm=-d[..., 0] * 1000.0, lateral_mm=np.hypot(d[..., 1], d[..., 2]) * 1000.0, **tilt_spin(q, n))
    out["seated"] = out["success"]
    return out


def truth_from_rel_spec(rel7, spec) -> dict:
    """truth_from_rel with the assembly's tip centre, symmetry and disassembly distance (assets.AssemblySpec)."""
    return truth_from_rel(rel7, spec.tip_centre_m, spec.symmetry_n, spec.disassembly_dist_m)


def success_steps(rel_seq, spec) -> dict:
    """AutoMate's episode success over a recorded trajectory (module docstring): rel_seq = eval_rel (T + 1, 7), row 0
    the start state (never counted), row k after control step k ->
        per_step            (T,) bool, S_k for k = 1..T
        success_any         OR_k S_k (AutoMate's ep_succeeded)
        success_end         S_T (False when T = 0: no control step, nothing evaluated)
        first_success_step  min k with S_k (control steps from the start, = the row index in eval_rel), None if none
        start_success       success_now(rel_seq[0]) (information: the start state already satisfies the rule)"""
    ok = np.asarray(truth_from_rel_spec(np.asarray(rel_seq, np.float64).reshape(-1, 7), spec)["success"], bool).reshape(-1)
    per = ok[1:]
    return {"per_step": per, "success_any": bool(per.any()), "success_end": bool(per[-1]) if len(per) else False,
            "first_success_step": int(np.argmax(per)) + 1 if per.any() else None,
            "start_success": bool(ok[0]) if len(ok) else False}


def success_any(rel_seq, spec) -> bool:
    """AutoMate's episode success: success_now after any control step of rel_seq = eval_rel (T + 1, 7); the start
    state rel_seq[0] is not counted (success_steps)."""
    return success_steps(rel_seq, spec)["success_any"]


def g_top_of(spec) -> float:
    """g_top = max(L_top, L) - L >= 0 (expert.py): how far the higher of {socket top, entrance} is above the entrance."""
    L = float(spec.pin_length_m)
    return max(float(spec.depth_top_m), L) - L


def start_box_success(spec, T_tcp_plug: np.ndarray, off, base_x_socket=(1.0, 0.0, 0.0),
                      gap_from: str | None = None) -> dict:
    """Does a start of the start box already satisfy AutoMate's rule (before any motion)? Simulation-free, in the
    socket frame (socket = identity, seated plug = socket, insertion direction a = -z):
        H*      = T_tcp_plug^-1                                   (TCP with the plug seated, grasp.py)
        F       = frames.frame(a, base_x_socket)                  (lateral axes; base_x_socket = the robot base x
                                                                   axis in the socket frame, R_socket^T e_x)
        T_start = frames.start_pose(H*, F, L, off, g_top, gap_from) (collect.py / Expert.start_pose; gap_from None =
                                                                   config.START_GAP_FROM; g_top = g_top_of(spec),
                                                                   read only for "socket_top")
        M       = T_start T_tcp_plug                              (plug in the socket frame at the start)
    -> automate_success of M (success, height_mm, keypoint_dist_mm) and the gap_from used. The H* used by the
    collector is the FK H* of a reference insertion, which differs from the model H* by 0.04-0.12 mm (smoke runs); not
    modelled here."""
    from . import frames
    if gap_from is None:
        gap_from = C.START_GAP_FROM
    g_top = g_top_of(spec) if gap_from == "socket_top" else None
    T_tcp_plug = np.asarray(T_tcp_plug, np.float64)
    H = np.linalg.inv(T_tcp_plug)
    F = frames.frame(np.array([0.0, 0.0, -1.0]), np.asarray(base_x_socket, np.float64))
    M = frames.start_pose(H, F, float(spec.pin_length_m), off, g_top, gap_from) @ T_tcp_plug
    out = automate_success(M[:3, 3], M[:3, :3], float(spec.disassembly_dist_m))
    return {"success": bool(out["success"]), "height_mm": float(out["height_mm"]),
            "keypoint_dist_mm": float(out["keypoint_dist_mm"]), "gap_from": gap_from}


class ContactReportError(RuntimeError):
    """PhysX's contact report has no plug-socket pair while the plug contact sensor reports force, or no robot-socket
    pair while the hand / finger contact sensor does (scene.py guard_contacts, check_contact_report): contact
    processing is not delivering the data the expert's guards need."""


REPORT_SENSOR_MIN_N = 1.0        # a sensor force above this needs report points of the same kind


def check_contact_report(c: dict, sensor_plug_N: float, sensor_finger_sum_N: float) -> dict:
    """The consistency rule of scene.AutomateScene.guard_contacts, on the socket_contacts dict c:
        plug:  c["plug_points"]  = 0  and  F_plug   > 1 N  ->  ContactReportError
        robot: c["robot_points"] = 0  and  F_finger > 1 N  ->  ContactReportError   (r8-expert-01: the 80 N guard on
               total_N = plug_N + robot_N and the 0.5 mm penetration guard need the robot-socket pairs too)
    F_plug = the plug contact sensor's net plug-socket force, F_finger = the hand / finger contact sensor's sum over
    hand, left and right finger of the net force with the socket [N]. Each kind is checked on its own, so points of
    one kind never hide a missing pair of the other. -> c with sensor_plug_net_N and sensor_finger_sum_N added."""
    if c["plug_points"] == 0 and sensor_plug_N > REPORT_SENSOR_MIN_N:
        raise ContactReportError(f"contact report has no plug-socket point while the plug contact sensor reports "
                                 f"{sensor_plug_N:.1f} N ({c['n_points']} other socket points)")
    if c["robot_points"] == 0 and sensor_finger_sum_N > REPORT_SENSOR_MIN_N:
        raise ContactReportError(f"contact report has no robot-socket point while the hand / finger contact sensor "
                                 f"reports {sensor_finger_sum_N:.1f} N ({c['n_points']} other socket points)")
    c["sensor_plug_net_N"] = float(sensor_plug_N)
    c["sensor_finger_sum_N"] = float(sensor_finger_sum_N)
    return c


class ResetError(RuntimeError):
    """A reset could not produce its state (no IK solution for the spawn, socket pose not taken by PhysX)."""


def truth(r) -> dict:
    """The truth of robot r now (truth_from_rel_spec of Robot.truth_rel); also the values at the plug frame origin
    (origin_*), as recorded before 2026-11-22."""
    rel = r.truth_rel().astype(np.float64)
    t = {k: (bool(v) if k in ("success", "seated") else float(v)) for k, v in truth_from_rel_spec(rel, r.port_spec).items()}
    t.update(origin_depth_left_mm=float(-rel[0] * 1000), origin_lateral_mm=float(np.hypot(rel[1], rel[2]) * 1000),
             success_rule="automate check_plug_inserted_in_socket")
    return t


def pair_force_sum(pairs, dt: float) -> float:
    """Chongyu's socket_contact_force_n: pairs = iterable of (k_p, 3) arrays of contact-point impulses [N s], one array
    per contact pair (pair of collision shapes) -> sum over pairs of |sum of the pair's impulses| / dt [N]. Points
    of one pair add as vectors (they can cancel within a pair, as in SAPIEN); separate pairs add as magnitudes."""
    return float(sum(np.linalg.norm(np.asarray(J, np.float64).reshape(-1, 3).sum(0)) for J in pairs) / dt)


def point_force_sum(pairs, dt: float) -> float:
    """Sum over every contact point of |impulse| / dt [N] (an upper bound of pair_force_sum; information)."""
    return float(sum(np.linalg.norm(np.asarray(J, np.float64).reshape(-1, 3), axis=1).sum() for J in pairs) / dt)


def force_magnitude_sum(force_matrix) -> float:
    """Contact sensor force matrix (..., 3) [N], one row per (body, filter body) -> sum of the rows' magnitudes. The
    Isaac Lab contact sensor reports ONE net force per (sensor body, filter body), the vector sum over every shape
    pair and contact point of that body pair: opposite walls of the receptacle cancel inside a row. Used only as
    information next to pair_force_sum (per shape pair, the guard's rule)."""
    f = np.asarray(force_matrix, np.float64).reshape(-1, 3)
    return float(np.linalg.norm(f, axis=1).sum())


def winding_number(mesh, pts: np.ndarray, budget: int = 2_000_000) -> np.ndarray:
    """Generalised winding number of `mesh` at each point (Van Oosterom-Strackee solid angles / 4 pi), evaluated in
    chunks of about `budget` point-triangle pairs."""
    pts = np.asarray(pts, np.float64)
    tri = np.asarray(mesh.triangles, np.float64)                  # (F, 3, 3)
    out = np.zeros(len(pts))
    step = max(1, budget // max(1, len(tri)))
    for i in range(0, len(pts), step):
        d = tri[None] - pts[i:i + step, None, None, :]           # (m, F, 3, 3)
        a, b, c = d[..., 0, :], d[..., 1, :], d[..., 2, :]
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        det = np.einsum("mfi,mfi->mf", a, np.cross(b, c))
        den = la * lb * lc + np.einsum("mfi,mfi->mf", a, b) * lc + np.einsum("mfi,mfi->mf", b, c) * la \
            + np.einsum("mfi,mfi->mf", c, a) * lb
        out[i:i + step] = 2.0 * np.arctan2(det, den).sum(-1) / (4.0 * np.pi)
    return out


_WINDING_FIXED: dict = {}


def winding_mesh(mesh):
    """(mesh, fixed): the mesh itself if its triangles are consistently oriented, else a cached copy repaired with
    trimesh.repair.fix_winding (module docstring)."""
    if mesh.is_winding_consistent:
        return mesh, False
    key = id(mesh)
    hit = _WINDING_FIXED.get(key)
    if hit is None or hit[0] is not mesh:
        import trimesh
        m = mesh.copy()
        trimesh.repair.fix_winding(m)
        hit = _WINDING_FIXED[key] = (mesh, m)
    return hit[1], True


def inside(mesh, pts: np.ndarray) -> np.ndarray:
    """Inside test of the module docstring: ray parity for a watertight mesh, |winding number| > 0.5 otherwise (on the
    winding-repaired copy when the orientation is inconsistent); points outside the bounding box are outside."""
    pts = np.asarray(pts, np.float64)
    lo, hi = mesh.bounds
    box = np.all((pts >= lo - 1e-9) & (pts <= hi + 1e-9), axis=1)
    res = np.zeros(len(pts), bool)
    if box.any():
        if mesh.is_watertight:
            res[box] = np.asarray(mesh.contains(pts[box]), bool)
        else:
            res[box] = np.abs(winding_number(winding_mesh(mesh)[0], pts[box])) > 0.5
    return res


def mesh_penetration(mesh, pts: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Depth [m] of each point inside `mesh` (0 outside): inside() (ray test, or winding number for a mesh that is not
    watertight), depth = unsigned distance to the closest point of the surface. In chunks of `chunk` points."""
    import trimesh
    pts = np.asarray(pts, np.float64)
    depth = np.zeros(len(pts))
    for i in range(0, len(pts), chunk):
        q = pts[i:i + chunk]
        ins = inside(mesh, q)
        if ins.any():
            _, dist, _ = trimesh.proximity.closest_point(mesh, q[ins])
            depth[i:i + chunk][ins] = dist
    return depth


def _samples(mesh, n: int) -> np.ndarray:
    import trimesh
    pts, _ = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    return np.vstack([pts, mesh.vertices])


def overlap(plug, sock, M: np.ndarray, n: int = 20000, chunk: int = 1000, samples=None) -> dict:
    """Penetration of plug and socket meshes with the plug placed at M (plug frame -> socket frame), both ways."""
    p_pts, s_pts = samples if samples is not None else (_samples(plug, n), _samples(sock, n))
    Mi = np.linalg.inv(M)
    d_plug = mesh_penetration(sock, p_pts @ M[:3, :3].T + M[:3, 3], chunk)        # plug points inside the socket
    d_sock = mesh_penetration(plug, s_pts @ Mi[:3, :3].T + Mi[:3, 3], chunk)      # socket points inside the plug
    d = np.concatenate([d_plug, d_sock])
    test = lambda m: "ray parity" if m.is_watertight else "winding number"
    fixed = lambda m: bool(not m.is_watertight and not m.is_winding_consistent)
    return {"max_penetration_mm": float(d.max(initial=0.0) * 1000), "n_points_inside_0.05mm": int((d > 5e-5).sum()),
            "n_points": int(len(d)), "max_plug_in_socket_mm": float(d_plug.max(initial=0.0) * 1000),
            "max_socket_in_plug_mm": float(d_sock.max(initial=0.0) * 1000),
            "plug_watertight": bool(plug.is_watertight), "socket_watertight": bool(sock.is_watertight),
            "inside_test": {"plug points in socket": test(sock), "socket points in plug": test(plug)},
            "winding_fixed": {"plug": fixed(plug), "socket": fixed(sock)}}


_MESH = {}


def penetration(r, n: int = 20000) -> dict:
    import trimesh
    key = r.assembly_id
    if key not in _MESH:
        plug = trimesh.load(A.fetch(f"{key}/plug.obj"), force="mesh")
        sock = trimesh.load(A.fetch(f"{key}/socket.obj"), force="mesh")
        _MESH.clear()
        _MESH[key] = (plug, sock, (_samples(plug, n), _samples(sock, n)))
    plug, sock, samples = _MESH[key]
    M = np.linalg.inv(r.sc.socket_T(r.env_id)) @ r.sc.plug_T(r.env_id)
    return overlap(plug, sock, M, samples=samples)


def drive_line(r, T0: np.ndarray, T1: np.ndarray, speed: float, rec=None, lag_stop: float | None = None,
               max_steps: int = 10_000) -> dict:
    """Commanded TCP pose on a straight line T0 -> T1 at `speed` [m/s] (rotation <= 2 deg per step), one IK target
    per control step; stops ("blocked") when the command leads the FK hand by more than lag_stop [m]."""
    dt, dr = pose_distance(T0, T1)
    n = max(1, int(np.ceil(max(dt / (speed * C.CONTROL_DT), dr / np.radians(2.0)))))
    seed, steps, why, cmd = r.qpos(), 0, "done", T0
    for k in range(1, min(n, max_steps) + 1):
        cmd = interpolate_T(T0, T1, k / n)
        q = r._ik(cmd, seed)
        if q is None:
            why = "ik"
            break
        r._command(q, 1)
        seed, steps = q, steps + 1
        if rec is not None:
            rec()
        if lag_stop is not None and np.linalg.norm(cmd[:3, 3] - r.hand_pose()[:3, 3]) > lag_stop:
            why = "blocked"
            break
    return {"steps": steps, "end": why, "q": seed, "cmd": cmd}


def along(r, H: np.ndarray, dist: float) -> np.ndarray:
    """H moved by -dist along the insertion direction (dist > 0: in front of H)."""
    T = H.copy()
    T[:3, 3] = H[:3, 3] - dist * r.axis()
    return T
