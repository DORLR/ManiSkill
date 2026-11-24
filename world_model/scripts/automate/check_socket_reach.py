"""Reachability of the AutoMate task with the socket randomised as in AutoMate (config.socket_pose: +-5 cm per axis,
yaw 0-10 deg), by inverse kinematics only (Pinocchio, wm.sim_automate.kinematics; no simulator).

For every assembly with a grasp box (grasp.RANGES_FILE) and every combination of
    socket pose    SOCKET_POS + (+-5, +-5, +-5) cm corners x yaw {0, 10} deg, and the nominal pose (17)
    grasp          the 8 corners of the box and its centre (fewer when a range has zero width)
the TCP poses checked are
    H*             T_socket T_hp^-1 (seated)
    spawn          the reset pose (plug aligned on the axis, lowest point max(L_top, L) + 12 mm above seated)
    transfer       H* moved back along a by L + g_top + 22 mm (the height of the unrecorded transfers)
    starts         Expert.start_pose for g0 in {6, 18} mm, lateral (+-10, +-10) mm, rotations (+-6, +-6, +-6) deg (64);
                   g0 from the socket top (config.START_GAP_FROM "socket_top", the user's decision 2026-11-23;
                   "entrance" only under diagnostics; recorded in switches)
    start_high     each start's T_high (expert.move_to_start leg 2 end: the start's attitude and lateral offset at the
                   transfer height, raised along a until the plug is TRANSFER_GAP_M above socket top / entrance)
    waypoints      random-motion waypoints in the expert's pivot convention (Execution.offset_rel): top gap {4, 18} mm,
                   lateral (+-10, +-10) mm, rotations (+-6)^3 deg (64)
    contact        contact offsets: top gap 3 mm, lateral 3 mm in 4 directions, rotations (+-3)^3 deg (32), and the
                   aligned stand-off (top gap 3 mm)
A pose is reachable when the IK converges (1e-5) within the joint limits from the H* solution, or else from any of
config.RESET_SEEDS (H* and the spawn always from RESET_SEEDS, as Robot.reset_ik). Unreachable combinations are
listed, never dropped; the smallest joint-limit margin of the H* solutions is reported.
Paths (--paths, socket poses of --path-sockets): the transfer legs T_lift -> T_high (robot.move_to) -> start (the
guarded descent, expert._retreat: one IK target per control step on the straight line), T_lift = H* raised to the
transfer height, checked as straight-line waypoints every 2 mm / 1 deg, each solved by
robot._ik's rule from the previous solution (tolerances 1e-5 .. 1e-3 with restarts; a jump > 0.25 rad in any joint
is refused). A path fails at its first refused waypoint.
Measured 2026-11-22 (round 2 audit): all 1,387 failures of the v5 run were "start" poses at the far-high socket
corner (+5, +-5, +5) cm, mostly with dtheta = +20 deg; unbounded least squares left 0.15-8.9 mm there with the arm
fully stretched (j4 = -0.467), so they are the arm's workspace limit, not IK convergence failures. reach.json
carries this note.

    python -u check_socket_reach.py --out reach.json [--ids a,b] [--workers 8] [--paths] [--path-sockets far|all]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import types
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_stub = types.ModuleType("wm.sim_automate.scene")                  # robot.py imports the Isaac scene; not used here
_stub.AutomateScene = object
sys.modules.setdefault("wm.sim_automate.scene", _stub)
from wm.env.sampling import Offset  # noqa: E402
from wm.sim_automate import assets as A  # noqa: E402
from wm.sim_automate import config as C  # noqa: E402
from wm.sim_automate import frames  # noqa: E402
from wm.sim_automate import grasp as G  # noqa: E402
from wm.sim_automate.expert import NEAR_GAP_M, TRANSFER_GAP_M, Execution, Expert, PlugGeometry, euler_R  # noqa: E402
from wm.sim_automate.kinematics import Kinematics  # noqa: E402
from wm.sim_automate.robot import R_G, Robot  # noqa: E402

TOL = 1e-5
PATH_STEP_M, PATH_STEP_RAD = 0.002, np.radians(1.0)
NOTE = ("v5 (2026-11-22): every unreachable pose was a 'start' at the far-high socket corner (+5, +-5, +5) cm; the "
        "round-2 audit found 0.15-8.9 mm residual with joint limits ignored and the arm fully stretched (j4 = -0.467): "
        "the arm's workspace limit, not IK convergence. In collection such a start fails the transfer and is redrawn "
        "(placement_failed per assembly is reported from grasp.json).")
OPTS = {"paths": False, "path_sockets": "far"}


class _IK:
    """robot.Robot._ik on a bare Kinematics (the same rule: restarts, tolerances, 0.25 rad branch limit)."""
    def __init__(self, kin):
        self.kin = kin
    _ik = Robot._ik


def _path(ik: _IK, poses: list, q0: np.ndarray):
    """Straight-line waypoints through `poses` every 2 mm / 1 deg, solved from q0 -> (ok, q, failing index)."""
    from wm.sim.geometry import interpolate_T, pose_distance
    q = q0
    for k in range(len(poses) - 1):
        dt, dr = pose_distance(poses[k], poses[k + 1])
        n = max(1, int(np.ceil(max(dt / PATH_STEP_M, dr / PATH_STEP_RAD))))
        for j in range(1, n + 1):
            sol = ik._ik(interpolate_T(poses[k], poses[k + 1], j / n), q)
            if sol is None:
                return False, q, (k, j, n)
            q = sol
    return True, q, None


def socket_poses(mount: str) -> list[tuple[str, np.ndarray]]:
    from scipy.spatial.transform import Rotation
    q = C.SOCKET_QUAT[mount]
    Rm = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    out = []
    for d in [(0.0, 0.0, 0.0)] + list(itertools.product(*[(-v, v) for v in C.SOCKET_POS_NOISE_M])):
        for yaw in ((0.0,) if d == (0.0, 0.0, 0.0) else (C.SOCKET_YAW_DEG, C.SOCKET_YAW_DEG + C.SOCKET_YAW_RANGE_DEG)):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ Rm
            T[:3, 3] = np.asarray(C.SOCKET_POS[mount]) + np.asarray(d)
            out.append((f"d={[round(x * 100) for x in d]}cm yaw={yaw:g}", T))
    return out


def grasp_points(box: dict) -> list[tuple]:
    ks = ("dtheta_deg", "dp_m", "dpsi_deg")
    pts = {tuple(float(x) for x in c) for c in itertools.product(*[sorted(set(box[k])) for k in ks])}
    pts.add(tuple(0.5 * (box[k][0] + box[k][1]) for k in ks))
    return sorted(pts)


def run(aid: str, mount: str = C.SOCKET_MOUNT) -> dict:
    t0 = time.time()
    rg = G.load_ranges(aid)
    if rg.get("box") is None:
        return {"assembly_id": aid, "skipped": "no box"}
    spec = A.load_spec(aid)
    kin = Kinematics()
    T0 = G.T_hp0(aid)
    L, g_top = float(spec.pin_length_m), max(float(spec.depth_top_m), float(spec.pin_length_m)) - float(spec.pin_length_m)
    lift = max(spec.depth_top_m, spec.pin_length_m) + C.SPAWN_GAP_M
    starts = [Offset(g, (y, z), rpy) for g in (0.006, 0.018) for y in (-0.01, 0.01) for z in (-0.01, 0.01)
              for rpy in itertools.product((-6.0, 6.0), repeat=3)]
    waypoints = [Offset(g, (y, z), rpy) for g in (0.004, 0.018) for y in (-0.01, 0.01) for z in (-0.01, 0.01)
                 for rpy in itertools.product((-6.0, 6.0), repeat=3)]
    contacts = [Offset(NEAR_GAP_M, (0.0, 0.0), (0.0, 0.0, 0.0))] + [
        Offset(NEAR_GAP_M, yz, rpy) for yz in ((0.003, 0.0), (-0.003, 0.0), (0.0, 0.003), (0.0, -0.003))
        for rpy in itertools.product((-3.0, 3.0), repeat=3)]
    import types as _t
    geom = PlugGeometry(_t.SimpleNamespace(port_spec=spec, assembly_id=aid, _fingers=G.FingerSocketCheck(aid)))
    seeds = [np.r_[s, 0.04, 0.04] for s in C.RESET_SEEDS]
    ik = _IK(kin)
    fails, n, margins, path_fails, n_paths = [], 0, [], [], 0

    def solve(T, seed_list):
        for s in seed_list:
            q = kin.ik(T, s, tol=TOL)
            if q is not None:
                return q
        return None
    for (sname, Ts), v in itertools.product(socket_poses(mount), grasp_points(rg["box"])):
        Thp = G.T_hp(T0, *v, axis_xy=G.spin_axis_xy(spec))         # the switch config.GRASP_SPIN_AXIS
        H = Ts @ np.linalg.inv(Thp)
        a = -Ts[:3, 2]
        n += 1
        qH = solve(H, seeds)
        if qH is None:
            fails.append({"socket": sname, "grasp": v, "pose": "H*"})
            continue
        margins.append(float(np.min(np.minimum(qH[:7] - kin.lower[:7], kin.upper[:7] - qH[:7]))))
        spawn = Ts @ C_trans([0.0, 0.0, lift]) @ np.linalg.inv(Thp)
        poses = [("spawn", spawn, seeds)]
        Ttr = H.copy()
        Ttr[:3, 3] = H[:3, 3] - a * (L + g_top + TRANSFER_GAP_M)
        poses.append(("transfer", Ttr, [qH] + seeds))
        F = frames.frame(a, kin.base_T[:3, 0])
        T_tcp_plug = Thp
        Gs = Ts @ R_G
        hi = np.linalg.inv(T_tcp_plug @ R_G)                        # Execution.tcp: G rel h^-1, h = TCP^-1 CH

        def high(T):                                                # expert.move_to_start: T_high of a start T
            rel_s = np.linalg.inv(Gs) @ T @ T_tcp_plug @ R_G
            Th = T.copy()
            Th[:3, 3] -= max(0.0, TRANSFER_GAP_M - geom.top_gap(rel_s)) * a
            return Th
        T_lift = H.copy()
        T_lift[:3, 3] -= max(0.0, TRANSFER_GAP_M - geom.top_gap(np.eye(4))) * a
        start_poses = []
        for off in starts:
            tag = f"g0={off.tip_gap_m * 1e3:.0f} yz={[round(x * 1e3) for x in off.yz_m]} rpy={list(off.rpy_deg)}"
            Tst = Expert.start_pose(H, F, L, off, g_top)              # follows config.START_GAP_FROM
            start_poses.append((tag, Tst))
            poses.append((f"start {tag}", Tst, [qH] + seeds))
            poses.append((f"start_high {tag}", high(Tst), [qH] + seeds))
        for off in waypoints:
            Tw = Gs @ geom.outside(off.tip_gap_m, off.yz_m, euler_R(off.rpy_deg)) @ hi
            poses.append((f"waypoint g={off.tip_gap_m * 1e3:.0f} yz={[round(x * 1e3) for x in off.yz_m]} "
                          f"rpy={list(off.rpy_deg)}", Tw, [qH] + seeds))
        for off in contacts:
            Tc = Gs @ geom.outside(off.tip_gap_m, off.yz_m, euler_R(off.rpy_deg)) @ hi
            poses.append((f"contact yz={[round(x * 1e3) for x in off.yz_m]} rpy={list(off.rpy_deg)}", Tc, [qH] + seeds))
        for name, T, sl in poses:
            n += 1
            if solve(T, sl) is None:
                fails.append({"socket": sname, "grasp": v, "pose": name})
        if OPTS["paths"] and (OPTS["path_sockets"] == "all" or sname.startswith("d=[0, 0, 0]") or
                              (sname.startswith("d=[5,") and sname.split("]")[0].endswith(", 5"))):
            q_lift = solve(T_lift, [qH] + seeds)
            for tag, Tst in start_poses:
                n_paths += 1
                ok, _, where = (False, None, "T_lift unreachable") if q_lift is None else \
                    _path(ik, [T_lift, high(Tst), Tst], q_lift)
                if not ok:
                    path_fails.append({"socket": sname, "grasp": v, "start": tag, "at": where})
    by_kind = {}
    for f in fails:
        k = f["pose"].split(" ")[0]
        by_kind[k] = by_kind.get(k, 0) + 1
    return {"assembly_id": aid, "checked": n, "unreachable": len(fails), "unreachable_by_kind": by_kind,
            "fails": fails[:200], "paths_checked": n_paths, "path_failures": len(path_fails),
            "path_fails": path_fails[:100], "min_joint_margin_H_star_rad": min(margins) if margins else None,
            "seconds": round(time.time() - t0, 1)}


def C_trans(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = v
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--paths", action="store_true", help="also check the transfer legs as straight-line paths")
    ap.add_argument("--path-sockets", default="far", choices=("far", "all"),
                    help="far: nominal socket and the (+5, +-5, +5) cm corners; all: every socket pose")
    args = ap.parse_args()
    OPTS.update(paths=args.paths, path_sockets=args.path_sockets)
    ids = args.ids.split(",") if args.ids else [a for a in A.assembly_ids() if G.ranges_file(a).exists()]
    with Pool(args.workers, initializer=OPTS.update, initargs=(dict(OPTS),)) as pool:
        rows = []
        for r in pool.imap_unordered(run, ids):
            rows.append(r)
            print(r["assembly_id"], {k: r.get(k) for k in ("checked", "unreachable", "min_joint_margin_H_star_rad", "seconds", "skipped")},
                  flush=True)
    rows.sort(key=lambda r: r["assembly_id"])
    summary = {"assemblies": len(rows), "with_unreachable": [r["assembly_id"] for r in rows if r.get("unreachable")],
               "checked": sum(r.get("checked", 0) for r in rows), "unreachable": sum(r.get("unreachable", 0) for r in rows),
               "paths_checked": sum(r.get("paths_checked", 0) for r in rows),
               "path_failures": sum(r.get("path_failures", 0) for r in rows), "note": NOTE,
               "switches": C.switches(), "ranges_file": G.RANGES_FILE, "spec_file": A.SPEC_FILE}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=float))
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
