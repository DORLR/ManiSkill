"""Lateral axes around the insertion direction (approved rule, milestone 1b). Plain numpy.

With a the unit insertion direction, up the gravity-up unit vector and x_b the robot base x axis (world frame):
    r  = up   if |a . up| < 0.9,   else  x_b
    e1 = normalise(r - (r . a) a)
    e2 = a x e1
frame(a) returns the 3 x 3 matrix with columns [a, e1 x a, e1] = [a, -e2, e1]: the column layout of the existing
wm.collect.explore.frame() and wm.sim.geometry.port_axes() (columns a, y, z with z the "up-like" lateral axis), so
that for |a . up| < 0.9 (every ManiSkill grasp) frame() computes exactly what explore.frame() computes, with the same
floating-point operations (checked bit-identical by scripts/automate/check_frames.py). For a vertical a (AutoMate's
table mount) the existing functions divide 0 by 0 (or, with an axis estimated from FK, take their lateral axes from
round-off); frame() uses the robot base x axis instead.

start_pose(H*, F, L, off, g_top, gap_from): the TCP pose of a start of the start box (task definition, section 6;
used by the expert, the collector and checks.start_box_success): H* moved back by b along a, shifted by (y, z) along
F's columns 1, 2 and rotated about the TCP by the static XYZ angles of the start, expressed in F:
    p = p* + F (-b, y, z),   R = F R_xyz(rpy) F^T R*
    b = L + g_top + g0   gap_from "socket_top" (default; option B of r4-expert-01, decided by the user 2026-11-23:
                                                g0 measured from the higher of {socket top, entrance}, so every start
                                                is outside the socket, as in AutoMate's evaluation)
    b = L + g0           gap_from "entrance"   (diagnostics only, WM_AUTOMATE_DIAGNOSTIC=1: g0 measured from the
                                                start of the tight fit, the box until 2026-11-23; for g_top > 6 mm some
                                                starts are inside the socket's lead-in)
with g_top = max(L_top, L) - L >= 0 (expert.py). gap_from None reads config.START_GAP_FROM (switch
WM_AUTOMATE_START_GAP_FROM); "socket_top" needs g_top, a missing g_top raises (never silently 0); "entrance" raises
outside diagnostics.
"""

from __future__ import annotations

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
BASE_X = np.array([1.0, 0.0, 0.0])
SWITCH = 0.9


def frame(axis: np.ndarray, base_x: np.ndarray = BASE_X) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    if abs(a[2]) < SWITCH:                       # a . up with up = (0, 0, 1); identical to explore.frame()
        z = np.array([0.0, 0.0, 1.0]) - a[2] * a
    else:
        r = np.asarray(base_x, np.float64)
        z = r - (r @ a) * a
    z /= np.linalg.norm(z)
    return np.stack([a, np.cross(z, a), z], 1)


def lateral_axes(axis: np.ndarray, base_x: np.ndarray = BASE_X) -> tuple[np.ndarray, np.ndarray]:
    """(e1, e2) of the rule above."""
    F = frame(axis, base_x)
    return F[:, 2].copy(), -F[:, 1]


def euler_R(rpy_deg) -> np.ndarray:
    """Static XYZ roll / pitch / yaw [deg] (as Chongyu's euler2quat(*radians(rpy)), axes x, y, z)."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()


START_GAP_FROM = ("socket_top", "entrance")


def start_back(L: float, g0: float, g_top: float | None = None, gap_from: str | None = None) -> float:
    """b of the module docstring: how far the start's TCP is moved back from H* along a."""
    from . import config as C
    if gap_from is None:
        gap_from = C.START_GAP_FROM
    if gap_from == "entrance":
        if not C.DIAGNOSTIC:
            raise ValueError("start gap from the entrance: diagnostics only (WM_AUTOMATE_DIAGNOSTIC=1); the user's "
                             "decision 2026-11-23 measures g0 from the socket top")
        return L + g0
    if gap_from == "socket_top":
        if g_top is None:
            raise ValueError("start gap from the socket top needs g_top (max(L_top, L) - L of the assembly)")
        if g_top < 0.0:
            raise ValueError(f"g_top {g_top} < 0")
        return L + float(g_top) + g0
    raise ValueError(f"gap_from {gap_from!r} not in {START_GAP_FROM}")


def start_pose(H_star: np.ndarray, F: np.ndarray, L: float, off, g_top: float | None = None,
               gap_from: str | None = None) -> np.ndarray:
    """TCP pose of a start (module docstring); off = wm.env.sampling.Offset (tip_gap_m, yz_m, rpy_deg)."""
    H_star = np.asarray(H_star, np.float64)
    T = H_star.copy()
    b = start_back(L, off.tip_gap_m, g_top, gap_from)
    T[:3, 3] = H_star[:3, 3] + F @ np.array([-b, off.yz_m[0], off.yz_m[1]])
    T[:3, :3] = F @ euler_R(off.rpy_deg) @ F.T @ H_star[:3, :3]
    return T
