"""Static pin and slot geometry for a single plug pose, in SI units.

The caller supplies the native goal pose, board * Translate(0, dy, dz) *
Rz(pi). Goal-frame +X points into the board and each slot's Y/Z center
matches its corresponding pin in the plug specification.
"""
from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ports import PortSpec


ROOT_MIN_X_M = -.001
ROOT_MAX_X_M = .0002
TIP_DEPTH_ALLOWANCE_M = .001


def _components(values, count):
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected a single finite pose in XYZ and WXYZ order") from exc
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError("Expected a single finite pose in XYZ and WXYZ order")
    return result


def _pose_parts(value):
    value = getattr(value, "sp", value)
    if hasattr(value, "p") and hasattr(value, "q"):
        position, quaternion = _components(value.p, 3), _components(value.q, 4)
    else:
        raw = _components(value, 7)
        position, quaternion = raw[:3], raw[3:]
    magnitude = math.hypot(*quaternion)
    if not 0. < magnitude < math.inf:
        raise ValueError("Expected a nonzero finite WXYZ quaternion")
    return position, tuple(component / magnitude for component in quaternion)


def _multiply_quaternions(left, right):
    w, x, y, z = left
    a, b, c, d = right
    return (
        w*a - x*b - y*c - z*d,
        w*b + x*a + y*d - z*c,
        w*c - x*d + y*a + z*b,
        w*d + x*c - y*b + z*a,
    )


def _rotation(quaternion):
    w, x, y, z = quaternion
    return (
        (1. - 2.*(y*y + z*z), 2.*(x*y - w*z), 2.*(x*z + w*y)),
        (2.*(x*y + w*z), 1. - 2.*(x*x + z*z), 2.*(y*z - w*x)),
        (2.*(x*z - w*y), 2.*(y*z + w*x), 1. - 2.*(x*x + y*y)),
    )


def _rotate(rotation, vector):
    x, y, z = vector
    return tuple(a*x + b*y + c*z for a, b, c in rotation)


def _slab_vertices(corners, board_depth):
    """Vertices of a transformed pin box clipped to 0 <= X <= board_depth."""
    inside = [point for point in corners if 0. <= point[0] <= board_depth]
    # The corner order is binary (X, Y, Z). Flipping one bit picks a box edge.
    for index, start in enumerate(corners):
        for bit in (1, 2, 4):
            if index & bit:
                continue
            end = corners[index | bit]
            for face_x in (0., board_depth):
                if min(start[0], end[0]) < face_x < max(start[0], end[0]):
                    weight = (face_x - start[0]) / (end[0] - start[0])
                    inside.append((
                        face_x,
                        start[1] + weight * (end[1] - start[1]),
                        start[2] + weight * (end[2] - start[2]),
                    ))
    return inside


def _slot_margin(points, center_y, center_z, half_y, half_z):
    return min(
        min(
            y - (center_y - half_y),
            (center_y + half_y) - y,
            z - (center_z - half_z),
            (center_z + half_z) - z,
        )
        for _, y, z in points
    )


def geometry_from_poses(charger_pose, goal_pose, spec: PortSpec) -> dict:
    """Measure one static configuration using the caller's connector geometry.

    Accepts XYZ/WXYZ sequences or single poses exposing p and q. The native
    depth and pose fields keep their meaning. The full-pin slot-fit diagnostic
    checks an infinitely extended slot, while the board-slab check describes
    only the part inside the actual board thickness (including its faces).
    A pin outside that slab has no board-slab fit or margin measurement.
    """
    plug_p, plug_q = _pose_parts(charger_pose)
    goal_p, goal_q = _pose_parts(goal_pose)
    goal_inv_q = (goal_q[0], -goal_q[1], -goal_q[2], -goal_q[3])
    world_delta = tuple(a - b for a, b in zip(plug_p, goal_p))
    relative_p = _rotate(_rotation(goal_inv_q), world_delta)
    relative_q = _multiply_quaternions(goal_inv_q, plug_q)
    rotation = _rotation(relative_q)
    pin_length = spec.pin_length_m
    board_depth = 2. * spec.socket_half_size_m[0]
    tip_depth_threshold = pin_length - TIP_DEPTH_ALLOWANCE_M
    pins = []
    # Keep the native negative-to-positive Y diagnostic order for A. The
    # explicit index still identifies the matching pin/slot in the spec.
    for pin_index, pin in sorted(
        enumerate(spec.pins), key=lambda entry: entry[1].center_yz_m
    ):
        center_y, center_z = pin.center_yz_m
        half_y, half_z = pin.half_size_yz_m
        slot_half_y, slot_half_z = half_y + spec.clearance_m, half_z + spec.clearance_m
        corners = [
            tuple(
                a + b
                for a, b in zip(
                    _rotate(rotation, (x, center_y + sy*half_y, center_z + sz*half_z)),
                    relative_p,
                )
            )
            for x, sy, sz in product((0., pin_length), (-1., 1.), (-1., 1.))
        ]
        axes = tuple(zip(*corners))
        lower = [min(axis) for axis in axes]
        upper = [max(axis) for axis in axes]
        slot = (center_y, center_z, slot_half_y, slot_half_z)
        margin = _slot_margin(corners, *slot)
        slab = _slab_vertices(corners, board_depth)
        slab_margin = _slot_margin(slab, *slot) if slab else None
        pins.append(
            {
                "pin_index": pin_index,
                "center_y_m": center_y,
                "center_z_m": center_z,
                "min_goal_xyz_m": lower,
                "max_goal_xyz_m": upper,
                "min_tip_goal_x_m": min(point[0] for point in corners[4:]),
                "slot_cross_section_fits": margin >= 0.,
                "minimum_slot_margin_mm": margin * 1000.,
                "intersects_board_slab": bool(slab),
                "board_slab_slot_fits": (
                    None if slab_margin is None else slab_margin >= 0.
                ),
                "minimum_board_slab_slot_margin_mm": (
                    None if slab_margin is None else slab_margin * 1000.
                ),
            }
        )
    board_fits = [
        pin["board_slab_slot_fits"] for pin in pins if pin["intersects_board_slab"]
    ]
    angle = 2. * math.atan2(math.hypot(*relative_q[1:]), abs(relative_q[0]))
    return {
        "pins": pins,
        "goal_relative_pose": list(relative_p + relative_q),
        "pin_slot_fit": all(pin["slot_cross_section_fits"] for pin in pins),
        "pin_in_board_slot_fit": all(board_fits) if board_fits else None,
        "depth_ok": (
            ROOT_MIN_X_M <= relative_p[0] <= ROOT_MAX_X_M
            and all(
                pin["min_goal_xyz_m"][0] >= ROOT_MIN_X_M
                and pin["min_tip_goal_x_m"] >= tip_depth_threshold
                for pin in pins
            )
        ),
        "pin_length_m": pin_length,
        "minimum_required_tip_goal_x_m": tip_depth_threshold,
        "axial_error_mm": relative_p[0] * 1000.,
        "lateral_error_mm": math.hypot(*relative_p[1:]) * 1000.,
        "position_error_mm": math.hypot(*world_delta) * 1000.,
        "angle_error_deg": math.degrees(angle),
    }
