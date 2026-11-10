"""Segmentation labels for the UNet (v4): plug pins and socket openings, per pixel.

The recorded per-pixel classes (data/schema.SEG_CLASSES: other, plug, robot, board, table) have no
"pin" or "opening" class. Those two are drawn here by projecting the port geometry into each image,
as a human annotator would outline them on a real image (pins on the plug, the openings in the
socket face). They are training labels only; the UNet's own prediction is what places the crops.

Geometry (mani_skill .../wminsertion_v2/ports.py): plug frame origin at the pin roots, +X toward the
tips; pin i is a box x in [0, 2L], centre (y_i, z_i), half sizes (hy_i, hz_i). When inserted the plug
frame equals the goal frame, the socket face is at x = 0 and opening i is the rectangle centre
(y_i, z_i), half sizes (hy_i + c, hz_i + c), c = clearance. Per image:
    charger(world) = goal(world) . rel        rel = recorded charger pose in the goal frame (eval_rel)
goal(world) is fixed for a grasp (one episode per grasp) and is read once from the simulator
(extract_grasp_geometry). It is checked against the recorded hand poses: hand(s)^-1 . charger(s)
must be constant (the plug is rigid in the hand).

Classes of the UNet target (CLASSES): the five recorded ones + pins + opening. A pin pixel is a
pixel inside a projected pin box that the renderer labelled "plug"; an opening pixel is inside a
projected opening rectangle and not covered by the plug or the robot.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("other", "plug", "robot", "board", "table", "pins", "opening")
PINS, OPENING = 5, 6
_REC_PLUG, _REC_ROBOT = 1, 2               # data.schema.SEG_CLASSES


def pose7_to_T(v: np.ndarray) -> np.ndarray:
    """(x, y, z, qw, qx, qy, qz) -> 4x4."""
    p, (w, x, y, z) = np.asarray(v[:3], np.float64), np.asarray(v[3:7], np.float64) / np.linalg.norm(v[3:7])
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = p
    return T


def pin_corners(geom: dict) -> list[np.ndarray]:
    """8 corners (plug frame, (8, 3)) of every pin box."""
    L2 = 2 * geom["pin_half_length_m"]
    out = []
    for pin in geom["pins"]:
        (cy, cz), (hy, hz) = pin["center_yz_m"], pin["half_size_yz_m"]
        out.append(np.array([[x, cy + sy * hy, cz + sz * hz] for x in (0.0, L2) for sy in (-1, 1) for sz in (-1, 1)]))
    return out


def opening_corners(geom: dict) -> list[np.ndarray]:
    """4 corners (goal frame, (4, 3)) of every socket opening, in polygon order."""
    c = geom["clearance_m"]
    out = []
    for pin in geom["pins"]:
        (cy, cz), (hy, hz) = pin["center_yz_m"], pin["half_size_yz_m"]
        hy, hz = hy + c, hz + c
        out.append(np.array([[0.0, cy - hy, cz - hz], [0.0, cy + hy, cz - hz], [0.0, cy + hy, cz + hz], [0.0, cy - hy, cz + hz]]))
    return out


def project(points_world: np.ndarray, cam_T_world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) world -> (N, 2) pixels of the crop and (N,) camera depth (m)."""
    pc = points_world @ cam_T_world[:3, :3].T + cam_T_world[:3, 3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


def _fill(mask: np.ndarray, pix: np.ndarray, depth: np.ndarray):
    """Fill the convex hull of projected points (skipped if any point is behind the camera)."""
    if (depth <= 1e-4).any():
        return
    hull = cv2.convexHull(np.round(pix * 16).astype(np.int32))           # 4 bits of sub-pixel precision
    cv2.fillConvexPoly(mask, hull, 1, lineType=cv2.LINE_8, shift=4)


def label_map(seg: np.ndarray, rel7: np.ndarray, cam_T_world: np.ndarray, K: np.ndarray, geom: dict) -> np.ndarray:
    """Target class map (H, W) uint8 for one image: recorded classes + pins + opening."""
    goal = np.asarray(geom["goal_world"], np.float64)
    charger = goal @ pose7_to_T(rel7)
    H, W = seg.shape
    pins, opening = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    for corners in pin_corners(geom):
        pw = corners @ charger[:3, :3].T + charger[:3, 3]
        _fill(pins, *project(pw, cam_T_world, K))
    for corners in opening_corners(geom):
        pw = corners @ goal[:3, :3].T + goal[:3, 3]
        _fill(opening, *project(pw, cam_T_world, K))
    out = seg.astype(np.uint8).copy()
    out[(opening > 0) & (seg != _REC_PLUG) & (seg != _REC_ROBOT)] = OPENING
    out[(pins > 0) & (seg == _REC_PLUG)] = PINS
    return out


# ---------------------------------------------------------------------------- per-grasp geometry
def extract_grasp_geometry(robot, meta: dict, check: dict | None = None) -> dict:
    """Reset the episode of a grasp (its seed, pitch, grasp position) and read the goal pose and the
    port geometry. `check` = one recorded trajectory's arrays (hand, eval_rel): the plug must be rigid
    in the hand, i.e. hand(s)^-1 . goal . rel(s) constant; its spread (mm) is returned."""
    from mani_skill.examples.motionplanning.panda.solutions.wminsertion_v2.scripted import _copy_pose
    robot.reset(meta["episode_seed"], meta["pitch_deg"], meta["grasp_pos_m"])
    goal = _copy_pose(robot.base.goal_pose).to_transformation_matrix().astype(np.float64)
    spec = robot.port_spec
    geom = {"goal_world": goal.tolist(), "pin_half_length_m": float(spec.pin_half_length_m),
            "clearance_m": float(spec.clearance_m), "pins": [p.as_dict() for p in spec.pins],
            "port": meta["port"], "grasp": meta["grasp"], "episode_seed": meta["episode_seed"]}
    if check is not None:
        hand, rel = check["hand"].astype(np.float64), check["eval_rel"]
        T = np.stack([np.linalg.inv(hand[s]) @ goal @ pose7_to_T(rel[s]) for s in range(0, len(rel), 5)])
        geom["rigid_spread_mm"] = float(np.linalg.norm(T[:, :3, 3] - T[:1, :3, 3], axis=1).max() * 1000)
        geom["rigid_rot_spread_deg"] = float(np.degrees(np.arccos(np.clip(
            (np.einsum("nij,ij->n", T[:, :3, :3], T[0, :3, :3]) - 1) / 2, -1, 1))).max())
    return geom


def load_geometry(root: str | Path, data_root: str | Path, traj_path: str | Path) -> dict:
    """Geometry file of the grasp a trajectory belongs to (<root>/<port>/<grasp>.json)."""
    rel = Path(traj_path).relative_to(data_root)
    return json.loads((Path(root) / rel.parts[0] / f"{rel.parts[1]}.json").read_text())
