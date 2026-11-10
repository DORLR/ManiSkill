"""Where to cut the two native-resolution crops (v4): plug pins and socket opening.

Per frame, from the UNet's class map (never from labels or simulator truth):
    plug crop     centred on the predicted pins (centroid of the "pins" pixels); if fewer than MIN_PX
                  pins pixels (pins inside the hole), the predicted plug body; if neither, the last
                  plug centre (the plug is rigid in the hand and the camera is on the hand, so its
                  image position barely changes)
    opening crop  centred on the predicted openings; their 3-D position (world) is remembered: ray of
                  the centroid x the median depth of the predicted board pixels around the openings
                  (the openings themselves are holes). When the openings are not visible (covered by
                  the plug near contact) the remembered point is projected with the current camera
                  pose (FK + hand-eye mount): the socket does not move. Before any opening was seen
                  with depth, the last 2-D opening centre, else the plug crop centre.
Crops are CROP x CROP pixels of the 320 x 320 image, at native resolution (no resizing), shifted to
stay inside the image. The crop origins (u0, v0) and the frame's intrinsics / camera pose give every
crop pixel its viewing ray, so the encoder knows where the two crops are relative to each other.
"""

from __future__ import annotations

import cv2
import numpy as np

from .labels import OPENING, PINS

CROP = 128
IMG = 320
MIN_PX = 10
_PLUG, _BOARD = 1, 3
DEPTH_UNIT_M = 1e-4


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    n = int(mask.sum())
    if n < MIN_PX:
        return None
    v, u = np.nonzero(mask)
    return np.array([u.mean(), v.mean()])


def _origin(centre: np.ndarray) -> np.ndarray:
    return np.clip(np.round(centre - CROP / 2), 0, IMG - CROP).astype(np.int32)


class CropPlacer:
    """Stateful over one episode / trajectory (the remembered opening point and plug centre)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.hole_world, self.hole_uv, self.plug_uv = None, None, None

    def __call__(self, classes: np.ndarray, depth: np.ndarray, K: np.ndarray, cam_T_world: np.ndarray) -> dict:
        """classes (320, 320) UNet argmax, depth (320, 320) uint16 [0.1 mm], K (3, 3), cam_T_world (4, 4)
        -> plug_uv0, hole_uv0 (2,) int crop origins, and how each centre was found (bookkeeping)."""
        K, cam = np.asarray(K, np.float64), np.asarray(cam_T_world, np.float64)
        # plug
        c = _centroid(classes == PINS)
        how_plug = "pins"
        if c is None:
            c, how_plug = _centroid(classes == _PLUG), "plug"
        if c is None:
            c, how_plug = (self.plug_uv if self.plug_uv is not None else np.array([IMG / 2, IMG / 2])), "memory"
        self.plug_uv = c
        # opening
        opening = classes == OPENING
        h = _centroid(opening)
        how_hole = "opening"
        if h is not None:
            self.hole_uv = h
            ring = cv2.dilate(opening.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
            d = depth[ring & (classes == _BOARD) & (depth > 0)]
            if len(d) >= MIN_PX:
                z = float(np.median(d)) * DEPTH_UNIT_M
                ray = np.linalg.solve(K, np.array([h[0], h[1], 1.0]))
                p_cam = ray * z                                               # z along the optical axis
                R_wc = cam[:3, :3].T
                self.hole_world = R_wc @ (p_cam - cam[:3, 3])
        elif self.hole_world is not None:
            pc = cam[:3, :3] @ self.hole_world + cam[:3, 3]
            if pc[2] > 1e-3:
                uv = K @ pc
                h, how_hole = uv[:2] / uv[2], "memory_3d"
        if h is None:
            h, how_hole = (self.hole_uv, "memory_2d") if self.hole_uv is not None else (c, "plug")
        return {"plug_uv0": _origin(c), "hole_uv0": _origin(h), "how_plug": how_plug, "how_hole": how_hole}


def crop_intrinsics(K: np.ndarray, uv0) -> np.ndarray:
    """Intrinsics of a crop whose top-left pixel is uv0 = (u0, v0) of the 320 image."""
    Kc = np.array(K, np.float64, copy=True)
    Kc[0, 2] -= uv0[0]
    Kc[1, 2] -= uv0[1]
    return Kc
