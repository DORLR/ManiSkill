"""Depth-sensor noise model applied to the wrist depth image before the method
sees it (RGB untouched, so the effect of depth/point-cloud noise is isolated).

Level "strong" (meant to be worse than a typical short-range RGB-D camera):
  * axial Gaussian noise growing with distance:
      sigma(z) = 0.5 mm + 2.0 mm * (z / 150 mm)^2   (1.4 mm @ 10 cm, 2.5 @ 15, 4.1 @ 20)
  * low-frequency wavy offset: 8x8 Gaussian grid (1.5 mm) bilinearly upsampled
  * depth edges (3x3 max-min > 5 mm): 60 % dropped, 20 % flying pixels (3x3 mean)
  * thin protrusions (> 4 mm in front of the 7x7 median, e.g. pins): 50 % dropped
  * 3 % random dropout
  * rounded to whole millimetres (the environment's int16 format)
Dropped pixels are 0 (no return), as the environment already uses for no depth.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

LEVELS = {
    "strong": dict(sigma0=0.5, sigma150=2.0, wave=1.5, edge_thresh=5.0, edge_drop=0.6, edge_fly=0.2,
                   thin_thresh=4.0, thin_drop=0.5, drop=0.03),
}


def noisy_depth(depth_mm: np.ndarray, rng: np.random.Generator, level: str = "strong") -> np.ndarray:
    p = LEVELS[level]
    d = depth_mm.astype(np.float64)
    valid = d > 0
    h, w = d.shape
    out = d.copy()
    # distance-dependent axial noise
    sigma = p["sigma0"] + p["sigma150"] * (d / 150.0) ** 2
    out += rng.normal(size=d.shape) * sigma
    # low-frequency wave
    grid = rng.normal(size=(8, 8)) * p["wave"]
    out += ndimage.zoom(grid, (h / 8, w / 8), order=1)[:h, :w]
    # edges: flying pixels and dropouts
    rng_local = ndimage.maximum_filter(d, 3) - ndimage.minimum_filter(np.where(valid, d, 1e9), 3)
    edge = valid & (rng_local > p["edge_thresh"])
    u = rng.random(d.shape)
    fly = edge & (u < p["edge_fly"])
    out[fly] = ndimage.uniform_filter(d, 3)[fly]
    drop = edge & (u >= p["edge_fly"]) & (u < p["edge_fly"] + p["edge_drop"])
    # thin structures in front of their surroundings
    thin = valid & (ndimage.median_filter(d, 7) - d > p["thin_thresh"]) & (rng.random(d.shape) < p["thin_drop"])
    drop |= thin
    drop |= rng.random(d.shape) < p["drop"]
    out[drop | ~valid] = 0
    return np.clip(np.round(out), 0, 32767).astype(np.int16)


def apply(obs, rng, level):
    """Return a copy of an Obs with a noisy depth image (level None = unchanged)."""
    if not level:
        return obs
    from dataclasses import replace
    return replace(obs, depth_mm=noisy_depth(obs.depth_mm, rng, level))
