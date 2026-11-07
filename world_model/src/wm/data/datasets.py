"""Torch datasets.

FrameDataset         one image per sample (VAE training): wrist RGB-D, camera, FK, joints,
                     port axes, and the grasp's inserted hand pose H_star (label only)
LatentWindowDataset  world-model samples from pre-computed latents: 4 context latents,
                     the target latent one step later, and the joint-target chunk between
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import schema

WM_STRIDE = 2      # images between world-model steps: 2 x 0.2 s = 0.4 s (8 control steps)
CONTEXT = 4
# The board is mounted upright; a fitted insertion axis with a vertical component above this
# means RANSAC found another plane (0.3 % of wm_v1 trajectories). Those trajectories are skipped.
MAX_AXIS_UP = 0.1


def port_axes_ok(entry: dict) -> bool:
    return entry["R_P"] is not None and abs(entry["R_P"][2][0]) <= MAX_AXIS_UP


def _jpeg(f, j: int) -> np.ndarray:
    off = f["rgb_offsets"]
    return np.asarray(Image.open(io.BytesIO(f["rgb_jpeg"][off[j]:off[j + 1]].tobytes())).convert("RGB"))


class FrameDataset(Dataset):
    def __init__(self, entries: list[dict], split: str, kinds: tuple[str, ...] | None = None):
        self.entries = [e for e in entries if e["split"] == split and port_axes_ok(e) and e["n_images"]
                        and (kinds is None or e["kind"] in kinds)]
        self.items = [(i, j) for i, e in enumerate(self.entries) for j in range(e["n_images"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k: int) -> dict:
        i, j = self.items[k]
        e = self.entries[i]
        with np.load(e["path"]) as f:
            meta = json.loads(str(f["meta"]))
            step = int(f["img_step"][j])
            out = {"rgb": _jpeg(f, j), "depth": f["depth"][j], "K": f["K"], "cam_T_world": f["cam_T_world"][j],
                   "hand": f["hand"][step], "qpos": f["qpos"][step]}
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
        out["depth"] = out["depth"].to(torch.int32)  # uint16 is not collatable on all torch builds
        out["R_P"] = torch.tensor(e["R_P"], dtype=torch.float32)
        out["h_star"] = torch.tensor(meta["H_star"], dtype=torch.float32)
        return out


class LatentWindowDataset(Dataset):
    """Reads <latents>/<relative trajectory path>.npz written by scripts/encode_latents.py."""

    def __init__(self, latent_root: str | Path, entries: list[dict], split: str, data_root: str | Path,
                 scale: float):
        self.scale = scale
        self.files, self.items = [], []
        for e in entries:
            if e["split"] != split or not e["n_images"] or not port_axes_ok(e):
                continue
            p = Path(latent_root) / Path(e["path"]).relative_to(data_root)
            if not p.exists():
                continue
            n = e["n_images"]
            fi = len(self.files)
            self.files.append(str(p))
            for j in range((CONTEXT - 1) * WM_STRIDE, n - WM_STRIDE):
                self.items.append((fi, j))
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _load(self, fi: int) -> dict:
        if fi not in self._cache:
            if len(self._cache) > 256:
                self._cache.clear()
            with np.load(self.files[fi]) as f:
                self._cache[fi] = {k: f[k] for k in f.files}
        return self._cache[fi]

    def __getitem__(self, k: int) -> dict:
        fi, j = self.items[k]
        d = self._load(fi)
        ctx = [j - WM_STRIDE * c for c in range(CONTEXT - 1, -1, -1)]
        s0, s1 = int(d["img_step"][j]), int(d["img_step"][j + WM_STRIDE])
        q0 = d["qpos"][s0][:7]
        act = d["action"][s0:s1] - q0                                   # (8, 7) targets relative to measured joints
        if act.shape[0] != schema.IMAGE_EVERY * WM_STRIDE:              # last image of a trajectory may be off-grid
            act = np.pad(act, ((0, schema.IMAGE_EVERY * WM_STRIDE - act.shape[0]), (0, 0)), mode="edge")
        return {"x_cond": torch.from_numpy(d["mean"][ctx].astype(np.float32)) * self.scale,
                "x": torch.from_numpy(d["mean"][j + WM_STRIDE].astype(np.float32)) * self.scale,
                "y": torch.from_numpy(act.reshape(-1).astype(np.float32)),
                "rel_t": torch.tensor(1.0)}
