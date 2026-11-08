"""Torch datasets.

FrameDataset         one image per sample (VAE training): wrist RGB-D, camera, FK, joints,
                     port axes, and the grasp's inserted hand pose H_star (label only)
LatentWindowDataset  world-model samples from pre-computed latents: 4 context latents,
                     the target latent one step later, and the action condition (see
                     action_condition: the joint-target chunk between, optionally with the
                     tracking history before it)
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
WM_STEP = schema.IMAGE_EVERY * WM_STRIDE   # control steps per world-model step (8)
HISTORY_STEPS = 24                         # tracking history given to the world model: 1.2 s


def action_condition(action: np.ndarray, qpos: np.ndarray, s0: int, history: bool) -> np.ndarray:
    """World-model action input at control step s0 (all real-robot quantities):
    the next WM_STEP joint targets relative to the measured joints at s0 (56 values), and, if
    `history`, for each of the HISTORY_STEPS control steps before s0 the commanded target minus the
    joints measured at that step (168 values; zeros before the trajectory starts). When the plug
    is blocked the commands keep moving and the joints do not, so this gap grows."""
    cur = action[s0:s0 + WM_STEP]
    if len(cur) == 0:
        cur = qpos[s0:s0 + 1, :7]
    if len(cur) < WM_STEP:
        cur = np.concatenate([cur, np.repeat(cur[-1:], WM_STEP - len(cur), 0)])
    parts = [(cur - qpos[s0][:7]).reshape(-1)]
    if history:
        h = np.zeros((HISTORY_STEPS, 7), np.float32)
        lo = max(0, s0 - HISTORY_STEPS)
        if s0 > lo:
            h[HISTORY_STEPS - (s0 - lo):] = action[lo:s0] - qpos[lo:s0, :7]
        parts.append(h.reshape(-1))
    return np.concatenate(parts).astype(np.float32)
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


POLICY_KINDS = ("reference", "insert", "perturbed")   # successful expert trajectories only
POLICY_CHUNK = 16                                      # 0.8 s of joint targets


class LatentPolicyDataset(Dataset):
    """FM policy samples from pre-computed latents: 4 context latents (0.4 s apart) and the next
    16 commanded joint targets relative to the measured joints (held at the end of a trajectory)."""

    def __init__(self, latent_root: str | Path, entries: list[dict], split: str, data_root: str | Path,
                 scale: float):
        self.scale = scale
        self.files, self.items = [], []
        for e in entries:
            if (e["split"] != split or not e["n_images"] or not port_axes_ok(e)
                    or e["kind"] not in POLICY_KINDS or e.get("outcome") != "inserted"):
                continue
            p = Path(latent_root) / Path(e["path"]).relative_to(data_root)
            if not p.exists():
                continue
            fi = len(self.files)
            self.files.append(str(p))
            self.items += [(fi, j) for j in range(e["n_images"])]
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
        ctx = [max(0, j - WM_STRIDE * c) for c in range(CONTEXT - 1, -1, -1)]   # repeat the first frame early on
        s0 = int(d["img_step"][j])
        act = d["action"][s0:s0 + POLICY_CHUNK]
        if len(act) == 0:
            act = d["qpos"][s0:s0 + 1, :7]
        if len(act) < POLICY_CHUNK:
            act = np.concatenate([act, np.repeat(act[-1:], POLICY_CHUNK - len(act), 0)])
        return {"z": torch.from_numpy(d["mean"][ctx].astype(np.float32)) * self.scale,
                "a": torch.from_numpy((act - d["qpos"][s0][:7]).reshape(-1).astype(np.float32))}


class ImagePolicyDataset(Dataset):
    """Latent-free policy samples from the raw trajectories: the same 4 frames (0.4 s apart) the latent
    policy sees, with everything the VAE encoder takes per frame, and the same 16-target chunk."""

    def __init__(self, entries: list[dict], split: str):
        self.entries = [e for e in entries if e["split"] == split and e["n_images"] and port_axes_ok(e)
                        and e["kind"] in POLICY_KINDS and e.get("outcome") == "inserted"]
        self.items = [(i, j) for i, e in enumerate(self.entries) for j in range(e["n_images"])]

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _chunk(f, s0):
        act = f["action"][s0:s0 + POLICY_CHUNK]
        if len(act) == 0:
            act = f["qpos"][s0:s0 + 1, :7]
        if len(act) < POLICY_CHUNK:
            act = np.concatenate([act, np.repeat(act[-1:], POLICY_CHUNK - len(act), 0)])
        return (act - f["qpos"][s0][:7]).reshape(-1).astype(np.float32)

    def action(self, k: int) -> np.ndarray:
        i, j = self.items[k]
        with np.load(self.entries[i]["path"]) as f:
            return self._chunk(f, int(f["img_step"][j]))

    def __getitem__(self, k: int) -> dict:
        i, j = self.items[k]
        e = self.entries[i]
        ctx = [max(0, j - WM_STRIDE * c) for c in range(CONTEXT - 1, -1, -1)]   # as LatentPolicyDataset
        with np.load(e["path"]) as f:
            steps = f["img_step"][ctx].astype(int)
            out = {"rgb": np.stack([_jpeg(f, c) for c in ctx]), "depth": f["depth"][ctx].astype(np.int32),
                   "K": f["K"].astype(np.float32), "cam_T_world": f["cam_T_world"][ctx].astype(np.float32),
                   "hand": f["hand"][steps].astype(np.float32), "qpos": f["qpos"][steps].astype(np.float32),
                   "a": self._chunk(f, int(steps[-1]))}
        out = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}
        out["R_P"] = torch.tensor(e["R_P"], dtype=torch.float32)
        return out


class LatentWindowDataset(Dataset):
    """Reads <latents>/<relative trajectory path>.npz written by scripts/encode_latents.py."""

    def __init__(self, latent_root: str | Path, entries: list[dict], split: str, data_root: str | Path,
                 scale: float, history: bool = False):
        self.scale, self.history = scale, history
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
        s0 = int(d["img_step"][j])
        return {"x_cond": torch.from_numpy(d["mean"][ctx].astype(np.float32)) * self.scale,
                "x": torch.from_numpy(d["mean"][j + WM_STRIDE].astype(np.float32)) * self.scale,
                "y": torch.from_numpy(action_condition(d["action"], d["qpos"], s0, self.history)),
                "rel_t": torch.tensor(1.0)}
