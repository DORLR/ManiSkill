"""Torch datasets (v3).

FrameDataset         autoencoder: one wrist frame (RGB-D, intrinsics, camera pose, hand pose, port
                     axes) + the label of the loss only (h_star, FK of this grasp's one insertion)
AdapterDataset       joint adapter: measured joints and tracking history at a control step, the
                     next 8 commanded targets -> the next 8 measured joints
LatentWindowDataset  world model: 4 context latents, a target latent at a random time shift
                     (NWM: up to +-16 s), the robot-state / commanded-path condition, rel_t
LatentPolicyDataset  latent FM policy: 4 context latents + measured joints -> next 16 targets
ImagePolicyDataset   latent-free baseline policy: the same 4 frames as raw inputs + joints

Every model input is a real-robot quantity; nothing here reads grasp parameters, shape identity or
simulator truth (tests/test_no_leak.py).
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
from .conditions import COND_DIM, rel_t_seconds, wm_condition

WM_STRIDE = 2                               # images between context frames: 2 x 0.2 s = 0.4 s
CONTEXT = 4
WM_STEP = schema.IMAGE_EVERY * WM_STRIDE    # control steps in 0.4 s (8)
HISTORY_STEPS = 24                          # adapter tracking history: 1.2 s
MAX_SHIFT_IMAGES = 80                       # NWM's +-16 s at 5 images/s
POLICY_KINDS = ("reference", "insert", "perturbed")   # successful expert trajectories only
POLICY_CHUNK = 16                                      # 0.8 s of joint targets
# The board is mounted upright; a fitted insertion axis with a vertical component above this means
# RANSAC found another plane (0.3 % of wm_v1 trajectories). Those trajectories are skipped.
MAX_AXIS_UP = 0.1


def port_axes_ok(entry: dict) -> bool:
    return entry["R_P"] is not None and abs(entry["R_P"][2][0]) <= MAX_AXIS_UP


def _jpeg(f, j: int) -> np.ndarray:
    off = f["rgb_offsets"]
    return np.asarray(Image.open(io.BytesIO(f["rgb_jpeg"][off[j]:off[j + 1]].tobytes())).convert("RGB"))


def _tensors(d: dict) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}


def context_indices(j: int) -> list[int]:
    """Image indices of the 4 context frames ending at j (oldest first, 0.4 s apart); before the
    start of a trajectory the first frame is repeated (used by the policies)."""
    return [max(0, j - WM_STRIDE * c) for c in range(CONTEXT - 1, -1, -1)]


# ---------------------------------------------------------------------------- autoencoder
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
            out = {"rgb": _jpeg(f, j), "depth": f["depth"][j].astype(np.int32), "K": f["K"].astype(np.float32),
                   "cam_T_world": f["cam_T_world"][j].astype(np.float32), "hand": f["hand"][step].astype(np.float32)}
        out = _tensors(out)
        out["R_P"] = torch.tensor(e["R_P"], dtype=torch.float32)
        out["h_star"] = torch.tensor(meta["H_star"], dtype=torch.float32)      # loss label only
        return out


# ---------------------------------------------------------------------------- joint adapter
def adapter_io(qpos: np.ndarray, action: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Adapter input / output at control step s.

    input  (231,) measured joints q[s] (7) | tracking history: for the 24 steps before s, commanded
           target minus the joints measured at that step (168; zeros before the start) | the next 8
           commanded targets relative to q[s] (56)
    output (56,)  the joints measured after each of the next 8 commands, relative to q[s]
    qpos has T+1 rows (q[s+1] is measured after command s), action has T rows."""
    q = qpos[s, :7]
    hist = np.zeros((HISTORY_STEPS, 7), np.float32)
    lo = max(0, s - HISTORY_STEPS)
    if s > lo:
        hist[HISTORY_STEPS - (s - lo):] = action[lo:s] - qpos[lo:s, :7]
    cmd = action[s:s + WM_STEP]
    nxt = qpos[s + 1:s + 1 + WM_STEP, :7]
    if len(cmd) < WM_STEP:                         # end of trajectory: hold the last command / joints
        cmd = np.concatenate([cmd, np.repeat(cmd[-1:], WM_STEP - len(cmd), 0)])
        nxt = np.concatenate([nxt, np.repeat(nxt[-1:], WM_STEP - len(nxt), 0)])
    x = np.concatenate([q, hist.reshape(-1), (cmd - q).reshape(-1)])
    return x.astype(np.float32), (nxt - q).reshape(-1).astype(np.float32)


class AdapterDataset(Dataset):
    """All control steps of all trajectories of a split (every kind: jams must be in the data)."""

    def __init__(self, entries: list[dict], split: str):
        self.q, self.a, self.items = [], [], []
        for e in entries:
            if e["split"] != split:
                continue
            with np.load(e["path"]) as f:
                qpos, action = f["qpos"].astype(np.float32), f["action"].astype(np.float32)
            if len(action) < 1:
                continue
            fi = len(self.q)
            self.q.append(qpos)
            self.a.append(action)
            self.items += [(fi, s) for s in range(len(action))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k: int) -> dict:
        fi, s = self.items[k]
        x, y = adapter_io(self.q[fi], self.a[fi], s)
        return {"x": torch.from_numpy(x), "y": torch.from_numpy(y)}


# ---------------------------------------------------------------------------- latents
class _LatentFiles:
    """Latent files of a split (written by training/encode.py), cached in memory per file."""

    def __init__(self, latent_root, entries, split, data_root, keep=lambda e: True):
        self.files, self.entries = [], []
        for e in entries:
            if e["split"] != split or not e["n_images"] or not port_axes_ok(e) or not keep(e):
                continue
            p = Path(latent_root) / Path(e["path"]).relative_to(data_root)
            if p.exists():
                self.files.append(str(p))
                self.entries.append(e)
        self._cache = {}

    def load(self, fi: int) -> dict:
        if fi not in self._cache:
            if len(self._cache) > 512:
                self._cache.clear()
            with np.load(self.files[fi]) as f:
                self._cache[fi] = {k: f[k] for k in f.files}
        return self._cache[fi]


class LatentWindowDataset(Dataset):
    """World-model samples. Context: images j-6, j-4, j-2, j (0.4 s apart; j >= 6). Target: image
    j + m, with a time shift m drawn per sample:
        with probability p_step  m = +2 (0.4 s: the step used when interacting with the policy)
        otherwise                m uniform over the valid shifts in [-80, +80] \\ {0} (+-16 s, NWM)
    `fixed_shift` (validation) keeps only windows where j + fixed_shift exists and uses that shift."""

    def __init__(self, latent_root, entries, split, data_root, scale: float, p_step: float = 0.5,
                 fixed_shift: int | None = None):
        self.lat = _LatentFiles(latent_root, entries, split, data_root)
        self.scale, self.p_step, self.fixed = scale, p_step, fixed_shift
        self.items = []
        for fi, e in enumerate(self.lat.entries):
            n = e["n_images"]
            for j in range((CONTEXT - 1) * WM_STRIDE, n):
                if fixed_shift is None or 0 <= j + fixed_shift < n:
                    self.items.append((fi, j))

    def __len__(self):
        return len(self.items)

    def _shift(self, j: int, n: int) -> int:
        if self.fixed is not None:
            return self.fixed
        # torch's generator: DataLoader seeds it differently in every worker (numpy's would repeat)
        if j + WM_STRIDE < n and float(torch.rand(())) < self.p_step:
            return WM_STRIDE
        lo, hi = max(-MAX_SHIFT_IMAGES, -j), min(MAX_SHIFT_IMAGES, n - 1 - j)
        m = 0
        while m == 0:
            m = int(torch.randint(lo, hi + 1, ()))
        return m

    def __getitem__(self, k: int) -> dict:
        fi, j = self.items[k]
        d, e = self.lat.load(fi), self.lat.entries[fi]
        n = len(d["img_step"])
        m = self._shift(j, n)
        ctx = [j - WM_STRIDE * c for c in range(CONTEXT - 1, -1, -1)]
        steps = d["img_step"].astype(int)
        y = wm_condition(d["hand"], d["hand_cmd"], steps[ctx], int(steps[j]), int(steps[j + m]),
                         np.asarray(e["R_P"], np.float64))
        return {"x_cond": torch.from_numpy(d["mean"][ctx].astype(np.float32)) * self.scale,
                "x": torch.from_numpy(d["mean"][j + m].astype(np.float32)) * self.scale,
                "y": torch.from_numpy(y),
                "rel_t": torch.tensor(rel_t_seconds(int(steps[j]), int(steps[j + m])), dtype=torch.float32)}


assert COND_DIM == 216


# ---------------------------------------------------------------------------- policies
def policy_chunk(qpos: np.ndarray, action: np.ndarray, s0: int) -> np.ndarray:
    """The next 16 commanded targets relative to the measured joints at s0 (held at the end)."""
    act = action[s0:s0 + POLICY_CHUNK]
    if len(act) == 0:
        act = qpos[s0:s0 + 1, :7]
    if len(act) < POLICY_CHUNK:
        act = np.concatenate([act, np.repeat(act[-1:], POLICY_CHUNK - len(act), 0)])
    return (act - qpos[s0][:7]).reshape(-1).astype(np.float32)


def _policy_entry(e: dict) -> bool:
    return e["kind"] in POLICY_KINDS and e.get("outcome") == "inserted"


class LatentPolicyDataset(Dataset):
    """FM policy samples: 4 context latents (repeating the first frame early on), the measured
    joints now, and the expert's next 16 joint targets (successful expert trajectories only)."""

    def __init__(self, latent_root, entries, split, data_root, scale: float):
        self.lat = _LatentFiles(latent_root, entries, split, data_root, keep=_policy_entry)
        self.scale = scale
        self.items = [(fi, j) for fi, e in enumerate(self.lat.entries) for j in range(e["n_images"])]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k: int) -> dict:
        fi, j = self.items[k]
        d = self.lat.load(fi)
        s0 = int(d["img_step"][j])
        return {"z": torch.from_numpy(d["mean"][context_indices(j)].astype(np.float32)) * self.scale,
                "q": torch.from_numpy(d["qpos"][s0, :7].astype(np.float32)),
                "a": torch.from_numpy(policy_chunk(d["qpos"], d["action"], s0))}


class ImagePolicyDataset(Dataset):
    """Latent-free baseline: the same 4 frames, as the encoder's raw inputs, + joints + targets."""

    def __init__(self, entries: list[dict], split: str):
        self.entries = [e for e in entries if e["split"] == split and e["n_images"] and port_axes_ok(e)
                        and _policy_entry(e)]
        self.items = [(i, j) for i, e in enumerate(self.entries) for j in range(e["n_images"])]

    def __len__(self):
        return len(self.items)

    def action(self, k: int) -> np.ndarray:
        i, j = self.items[k]
        with np.load(self.entries[i]["path"]) as f:
            return policy_chunk(f["qpos"], f["action"], int(f["img_step"][j]))

    def __getitem__(self, k: int) -> dict:
        i, j = self.items[k]
        e = self.entries[i]
        ctx = context_indices(j)
        with np.load(e["path"]) as f:
            steps = f["img_step"][ctx].astype(int)
            out = {"rgb": np.stack([_jpeg(f, c) for c in ctx]), "depth": f["depth"][ctx].astype(np.int32),
                   "K": f["K"].astype(np.float32), "cam_T_world": f["cam_T_world"][ctx].astype(np.float32),
                   "hand": f["hand"][steps].astype(np.float32),
                   "q": f["qpos"][steps[-1], :7].astype(np.float32),
                   "a": policy_chunk(f["qpos"], f["action"], int(steps[-1]))}
        out = _tensors(out)
        out["R_P"] = torch.tensor(e["R_P"], dtype=torch.float32)
        return out
