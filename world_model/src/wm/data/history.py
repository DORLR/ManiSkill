"""History samples for the world model and the success classifier (v4).

The model sees the last HIST slots of 0.2 s (6.2 s up to now, one slot per recorded image):
    vision   the latent of the image in each slot, or nothing:
               - before the start of the trajectory (padding)
               - masked at random when training (below)
               - in imagination, the 0.2 s slots between two imagined 0.4 s frames
             attention only sees the slots that have a latent (time-series classification with
             irregular sampling: mTAN / Raindrop / masked transformers); no "not seen" flag is given
    proprio  every slot (never masked): measured and commanded hand pose at that slot's control step,
             relative to the current measured hand pose, in port axes (2 x 9 values), and the slot's
             time (seconds before now, <= 0)

Vision masking when training (the model must not depend on seeing the last frames -- near contact the
plug covers the opening for the rest of the insertion):
    with p_tail   the last L slots, L uniform in 1..(available - 1)   (main case: occlusion until the end)
    with p_mid    a random contiguous block, not touching the oldest available slot
    always        each remaining slot dropped with p_drop (irregular sampling)
At least one slot keeps its latent.
"""

from __future__ import annotations

import numpy as np
import torch

from .conditions import N_WAYPOINTS, POSE_DIM, pose_feature

HIST = 32                     # slots of 0.2 s (one recorded image each): 6.2 s up to now
PROPRIO_DIM = 2 * POSE_DIM    # measured + commanded hand pose
PATH_DIM = N_WAYPOINTS * POSE_DIM


def vision_mask(avail: np.ndarray, rng: np.random.Generator, p_tail: float, p_mid: float, p_drop: float) -> np.ndarray:
    """avail (HIST,) bool, oldest first -> seen (HIST,) bool (a subset of avail, never empty)."""
    seen = avail.copy()
    idx = np.nonzero(avail)[0]
    n = len(idx)
    if n >= 2:
        u = rng.random()
        if u < p_tail:
            L = int(rng.integers(1, n))
            seen[idx[n - L:]] = False
        elif u < p_tail + p_mid:
            L = int(rng.integers(1, n))
            a = int(rng.integers(1, n - L + 1))
            seen[idx[a:a + L]] = False
        drop = rng.random(HIST) < p_drop
        keep_one = seen.copy()
        seen &= ~drop
        if not seen.any():
            seen[np.nonzero(keep_one)[0][0]] = True
    return seen


def history(d: dict, j: int, R_P: np.ndarray, scale: float, seen: np.ndarray | None = None) -> dict:
    """History ending at image j of a latent file d.
    -> z (HIST, 64, C) scaled latents (zeros where unseen), seen (HIST,) bool, proprio (HIST, 18),
       dt (HIST,) seconds (<= 0), avail (HIST,) bool."""
    steps = d["img_step"].astype(int)
    slots = j - np.arange(HIST - 1, -1, -1)
    avail = slots >= 0
    sl = np.clip(slots, 0, None)
    ref = d["hand"][steps[j]].astype(np.float64)
    st = steps[sl]
    meas = d["hand"][st].astype(np.float64)
    cmd = d["hand_cmd"][np.clip(st, 0, len(d["hand_cmd"]) - 1)].astype(np.float64)
    proprio = np.concatenate([pose_feature(meas, ref, R_P), pose_feature(cmd, ref, R_P)], -1)
    proprio[~avail] = 0
    dt = (st - steps[j]) / 20.0
    dt[~avail] = 0
    seen = avail if seen is None else seen & avail
    z = d["mean"][sl].astype(np.float32) * scale
    z[~seen] = 0
    return {"z": z, "seen": seen, "proprio": proprio.astype(np.float32), "dt": dt.astype(np.float32), "avail": avail}


def to_tensors(h: dict) -> dict:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in h.items()}


# ---------------------------------------------------------------------------- datasets
class LatentFiles:
    """v4 latent files (training/encode4.py or online/actor.py), cached in memory per file."""

    def __init__(self, paths: list[str]):
        self.paths = list(paths)
        self._cache = {}

    def add(self, paths: list[str]):
        self.paths += list(paths)

    def __len__(self):
        return len(self.paths)

    def load(self, fi: int) -> dict:
        if fi not in self._cache:
            if len(self._cache) > 2048:
                self._cache.clear()
            with np.load(self.paths[fi]) as f:
                self._cache[fi] = {k: f[k] for k in f.files}
        return self._cache[fi]

    def n_images(self, fi: int) -> int:
        return len(self.load(fi)["img_step"])


def _rng() -> np.random.Generator:
    # torch's generator is seeded differently in every DataLoader worker (numpy's would repeat)
    return np.random.default_rng(int(torch.randint(0, 2 ** 62, ())))


MAX_SHIFT_IMAGES = 80          # NWM's +-16 s at 5 images/s
WM_STRIDE_IMAGES = 2           # 0.4 s, the step used with the policy


class HistoryWindowDataset(torch.utils.data.Dataset):
    """World-model samples: history ending at image j, a target image j + m (half the samples m = +2,
    0.4 s; the rest uniform over the valid shifts in +-16 s, NWM), the commanded hand path from now to
    the target (16 waypoints, relative to the current measured hand, port axes), rel_t, and the target
    frame's geometry (only to read the plug pose from latents in the loss). `fixed_shift`: validation."""

    def __init__(self, files: LatentFiles, scale: float, mask: dict, p_step: float = 0.5,
                 fixed_shift: int | None = None, items: list | None = None):
        self.files, self.scale, self.mask, self.p_step, self.fixed = files, scale, mask, p_step, fixed_shift
        self.items = items if items is not None else self.index(files, fixed_shift)

    @staticmethod
    def index(files: LatentFiles, fixed_shift=None, first: int = 0) -> list:
        out = []
        for fi in range(first, len(files)):
            n = files.n_images(fi)
            out += [(fi, j) for j in range(n) if n > 1 and (fixed_shift is None or 0 <= j + fixed_shift < n)]
        return out

    def __len__(self):
        return len(self.items)

    def _shift(self, j, n, rng):
        if self.fixed is not None:
            return self.fixed
        if j + WM_STRIDE_IMAGES < n and rng.random() < self.p_step:
            return WM_STRIDE_IMAGES
        lo, hi = max(-MAX_SHIFT_IMAGES, -j), min(MAX_SHIFT_IMAGES, n - 1 - j)
        m = 0
        while m == 0:
            m = int(rng.integers(lo, hi + 1))
        return m

    def __getitem__(self, k):
        fi, j = self.items[k]
        d = self.files.load(fi)
        rng = _rng()
        n = len(d["img_step"])
        m = self._shift(j, n, rng)
        R_P = d["R_P"].astype(np.float64)
        avail = (j - np.arange(HIST - 1, -1, -1)) >= 0
        seen = vision_mask(avail, rng, **self.mask) if self.mask else avail
        h = history(d, j, R_P, self.scale, seen)
        steps = d["img_step"].astype(int)
        s_now, s_tgt = int(steps[j]), int(steps[j + m])
        ref = d["hand"][s_now].astype(np.float64)
        wp = np.rint(np.linspace(s_now, s_tgt, N_WAYPOINTS + 1)[1:]).astype(int)
        cmd = d["hand_cmd"][np.clip(wp, 0, len(d["hand_cmd"]) - 1)].astype(np.float64)
        path = pose_feature(cmd, ref, R_P).reshape(-1)
        t = int(j + m)
        out = to_tensors(h)
        out.update(x=torch.from_numpy(d["mean"][t].astype(np.float32)) * self.scale, path=torch.from_numpy(path),
                   rel_t=torch.tensor((s_tgt - s_now) / 20.0, dtype=torch.float32),
                   geom={"K": torch.from_numpy(d["K"].astype(np.float32)),
                         "cam_T_world": torch.from_numpy(d["cam_T_world"][t].astype(np.float32)),
                         "hand": torch.from_numpy(d["hand"][s_tgt].astype(np.float32)),
                         "R_P": torch.from_numpy(R_P.astype(np.float32)),
                         "plug_uv0": torch.from_numpy(d["plug_uv0"][t].astype(np.int64)),
                         "hole_uv0": torch.from_numpy(d["hole_uv0"][t].astype(np.int64))})
        return out


class SuccessDataset(torch.utils.data.Dataset):
    """Success-classifier samples: history ending at image j (same masking) and the label "inserted at
    image j" (simulator truth at training time; the contact switch on the real robot)."""

    def __init__(self, files: LatentFiles, scale: float, mask: dict, items: list | None = None):
        self.files, self.scale, self.mask = files, scale, mask
        self.items = items if items is not None else self.index(files)

    @staticmethod
    def index(files: LatentFiles, first: int = 0) -> list:
        out = []
        for fi in range(first, len(files)):
            d = files.load(fi)
            out += [(fi, j, bool(d["inserted"][j])) for j in range(len(d["img_step"]))]
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        fi, j, _ = self.items[k]
        d = self.files.load(fi)
        rng = _rng()
        avail = (j - np.arange(HIST - 1, -1, -1)) >= 0
        seen = vision_mask(avail, rng, **self.mask) if self.mask else avail
        out = to_tensors(history(d, j, d["R_P"].astype(np.float64), self.scale, seen))
        out["label"] = torch.tensor(float(d["inserted"][j]))
        return out
