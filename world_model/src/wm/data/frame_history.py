"""History samples from the recorded images themselves (v6: the encoder is trained with the world model, so
features cannot be stored beforehand). The same history, vision masking, time shifts, commanded path and
proprioception as data/history.py HistoryWindowDataset; the latents are replaced by the frames to encode.

Per sample
    frames_rgb / frames_depth (F, 320, 320[, 3])  the seen slots' images, newest first, padded; frame_slot (F,) their
                  slot, frame_ok (F,) which are real. At most `max_frames` are kept (the newest and a random choice of
                  the others): every kept frame goes through the encoder at every training step.
    seen / avail / proprio / dt / path / rel_t      as HistoryWindowDataset (seen = after the cap)
    tgt_rgb / tgt_depth                              the target image
    label_*  (prefixes g = the newest seen frame, t = the target frame)   hand, cam_T_world, inserted: for the label
                  head (remaining motion to inserted by FK; "inserted now")
    res / res_ok  for a +0.4 s target: where the hand really is at +0.2 s and +0.4 s relative to the command that led
                  there (pose features, position in mm x 1, rotation 6D x 10) -- the world model's second output,
                  replacing the joint adapter (decided with the user 2026-11-12)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import index as data_index
from .conditions import pose_feature
from .datasets import port_axes_ok
from .history import HIST, MAX_SHIFT_IMAGES, WM_STRIDE_IMAGES, _rng, commanded_path, history, vision_mask
from .outcome import inserted

RES_SCALE = 10.0            # pose features are in cm / unit vectors: x 10 -> mm, and rotations on a similar scale


def residual_feature(hand: np.ndarray, hand_cmd: np.ndarray, steps, ref: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """Measured hand pose at each control step of `steps` minus the commanded pose that led there (command
    index step - 1), as pose features relative to `ref` in port axes, x RES_SCALE -> (len(steps) * 9,)."""
    s = np.asarray(steps)
    c = np.clip(s - 1, 0, len(hand_cmd) - 1)
    return ((pose_feature(hand[s].astype(np.float64), ref, R_P) - pose_feature(hand_cmd[c].astype(np.float64), ref, R_P))
            * RES_SCALE).reshape(-1).astype(np.float32)


class FrameHistoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_root: str, ee_root: str, split: str, mask: dict | None, p_step: float = 0.5,
                 max_frames: int = 10, fixed_shift: int | None = None, tail: int = 0):
        self.root, self.ee = data_root, ee_root
        self.entries = [e for e in data_index.load(data_root) if e["split"] == split and e["n_images"] > 1 and port_axes_ok(e)
                        and (Path(ee_root) / Path(e["path"]).relative_to(data_root)).exists()]
        self.mask, self.p_step, self.max_frames, self.fixed, self.tail = mask, p_step, max_frames, fixed_shift, tail
        self.items = [(i, j) for i, e in enumerate(self.entries) for j in range(e["n_images"])
                      if fixed_shift is None or 0 <= j + fixed_shift < e["n_images"]]

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
        i, j = self.items[k]
        e = self.entries[i]
        rng = _rng()
        with np.load(e["path"]) as f:
            meta = json.loads(str(f["meta"]))
            steps, hand, rel = f["img_step"].astype(int), f["hand"], f["eval_rel"]
            cam, depth, jpeg, off = f["cam_T_world"], f["depth"], f["rgb_jpeg"], f["rgb_offsets"]
        with np.load(Path(self.ee) / Path(e["path"]).relative_to(self.root)) as g:
            hand_cmd = g["hand_cmd"]
        n = len(steps)
        m = self._shift(j, n, rng)
        R_P = np.asarray(e["R_P"], np.float64)
        avail = (j - np.arange(HIST - 1, -1, -1)) >= 0
        seen = vision_mask(avail, rng, **self.mask) if self.mask else avail.copy()
        if self.tail:                                              # validation: the last `tail` slots of vision hidden
            idx = np.nonzero(seen)[0]
            if len(idx) > 1:
                seen[idx[-min(self.tail, len(idx) - 1):]] = False
        idx = np.nonzero(seen)[0]
        if len(idx) > self.max_frames:                             # the newest and a random choice of the others
            keep = np.r_[rng.choice(idx[:-1], self.max_frames - 1, replace=False), idx[-1]]
            seen = np.zeros(HIST, bool)
            seen[keep] = True
            idx = np.nonzero(seen)[0]
        d = {"img_step": steps, "hand": hand, "hand_cmd": hand_cmd, "mean": np.zeros((n, 1, 1), np.float32)}
        h = history(d, j, R_P, 1.0, seen)
        slots = idx[::-1]                                          # newest first
        imgs = j - (HIST - 1 - slots)
        image = lambda t: np.asarray(Image.open(io.BytesIO(jpeg[off[t]:off[t + 1]].tobytes())).convert("RGB"))
        F = self.max_frames
        rgb, dep = np.zeros((F, 320, 320, 3), np.uint8), np.zeros((F, 320, 320), np.int32)
        for a, t in enumerate(imgs):
            rgb[a], dep[a] = image(t), depth[t]
        slot, ok = np.zeros(F, np.int64), np.zeros(F, bool)
        slot[:len(slots)], ok[:len(slots)] = slots, True
        t, jg = j + m, int(imgs[0])
        s_now, s_tgt, s_g = int(steps[j]), int(steps[t]), int(steps[jg])
        res_ok = (m == WM_STRIDE_IMAGES) and (s_tgt - s_now == 8) and len(hand_cmd) >= s_tgt
        res = residual_feature(hand, hand_cmd, [s_now + 4, s_tgt], hand[s_now].astype(np.float64), R_P) if res_ok else np.zeros(18, np.float32)
        T = torch.from_numpy
        return {"frames_rgb": T(rgb), "frames_depth": T(dep), "frame_slot": T(slot), "frame_ok": T(ok),
                "seen": T(h["seen"]), "avail": T(h["avail"]), "proprio": T(h["proprio"]), "dt": T(h["dt"]),
                "path": T(commanded_path(d, s_now, s_tgt, R_P)), "rel_t": torch.tensor((s_tgt - s_now) / 20.0, dtype=torch.float32),
                "tgt_rgb": T(image(t)), "tgt_depth": T(depth[t].astype(np.int32)),
                "h_star": torch.tensor(meta["H_star"], dtype=torch.float32), "R_P": T(R_P.astype(np.float32)),
                "g_hand": T(hand[s_g].astype(np.float32)), "g_cam": T(cam[jg].astype(np.float32)), "g_inserted": torch.tensor(bool(inserted(rel[s_g]))),
                "t_hand": T(hand[s_tgt].astype(np.float32)), "t_cam": T(cam[t].astype(np.float32)), "t_inserted": torch.tensor(bool(inserted(rel[s_tgt]))),
                "res": T(res), "res_ok": torch.tensor(bool(res_ok))}
