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
    res / res_ok  for a +0.4 s target: how the hand really moves from now to +0.2 s and to +0.4 s (pose features relative
                  to the hand now, minus the feature of "did not move"; position in mm, rotation 6D x 10) -- the world
                  model's second output, replacing the joint adapter (decided with the user 2026-11-12).
                  First written as "measured minus the command that led there" (runs joint_v6): then "the hand stays" is
                  the answer -lead, and the model learned "the lag stays as it was" instead -- recorded commands replayed
                  gave 0.01 mm per held decision, but a script pushing from the imagined hand with another lead than the
                  state had seen moved it 0.37 mm per decision, 14.5 mm in 16 s (check_wm6_push.py, 10 000 steps).
                  Written as the hand's own motion, "the hand stays" is zero whatever the command.
    state / t_pressed   (decided with the user 2026-11-12) the contact state now, (pressed, inserted), given to the world
                  model as a condition, and "pressed" at the target time, its third extra output (next to "inserted").
                  Why: with the motion output alone (run joint_v6b, 5000 steps) the pushing script still moved the imagined
                  hand 11 mm in 16 s. The model has no state "I am pressed against something"; it guesses it anew at
                  every step from what commands and history look like, and once it has guessed "free" the history says
                  so too. Given as a condition and carried from each step's output to the next step's input, pressed is a
                  state that persists until something ends it. It is the present state, not the outcome ("inserted
                  within 8 s" as a condition would fix the answer whatever the actions). pressed: pressed_at above;
                  inserted: the contact switch (simulator truth in training).
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


STILL = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], np.float32)       # the pose feature of a hand that has not moved
PRESS_STEPS, PRESS_STILL_MM, PRESS_LEAD_MM = 8, 0.05, 0.1


def pressed_at(hand: np.ndarray, hand_cmd: np.ndarray, s: int, axis: np.ndarray) -> bool:
    """The contact state "pressed" at control step s, from proprioception only (also on a real robot): over the last
    0.4 s the hand did not move along the insertion axis (< 0.05 mm) although the last command was ahead of it
    (> 0.1 mm). Free motion, a command that just holds the hand where it is, and backing off are not pressed."""
    if s < PRESS_STEPS or s - 1 >= len(hand_cmd):
        return False
    moved = abs(float((hand[s, :3, 3] - hand[s - PRESS_STEPS, :3, 3]) @ axis)) * 1000
    lead = float((hand_cmd[s - 1, :3, 3] - hand[s, :3, 3]) @ axis) * 1000
    return moved < PRESS_STILL_MM and lead > PRESS_LEAD_MM


def motion_feature(hand: np.ndarray, steps, ref: np.ndarray, R_P: np.ndarray) -> np.ndarray:
    """The measured hand's motion from `ref` (the hand now) to each control step of `steps`: pose features relative
    to `ref` in port axes minus STILL, x RES_SCALE -> (len(steps) * 9,). All zeros = the hand did not move."""
    return ((pose_feature(hand[np.asarray(steps)].astype(np.float64), ref, R_P) - STILL) * RES_SCALE).reshape(-1).astype(np.float32)


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
        res = motion_feature(hand, [s_now + 4, s_tgt], hand[s_now].astype(np.float64), R_P) if res_ok else np.zeros(18, np.float32)
        hand64, axis = hand.astype(np.float64), R_P[:, 0]
        state = np.array([pressed_at(hand64, hand_cmd, s_now, axis), bool(inserted(rel[s_now]))], np.float32)
        T = torch.from_numpy
        return {"frames_rgb": T(rgb), "frames_depth": T(dep), "frame_slot": T(slot), "frame_ok": T(ok),
                "seen": T(h["seen"]), "avail": T(h["avail"]), "proprio": T(h["proprio"]), "dt": T(h["dt"]),
                "path": T(commanded_path(d, s_now, s_tgt, R_P)), "rel_t": torch.tensor((s_tgt - s_now) / 20.0, dtype=torch.float32),
                "tgt_rgb": T(image(t)), "tgt_depth": T(depth[t].astype(np.int32)),
                "h_star": torch.tensor(meta["H_star"], dtype=torch.float32), "R_P": T(R_P.astype(np.float32)),
                "g_hand": T(hand[s_g].astype(np.float32)), "g_cam": T(cam[jg].astype(np.float32)), "g_inserted": torch.tensor(bool(inserted(rel[s_g]))),
                "t_hand": T(hand[s_tgt].astype(np.float32)), "t_cam": T(cam[t].astype(np.float32)), "t_inserted": torch.tensor(bool(inserted(rel[s_tgt]))),
                "res": T(res), "res_ok": torch.tensor(bool(res_ok)), "state": T(state),
                "t_pressed": torch.tensor(bool(pressed_at(hand64, hand_cmd, s_tgt, axis)))}
