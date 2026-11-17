"""The branch set as decision-level transitions.

One step = one decision of the policy (8 control steps, 0.4 s). For decision k of a branch with n decisions:
  observation  the pictures at image index 2k, 2k-1, 2k-2 (0.2 s apart) and the joints measured at those instants
  action       the 8 executed joint targets minus the joints measured at the decision (56 numbers), through
               asinh(x / median size) scaled to [-1, 1]: near the board the moves are about 2 % of the largest ones
               (measured), a plain division by the largest would leave them, and the differences between candidates,
               at the noise floor of the network's input
  reward       gain of insertion depth over the decision in mm, from the FK label `depth_fk_m` (pin length minus the
               distance of the hand to the reference pose H*; never the simulator's truth), plus BONUS at the last
               decision of a branch whose FK label says seated
  terminated   last decision of a branch that stopped for another reason than the time limit
The observation after the last decision is the picture at image index 2n (control step 8n, before the short hold).
The simulator's truth (`outcome`) is carried along for the reports only.
"""

from __future__ import annotations

import glob
import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

from wm.data.splits import split_of

EXECUTE, IMAGE_EVERY, JOINTS = 8, 4, 7
ACTION_DIM = EXECUTE * JOINTS
BONUS = 10.0
FACE_MARGIN_M = 0.0005
DEV_ONE_IN = 20
EVAL_OFFSETS = (-10, -5, -2, 0, 3)          # decisions relative to the one at which the pins reach the board face


def split_name(port: str, condition: str, grasp: str) -> str:
    """train / ood_val / test as wm.data.splits; one training grasp in 20 is set aside as `dev` (seen shapes and
    grasp bands, unseen grasps) to tell overfitting from the shift to unseen shapes."""
    s = split_of(port, condition)
    if s == "train" and zlib.crc32(grasp.encode()) % DEV_ONE_IN == 0:
        return "dev"
    return s


def list_branches(root: str | Path, split: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(f"{root}/*/*/grasp.json")):
        g = json.loads(Path(f).read_text())
        if g.get("error") or split_name(g["port"], g["condition"], g["grasp"]) != split:
            continue
        for s in g["starts"]:
            for b in s.get("branches", []):
                p = Path(f).parent / f"s{s['idx']:02d}_b{b['idx']}.npz"
                if p.exists() and b["decisions"] >= 1:
                    out.append({"path": str(p), "rel": f"{g['port']}/{g['grasp']}/s{s['idx']:02d}_b{b['idx']}",
                                "grasp": g["grasp"], "port": g["port"], "start": s["idx"], "branch": b["idx"],
                                "sigma_m": b["noise"]["sigma_m"], "decisions": b["decisions"], "stopped": b["stopped"],
                                "seated_fk": bool(b["seated_fk"]), "inserted": b["outcome"] == "inserted",
                                "outcome": b["outcome"], "depth_left_mm": b.get("depth_left_mm"), "lateral_mm": b.get("lateral_mm")})
    return out


def decision_arrays(z) -> dict | None:
    """From an opened branch file: everything but the pictures. None if the file does not have the regular layout."""
    dec, img_step, act = z["dec_step"], z["img_step"], z["action"]
    n, steps = len(dec), len(act)
    if (n < 1 or steps < EXECUTE * n or len(img_step) < 2 * n + 1 or not np.array_equal(dec, EXECUTE * np.arange(n))
            or not np.array_equal(img_step[:2 * n + 1], IMAGE_EVERY * np.arange(2 * n + 1))):
        return None
    meta = json.loads(str(z["meta"]))
    qpos, d = z["qpos"][:, :JOINTS], z["depth_fk_m"]
    delta = act[:EXECUTE * n].reshape(n, EXECUTE, JOINTS) - qpos[dec][:, None, :]
    edge = np.append(dec, steps)                                   # the last decision's result includes the hold
    reward = (d[edge[1:]] - d[edge[:-1]]) * 1000.0
    if meta["seated_fk"]:
        reward[-1] += BONUS
    terminated = np.zeros(n, np.float32)
    terminated[-1] = float(meta["stopped"] != "time")
    H, axis, L = np.asarray(meta["H_star"]), np.asarray(meta["axis"]), meta["pin_length_m"]
    along = (z["hand"][dec, :3, 3] - H[:3, 3]) @ axis
    at = np.nonzero(along >= -(L + FACE_MARGIN_M))[0]
    return {"n": n, "delta": delta.reshape(n, ACTION_DIM).astype(np.float32), "reward": reward.astype(np.float32),
            "terminated": terminated, "q_img": qpos[img_step[:2 * n + 1]].astype(np.float32),
            "face_k": int(at[0]) if len(at) else -1, "meta": meta}


def returns(reward: np.ndarray, gamma: float) -> np.ndarray:
    """Discounted return from each decision on; one more entry (0) for the state after the last decision."""
    g = np.zeros(len(reward) + 1, np.float32)
    for k in range(len(reward) - 1, -1, -1):
        g[k] = reward[k] + gamma * g[k + 1]
    return g


def compute_stats(branches: list[dict], n: int = 600, seed: int = 0) -> dict:
    """Scales from a sample of training branches: per action dimension the median and the 99.9th percentile of the
    magnitude; the joints' mean and spread; the spread of the joints' change between two pictures."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(branches), min(n, len(branches)), replace=False)
    delta, q, dq = [], [], []
    for i in pick:
        with np.load(branches[i]["path"]) as z:
            a = decision_arrays(z)
        if a is None:
            continue
        delta.append(a["delta"])
        q.append(a["q_img"])
        dq.append(np.diff(a["q_img"], axis=0))
    delta, q, dq = np.concatenate(delta), np.concatenate(q), np.concatenate(dq)
    s0 = np.maximum(np.percentile(np.abs(delta), 50, axis=0), 1e-5)
    return {"action_s0": s0.tolist(), "action_max": np.maximum(np.percentile(np.abs(delta), 99.9, axis=0), 2 * s0).tolist(),
            "q_mean": q.mean(0).tolist(), "q_std": np.maximum(q.std(0), 1e-3).tolist(),
            "dq_std": np.maximum(dq.std(0), 1e-5).tolist(), "branches_used": len(pick), "decisions_used": len(delta)}


def scale_action(delta: np.ndarray, stats: dict) -> np.ndarray:
    s0, top = np.asarray(stats["action_s0"], np.float64), np.asarray(stats["action_max"], np.float64)
    return np.clip(np.arcsinh(delta / s0) / np.arcsinh(top / s0), -1.0, 1.0).astype(np.float32)


def decode_rgb(blob: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(blob[offsets[i]:offsets[i + 1]], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def eval_points(n: int, face_k: int) -> list[tuple[str, int]]:
    """(tag, decision) at which a branch is scored in the reports."""
    pts = [("start", 0)]
    if face_k >= 0:
        pts += [(f"face{o:+d}", face_k + o) for o in EVAL_OFFSETS if 0 < face_k + o < n]
    return pts


class _Pictures:
    """The pictures of one opened branch file, decoded on demand (the depth array is unpacked once)."""

    def __init__(self, z):
        self.blob, self.off, self.depth = z["rgb_jpeg"], z["rgb_offsets"], z["depth"]

    def get(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        memo = {int(i): decode_rgb(self.blob, self.off, int(i)) for i in np.unique(idx)}
        return np.stack([memo[int(i)] for i in idx]), np.minimum(self.depth[idx], 32767).astype(np.int16)


class Windows(torch.utils.data.Dataset):
    """Training windows of `horizon` decisions with the full pictures. One item = `per_file` windows of one branch
    file (reading a file costs about half a second, mostly unpacking its depth pictures).
    Per window: rgb (F, 640, 640, 3) uint8 and depth (F, 640, 640) int16 [0.1 mm] for F = 2 horizon + 3 consecutive
    pictures (the observation of step j uses pictures 2j+2, 2j+1, 2j of the window), q (F, 7), action (horizon, 56),
    reward (horizon,), ret (horizon + 1,), terminated (horizon,), head_mask (heads,)."""

    def __init__(self, branches, stats, horizon, gamma, per_file, length, seed, heads=5, keep=0.8):
        self.b = [x for x in branches if x["decisions"] >= horizon]
        w = np.array([x["decisions"] - horizon + 1 for x in self.b], np.float64)
        self.p = w / w.sum()
        self.stats, self.h, self.gamma, self.per_file, self.length, self.seed = stats, horizon, gamma, per_file, length, seed
        self.heads, self.keep = heads, keep

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        rng = np.random.default_rng((self.seed, i))
        for _ in range(20):
            bi = int(rng.choice(len(self.b), p=self.p))
            item = self._read(bi, rng)
            if item is not None:
                return item
        raise RuntimeError("20 unreadable branch files in a row")

    def _read(self, bi, rng):
        h = self.h
        with np.load(self.b[bi]["path"]) as z:
            a = decision_arrays(z)
            if a is None or a["n"] < h:
                return None
            pics = _Pictures(z)
        ks = rng.integers(0, a["n"] - h + 1, self.per_file)
        act, ret = scale_action(a["delta"], self.stats), returns(a["reward"], self.gamma)
        mask = (np.random.default_rng(zlib.crc32(self.b[bi]["rel"].encode())).random(self.heads) < self.keep)
        out = {k: [] for k in ("rgb", "depth", "q", "action", "reward", "ret", "terminated")}
        for k in ks:
            idx = np.maximum(np.arange(2 * k - 2, 2 * (k + h) + 1), 0)
            rgb, depth = pics.get(idx)
            out["rgb"].append(rgb)
            out["depth"].append(depth)
            out["q"].append(a["q_img"][idx])
            out["action"].append(act[k:k + h])
            out["reward"].append(a["reward"][k:k + h])
            out["ret"].append(ret[k:k + h + 1])
            out["terminated"].append(a["terminated"][k:k + h])
        item = {k: torch.from_numpy(np.stack(v)) for k, v in out.items()}
        item["head_mask"] = torch.from_numpy(np.repeat(mask[None].astype(np.float32), len(ks), 0))
        return item


def collate_cat(items: list[dict]) -> dict:
    return {k: torch.cat([it[k] for it in items]) for k in items[0]}


class EvalPoints(torch.utils.data.Dataset):
    """For the reports: one item = the scored decisions of one branch with the full pictures.
    rgb (P, 3, 640, 640, 3), depth (P, 3, 640, 640), q (P, 3, 7) for the pictures [t-2, t-1, t]; action (P, hmax, 56)
    the executed actions from the decision on (zeros past the end), valid (P,) how many of them exist."""

    def __init__(self, branches, stats, gamma, hmax):
        self.b, self.stats, self.gamma, self.hmax = branches, stats, gamma, hmax

    def __len__(self):
        return len(self.b)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        with np.load(self.b[i]["path"]) as z:
            a = decision_arrays(z)
            if a is None:
                return None
            pics = _Pictures(z)
        return _eval_item(a, pics, self.stats, self.gamma, self.hmax, i)


def point_arrays(a: dict, stats: dict, gamma: float, hmax: int, branch_index: int) -> dict:
    """The non-picture part of a branch's scored decisions (shared by both models' reports)."""
    pts = eval_points(a["n"], a["face_k"])
    act, ret = scale_action(a["delta"], stats), returns(a["reward"], gamma)
    action = np.zeros((len(pts), hmax, ACTION_DIM), np.float32)
    for j, (_, k) in enumerate(pts):
        m = min(hmax, a["n"] - k)
        action[j, :m] = act[k:k + m]
    ks = np.array([k for _, k in pts])
    return {"action": torch.from_numpy(action), "valid": torch.from_numpy(np.minimum(hmax, a["n"] - ks)),
            "ret": torch.from_numpy(ret[ks]), "k": torch.from_numpy(ks),
            "branch": torch.full((len(pts),), branch_index, dtype=torch.long), "tags": [t for t, _ in pts]}


def _eval_item(a, pics, stats, gamma, hmax, i):
    item = point_arrays(a, stats, gamma, hmax, i)
    rgb, depth, q = [], [], []
    for k in item["k"].tolist():
        idx = np.maximum(np.arange(2 * k - 2, 2 * k + 1), 0)
        r, d = pics.get(idx)
        rgb.append(r)
        depth.append(d)
        q.append(a["q_img"][idx])
    item.update(rgb=torch.from_numpy(np.stack(rgb)), depth=torch.from_numpy(np.stack(depth)), q=torch.from_numpy(np.stack(q)))
    return item


def collate_points(items: list[dict | None]) -> dict:
    items = [it for it in items if it is not None]
    out = {k: torch.cat([it[k] for it in items]) for k in items[0] if k != "tags"}
    out["tags"] = [t for it in items for t in it["tags"]]
    return out
