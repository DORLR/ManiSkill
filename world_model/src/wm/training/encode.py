"""Frozen autoencoder -> latent of every recorded image, saved with the trajectory's robot data.

Writes <out>/<relative trajectory path>.npz with
    mean      (N, 64, C) float16  latent of each image (deterministic encoder; name kept from v1)
    img_step  (N,)                control step of each image
    qpos      (T+1, 9)            measured joints
    action    (T, 7)              commanded joint targets
    hand      (T+1, 4, 4)         measured hand pose (FK of qpos)
    hand_cmd  (T, 4, 4)           commanded hand pose (FK of the targets; data/kinematics.py)
and <out>/stats.json with the latent scale (1 / RMS over the train split, as Stable Diffusion's
0.18215). Trajectories are read and decoded by DataLoader workers; the GPU encodes in batches.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import store
from ..data.datasets import port_axes_ok
from ..models.vae import TokenVAE


class _Trajectories(Dataset):
    def __init__(self, entries: list[dict], data_root: Path, ee_root: Path, out: Path):
        self.items = [(e, out / Path(e["path"]).relative_to(data_root), ee_root / Path(e["path"]).relative_to(data_root))
                      for e in entries if e["n_images"] and port_axes_ok(e)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, dst, ee = self.items[i]
        if dst.exists():
            return {"dst": str(dst), "cached": True, "split": e["split"]}
        arrays, _ = store.load(e["path"])
        steps = arrays["img_step"]
        with np.load(ee) as f:
            hand_cmd = f["hand_cmd"]
        return {"dst": str(dst), "cached": False, "split": e["split"], "R_P": np.asarray(e["R_P"], np.float32),
                "rgb": arrays["rgb"], "depth": arrays["depth"].astype(np.int32), "K": arrays["K"],
                "cam_T_world": arrays["cam_T_world"], "hand_img": arrays["hand"][steps], "img_step": steps,
                "qpos": arrays["qpos"], "action": arrays["action"], "hand": arrays["hand"], "hand_cmd": hand_cmd}


def _keep(sample):
    """Keep numpy arrays as they are (no automatic tensor conversion)."""
    return sample


@torch.no_grad()
def _encode(model: TokenVAE, t: dict, device, batch: int) -> np.ndarray:
    n = len(t["img_step"])
    out = []
    for a in range(0, n, batch):
        sl = slice(a, min(n, a + batch))
        m = sl.stop - sl.start
        b = {"rgb": torch.from_numpy(t["rgb"][sl]), "depth": torch.from_numpy(t["depth"][sl]),
             "K": torch.from_numpy(t["K"]).float().expand(m, 3, 3), "cam_T_world": torch.from_numpy(t["cam_T_world"][sl]).float(),
             "hand": torch.from_numpy(t["hand_img"][sl]).float(), "R_P": torch.from_numpy(t["R_P"]).expand(m, 3, 3)}
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = model.encode(b)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def run(model: TokenVAE, entries: list[dict], data_root: Path, ee_root: Path, out: Path, device,
        workers: int = 12, batch: int = 64) -> float:
    model.eval().to(device)
    loader = DataLoader(_Trajectories(entries, data_root, ee_root, out), batch_size=None, shuffle=False,
                        num_workers=workers, prefetch_factor=2 if workers else None, collate_fn=_keep)
    sq, cnt = 0.0, 0
    for k, t in enumerate(loader):
        dst = Path(t["dst"])
        if t["cached"]:
            mean = np.load(dst)["mean"]
        else:
            mean = _encode(model, t, device, batch)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, mean=mean, img_step=t["img_step"], qpos=t["qpos"], action=t["action"],
                     hand=t["hand"].astype(np.float32), hand_cmd=t["hand_cmd"].astype(np.float32))
        if t["split"] == "train":
            m = mean.astype(np.float64)
            sq, cnt = sq + float((m ** 2).sum()), cnt + m.size
        if k % 500 == 0:
            print(f"encoded {k}/{len(loader)}", flush=True)
    if cnt == 0:
        raise RuntimeError("no train-split latents: cannot compute the latent scale")
    scale = 1.0 / max(np.sqrt(sq / cnt), 1e-6)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps({"scale": scale, "train_values": cnt}))
    return scale
