"""Frozen crop autoencoder (v4) -> latent of every recorded image, saved with the trajectory's robot data.

Writes <out>/<relative trajectory path>.npz with
    mean        (N, 64, C) float16   latent of each image
    img_step    (N,)                 control step of each image
    qpos (T+1, 9), action (T, 7), hand (T+1, 4, 4), hand_cmd (T, 4, 4)   robot data
    cam_T_world (N, 4, 4), K (3, 3), plug_uv0 / hole_uv0 (N, 2)          to read a pose from a latent
    inserted    (N,) bool            success label at each image (data/outcome.py; simulator truth,
                                     label only -- on the real robot the connector's contact switch)
    R_P         (3, 3)               port axes (board-plane fit of the first image)
and <out>/stats.json with the latent scale (1 / RMS over the train split).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data import store
from ..data.datasets import crops_path, port_axes_ok
from ..data.outcome import inserted
from ..models.crop_ae import CropAE


class _Trajectories(Dataset):
    def __init__(self, entries, data_root, ee_root, crops_root, out):
        self.items = []
        for e in entries:
            if not (e["n_images"] and port_axes_ok(e)):
                continue
            rel = Path(e["path"]).relative_to(data_root)
            cp = crops_path(crops_root, data_root, e["path"])
            if cp.exists():
                self.items.append((e, out / rel, ee_root / rel, cp))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, dst, ee, cp = self.items[i]
        if dst.exists():
            return {"dst": str(dst), "cached": True, "split": e["split"]}
        arrays, _ = store.load(e["path"])
        steps = arrays["img_step"]
        with np.load(ee) as f:
            hand_cmd = f["hand_cmd"]
        with np.load(cp) as f:
            plug_uv0, hole_uv0 = f["plug_uv0"], f["hole_uv0"]
        return {"dst": str(dst), "cached": False, "split": e["split"], "R_P": np.asarray(e["R_P"], np.float32),
                "rgb": arrays["rgb"], "depth": arrays["depth"].astype(np.int32), "K": arrays["K"],
                "cam_T_world": arrays["cam_T_world"], "hand_img": arrays["hand"][steps], "img_step": steps,
                "qpos": arrays["qpos"], "action": arrays["action"], "hand": arrays["hand"], "hand_cmd": hand_cmd,
                "plug_uv0": plug_uv0, "hole_uv0": hole_uv0, "inserted": inserted(arrays["eval_rel"][steps])}


@torch.no_grad()
def encode_frames(model: CropAE, t: dict, device, batch: int) -> np.ndarray:
    """All images of one trajectory -> (N, 64, C) float16 latents."""
    n = len(t["img_step"])
    out = []
    for a in range(0, n, batch):
        sl = slice(a, min(n, a + batch))
        m = sl.stop - sl.start
        b = {"rgb": torch.from_numpy(t["rgb"][sl]), "depth": torch.from_numpy(t["depth"][sl]),
             "plug_uv0": torch.from_numpy(t["plug_uv0"][sl].astype(np.int64)),
             "hole_uv0": torch.from_numpy(t["hole_uv0"][sl].astype(np.int64)),
             "K": torch.from_numpy(np.asarray(t["K"])).float().expand(m, 3, 3),
             "cam_T_world": torch.from_numpy(t["cam_T_world"][sl]).float(),
             "hand": torch.from_numpy(t["hand_img"][sl]).float(), "R_P": torch.from_numpy(t["R_P"]).expand(m, 3, 3)}
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, _ = model.encode(b)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out).astype(np.float16)


def save_latent_file(dst: Path, mean, t: dict):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, mean=mean, img_step=t["img_step"], qpos=t["qpos"], action=t["action"],
             hand=np.asarray(t["hand"], np.float32), hand_cmd=np.asarray(t["hand_cmd"], np.float32),
             cam_T_world=np.asarray(t["cam_T_world"], np.float32), K=np.asarray(t["K"], np.float32),
             plug_uv0=np.asarray(t["plug_uv0"], np.int16), hole_uv0=np.asarray(t["hole_uv0"], np.int16),
             inserted=np.asarray(t["inserted"], bool), R_P=np.asarray(t["R_P"], np.float32))
    tmp.rename(dst)


def run(model: CropAE, entries, data_root: Path, ee_root: Path, crops_root: Path, out: Path, device,
        workers: int = 12, batch: int = 64) -> float:
    model.eval().to(device)
    loader = DataLoader(_Trajectories(entries, data_root, ee_root, crops_root, out), batch_size=None, shuffle=False,
                        num_workers=workers, prefetch_factor=2 if workers else None, collate_fn=lambda s: s)
    sq, cnt = 0.0, 0
    for k, t in enumerate(loader):
        dst = Path(t["dst"])
        if t["cached"]:
            mean = np.load(dst)["mean"]
        else:
            mean = encode_frames(model, t, device, batch)
            save_latent_file(dst, mean, t)
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
