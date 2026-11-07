"""Frozen VAE -> latent mean of every recorded image, saved next to the trajectory's
control data. Writes <out>/<relative trajectory path>.npz and <out>/stats.json with the
latent scale (1 / std over the train split, as Stable Diffusion's 0.18215)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..data import store
from ..models.vae import TokenVAE


@torch.no_grad()
def encode_trajectory(model: TokenVAE, entry: dict, device, batch: int = 32) -> dict:
    arrays, _ = store.load(entry["path"])
    steps = arrays["img_step"]
    n = len(steps)
    R_P = torch.tensor(entry["R_P"], dtype=torch.float32, device=device)
    means = []
    for a in range(0, n, batch):
        sl = slice(a, min(n, a + batch))
        b = {"rgb": torch.from_numpy(arrays["rgb"][sl]), "depth": torch.from_numpy(arrays["depth"][sl].astype(np.int32)),
             "K": torch.from_numpy(arrays["K"]).expand(sl.stop - sl.start, 3, 3),
             "cam_T_world": torch.from_numpy(arrays["cam_T_world"][sl]),
             "hand": torch.from_numpy(arrays["hand"][steps[sl]]), "qpos": torch.from_numpy(arrays["qpos"][steps[sl]])}
        b = {k: v.to(device) for k, v in b.items()}
        b["R_P"] = R_P.expand(sl.stop - sl.start, 3, 3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mean, _, _ = model.encode(b)
        means.append(mean.float().cpu().numpy())
    return {"mean": np.concatenate(means).astype(np.float16), "img_step": steps,
            "qpos": arrays["qpos"], "action": arrays["action"], "hand": arrays["hand"][steps]}


def run(model: TokenVAE, entries: list[dict], data_root: Path, out: Path, device) -> float:
    model.eval().to(device)
    sq, cnt = 0.0, 0
    for e in entries:
        if not e["n_images"] or e["R_P"] is None:
            continue
        dst = out / Path(e["path"]).relative_to(data_root)
        if dst.exists():
            d = dict(np.load(dst))
        else:
            d = encode_trajectory(model, e, device)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **d)
        if e["split"] == "train":
            m = d["mean"].astype(np.float64)
            sq, cnt = sq + float((m ** 2).sum()), cnt + m.size
    scale = 1.0 / max(np.sqrt(sq / max(cnt, 1)), 1e-6)
    (out / "stats.json").write_text(json.dumps({"scale": scale, "train_values": cnt}))
    return scale
