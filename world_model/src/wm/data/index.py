"""Dataset index: one entry per trajectory with its split and port axes.

Port axes (board-plane RANSAC + gravity) are estimated once per trajectory from its
first image, a real-robot quantity; they are stored in the index so training does
not repeat the fit.

    python world_model/scripts/build_index.py $Q/data/wm_v1      -> $Q/data/wm_v1/index.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from . import store
from .splits import split_of


def _entry(path: str) -> dict:
    from ..models.perception import port_axes
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        n_img = int(f["img_step"].shape[0]) if "img_step" in f.files else 0
        R = None
        if n_img:
            off = f["rgb_offsets"]
            first = store._decode_jpeg(f["rgb_jpeg"][: off[1]], off[:2])[0]
            R = port_axes(first, f["depth"][0], f["K"], f["cam_T_world"][0])
    return {"path": path, "port": meta["port"], "grasp": meta["grasp"], "condition": meta["condition"],
            "kind": meta["kind"], "outcome": meta.get("outcome"), "n_images": n_img,
            "split": split_of(meta["port"], meta["condition"]),
            "R_P": None if R is None else np.asarray(R).tolist()}


def build(root: str | Path, workers: int = 16) -> list[dict]:
    paths = [str(p) for p in sorted(Path(root).rglob("t*.npz")) if not p.name.endswith(".tmp.npz")]
    with mp.get_context("spawn").Pool(workers) as pool:
        entries = pool.map(_entry, paths, chunksize=8)
    (Path(root) / "index.json").write_text(json.dumps(entries))
    return entries


def load(root: str | Path) -> list[dict]:
    return json.loads((Path(root) / "index.json").read_text())
