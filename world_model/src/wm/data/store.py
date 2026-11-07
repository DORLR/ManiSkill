"""Trajectory files: write (atomic, compressed) and read.

RGB is stored as JPEG (quality 95, no chroma subsampling) to fit the disk budget;
depth stays lossless (uint16, zlib inside the .npz).
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from . import schema


def _encode_jpeg(images: np.ndarray, quality: int) -> tuple[np.ndarray, np.ndarray]:
    blobs = []
    for im in images:
        buf = io.BytesIO()
        Image.fromarray(im).save(buf, format="JPEG", quality=quality, subsampling=0)
        blobs.append(np.frombuffer(buf.getvalue(), np.uint8))
    offsets = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    data = np.concatenate(blobs) if blobs else np.zeros(0, np.uint8)
    return data, offsets


def _decode_jpeg(data: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(Image.open(io.BytesIO(data[a:b].tobytes())).convert("RGB"))
                     for a, b in zip(offsets[:-1], offsets[1:])])


def save(path: str | Path, arrays: dict, meta: dict, rgb_codec: str = "jpeg", quality: int = 95) -> int:
    """Write one trajectory; returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(arrays)
    meta = {"schema": schema.SCHEMA_VERSION, "rgb_codec": rgb_codec if "rgb" in arrays else None, **meta}
    if "rgb" in arrays and rgb_codec == "jpeg":
        arrays["rgb_jpeg"], arrays["rgb_offsets"] = _encode_jpeg(arrays.pop("rgb"), quality)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta)), **arrays)
    os.replace(tmp, path)
    return path.stat().st_size


def read_meta(path: str | Path) -> dict:
    with np.load(path) as f:
        return json.loads(str(f["meta"]))


def load(path: str | Path, decode_rgb: bool = True) -> tuple[dict, dict]:
    with np.load(path) as f:
        meta = json.loads(str(f["meta"]))
        arrays = {k: f[k] for k in f.files if k != "meta"}
    if decode_rgb and "rgb_jpeg" in arrays:
        arrays["rgb"] = _decode_jpeg(arrays.pop("rgb_jpeg"), arrays.pop("rgb_offsets"))
    return arrays, meta


def index(root: str | Path) -> list[dict]:
    """Metadata of every trajectory under root (reads only the meta entry)."""
    return [{"path": str(p), **read_meta(p)} for p in sorted(Path(root).rglob("*.npz"))
            if not p.name.endswith(".tmp.npz")]
