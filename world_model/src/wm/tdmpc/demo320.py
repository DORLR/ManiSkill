"""Data logic of the 320 px policy: the fast copy of the demonstrations, the split, the policy's samples.
numpy + PIL + the standard library only (no torch, no OpenCV), so that all of it runs in a test without a GPU.

Fast copy ("wm-fast-1"), written once by scripts/data/cache_wm_v1_fast.py: one uncompressed .npz per trajectory with
the stored JPEG pictures byte for byte, every depth picture as a lossless 16-bit PNG, the joints, hand poses, joint
targets and the meta. Why: the policy is trained on independent samples (as the old policy was), one file read per
sample; the original files inflate the whole depth array (18-40 MB) at every read. The simulator's truth (`eval_rel`)
is not copied, so nothing that trains from the fast copy can read it; trajectories of the test ports A-D are not
copied at all.

Split (decided 2026-11-17): held out = grasps with pitch in (-42, -38) deg or position in (-21, -19) mm, and the 13
held-out shapes; ports A-D = final test; of the remaining grasps one in twenty (crc32 of the grasp's name, the rule
of the previous model's "dev" split) is the SELECTION split: not trained on, used to choose checkpoints, so that
nothing is chosen on the held-out set.
"""

from __future__ import annotations

import io
import json
import os
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from ..data.splits import OOD_VAL_SHAPES, TEST_PORTS

IMAGE_EVERY, JOINTS = 4, 7
CHUNK = 16                           # joint targets per policy output (0.8 s at 20 Hz); 8 are executed
CONTEXT, STRIDE_IMG = 4, 2           # the policy's pictures: 4, every second stored picture (0.4 s apart)
POLICY_KINDS = ("reference", "insert", "perturbed")       # successful expert trajectories, as the old policy
SELECT_ONE_IN = 20
FORMAT = "wm-fast-1"
SUCCESS = {"depth_left_mm": 1.25, "angle_deg": 4.0, "lateral_mm": 2.0}      # the one success criterion (truth, at the end)


def is_success(truth: dict) -> bool:
    """The success criterion on the simulator's final state: every quantity strictly below its bound."""
    return all(float(truth[k]) < v for k, v in SUCCESS.items())


def split320(port: str, pitch_deg: float, pos_m: float, grasp: str) -> str:
    if port in TEST_PORTS:
        return "test"
    if port in OOD_VAL_SHAPES or -42.0 < pitch_deg < -38.0 or -21.0 < pos_m * 1000 < -19.0:
        return "heldout"
    return "select" if zlib.crc32(grasp.encode()) % SELECT_ONE_IN == 0 else "train"


def group320(port: str, pitch_deg: float, pos_m: float, grasp: str) -> str:
    """Reporting group of a grasp: unseen_shape, interior_strip, test_port, select, train_outer, train_band."""
    s, x = split320(port, pitch_deg, pos_m, grasp), pos_m * 1000
    if s == "test":
        return "test_port"
    if s == "heldout":
        return "unseen_shape" if port in OOD_VAL_SHAPES else "interior_strip"
    if s == "select":
        return "select"
    return "train_outer" if pitch_deg < -55 or pitch_deg > -25 or x < -28 or x > -12 else "train_band"


# ---------------------------------------------------------------- pictures
def decode_rgb(blob: np.ndarray, off: np.ndarray, i: int) -> np.ndarray:
    """Picture i of a stored JPEG blob -> (320, 320, 3) uint8, decoded as wm/data/store.py does (PIL, RGB)."""
    return np.asarray(Image.open(io.BytesIO(blob[off[i]:off[i + 1]].tobytes())).convert("RGB"))


def encode_depth(d: np.ndarray) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(d, np.uint16)).save(buf, format="PNG")
    return np.frombuffer(buf.getvalue(), np.uint8)


def decode_depth(blob: np.ndarray, off: np.ndarray, i: int) -> np.ndarray:
    """Depth picture i -> (320, 320) uint16 in 0.1 mm, exactly as recorded."""
    return np.asarray(Image.open(io.BytesIO(blob[off[i]:off[i + 1]].tobytes()))).astype(np.uint16)


def jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """A rendered picture through the store's own codec (quality 95, no chroma subsampling), as every training picture."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=95, subsampling=0)
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))


# ---------------------------------------------------------------- conversion
def convert(src: str | Path, dst: str | Path, pin_length_m: float, rel: str) -> dict | None:
    """One trajectory of wm_v1 -> its fast copy; returns the manifest entry, None if it has no pictures.
    Every depth picture is decoded again and compared with the source before the file is written."""
    with np.load(src) as z:
        meta = json.loads(str(z["meta"]))
        if "img_step" not in z.files or len(z["img_step"]) == 0:
            return None
        depth, img = z["depth"], z["img_step"]
        keep = {k: z[k] for k in ("rgb_jpeg", "rgb_offsets", "img_step", "qpos", "hand", "action")}
    if not len(depth) == len(img) == len(keep["rgb_offsets"]) - 1:
        raise ValueError(f"{src}: {len(depth)} depth pictures, {len(img)} image steps, {len(keep['rgb_offsets']) - 1} colour pictures")
    blobs = [encode_depth(d) for d in depth]
    blob, off = np.concatenate(blobs), np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    for i, d in enumerate(depth):
        if not np.array_equal(decode_depth(blob, off, i), d):
            raise ValueError(f"{src}: depth picture {i} does not survive the PNG round trip")
    meta.update(pin_length_m=float(pin_length_m), format=FORMAT, source=rel)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, meta=np.asarray(json.dumps(meta)), depth_png=blob, depth_offsets=off, **keep)
    os.replace(tmp, dst)
    return entry_of(meta, rel, len(img))


def entry_of(meta: dict, rel: str, n_images: int) -> dict:
    return {"path": rel, "n_images": int(n_images), "steps": int(meta["steps"]),
            **{k: meta[k] for k in ("port", "grasp", "condition", "kind", "outcome", "pitch_deg", "grasp_pos_m", "pin_length_m")}}


def load_entries(root: str | Path) -> list[dict]:
    """The fast copy's trajectories with their split and `bc` (the policy imitates it: a successful expert trajectory)."""
    out = []
    for e in json.loads((Path(root) / "manifest.json").read_text())["entries"]:
        out.append({**e, "split": split320(e["port"], e["pitch_deg"], e["grasp_pos_m"], e["grasp"]),
                    "bc": e["kind"] in POLICY_KINDS and e.get("outcome") == "inserted"})
    return out


# ---------------------------------------------------------------- the policy's samples (as the old policy's data set)
def context_indices(j: int) -> list[int]:
    """Pictures of a policy sample at picture j: [j-6, j-4, j-2, j], the first picture repeated at the start."""
    return [max(0, j - STRIDE_IMG * c) for c in range(CONTEXT - 1, -1, -1)]


def policy_chunk(qpos: np.ndarray, action: np.ndarray, s0: int) -> np.ndarray:
    """The next 16 commanded targets relative to the measured joints at s0 (held at the end)."""
    act = action[s0:s0 + CHUNK]
    if len(act) == 0:
        act = qpos[s0:s0 + 1, :JOINTS]
    if len(act) < CHUNK:
        act = np.concatenate([act, np.repeat(act[-1:], CHUNK - len(act), 0)])
    return (act - qpos[s0][:JOINTS]).reshape(-1).astype(np.float32)


def policy_items(entries: list[dict]) -> np.ndarray:
    """(entry index, picture index) of every picture of every trajectory in `entries`, the terminal picture included."""
    return np.array([(i, j) for i, e in enumerate(entries) for j in range(e["n_images"])], np.int64).reshape(-1, 2)


def policy_sample(path: str | Path, j: int) -> dict:
    """rgb (4, 320, 320, 3) uint8, depth (4, 320, 320) int16 [0.1 mm], q (7,) the joints measured at picture j,
    a (112,) the next 16 joint targets minus q."""
    with np.load(path) as z:
        img, qpos, act = z["img_step"], z["qpos"], z["action"]
        jb, jo, pb, po = z["rgb_jpeg"], z["rgb_offsets"], z["depth_png"], z["depth_offsets"]
    idx = context_indices(j)
    rgb = {i: decode_rgb(jb, jo, i) for i in set(idx)}
    dep = {i: np.minimum(decode_depth(pb, po, i), 32767).astype(np.int16) for i in set(idx)}
    s0 = int(img[j])
    return {"rgb": np.stack([rgb[i] for i in idx]), "depth": np.stack([dep[i] for i in idx]),
            "q": qpos[s0, :JOINTS].astype(np.float32), "a": policy_chunk(qpos, act, s0)}


def action_stats(root: str | Path, entries: list[dict], n: int = 20000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and spread (+1e-4) of the action chunk per dimension over n samples drawn without replacement
    (the old policy's rule); reads joints and targets only."""
    items = policy_items(entries)
    pick = items[np.sort(np.random.default_rng(seed).choice(len(items), size=min(n, len(items)), replace=False))]
    a = []
    for i in np.unique(pick[:, 0]):
        with np.load(Path(root) / entries[i]["path"]) as z:
            img, qpos, act = z["img_step"], z["qpos"], z["action"]
        a += [policy_chunk(qpos, act, int(img[j])) for j in pick[pick[:, 0] == i, 1]]
    a = np.stack(a)
    return a.mean(0), a.std(0) + 1e-4
