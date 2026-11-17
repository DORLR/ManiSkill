"""Checks of wm/tdmpc/demo320.py on a small sample of the real data (no GPU). Environment: WM_SAMPLE_ROOT = a folder
in wm_v1's layout (index.json, <port>/<grasp>/t*.npz, pin_length_by_port.json), WM_FIX_ROOT = where the fast copy of
the sample goes (data/wm_v1_fast_sample) and where data/grasps_table.json lies.

    python world_model/tests/test_demo320.py
"""

import io
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from wm.data.splits import OOD_VAL_SHAPES, TEST_PORTS
from wm.tdmpc import demo320 as D

SRC, FIX = Path(os.environ["WM_SAMPLE_ROOT"]), Path(os.environ["WM_FIX_ROOT"])
FAST = FIX / "data" / "wm_v1_fast_sample"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "data" / "cache_wm_v1_fast.py"


def old_context_indices(j):                       # verbatim: data/datasets.py of git 4a3a6d3 (WM_STRIDE 2, CONTEXT 4)
    return [max(0, j - 2 * c) for c in range(4 - 1, -1, -1)]


def old_policy_chunk(qpos, action, s0):           # verbatim: data/datasets.py of git 4a3a6d3 (POLICY_CHUNK 16)
    act = action[s0:s0 + 16]
    if len(act) == 0:
        act = qpos[s0:s0 + 1, :7]
    if len(act) < 16:
        act = np.concatenate([act, np.repeat(act[-1:], 16 - len(act), 0)])
    return (act - qpos[s0][:7]).reshape(-1).astype(np.float32)


def build():
    subprocess.run([sys.executable, str(SCRIPT), "--src", str(SRC), "--dst", str(FAST), "--pins", str(SRC / "pin_length_by_port.json"),
                    "--workers", "2"], check=True, stdout=subprocess.DEVNULL)


def test_conversion():
    build()
    man = json.loads((FAST / "manifest.json").read_text())
    index = json.loads((SRC / "index.json").read_text())
    want = [e for e in index if e["port"] not in TEST_PORTS and e["n_images"]]
    assert len(man["entries"]) == len(want) and man["skipped"]["test_ports"] == sum(e["port"] in TEST_PORTS for e in index)
    assert not any(e["port"] in TEST_PORTS for e in man["entries"]) and not list(FAST.glob("A/*/*.npz"))
    pictures = 0
    for e in man["entries"]:
        with np.load(SRC / e["path"]) as s, np.load(FAST / e["path"]) as f:
            assert "eval_rel" not in f.files and "seg" not in f.files
            assert np.array_equal(s["rgb_jpeg"], f["rgb_jpeg"]) and np.array_equal(s["rgb_offsets"], f["rgb_offsets"])
            for k in ("img_step", "qpos", "hand", "action"):
                assert np.array_equal(s[k], f[k]), k
            depth, pb, po = s["depth"], f["depth_png"], f["depth_offsets"]
            assert len(depth) == e["n_images"] == len(po) - 1
            for i in range(len(depth)):
                d = D.decode_depth(pb, po, i)
                assert d.dtype == np.uint16 and np.array_equal(d, depth[i])
            meta = json.loads(str(f["meta"]))
            assert meta["format"] == D.FORMAT and meta["pin_length_m"] == e["pin_length_m"] and e["steps"] == len(s["action"])
            pictures += len(depth)
    t = os.path.getmtime(FAST / man["entries"][0]["path"])
    build()                                       # a second run keeps the files
    assert os.path.getmtime(FAST / man["entries"][0]["path"]) == t
    return f"{len(man['entries'])} trajectories, {pictures} depth pictures exact, colour bytes identical, test port not copied"


def test_policy_samples():
    ents = [e for e in D.load_entries(FAST) if e["bc"]]
    items = D.policy_items(ents)
    assert len(items) == sum(e["n_images"] for e in ents)
    n = 0
    for i, e in enumerate(ents):
        with np.load(SRC / e["path"]) as s:
            img, qpos, act, depth, jb, jo = s["img_step"], s["qpos"], s["action"], s["depth"], s["rgb_jpeg"], s["rgb_offsets"]
        assert int(img[-1]) == len(act), "the terminal picture is at the last control step"
        for j in sorted({0, 1, 2, 5, len(img) // 2, len(img) - 2, len(img) - 1}):
            got, idx, s0 = D.policy_sample(FAST / e["path"], j), old_context_indices(j), int(img[j])
            assert D.context_indices(j) == idx
            assert np.array_equal(got["q"], qpos[s0, :7]) and np.array_equal(got["a"], old_policy_chunk(qpos, act, s0))
            assert np.array_equal(got["depth"], np.minimum(depth[idx], 32767).astype(np.int16))
            old_rgb = np.stack([np.asarray(Image.open(io.BytesIO(jb[jo[k]:jo[k + 1]].tobytes())).convert("RGB")) for k in idx])
            assert got["rgb"].dtype == np.uint8 and np.array_equal(got["rgb"], old_rgb)
            n += 1
        last = D.policy_sample(FAST / e["path"], len(img) - 1)
        assert not last["a"].any(), "terminal sample: hold"
    mean, std = D.action_stats(FAST, ents, n=500, seed=0)
    assert mean.shape == std.shape == (112,) and (std > 0).all()
    return f"{n} samples of {len(ents)} imitated trajectories equal the old data set's (pictures, joints, action chunk), terminal sample = hold"


def test_split():
    gs = [g for g in json.loads((FIX / "data" / "grasps_table.json").read_text()) if g["reference_ok"]]
    split = Counter(D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) for g in gs)
    group = Counter(D.group320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) for g in gs)
    assert split["train"] + split["select"] == 705 and split["heldout"] == 297, split
    for g in gs:
        s, p, x = D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]), g["pitch_deg"], g["grasp_pos_m"] * 1000
        inside = -42 < p < -38 or -21 < x < -19
        assert (s == "test") == (g["port"] in TEST_PORTS)
        if s in ("train", "select"):
            assert g["port"] not in OOD_VAL_SHAPES and not inside
        if s == "heldout":
            assert g["port"] in OOD_VAL_SHAPES or inside
    assert group["unseen_shape"] + group["interior_strip"] == 297 and group["select"] == split["select"]
    assert D.is_success({"depth_left_mm": 1.249, "angle_deg": 3.99, "lateral_mm": 1.99})
    assert not D.is_success({"depth_left_mm": 1.25, "angle_deg": 0, "lateral_mm": 0}) and not D.is_success({"depth_left_mm": 0, "angle_deg": 4.0, "lateral_mm": 0})
    return f"grasps with a reference: {dict(split)}; groups {dict(group)}"


if __name__ == "__main__":
    for t in (test_conversion, test_policy_samples, test_split):
        print(t.__name__, "OK:", t(), flush=True)
