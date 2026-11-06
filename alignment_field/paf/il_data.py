"""Training data for the IL insertion policies (paf.expert_demo format).

Frames: RGB crop, 7 joint angles; labels: chunk of joint targets relative to
the current joint angles (radians), divided by ACT_SCALE.
Memory: the container has a 62 GB limit, so the frames are counted first and
written into preallocated arrays (no list + concatenate copy); frame_stride
keeps every k-th recorded frame (recorded frames are 2 control ticks apart).
"""

import glob
import json

import numpy as np

from .fm_policy import ACT_SCALE


def load_demos(pattern, ports, frame_stride=1):
    files = []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if meta.get("port") in ports and "act" in d.files:
            files.append((f, len(d["act"][::frame_stride])))
    if not files:
        raise ValueError(f"no demonstrations for {ports}")
    n = sum(m for _, m in files)
    d0 = np.load(files[0][0])
    rgb = np.empty((n, *d0["rgb"].shape[1:]), np.uint8)
    prop = np.empty((n, 7), np.float32)
    act = np.empty((n, *d0["act"].shape[1:]), np.float32)
    grasp = np.empty(n, np.int64)
    k = 0
    for gi, (f, m) in enumerate(files):
        d = np.load(f, allow_pickle=False)
        rgb[k:k + m] = d["rgb"][::frame_stride]
        prop[k:k + m] = d["qpos"][::frame_stride]
        act[k:k + m] = d["act"][::frame_stride] / ACT_SCALE
        grasp[k:k + m] = gi
        k += m
    return rgb, prop, act, grasp
