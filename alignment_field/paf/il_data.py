"""Training data for the IL insertion policies (paf.expert_demo format).

Frames: RGB crop, 7 joint angles; labels: chunk of joint targets relative to
the current joint angles (radians), divided by ACT_SCALE.
"""

import glob
import json

import numpy as np

from .fm_policy import ACT_SCALE


def load_demos(pattern, ports):
    rgb, prop, act, grasp = [], [], [], []
    for gi, f in enumerate(sorted(glob.glob(pattern))):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if meta.get("port") not in ports or "act" not in d.files:
            continue
        rgb.append(d["rgb"])
        prop.append(d["qpos"].astype(np.float32))
        act.append(d["act"].astype(np.float32) / ACT_SCALE)
        grasp.append(np.full(len(d["act"]), gi))
    return np.concatenate(rgb), np.concatenate(prop), np.concatenate(act), np.concatenate(grasp)
