"""Training data for the IL insertion policies (labels only; never used at run time).

Demo actions were recorded in port axes; they are converted to the hand frame
with R_hand^T R_P, where R_hand is the hand orientation of that frame
reconstructed from the recorded state u, the pre-insertion pose H_pre and the
port axes R_P. H_pre and R_P come from the one insertion of the grasp (FK).
"""

import glob
import json

import numpy as np
from scipy.spatial.transform import Rotation

from . import config as C
from .fm_policy import ACT_SCALE, H_CHUNK


def load_demos(pattern, ports):
    """Demo frames (RGB crop, joint angles) with hand-frame action chunks."""
    rgb, prop, chunks, grasp = [], [], [], []
    for gi, f in enumerate(sorted(glob.glob(pattern))):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if meta.get("port") not in ports or "demo_qpos" not in d.files or len(d["demo_act"]) == 0:
            continue  # shards collected before joint angles were stored are not used for IL
        R_P, H_pre = d["R_P"], d["H_pre"]
        act, did, U = d["demo_act"], d["demo_id"], d["demo_u"].astype(np.float64)
        q = U * C.SIGMA
        R_hand = R_P[None] @ Rotation.from_rotvec(q[:, 3:]).as_matrix() @ R_P.T[None] @ H_pre[None, :3, :3]
        p_hand = H_pre[None, :3, 3] + q[:, :3] @ R_P.T
        M = np.transpose(R_hand, (0, 2, 1)) @ R_P[None]
        a_hand = np.concatenate([np.einsum("nij,nj->ni", M, act[:, :3]), np.einsum("nij,nj->ni", M, act[:, 3:])], 1)
        for k in np.unique(did):
            idx = np.nonzero(did == k)[0]
            a = a_hand[idx].astype(np.float32)
            pad = np.concatenate([a, np.zeros((H_CHUNK, 6), np.float32)])
            chunks.append(np.stack([pad[t:t + H_CHUNK] for t in range(len(idx))]))
            rgb.append(d["demo_rgb"][idx])
            prop.append(d["demo_qpos"][idx].astype(np.float32))
            grasp.append(np.full(len(idx), gi))
    return (np.concatenate(rgb), np.concatenate(prop), np.concatenate(chunks) / ACT_SCALE, np.concatenate(grasp))


