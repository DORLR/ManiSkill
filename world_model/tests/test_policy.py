"""Policy shapes, loss and sampling (CPU)."""

import numpy as np
import torch

from wm.data.datasets import POLICY_CHUNK, LatentPolicyDataset
from wm.models.policy import ACTION_DIM, CHUNK, LatentFMPolicy


def test_policy_loss_and_sample():
    torch.manual_seed(0)
    m = LatentFMPolicy(d=64, layers=1, heads=4, hidden=128)
    z = torch.randn(3, 4, 64, 8)
    a = torch.randn(3, CHUNK * ACTION_DIM)
    loss = m.loss(z, a)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.sample(z, steps=3, n=2).shape == (3, 2, CHUNK * ACTION_DIM)


def test_policy_dataset(tmp_path):
    root, lat = tmp_path / "data", tmp_path / "lat"
    (lat / "G00" / "g").mkdir(parents=True)
    n, T = 5, 20
    qpos = np.zeros((T, 9), np.float32)
    action = np.ones((T - 1, 7), np.float32)
    np.savez(lat / "G00" / "g" / "t.npz", mean=np.zeros((n, 64, 8), np.float16), img_step=np.arange(n) * 4,
             qpos=qpos, action=action, hand=np.zeros((n, 4, 4)))
    e = {"path": str(root / "G00" / "g" / "t.npz"), "split": "train", "n_images": n, "kind": "insert",
         "outcome": "inserted", "R_P": np.eye(3).tolist()}
    ds = LatentPolicyDataset(lat, [e, {**e, "kind": "contact"}], "train", root, scale=1.0)
    assert len(ds) == n                         # contact trials are not policy data
    item = ds[n - 1]                            # last image: chunk padded by holding the last target
    assert item["z"].shape == (4, 64, 8) and item["a"].shape == (POLICY_CHUNK * 7,)
    assert torch.allclose(item["a"], torch.ones(POLICY_CHUNK * 7))
