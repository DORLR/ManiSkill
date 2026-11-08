"""Policy shapes, loss and sampling (CPU), and the datasets' targets."""

import numpy as np
import torch

from wm.data.datasets import HISTORY_STEPS, POLICY_CHUNK, LatentPolicyDataset, adapter_io
from wm.models.policy import ACTION_DIM, CHUNK, LatentFMPolicy


def test_policy_loss_and_sample():
    torch.manual_seed(0)
    m = LatentFMPolicy(d=64, layers=1, heads=4, hidden=128)
    z, q = torch.randn(3, 4, 64, 8), torch.randn(3, 7)
    a = torch.randn(3, CHUNK * ACTION_DIM)
    loss = m.loss(z, q, a)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.sample(z, q, steps=3, n=2).shape == (3, 2, CHUNK * ACTION_DIM)


def _latent_file(lat, n=5, T=20):
    (lat / "G00" / "g").mkdir(parents=True)
    np.savez(lat / "G00" / "g" / "t.npz", mean=np.zeros((n, 64, 8), np.float16), img_step=np.arange(n) * 4,
             qpos=np.zeros((T, 9), np.float32), action=np.ones((T - 1, 7), np.float32),
             hand=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), hand_cmd=np.tile(np.eye(4, dtype=np.float32), (T - 1, 1, 1)),
             cam_T_world=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), K=np.eye(3, dtype=np.float32))


def test_policy_dataset(tmp_path):
    root, lat = tmp_path / "data", tmp_path / "lat"
    n = 5
    _latent_file(lat, n)
    e = {"path": str(root / "G00" / "g" / "t.npz"), "split": "train", "n_images": n, "kind": "insert",
         "outcome": "inserted", "R_P": np.eye(3).tolist()}
    ds = LatentPolicyDataset(lat, [e, {**e, "kind": "contact"}], "train", root, scale=1.0)
    assert len(ds) == n                         # contact trials are not policy data
    item = ds[n - 1]                            # last image: chunk padded by holding the last target
    assert item["z"].shape == (4, 64, 8) and item["q"].shape == (7,) and item["a"].shape == (POLICY_CHUNK * 7,)
    assert torch.allclose(item["a"], torch.ones(POLICY_CHUNK * 7))


def test_adapter_io_blocked_history():
    T = 40
    qpos = np.zeros((T + 1, 9), np.float32)
    action = np.ones((T, 7), np.float32)       # commands 1 rad ahead, joints never move: blocked
    x, y = adapter_io(qpos, action, 30)
    assert x.shape == (7 + 7 * HISTORY_STEPS + 56,) and y.shape == (56,)
    assert np.allclose(x[7:7 + 7 * HISTORY_STEPS], 1) and np.allclose(y, 0)
    x0, _ = adapter_io(qpos, action, 0)
    assert np.allclose(x0[7:7 + 7 * HISTORY_STEPS], 0)     # no history before the start
