"""Leakage checks: models only see real-robot inputs, never grasp parameters, shape identity or
simulator truth; labels from the one insertion (h_star) are used only by the loss; the test ports
A-D never reach training or model selection."""

import inspect

import numpy as np

from wm.data import splits, store
from wm.data.datasets import FrameDataset, LatentWindowDataset
from wm.models.encoder import ImageEncoder
from wm.models.vae import TokenVAE

MODEL_INPUTS = {"rgb", "depth", "K", "cam_T_world", "hand", "R_P"}
LOSS_ONLY = {"h_star", "seg"}
# Label / metadata / truth accessors (the word "port" alone is legitimate: port axes, socket pixels).
FORBIDDEN = ("h_star", "H_star", "eval_", "pitch", "grasp_pos", "condition", '["port"]', "meta",
             "goal_pose", "charger", "seg")


def _fake_trajectory(path, port="G00", condition="train"):
    n, T = 3, 9
    arrays = {"rgb": np.zeros((n, 8, 8, 3), np.uint8), "depth": np.ones((n, 8, 8), np.uint16),
              "seg": np.zeros((n, 8, 8), np.uint8), "img_step": np.array([0, 4, 8], np.int32),
              "cam_T_world": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), "K": np.eye(3, dtype=np.float32),
              "qpos": np.zeros((T, 9), np.float32), "hand": np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)),
              "action": np.zeros((T - 1, 7), np.float32), "phase": np.zeros(T - 1, np.uint8),
              "eval_rel": np.zeros((T, 7), np.float32)}
    meta = {"port": port, "grasp": f"{port}_g00", "condition": condition, "kind": "insert",
            "pitch_deg": -40.0, "grasp_pos_m": -0.02, "H_star": np.eye(4).tolist()}
    store.save(path, arrays, meta)
    return {"path": str(path), "port": port, "grasp": meta["grasp"], "condition": condition, "kind": "insert",
            "outcome": "inserted", "n_images": n, "split": splits.split_of(port, condition),
            "R_P": np.eye(3).tolist()}


def test_frame_dataset_yields_only_inputs_and_loss_label(tmp_path):
    e = _fake_trajectory(tmp_path / "t.npz")
    item = FrameDataset([e], "train")[0]
    assert set(item) == MODEL_INPUTS | LOSS_ONLY


def test_encoder_never_reads_labels_or_truth():
    src = inspect.getsource(TokenVAE.encode) + inspect.getsource(ImageEncoder)
    for word in FORBIDDEN:
        assert word not in src, f"encoder source mentions {word!r}"
    params = set(inspect.signature(ImageEncoder.forward).parameters) - {"self"}
    assert params == MODEL_INPUTS


def test_world_model_samples_contain_no_labels(tmp_path):
    root, lat = tmp_path / "data", tmp_path / "lat"
    e = _fake_trajectory(root / "G00" / "g" / "t.npz")
    (lat / "G00" / "g").mkdir(parents=True)
    n, T = 12, 60
    np.savez(lat / "G00" / "g" / "t.npz", mean=np.zeros((n, 64, 8), np.float16), img_step=np.arange(n) * 4,
             qpos=np.zeros((T, 9), np.float32), action=np.zeros((T - 1, 7), np.float32),
             hand=np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), hand_cmd=np.tile(np.eye(4, dtype=np.float32), (T - 1, 1, 1)),
             cam_T_world=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)), K=np.eye(3, dtype=np.float32))
    e["n_images"] = n
    ds = LatentWindowDataset(lat, [e], "train", root, scale=1.0)
    item = ds[0]
    assert set(item) == {"x_cond", "x", "y", "rel_t", "geom"}
    assert set(item["geom"]) == {"K", "cam_T_world", "hand", "R_P"}        # real-robot geometry, for reading poses
    assert -16.0 <= float(item["rel_t"]) <= 16.0 and float(item["rel_t"]) != 0


def test_test_ports_never_train_or_validate():
    for p in splits.TEST_PORTS:
        for c in ("train", "pitch_extrap", "combined"):
            assert splits.split_of(p, c) == "test"
    assert splits.split_of("G00", "pitch_extrap") == "ood_val"
    assert splits.split_of("G46", "train") == "ood_val"
    assert splits.split_of("G00", "train") == "train"
