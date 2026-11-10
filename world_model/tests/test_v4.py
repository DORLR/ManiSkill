"""v4 perception: crop cutting, crop placement, encoder inputs, autoencoder shapes and pose solve."""

import inspect

import numpy as np
import torch

from wm.models.crop_ae import CropAE
from wm.models.crop_encoder import CROP, CropEncoder, cut
from wm.perception.crops import CropPlacer
from wm.perception.labels import OPENING, PINS

MODEL_INPUTS = {"rgb", "depth", "plug_uv0", "hole_uv0", "K", "cam_T_world", "hand", "R_P"}
FORBIDDEN = ("h_star", "H_star", "eval_", "pitch", "grasp_pos", "condition", "meta", "goal", "charger", "seg", "label")


def test_cut_matches_slicing():
    img = torch.arange(2 * 320 * 320).view(2, 320, 320)
    uv0 = torch.tensor([[10, 50], [192, 0]])
    c = cut(img, uv0)
    assert torch.equal(c[0], img[0, 50:50 + CROP, 10:10 + CROP])
    assert torch.equal(c[1], img[1, 0:CROP, 192:192 + CROP])


def test_encoder_inputs_are_real_robot_quantities():
    params = set(inspect.signature(CropEncoder.forward).parameters) - {"self"}
    assert params == MODEL_INPUTS
    src = inspect.getsource(CropEncoder) + inspect.getsource(CropAE.encode)
    for w in FORBIDDEN:
        assert w not in src, w


def _batch(B=2):
    K = torch.tensor([[337.0, 0, 160], [0, 337.0, 160], [0, 0, 1]]).expand(B, 3, 3)
    cam = torch.eye(4).expand(B, 4, 4).clone()
    cam[:, 2, 3] = 0.1
    return {"rgb": torch.randint(0, 255, (B, 320, 320, 3), dtype=torch.uint8),
            "depth": torch.randint(0, 3000, (B, 320, 320), dtype=torch.int32),
            "plug_uv0": torch.tensor([[100, 150]] * B), "hole_uv0": torch.tensor([[60, 40]] * B),
            "K": K, "cam_T_world": cam, "hand": torch.eye(4).expand(B, 4, 4), "R_P": torch.eye(3).expand(B, 3, 3)}


def test_autoencoder_shapes_and_pose():
    torch.manual_seed(0)
    m = CropAE(encoder={"dino": False, "d": 64, "heads": 4, "crop_layers": 1, "relation_layers": 1,
                        "joint_layers": 1, "query_layers": 1}, latent_ch=16, dec_d=64, dec_layers=1, patch_layers=1, heads=4)
    out = m(_batch())
    assert out["z"].shape == (2, 64, 16)
    assert out["R"].shape == (2, 3, 3) and out["t"].shape == (2, 3)
    assert torch.isfinite(out["R"]).all() and torch.isfinite(out["t"]).all()
    geom = {k: v for k, v in _batch().items() if k not in ("rgb", "depth")}
    R, t = m.decode_pose(out["z"], geom)
    assert torch.isfinite(R).all()


def test_crop_placer_uses_memory_when_opening_hidden():
    K = np.array([[337.0, 0, 160], [0, 337.0, 160], [0, 0, 1]])
    cam = np.eye(4)
    cls = np.zeros((320, 320), np.uint8)
    cls[200:230, 150:170] = PINS
    cls[80:90, 100:120] = OPENING
    cls[60:110, 80:140][cls[60:110, 80:140] == 0] = 3            # board around the opening
    depth = np.full((320, 320), 1000, np.uint16)                  # 0.1 m
    p = CropPlacer()
    r1 = p(cls, depth, K, cam)
    cls2 = cls.copy()
    cls2[cls2 == OPENING] = 3                                     # opening now covered
    r2 = p(cls2, depth, K, cam)
    assert r2["how_hole"] == "memory_3d"
    assert np.abs(r1["hole_uv0"] - r2["hole_uv0"]).max() <= 1
    assert r1["how_plug"] == "pins"
