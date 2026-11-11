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


def test_success_label_horizon():
    from wm.data.history import SuccessDataset
    ins = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], bool)
    assert SuccessDataset.labels(ins, 0).tolist() == ins.tolist()                       # "inserted now"
    assert SuccessDataset.labels(ins, 2).tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]     # now or within 2 images
    assert SuccessDataset.labels(ins, 100).tolist() == [1] * 7 + [0] * 3                # never looks back
    assert SuccessDataset.labels(np.zeros(5, bool), 40).sum() == 0


def test_exploration_segments_do_not_use_the_scene():
    """Starts and maneuvers are drawn from a random generator only (no robot, no simulator state)."""
    import inspect
    from wm.collect import explore
    for fn in (explore.sample_start, explore.sample_segments):
        assert list(inspect.signature(fn).parameters) == ["rng"]
    a = explore.sample_segments(np.random.default_rng(3))
    b = explore.sample_segments(np.random.default_rng(3))
    assert a == b
    src = inspect.getsource(explore.Explorer)
    for w in ("goal_pose", "charger", "eval_rel", "relative("):
        assert w not in src, w


def test_alternating_mask_matches_imagination():
    from wm.data.history import HIST, vision_mask
    avail = np.ones(HIST, bool)
    hits = 0
    for seed in range(200):
        seen = vision_mask(avail, np.random.default_rng(seed), p_tail=0.0, p_mid=0.0, p_drop=0.0, p_alt=1.0)
        assert seen[-1]                                   # "now" always has its frame
        k = 1
        while k < HIST and seen[-1 - k] != seen[-k]:      # alternating run counted back from now
            k += 1
        assert k >= 2 and not seen[-2]
        hits += 1
    assert hits == 200


def test_vision_adapter_ignores_unseen_slots_and_keeps_the_blind_path():
    """The vision-conditioned joint adapter: output shape; latents of unseen slots do not matter (attention over
    the seen slots only, no flag); seen latents, the commanded path and the numeric input do."""
    from wm.data.history import HIST, PATH_DIM, PROPRIO_DIM
    from wm.models.adapter import IN_DIM, OUT_DIM, VisionJointAdapter
    torch.manual_seed(0)
    m = VisionJointAdapter(latent_ch=16, d=64, layers=2, heads=4, hidden=64).eval()
    B = 3
    x, path = torch.randn(B, IN_DIM), torch.randn(B, PATH_DIM)
    z, prop, dt = torch.randn(B, HIST, 64, 16), torch.randn(B, HIST, PROPRIO_DIM), -torch.rand(B, HIST)
    avail = torch.ones(B, HIST, dtype=torch.bool)
    seen = avail.clone()
    seen[:, -6:] = False
    with torch.no_grad():
        y = m(x, z, seen, prop, dt, avail, path)
        z2 = z.clone()
        z2[:, -6:] = torch.randn(B, 6, 64, 16)
        assert y.shape == (B, OUT_DIM)
        assert torch.allclose(y, m(x, z2, seen, prop, dt, avail, path), atol=1e-5)
        z3 = z.clone()
        z3[:, 0] += 1.0
        assert not torch.allclose(y, m(x, z3, seen, prop, dt, avail, path), atol=1e-5)
        assert not torch.allclose(y, m(x, z, seen, prop, dt, avail, path + 1.0), atol=1e-5)
        assert not torch.allclose(y, m(x + 1.0, z, seen, prop, dt, avail, path), atol=1e-5)


def test_success_head_features_unchanged_by_refactor():
    """SuccessHead.forward = out(features): the logit of a checkpoint does not depend on the refactor."""
    from wm.data.history import HIST, PROPRIO_DIM
    from wm.models.success import SuccessHead
    torch.manual_seed(0)
    m = SuccessHead(latent_ch=16, d=64, layers=2, heads=4).eval()
    z, prop, dt = torch.randn(2, HIST, 64, 16), torch.randn(2, HIST, PROPRIO_DIM), -torch.rand(2, HIST)
    seen = torch.ones(2, HIST, dtype=torch.bool)
    with torch.no_grad():
        assert torch.equal(m(z, seen, prop, dt, seen), m.out(m.features(z, seen, prop, dt, seen))[:, 0])
