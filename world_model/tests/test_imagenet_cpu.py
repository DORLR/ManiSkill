"""The ImageNet start of the picture encoder (policy320.PictureTokens.init_imagenet, the user's decision 2026-11-18)
on a CPU with the sample data: which tensors are copied, the RGB normalisation, both training scripts with
--imagenet (the flag stored in the checkpoint, the loaders rebuilding the same model, a resume with a different flag
refused), and old checkpoints (no flag) loading as before. Needs torchvision's IMAGENET1K_V1 file (downloaded once
into $TORCH_HOME) and the environment of test_demo320.py.

    python world_model/tests/test_imagenet_cpu.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_policy_cpu as TP  # noqa: E402  (its sample split and run() helper)
import test_wm_cpu as TW  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.tdmpc.policy320 import PictureTokens, ResnetFMPolicy  # noqa: E402
from wm.tdmpc.wm320 import load  # noqa: E402


def test_init():
    src = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    for stem in ("standard", "nopool"):
        torch.manual_seed(0)
        pt = PictureTokens(stem, imagenet=True)
        assert pt.init_imagenet() == 20, "first layer + 19 trunk convolutions"
        w = src.conv1.weight
        assert torch.equal(pt.stem[0].weight[:, :3], w) and torch.allclose(pt.stem[0].weight[:, 3], w.mean(1))
        layers = [m for m in pt.trunk if isinstance(m, torch.nn.Sequential)]
        for k, tgt in enumerate(layers, 1):
            ref = getattr(src, f"layer{k}").state_dict()
            got = tgt.state_dict()
            for name, v in ref.items():
                if name.endswith(("conv1.weight", "conv2.weight", "downsample.0.weight")):
                    assert torch.equal(got[name], v), (stem, k, name)
            for name, mod in tgt.named_modules():
                if isinstance(mod, torch.nn.GroupNorm):
                    assert torch.all(mod.weight == 1) and torch.all(mod.bias == 0), "group norms keep their fresh values"
        rgb = torch.randint(0, 256, (2, 3, 320, 320), dtype=torch.uint8)
        depth = torch.randint(1000, 3000, (2, 320, 320), dtype=torch.int32)
        out = pt.eval()(rgb, depth)
        assert out.shape == (2, 100, 32) and torch.isfinite(out).all()
        assert set(pt.state_dict()) == set(PictureTokens(stem).state_dict()), "the normalisation is not in the checkpoint"
        plain = PictureTokens(stem)
        plain.load_state_dict(pt.state_dict())
        assert not torch.allclose(plain.eval()(rgb, depth), out), "the ImageNet model normalises RGB differently"
        try:
            PictureTokens(stem).init_imagenet()
            raise AssertionError("init_imagenet on a model built without imagenet=True")
        except ValueError:
            pass
    return "20 tensors copied per stem, depth weights = RGB mean, group norms fresh, own RGB normalisation"


def test_policy(tmp):
    out = tmp / "fm"
    TP.run(out, "--steps", "1", "--imagenet")
    ck = torch.load(out / "step1.pt", weights_only=False)
    assert ck["model"] == {"stem": "standard", "imagenet": True} and ck["info"]["imagenet"] and ck["info"]["imagenet_tensors_copied"] == 20
    m = ResnetFMPolicy(**ck["model"])
    m.load_state_dict(ck["ema"])
    assert m.enc.imagenet
    (out / "DONE.json").unlink()
    try:
        TP.run(out, "--steps", "2", "--resume")
        raise AssertionError("a resume without --imagenet was accepted")
    except SystemExit as e:
        assert "imagenet" in str(e.code), e.code
    TP.run(out, "--steps", "2", "--resume", "--imagenet")
    c2 = torch.load(out / "step2.pt", weights_only=False)
    assert c2["step"] == 2 and c2["info"]["imagenet_tensors_copied"] == 20, "the count survives a resume"
    old = tmp / "fm_old"                                   # no flag: as before, and the old checkpoint format still loads
    TP.run(old, "--steps", "1")
    c = torch.load(old / "step1.pt", weights_only=False)
    assert c["model"] == {"stem": "standard"}, "a plain checkpoint keeps the old format"
    m = ResnetFMPolicy(**c["model"])
    m.load_state_dict(c["ema"])
    assert not m.enc.imagenet and c["info"]["imagenet_tensors_copied"] == 0
    return "FM: flag stored and rebuilt, resume with another flag refused, a checkpoint without the flag loads as before"


def test_wm(tmp):
    out = tmp / "wm"
    TW.run(out, "--steps", "2", "--imagenet")
    start = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()][0]
    assert start["event"] == "start" and start["imagenet"] is True and start["imagenet_tensors_copied"] == 20
    ck = torch.load(out / "step2.pt", weights_only=False)
    assert ck["imagenet"] is True
    m = load(str(out / "step2.pt"), "cpu")
    assert m.enc.pic.imagenet
    ag = WMAgent(str(out / "step2.pt"), "cpu")
    assert ag.m.enc.pic.imagenet
    sd = m.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in ag.m.state_dict().items() if torch.is_tensor(v))   # (the ensembles also keep sizes)
    (out / "DONE.json").unlink()
    try:
        TW.run(out, "--steps", "3", "--resume")
        raise AssertionError("a resume without --imagenet was accepted")
    except SystemExit as e:
        assert "imagenet" in str(e.code)
    TW.run(out, "--steps", "3", "--resume", "--imagenet")
    ck.pop("imagenet")                                     # an old checkpoint (no flag) builds the plain model
    p = tmp / "old.pt"
    torch.save(ck, p)
    assert not load(str(p), "cpu").enc.pic.imagenet
    return "WM: flag stored, wm320.load and WMAgent rebuild it, resume with another flag refused, old checkpoints plain"


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="imagenet320_"))
    try:
        print("test_init OK:", test_init(), flush=True)
        print("test_policy OK:", test_policy(tmp), flush=True)
        print("test_wm OK:", test_wm(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
