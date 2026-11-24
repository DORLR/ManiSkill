"""The picture-encoder switch of the world model (policy320.PictureTokens(encoder=...), wm/tdmpc/encoders320.py,
2026-11-21) on a CPU with small batches:
  1 legacy bit-identity: the default (ResNet-18) model of this code and of git 6b37757, built with the same seed, have
    the same state_dict keys and tensors and give identical outputs; an old checkpoint (no encoder keys) loads into the
    same model through wm320.load and agent320.WMAgent; the same for the stand-alone policy (policy320.ResnetFMPolicy)
  2 shapes: (N, 100, token_dim) for resnet18 / resnet34, (N, 400, token_dim) for segformer_b1; the world model's
    token positions and input layer follow
  3 zero depth start: after init_pretrained the first layer's output, and the encoder's output, do not depend on depth
  4 pretrained weights: the first layer's RGB slices and deep tensors equal the published checkpoints; the SegFormer's
    configuration and RGB normalisation equal the Hub's files; the kept map equals the original model's fused map
  5 locality: the gradient of token (i, j) with respect to the fused map is non-zero only inside cell (i, j); the
    reshaped attention equals an explicit per-cell computation
  6 checkpoint round trip: save and wm320.load / WMAgent reproduce the outputs for all three encoders
  7 refusal: train_wm320.py --resume with other encoder settings than the checkpoint's refuses
  8 (smoke) one loss and backward pass of the world model per encoder on random windows
  9 the SegFormer decoder's batch norm keeps its ADE20K statistics: unchanged (bit for bit) by training steps (loss,
    backward, optimiser step) in train(); the same output in train() and eval(); its weight and bias get gradients
    and move; the statistics survive a checkpoint round trip
  10 AutoMate command line: a missing --encoder / --token-dim and resnet18 are refused on AutoMate data; invalid depth
    defaults to (0 - mean) / std and must lie below the smallest valid z; placeholders refused; 320 x 320 only;
    scripts/automate/depth_stats.py against numpy; a statistics file of another split refused; on AutoMate data the
    --depth-* flags are refused (the statistics file only), an undecided split ("unassigned") is refused by
    train_wm320 and depth_stats, and fewer than MIN_SELECT_GRASPS selection grasps or no held-out trajectory is
    refused; on ManiSkill data flags next to a statistics file get its split hash checked; a statistics file of
    another split identity refused; after depth_stats.py --distribution-only (no depth_stats.json) train_wm320 refuses
    (r10-data-01); a refused depth_stats.py run with a D_max leaves no earlier depth_stats.json (r11-data-01)
  11 AutoMate training on a CPU: raw trajectories with collect.py's meta (simulator truth included) -> demo320.convert
    -> manifest with a split identity -> depth_stats.py -> train_wm320.main --encoder resnet34 --token-dim 8 --steps 2
    (optimiser steps through stats_for, WindowData, the imitation loader and one validation, heldout_shape /
    heldout_grasp logged separately); last.pt and step2.pt hold the encoder settings and provenance (this tree's wm
    file, the split identity, the depth statistics sha256, the completeness record, the converter's code record
    "fast_copy_code") and the offset label "gap_mm" (decision 6 of 2026-11-23); a manifest without a split
    identity, without a completeness record or with a partial one refused (--allow-partial passes that check);
    --resume of that checkpoint on a fast copy of another split identity refused. r11-integration-01: every
    validation split logs face_items_kept = [contact pushes in the face AUC, contact pushes in the split].
    Without scikit-learn in the environment the test installs a stand-in roc_auc_score (rank formula) for the run
    and says so; train_wm320 itself refuses to start without scikit-learn.
Needs: torchvision's IMAGENET1K_V1 files for ResNet-18 / ResNet-34 ($TORCH_HOME), transformers and the Hub files of
nvidia/segformer-b1-finetuned-ade-512-512 ($HF_HOME), TD-MPC2's code (wm.tdmpc.use_tdmpc2), and the code of git 6b37757:
by default `git archive` of this repository, or WM_LEGACY_SRC = a folder holding that commit's world_model/src/wm
(needed where the code is a copy without git history, e.g. rs2: `git archive 6b37757 world_model/src/wm | tar -x -C <dir>`).

    python world_model/tests/test_encoders_automate.py
"""

import importlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts" / "train"))
import train_wm320 as T  # noqa: E402
from wm.online.agent320 import WMAgent  # noqa: E402
from wm.tdmpc import encoders320 as E  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.policy320 import ENCODER_DEFAULTS, PictureTokens, ResnetFMPolicy  # noqa: E402
from wm.tdmpc.wm320 import WM320, load  # noqa: E402

LEGACY_COMMIT = "6b37757"
STATS = {"s0": np.full(WD.ACTION_DIM, 0.05, np.float32), "top": np.full(WD.ACTION_DIM, 0.5, np.float32),
         "a_mean": np.zeros(16 * 7, np.float32), "a_std": np.full(16 * 7, 0.1, np.float32)}
# non-default, to see them travel; the invalid value lies below the smallest valid z, (0.07 - 0.4) / 0.2 = -1.65
DEPTH = {"depth_max_m": 1.5, "depth_mean_m": 0.4, "depth_std_m": 0.2, "depth_invalid": -2.5}
NEW = {"resnet34": {"encoder": "resnet34", "token_dim": 128, **DEPTH}, "segformer_b1": {"encoder": "segformer_b1", "token_dim": 128, **DEPTH}}


def pictures(n: int, seed: int = 0):
    """n random pictures: rgb (n, 320, 320, 3) uint8, depth (n, 320, 320) int16 in 0.1 mm with invalid (0) pixels."""
    g = torch.Generator().manual_seed(seed)
    rgb = torch.randint(0, 256, (n, 320, 320, 3), dtype=torch.uint8, generator=g)
    depth = torch.randint(500, 20000, (n, 320, 320), dtype=torch.int16, generator=g)
    depth[:, :40, :40] = 0
    return rgb, depth


def legacy_package(tmp: Path):
    """git 6b37757's world_model/src/wm, importable as the package wm_legacy."""
    dst = tmp / "legacy"
    if not (dst / "wm_legacy").exists():
        src = os.environ.get("WM_LEGACY_SRC")
        if src:
            shutil.copytree(Path(src) / "world_model" / "src" / "wm", dst / "wm_legacy")
        else:
            arc = tmp / "legacy.tar"
            with open(arc, "wb") as f:
                r = subprocess.run(["git", "-C", str(HERE.parents[1]), "archive", LEGACY_COMMIT, "world_model/src/wm"],
                                   stdout=f, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"test 1 needs git {LEGACY_COMMIT}'s world_model/src/wm: `git archive` failed here "
                                   f"({r.stderr.strip()[:200]}); set WM_LEGACY_SRC to a folder holding that commit's "
                                   "world_model/src/wm (not skipped: the legacy identity must be checked)")
            with tarfile.open(arc) as t:
                t.extractall(dst / "x", filter="data")
            shutil.move(str(dst / "x" / "world_model" / "src" / "wm"), str(dst / "wm_legacy"))
    if str(dst) not in sys.path:
        sys.path.insert(0, str(dst))
    return (importlib.import_module("wm_legacy.tdmpc.wm320"), importlib.import_module("wm_legacy.tdmpc.policy320"),
            importlib.import_module("wm_legacy.online.agent320"))


def same_state(a: torch.nn.Module, b: torch.nn.Module) -> None:
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb), (set(sa) ^ set(sb))
    for k in sa:                                     # (TD-MPC2's ensembles also store their sizes, as torch.Size)
        if torch.is_tensor(sa[k]):
            assert sa[k].dtype == sb[k].dtype and torch.equal(sa[k], sb[k]), k
        else:
            assert sa[k] == sb[k], k


@torch.no_grad()
def wm_outputs(m, rgb, depth, q, a):
    """Everything a decision reads: tokens, z and the heads, for one observation per row (rgb (B*4, ...))."""
    tok = m.enc.tokens(rgb, depth)
    tok = tok.view(q.shape[0], 4, *tok.shape[1:])
    z = m.enc.fuse(tok, q)
    return {"tok": tok, "z": z, "act": m.act_chunk(z), "V": m.V(z), "Q": m.Q(z, a), "next": m.next(z, a), "r": m.reward(z, a),
            "done": m._done(z), "off": m._offset(z) if m._offset is not None else torch.zeros(1)}


def same_outputs(x: dict, y: dict) -> None:
    for k in x:
        assert torch.equal(x[k], y[k]), k


def ckpt_of(m: WM320, stem: str, imagenet: bool, offset_weight: float, enc: dict | None) -> dict:
    """A checkpoint as train_wm320.py writes it; enc None: an old checkpoint (before the encoder keys)."""
    return {"format": "wm320-1", "labels": WD.LABELS, "stem": stem, "imagenet": imagenet, "offset_weight": offset_weight, **(enc or {}),
            "shaping": 0.3, "cfg": {"gamma": 0.967}, "stats": STATS, "state": m.state_dict(), "step": 1}


# ------------------------------------------------------------------------------------------------------------- 1
def test_legacy_identity(tmp: Path):
    OW, OP, OA = legacy_package(tmp)
    rgb, depth = pictures(8)
    q, a = torch.randn(2, 7), torch.rand(2, WD.ACTION_DIM) * 2 - 1
    n = 0
    for stem in ("standard", "nopool"):
        for imagenet in (False, True):
            for ow in (0.0, 1.0):
                torch.manual_seed(0)
                old = OW.WM320(stem, STATS, 0.3, imagenet=imagenet, offset_weight=ow, gamma=0.967)
                c_old = old.enc.pic.init_imagenet() if imagenet else 0
                rng_old = torch.random.get_rng_state()
                torch.manual_seed(0)
                new = WM320(stem, STATS, 0.3, imagenet=imagenet, offset_weight=ow, gamma=0.967)
                c_new = new.enc.pic.init_pretrained() if imagenet else 0
                assert c_old == c_new and torch.equal(rng_old, torch.random.get_rng_state()), "same random numbers drawn"
                same_state(old, new)
                assert new.enc.pic.n_tokens == 100 and new.enc.pic.token_dim == 32
                old.eval(), new.eval()
                same_outputs(wm_outputs(old, rgb, depth, q, a), wm_outputs(new, rgb, depth, q, a))
                n += 1
    # an old checkpoint (no encoder keys) through the new loaders = the old loaders
    torch.manual_seed(1)
    old = OW.WM320("standard", STATS, 0.3, imagenet=True, offset_weight=1.0, gamma=0.967)
    old.enc.pic.init_imagenet()
    p = tmp / "old.pt"
    torch.save(ckpt_of(old, "standard", True, 1.0, None), p)
    m_old, m_new = OW.load(str(p), "cpu"), load(str(p), "cpu")
    same_state(m_old, m_new)
    same_outputs(wm_outputs(m_old, rgb, depth, q, a), wm_outputs(m_new, rgb, depth, q, a))
    a_old, a_new = OA.WMAgent(str(p), "cpu"), WMAgent(str(p), "cpu")
    same_state(a_old.m, a_new.m)
    obs = (rgb[:4].numpy(), depth[:4].numpy().astype(np.uint16), np.random.default_rng(0).normal(size=7).astype(np.float32))
    assert torch.equal(a_old.encode(*obs), a_new.encode(*obs))
    assert m_new.enc.pic.encoder == "resnet18" and m_new.enc.pic.token_dim == 32
    # the stand-alone policy (policy320.ResnetFMPolicy), which reads the encoder's sizes now
    for stem in ("standard", "nopool"):
        torch.manual_seed(2)
        po = OP.ResnetFMPolicy(stem, imagenet=True, offset=True)
        torch.manual_seed(2)
        pn = ResnetFMPolicy(stem, imagenet=True, offset=True)
        same_state(po, pn)
        po.eval(), pn.eval()
        r5, d5 = rgb.view(2, 4, 320, 320, 3), depth.view(2, 4, 320, 320)
        with torch.no_grad():
            assert torch.equal(po.tokens(r5, d5), pn.tokens(r5, d5))
            torch.manual_seed(3)
            lo = po.loss(r5, d5, q, torch.randn(2, 16 * 7))
            torch.manual_seed(3)
            ln = pn.loss(r5, d5, q, torch.randn(2, 16 * 7))
            assert torch.equal(lo, ln)
            assert torch.equal(po.read_offset(r5, d5), pn.read_offset(r5, d5))
    return (f"{n} world models (2 stems x imagenet x offset head) and 2 policies: same keys, tensors and outputs as git "
            f"{LEGACY_COMMIT}; an old checkpoint loads identically through wm320.load and WMAgent")


# ------------------------------------------------------------------------------------------------------------- 2
def test_shapes():
    rgb, depth = pictures(2)
    r, d = rgb.permute(0, 3, 1, 2), depth
    out = []
    for kw, n_tok in (({}, 100), ({"imagenet": True, **NEW["resnet34"]}, 100), ({"imagenet": True, **NEW["segformer_b1"]}, 400),
                      ({"imagenet": True, **NEW["segformer_b1"], "token_dim": 64}, 400)):
        torch.manual_seed(0)
        pt = PictureTokens("standard", **kw).eval()
        with torch.no_grad():
            t = pt(r, d)
        td = kw.get("token_dim", 32)
        assert t.shape == (2, n_tok, td) and t.dtype == torch.float32 and torch.isfinite(t).all(), t.shape
        assert (pt.n_tokens, pt.token_dim) == (n_tok, td)
        out.append(f"{kw.get('encoder', 'resnet18')}/{td}: {tuple(t.shape)}")
    for enc in ("resnet34", "segformer_b1"):
        m = WM320("standard", STATS, 0.3, imagenet=True, gamma=0.967, **NEW[enc])
        assert m.enc.inp.in_features == 128 and m.enc.token_pos.shape == (m.enc.pic.n_tokens, 256)
        for bad in ({"imagenet": False}, {"stem": "nopool"}):
            try:
                PictureTokens(**{"stem": "standard", "imagenet": True, **NEW[enc], **bad})
                raise AssertionError(f"{enc} accepted {bad}")
            except ValueError:
                pass
    return "; ".join(out) + "; world model: inp 128 -> 256, token_pos (n_tokens, 256); imagenet=False or stem nopool refused"


# ------------------------------------------------------------------------------------------------------------- 3
def test_zero_depth():
    rgb, depth = pictures(2)
    r = rgb.permute(0, 3, 1, 2)
    msgs = []
    for enc in ("resnet34", "segformer_b1"):
        torch.manual_seed(0)
        pt = PictureTokens("standard", imagenet=True, **NEW[enc]).eval()
        pt.init_pretrained()
        conv = pt.stem[0] if enc == "resnet34" else pt.seg.segformer.encoder.patch_embeddings[0].proj
        assert torch.all(conv.weight[:, 3] == 0)
        c = (r.float() / 255.0 - pt.rgb_mean) / pt.rgb_std
        with torch.no_grad():
            firsts = [conv(torch.cat([c, E.depth_z(dd, 1.5, 0.4, 0.2, -1.0).unsqueeze(1)], 1)) for dd in (depth, depth.flip(1) // 3)]
            assert torch.equal(firsts[0], firsts[1]), "first layer output depends on depth"
            outs = [pt(r, dd) for dd in (depth, torch.zeros_like(depth), torch.full_like(depth, 3000), torch.full_like(depth, 30000))]
        for o in outs[1:]:
            assert torch.equal(outs[0], o), "encoder output depends on depth"
        msgs.append(f"{enc}: first layer and all {outs[0].shape[1]} tokens bit-identical for random / zero / constant depth")
    # the normalisation itself
    d = torch.tensor([[0, 1, 4000, 15000, 30000]], dtype=torch.int16)
    z = E.depth_z(d, 1.5, 0.4, 0.2, -1.0)
    want = torch.tensor([[-1.0, (1e-4 - 0.4) / 0.2, (0.4 - 0.4) / 0.2, (1.5 - 0.4) / 0.2, (1.5 - 0.4) / 0.2]])
    assert torch.allclose(z, want, atol=1e-6), z
    return "; ".join(msgs) + "; depth_z: 0 -> c, metres, clipped at D_MAX, (d - mu) / sigma"


# ------------------------------------------------------------------------------------------------------------- 4
def test_pretrained():
    torch.manual_seed(0)
    pt = PictureTokens("standard", imagenet=True, **NEW["resnet34"])
    n34 = pt.init_pretrained()
    assert n34 == 36, "first layer + 35 trunk convolutions"
    src = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)
    assert torch.equal(pt.stem[0].weight[:, :3], src.conv1.weight) and torch.all(pt.stem[0].weight[:, 3] == 0)
    layers = [m for m in pt.trunk if isinstance(m, torch.nn.Sequential)]
    assert [len(x) for x in layers] == [3, 4, 6, 3]
    for k, tgt in enumerate(layers, 1):
        ref, got = getattr(src, f"layer{k}").state_dict(), tgt.state_dict()
        for name, v in ref.items():
            if name.endswith(("conv1.weight", "conv2.weight", "downsample.0.weight")):
                assert torch.equal(got[name], v), (k, name)
        for mod in tgt.modules():
            assert not isinstance(mod, torch.nn.BatchNorm2d)
            if isinstance(mod, torch.nn.GroupNorm):
                assert mod.num_groups == 8 and torch.all(mod.weight == 1) and torch.all(mod.bias == 0)
    assert isinstance(pt.trunk[0], torch.nn.MaxPool2d) and pt.squeeze.in_channels == 512 and pt.squeeze.out_channels == 128

    from huggingface_hub import hf_hub_download
    from transformers import SegformerForSemanticSegmentation
    hub = json.loads(Path(hf_hub_download(E.SEGFORMER_B1, "config.json", revision=E.SEGFORMER_B1_REVISION, local_files_only=True)).read_text())
    pre = json.loads(Path(hf_hub_download(E.SEGFORMER_B1, "preprocessor_config.json", revision=E.SEGFORMER_B1_REVISION,
                                          local_files_only=True)).read_text())
    for k, v in E.SEGFORMER_B1_CONFIG.items():
        assert (len(hub["id2label"]) if k == "num_labels" else hub[k]) == v, k
    assert tuple(pre["image_mean"]) == E.SEGFORMER_B1_MEAN and tuple(pre["image_std"]) == E.SEGFORMER_B1_STD and pre["do_normalize"]
    torch.manual_seed(0)
    ps = PictureTokens("standard", imagenet=True, **NEW["segformer_b1"])
    nseg = ps.init_pretrained()
    orig = SegformerForSemanticSegmentation.from_pretrained(E.SEGFORMER_B1, **E.SEGFORMER_B1_LOAD).eval()
    ref, got = orig.state_dict(), ps.seg.state_dict()
    assert set(got) == set(ref) - {"decode_head.classifier.weight", "decode_head.classifier.bias"} and nseg == len(got)
    k0 = "segformer.encoder.patch_embeddings.0.proj.weight"
    assert torch.equal(got[k0][:, :3], ref[k0]) and torch.all(got[k0][:, 3] == 0)
    deep = ["segformer.encoder.patch_embeddings.0.proj.bias", "segformer.encoder.block.3.1.mlp.dense2.weight",
            "segformer.encoder.block.2.1.attention.self.sr.weight", "segformer.encoder.layer_norm.3.weight",
            "decode_head.linear_c.3.proj.weight", "decode_head.linear_fuse.weight", "decode_head.batch_norm.running_mean",
            "decode_head.batch_norm.running_var", "decode_head.batch_norm.weight"]
    for k in deep:
        assert torch.equal(got[k], ref[k]), k
    for k in ref:                                   # in fact every tensor but the first layer's and the classifier
        if k != k0 and not k.startswith("decode_head.classifier."):
            assert torch.equal(got[k], ref[k]), k
    assert isinstance(ps.seg.decode_head.batch_norm, E.FrozenStatsBatchNorm2d)          # a subclass of nn.BatchNorm2d
    assert [n for n, mod in ps.named_modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)] == ["seg.decode_head.batch_norm"]
    # the kept map is the original model's fused map: the output of decode_head.activation (after linear_fuse and
    # batch_norm), taken from the unmodified model by a hook, for the same RGB (depth has zero weights)
    rgb, depth = pictures(1)
    r = rgb.permute(0, 3, 1, 2)
    ps.eval()
    x3 = (r.float() / 255.0 - ps.rgb_mean) / ps.rgb_std
    box = {}
    h = orig.decode_head.activation.register_forward_hook(lambda mod, i, o: box.update(f=o))
    with torch.no_grad():
        orig(pixel_values=x3)
        h.remove()
        x4 = torch.cat([x3, E.depth_z(depth, *DEPTH.values()).unsqueeze(1)], 1)
        mine = E.segformer_fused(ps.seg, x4)
    assert mine.shape == (1, 256, 80, 80) and box["f"].shape == (1, 256, 80, 80)
    err = float((mine - box["f"]).abs().max())
    assert err < 1e-5, err
    return (f"resnet34: {n34} tensors copied (first layer RGB, depth 0; 35 convolutions), group norms fresh; segformer_b1: "
            f"{nseg} tensors copied (all but the ADE20K classifier; first layer RGB, depth 0), config and RGB normalisation = "
            f"the Hub's; kept map = the original's fused map (max |diff| {err:.1e})")


# ------------------------------------------------------------------------------------------------------------- 5
def test_locality():
    torch.manual_seed(0)
    lr = E.LocalRefine(256, 4, 128).double()
    f = torch.randn(2, 256, 80, 80, dtype=torch.float64, requires_grad=True)
    t = lr(f)
    assert t.shape == (2, 400, 128)
    for b, i, j in ((0, 0, 0), (1, 7, 13), (0, 19, 19), (1, 19, 0)):
        g, = torch.autograd.grad(t[b, 20 * i + j].sum(), f, retain_graph=True)
        nz = g.abs().sum(1) > 0                                                  # (2, 80, 80)
        inside = torch.zeros_like(nz)
        inside[b, 4 * i:4 * i + 4, 4 * j:4 * j + 4] = True
        assert not (nz & ~inside).any(), (b, i, j)
        assert nz[inside].all(), "every fine feature of the cell contributes"
    # the reshaped attention equals the per-cell computation written out
    with torch.no_grad():
        for b, i, j in ((0, 3, 5), (1, 19, 19)):
            cell = f[b, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4].reshape(256, 16).T + lr.sub_pos        # (16, C), row-major
            qv = f[b, :, 4 * i:4 * i + 4, 4 * j:4 * j + 4].mean((1, 2))[None, None]
            kv = cell[None]
            ref = lr.mlp(qv + lr.attn(lr.ln_q(qv), lr.ln_k(kv), lr.ln_v(kv), need_weights=False)[0])[0, 0]
            assert torch.allclose(ref, t[b, 20 * i + j], atol=1e-10)
    return "gradient of each tested token non-zero exactly on its own 4 x 4 cell; reshaped attention = per-cell attention"


# ------------------------------------------------------------------------------------------------------------- 6
def test_round_trip(tmp: Path):
    rgb, depth = pictures(4)
    q, a = torch.randn(1, 7), torch.rand(1, WD.ACTION_DIM) * 2 - 1
    out = []
    for enc in ("resnet18", "resnet34", "segformer_b1"):
        settings = {**ENCODER_DEFAULTS, **NEW.get(enc, {})}
        imagenet = enc != "resnet18"
        torch.manual_seed(0)
        m = WM320("standard", STATS, 0.3, imagenet=imagenet, offset_weight=1.0, gamma=0.967, **settings)
        if enc == "segformer_b1":                          # statistics off their defaults, to see them travel
            bn = m.enc.pic.seg.decode_head.batch_norm
            g = torch.Generator().manual_seed(7)
            bn.running_mean.copy_(torch.randn(bn.running_mean.shape, generator=g))
            bn.running_var.copy_(torch.rand(bn.running_var.shape, generator=g) + 0.5)
        m.eval()
        p = tmp / f"{enc}.pt"
        torch.save(ckpt_of(m, "standard", imagenet, 1.0, settings), p)
        ck = torch.load(p, weights_only=False)
        assert all(ck[k] == v for k, v in settings.items())
        ld = load(str(p), "cpu")
        same_state(m, ld)
        assert {k: getattr(ld.enc.pic, k) for k in settings} == settings
        same_outputs(wm_outputs(m, rgb, depth, q, a), wm_outputs(ld, rgb, depth, q, a))
        ag = WMAgent(str(p), "cpu")
        same_state(m, ag.m)
        obs = (rgb.numpy(), depth.numpy().astype(np.uint16), q[0].numpy())
        z_ag = ag.encode(*obs)
        assert torch.equal(z_ag, wm_outputs(m, rgb, depth, q, a)["z"])
        out.append(f"{enc} ({ld.enc.pic.n_tokens} x {ld.enc.pic.token_dim})")
    return "save -> wm320.load and WMAgent: same tensors and outputs for " + ", ".join(out)


# ------------------------------------------------------------------------------------------------------------- 7
def run_train(data: Path, out: Path, *extra):
    argv = ["train_wm320.py", "--data", str(data), "--out", str(out), "--stem", "standard", "--shift-px", "4", "--shaping", "0.3",
            "--wm-batch", "1", "--bc-batch", "1", "--accum", "1", "--workers-wm", "0", "--workers-bc", "0", "--device", "cpu", *extra]
    old, sys.argv = sys.argv, argv
    try:
        T.main()
    finally:
        sys.argv = old


def refused(data, out, *extra) -> str:
    try:
        run_train(data, out, *extra)
    except SystemExit as e:
        return str(e.code)
    raise AssertionError(f"accepted: {extra}")


def test_refusal(tmp: Path):
    data = tmp / "data"
    data.mkdir(exist_ok=True)
    (data / "manifest.json").write_text(json.dumps({"entries": []}))
    dflags = ["--depth-max-m", "1.5", "--depth-mean-m", "0.4", "--depth-std-m", "0.2", "--depth-invalid", "-2.5"]
    runs = {"r34": {"imagenet": True, **NEW["resnet34"]}, "seg": {"imagenet": True, **NEW["segformer_b1"]},
            "old": {"imagenet": True}}                                                            # old: no encoder keys
    for name, keys in runs.items():
        (tmp / name).mkdir(exist_ok=True)
        torch.save({"labels": WD.LABELS, "offset_weight": 0.0, **keys}, tmp / name / "last.pt")
    msgs = [refused(data, tmp / "r34", "--resume", "--imagenet", "--encoder", "segformer_b1", "--token-dim", "128", *dflags),
            refused(data, tmp / "seg", "--resume", "--imagenet", "--encoder", "resnet34", "--token-dim", "128", *dflags),
            refused(data, tmp / "old", "--resume", "--imagenet", "--encoder", "resnet34", "--token-dim", "128", *dflags),
            refused(data, tmp / "r34", "--resume", "--imagenet"),
            refused(data, tmp / "r34", "--resume", "--imagenet", "--encoder", "resnet34", "--token-dim", "64", *dflags),
            refused(data, tmp / "r34", "--resume", "--imagenet", "--encoder", "resnet34", "--token-dim", "128", *dflags[:-2],
                    "--depth-invalid", "-3")]
    assert "encoder=resnet34" in msgs[0] and "--encoder is segformer_b1" in msgs[0], msgs[0]
    assert "encoder=segformer_b1" in msgs[1], msgs[1]
    assert "encoder=resnet18" in msgs[2], msgs[2]
    assert "encoder=resnet34" in msgs[3] and "--encoder is resnet18" in msgs[3], msgs[3]
    assert "token_dim=128" in msgs[4], msgs[4]
    assert "depth_invalid=-2.5" in msgs[5], msgs[5]
    bad = [refused(data, tmp / "new1", "--encoder", "resnet34", "--token-dim", "128", *dflags),           # no --imagenet
           refused(data, tmp / "new2", "--encoder", "resnet34", "--imagenet", "--token-dim", "128"),     # no depth settings
           refused(data, tmp / "new3", "--depth-max-m", "1.5"),                                          # depth with resnet18
           refused(data, tmp / "new4", "--encoder", "resnet34", "--imagenet", *dflags),                  # no --token-dim
           refused(data, tmp / "new5", "--encoder", "resnet34", "--imagenet", "--token-dim", "128", *dflags[:-2],
                   "--depth-invalid", "-1")]                                                             # invalid above valid z
    assert "--imagenet" in bad[0] and "depth_stats" in bad[1] and "legacy ResNet-18" in bad[2], bad
    assert "--token-dim" in bad[3] and "smallest valid z" in bad[4], bad
    assert not any((tmp / f"new{i}").exists() for i in (1, 2, 3, 4, 5)), "refused before anything is written"
    return "resume refused for another encoder (both ways and from an old checkpoint), token_dim or depth setting: " + msgs[0]


# ------------------------------------------------------------------------------------------------------------- 8
def test_loss_smoke():
    B, S = 1, WD.H + 4
    g = torch.Generator().manual_seed(0)
    rgb, depth = pictures(B * S)
    W = {"rgb": rgb.view(B, S, 320, 320, 3), "depth": depth.view(B, S, 320, 320), "q": torch.randn(B, WD.H + 1, 7, generator=g),
         "action": torch.rand(B, WD.H, WD.ACTION_DIM, generator=g) * 2 - 1, "reward": torch.randn(B, WD.H, generator=g),
         "term": torch.zeros(B, WD.H), "valid": torch.ones(B, WD.H), "obs_valid": torch.ones(B, WD.H + 1), "v_valid": torch.ones(B, WD.H + 1),
         "a_next": torch.rand(B, WD.H + 1, WD.ACTION_DIM, generator=g) * 2 - 1, "a_next_valid": torch.ones(B, WD.H + 1),
         "done": torch.zeros(B, WD.H + 1), "probe": torch.randn(B, WD.H + 1, 4, generator=g)}
    W["offset"] = W["probe"]                          # the offset head's target (wmdata320 "offset")
    r2, d2 = pictures(4, 9)
    Bc = {"rgb": r2.view(1, 4, 320, 320, 3), "depth": d2.view(1, 4, 320, 320), "q": torch.randn(1, 7, generator=g),
          "a": torch.randn(1, 16 * 7, generator=g) * 0.1}
    out = []
    for enc in ("resnet18", "resnet34", "segformer_b1"):
        torch.manual_seed(0)
        m = WM320("standard", STATS, 0.3, imagenet=enc != "resnet18", offset_weight=1.0, gamma=0.967, **NEW.get(enc, {})).train()
        total, info = m.loss(W, Bc, 0, 5, 1.0)
        total.backward()
        gp = [p for p in m.enc.pic.parameters() if p.requires_grad]
        assert torch.isfinite(total) and all(p.grad is not None for p in gp), "every encoder parameter gets a gradient"
        out.append(f"{enc} loss {float(total):.3g}")
    return "loss + backward with windows and imitation batch, every encoder parameter has a gradient: " + ", ".join(out)


# ------------------------------------------------------------------------------------------------------------- 9
def windows(seed: int):
    """One random training window and one imitation sample, as test_loss_smoke."""
    B, S = 1, WD.H + 4
    g = torch.Generator().manual_seed(seed)
    rgb, depth = pictures(B * S, seed)
    W = {"rgb": rgb.view(B, S, 320, 320, 3), "depth": depth.view(B, S, 320, 320), "q": torch.randn(B, WD.H + 1, 7, generator=g),
         "action": torch.rand(B, WD.H, WD.ACTION_DIM, generator=g) * 2 - 1, "reward": torch.randn(B, WD.H, generator=g),
         "term": torch.zeros(B, WD.H), "valid": torch.ones(B, WD.H), "obs_valid": torch.ones(B, WD.H + 1), "v_valid": torch.ones(B, WD.H + 1),
         "a_next": torch.rand(B, WD.H + 1, WD.ACTION_DIM, generator=g) * 2 - 1, "a_next_valid": torch.ones(B, WD.H + 1),
         "done": torch.zeros(B, WD.H + 1)}
    r2, d2 = pictures(4, seed + 100)
    Bc = {"rgb": r2.view(1, 4, 320, 320, 3), "depth": d2.view(1, 4, 320, 320), "q": torch.randn(1, 7, generator=g),
          "a": torch.randn(1, 16 * 7, generator=g) * 0.1}
    return W, Bc


def test_frozen_bn(tmp: Path):
    from transformers import SegformerForSemanticSegmentation
    ade = SegformerForSemanticSegmentation.from_pretrained(E.SEGFORMER_B1, **E.SEGFORMER_B1_LOAD).decode_head.batch_norm
    keys = ("running_mean", "running_var", "num_batches_tracked")
    torch.manual_seed(0)
    m = WM320("standard", STATS, 0.3, imagenet=True, gamma=0.967, **NEW["segformer_b1"])
    m.enc.pic.init_pretrained()
    bn = m.enc.pic.seg.decode_head.batch_norm
    assert set(bn.state_dict()) == {"weight", "bias", *keys}, "the same checkpoint keys as nn.BatchNorm2d"
    for k in keys:
        assert torch.equal(getattr(bn, k), getattr(ade, k)), k
    before = {k: getattr(bn, k).clone() for k in keys}
    w0, b0 = bn.weight.detach().clone(), bn.bias.detach().clone()
    x = torch.randn(3, 256, 80, 80) * 2 + 0.5                       # far from the ADE20K statistics
    m.eval()
    with torch.no_grad():
        y_eval = bn(x)
    m.train()
    with torch.no_grad():
        y_train = bn(x)
    assert bn.training and torch.equal(y_train, y_eval), "train() and eval() give the same output"
    want = (x - ade.running_mean[:, None, None]) / torch.sqrt(ade.running_var[:, None, None] + ade.eps) * ade.weight[:, None, None] + ade.bias[:, None, None]
    assert torch.allclose(y_eval, want, atol=1e-4), "normalised with the ADE20K statistics"
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    for step in range(3):                                           # training steps in train(): two forward calls each
        W, Bc = windows(step)
        opt.zero_grad(set_to_none=True)
        total, _ = m.loss(W, Bc, step, 5, 1.0)
        total.backward()
        assert bn.weight.grad is not None and bn.bias.grad is not None and bn.weight.grad.abs().sum() > 0 and bn.bias.grad.abs().sum() > 0
        opt.step()
        for k in keys:
            assert torch.equal(getattr(bn, k), before[k]), (step, k)
    assert not torch.equal(bn.weight.detach(), w0) and not torch.equal(bn.bias.detach(), b0), "the affine parameters are trained"
    m.eval()
    p = tmp / "frozen_bn.pt"
    torch.save(ckpt_of(m, "standard", True, 0.0, {**ENCODER_DEFAULTS, **NEW["segformer_b1"]}), p)
    for ld in (load(str(p), "cpu"), WMAgent(str(p), "cpu").m):
        lb = ld.enc.pic.seg.decode_head.batch_norm
        assert isinstance(lb, E.FrozenStatsBatchNorm2d)
        for k in keys:
            assert torch.equal(getattr(lb, k), before[k]), k
        assert torch.equal(lb.weight, bn.weight) and torch.equal(lb.bias, bn.bias)
        ld.train()
        with torch.no_grad():
            assert torch.equal(lb(x), bn(x))
    return ("running mean / var and num_batches_tracked = ADE20K's, bit-identical after 3 training steps (6 forward calls in "
            "train()); train() output = eval() output; weight and bias get gradients and change; statistics kept by wm320.load and WMAgent")


# ------------------------------------------------------------------------------------------------------------ 10
def fast_copy(root: Path, simulator: str | None, depths: list, condition: str = "train") -> Path:
    """A one-trajectory fast copy (manifest + npz with the depth pictures only), port 00015 (a training grasp: its
    stored condition is "train"; AutoMate's split is the stored condition, demo320.split_of_entry)."""
    from wm.tdmpc import demo320 as D
    root.mkdir(parents=True, exist_ok=True)
    blobs = [D.encode_depth(d) for d in depths]
    off = np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64)
    np.savez(root / "t.npz", depth_png=np.concatenate(blobs), depth_offsets=off)
    e = {"path": "t.npz", "n_images": len(depths), "steps": 40, "port": "00015", "grasp": "00015_g01", "condition": condition,
         "kind": "insert", "outcome": "inserted", "pitch_deg": 1.0, "grasp_pos_m": 0.001, "pin_length_m": 0.02,
         **({"simulator": simulator, "expert_completed": True, "automate_success_any": True} if simulator else {})}
    split = {"name": "automate_v1", "file": "split_automate_v1.json", "sha256": "f" * 64, "plan_sha": "p"} if simulator else None
    (root / "manifest.json").write_text(json.dumps({"entries": [e], **({"split": split} if split else {})}))
    return root


def test_automate_cli(tmp: Path):
    """Encoder command line on AutoMate data (encoder-1/2/3/10): no silent legacy defaults, the invalid-depth default and
    its check, the placeholders refused, the depth statistics of the training split, 320 x 320 only."""
    import importlib.util
    from wm.tdmpc import policy320 as P
    g = np.random.default_rng(0)
    depths = [np.where(g.random((320, 320)) < 0.1, 0, g.integers(700, 4500, (320, 320))).astype(np.uint16) for _ in range(3)]
    data = fast_copy(tmp / "am", "automate", depths)
    dflags = ["--depth-max-m", "0.4", "--depth-mean-m", "0.3", "--depth-std-m", "0.06"]
    msgs = [refused(data, tmp / "a1", "--imagenet"),                                                      # no --encoder
            refused(data, tmp / "a2", "--encoder", "resnet18"),                                            # dropped
            refused(data, tmp / "a3", "--encoder", "segformer_b1", "--imagenet", *dflags)]                 # no --token-dim
    assert "state --encoder" in msgs[0] and "dropped" in msgs[1] and "--token-dim" in msgs[2], msgs
    # the same omitted --encoder on ManiSkill data stays the legacy resnet18 (refused later, for its depth flags only)
    ms = fast_copy(tmp / "ms", None, depths)
    assert "legacy ResNet-18" in refused(ms, tmp / "m1", "--depth-max-m", "0.4")
    # PictureTokens: depth_invalid default (0 - mean) / std; above the smallest valid z, placeholders, missing: refused
    pt = PictureTokens("standard", imagenet=True, encoder="resnet34", token_dim=8, depth_max_m=0.4, depth_mean_m=0.3, depth_std_m=0.06)
    assert abs(pt.depth_invalid - (0 - 0.3) / 0.06) < 1e-12 and pt.depth_invalid < (P.DEPTH_MIN_VALID_M - 0.3) / 0.06
    for kw in ({"depth_invalid": -1.0}, {"depth_max_m": P.DEPTH_MAX_M, "depth_mean_m": P.DEPTH_MEAN_M, "depth_std_m": P.DEPTH_STD_M,
                                           "depth_invalid": P.DEPTH_INVALID}, {"depth_mean_m": None}):
        try:
            PictureTokens("standard", imagenet=True, encoder="resnet34", token_dim=8,
                          **{"depth_max_m": 0.4, "depth_mean_m": 0.3, "depth_std_m": 0.06, **kw})
            raise AssertionError(f"accepted {kw}")
        except ValueError:
            pass
    try:                                                                                              # 320 x 320 only
        pt.eval()(torch.zeros(1, 3, 160, 160, dtype=torch.uint8), torch.zeros(1, 160, 160, dtype=torch.int16))
        raise AssertionError("accepted 160 x 160")
    except ValueError as e:
        assert "320 x 320" in str(e)
    # depth statistics of the training split (scripts/automate/depth_stats.py) against numpy
    spec = importlib.util.spec_from_file_location("depth_stats", HERE.parent / "scripts" / "automate" / "depth_stats.py")
    DS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DS)
    st = DS.compute(str(data), 0.4)
    v = np.minimum(np.concatenate([d[d > 0] for d in depths]).astype(np.float64) * 1e-4, 0.4)
    assert st["n_valid_pixels"] == v.size and abs(st["depth_mean_m"] - v.mean()) < 1e-12 and abs(st["depth_std_m"] - v.std()) < 1e-9
    assert abs(st["depth_invalid"] - (0 - v.mean()) / v.std()) < 1e-9 and st["depth_invalid"] < st["smallest_valid_z"]
    # r10-data-01: a distribution-only run (prep_automate_data.sh with D_max none) leaves no statistics file, and
    # train_wm320 refuses until depth_stats.py has run with a D_max
    old, sys.argv = sys.argv, ["depth_stats.py", "--data", str(data), "--distribution-only"]
    try:
        DS.main()
    finally:
        sys.argv = old
    assert (data / "depth_distribution.json").exists() and not (data / "depth_stats.json").exists()
    # r11-data-01: a run with a D_max deletes an earlier depth_stats.json before its checks, so a refused run (here:
    # check_depth_max, and check_depth_settings inside compute) leaves no statistics of the earlier D_max
    keep_check = DS.check_depth_settings
    for dmax, patched, text in (("5.0", False, "must lie in"), ("0.4", True, "unusable statistics")):
        (data / "depth_stats.json").write_text(json.dumps({**st, "depth_max_m": 0.3}))
        if patched:
            DS.check_depth_settings = lambda *a: "refused for the test"
        old, sys.argv = sys.argv, ["depth_stats.py", "--data", str(data), "--depth-max-m", dmax]
        try:
            DS.main()
            raise AssertionError(f"D_max {dmax} accepted")
        except SystemExit as e:
            assert text in str(e.code), e.code
        finally:
            sys.argv, DS.check_depth_settings = old, keep_check
        assert not (data / "depth_stats.json").exists(), f"the earlier depth_stats.json survived a refused D_max {dmax}"
    m = refused(data, tmp / "a3c", "--encoder", "resnet34", "--imagenet", "--token-dim", "8")
    assert "needs the depth statistics" in m and not (tmp / "a3c").exists(), m
    (data / "depth_stats.json").write_text(json.dumps({**st, "train_files_sha256": "another split"}))
    m = refused(data, tmp / "a4", "--encoder", "resnet34", "--imagenet", "--token-dim", "8")
    assert "another training split" in m, m
    (data / "depth_stats.json").write_text(json.dumps({**st, "split": {**st["split"], "sha256": "0" * 64}}))
    m = refused(data, tmp / "a4b", "--encoder", "resnet34", "--imagenet", "--token-dim", "8")
    assert "carry the split" in m, m                                       # a statistics file of another split identity
    (data / "depth_stats.json").write_text(json.dumps(st))
    m = refused(data, tmp / "a5", "--encoder", "resnet34", "--imagenet", "--token-dim", "8", "--depth-max-m", "0.5",
                "--depth-mean-m", "0.3", "--depth-std-m", "0.06")
    assert "statistics file only" in m, m                                   # AutoMate: no hand-typed depth flags
    m = refused(data, tmp / "a6", "--encoder", "resnet34", "--imagenet", "--token-dim", "8")
    assert "selection grasps" in m, m                                       # the right file; no selection / held-out set
    # an undecided AutoMate split is refused by depth_stats and train_wm320, before anything is written
    un = fast_copy(tmp / "un", "automate", depths, condition="unassigned")
    for f in (lambda: DS.compute(str(un), 0.4), lambda: run_train(un, tmp / "a7", "--encoder", "resnet34", "--imagenet", "--token-dim", "8")):
        try:
            f()
            raise AssertionError("an undecided split was accepted")
        except SystemExit as e:
            assert "no decided split" in str(e.code), e.code
    assert not (tmp / "a7").exists()
    # ManiSkill data: flags next to a statistics file must agree with it, and its split hash is checked too
    sm = DS.compute(str(ms), 0.4)
    (ms / "depth_stats.json").write_text(json.dumps(sm))
    mflags = ["--depth-max-m", "0.4", "--depth-mean-m", repr(sm["depth_mean_m"]), "--depth-std-m", repr(sm["depth_std_m"])]
    m = refused(ms, tmp / "m2", "--encoder", "resnet34", "--imagenet", "--token-dim", "8", "--depth-max-m", "0.5", *mflags[2:])
    assert "differs from" in m, m
    (ms / "depth_stats.json").write_text(json.dumps({**sm, "train_files_sha256": "another split"}))
    m = refused(ms, tmp / "m3", "--encoder", "resnet34", "--imagenet", "--token-dim", "8", *mflags)
    assert "another training split" in m, m
    return (f"AutoMate data refuses a missing --encoder, resnet18 and a missing --token-dim; invalid depth default "
            f"{pt.depth_invalid:.3f} = (0 - mean) / std, invalid above the smallest valid z / placeholders / missing statistics "
            f"refused; 160 x 160 refused; no statistics file after --distribution-only refused; depth_stats mean {st['depth_mean_m']:.4f} std {st['depth_std_m']:.4f} = numpy; a stats file of "
            "another split refused; AutoMate: depth flags, an undecided split and a missing selection / held-out set refused; "
            "ManiSkill: disagreeing flags and a stats file of another split refused")


# ------------------------------------------------------------------------------------------------------------ 11
def raw_automate(path: Path, meta: dict, T: int = 16, seed: int = 0):
    """A raw AutoMate trajectory as collect.py stores it (real JPEG pictures, depth, joints, FK hands, actions, the
    eval_* truth arrays) with its meta."""
    import io
    from PIL import Image
    g = np.random.default_rng(seed)
    steps = list(range(0, T + 1, 4))
    blobs = []
    for _ in steps:
        buf = io.BytesIO()
        Image.fromarray(g.integers(0, 255, (320, 320, 3)).astype(np.uint8)).save(buf, format="JPEG", quality=95)
        blobs.append(np.frombuffer(buf.getvalue(), np.uint8))
    hand = np.tile(np.eye(4), (T + 1, 1, 1))
    hand[:, 2, 3] = np.linspace(0.03, 0.0, T + 1)                       # down the axis onto H* = identity
    qpos = g.normal(size=(T + 1, 9)) * 0.05
    np.savez(path, meta=np.asarray(json.dumps({**meta, "steps": T})), img_step=np.asarray(steps, np.int64),
             depth=g.integers(700, 4500, (len(steps), 320, 320)).astype(np.uint16), rgb_jpeg=np.concatenate(blobs),
             rgb_offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64), qpos=qpos, hand=hand,
             action=qpos[1:, :7] + g.normal(size=(T, 7)) * 0.01, eval_rel=g.normal(size=(T + 1, 7)),
             eval_tip=np.zeros((T + 1, 9), np.float32))


def sklearn_stand_in() -> str | None:
    """A stand-in sklearn.metrics.roc_auc_score (Mann-Whitney rank formula) when scikit-learn is not installed."""
    import importlib.util
    import types
    if importlib.util.find_spec("sklearn") is not None:
        return None
    from scipy.stats import rankdata

    def roc_auc_score(y, s):
        y, r = np.asarray(y, bool), rankdata(np.asarray(s, np.float64))
        n1, n0 = int(y.sum()), int((~y).sum())
        return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    try:                                              # transformers probes for sklearn at import: let it see the truth
        import transformers  # noqa: F401
    except ImportError:
        pass
    from importlib.machinery import ModuleSpec
    sk, met = types.ModuleType("sklearn"), types.ModuleType("sklearn.metrics")
    sk.__spec__, met.__spec__ = ModuleSpec("sklearn", None), ModuleSpec("sklearn.metrics", None)
    met.roc_auc_score, sk.metrics = roc_auc_score, met
    sys.modules["sklearn"], sys.modules["sklearn.metrics"] = sk, met
    return "scikit-learn missing: a stand-in roc_auc_score was used for this test"


def test_automate_train(tmp: Path):
    """r3-data-07: train_wm320 past its refusals on AutoMate-shaped data (two optimiser steps, one validation)."""
    import importlib.util
    from wm.tdmpc import demo320 as D
    note = sklearn_stand_in()
    raw, data = tmp / "am_raw", tmp / "am_fast"
    truth = {"eval_socket_pose": [[1, 0, 0, 0.03]], "truth": {"depth_left_mm": 0.1}, "episode_seed": 7, "seed_offset": 3,
             "rng": {"entropy": 0}, "peak": {"plug_socket_N": 3.0}, "stages": [], "start_error_mm": 0.1, "depth_left_mm": 0.2,
             "tip_centre_m": [0, 0, -0.02], "disassembly_dist_m": 0.03, "file": "t01_insert.npz", "physics_device": "cpu",
             "success": {"inserted_any_step": True}, "eval_tip_columns": ["success"], "wall_s": 1.0}
    conds = ["train"] * 3 + ["select"] * 20 + ["heldout_grasp", "heldout_shape", "test"]
    entries = []
    for i, cond in enumerate(conds):
        grasp = f"{15 + i // 3:05d}_g{i % 3:02d}"
        meta = {"grasp": grasp, "port": grasp[:5], "condition": cond, "kind": "insert", "outcome": "inserted",
                "expert_completed": True, "automate_success_any": True, "idx": 1, "H_star": np.eye(4).tolist(),
                "axis": [0.0, 0.0, -1.0], "pin_length_m": 0.02, "pitch_deg": 1.0, "grasp_pos_m": 0.001, "spin_deg": 2.0,
                "u": {"dtheta": 0.3}, "simulator": "automate", "symmetry_n": 4, "switches": {"socket_draw": "per_reset"},
                "start": {"g0": 0.01}, "assembly_id": grasp[:5], "config_sha": "c", "plan_sha": "p", **truth}
        src = raw / grasp[:5] / grasp / "t01_insert.npz"
        src.parent.mkdir(parents=True, exist_ok=True)
        raw_automate(src, meta, seed=i)
        rel = str(src.relative_to(raw))
        if cond == "test":
            try:
                D.convert(src, data / rel, 0.02, rel)
                raise AssertionError("a test trajectory was converted")
            except ValueError as e:
                assert "condition 'test'" in str(e)
            continue
        entries.append(D.convert(src, data / rel, 0.02, rel))
    split = {"name": "automate_v1", "file": "split_automate_v1.json", "sha256": "a" * 64, "plan_sha": "p"}
    (data / "manifest.json").write_text(json.dumps({"format": D.FORMAT, "source": str(raw), "skipped": {}, "entries": entries}))
    spec = importlib.util.spec_from_file_location("depth_stats", HERE.parent / "scripts" / "automate" / "depth_stats.py")
    DS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DS)
    (data / "depth_stats.json").write_text(json.dumps(DS.compute(str(data), 0.5)))
    args = ("--encoder", "resnet34", "--imagenet", "--token-dim", "8", "--steps", "2", "--save-every", "1",
            "--eval-every", "1000", "--eval-windows", "4", "--keep-every", "1000")
    m = refused(data, tmp / "t11a", *args)                       # no split identity in the manifest
    assert "split identity" in m, m
    man = {"format": D.FORMAT, "source": str(raw), "skipped": {}, "entries": entries, "split": split}
    (data / "manifest.json").write_text(json.dumps(man))
    (data / "depth_stats.json").write_text(json.dumps(DS.compute(str(data), 0.5)))
    m = refused(data, tmp / "t11b", *args)                       # r5-data-01: no completeness record
    assert "no completeness record" in m and "--allow-partial" in m, m
    comp = {"planned": 26, "with_grasp_json": 26, "skipped_assemblies": {}, "partial": False, "problems": {}}
    (data / "manifest.json").write_text(json.dumps({**man, "completeness": {**comp, "partial": True,
                                                                            "problems": {"unfinished": "x"}}}))
    m = refused(data, tmp / "t11c", *args)                       # a partial collection
    assert "running or unfinished" in m, m
    assert not (tmp / "t11b").exists() and not (tmp / "t11c").exists()
    code = {"root": "/x/code", "md5": {"src/wm/tdmpc/demo320.py": "d" * 32}, "earlier_writers": []}   # r6-data-04
    (data / "manifest.json").write_text(json.dumps({**man, "completeness": comp, "code": code}))
    (data / "depth_stats.json").write_text(json.dumps(DS.compute(str(data), 0.5)))
    run_train(data, tmp / "t11", *args)
    ck = torch.load(tmp / "t11" / "last.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 2 and ck["encoder"] == "resnet34" and ck["token_dim"] == 8, {k: ck.get(k) for k in ("step", "encoder")}
    pv = ck["provenance"]
    assert pv["split"] == split and Path(pv["wm_file"]).is_relative_to((HERE.parent / "src").resolve()), pv
    import hashlib
    assert pv["depth_stats"]["sha256"] == hashlib.sha256((data / "depth_stats.json").read_bytes()).hexdigest()
    assert pv["completeness"] == comp and pv["allow_partial"] is False and pv["fast_copy_code"] == code, pv
    assert ck["offset_label"] == pv["offset_label"] == WD.OFFSET_GAP, "AutoMate: the offset head learns the gap (decision 6)"
    assert (tmp / "t11" / "step2.pt").exists() and (tmp / "t11" / "DONE.json").exists()
    log = [json.loads(x) for x in (tmp / "t11" / "log.jsonl").read_text().splitlines()]
    start = [r for r in log if r.get("event") == "start"][0]
    assert start["trajectories"]["train"] == 3 and start["trajectories"]["select"] == 20, start["trajectories"]
    assert start["trajectories"]["heldout"] == 2 and start["trajectories"]["heldout_shape"] == 1, start["trajectories"]
    val = [r for r in log if r.get("event") == "val"][0]
    assert {"select", "heldout", "heldout_shape", "heldout_grasp"} <= set(val), sorted(val)
    for name in ("select", "heldout", "heldout_shape", "heldout_grasp"):   # r11-integration-01: coverage logged
        assert val[name]["face_items_kept"] == [0, 0], (name, val[name].get("face_items_kept"))   # no contact pushes here
    # r9-data-03 (decision 6 end to end): AutoMate data with the offset head on, 2 steps; the gap label of WindowData
    # reaches WM320.loss. The training log writes its loss rows every 250 steps only, so offset_loss is read from
    # WM320.loss on a WindowData batch of the trained checkpoint (the call train_wm320 makes at every step)
    run_train(data, tmp / "t11o", *args, "--offset-weight", "1.0")
    cko = torch.load(tmp / "t11o" / "last.pt", map_location="cpu", weights_only=False)
    assert cko["step"] == 2 and cko["offset_weight"] == 1.0, {k: cko.get(k) for k in ("step", "offset_weight")}
    assert cko["offset_label"] == cko["provenance"]["offset_label"] == WD.OFFSET_GAP, cko["provenance"]
    from wm.tdmpc.wm320 import WindowData, window_items
    mo = load(str(tmp / "t11o" / "last.pt"), device="cpu")
    assert mo._offset is not None and mo.offset_weight == 1.0
    tr_ents = [e for e in D.load_entries(str(data)) if e["split"] == "train" and e["steps"] >= WD.EXECUTE]
    wd = WindowData(str(data), tr_ents, window_items(tr_ents), cko["stats"], cko["shaping"], cko["cfg"]["gamma"])
    item = wd[0]
    with np.load(data / tr_ents[0]["path"]) as z:
        meta0 = json.loads(str(z["meta"]))
    assert meta0["simulator"] == "automate", meta0.get("simulator")
    off, probe = item["offset"].numpy(), item["probe"].numpy()
    assert np.array_equal(off[:, 0], (-probe[:, 0].astype(np.float64) - 1000.0 * meta0["pin_length_m"]).astype(np.float32)), \
        (off[:, 0], probe[:, 0])
    assert np.array_equal(off[:, 1:], probe[:, 1:]), "lateral x, y and tilt as probe"
    batch = torch.utils.data.default_collate([item])
    mo.train()
    total, info = mo.loss(batch, None, 0, 5000, 0.0)
    assert np.isfinite(info["offset_loss"]) and info["offset_loss"] > 0 and np.isfinite(total.item()), info
    # r4-data-08 (a): an AutoMate checkpoint is not resumed on a fast copy of another split identity
    res = tmp / "t11r"
    res.mkdir()
    shutil.copy(tmp / "t11" / "last.pt", res / "last.pt")
    man = {"format": D.FORMAT, "source": str(raw), "skipped": {}, "entries": entries}
    (data / "manifest.json").write_text(json.dumps({**man, "split": {**split, "sha256": "b" * 64},
                                                    "completeness": {**comp, "partial": True}}))
    (data / "depth_stats.json").write_text(json.dumps(DS.compute(str(data), 0.5)))
    m = refused(data, res, *args, "--resume", "--allow-partial")   # --allow-partial passes the completeness check
    assert "was trained on the split" in m, m
    (data / "manifest.json").write_text(json.dumps({**man, "split": split, "completeness": comp}))
    (data / "depth_stats.json").write_text(json.dumps(DS.compute(str(data), 0.5)))
    return (f"2 optimiser steps on 3 training trajectories (resnet34, token_dim 8), validation over select / heldout / "
            f"heldout_shape / heldout_grasp; provenance with this tree's wm, the split identity, the stats sha256, the "
            f"completeness and the converter's code record; face_items_kept logged per split; with --offset-weight 1.0: offset_weight 1.0, offset_label gap_mm, "
            f"WindowData offset[:, 0] = -probe[:, 0] - 1000 L and WM320.loss offset_loss {info['offset_loss']:.4f} > 0; "
            f"test trajectory not converted; no split identity, no completeness record and a partial "
            f"collection refused (--allow-partial passes); resume on another split refused"
            + (f"; NOTE {note}" if note else ""))


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp(prefix="encoders_automate_"))
    torch.set_num_threads(int(os.environ.get("TEST_THREADS", "4")))
    try:
        print("1 test_legacy_identity OK:", test_legacy_identity(tmp), flush=True)
        print("2 test_shapes OK:", test_shapes(), flush=True)
        print("3 test_zero_depth OK:", test_zero_depth(), flush=True)
        print("4 test_pretrained OK:", test_pretrained(), flush=True)
        print("5 test_locality OK:", test_locality(), flush=True)
        print("6 test_round_trip OK:", test_round_trip(tmp), flush=True)
        print("7 test_refusal OK:", test_refusal(tmp), flush=True)
        print("8 test_loss_smoke OK:", test_loss_smoke(), flush=True)
        print("9 test_frozen_bn OK:", test_frozen_bn(tmp), flush=True)
        print("10 test_automate_cli OK:", test_automate_cli(tmp), flush=True)
        print("11 test_automate_train OK:", test_automate_train(tmp), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
