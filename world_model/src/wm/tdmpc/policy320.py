"""The flow-matching policy with its own ResNet at 320 px: model, data loader, validation, closed-loop agent.

Input: 4 pictures 0.4 s apart (320 x 320 RGB + depth) -> 100 tokens of 32 numbers each, and the 7 measured joints.
Output: 16 joint targets relative to the measured joints, 8 executed (models/policy.py's LatentFMPolicy).
Stem: "standard" = the usual ResNet entry (7x7 convolution with step 2, then max-pool; last map 10 x 10);
      "nopool"   = the same without the max-pool (the trunk works at twice the resolution; its 20 x 20 map is
                   averaged 2 x 2 to the same 100 tokens).

The samples are the old policy's (git 4a3a6d3, data/datasets.py LatentPolicyDataset): every picture of every
successful expert trajectory (reference, insert, perturbed), the terminal picture included, drawn independently
without replacement; pictures [j-6, j-4, j-2, j]; joints and the next 16 targets from picture j's control step.
The data logic is in demo320.py (numpy + PIL), tested there against the old functions.

What differs from the old policy, on purpose (see scripts/train/train_policy320.py for the training side):
  encoder    a ResNet-18 (group norm, RGB + depth) per picture instead of frozen DINOv2 latents, trained with the
             policy; from scratch, or (imagenet=True, the user's decision 2026-11-18) starting from torchvision's
             ImageNet ResNet-18 (IMAGENET1K_V1): its convolutions copied, the first layer's depth input from the mean of
             its RGB weights, the group norms fresh (as Diffusion Policy replaces batch norm by group norm), RGB
             normalised with ImageNet's mean and std. Nothing is frozen.
  offset     optional (offset=True, the user's decision 2026-11-18): a training-only head (two layers on the 100 tokens
             of each context picture) reads the hand's offset from H* (forward kinematics: along the insertion axis,
             two lateral [mm], tilt [deg]; wmdata320.decision_labels' "probe"); its loss trains the ResNet to see the
             offset. Nothing at test time uses it (the agent never calls it); validation reports its error.
  joints     embedded in float32 (models/policy.py); under bfloat16 the old code rounded joints of 2-3 rad by up to 8 mrad
  test time  the agent passes the rendered picture through the store's JPEG codec, as every training picture was
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..models.policy import ACTION_DIM, CHUNK, LatentFMPolicy
from . import demo320 as D
from . import wmdata320 as WD
from .demo320 import CONTEXT, POLICY_KINDS, STRIDE_IMG  # noqa: F401  (re-exported: joint320.py imports them from here)

TOKENS, TOKEN_DIM = 100, 32
OFFSET_SCALE = (10.0, 10.0, 10.0, 5.0)           # [mm, mm, mm, deg] -> offsets of about 1 for the read-out loss
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _amp(x: torch.Tensor, on: bool = True):
    """bfloat16 autocast on CUDA; off on a CPU (tests)."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=on and x.is_cuda)


class PictureTokens(nn.Module):
    """One picture (RGB + depth, 320 x 320) -> 100 tokens of 32 numbers. As FrameNet of wm/tdmpc/model.py up to its map."""

    def __init__(self, stem: str = "standard", amp: bool = True, imagenet: bool = False):
        super().__init__()
        assert stem in ("standard", "nopool")
        self.imagenet = imagenet                                          # RGB normalisation of the ImageNet weights
        self.register_buffer("rgb_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)   # not in the
        self.register_buffer("rgb_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)     # checkpoint
        net = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(8, c))
        self.stem = nn.Sequential(nn.Conv2d(4, 64, 7, 2, 3, bias=False), nn.GroupNorm(8, 64), nn.ReLU())
        self.trunk = nn.Sequential(*([net.maxpool] if stem == "standard" else []), net.layer1, net.layer2, net.layer3, net.layer4)
        self.squeeze = nn.Conv2d(512, TOKEN_DIM, 1)
        self.pool = nn.Identity() if stem == "standard" else nn.AvgPool2d(2)
        self.amp = amp

    def init_imagenet(self) -> int:
        """Start from torchvision's ImageNet ResNet-18 (IMAGENET1K_V1): every convolution of the trunk copied, the
        first layer's RGB weights copied and its depth weights set to their mean; group norms keep their fresh
        values. For a new training only (a checkpoint carries its own weights). Returns the number of tensors copied."""
        if not self.imagenet:
            raise ValueError("built without imagenet=True: its RGB normalisation would not match the ImageNet weights")
        src = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        with torch.no_grad():
            w = src.conv1.weight
            self.stem[0].weight.copy_(torch.cat([w, w.mean(1, keepdim=True)], 1))
        layers = [m for m in self.trunk if isinstance(m, nn.Sequential)]
        assert len(layers) == 4
        n = 1
        for k, tgt in enumerate(layers, 1):
            sd = {name: v for name, v in getattr(src, f"layer{k}").state_dict().items()
                  if name.endswith(("conv1.weight", "conv2.weight", "downsample.0.weight"))}
            res = tgt.load_state_dict(sd, strict=False)
            left = [key for key in res.missing_keys if not (".bn1." in f".{key}" or ".bn2." in f".{key}" or "downsample.1." in key)]
            if res.unexpected_keys or left:
                raise RuntimeError(f"layer{k}: unexpected {res.unexpected_keys}, convolutions not copied {left}")
            n += len(sd)
        return n

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (N, 3, 320, 320) uint8, depth (N, 320, 320) in 0.1 mm -> (N, 100, 32)."""
        if self.imagenet:
            c = (rgb.float() / 255.0 - self.rgb_mean) / self.rgb_std
        else:
            c = rgb.float() / 255.0 - 0.5
        x = torch.cat([c, (depth.float() * 1e-3).clamp(0, 3).unsqueeze(1)], 1)
        x = self.stem(x.contiguous(memory_format=torch.channels_last))     # float32: 0.1 mm of depth must survive
        with _amp(x, self.amp):
            x = self.pool(self.squeeze(self.trunk(x)))
        return x.float().flatten(2).transpose(1, 2)


class ResnetFMPolicy(nn.Module):
    def __init__(self, stem: str = "standard", d: int = 256, layers: int = 4, heads: int = 8, hidden: int = 512, imagenet: bool = False,
                 offset: bool = False):
        super().__init__()
        self.enc = PictureTokens(stem, imagenet=imagenet)
        self.head = LatentFMPolicy(latent_ch=TOKEN_DIM, context=CONTEXT, d=d, layers=layers, heads=heads, hidden=hidden, n_tokens=TOKENS)
        self.off = None
        if offset:
            with torch.random.fork_rng(devices=[]):         # the other parts start from the same weights as without it
                self.off = nn.Sequential(nn.Linear(TOKENS * TOKEN_DIM, 256), nn.SiLU(), nn.Linear(256, 4))
        self.register_buffer("off_scale", torch.tensor(OFFSET_SCALE), persistent=False)

    def tokens(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """rgb (B, 4, 320, 320, 3) uint8, depth (B, 4, 320, 320) -> (B, 4, 100, 32)."""
        B, C = depth.shape[:2]
        t = self.enc(rgb.flatten(0, 1).permute(0, 3, 1, 2), depth.flatten(0, 1))
        return t.view(B, C, *t.shape[1:])

    def loss(self, rgb, depth, q, a1) -> torch.Tensor:
        z = self.tokens(rgb, depth)
        with _amp(z):
            return self.head.loss(z, q, a1)

    def losses(self, rgb, depth, q, a1, off: torch.Tensor | None):
        """The flow-matching loss and, with the offset head and labels `off` (B, 4, 4), the read-out loss over the 4
        context pictures (smooth L1 on OFFSET_SCALE units); None without them. One pass through the ResNet."""
        z = self.tokens(rgb, depth)
        with _amp(z):
            fl = self.head.loss(z, q, a1)
        if self.off is None or off is None:
            return fl, None
        return fl, F.smooth_l1_loss(self.off(z.flatten(2)), off / self.off_scale)

    @torch.no_grad()
    def read_offset(self, rgb, depth) -> torch.Tensor:
        """The offset head's reading of every context picture, (B, 4, 4) [mm, mm, mm, deg]. Validation only."""
        return self.off(self.tokens(rgb, depth).flatten(2)) * self.off_scale

    @torch.no_grad()
    def sample(self, rgb, depth, q, steps: int = 10, n: int = 1) -> torch.Tensor:
        z = self.tokens(rgb, depth)
        with _amp(z):
            return self.head.sample(z, q, steps=steps, n=n).float()


def shift(rgb: torch.Tensor, depth: torch.Tensor, px: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Random shift of up to `px` pixels, one per sample (the same for all its pictures), edges repeated."""
    if not px:
        return rgb, depth
    B, H = rgb.shape[0], rgb.shape[2]
    ar = torch.arange(H, device=rgb.device)
    s = torch.randint(-px, px + 1, (B, 2), device=rgb.device)
    iy = (ar[None] + s[:, :1]).clamp(0, H - 1)
    ix = (ar[None] + s[:, 1:]).clamp(0, H - 1)
    b = torch.arange(B, device=rgb.device)[:, None, None]
    return rgb[b, :, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2, 4), depth[b, :, iy[:, :, None], ix[:, None, :]].permute(0, 3, 1, 2)


def collate_cat(items: list[dict]) -> dict:
    return {k: torch.cat([it[k] for it in items]) for k in items[0]}


class PolicyData(torch.utils.data.Dataset):
    """One item = one policy sample (demo320.policy_sample) of the fast copy. `items`: rows (entry index, picture
    index); default every picture of every trajectory in `entries`."""

    def __init__(self, root: str | Path, entries: list[dict], items: np.ndarray | None = None, offset: bool = False):
        self.paths = [str(Path(root) / e["path"]) for e in entries]
        self.items = D.policy_items(entries) if items is None else np.asarray(items, np.int64)
        self.offset = offset

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e, j = self.items[i]
        s = D.policy_sample(self.paths[e], int(j))
        if self.offset:
            s["off"] = context_offsets(self.paths[e], int(j))
        return {k: torch.from_numpy(v) for k, v in s.items()}


def context_offsets(path: str | Path, j: int) -> np.ndarray:
    """(4, 4) float32: for each context picture of sample j, the hand's offset from H* by forward kinematics
    (wmdata320.decision_labels' "probe": along the axis, lateral x, lateral y [mm], tilt [deg])."""
    with np.load(path) as z:
        meta, img, hand = json.loads(str(z["meta"])), z["img_step"], z["hand"]
    return WD.decision_labels(meta, hand.astype(np.float64), np.asarray(img[D.context_indices(j)], np.int64))["probe"]


def policy_loader(root, entries, batch: int, workers: int, seed: int = 0, shuffle: bool = True, items=None, offset: bool = False):
    """Training: independent samples, shuffled without replacement (a seeded generator), incomplete batches dropped.
    Validation (shuffle=False): the given items in their order, all of them."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(PolicyData(root, entries, items, offset), batch_size=batch, shuffle=shuffle, generator=g if shuffle else None,
                                       drop_last=shuffle, num_workers=workers, persistent_workers=shuffle and workers > 0,
                                       prefetch_factor=2 if workers else None)


@torch.no_grad()
def evaluate_policy(model: ResnetFMPolicy, loader, mean: torch.Tensor, std: torch.Tensor, device, seed: int = 1234) -> dict:
    """Flow-matching loss and the error of one sampled chunk against the expert's [mrad], over the loader's samples.
    The random numbers are fixed (and the generator restored afterwards), so that two validations are comparable."""
    was, cpu = model.training, torch.random.get_rng_state()
    gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    model.eval()
    torch.manual_seed(seed)
    fl, err, n = 0.0, 0.0, 0
    oe = []                                          # offset read-out errors of the current picture: along, lateral, tilt
    for b in loader:
        rgb, depth, q, a = (b[k].to(device) for k in ("rgb", "depth", "q", "a"))
        fl += float(model.loss(rgb, depth, q, (a - mean) / std)) * len(a)
        err += float((model.sample(rgb, depth, q)[:, 0] * std + mean - a).abs().mean()) * 1000 * len(a)
        n += len(a)
        if model.off is not None and "off" in b:
            e = (model.read_offset(rgb, depth)[:, -1] - b["off"][:, -1].to(device)).float()
            oe.append(torch.stack([e[:, 0].abs(), e[:, 1:3].norm(dim=-1), e[:, 3].abs()], 1).cpu())
    torch.random.set_rng_state(cpu)
    if gpu is not None:
        torch.cuda.set_rng_state_all(gpu)
    model.train(was)
    out = {"flow_loss": fl / max(n, 1), "action_err_mrad": err / max(n, 1), "samples": n}
    if oe:
        m = torch.cat(oe).mean(0).tolist()
        out.update(offset_along_err_mm=m[0], offset_lateral_err_mm=m[1], offset_tilt_err_deg=m[2])
    return out


class Policy320Agent:
    """Closed loop: reads the recorder's 320 x 320 picture (`frame`) and the measured joints. `jpeg_roundtrip`: the
    picture goes through the store's JPEG codec first, as every training picture did."""

    def __init__(self, policy_ckpt: str, device: str = "cuda", flow_steps: int = 10, jpeg_roundtrip: bool = True):
        self.device = torch.device(device)
        p = torch.load(policy_ckpt, map_location="cpu", weights_only=False)
        self.model = ResnetFMPolicy(**p["model"])
        self.model.load_state_dict(p["ema"])
        self.model.to(self.device).eval()
        self.mean, self.std = torch.tensor(p["mean"], device=self.device), torch.tensor(p["std"], device=self.device)
        self.flow_steps, self.jpeg, self.step, self.context = flow_steps, jpeg_roundtrip, p.get("step"), deque(maxlen=CONTEXT)

    def reset(self, first=None, seed: int = 0) -> bool:
        self.context.clear()
        return True

    @torch.no_grad()
    def act(self, obs, n_samples: int = 1) -> np.ndarray:
        """Joint targets (n_samples, CHUNK, 7), absolute. `obs`: collect.branch.FullRecorder or its `frame`."""
        f = obs.frame if hasattr(obs, "buf") else obs
        if f.rgb.shape[:2] != (320, 320):
            raise ValueError(f"the policy reads 320 px pictures, got {f.rgb.shape}")
        rgb_np = D.jpeg_roundtrip(np.ascontiguousarray(f.rgb)) if self.jpeg else np.ascontiguousarray(f.rgb)
        rgb = torch.from_numpy(rgb_np)[None].to(self.device).permute(0, 3, 1, 2)
        depth = torch.from_numpy(np.minimum(f.depth, 32767).astype(np.int16))[None].to(self.device)
        t = self.model.enc(rgb, depth)[0]
        if not self.context:
            self.context.extend([t] * CONTEXT)
        else:
            self.context.append(t)
        q = np.asarray(f.qpos[:ACTION_DIM], np.float32)
        z = torch.stack(list(self.context))[None]
        with _amp(z):
            a = self.model.head.sample(z, torch.from_numpy(q)[None].to(self.device), steps=self.flow_steps, n=n_samples)[0].float()
        a = a * self.std + self.mean
        return a.view(n_samples, CHUNK, ACTION_DIM).cpu().numpy() + q
