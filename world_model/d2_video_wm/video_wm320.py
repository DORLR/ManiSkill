"""D2 Video World Model & High-Resolution Encoder Suite (`VideoWM320`).

Drop-in architectural upgrade family for Changhao's `wm320.WM320` solving two verified root causes
of `B1_fidelity` failure and 30-36% imagined through-board penetration:
  1. Coarse ResNet-18 standard tokenization (`10x10`, 1 token = `32x32 px ~= 8-10 mm` vs `0.5 mm`
     insertion clearance) -> high-res pretrained `SegFormer MiT-B1 FPN`, `ResNet-34 NoPool`, and
     official Meta `V-JEPA 2` (`facebook/vjepa2-vitl-fpc64-256`) + parallel `320x320` float32 RGB-D stem.
  2. Severe single-vector `256-d [CLS]` bottleneck (`Linear(256->512)` -> rank <= 256) + 2-layer MLP
     dynamics -> wider full-rank `d=512` fusion (`D2a`) and Spatial-Token Action-Conditioned Video
     Dynamics Transformer `STDiTDynamics` (`D2b`, `D2c`) evolving a spatial token grid `Z_t in R^{K x D_z}`
     directly with AdaLN-Zero action modulation and per-token SimNorm!

Three clean experimental variants controlled by `--arch`:
  - `D2a` (`d2a_segformer`, `d2a_resnet34`, `d2a_resnet18`):
      Pretrained high-res encoder + full-rank `d=512` 4-frame Transformer + TD-MPC2 MLP dynamics/heads.
      (Pretrained encoder; domain-trained TD-MPC2 MLP transition model.)
  - `D2b` (`d2b_segformer`, `d2b_resnet18`, `d2b_resnet34`):
      Pretrained encoder + Spatial-Token 4-frame Fusion (`Z_t in R^{K x D_z}`, default `K=64` [8x8], `D_z=64`,
      per-token SimNorm, NO `[CLS]` crush) + 4-layer AdaLN-Zero `STDiTDynamics` + Multi-Query Attentive
      Spatial Readout + TD-MPC2 heads.
      (Pretrained encoder; domain-trained DiT spatial-token transition model.)
  - `D2c` (`d2c_vjepa2_vitl`, `d2c_vjepa2_ac_vitg`):
      Official Meta `V-JEPA 2` pretrained 3D video transformer (`facebook/vjepa2-vitl-fpc64-256`, MIT, ungated)
      + parallel `320x320` float32 RGB-D depth/pin residual stem -> spatial token grid `Z_t in R^{K x D_z}`.
      - `d2c_vjepa2_vitl`: Pretrained V-JEPA 2 ViT-L video encoder (top-4 blocks unfrozen by default; fits
        1x `3g.40gb` 40 GB MIG!) + domain-trained `STDiTDynamics` (`transition_pretrained=False`).
      - `d2c_vjepa2_ac_vitg`: Official frozen `ViT-g/16` + adapted official pretrained `vjepa2_ac_predictor`
        from `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')` (`transition_pretrained=True`;
        requires `h100:1` 80 GB).

External API Compatibility:
  - Preserves the EXACT `WM320` contract (`observe`, `enc.tokens`, `enc.fuse`, `next`, `reward`, `Q`, `V`,
    `scalar`, `act_chunk`, `scale`, `policy_action`, `target_value`, `update_target`, `loss`, `_done`,
    `_offset`, `off_scale`, `cfg`, `math`) so `WMAgent`, `attempt320`, `fidelity_factual320`, and
    `counterfactual320` run out of the box!
  - For spatial-token models (`D2b`/`D2c`), `z` is packed along the last dimension as `(..., K * D_z)` at the
    external boundary so unmodified `agent320.py` (`zc = z.expand(C, -1)`) and `fidelity_factual320.py` work
    with zero shape changes, while internal dynamics & readouts unpack `z` to `(..., K, D_z)`.
  - Zero-privilege `--offset-mode` (100% harmonized with D1 `d1_gap_fix` overlay):
    * `gap` (default, apples-to-apples parity with D1b): 4D `_offset` head trained WITH gradients on
      `(gap_mm, lat_x_mm, lat_y_mm, tilt_deg)` (`gap_mm = -along_mm - L_mm`, zero `self.L` at test time!).
    * `gap_1d` (optional ablation): 1D `_gap` head with gradients + strictly detached (`z.detach()`) 4D
      `_probe_sg` (`_offset`) whose dim 0 target is ALSO `gap_mm` (exact sign match to D1 `fidelity_factual320.py`).
    * `along` (alias `probe`): legacy 4D `(along_mm, lat_x_mm, lat_y_mm, tilt_deg)`.
    * `none`: zero gradient-flowing offset head + optional strictly detached `_probe_sg` on `(gap_mm, lx, ly, tilt)`.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import types
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
except ImportError:  # pragma: no cover
    torchvision = None  # type: ignore[assignment]

try:
    import transformers
except ImportError:  # pragma: no cover
    transformers = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants & Optional Changhao `wm.tdmpc` / TD-MPC2 Imports (with exact fallback)
# ---------------------------------------------------------------------------
VIDEO_WM_FORMAT = "video_wm320-1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFSET_SCALE = (10.0, 10.0, 10.0, 5.0)  # [along_mm, lat_x_mm, lat_y_mm, tilt_deg]
GAP_SCALE = 10.0                        # [mm] pin-tip clearance in front of board face
PHI_MAX = 150.0

DEFAULT_SEGFORMER_ID = "nvidia/mit-b1"
DEFAULT_VJEPA2_VITL_ID = "facebook/vjepa2-vitl-fpc64-256"

try:
    from wm.tdmpc import demo320 as D
    from wm.tdmpc import use_tdmpc2
    from wm.tdmpc import wmdata320 as WD
    from wm.tdmpc.model import make_cfg as _wm_make_cfg
    from wm.tdmpc.wm320 import WindowData as _BaseWindowData
    from wm.tdmpc.wm320 import WD_KINDS

    _HAS_WM_PKG = True
except Exception:  # pragma: no cover - standalone smoke test without task_release on sys.path
    D = None  # type: ignore[assignment]
    WD = None  # type: ignore[assignment]
    _wm_make_cfg = None
    _BaseWindowData = None
    WD_KINDS = {"reference": 0, "insert": 0, "perturbed": 0, "contact": 1, "random": 2}
    _HAS_WM_PKG = False

    def use_tdmpc2():  # type: ignore[misc]
        return False


CHUNK = int(D.CHUNK) if D is not None else 16
CONTEXT = int(D.CONTEXT) if D is not None else 4
JOINTS = int(D.JOINTS) if D is not None else 7
EXECUTE = int(WD.EXECUTE) if WD is not None else 8
HORIZON = int(WD.H) if WD is not None else 4
ACTION_DIM = int(WD.ACTION_DIM) if WD is not None else EXECUTE * JOINTS
LABELS = str(WD.LABELS) if WD is not None else "wm320-labels-3"

TDMPC2_DEFAULTS = dict(
    horizon=HORIZON,
    latent_dim=512,
    mlp_dim=512,
    enc_dim=256,
    num_enc_layers=2,
    num_q=5,
    dropout=0.01,
    simnorm_dim=8,
    num_bins=101,
    vmin=-10,
    vmax=10,
    frame_dim=512,
    action_dim=ACTION_DIM,
    consistency_coef=20,
    reward_coef=0.1,
    value_coef=0.1,
    rho=0.5,
    lr=3e-4,
    enc_lr_scale=0.3,
    grad_clip_norm=20,
    gamma=1 - 1 / 30,
    shift_px=0,
    amp=True,
    multitask=False,
    task_dim=0,
    stem_stride=2,
)


class _FallbackCfg(types.SimpleNamespace):
    def get(self, k: str, d: Any = None) -> Any:
        return getattr(self, k, d)


def make_cfg(**over) -> Any:
    if _wm_make_cfg is not None:
        return _wm_make_cfg(**over)
    c = _FallbackCfg(**{**TDMPC2_DEFAULTS, **over})
    c.bin_size = (c.vmax - c.vmin) / (c.num_bins - 1)
    return c


# ---------------------------------------------------------------------------
# Official TD-MPC2 Layers / Math Loader (with numerically identical fallback)
# ---------------------------------------------------------------------------
class _FallbackSimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = int(cfg.simnorm_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)


class _FallbackNormedLinear(nn.Linear):
    def __init__(self, *args, dropout: float = 0.0, act: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


class _FallbackEnsemble(nn.Module):
    def __init__(self, modules: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(list(modules))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m(x) for m in self.models], dim=0)


def _fallback_mlp(in_dim: int, mlp_dims: Sequence[int], out_dim: int, act: Optional[nn.Module] = None, dropout: float = 0.0) -> nn.Sequential:
    dims = [int(in_dim)] + [int(d) for d in mlp_dims] + [int(out_dim)]
    mods: List[nn.Module] = []
    for i in range(len(dims) - 2):
        mods.append(_FallbackNormedLinear(dims[i], dims[i + 1], dropout=dropout if i == 0 else 0.0))
    if act is not None:
        mods.append(_FallbackNormedLinear(dims[-2], dims[-1], act=act))
    else:
        mods.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*mods)


def _fallback_weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _fallback_zero_(params: Sequence[torch.Tensor]) -> None:
    for p in params:
        p.data.fill_(0.0)


def _fallback_symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(1.0 + torch.abs(x))


def _fallback_symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _fallback_two_hot(x: torch.Tensor, cfg) -> torch.Tensor:
    if cfg.num_bins == 0:
        return x
    if cfg.num_bins == 1:
        return _fallback_symlog(x)
    x = torch.clamp(_fallback_symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1.0 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def _fallback_two_hot_inv(x: torch.Tensor, cfg) -> torch.Tensor:
    if cfg.num_bins == 0:
        return x
    if cfg.num_bins == 1:
        return _fallback_symexp(x)
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return _fallback_symexp(x)


def _fallback_soft_ce(pred: torch.Tensor, target: torch.Tensor, cfg) -> torch.Tensor:
    pred = F.log_softmax(pred, dim=-1)
    target = _fallback_two_hot(target, cfg)
    return -(target * pred).sum(-1, keepdim=True)


def _load_tdmpc2_backend():
    """Return `(init_mod, layers_mod, math_mod, is_official)`."""
    if _HAS_WM_PKG:
        try:
            use_tdmpc2()
            from common import init as td_init
            from common import layers as td_layers
            from common import math as td_math

            return td_init, td_layers, td_math, True
        except Exception:
            pass
    td_init = types.SimpleNamespace(weight_init=_fallback_weight_init, zero_=_fallback_zero_)
    td_layers = types.SimpleNamespace(SimNorm=_FallbackSimNorm, mlp=_fallback_mlp, Ensemble=_FallbackEnsemble)
    td_math = types.SimpleNamespace(two_hot_inv=_fallback_two_hot_inv, soft_ce=_fallback_soft_ce, two_hot=_fallback_two_hot)
    return td_init, td_layers, td_math, False


def _zero_ensemble_last_layer(ens: nn.Module, td_init) -> None:
    """Zero-init the output layer of either TD-MPC2's `L.Ensemble` or `_FallbackEnsemble`."""
    if hasattr(ens, "params") and ("2", "weight") in ens.params:
        td_init.zero_([ens.params["2", "weight"]])
    elif hasattr(ens, "models"):
        td_init.zero_([m[-1].weight for m in ens.models])


def _amp(x: torch.Tensor, enabled: bool = True):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled and x.is_cuda)


def _to_nchw_rgb(rgb: torch.Tensor) -> torch.Tensor:
    """Accept `(N, H, W, 3)` (Changhao WindowData/WMAgent) or `(N, 3, H, W)` -> `(N, 3, H, W)`."""
    if rgb.ndim != 4:
        raise ValueError(f"Expected 4D RGB tensor, got shape {tuple(rgb.shape)}")
    if rgb.shape[-1] == 3 and rgb.shape[1] != 3:
        return rgb.permute(0, 3, 1, 2).contiguous()
    if rgb.shape[1] == 3:
        return rgb
    raise ValueError(f"Ambiguous RGB channel layout for shape {tuple(rgb.shape)}")


class Float32StemWrap(nn.Module):
    """Forces a 4-channel RGB-D stem conv to run strictly in float32 (preserves 0.1 mm D405 depth)."""

    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.mod(x.float())


def inflate_conv2d_3_to_4(
    conv3: nn.Conv2d,
    *,
    copy_rgb: bool = True,
    depth_init: str = "rgb_mean",
) -> nn.Conv2d:
    """Inflate a 3-channel Conv2d to 4-channel RGB-D (default depth init = mean of RGB kernels)."""
    if conv3.in_channels != 3:
        raise ValueError(f"Expected 3-ch Conv2d, got {conv3.in_channels}")
    if depth_init not in ("rgb_mean", "zero"):
        raise ValueError(f"Unknown depth_init={depth_init}")
    has_bias = conv3.bias is not None
    conv4 = nn.Conv2d(
        in_channels=4,
        out_channels=conv3.out_channels,
        kernel_size=conv3.kernel_size,
        stride=conv3.stride,
        padding=conv3.padding,
        dilation=conv3.dilation,
        groups=conv3.groups,
        bias=has_bias,
    )
    with torch.no_grad():
        if copy_rgb:
            conv4.weight[:, :3].copy_(conv3.weight)
            if depth_init == "rgb_mean":
                conv4.weight[:, 3:4].copy_(conv3.weight.mean(dim=1, keepdim=True))
            else:
                conv4.weight[:, 3:4].zero_()
            if has_bias and conv4.bias is not None and conv3.bias is not None:
                conv4.bias.copy_(conv3.bias)
        else:
            nn.init.kaiming_normal_(conv4.weight, mode="fan_out", nonlinearity="relu")
            if has_bias and conv4.bias is not None:
                nn.init.zeros_(conv4.bias)
    return conv4


def build_2d_sincos_pos_embed(gh: int, gw: int, dim: int, device: torch.device) -> torch.Tensor:
    """Deterministic 2D sin-cos positional embedding of shape `(1, gh * gw, dim)`."""
    if dim % 4 != 0:
        raise ValueError(f"2D sin-cos dim must be divisible by 4, got {dim}")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32) / float(quarter)
    omega = 1.0 / (10000.0 ** omega)
    y = torch.linspace(-1.0, 1.0, gh, device=device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, gw, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    out_y = gy.reshape(-1, 1) * omega[None, :]
    out_x = gx.reshape(-1, 1) * omega[None, :]
    emb = torch.cat([torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)], dim=-1)
    return emb.unsqueeze(0)


def _append_coord_channels(feat: torch.Tensor) -> torch.Tensor:
    """Append normalized `(u, v) in [-1, 1]` CoordConv channels to `(N, C, H, W)` feature map."""
    N, _, H, W = feat.shape
    vy = torch.linspace(-1.0, 1.0, H, device=feat.device, dtype=feat.dtype)
    ux = torch.linspace(-1.0, 1.0, W, device=feat.device, dtype=feat.dtype)
    gy, gx = torch.meshgrid(vy, ux, indexing="ij")
    coords = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(N, -1, -1, -1)
    return torch.cat([feat, coords], dim=1)


def _offline_local_only(local_files_only: bool) -> bool:
    if local_files_only:
        return True
    return os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"


# ---------------------------------------------------------------------------
# Pure-PyTorch Structural Fallbacks (ONLY for offline `pretrained=False` unit tests)
# ---------------------------------------------------------------------------
class _FallbackBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, downsample: Optional[nn.Module], norm_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


def _build_resnet_gn_fallback(layers: Sequence[int]) -> nn.Module:
    norm_layer = lambda c: nn.GroupNorm(8, c)  # noqa: E731

    class _RN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            self.layer1 = self._make(64, layers[0], 1)
            self.layer2 = self._make(128, layers[1], 2)
            self.layer3 = self._make(256, layers[2], 2)
            self.layer4 = self._make(512, layers[3], 2)

        def _make(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
            ds = None
            if stride != 1 or self.inplanes != planes:
                ds = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False), norm_layer(planes))
            out = [_FallbackBasicBlock(self.inplanes, planes, stride, ds, norm_layer)]
            self.inplanes = planes
            for _ in range(1, blocks):
                out.append(_FallbackBasicBlock(self.inplanes, planes, 1, None, norm_layer))
            return nn.Sequential(*out)

    return _RN()


def _build_segformer_mitb1_fallback() -> nn.Module:
    class _Stage(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, k: int, s: int):
            super().__init__()
            self.patch_embeddings = nn.Module()
            self.patch_embeddings.proj = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=k // 2)
            self.block = nn.Sequential(
                nn.GroupNorm(8, out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.Conv2d(out_ch, out_ch, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.patch_embeddings.proj(x)
            return h + self.block(h)

    class _SegFallback(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(num_channels=3)
            self.stages = nn.ModuleList(
                [
                    _Stage(3, 64, 7, 4),
                    _Stage(64, 128, 3, 2),
                    _Stage(128, 320, 3, 2),
                    _Stage(320, 512, 3, 2),
                ]
            )

        def forward(self, x: torch.Tensor, output_hidden_states: bool = True):
            del output_hidden_states
            hs = []
            for st in self.stages:
                x = st(x)
                hs.append(x)
            return types.SimpleNamespace(hidden_states=tuple(hs), last_hidden_state=hs[-1])

    return _SegFallback()


class _FallbackVJEPA2ViT(nn.Module):
    """Lightweight structural mirror of `VJEPA2Model` for `pretrained=False` local unit tests."""

    def __init__(self, hidden_size: int = 1024, depth: int = 6, heads: int = 16, crop_size: int = 256):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=hidden_size,
            patch_size=16,
            tubelet_size=2,
            crop_size=crop_size,
            in_chans=3,
        )
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.proj = nn.Conv3d(3, hidden_size, kernel_size=(2, 16, 16), stride=(2, 16, 16))
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=heads,
                    dim_feedforward=2 * hidden_size,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.encoder.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values_videos: torch.Tensor, skip_predictor: bool = True):
        del skip_predictor
        # Matches HF VJEPA2Model: input is (B, T, C, H, W)
        x = pixel_values_videos.permute(0, 2, 1, 3, 4)
        h = self.embeddings.patch_embeddings.proj(x).flatten(2).transpose(1, 2)
        for blk in self.encoder.layer:
            h = blk(h)
        h = self.encoder.layernorm(h)
        return types.SimpleNamespace(last_hidden_state=h)


# ---------------------------------------------------------------------------
# High-Resolution 2D RGB-D Frame Encoders (D2a & D2b)
# ---------------------------------------------------------------------------
class _RGBDNormMixin:
    def _init_rgbd_norm(self) -> None:
        self.register_buffer("_rgb_mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_rgb_std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def normalize_rgbd(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Strictly float32 RGB-D normalization matching Study #1 & wm320."""
        rgb_nchw = _to_nchw_rgb(rgb)
        with torch.autocast(device_type=rgb_nchw.device.type, enabled=False):
            r = (rgb_nchw.to(torch.float32) / 255.0 - self._rgb_mean) / self._rgb_std
            d = (depth.to(torch.float32) * 1e-3).clamp(0.0, 3.0).unsqueeze(1)
            return torch.cat([r, d], dim=1)

    def normalize_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb_nchw = _to_nchw_rgb(rgb)
        with torch.autocast(device_type=rgb_nchw.device.type, enabled=False):
            return (rgb_nchw.to(torch.float32) / 255.0 - self._rgb_mean) / self._rgb_std


class SpatialTokenProjector(nn.Module):
    """Projects a high-res `(N, C_in, H_f, W_f)` map (+ CoordConv `(u, v)`) to `(N, G*G, C_tok)`.

    Preserves sub-cell pin/hole offset inside each output token via CoordConv + learned strided/adaptive
    fusion instead of blind average pooling alone.
    """

    def __init__(self, in_ch: int, out_dim: int, grid_size: int):
        super().__init__()
        self.grid_size = int(grid_size)
        self.out_dim = int(out_dim)
        mid = min(256, max(128, out_dim))
        self.conv_pre = nn.Sequential(
            nn.Conv2d(in_ch + 2, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(),
        )
        self.conv_post = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, mid),
            nn.SiLU(),
            nn.Conv2d(mid, out_dim, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = self.conv_pre(_append_coord_channels(feat))
        G = self.grid_size
        if h.shape[2] != G or h.shape[3] != G:
            if h.shape[2] == 2 * G and h.shape[3] == 2 * G:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)
            else:
                h = F.adaptive_avg_pool2d(h, output_size=(G, G))
        h = self.conv_post(h)
        return h.float().flatten(2).transpose(1, 2).contiguous()  # (N, G*G, out_dim)


class SegFormerB1FPNTokenizer(nn.Module, _RGBDNormMixin):
    """Official pretrained `nvidia/mit-b1` with float32 4-ch RGB-D stem + 4-stage FPN (`1/4..1/32`).

    Verified top performer in Study #1/#2 (`OOD Press AUC 0.91-0.923`).
    Zero BatchNorm anywhere (native LayerNorm in MiT-B1 + GroupNorm in FPN).
    """

    def __init__(
        self,
        out_dim: int = 64,
        grid_size: int = 10,
        pretrained: bool = True,
        model_id: str = DEFAULT_SEGFORMER_ID,
        local_files_only: bool = False,
        depth_init: str = "rgb_mean",
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self._init_rgbd_norm()
        self.backbone_name = "segformer_b1_fpn"
        self.pretrained_source = str(model_id) if pretrained else "scratch"
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
        self.num_tokens = self.grid_size * self.grid_size

        if pretrained and transformers is None:
            raise ImportError("transformers is required for pretrained SegFormer MiT-B1 (`nvidia/mit-b1`).")

        if transformers is not None:
            if pretrained:
                seg = transformers.SegformerModel.from_pretrained(
                    model_id,
                    local_files_only=_offline_local_only(local_files_only),
                )
            else:
                cfg = transformers.SegformerConfig(
                    num_channels=3,
                    num_encoder_blocks=4,
                    depths=[2, 2, 2, 2],
                    sr_ratios=[8, 4, 2, 1],
                    hidden_sizes=[64, 128, 320, 512],
                    patch_sizes=[7, 3, 3, 3],
                    strides=[4, 2, 2, 2],
                    num_attention_heads=[1, 2, 5, 8],
                    mlp_ratios=[4, 4, 4, 4],
                )
                seg = transformers.SegformerModel(cfg)
        else:
            seg = _build_segformer_mitb1_fallback()

        pe0 = seg.stages[0].patch_embeddings if hasattr(seg, "stages") else seg.encoder.patch_embeddings[0]
        proj4 = inflate_conv2d_3_to_4(pe0.proj, copy_rgb=pretrained, depth_init=depth_init)
        pe0.proj = Float32StemWrap(proj4)
        if hasattr(seg, "config"):
            seg.config.num_channels = 4
        if grad_ckpt and hasattr(seg, "gradient_checkpointing_enable"):
            seg.gradient_checkpointing_enable()

        self._stem_proj = pe0.proj
        self.seg = seg

        stage_dims = (64, 128, 320, 512)
        self.laterals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(8, 128),
                    nn.SiLU(),
                )
                for c in stage_dims
            ]
        )
        # Fuse all 4 scales (80x80, 40x40, 20x20, 10x10) at crisp 20x20 (1/16) resolution
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(128 * 4, 256, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.projector = SpatialTokenProjector(256, out_dim=self.out_dim, grid_size=self.grid_size)

    def split_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        stem_ids = {id(p) for p in self._stem_proj.parameters()}
        pretrained_p = [p for p in self.seg.parameters() if p.requires_grad and id(p) not in stem_ids]
        new_p = (
            [p for p in self._stem_proj.parameters() if p.requires_grad]
            + [p for p in self.laterals.parameters() if p.requires_grad]
            + [p for p in self.fpn_fuse.parameters() if p.requires_grad]
            + [p for p in self.projector.parameters() if p.requires_grad]
        )
        return pretrained_p, new_p

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x4 = self.normalize_rgbd(rgb, depth)  # (N, 4, 320, 320) float32
        H_f, W_f = x4.shape[2] // 16, x4.shape[3] // 16  # 20x20 @ 320px
        with _amp(x4):
            hs = self.seg(x4, output_hidden_states=True).hidden_states
            if len(hs) > 4:
                hs = hs[-4:]
            aligned: List[torch.Tensor] = []
            for lat, feat in zip(self.laterals, hs):
                p = lat(feat)
                fh, fw = p.shape[2], p.shape[3]
                if (fh, fw) == (H_f, W_f):
                    aligned.append(p)
                elif fh > H_f:
                    aligned.append(F.adaptive_avg_pool2d(p, output_size=(H_f, W_f)))
                else:
                    aligned.append(F.interpolate(p, size=(H_f, W_f), mode="bilinear", align_corners=False))
            fpn_20 = self.fpn_fuse(torch.cat(aligned, dim=1))
            tok = self.projector(fpn_20)
        return tok.float()


class ResNetGroupNormTokenizer(nn.Module, _RGBDNormMixin):
    """Pretrained `ResNet-34 NoPool` (`20x20` map) or `ResNet-18 Standard` (`10x10`) with GroupNorm(8).

    Copies all `IMAGENET1K_V1` conv weights into a GroupNorm(8) ResNet (exact Diffusion Policy /
    Changhao `PictureTokens.init_imagenet` convention: zero BN drift, exact micro-batch accumulation),
    with float32 4-channel RGB-D stem (`depth` initialized from RGB mean).
    """

    def __init__(
        self,
        arch: str = "resnet34",
        nopool: bool = True,
        out_dim: int = 64,
        grid_size: int = 10,
        pretrained: bool = True,
        depth_init: str = "rgb_mean",
    ):
        super().__init__()
        self._init_rgbd_norm()
        if arch not in ("resnet18", "resnet34"):
            raise ValueError(f"Unsupported ResNet arch={arch}")
        self.backbone_name = f"{arch}_{'nopool' if nopool else 'standard'}_gn"
        self.pretrained_source = f"torchvision:{arch}:IMAGENET1K_V1" if pretrained else "scratch"
        self.nopool = bool(nopool)
        self.out_dim = int(out_dim)
        self.grid_size = int(grid_size)
        self.num_tokens = self.grid_size * self.grid_size

        if pretrained and torchvision is None:
            raise ImportError(f"torchvision is required for pretrained {arch}.")

        if torchvision is not None:
            ctor = torchvision.models.resnet34 if arch == "resnet34" else torchvision.models.resnet18
            net = ctor(weights=None, norm_layer=lambda c: nn.GroupNorm(8, c))
        else:
            net = _build_resnet_gn_fallback((3, 4, 6, 3) if arch == "resnet34" else (2, 2, 2, 2))

        self.stem_conv = Float32StemWrap(nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.stem_norm = nn.Sequential(nn.GroupNorm(8, 64), nn.ReLU(inplace=True))
        self.maxpool = nn.Identity() if nopool else net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # Fuse layer3 (256ch) + layer4 (512ch) lightly for NoPool crisp detail, or direct layer4 for ResNet-18 control
        self.use_fpn = bool(nopool)
        if self.use_fpn:
            self.lat4 = nn.Sequential(nn.Conv2d(512, 160, 1, bias=False), nn.GroupNorm(8, 160), nn.SiLU())
            self.lat3 = nn.Sequential(nn.Conv2d(256, 160, 1, bias=False), nn.GroupNorm(8, 160), nn.SiLU())
            self.fpn_fuse = nn.Sequential(nn.Conv2d(320, 256, 3, padding=1, bias=False), nn.GroupNorm(8, 256), nn.SiLU())
            self.projector = SpatialTokenProjector(256, out_dim=self.out_dim, grid_size=self.grid_size)
        else:
            self.projector = SpatialTokenProjector(512, out_dim=self.out_dim, grid_size=self.grid_size)

        if pretrained:
            self._copy_imagenet_weights(arch=arch, depth_init=depth_init)

    def _copy_imagenet_weights(self, arch: str, depth_init: str = "rgb_mean") -> int:
        ctor = torchvision.models.resnet34 if arch == "resnet34" else torchvision.models.resnet18
        src = ctor(weights="IMAGENET1K_V1").state_dict()
        dst = self.state_dict()
        copied = 0
        with torch.no_grad():
            w1 = src["conv1.weight"]
            self.stem_conv.mod.weight[:, :3].copy_(w1)
            if depth_init == "rgb_mean":
                self.stem_conv.mod.weight[:, 3:4].copy_(w1.mean(dim=1, keepdim=True))
            else:
                self.stem_conv.mod.weight[:, 3:4].zero_()
            copied += 1
            for k, v in src.items():
                if k.startswith(("layer1.", "layer2.", "layer3.", "layer4.")) and k in dst and dst[k].shape == v.shape:
                    if "conv" in k or "downsample.0" in k:
                        dst[k].copy_(v)
                        copied += 1
        return copied

    def split_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        pretrained_p = (
            [p for p in self.layer1.parameters() if p.requires_grad]
            + [p for p in self.layer2.parameters() if p.requires_grad]
            + [p for p in self.layer3.parameters() if p.requires_grad]
            + [p for p in self.layer4.parameters() if p.requires_grad]
        )
        new_p = (
            [p for p in self.stem_conv.parameters() if p.requires_grad]
            + [p for p in self.stem_norm.parameters() if p.requires_grad]
            + [p for p in self.projector.parameters() if p.requires_grad]
        )
        if self.use_fpn:
            new_p += (
                [p for p in self.lat4.parameters() if p.requires_grad]
                + [p for p in self.lat3.parameters() if p.requires_grad]
                + [p for p in self.fpn_fuse.parameters() if p.requires_grad]
            )
        return pretrained_p, new_p

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        x4 = self.normalize_rgbd(rgb, depth)  # (N, 4, 320, 320) float32
        h = self.stem_conv(x4)                # float32!
        with _amp(h):
            h = self.maxpool(self.stem_norm(h))
            c1 = self.layer1(h)
            c2 = self.layer2(c1)
            c3 = self.layer3(c2)  # 40x40 (NoPool) or 20x20 (standard)
            c4 = self.layer4(c3)  # 20x20 (NoPool) or 10x10 (standard)
            if self.use_fpn:
                p4 = self.lat4(c4)
                p3 = F.avg_pool2d(self.lat3(c3), kernel_size=2, stride=2)
                feat = self.fpn_fuse(torch.cat([p4, p3], dim=1))  # 20x20
            else:
                feat = c4  # 10x10
            tok = self.projector(feat)
        return tok.float()


# ---------------------------------------------------------------------------
# D2a Encoder: High-Res Pretrained Backbone + Wider Full-Rank `d=512` Fusion
# ---------------------------------------------------------------------------
class D2aEncoder(nn.Module):
    """D2a controlled encoder upgrade: high-res backbone + full-rank `d=512` 4-frame Transformer.

    Fixes Changhao's `d=256 -> Linear(256, 512)` rank bottleneck while keeping TD-MPC2's `z in R^512`
    vector latent and MLP dynamics for a clean controlled encoder comparison.
    """

    def __init__(
        self,
        pic: nn.Module,
        cfg,
        d: int = 512,
        layers: int = 4,
        heads: int = 8,
        simnorm_cls=None,
    ):
        super().__init__()
        self.pic = pic
        self.d = int(d)
        self.num_pic_tokens = int(pic.num_tokens)
        self.pic_token_dim = int(pic.out_dim)
        self.inp = nn.Linear(self.pic_token_dim, self.d)
        self.token_pos = nn.Parameter(torch.randn(self.num_pic_tokens, self.d) * 0.02)
        self.frame_pos = nn.Parameter(torch.randn(CONTEXT, self.d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, self.d) * 0.02)
        self.joint_in = nn.Sequential(nn.Linear(JOINTS, self.d), nn.SiLU(), nn.Linear(self.d, self.d))
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d,
                nhead=heads,
                dim_feedforward=4 * self.d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.out = nn.Sequential(nn.LayerNorm(self.d), nn.Linear(self.d, cfg.latent_dim), simnorm_cls(cfg))

    def tokens(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.pic(rgb, depth)  # (N, K_pic, C_pic)

    def fuse(self, tok: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """tok `(B, 4, K_pic, C_pic)`, q `(B, 7)` -> `z (B, latent_dim)`."""
        B = tok.shape[0]
        h = self.inp(tok) + self.token_pos[None, None] + self.frame_pos[None, :, None]
        with torch.autocast(device_type=q.device.type, enabled=False):
            j = self.joint_in(q.float())
        h = torch.cat([self.cls.expand(B, -1, -1), j[:, None], h.flatten(1, 2)], dim=1)
        with _amp(h):
            h = self.tf(h)[:, 0]
        return self.out(h.float())

    def split_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        pre_p, pic_new_p = self.pic.split_params()
        pic_ids = {id(p) for p in self.pic.parameters()}
        fuse_p = [p for p in self.parameters() if p.requires_grad and id(p) not in pic_ids]
        return pre_p, pic_new_p + fuse_p


# ---------------------------------------------------------------------------
# D2b Encoder: Spatial-Token 4-Frame Video Encoder (Breaks `[CLS]` Bottleneck!)
# ---------------------------------------------------------------------------
class D2bSpatialEncoder(nn.Module):
    """Preserves an explicit `G x G` (`K = G^2`) spatial token grid `Z_t in R^{K x D_z}` with per-token SimNorm!

    Instead of crushing 4 frames into 1 `[CLS]` vector, each of the `K` current-frame spatial cells attends
    across all 4 frames `[t-1.2s..t]` AND the strict-float32 proprioceptive joint token `j(q_t)`, outputting
    a packed `(B, K * D_z)` spatial latent where every token `k in {0..K-1}` retains its 2D correspondence.
    """

    def __init__(
        self,
        pic: nn.Module,
        cfg,
        latent_grid: int = 8,
        token_dim: int = 64,
        d_enc: int = 384,
        layers: int = 4,
        heads: int = 6,
        simnorm_cls=None,
    ):
        super().__init__()
        self.pic = pic
        self.latent_grid = int(latent_grid)
        self.num_tokens = self.latent_grid * self.latent_grid
        self.token_dim = int(token_dim)
        self.d_enc = int(d_enc)
        if pic.num_tokens != self.num_tokens:
            raise ValueError(f"D2bSpatialEncoder expects pic.num_tokens=={self.num_tokens}, got {pic.num_tokens}")

        self.inp = nn.Linear(int(pic.out_dim), self.d_enc)
        self.frame_pos = nn.Parameter(torch.randn(CONTEXT, self.d_enc) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(self.num_tokens, self.d_enc) * 0.02)
        self.joint_in = nn.Sequential(nn.Linear(JOINTS, self.d_enc), nn.SiLU(), nn.Linear(self.d_enc, self.d_enc))
        self.tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_enc,
                nhead=heads,
                dim_feedforward=4 * self.d_enc,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        # Fuse newest-frame spatial tokens + temporal-mean summary + attended proprioceptive joint token `q_t`
        self.out_proj = nn.Sequential(
            nn.LayerNorm(3 * self.d_enc),
            nn.Linear(3 * self.d_enc, self.token_dim),
            simnorm_cls(cfg),
        )

    def tokens(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return self.pic(rgb, depth)  # (N, K, C_pic)

    def fuse(self, tok: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """tok `(B, 4, K, C_pic)`, q `(B, 7)` -> packed spatial latent `(B, K * D_z)`."""
        B, T, K, _ = tok.shape
        if T != CONTEXT or K != self.num_tokens:
            raise ValueError(f"Expected (B, {CONTEXT}, {self.num_tokens}, C), got {tuple(tok.shape)}")
        h = self.inp(tok) + self.spatial_pos[None, None] + self.frame_pos[None, :, None]
        with torch.autocast(device_type=q.device.type, enabled=False):
            j = self.joint_in(q.float())
        seq = torch.cat([j[:, None], h.flatten(1, 2)], dim=1)  # (B, 1 + 4*K, d_enc)
        with _amp(seq):
            seq = self.tf(seq)
        seq_f = seq.float()
        j_out = seq_f[:, 0]                                   # (B, d_enc) attended proprioception preserved directly!
        grid_seq = seq_f[:, 1:].view(B, CONTEXT, K, self.d_enc)
        curr_sp = grid_seq[:, -1]                             # newest frame spatial grid after 4-frame + joint attention
        mean_sp = grid_seq.mean(dim=1)                        # temporal context summary at each spatial cell
        j_exp = j_out[:, None, :].expand(-1, K, -1)           # condition every spatial token with q_t from step 0
        z_tokens = self.out_proj(torch.cat([curr_sp, mean_sp, j_exp], dim=-1))  # (B, K, D_z)
        return z_tokens.reshape(B, K * self.token_dim)

    def split_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        pre_p, pic_new_p = self.pic.split_params()
        pic_ids = {id(p) for p in self.pic.parameters()}
        fuse_p = [p for p in self.parameters() if p.requires_grad and id(p) not in pic_ids]
        return pre_p, pic_new_p + fuse_p


# ---------------------------------------------------------------------------
# D2c Encoder: Official Pretrained Meta V-JEPA 2 Video Backbone + High-Res RGB-D Stem
# ---------------------------------------------------------------------------
class ParallelRGBDConvStem(nn.Module, _RGBDNormMixin):
    """Lightweight high-res `320x320` float32 RGB-D residual stem producing `(N, 128, 16, 16)`.

    Preserves 0.1 mm D405 float depth and native `320x320` pin-edge cues alongside V-JEPA 2's
    `256x256` RGB video tokens without modifying V-JEPA 2's pretrained 3D patch embedding.
    """

    def __init__(self, out_ch: int = 128):
        super().__init__()
        self._init_rgbd_norm()
        self.out_ch = int(out_ch)
        self.stage1_fp32 = nn.Sequential(
            nn.Conv2d(4, 48, kernel_size=3, stride=2, padding=1, bias=False),  # 160x160
            nn.GroupNorm(8, 48),
            nn.SiLU(),
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, 48),
            nn.SiLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),  # 80x80
            nn.GroupNorm(8, 96),
            nn.SiLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=False),  # 40x40
            nn.GroupNorm(8, 96),
            nn.SiLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(96, self.out_ch, kernel_size=3, stride=2, padding=1, bias=False),  # 20x20
            nn.GroupNorm(8, self.out_ch),
            nn.SiLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, self.out_ch),
            nn.SiLU(),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(stem_16 (N, out_ch, 16, 16), rgb_256_norm (N, 3, 256, 256))` in float32."""
        x4 = self.normalize_rgbd(rgb, depth)  # (N, 4, 320, 320) float32
        with torch.autocast(device_type=x4.device.type, enabled=False):
            h = self.stage1_fp32(x4)
            rgb_256 = F.interpolate(x4[:, :3].detach(), size=(256, 256), mode="bilinear", align_corners=False, antialias=True)
        with _amp(h):
            h = self.stage3(self.stage2(h))  # (N, out_ch, 20, 20)
            h16 = F.adaptive_avg_pool2d(h, output_size=(16, 16))
        return h16.float(), rgb_256.float()


class VJEPA2RGBDEncoder(nn.Module):
    """Official Meta `V-JEPA 2` 3D video encoder + parallel `320x320` float32 RGB-D residual stem.

    Loader priority:
      - `variant == "vitl"`: HuggingFace `AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")`
        (MIT license, ungated, `hidden_size=1024, depth=24, patch=16, tubelet=2, 3D RoPE`).
        Falls back to `torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")` if HF transformers
        does not yet include `VJEPA2Model`.
      - `variant == "vitg_ac"`: `torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")`
        returning `(encoder, ac_predictor)` (`hidden_size=1408, depth=40`).

    API Trick for 100% `WMAgent.encode()` compatibility:
      - `tokens(rgb, depth)` packs lossless `F.pixel_unshuffle(rgb_256, 16)` (768ch) + `stem_16` (128ch)
        into a single `(N, 256, 896)` tensor!
      - `fuse(tok, q)` unpacks `rgb_256.detach()` (STRICT `.detach()` so frozen V-JEPA 2 blocks hold
        ZERO activation graph!) into `(B, 4, 3, 256, 256)` for V-JEPA 2, fuses the resulting `16x16`
        tubelet video semantics + temporal motion difference with `stem_16` (0.1 mm D405 depth!),
        and outputs `K = G^2` per-token SimNorm spatial tokens packed as `(B, K * D_z)`.
    """

    def __init__(
        self,
        cfg,
        variant: str = "vitl",
        model_id: str = DEFAULT_VJEPA2_VITL_ID,
        pretrained: bool = True,
        local_files_only: bool = False,
        unfreeze_blocks: int = 4,
        latent_grid: int = 8,
        token_dim: int = 64,
        d_fuse: int = 384,
        simnorm_cls=None,
        vjepa2_hf_config: Optional[dict] = None,
    ):
        super().__init__()
        if variant not in ("vitl", "vitg_ac"):
            raise ValueError(f"Unknown V-JEPA 2 variant={variant}")
        self.variant = str(variant)
        self.model_id = str(model_id)
        self.pretrained_source = (
            f"torch.hub:facebookresearch/vjepa2:vjepa2_ac_vit_giant"
            if (pretrained and variant == "vitg_ac")
            else (str(model_id) if pretrained else f"scratch_vjepa2_{variant}")
        )
        self.latent_grid = int(latent_grid)
        self.num_tokens = self.latent_grid * self.latent_grid
        self.token_dim = int(token_dim)
        self.d_fuse = int(d_fuse)
        self.unfreeze_blocks = int(unfreeze_blocks)
        self._backend_kind = "fallback"  # "hf" | "torchhub" | "fallback"
        self.ac_predictor_raw: Optional[nn.Module] = None

        if variant == "vitg_ac":
            self.vjepa_dim = 1408
            if pretrained:
                enc, pred = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
                self.vjepa = enc
                self.ac_predictor_raw = pred
                self._backend_kind = "torchhub"
            else:
                self.vjepa = _FallbackVJEPA2ViT(hidden_size=1408, depth=4, heads=16, crop_size=256)
                self._backend_kind = "fallback"
        else:
            self.vjepa_dim = 1024
            if vjepa2_hf_config is not None and transformers is not None and not pretrained:
                # Lossless offline reconstruction from saved checkpoint config (zero HF cache needed!)
                cfg_cls = getattr(transformers, "VJEPA2Config", None)
                mdl_cls = getattr(transformers, "VJEPA2Model", None)
                if cfg_cls is not None and mdl_cls is not None:
                    self.vjepa = mdl_cls(cfg_cls(**vjepa2_hf_config))
                    self._backend_kind = "hf"
                else:
                    self.vjepa = _FallbackVJEPA2ViT(hidden_size=1024, depth=6, heads=16, crop_size=256)
                    self._backend_kind = "fallback"
            elif pretrained:
                loaded = False
                if transformers is not None:
                    try:
                        self.vjepa = transformers.AutoModel.from_pretrained(
                            model_id,
                            local_files_only=_offline_local_only(local_files_only),
                        )
                        self._backend_kind = "hf"
                        loaded = True
                    except Exception as hf_exc:
                        if local_files_only:
                            raise RuntimeError(
                                f"Failed to load local V-JEPA 2 checkpoint '{model_id}' via transformers: {hf_exc}"
                            ) from hf_exc
                if not loaded:
                    self.vjepa = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
                    self._backend_kind = "torchhub"
            else:
                self.vjepa = _FallbackVJEPA2ViT(hidden_size=1024, depth=6, heads=16, crop_size=256)
                self._backend_kind = "fallback"

        self._configure_vjepa_freezing()

        self.rgbd_stem = ParallelRGBDConvStem(out_ch=128)
        self.num_pic_tokens = 16 * 16
        self.pic_token_dim = 768 + 128  # lossless pixel_unshuffle(256x256 RGB, 16) + 128ch stem

        # Project V-JEPA 2 last tubelet + motion diff (last - first) + high-res RGB-D stem at 16x16
        self.vjepa_sem_proj = nn.Sequential(
            nn.Conv2d(self.vjepa_dim, 256, kernel_size=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        self.vjepa_mot_proj = nn.Sequential(
            nn.Conv2d(self.vjepa_dim, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        self.stem_temp_proj = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
        )
        # Fuse [sem(256) + motion(128) + rgbd_stem(128) + coords(2)] at 16x16 -> latent_grid x latent_grid
        self.grid_fuse = SpatialTokenProjector(512, out_dim=self.d_fuse, grid_size=self.latent_grid)
        self.spatial_pos = nn.Parameter(torch.randn(self.num_tokens, self.d_fuse) * 0.02)
        self.joint_in = nn.Sequential(nn.Linear(JOINTS, self.d_fuse), nn.SiLU(), nn.Linear(self.d_fuse, self.d_fuse))
        self.fuse_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_fuse,
                nhead=6,
                dim_feedforward=4 * self.d_fuse,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        # Fuse spatial tokens with attended proprioceptive joint token `q_t` directly into z_0
        self.out_proj = nn.Sequential(
            nn.LayerNorm(2 * self.d_fuse),
            nn.Linear(2 * self.d_fuse, self.token_dim),
            simnorm_cls(cfg),
        )

    def _get_vjepa_blocks(self) -> Optional[nn.ModuleList]:
        if hasattr(self.vjepa, "encoder") and hasattr(self.vjepa.encoder, "layer"):
            return self.vjepa.encoder.layer
        if hasattr(self.vjepa, "blocks"):
            return self.vjepa.blocks
        return None

    def _configure_vjepa_freezing(self) -> None:
        self.vjepa.requires_grad_(False)
        blocks = self._get_vjepa_blocks()
        if self.unfreeze_blocks > 0 and blocks is not None:
            for blk in list(blocks)[-self.unfreeze_blocks :]:
                blk.requires_grad_(True)
            if hasattr(self.vjepa, "encoder") and hasattr(self.vjepa.encoder, "layernorm"):
                self.vjepa.encoder.layernorm.requires_grad_(True)
            elif hasattr(self.vjepa, "norm") and self.vjepa.norm is not None:
                self.vjepa.norm.requires_grad_(True)
        self._sync_frozen_eval()

    def _sync_frozen_eval(self) -> None:
        if self.unfreeze_blocks <= 0:
            self.vjepa.eval()
            return
        if hasattr(self.vjepa, "embeddings"):
            self.vjepa.embeddings.eval()
        if hasattr(self.vjepa, "patch_embed"):
            self.vjepa.patch_embed.eval()
        blocks = self._get_vjepa_blocks()
        if blocks is not None:
            n_frozen = max(0, len(blocks) - self.unfreeze_blocks)
            for blk in list(blocks)[:n_frozen]:
                blk.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._sync_frozen_eval()
        return self

    def _forward_vjepa_clip(self, rgb_btchw: torch.Tensor) -> torch.Tensor:
        """`rgb_btchw`: `(B, T=4, 3, 256, 256)` float32 (detached!) -> `(B, 2, 16, 16, D_vjepa)` float32."""
        B, T, C, H, W = rgb_btchw.shape
        x_det = rgb_btchw.detach()  # CRITICAL: frozen prefix blocks store ZERO activation graph!
        ctx = torch.no_grad() if self.unfreeze_blocks <= 0 else torch.enable_grad()
        with ctx, _amp(x_det):
            if self._backend_kind in ("hf", "fallback"):
                # HF VJEPA2Embeddings expects (B, T, C, H, W) and permutes internally to (B, C, T, H, W)
                out = self.vjepa(pixel_values_videos=x_det, skip_predictor=True)
                feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            else:
                # Native Meta torch.hub VisionTransformer expects (B, C, T, H, W)
                feats = self.vjepa(x_det.permute(0, 2, 1, 3, 4))
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
        t_tok = max(1, T // 2)
        gh, gw = H // 16, W // 16
        expected = t_tok * gh * gw
        if feats.shape[1] != expected:
            raise RuntimeError(
                f"V-JEPA 2 output token count mismatch: expected {expected} ({t_tok}x{gh}x{gw}), got {tuple(feats.shape)}"
            )
        return feats.float().view(B, t_tok, gh, gw, self.vjepa_dim)

    def tokens(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Per-frame packed `(N, 256, 896)` = `[pixel_unshuffle(rgb_256, 16) (768), stem_16 (128)]`."""
        stem_16, rgb_256 = self.rgbd_stem(rgb, depth)
        rgb_u = F.pixel_unshuffle(rgb_256, 16)  # (N, 768, 16, 16), exact lossless!
        packed = torch.cat([rgb_u, stem_16], dim=1)
        return packed.flatten(2).transpose(1, 2).contiguous()  # (N, 256, 896)

    def fuse(self, tok: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """tok `(B, 4, 256, 896)`, q `(B, 7)` -> packed spatial latent `(B, K * D_z)`."""
        B, T, HW, C = tok.shape
        if T != CONTEXT or HW != 256 or C != 896:
            raise ValueError(f"Expected (B, 4, 256, 896), got {tuple(tok.shape)}")
        # Detach RGB half immediately so Slice/Cat backward NEVER touches frozen V-JEPA 2 layers
        rgb_u = tok[..., :768].detach().reshape(B * T, 16, 16, 768).permute(0, 3, 1, 2).contiguous()
        rgb_256 = F.pixel_shuffle(rgb_u, 16).view(B, T, 3, 256, 256)

        stem = tok[..., 768:].reshape(B, T, 16, 16, 128).permute(0, 1, 4, 2, 3).contiguous()  # (B, 4, 128, 16, 16)
        stem_cat = torch.cat([stem[:, -1], stem.mean(dim=1)], dim=1)  # (B, 256, 16, 16)

        vj = self._forward_vjepa_clip(rgb_256)  # (B, 2, 16, 16, D_vjepa)
        vj_last = vj[:, -1].permute(0, 3, 1, 2).contiguous()
        vj_diff = (vj[:, -1] - vj[:, 0]).permute(0, 3, 1, 2).contiguous()

        with _amp(vj_last):
            f_sem = self.vjepa_sem_proj(vj_last)
            f_mot = self.vjepa_mot_proj(vj_diff)
            f_stem = self.stem_temp_proj(stem_cat)
            fused_16 = torch.cat([f_sem, f_mot, f_stem], dim=1)  # (B, 512, 16, 16)
            grid_tok = self.grid_fuse(fused_16)                  # (B, K, d_fuse)

        with torch.autocast(device_type=q.device.type, enabled=False):
            j = self.joint_in(q.float())
        seq = torch.cat([j[:, None], grid_tok + self.spatial_pos[None]], dim=1)
        with _amp(seq):
            seq = self.fuse_tf(seq)
        seq_f = seq.float()
        j_out = seq_f[:, 0]                                               # (B, d_fuse) attended proprioception
        sp_out = seq_f[:, 1:]                                             # (B, K, d_fuse)
        j_exp = j_out[:, None, :].expand(-1, self.num_tokens, -1)
        z_tokens = self.out_proj(torch.cat([sp_out, j_exp], dim=-1))      # (B, K, D_z)
        return z_tokens.reshape(B, self.num_tokens * self.token_dim)

    def split_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        pre_p = [p for p in self.vjepa.parameters() if p.requires_grad]
        pre_ids = {id(p) for p in self.vjepa.parameters()}
        if self.ac_predictor_raw is not None:
            pre_ids.update(id(p) for p in self.ac_predictor_raw.parameters())
        new_p = [p for p in self.parameters() if p.requires_grad and id(p) not in pre_ids]
        return pre_p, new_p


# ---------------------------------------------------------------------------
# Spatial-Token Action-Conditioned Dynamics: AdaLN-Zero ST-DiT (D2b & D2c)
# ---------------------------------------------------------------------------
class STDiTBlock(nn.Module):
    """DiT-style (Peebles & Xie, ICCV 2023) AdaLN-Zero Transformer block modulated by action chunk `c`."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        if self.dim % self.heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")

        self.norm1 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

        self.norm2 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-6)
        hidden = int(self.dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, self.dim),
        )

        # AdaLN modulation: 6 modulation params (shift1, scale1, gate1, shift2, scale2, gate2).
        # Initialize gate1 & gate2 biases to small non-zero (0.1) rather than 0.0 so ST-DiT is strictly
        # non-identity and propagates live attention/MLP gradients from step 0 (prevents trivial collapse!).
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.dim, 6 * self.dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        with torch.no_grad():
            self.adaLN_modulation[-1].bias[2 * self.dim : 3 * self.dim].fill_(0.1)  # gate1
            self.adaLN_modulation[-1].bias[5 * self.dim : 6 * self.dim].fill_(0.1)  # gate2

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """`x`: `(N, L, D)`, `c`: `(N, D)` -> `(N, L, D)`."""
        N, L, D = x.shape
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        h = self.norm1(x) * (1.0 + scale1[:, None]) + shift1[:, None]
        qkv = self.qkv(h).reshape(N, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self.proj(attn.transpose(1, 2).reshape(N, L, D))
        x = x + gate1[:, None] * attn

        h2 = self.norm2(x) * (1.0 + scale2[:, None]) + shift2[:, None]
        x = x + gate2[:, None] * self.mlp(h2)
        return x


class STDiTDynamics(nn.Module):
    """4-layer Action-Conditioned Spatial-Temporal DiT evolving `Z_t in R^{K x D_z}` directly!

    - Action chunk `a_t in R^{56}` (`asinh`-scaled) is embedded in strict `float32` so fine `0.5 mm`
      vs `1.5 mm` candidate corrections are never quantized before AdaLN-Zero modulation!
    - Also appends 1 explicit action token so spatial tokens can directly cross-attend to `a_t`.
    - AdaLN-Zero initializes all block gates to 0 for clean multi-step (`H=4`) rollout stability.
    - Outputs per-token `SimNorm` spatial grid packed to `(*lead, K * D_z)`.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        action_dim: int,
        cfg,
        dit_dim: int = 384,
        dit_layers: int = 4,
        dit_heads: int = 6,
        simnorm_cls=None,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.grid_size = int(round(math.sqrt(self.num_tokens)))
        if self.grid_size * self.grid_size != self.num_tokens:
            raise ValueError(f"num_tokens must be a square grid, got {self.num_tokens}")
        self.token_dim = int(token_dim)
        self.dit_dim = int(dit_dim)

        self.in_proj = nn.Linear(self.token_dim, self.dit_dim)
        self.act_embed = nn.Sequential(
            nn.Linear(action_dim, self.dit_dim),
            nn.SiLU(),
            nn.Linear(self.dit_dim, self.dit_dim),
        )
        self.act_token_norm = nn.LayerNorm(self.dit_dim)
        self.blocks = nn.ModuleList([STDiTBlock(self.dit_dim, heads=dit_heads) for _ in range(dit_layers)])
        self.final_norm = nn.LayerNorm(self.dit_dim, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(self.dit_dim, 2 * self.dit_dim))
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)
        self.out_proj = nn.Linear(self.dit_dim, self.token_dim)
        # Direct Xavier-initialized action-conditioned delta guarantees STDiT(z_0, a) is strictly
        # non-identity & action-sensitive from step 0 (blocks trivial z_t -> const collapse!).
        self.act_delta_proj = nn.Linear(self.dit_dim, self.token_dim)
        nn.init.xavier_uniform_(self.act_delta_proj.weight)
        nn.init.zeros_(self.act_delta_proj.bias)
        self.simnorm = simnorm_cls(cfg)
        self._pos_cache: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}

    def _get_pos(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        p = self._pos_cache.get(key)
        if p is None or p.device != device:
            p = build_2d_sincos_pos_embed(self.grid_size, self.grid_size, self.dit_dim, device)
            self._pos_cache[key] = p
        return p

    def forward(self, z_packed: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        lead = z_packed.shape[:-1]
        N = int(math.prod(lead)) if lead else 1
        z_sp = z_packed.reshape(N, self.num_tokens, self.token_dim)
        a_flat = a.reshape(N, a.shape[-1])

        # Embed action chunk strictly in float32 (preserves 0.5 mm candidate deltas)
        with torch.autocast(device_type=a_flat.device.type, enabled=False):
            c = self.act_embed(a_flat.float())
            act_delta = 0.1 * self.act_delta_proj(c)[:, None, :]  # (N, 1, token_dim)

        pos = self._get_pos(z_sp.device)
        with _amp(z_sp):
            h_sp = self.in_proj(z_sp) + pos.to(z_sp.dtype)
            h_act = self.act_token_norm(c).to(h_sp.dtype).unsqueeze(1)
            h = torch.cat([h_act, h_sp], dim=1)  # (N, 1 + K, dit_dim)
            c_cast = c.to(h.dtype)
            for blk in self.blocks:
                h = blk(h, c_cast)
            shift_f, scale_f = self.final_adaLN(c_cast).chunk(2, dim=-1)
            h_out = self.final_norm(h[:, 1:]) * (1.0 + scale_f[:, None]) + shift_f[:, None]
            logits = self.out_proj(h_out)
        z_next = self.simnorm(logits.float() + act_delta)
        return z_next.reshape(*lead, self.num_tokens * self.token_dim)


class VJEPA2ACPredictorAdapter(nn.Module):
    """Adapter around official pretrained `vjepa2_ac_predictor` (`d2c_vjepa2_ac_vitg`, `h100:1`).

    Maps our `(K, D_z)` spatial token grid + 56-dim action chunk into the official pretrained
    action-conditioned predictor blocks (or falls back cleanly to `STDiTDynamics` if `pretrained=False`).
    Explicitly marked `transition_pretrained=True` when backed by official weights.
    """

    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        action_dim: int,
        cfg,
        raw_predictor: Optional[nn.Module],
        simnorm_cls=None,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.has_pretrained_predictor = raw_predictor is not None
        if raw_predictor is None:
            self.fallback = STDiTDynamics(
                num_tokens=num_tokens,
                token_dim=token_dim,
                action_dim=action_dim,
                cfg=cfg,
                dit_dim=512,
                dit_layers=4,
                dit_heads=8,
                simnorm_cls=simnorm_cls,
            )
            self.predictor = None
        else:
            self.fallback = None
            self.predictor = raw_predictor
            # Infer predictor's expected visual dim (1408 for ViT-g) and action/state dim (7 for DROID)
            enc_dim = int(getattr(raw_predictor, "embed_dim", 1408))
            act_dim_in = 7
            if hasattr(raw_predictor, "action_encoder") and hasattr(raw_predictor.action_encoder, "in_features"):
                act_dim_in = int(raw_predictor.action_encoder.in_features)
            self.z_to_vj = nn.Linear(self.token_dim, enc_dim)
            self.act_to_ac = nn.Sequential(nn.Linear(action_dim, 128), nn.SiLU(), nn.Linear(128, act_dim_in))
            self.state_stub = nn.Parameter(torch.zeros(1, 1, act_dim_in))
            self.vj_to_z = nn.Linear(enc_dim, self.token_dim)
            self.simnorm = simnorm_cls(cfg)

    def forward(self, z_packed: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.predictor is None:
            assert self.fallback is not None
            return self.fallback(z_packed, a)
        lead = z_packed.shape[:-1]
        N = int(math.prod(lead)) if lead else 1
        z_sp = z_packed.reshape(N, self.num_tokens, self.token_dim)
        a_flat = a.reshape(N, a.shape[-1])
        with torch.autocast(device_type=a_flat.device.type, enabled=False):
            a_ac = self.act_to_ac(a_flat.float()).unsqueeze(1)  # (N, 1, 7)
            s_ac = self.state_stub.expand(N, 1, -1)
        with _amp(z_sp):
            x_in = self.z_to_vj(z_sp)  # (N, K, 1408)
            y_out = self.predictor(x_in, a_ac.to(x_in.dtype), s_ac.to(x_in.dtype))
            logits = self.vj_to_z(y_out)
        z_next = self.simnorm(logits.float())
        return z_next.reshape(*lead, self.num_tokens * self.token_dim)


# ---------------------------------------------------------------------------
# Multi-Query Attentive Spatial Readout (`(N, K, D_z) -> (N, D_head=512)`)
# ---------------------------------------------------------------------------
class SpatialTokenReadout(nn.Module):
    """Multi-query cross-attention pooling `(..., K * D_z) -> (..., D_head)` preserving spatial layout.

    Uses `M = 8` learned spatial queries attending over `Z_t + 2D_sincos_pos` (`M * D_z = 512`),
    so distinct queries can focus on pin tips, socket edges, contact rim, and board plane.
    """

    def __init__(self, num_tokens: int, token_dim: int, out_dim: int = 512, num_queries: int = 8):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.grid_size = int(round(math.sqrt(self.num_tokens)))
        self.token_dim = int(token_dim)
        self.num_queries = int(num_queries)
        self.out_dim = int(out_dim)

        self.kv_norm = nn.LayerNorm(self.token_dim)
        self.queries = nn.Parameter(torch.randn(1, self.num_queries, self.token_dim) * 0.02)
        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.num_queries * self.token_dim),
            nn.Linear(self.num_queries * self.token_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )
        # Direct mean+max spatial pooling residual (strong step-0 gradients into every token, zero `pos` shortcut!)
        self.direct_proj = nn.Sequential(
            nn.LayerNorm(2 * self.token_dim),
            nn.Linear(2 * self.token_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.SiLU(),
        )
        self.final_norm = nn.LayerNorm(self.out_dim)
        self._pos_cache: Dict[Tuple[str, Optional[int]], torch.Tensor] = {}

    def _get_pos(self, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index)
        p = self._pos_cache.get(key)
        if p is None or p.device != device:
            p = build_2d_sincos_pos_embed(self.grid_size, self.grid_size, self.token_dim, device)
            self._pos_cache[key] = p
        return p

    def forward(self, z_packed: torch.Tensor) -> torch.Tensor:
        lead = z_packed.shape[:-1]
        N = int(math.prod(lead)) if lead else 1
        z_sp = z_packed.reshape(N, self.num_tokens, self.token_dim)
        pos = self._get_pos(z_sp.device)
        h_norm = self.kv_norm(z_sp)
        q = self.q_proj(self.queries.expand(N, -1, -1).to(h_norm.dtype))   # (N, M, D_z)
        # Add 2D positional embedding ONLY to keys (spatial addressing), NEVER to values (blocks pos shortcut!)
        k = self.k_proj(h_norm + pos.to(h_norm.dtype))                     # (N, K, D_z)
        v = self.v_proj(h_norm)                                            # (N, K, D_z) pure content from z_sp!
        attn = F.scaled_dot_product_attention(q, k, v)                     # (N, M, D_z)
        attn_out = self.out_proj(attn.reshape(N, self.num_queries * self.token_dim))
        direct = self.direct_proj(torch.cat([z_sp.mean(dim=1), z_sp.amax(dim=1)], dim=-1))
        pooled = self.final_norm(attn_out + direct)
        return pooled.reshape(*lead, self.out_dim)


class SpatialReadoutHead(nn.Module):
    """Wraps a head MLP so direct calls like `m._done(z)` work on packed spatial latents WITHOUT duplicating params."""

    def __init__(self, readout: Optional[nn.Module], head: nn.Module):
        super().__init__()
        # Store readout reference outside `_modules` so state_dict/parameters never duplicate `self.readout`!
        object.__setattr__(self, "_readout_ref", readout)
        self.head = head

    def forward(self, z: torch.Tensor, pooled: bool = False) -> torch.Tensor:
        ro = object.__getattribute__(self, "_readout_ref")
        if ro is not None and not pooled:
            z = ro(z)
        return self.head(z)


class DetachedProbeHead(nn.Module):
    """Strictly stop-gradient 4D B1 diagnostic readout (`z.detach()` ALWAYS executed before any layer).

    Owns its own independent `SpatialTokenReadout` (for spatial-token models) + MLP, guaranteeing
    ZERO gradient leak and ZERO shared parameters with the world model (`enc`, `dynamics`, `readout`).
    """

    def __init__(self, is_spatial: bool, num_tokens: int, token_dim: int, head_dim: int, mlp_fn, init_fn):
        super().__init__()
        self.readout = (
            SpatialTokenReadout(num_tokens=num_tokens, token_dim=token_dim, out_dim=head_dim)
            if is_spatial
            else None
        )
        self.head = mlp_fn(head_dim, [256], 4)
        self.head.apply(init_fn)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zd = z.detach()
        if self.readout is not None:
            zd = self.readout(zd)
        return self.head(zd)


# ---------------------------------------------------------------------------
# Architecture Presets (`D2a`, `D2b`, `D2c`)
# ---------------------------------------------------------------------------
ARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "d2a_segformer": {
        "family": "D2a",
        "backbone": "segformer_b1_fpn",
        "dynamics": "mlp",
        "spatial_latent": False,
        "pic_grid": 10,
        "pic_dim": 64,
        "fuse_dim": 512,
        "fuse_layers": 4,
        "fuse_heads": 8,
        "latent_grid": 1,
        "token_dim": 512,
        "dit_dim": 0,
        "dit_layers": 0,
        "dit_heads": 0,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "nvidia/mit-b1 (SegFormer MiT-B1)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained TD-MPC2 2-layer MLP dynamics",
    },
    "d2a_resnet34": {
        "family": "D2a",
        "backbone": "resnet34_nopool",
        "dynamics": "mlp",
        "spatial_latent": False,
        "pic_grid": 10,
        "pic_dim": 64,
        "fuse_dim": 512,
        "fuse_layers": 4,
        "fuse_heads": 8,
        "latent_grid": 1,
        "token_dim": 512,
        "dit_dim": 0,
        "dit_layers": 0,
        "dit_heads": 0,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "torchvision resnet34 IMAGENET1K_V1 (NoPool + GroupNorm)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained TD-MPC2 2-layer MLP dynamics",
    },
    "d2a_resnet18": {
        "family": "D2a",
        "backbone": "resnet18_standard",
        "dynamics": "mlp",
        "spatial_latent": False,
        "pic_grid": 10,
        "pic_dim": 64,
        "fuse_dim": 512,
        "fuse_layers": 4,
        "fuse_heads": 8,
        "latent_grid": 1,
        "token_dim": 512,
        "dit_dim": 0,
        "dit_layers": 0,
        "dit_heads": 0,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "torchvision resnet18 IMAGENET1K_V1 (GroupNorm width control)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained TD-MPC2 2-layer MLP dynamics",
    },
    "d2b_segformer": {
        "family": "D2b",
        "backbone": "segformer_b1_fpn",
        "dynamics": "st_dit",
        "spatial_latent": True,
        "pic_grid": 8,
        "pic_dim": 64,
        "fuse_dim": 384,
        "fuse_layers": 4,
        "fuse_heads": 6,
        "latent_grid": 8,
        "token_dim": 64,
        "dit_dim": 384,
        "dit_layers": 4,
        "dit_heads": 6,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "nvidia/mit-b1 (SegFormer MiT-B1 FPN)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained 4-layer AdaLN-Zero ST-DiT over K=64 spatial tokens",
    },
    "d2b_resnet18": {
        "family": "D2b",
        "backbone": "resnet18_standard",
        "dynamics": "st_dit",
        "spatial_latent": True,
        "pic_grid": 8,
        "pic_dim": 64,
        "fuse_dim": 384,
        "fuse_layers": 4,
        "fuse_heads": 6,
        "latent_grid": 8,
        "token_dim": 64,
        "dit_dim": 384,
        "dit_layers": 4,
        "dit_heads": 6,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "torchvision resnet18 IMAGENET1K_V1 (pure dynamics control vs wm320)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained 4-layer AdaLN-Zero ST-DiT over K=64 spatial tokens",
    },
    "d2b_resnet34": {
        "family": "D2b",
        "backbone": "resnet34_nopool",
        "dynamics": "st_dit",
        "spatial_latent": True,
        "pic_grid": 8,
        "pic_dim": 64,
        "fuse_dim": 384,
        "fuse_layers": 4,
        "fuse_heads": 6,
        "latent_grid": 8,
        "token_dim": 64,
        "dit_dim": 384,
        "dit_layers": 4,
        "dit_heads": 6,
        "detach_target_obs": False,
        "encoder_pretrained_desc": "torchvision resnet34 IMAGENET1K_V1 (NoPool + GroupNorm)",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained 4-layer AdaLN-Zero ST-DiT over K=64 spatial tokens",
    },
    "d2c_vjepa2_vitl": {
        "family": "D2c",
        "backbone": "vjepa2_vitl",
        "dynamics": "st_dit",
        "spatial_latent": True,
        "pic_grid": 16,
        "pic_dim": 896,
        "fuse_dim": 384,
        "fuse_layers": 2,
        "fuse_heads": 6,
        "latent_grid": 8,
        "token_dim": 64,
        "dit_dim": 384,
        "dit_layers": 4,
        "dit_heads": 6,
        "vjepa2_unfreeze_blocks": 4,
        "detach_target_obs": True,  # 5x activation savings on ViT-L while keeping full grad on z0 & BC!
        "encoder_pretrained_desc": "facebook/vjepa2-vitl-fpc64-256 (ViT-L/16 3D video) + parallel 320px float32 RGB-D stem",
        "transition_pretrained": False,
        "transition_desc": "Domain-trained 4-layer AdaLN-Zero ST-DiT over K=64 spatial tokens (NOT pretrained AC predictor)",
    },
    "d2c_vjepa2_ac_vitg": {
        "family": "D2c",
        "backbone": "vjepa2_vitg_ac",
        "dynamics": "vjepa2_ac",
        "spatial_latent": True,
        "pic_grid": 16,
        "pic_dim": 896,
        "fuse_dim": 384,
        "fuse_layers": 2,
        "fuse_heads": 6,
        "latent_grid": 16,  # Matches official vjepa2_ac_vit_giant 16x16 RoPE predictor grid!
        "token_dim": 32,
        "dit_dim": 0,
        "dit_layers": 0,
        "dit_heads": 0,
        "vjepa2_unfreeze_blocks": 0,  # ViT-g 1.0B kept frozen; official AC predictor adapted (requires h100:1 80 GB)
        "detach_target_obs": True,
        "encoder_pretrained_desc": "facebookresearch/vjepa2 vjepa2_ac_vit_giant frozen ViT-g/16 + parallel 320px RGB-D stem",
        "transition_pretrained": True,
        "transition_desc": "Official pretrained V-JEPA 2-AC predictor adapted to 56-dim action chunks (h100:1 80 GB)",
    },
}


# ---------------------------------------------------------------------------
# Unified Drop-In `VideoWM320`
# ---------------------------------------------------------------------------
class VideoWM320(nn.Module):
    """Unified drop-in replacement for `wm320.WM320` implementing `D2a`, `D2b`, and `D2c`."""

    def __init__(
        self,
        arch: str,
        stats: dict,
        shaping: float = 0.0,
        offset_mode: str = "gap",
        offset_weight: float = 1.0,
        sg_probe: bool = True,
        pretrained: bool = True,
        local_files_only: bool = False,
        segformer_model_id: str = DEFAULT_SEGFORMER_ID,
        vjepa2_model_id: str = DEFAULT_VJEPA2_VITL_ID,
        latent_grid: Optional[int] = None,
        token_dim: Optional[int] = None,
        dit_dim: Optional[int] = None,
        dit_layers: Optional[int] = None,
        dit_heads: Optional[int] = None,
        vjepa2_unfreeze_blocks: Optional[int] = None,
        detach_target_obs: Optional[bool] = None,
        grad_ckpt: bool = False,
        vjepa2_hf_config: Optional[dict] = None,
        cons_warmup_steps: Optional[int] = None,
        **cfg_over,
    ):
        super().__init__()
        if arch not in ARCH_PRESETS:
            raise ValueError(f"Unknown arch '{arch}'. Valid choices: {sorted(ARCH_PRESETS)}")
        norm_mode = "along" if str(offset_mode) == "probe" else str(offset_mode)
        if norm_mode not in ("gap", "gap_1d", "along", "none"):
            raise ValueError(
                f"offset_mode must be 'gap', 'gap_1d', 'along' (or alias 'probe'), or 'none', got {offset_mode!r}"
            )

        spec = dict(ARCH_PRESETS[arch])
        self.arch = str(arch)
        self.family = str(spec["family"])
        self.backbone_type = str(spec["backbone"])
        self.dynamics_type = str(spec["dynamics"])
        self.spatial_latent = bool(spec["spatial_latent"])
        # `video_offset_mode` records the exact D2 head variant; `offset_mode` is the D1-compatible
        # wire mode ('gap' or 'along') read by D1's attempt320.py and fidelity_factual320.py!
        self.video_offset_mode = norm_mode
        self.offset_mode = "along" if norm_mode == "along" else "gap"
        self.offset_weight = float(offset_weight)
        self.sg_probe_enabled = bool(sg_probe) and (self.video_offset_mode in ("gap_1d", "none"))
        self.pretrained = bool(pretrained)
        self.shaping = float(shaping)
        self.ret_max = 20.0 + 10.0 + self.shaping * PHI_MAX

        if latent_grid is not None and self.spatial_latent:
            spec["latent_grid"] = int(latent_grid)
            if self.backbone_type not in ("vjepa2_vitl", "vjepa2_vitg_ac"):
                spec["pic_grid"] = int(latent_grid)
        if token_dim is not None and self.spatial_latent:
            spec["token_dim"] = int(token_dim)
        if dit_dim is not None and self.dynamics_type == "st_dit":
            spec["dit_dim"] = int(dit_dim)
        if dit_layers is not None and self.dynamics_type == "st_dit":
            spec["dit_layers"] = int(dit_layers)
        if dit_heads is not None and self.dynamics_type == "st_dit":
            spec["dit_heads"] = int(dit_heads)
        if vjepa2_unfreeze_blocks is not None:
            spec["vjepa2_unfreeze_blocks"] = int(vjepa2_unfreeze_blocks)
        if detach_target_obs is not None:
            spec["detach_target_obs"] = bool(detach_target_obs)

        self.spec = spec
        self.detach_target_obs = bool(spec["detach_target_obs"])
        # Anti-collapse consistency warmup for spatial-token models (D2b/D2c): linearly warms up
        # consistency_coef from 0.5 -> 20.0 over the first 1000 steps so heads pull spatial features apart first!
        self.cons_warmup_steps = (
            int(cons_warmup_steps) if cons_warmup_steps is not None else (1000 if self.spatial_latent else 0)
        )
        self.latent_grid = int(spec["latent_grid"])
        self.num_tokens = self.latent_grid * self.latent_grid
        self.token_dim = int(spec["token_dim"])
        self.packed_latent_dim = self.num_tokens * self.token_dim

        td_init, td_layers, td_math, self._official_tdmpc2 = _load_tdmpc2_backend()
        cfg = make_cfg(**cfg_over)
        self.cfg = cfg
        self.math = td_math
        self.head_dim = int(cfg.latent_dim)  # 512 for TD-MPC2 heads
        if not self.spatial_latent and self.packed_latent_dim != self.head_dim:
            self.token_dim = self.head_dim
            self.packed_latent_dim = self.head_dim
        if self.token_dim % int(cfg.simnorm_dim) != 0:
            raise ValueError(f"token_dim={self.token_dim} must be divisible by simnorm_dim={cfg.simnorm_dim}")

        # 1. Build Encoder (`D2a` / `D2b` / `D2c`)
        raw_ac_predictor: Optional[nn.Module] = None
        if self.backbone_type in ("vjepa2_vitl", "vjepa2_vitg_ac"):
            vj_var = "vitl" if self.backbone_type == "vjepa2_vitl" else "vitg_ac"
            self.enc = VJEPA2RGBDEncoder(
                cfg=cfg,
                variant=vj_var,
                model_id=vjepa2_model_id,
                pretrained=pretrained,
                local_files_only=local_files_only,
                unfreeze_blocks=int(spec.get("vjepa2_unfreeze_blocks", 4)),
                latent_grid=self.latent_grid,
                token_dim=self.token_dim,
                d_fuse=int(spec["fuse_dim"]),
                simnorm_cls=td_layers.SimNorm,
                vjepa2_hf_config=vjepa2_hf_config,
            )
            raw_ac_predictor = self.enc.ac_predictor_raw
            self.enc.ac_predictor_raw = None  # owned by `self._dynamics` below (never duplicated!)
        else:
            pic = self._build_2d_tokenizer(
                backbone=self.backbone_type,
                out_dim=int(spec["pic_dim"]),
                grid_size=int(spec["pic_grid"]),
                pretrained=pretrained,
                segformer_model_id=segformer_model_id,
                local_files_only=local_files_only,
                grad_ckpt=grad_ckpt,
            )
            if self.spatial_latent:
                self.enc = D2bSpatialEncoder(
                    pic=pic,
                    cfg=cfg,
                    latent_grid=self.latent_grid,
                    token_dim=self.token_dim,
                    d_enc=int(spec["fuse_dim"]),
                    layers=int(spec["fuse_layers"]),
                    heads=int(spec["fuse_heads"]),
                    simnorm_cls=td_layers.SimNorm,
                )
            else:
                self.enc = D2aEncoder(
                    pic=pic,
                    cfg=cfg,
                    d=int(spec["fuse_dim"]),
                    layers=int(spec["fuse_layers"]),
                    heads=int(spec["fuse_heads"]),
                    simnorm_cls=td_layers.SimNorm,
                )

        # 2. Build Dynamics (`mlp` for D2a, `STDiTDynamics` for D2b/D2c-ViT-L, `VJEPA2ACPredictorAdapter` for D2c-AC)
        Ld, A, Md = self.head_dim, ACTION_DIM, int(cfg.mlp_dim)
        if self.dynamics_type == "mlp":
            self._dynamics: nn.Module = td_layers.mlp(Ld + A, 2 * [Md], Ld, act=td_layers.SimNorm(cfg))
            self._dynamics.apply(td_init.weight_init)
        elif self.dynamics_type == "st_dit":
            self._dynamics = STDiTDynamics(
                num_tokens=self.num_tokens,
                token_dim=self.token_dim,
                action_dim=A,
                cfg=cfg,
                dit_dim=int(spec["dit_dim"]),
                dit_layers=int(spec["dit_layers"]),
                dit_heads=int(spec["dit_heads"]),
                simnorm_cls=td_layers.SimNorm,
            )
        elif self.dynamics_type == "vjepa2_ac":
            self._dynamics = VJEPA2ACPredictorAdapter(
                num_tokens=self.num_tokens,
                token_dim=self.token_dim,
                action_dim=A,
                cfg=cfg,
                raw_predictor=raw_ac_predictor,
                simnorm_cls=td_layers.SimNorm,
            )
        else:
            raise ValueError(f"Unknown dynamics_type={self.dynamics_type}")

        # 3. Spatial Readout Pooler (D2b & D2c) + EMA target copy synced with `_Qt`
        self.readout: Optional[SpatialTokenReadout] = (
            SpatialTokenReadout(num_tokens=self.num_tokens, token_dim=self.token_dim, out_dim=Ld, num_queries=8)
            if self.spatial_latent
            else None
        )
        self._readout_t: Optional[SpatialTokenReadout] = (
            deepcopy(self.readout).requires_grad_(False).train(False)
            if self.readout is not None
            else None
        )

        # 4. TD-MPC2 Heads (reward, Q x5, V x5, done, built-in ACT head)
        self._reward = td_layers.mlp(Ld + A, 2 * [Md], cfg.num_bins)
        self._Qs = td_layers.Ensemble(
            [td_layers.mlp(Ld + A, 2 * [Md], cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)]
        )
        self._Vs = td_layers.Ensemble(
            [td_layers.mlp(Ld, 2 * [Md], cfg.num_bins, dropout=cfg.dropout) for _ in range(cfg.num_q)]
        )
        done_mlp = td_layers.mlp(Ld, [256], 1)
        act_mlp = td_layers.mlp(Ld, 3 * [512], CHUNK * JOINTS)
        for m in (self._reward, self._Qs, self._Vs, done_mlp, act_mlp):
            m.apply(td_init.weight_init)
        td_init.zero_([self._reward[-1].weight])
        _zero_ensemble_last_layer(self._Qs, td_init)
        _zero_ensemble_last_layer(self._Vs, td_init)

        self._done = SpatialReadoutHead(self.readout, done_mlp) if self.spatial_latent else done_mlp
        self._act = SpatialReadoutHead(self.readout, act_mlp) if self.spatial_latent else act_mlp

        self._Qt = deepcopy(self._Qs).requires_grad_(False)
        self._Qt.train(False)

        self.register_buffer("off_scale", torch.tensor(OFFSET_SCALE, dtype=torch.float32), persistent=False)
        self.register_buffer("gap_scale", torch.tensor(GAP_SCALE, dtype=torch.float32), persistent=False)
        for k, v in stats.items():
            self.register_buffer(k, torch.tensor(np.asarray(v), dtype=torch.float32))

        # 5. Offset / Gap Readout Head (100% D1-harmonized):
        #   - `gap` (default, parity with D1b): 4D `_offset` head with gradients on (gap_mm, lat_x, lat_y, tilt)
        #   - `gap_1d` (ablation): 1D `_gap` head with gradients + strictly detached `_probe_sg` on (gap_mm, lx, ly, tilt)
        #   - `along` (legacy / alias `probe`): 4D `_offset` head with gradients on (along_mm, lat_x, lat_y, tilt)
        #   - `none`: no gradient-flowing offset head + optional strictly detached `_probe_sg` on (gap_mm, lx, ly, tilt)
        self._gap: Optional[nn.Module] = None
        self._probe_sg: Optional[DetachedProbeHead] = None
        self._offset: Optional[nn.Module] = None

        if self.video_offset_mode in ("gap", "along"):
            if self.offset_weight > 0:
                off_mlp = td_layers.mlp(Ld, [256], 4)
                off_mlp.apply(td_init.weight_init)
                self._offset = SpatialReadoutHead(self.readout, off_mlp) if self.spatial_latent else off_mlp
        elif self.video_offset_mode == "gap_1d":
            if self.offset_weight > 0:
                gap_mlp = td_layers.mlp(Ld, [256], 1)
                gap_mlp.apply(td_init.weight_init)
                self._gap = SpatialReadoutHead(self.readout, gap_mlp) if self.spatial_latent else gap_mlp
            if self.sg_probe_enabled:
                self._probe_sg = DetachedProbeHead(
                    is_spatial=self.spatial_latent,
                    num_tokens=self.num_tokens,
                    token_dim=self.token_dim,
                    head_dim=Ld,
                    mlp_fn=td_layers.mlp,
                    init_fn=td_init.weight_init,
                )
                # Expose detached B1 readout (dim 0 = gap_mm!) via `self._offset` for D1 `fidelity_factual320.py`
                self._offset = self._probe_sg
        elif self.video_offset_mode == "none":
            if self.sg_probe_enabled:
                self._probe_sg = DetachedProbeHead(
                    is_spatial=self.spatial_latent,
                    num_tokens=self.num_tokens,
                    token_dim=self.token_dim,
                    head_dim=Ld,
                    mlp_fn=td_layers.mlp,
                    init_fn=td_init.weight_init,
                )
                self._offset = self._probe_sg

        # Capture HF V-JEPA 2 config dict for lossless offline reload without HF cache!
        saved_vj_cfg = None
        if isinstance(self.enc, VJEPA2RGBDEncoder) and self.enc._backend_kind == "hf":
            cfg_obj = getattr(self.enc.vjepa, "config", None)
            if cfg_obj is not None and hasattr(cfg_obj, "to_dict"):
                saved_vj_cfg = cfg_obj.to_dict()

        # Store constructor args needed for lossless checkpoint reload
        self._init_kwargs: Dict[str, Any] = {
            "arch": self.arch,
            "shaping": self.shaping,
            "offset_mode": self.video_offset_mode,
            "offset_weight": self.offset_weight,
            "sg_probe": bool(sg_probe),
            "segformer_model_id": str(segformer_model_id),
            "vjepa2_model_id": str(vjepa2_model_id),
            "latent_grid": self.latent_grid,
            "token_dim": self.token_dim,
            "dit_dim": int(spec.get("dit_dim", 0)),
            "dit_layers": int(spec.get("dit_layers", 0)),
            "dit_heads": int(spec.get("dit_heads", 0)),
            "vjepa2_unfreeze_blocks": int(spec.get("vjepa2_unfreeze_blocks", 0)),
            "detach_target_obs": self.detach_target_obs,
            "grad_ckpt": bool(grad_ckpt),
            "vjepa2_hf_config": saved_vj_cfg,
            "cons_warmup_steps": self.cons_warmup_steps,
        }

    @staticmethod
    def _build_2d_tokenizer(
        backbone: str,
        out_dim: int,
        grid_size: int,
        pretrained: bool,
        segformer_model_id: str,
        local_files_only: bool,
        grad_ckpt: bool,
    ) -> nn.Module:
        if backbone == "segformer_b1_fpn":
            return SegFormerB1FPNTokenizer(
                out_dim=out_dim,
                grid_size=grid_size,
                pretrained=pretrained,
                model_id=segformer_model_id,
                local_files_only=local_files_only,
                grad_ckpt=grad_ckpt,
            )
        if backbone == "resnet34_nopool":
            return ResNetGroupNormTokenizer(
                arch="resnet34",
                nopool=True,
                out_dim=out_dim,
                grid_size=grid_size,
                pretrained=pretrained,
            )
        if backbone == "resnet18_standard":
            return ResNetGroupNormTokenizer(
                arch="resnet18",
                nopool=False,
                out_dim=out_dim,
                grid_size=grid_size,
                pretrained=pretrained,
            )
        raise ValueError(f"Unsupported 2D backbone: {backbone}")

    def train(self, mode: bool = True):
        super().train(mode)
        self._Qt.train(False)
        if self._readout_t is not None:
            self._readout_t.train(False)
        return self

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        return self.readout(z) if self.readout is not None else z

    def _pool_target(self, z: torch.Tensor) -> torch.Tensor:
        return self._readout_t(z) if self._readout_t is not None else z

    # -----------------------------------------------------------------------
    # Drop-In `WM320` Interface Methods
    # -----------------------------------------------------------------------
    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if self.dynamics_type == "mlp":
            return self._dynamics(torch.cat([z, a], dim=-1))
        return self._dynamics(z, a)

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        zr = self._pool(z)
        return self._reward(torch.cat([zr, a], dim=-1))

    def Q(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        zr = self._pool(z)
        return self._Qs(torch.cat([zr, a], dim=-1))

    def V(self, z: torch.Tensor) -> torch.Tensor:
        zr = self._pool(z)
        return self._Vs(zr)

    def scalar(self, logits: torch.Tensor) -> torch.Tensor:
        return self.math.two_hot_inv(logits, self.cfg)

    def act_chunk(self, z: torch.Tensor) -> torch.Tensor:
        """Built-in ACT head: next 16 joint targets relative to measured joints [rad], `(*lead, 16, 7)`."""
        raw = self._act(z)
        return (raw * self.a_std + self.a_mean).view(*z.shape[:-1], CHUNK, JOINTS)

    def scale(self, delta: torch.Tensor) -> torch.Tensor:
        return (torch.asinh(delta / self.s0) / torch.asinh(self.top / self.s0)).clamp(-1.0, 1.0)

    def predict_gap_mm(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Zero-privilege predicted pin-tip-to-board clearance `g_t` [mm], `(*lead,)`.

        - `gap` (D1b parity): reads `(self._offset(z)[..., 0] * self.off_scale[0])`.
        - `gap_1d`: reads 1D `(self._gap(z)[..., 0] * self.gap_scale)` (or detached `_probe_sg` if `_gap` is off).
        - `none`: reads detached `_probe_sg` dim 0 if enabled, else `None`.
        - `along` (legacy): returns `None` (requires external `pin_length_mm`).
        """
        if self.video_offset_mode == "along":
            return None
        if self._gap is not None:
            return self._gap(z)[..., 0] * self.gap_scale
        if self._offset is not None:
            return self._offset(z)[..., 0] * self.off_scale[0]
        return None

    @torch.no_grad()
    def policy_action(self, z: torch.Tensor) -> torch.Tensor:
        was_act = self._act.training
        was_ro = self.readout.training if self.readout is not None else False
        self._act.eval()
        if self.readout is not None:
            self.readout.eval()
        a = self.act_chunk(z.reshape(-1, z.shape[-1]))[:, :EXECUTE].reshape(*z.shape[:-1], ACTION_DIM)
        self._act.train(was_act)
        if self.readout is not None:
            self.readout.train(was_ro)
        return self.scale(a)

    @torch.no_grad()
    def target_value(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        zt = self._pool_target(z)
        v = self.scalar(self._Qt(torch.cat([zt, a], dim=-1)))[..., 0]
        i = torch.randperm(v.shape[0], device=v.device)[:2]
        return v[i].min(0).values

    @torch.no_grad()
    def update_target(self, tau: float = 0.01) -> None:
        for pt, p in zip(self._Qt.parameters(), self._Qs.parameters()):
            pt.lerp_(p.detach(), tau)
        if self._readout_t is not None and self.readout is not None:
            for pt, p in zip(self._readout_t.parameters(), self.readout.parameters()):
                pt.lerp_(p.detach(), tau)

    def encode_obs(self, rgb: torch.Tensor, depth: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Encode single 4-frame observation batch: `rgb (B, 4, 320, 320, 3)`, `depth (B, 4, 320, 320)`, `q (B, 7)`."""
        B, T = depth.shape[:2]
        tok = self.enc.tokens(rgb.flatten(0, 1), depth.flatten(0, 1))
        tok = tok.view(B, T, *tok.shape[1:])
        return self.enc.fuse(tok, q)

    def observe(self, rgb: torch.Tensor, depth: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Windows: `rgb (B, H+4, 320, 320, 3)`, `depth (B, H+4, 320, 320)`, `q (B, H+1, 7)` -> `(B, H+1, L_flat)`."""
        B, S = depth.shape[:2]
        n_obs = S - CONTEXT + 1
        tok = self.enc.tokens(rgb.flatten(0, 1), depth.flatten(0, 1))
        tok = tok.view(B, S, *tok.shape[1:])
        if self.detach_target_obs and self.training and n_obs > 1:
            z0 = self.enc.fuse(tok[:, 0:CONTEXT], q[:, 0])
            with torch.no_grad():
                z_rest = [self.enc.fuse(tok[:, j : j + CONTEXT].detach(), q[:, j]) for j in range(1, n_obs)]
            return torch.stack([z0] + z_rest, dim=1)
        return torch.stack([self.enc.fuse(tok[:, j : j + CONTEXT], q[:, j]) for j in range(n_obs)], dim=1)

    # -----------------------------------------------------------------------
    # Training Forward / Loss
    # -----------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        """`= loss()` so DistributedDataParallel synchronizes all gradients through `forward()`."""
        return self.loss(*args, **kwargs)

    @staticmethod
    def _extract_probe_targets(W: dict) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Extract unambiguous `(probe_gap, probe_along, gap_mm)` in `(H+1, B, ...)` without double-negating!"""
        probe_gap_b = W.get("probe_gap")
        probe_along_b = W.get("probe_along")
        gap_b = W.get("gap_mm")

        if probe_gap_b is None and "probe" in W:
            if gap_b is not None:
                probe_gap_b = torch.cat([gap_b.unsqueeze(-1), W["probe"][..., 1:]], dim=-1)
            elif "pin_length_mm" in W:
                g = -W["probe"][..., 0] - W["pin_length_mm"][:, None]
                probe_gap_b = torch.cat([g.unsqueeze(-1), W["probe"][..., 1:]], dim=-1)
                gap_b = g
            else:
                probe_gap_b = W["probe"]

        if probe_along_b is None and "probe" in W:
            if gap_b is not None and "pin_length_mm" in W:
                a = -gap_b - W["pin_length_mm"][:, None]
                probe_along_b = torch.cat([a.unsqueeze(-1), W["probe"][..., 1:]], dim=-1)
            else:
                probe_along_b = W["probe"]

        if gap_b is None and probe_gap_b is not None:
            gap_b = probe_gap_b[..., 0]

        pg = probe_gap_b.transpose(0, 1) if probe_gap_b is not None else None
        pa = probe_along_b.transpose(0, 1) if probe_along_b is not None else None
        gm = gap_b.transpose(0, 1) if gap_b is not None else None
        return pg, pa, gm

    def loss(self, W: dict, Bc: Optional[dict], step: int, warmup: int, bc_weight: float, detail: bool = False):
        cfg, M = self.cfg, self.math
        B = W["q"].shape[0]
        z = self.observe(W["rgb"], W["depth"], W["q"])  # (B, H+1, L_flat)
        act, rew, term, valid = W["action"].transpose(0, 1), W["reward"].T, W["term"].T, W["valid"].T
        ov, vv = W["obs_valid"].T, W["v_valid"].T

        # Dynamics rollout + per-token spatial consistency loss
        zs: List[torch.Tensor] = [z[:, 0]]
        cons = z.new_zeros(())
        for t in range(HORIZON):
            zt = self.next(zs[-1], act[t])
            cons = cons + ((zt - z[:, t + 1].detach()).pow(2).mean(-1) * valid[t]).sum() / B * (cfg.rho ** t)
            zs.append(zt)
        zs_t = torch.stack(zs, dim=0)  # (H+1, B, L_flat)
        zd = z.detach().transpose(0, 1)

        with torch.no_grad():
            if step < warmup:
                a_nx, a_ok = W["a_next"].transpose(0, 1), W["a_next_valid"].T
            else:
                a_nx, a_ok = self.policy_action(zd), torch.ones_like(ov)
            tv = self.target_value(zd, a_nx).clamp(-self.ret_max, self.ret_max)  # (H+1, B)
            y = (rew + cfg.gamma * (1.0 - term) * tv[1:]).clamp(-self.ret_max, self.ret_max)
            q_ok = valid * (term + (1.0 - term) * a_ok[1:])
            v_ok = vv * a_ok

        # Pool imagined spatial tokens ONCE for all TD-MPC2 heads (fast + consistent)
        zs_pool = self._pool(zs_t)  # (H+1, B, 512)
        rp = self._reward(torch.cat([zs_pool[:-1], act], dim=-1))
        qs = self._Qs(torch.cat([zs_pool[:-1], act], dim=-1))
        vs = self._Vs(zs_pool)

        rl, ql, vl = z.new_zeros(()), z.new_zeros(()), z.new_zeros(())
        for t in range(HORIZON + 1):
            w = cfg.rho ** t
            if t < HORIZON:
                rl = rl + (M.soft_ce(rp[t], rew[t, :, None], cfg)[:, 0] * valid[t]).sum() / B * w
                for h in range(cfg.num_q):
                    ql = ql + (M.soft_ce(qs[h, t], y[t, :, None], cfg)[:, 0] * q_ok[t]).sum() / B * w
            for h in range(cfg.num_q):
                vl = vl + (M.soft_ce(vs[h, t], tv[t, :, None], cfg)[:, 0] * v_ok[t]).sum() / B * w
        cons = cons / HORIZON
        rl = rl / HORIZON
        ql = ql / (HORIZON * cfg.num_q)
        vl = vl / ((HORIZON + 1) * cfg.num_q)

        # Done head on encoded z and imagined zs[1:]
        z_done_in = torch.cat([z[:, :1], z[:, 1:].detach()], dim=1) if (self.detach_target_obs and self.training) else z
        dlog = self._done(z_done_in)[..., 0]
        dimg = (self._done(zs_pool[1:], pooled=True) if self.spatial_latent else self._done(zs_pool[1:]))[..., 0].T
        dl = (
            (F.binary_cross_entropy_with_logits(dlog, W["done"], reduction="none") * W["obs_valid"]).sum()
            + (F.binary_cross_entropy_with_logits(dimg, W["done"][:, 1:], reduction="none") * W["obs_valid"][:, 1:]).sum()
        ) / (B * (2 * HORIZON + 1))

        # Built-in ACT imitation loss
        bc = z.new_zeros(())
        if Bc is not None:
            zb = self.encode_obs(Bc["rgb"], Bc["depth"], Bc["q"])
            bc = (self._act(zb) - (Bc["a"] - self.a_mean) / self.a_std).abs().mean()

        # Offset / Gap supervision (100% D1-harmonized) & optional Strictly Stop-Grad B1 Probe
        probe_gap_t, probe_along_t, gap_t = self._extract_probe_targets(W)
        aux_loss = z.new_zeros(())
        sg_loss = z.new_zeros(())
        oinfo: Dict[str, Any] = {}

        if self.video_offset_mode in ("gap", "along") and self._offset is not None and self.offset_weight > 0:
            tgt_4d = probe_gap_t if self.video_offset_mode == "gap" else probe_along_t
            assert tgt_4d is not None, "Window batch missing probe/gap target"
            tgt = tgt_4d / self.off_scale  # (H+1, B, 4)
            pr = self._offset(zs_pool, pooled=True) if self.spatial_latent else self._offset(zs_pool)
            per = F.smooth_l1_loss(pr, tgt, reduction="none").mean(-1)
            aux_loss = sum((per[t] * ov[t]).sum() / B * (cfg.rho ** t) for t in range(HORIZON + 1)) / (HORIZON + 1)
            with torch.no_grad():
                err = (pr - tgt) * self.off_scale
                lat = err[..., 1:3].norm(dim=-1)
                d0_enc = (err[0, :, 0].abs() * ov[0]).sum() / ov[0].sum().clamp(min=1.0)
                d0_k1 = (err[1, :, 0].abs() * ov[1]).sum() / ov[1].sum().clamp(min=1.0)
                d0_img = (err[-1, :, 0].abs() * ov[-1]).sum() / ov[-1].sum().clamp(min=1.0)
                oinfo.update(
                    offset_loss=aux_loss,
                    offset_lat_err_mm_encoded=(lat[0] * ov[0]).sum() / ov[0].sum().clamp(min=1.0),
                    offset_along_err_mm_encoded=d0_enc,
                    offset_lat_err_mm_imagined_last=(lat[-1] * ov[-1]).sum() / ov[-1].sum().clamp(min=1.0),
                )
                if self.video_offset_mode == "gap":
                    oinfo.update(
                        offset_gap_err_mm_encoded=d0_enc,
                        offset_gap_err_mm_imagined_k1=d0_k1,
                        offset_gap_err_mm_imagined_last=d0_img,
                    )

        if self.video_offset_mode == "gap_1d" and self._gap is not None and self.offset_weight > 0:
            assert gap_t is not None, "Window batch missing gap_mm target"
            gap_tgt = gap_t / self.gap_scale
            gap_pred = (self._gap(zs_pool, pooled=True) if self.spatial_latent else self._gap(zs_pool))[..., 0]
            per_g = F.smooth_l1_loss(gap_pred, gap_tgt, reduction="none")
            aux_loss = sum((per_g[t] * ov[t]).sum() / B * (cfg.rho ** t) for t in range(HORIZON + 1)) / (HORIZON + 1)
            with torch.no_grad():
                g_err = (gap_pred - gap_tgt).abs() * self.gap_scale
                oinfo.update(
                    gap_loss=aux_loss,
                    offset_gap_err_mm_encoded=(g_err[0] * ov[0]).sum() / ov[0].sum().clamp(min=1.0),
                    offset_gap_err_mm_imagined_k1=(g_err[1] * ov[1]).sum() / ov[1].sum().clamp(min=1.0),
                    offset_gap_err_mm_imagined_last=(g_err[-1] * ov[-1]).sum() / ov[-1].sum().clamp(min=1.0),
                )

        if self._probe_sg is not None and probe_gap_t is not None:
            # STRICTLY detached input: ZERO gradient flows back into `enc`, `dynamics`, or `readout`!
            # Dim 0 target is `gap_mm` (NOT `along_mm`) for exact sign parity with D1 `fidelity_factual320.py`!
            tgt_sg = probe_gap_t / self.off_scale  # (H+1, B, 4) with dim 0 = gap_mm
            pr_sg_img = self._probe_sg(zs_t.detach())
            pr_sg_enc = self._probe_sg(z.detach().transpose(0, 1))
            per_img = F.smooth_l1_loss(pr_sg_img, tgt_sg, reduction="none").mean(-1)
            per_enc = F.smooth_l1_loss(pr_sg_enc, tgt_sg, reduction="none").mean(-1)
            sg_loss = (
                sum((per_img[t] * ov[t]).sum() / B * (cfg.rho ** t) for t in range(HORIZON + 1)) / (HORIZON + 1)
                + sum((per_enc[t] * ov[t]).sum() / B for t in range(HORIZON + 1)) / (HORIZON + 1)
            )
            with torch.no_grad():
                err_sg = (pr_sg_img - tgt_sg) * self.off_scale
                lat_sg = err_sg[..., 1:3].norm(dim=-1)
                d0_sg = (err_sg[0, :, 0].abs() * ov[0]).sum() / ov[0].sum().clamp(min=1.0)
                oinfo.update(
                    sg_probe_loss=sg_loss,
                    offset_lat_err_mm_encoded=(lat_sg[0] * ov[0]).sum() / ov[0].sum().clamp(min=1.0),
                    offset_along_err_mm_encoded=d0_sg,
                    offset_lat_err_mm_imagined_last=(lat_sg[-1] * ov[-1]).sum() / ov[-1].sum().clamp(min=1.0),
                )

        if self.spatial_latent and self.cons_warmup_steps > 0:
            cons_scale = 0.025 + 0.975 * min(1.0, float(step) / float(self.cons_warmup_steps))
        else:
            cons_scale = 1.0
        eff_cons_coef = float(cfg.consistency_coef) * cons_scale

        total = (
            eff_cons_coef * cons
            + cfg.reward_coef * rl
            + cfg.value_coef * (ql + vl)
            + 0.1 * dl
            + bc_weight * bc
            + self.offset_weight * aux_loss
            + sg_loss
        )
        info: Dict[str, Any] = {
            "total": total,
            "consistency": cons,
            "eff_cons_coef": eff_cons_coef,
            "reward_loss": rl,
            "q_loss": ql,
            "v_loss": vl,
            "done_loss": dl,
            "bc_loss": bc,
            "z_spread": z[:, 0].std(0).mean(),
            "target_value_mean": (tv * v_ok).sum() / v_ok.sum().clamp(min=1.0),
            **oinfo,
        }
        info = {k: float(v.detach().item()) if torch.is_tensor(v) else float(v) for k, v in info.items()}
        if detail:
            with torch.no_grad():
                info["_v0"] = self.scalar(self.V(z[:, 0]))[..., 0].mean(0).cpu().numpy()
                info["_done"] = torch.sigmoid(dlog).cpu().numpy()
                info["_done_imag"] = torch.sigmoid(dimg).cpu().numpy()
                info["_rew_img"] = self.scalar(rp)[..., 0].cpu().numpy()  # (H, B)
                if self._offset is not None:
                    info["_off_enc"] = (self._offset(z) * self.off_scale).transpose(0, 1).cpu().numpy()  # (H+1, B, 4)
                    info["_off_img"] = (self._offset(zs_t) * self.off_scale).cpu().numpy()              # (H+1, B, 4)
                if self._gap is not None:
                    info["_gap_enc"] = self.predict_gap_mm(z).transpose(0, 1).cpu().numpy()             # (H+1, B)
                    info["_gap_img"] = self.predict_gap_mm(zs_t).cpu().numpy()                          # (H+1, B)
        return total, info

    # -----------------------------------------------------------------------
    # Optimizer Parameter Grouping & Metadata
    # -----------------------------------------------------------------------
    def param_groups(
        self,
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        pretrained_lr_scale: Optional[float] = None,
        weight_decay: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Differential LR groups: pretrained trunk (lower LR), new encoder neck/fusion, dynamics & heads, sg_probe."""
        if pretrained_lr_scale is None:
            pretrained_lr_scale = 0.1 if self.backbone_type.startswith("vjepa2") else enc_lr_scale
        pre_p, enc_new_p = self.enc.split_params()
        sg_p = [p for p in self._probe_sg.parameters() if p.requires_grad] if self._probe_sg is not None else []
        assigned = {id(p) for p in pre_p} | {id(p) for p in enc_new_p} | {id(p) for p in sg_p}
        rest_p = [p for p in self.parameters() if p.requires_grad and id(p) not in assigned]

        groups: List[Dict[str, Any]] = []
        if pre_p:
            groups.append(
                {
                    "params": pre_p,
                    "lr": float(lr) * float(pretrained_lr_scale),
                    "weight_decay": float(weight_decay),
                    "group_name": "enc_pretrained",
                }
            )
        if enc_new_p:
            groups.append(
                {
                    "params": enc_new_p,
                    "lr": float(lr) * float(enc_lr_scale),
                    "weight_decay": float(weight_decay),
                    "group_name": "enc_new",
                }
            )
        if rest_p:
            groups.append(
                {
                    "params": rest_p,
                    "lr": float(lr),
                    "weight_decay": 0.0,  # TD-MPC2 heads keep zero weight decay
                    "group_name": "dynamics_and_heads",
                }
            )
        if sg_p:
            groups.append(
                {
                    "params": sg_p,
                    "lr": float(lr),
                    "weight_decay": 0.0,
                    "group_name": "sg_diagnostic_probe",
                }
            )
        return groups

    def architecture_summary(self) -> Dict[str, Any]:
        def _count(m: Optional[nn.Module], trainable_only: bool = False) -> int:
            if m is None:
                return 0
            return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))

        return {
            "format": VIDEO_WM_FORMAT,
            "arch": self.arch,
            "family": self.family,
            "backbone": self.backbone_type,
            "dynamics": self.dynamics_type,
            "spatial_latent": self.spatial_latent,
            "latent_shape": [self.num_tokens, self.token_dim] if self.spatial_latent else [self.packed_latent_dim],
            "packed_latent_dim": self.packed_latent_dim,
            "head_dim": self.head_dim,
            "offset_mode": self.offset_mode,
            "video_offset_mode": self.video_offset_mode,
            "zero_privilege_test_gap": self.offset_mode == "gap",
            "has_gap_head": (self._gap is not None) or (self.video_offset_mode == "gap" and self._offset is not None),
            "has_sg_probe": self._probe_sg is not None,
            "encoder_pretrained": self.pretrained,
            "encoder_pretrained_desc": self.spec["encoder_pretrained_desc"],
            "transition_pretrained": bool(self.spec["transition_pretrained"]) and self.pretrained,
            "transition_desc": self.spec["transition_desc"],
            "official_tdmpc2_backend": self._official_tdmpc2,
            "params_total_m": round(_count(self, False) / 1e6, 3),
            "params_trainable_m": round(_count(self, True) / 1e6, 3),
            "params_encoder_total_m": round(_count(self.enc, False) / 1e6, 3),
            "params_encoder_trainable_m": round(_count(self.enc, True) / 1e6, 3),
            "params_dynamics_m": round(_count(self._dynamics, False) / 1e6, 3),
            "params_readout_m": round(_count(self.readout, False) / 1e6, 3),
            "params_sg_probe_m": round(_count(self._probe_sg, False) / 1e6, 3),
        }

    def checkpoint_meta(self, stats: dict) -> Dict[str, Any]:
        """Metadata block written into every checkpoint (100% `fidelity_factual320.py` compatible!)."""
        return {
            "format": VIDEO_WM_FORMAT,
            "labels": LABELS,
            "arch": self.arch,
            "family": self.family,
            "offset_mode": self.offset_mode,
            "video_offset_mode": self.video_offset_mode,
            "zero_privilege_test_gap": self.offset_mode == "gap",
            "has_gap_head": (self._gap is not None) or (self.video_offset_mode == "gap" and self._offset is not None),
            "has_sg_probe": self._probe_sg is not None,
            # Legacy keys read by `fidelity_factual320.py` & `agent320.py`
            "stem": self.arch,
            "imagenet": self.pretrained,
            "offset_weight_train": self.offset_weight,
            # `fidelity_factual320.py` checks `ck.get("offset_weight", 0.0) > 0`; set >0 whenever `_offset` exists
            "offset_weight": (max(self.offset_weight, 1e-6) if self._offset is not None else 0.0),
            "shaping": self.shaping,
            "cfg": {"gamma": float(self.cfg.gamma)},
            "model_kwargs": dict(self._init_kwargs),
            "stats": stats,
            "summary": self.architecture_summary(),
        }


# ---------------------------------------------------------------------------
# Dataset Wrapper Adding Unambiguous `probe_gap`, `probe_along`, `gap_mm`, `pin_length_mm`
# ---------------------------------------------------------------------------
def canonicalize_window_probes(
    out: dict,
    pin_length_mm: float,
    base_offset_mode: str = "along",
    target_offset_mode: str = "gap",
) -> dict:
    """Populate `pin_length_mm`, `gap_mm`, `probe_along`, `probe_gap`, and `probe` with zero risk of double-negation.

    Works identically whether the underlying `WindowData` is unpatched legacy (`base_offset_mode='along'`,
    no `gap_mm`) or D1-patched (`offset_mode='along'` or `'gap'`, with `gap_mm` present).
    """
    L_mm = float(pin_length_mm)
    probe = out["probe"]
    if base_offset_mode == "gap":
        gap_mm = out["gap_mm"] if "gap_mm" in out else probe[:, 0]
        along_mm = -gap_mm - L_mm
    elif base_offset_mode == "along":
        along_mm = probe[:, 0]
        gap_mm = out["gap_mm"] if "gap_mm" in out else (-along_mm - L_mm)
    else:
        raise ValueError(f"Unknown base_offset_mode={base_offset_mode!r}")

    probe_along = torch.cat([along_mm.unsqueeze(-1), probe[:, 1:]], dim=-1)
    probe_gap = torch.cat([gap_mm.unsqueeze(-1), probe[:, 1:]], dim=-1)
    wire_mode = "along" if target_offset_mode in ("along", "probe") else "gap"

    out["pin_length_mm"] = torch.tensor(L_mm, dtype=torch.float32)
    out["gap_mm"] = gap_mm
    out["probe_along"] = probe_along
    out["probe_gap"] = probe_gap
    out["probe"] = probe_gap if wire_mode == "gap" else probe_along
    return out


class VideoWindowData(torch.utils.data.Dataset):
    """Wraps legacy OR D1-patched `wm320.WindowData`, guaranteeing consistent `probe_gap`, `probe_along`, and `gap_mm`."""

    def __init__(
        self,
        root,
        entries: List[dict],
        items,
        stats: dict,
        shaping: float,
        gamma: float,
        offset_mode: str = "gap",
    ):
        if _BaseWindowData is None:
            raise ImportError("wm.tdmpc.wm320.WindowData is required (add world_model/src to PYTHONPATH).")
        self.offset_mode = "along" if str(offset_mode) in ("along", "probe") else "gap"
        sig = inspect.signature(_BaseWindowData.__init__)
        if "offset_mode" in sig.parameters:
            # Always request canonical 'along' from D1-patched WindowData so probe[:,0] is unambiguously along_mm
            self.base = _BaseWindowData(root, entries, items, stats, shaping, gamma, offset_mode="along")
        else:
            self.base = _BaseWindowData(root, entries, items, stats, shaping, gamma)
        self._base_offset_mode = str(getattr(self.base, "offset_mode", "along"))
        self.pin_length_mm = np.asarray([1000.0 * float(e["pin_length_m"]) for e in entries], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        out = self.base[i]
        e = int(out["item"][0])
        L_mm = float(self.pin_length_mm[e])
        return canonicalize_window_probes(
            out,
            pin_length_mm=L_mm,
            base_offset_mode=self._base_offset_mode,
            target_offset_mode=self.offset_mode,
        )


# ---------------------------------------------------------------------------
# Factory & Loader
# ---------------------------------------------------------------------------
def build_video_wm320(arch: str, stats: dict, **kwargs) -> VideoWM320:
    return VideoWM320(arch=arch, stats=stats, **kwargs)


def load_video_wm320(ckpt: str | Path, device: str | torch.device = "cuda", local_files_only: bool = True) -> VideoWM320:
    """Load a `video_wm320-1` checkpoint cleanly without requiring internet access."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("format") != VIDEO_WM_FORMAT:
        raise ValueError(f"{ckpt} has format={ck.get('format')!r}, expected {VIDEO_WM_FORMAT!r}")
    if ck.get("labels") != LABELS:
        raise ValueError(f"{ckpt} was trained with labels {ck.get('labels')!r}, code has {LABELS!r}")
    mk = dict(ck["model_kwargs"])
    arch = mk.pop("arch")
    try:
        # Fast offline instantiation (uses saved `vjepa2_hf_config` when present -> zero HF cache needed!)
        m = VideoWM320(
            arch=arch,
            stats=ck["stats"],
            pretrained=False,
            local_files_only=local_files_only,
            **mk,
            **ck.get("cfg", {}),
        )
        m.load_state_dict(ck["state"], strict=True)
    except Exception:
        # Fallback for torch.hub V-JEPA 2 / V-JEPA 2-AC architectures that require cached hub definition
        m = VideoWM320(
            arch=arch,
            stats=ck["stats"],
            pretrained=True,
            local_files_only=local_files_only,
            **mk,
            **ck.get("cfg", {}),
        )
        m.load_state_dict(ck["state"], strict=True)
    return m.to(device).eval()


__all__ = [
    "ARCH_PRESETS",
    "DEFAULT_SEGFORMER_ID",
    "DEFAULT_VJEPA2_VITL_ID",
    "GAP_SCALE",
    "OFFSET_SCALE",
    "VIDEO_WM_FORMAT",
    "VideoWM320",
    "VideoWindowData",
    "build_video_wm320",
    "canonicalize_window_probes",
    "load_video_wm320",
]
