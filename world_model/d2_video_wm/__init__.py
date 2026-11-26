"""D2 Video World Model & High-Resolution Encoder Suite (`VideoWM320`).

Drop-in upgrades to Changhao's `wm320.WM320` solving two diagnosed bottlenecks:
  1. Coarse ResNet-18 standard `10x10` tokenization (`32x32 px ≈ 8-10 mm` vs `0.5 mm` clearance)
     -> High-res pretrained `SegFormer MiT-B1 FPN` (`nvidia/mit-b1`), `ResNet-34 NoPool` (`IMAGENET1K_V1`),
     and official Meta `V-JEPA 2` (`facebook/vjepa2-vitl-fpc64-256` + parallel high-res 320px RGB-D stem).
  2. Single-vector `256-d [CLS]` crush + 2-layer MLP dynamics -> full-rank `d=512` (`D2a`) and
     Spatial-Token Action-Conditioned Video Dynamics Transformer `STDiTDynamics` (`D2b`, `D2c`).
"""

from .video_wm320 import (
    ARCH_PRESETS,
    GAP_SCALE,
    VIDEO_WM_FORMAT,
    VideoWM320,
    VideoWindowData,
    build_video_wm320,
    canonicalize_window_probes,
    load_video_wm320,
)

__all__ = [
    "ARCH_PRESETS",
    "GAP_SCALE",
    "VIDEO_WM_FORMAT",
    "VideoWM320",
    "VideoWindowData",
    "build_video_wm320",
    "canonicalize_window_probes",
    "load_video_wm320",
]
