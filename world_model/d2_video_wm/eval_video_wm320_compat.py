"""Drop-in compatibility adapter for running unmodified Changhao evaluation & online scripts on `VideoWM320`.

Works seamlessly with:
  - `world_model/src/wm/online/agent320.py` (`WMAgent`)
  - `world_model/src/wm/online/attempt320.py` (`OnlineTester`)
  - `world_model/scripts/online/run_wm320_attempts.py` (including `mp.get_context("spawn")` workers!)
  - `world_model/scripts/analysis/fidelity_factual320.py` (`B1`)
  - `world_model/scripts/analysis/counterfactual320.py` (`B2`)

Without modifying a single file inside `task_release/`!

Usage:
  1. Run ANY unmodified Changhao script with automatic `VideoWM320` + legacy `WM320` loading
     (propagates into `spawn` workers via `compat_hook/sitecustomize.py` when `VIDEO_WM320_COMPAT=1`):
       python eval_video_wm320_compat.py run world_model/scripts/analysis/fidelity_factual320.py \\
           --ckpt <video_wm.pt> --fast $SCRATCH/data/wm_v1_fast --out <b1.json>

       python eval_video_wm320_compat.py run world_model/scripts/online/run_wm320_attempts.py \\
           --grasps $SCRATCH/data/wm_v1 --ckpt <video_wm.pt> --mode wm --split heldout ...

  2. Inspect any checkpoint (`VideoWM320` or legacy `WM320`):
       python eval_video_wm320_compat.py inspect <ckpt.pt>

  3. Programmatic Python usage:
       from eval_video_wm320_compat import load_any_wm320, VideoWMAgent, install_compat_hooks
       install_compat_hooks()
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_COMPAT_HOOK_DIR = _THIS_DIR / "compat_hook"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from video_wm320 import (  # noqa: E402
    ACTION_DIM,
    JOINTS,
    LABELS,
    VIDEO_WM_FORMAT,
    VideoWM320,
    load_video_wm320,
)

_ORIG_WM320_LOAD = None
_HOOK_INSTALLED = False


def is_video_wm320_checkpoint(ckpt: str | Path) -> bool:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    return ck.get("format") == VIDEO_WM_FORMAT


def load_any_wm320(ckpt: str | Path, device: str | torch.device = "cuda", local_files_only: bool = True):
    """Auto-detect and load either a `video_wm320-1` checkpoint OR a legacy `wm320-1` checkpoint."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    if ck.get("format") == VIDEO_WM_FORMAT:
        return load_video_wm320(ckpt, device=device, local_files_only=local_files_only)
    if _ORIG_WM320_LOAD is not None:
        return _ORIG_WM320_LOAD(str(ckpt), device=device)
    from wm.tdmpc.wm320 import WM320

    if ck.get("labels") != LABELS:
        raise ValueError(f"{ckpt} was trained with labels {ck.get('labels')}, this code has {LABELS}")
    wm_kwargs = dict(
        imagenet=ck.get("imagenet", False),
        offset_weight=ck.get("offset_weight", 0.0),
        **ck["cfg"],
    )
    if "offset_mode" in inspect.signature(WM320.__init__).parameters:
        wm_kwargs["offset_mode"] = str(ck.get("offset_mode", "along"))
    m = WM320(ck["stem"], ck["stats"], ck.get("shaping", 0.0), **wm_kwargs)
    m.load_state_dict(ck["state"])
    return m.to(device).eval()


def _normalize_agent_offset_mode(raw: object) -> str:
    s = str(raw or "along")
    if s in ("along", "probe"):
        return "along"
    if s in ("gap", "gap_1d", "none"):
        return "gap"
    raise ValueError(f"Unknown checkpoint offset_mode {raw!r}; expected 'gap' or 'along'")


class VideoWMAgent:
    """Drop-in replacement for `wm.online.agent320.WMAgent` supporting BOTH `VideoWM320` and D1/legacy `WM320`.

    Strictly implements D1's zero-privilege `offset_mode` contract (`attempt320._read_agent_offset_and_gap`):
      - `self.offset_mode`: `'gap'` (zero-privilege; never reads/accepts `pin_length_mm`) or `'along'` (legacy).
      - `encode(rgb, depth, q) -> z (1, L_flat)` (D2a vector latents AND D2b/D2c packed spatial tokens)
      - `propose(z) -> (16, 7)` float64
      - `done(z) -> float`
      - `offset_mm(z) -> (4,)` float64 or `None` (dim 0 is `gap_mm` in `'gap'` mode, `along_mm` in `'along'` mode)
      - `gap_mm(z, pin_length_mm=None) -> float` or `None` (raises `ValueError` if `pin_length_mm` passed in `'gap'` mode!)
      - `values(z) -> (num_q,)` float64
      - `score(z, deltas, parts=False) -> (C, num_q)` float64 (or `(scores, G)`)
    """

    def __init__(self, ckpt: str, device: str = "cuda", horizon: int = 3, local_files_only: bool = True):
        self.device = torch.device(device)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        self.format = str(ck.get("format", "wm320-1"))
        self.arch = str(ck.get("arch", ck.get("stem", "wm320")))
        self.offset_mode = _normalize_agent_offset_mode(ck.get("offset_mode", "along"))
        self.video_offset_mode = str(ck.get("video_offset_mode", self.offset_mode))
        self.step = int(ck.get("step", -1))
        self.shaping = float(ck.get("shaping", 0.0))
        self.horizon = int(horizon)
        del ck

        self.m = load_any_wm320(ckpt, device=self.device, local_files_only=local_files_only)
        self.gamma = float(self.m.cfg.gamma)

    @torch.no_grad()
    def encode(self, rgb: np.ndarray, depth: np.ndarray, q: np.ndarray) -> torch.Tensor:
        """`rgb (4, 320, 320, 3)` uint8, `depth (4, 320, 320)` uint16/int16 [0.1 mm], `q (7,)` -> `z (1, L_flat)`."""
        r = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        d = torch.from_numpy(np.minimum(depth, 32767).astype(np.int16)).to(self.device)
        tok = self.m.enc.tokens(r, d)[None]
        q_t = torch.from_numpy(np.asarray(q, np.float32)[:JOINTS])[None].to(self.device)
        return self.m.enc.fuse(tok, q_t)

    @torch.no_grad()
    def propose(self, z: torch.Tensor) -> np.ndarray:
        return self.m.act_chunk(z)[0].double().cpu().numpy()

    @torch.no_grad()
    def done(self, z: torch.Tensor) -> float:
        return float(torch.sigmoid(self.m._done(z))[0, 0])

    @torch.no_grad()
    def offset_mm(self, z: torch.Tensor) -> Optional[np.ndarray]:
        if self.m._offset is None:
            return None
        return (self.m._offset(z) * self.m.off_scale)[0].double().cpu().numpy()

    @torch.no_grad()
    def gap_mm(self, z: torch.Tensor, pin_length_mm: Optional[float] = None) -> Optional[float]:
        """Estimated pin-tip clearance [mm] in front of board face (exact D1 zero-privilege contract).

        - `offset_mode == 'gap'`: raises `ValueError` if `pin_length_mm` is not None; returns `predict_gap_mm(z)`
          (or `float(o[0])` for D1-patched `WM320` checkpoints).
        - `offset_mode == 'along'` (legacy): requires `pin_length_mm` and returns `float(-o[0] - pin_length_mm)`.
        """
        if self.offset_mode == "gap":
            if pin_length_mm is not None:
                raise ValueError("zero-privilege violation: gap mode must not receive pin_length_mm")
            if isinstance(self.m, VideoWM320):
                g = self.m.predict_gap_mm(z)
                return None if g is None else float(g[0])
            o = self.offset_mm(z)
            return None if o is None else float(o[0])
        if self.offset_mode == "along":
            if pin_length_mm is None:
                raise ValueError(
                    "legacy offset_mode='along' requires pin_length_mm; use an offset_mode='gap' checkpoint for zero-privilege inference"
                )
            o = self.offset_mm(z)
            return None if o is None else float(-o[0] - float(pin_length_mm))
        raise ValueError(f"Unknown offset_mode {self.offset_mode!r}")

    @torch.no_grad()
    def values(self, z: torch.Tensor) -> np.ndarray:
        return self.m.scalar(self.m.V(z))[:, 0, 0].double().cpu().numpy()

    @torch.no_grad()
    def score(self, z: torch.Tensor, deltas: np.ndarray, parts: bool = False):
        """`deltas (C, 8, 7)` -> `(C, num_q)` float64 (or `(total, G)` when `parts=True`)."""
        m = self.m
        C = len(deltas)
        a = m.scale(torch.from_numpy(np.asarray(deltas, np.float32).reshape(C, ACTION_DIM)).to(self.device))
        zc = z.expand(C, -1)
        G = torch.zeros(C, device=self.device)
        cont = torch.ones(C, device=self.device)
        for t in range(self.horizon):
            if t > 0:
                a = m.policy_action(zc)
            G = G + cont * (self.gamma ** t) * m.scalar(m.reward(zc, a))[:, 0]
            zc = m.next(zc, a)
            cont = cont * (1.0 - torch.sigmoid(m._done(zc))[:, 0])
        V = m.scalar(m.V(zc))[..., 0]  # (num_q, C)
        total = (G[None] + (cont * (self.gamma ** self.horizon))[None] * V).T.double().cpu().numpy()
        return (total, G.double().cpu().numpy()) if parts else total


# ---------------------------------------------------------------------------
# Non-Invasive Import Hook (`wm.tdmpc.wm320.load` & `wm.online.agent320.WMAgent`)
# ---------------------------------------------------------------------------
def _patch_module(fullname: str, mod) -> None:
    global _ORIG_WM320_LOAD
    if fullname == "wm.tdmpc.wm320":
        if getattr(mod, "load", None) is not load_any_wm320:
            _ORIG_WM320_LOAD = getattr(mod, "load", None)
            mod.load = load_any_wm320
    elif fullname == "wm.online.agent320":
        if getattr(mod, "WMAgent", None) is not VideoWMAgent:
            mod._OrigWMAgent = getattr(mod, "WMAgent", None)
            mod.WMAgent = VideoWMAgent


class _CompatPostImportFinder(importlib.abc.MetaPathFinder):
    """Patches `wm.tdmpc.wm320.load` and `wm.online.agent320.WMAgent` immediately upon module execution."""

    TARGETS = {"wm.tdmpc.wm320", "wm.online.agent320"}

    def __init__(self):
        self._busy = set()

    def find_spec(self, fullname: str, path=None, target=None):
        if fullname not in self.TARGETS or fullname in self._busy:
            return None
        self._busy.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        orig_loader = spec.loader

        class _PatchLoader(importlib.abc.Loader):
            def create_module(self, s):
                return orig_loader.create_module(s) if hasattr(orig_loader, "create_module") else None

            def exec_module(self, module):
                orig_loader.exec_module(module)
                _patch_module(fullname, module)

        spec.loader = _PatchLoader()
        return spec


def install_compat_hooks() -> None:
    """Install post-import hooks AND patch already-imported `wm` modules."""
    global _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.meta_path.insert(0, _CompatPostImportFinder())
        _HOOK_INSTALLED = True
    for name in ("wm.tdmpc.wm320", "wm.online.agent320"):
        if name in sys.modules:
            _patch_module(name, sys.modules[name])


# ---------------------------------------------------------------------------
# CLI: `run` (wrap any Changhao script) & `inspect` (show checkpoint metadata)
# ---------------------------------------------------------------------------
def _cmd_run(argv: list[str]) -> int:
    if not argv:
        raise SystemExit("Usage: eval_video_wm320_compat.py run <script.py> [script args...]")
    script = argv[0]
    rest = argv[1:]
    env = os.environ.copy()
    env["VIDEO_WM320_COMPAT"] = "1"
    existing_pp = env.get("PYTHONPATH", "")
    # Prepend compat_hook (for sitecustomize.py) and d2_video_wm; preserve existing PYTHONPATH
    parts = [str(_COMPAT_HOOK_DIR), str(_THIS_DIR)]
    if existing_pp:
        parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    cmd = [sys.executable, script, *rest]
    return subprocess.call(cmd, env=env)


def _cmd_inspect(argv: list[str]) -> int:
    if len(argv) != 1:
        raise SystemExit("Usage: eval_video_wm320_compat.py inspect <ckpt.pt>")
    ck = torch.load(argv[0], map_location="cpu", weights_only=False)
    out = {
        "ckpt": argv[0],
        "format": ck.get("format"),
        "labels": ck.get("labels"),
        "step": ck.get("step"),
        "arch": ck.get("arch", ck.get("stem")),
        "family": ck.get("family", "legacy_wm320"),
        "offset_mode": ck.get("offset_mode", "along"),
        "video_offset_mode": ck.get("video_offset_mode", ck.get("offset_mode", "along")),
        "zero_privilege_test_gap": ck.get("offset_mode", "along") == "gap",
        "has_gap_head": ck.get("has_gap_head", ck.get("offset_mode") == "gap" and ck.get("offset_weight", 0.0) > 0),
        "has_sg_probe": ck.get("has_sg_probe", False),
        "offset_weight_b1_compat": ck.get("offset_weight", 0.0),
        "shaping": ck.get("shaping"),
        "cfg": ck.get("cfg"),
        "summary": ck.get("summary"),
    }
    print(json.dumps(out, indent=2))
    return 0


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)
    cmd = sys.argv[1]
    if cmd == "run":
        raise SystemExit(_cmd_run(sys.argv[2:]))
    if cmd == "inspect":
        raise SystemExit(_cmd_inspect(sys.argv[2:]))
    raise SystemExit(f"Unknown subcommand {cmd!r}. Expected 'run' or 'inspect'.")


__all__ = [
    "VideoWMAgent",
    "install_compat_hooks",
    "is_video_wm320_checkpoint",
    "load_any_wm320",
]

if __name__ == "__main__":
    main()
