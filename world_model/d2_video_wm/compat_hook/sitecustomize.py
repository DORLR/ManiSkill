"""Zero-overhead, stdlib-only startup hook for `VideoWM320` compatibility in unmodified `task_release/` scripts.

Activates ONLY when `VIDEO_WM320_COMPAT=1` is set in the environment.
Crucially does NOT import `torch` or `numpy` at interpreter startup, so `run_wm320_attempts.py`
`spawn` workers set `CUDA_VISIBLE_DEVICES` before `torch` is ever imported!
"""

import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path

if os.environ.get("VIDEO_WM320_COMPAT") == "1":
    _D2_DIR = str(Path(__file__).resolve().parents[1])

    def _lazy_patch(fullname: str, module) -> None:
        if _D2_DIR not in sys.path:
            sys.path.insert(0, _D2_DIR)
        from eval_video_wm320_compat import _patch_module

        _patch_module(fullname, module)

    class _LazyVideoWMCompatFinder(importlib.abc.MetaPathFinder):
        TARGETS = {"wm.tdmpc.wm320", "wm.online.agent320"}

        def __init__(self):
            self._busy = set()

        def find_spec(self, fullname, path=None, target=None):
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

                def exec_module(self, mod):
                    orig_loader.exec_module(mod)
                    _lazy_patch(fullname, mod)

            spec.loader = _PatchLoader()
            return spec

    if not any(type(f).__name__ == "_LazyVideoWMCompatFinder" for f in sys.meta_path):
        sys.meta_path.insert(0, _LazyVideoWMCompatFinder())
