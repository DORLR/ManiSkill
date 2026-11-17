"""TD-MPC2 on the branch set (wm/collect/branch.py).

The official code (github.com/nicklashansen/tdmpc2, commit 332f19e) lies unmodified in $Q/code/tdmpc2_orig; its one
missing dependency (tensordict) is in a separate folder so that the shared environment is not touched.
  data.py     one step = one decision of the policy; observations, actions, rewards from joints only
  model.py    the agreed model: TD-MPC2's latent structure with a picture encoder (RGB + depth, 640 x 640) and values
              learned from the real returns
  original.py TD-MPC2 itself, unchanged, fed from the same data (the control run)
  report.py   the offline checks shared by both
"""

import os
import sys
from pathlib import Path

TDMPC2_DIR = Path(os.environ.get("TDMPC2_DIR", "/mnt/storage2/qingxin/code/tdmpc2_orig/tdmpc2"))
TDMPC2_EXTRA = Path(os.environ.get("TDMPC2_EXTRA", "/mnt/storage2/qingxin/envs/tdmpc2_extra"))


def use_tdmpc2() -> None:
    """Make TD-MPC2's modules (`common`, `tdmpc2`) importable; the extra packages go last so they shadow nothing."""
    if str(TDMPC2_DIR) not in sys.path:
        sys.path.insert(0, str(TDMPC2_DIR))
    if str(TDMPC2_EXTRA) not in sys.path:
        sys.path.append(str(TDMPC2_EXTRA))
