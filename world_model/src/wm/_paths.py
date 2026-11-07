"""Repository locations. The environment adapter is reused from alignment_field (sibling folder)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ALIGNMENT_FIELD = REPO_ROOT / "alignment_field"


def use_alignment_field() -> None:
    """Make `paf` (alignment_field/paf) importable."""
    path = str(ALIGNMENT_FIELD)
    if path not in sys.path:
        sys.path.insert(0, path)
