"""Training splits, decided from recorded metadata (fixed before any training).

    test      ports A, B, C, D (never used for training or model selection)
    ood_val   held-out generated shapes (all grasp conditions), and on training shapes
              every grasp outside the training bands (interp / extrap conditions)
    train     training shapes x training-band grasps
"""

from __future__ import annotations

TEST_PORTS = ("A", "B", "C", "D")
OOD_VAL_SHAPES = ("G46", "G47", "G48") + tuple(f"G{i}" for i in range(55, 150, 10))  # G55, G65, ..., G145


def split_of(port: str, condition: str) -> str:
    if port in TEST_PORTS:
        return "test"
    if port in OOD_VAL_SHAPES or condition != "train":
        return "ood_val"
    return "train"
