"""Test grasps for every combination of unseen factors, paired across cells (2026-11-19, the user's request; design
after an adversarial review).

The three factors: port shape (13 held-out shapes OOD_VAL_SHAPES), grasp pitch (held-out strip (-42, -38) deg) and
grasp position on the plug (held-out strip (-21, -19) mm) -- demo320.split320. wm_v1's held-out grasps miss "angle +
position" and "shape + angle + position" and have only 12 grasps of "shape + angle" / "shape + position". A test needs
no demonstration: run_wm320_attempts.py does the reference insertion itself (with collection's seed retries) and, for
a grasp file without "axis", measures the axis from it. So this script only writes grasp files.

8 cells = "none" + the 7 combinations. Shared draws per index i (paired design: cell c's grasp i differs from cell c''s
grasp i only in the factors that differ between the cells):
    T_i  training shape (ports of wm_v1's training split, never OOD_VAL_SHAPES or the test ports A-D), spread evenly
    O_i  held-out shape, OOD_VAL_SHAPES[i mod 13]
    p_i  pitch next to the strip, (-46, -42) u (-38, -34) deg      p*_i  pitch in the strip (-42, -38)
    x_i  position next to the strip, (-23, -21) u (-19, -17) mm    x*_i  position in the strip (-21, -19)
    s_i  episode seed (SEED0 + 1000 i; scene randomisation)
    cell "shape+angle" grasp i = (O_i, p*_i, x_i, s_i); cell "none" grasp i = (T_i, p_i, x_i, s_i); and so on.
The seen values are drawn next to the strips (same widths as the strips, inside alignment_field/paf/config.py's
"train" bands) so that a factor effect compares an unseen value with seen values of about the same difficulty
(success varies along pitch and position: second review, 2026-11-19); the outer bands (pitch -60..-55 / -25..-20,
position -30..-28 / -12..-10) are a separate (extrapolation) question and are not drawn. MARGIN keeps values away from every band edge. Grasp names
<port>_c<cell><i:03d>; condition = the cell. "none" grasps are new grasps of training shapes with training values
(in-distribution control, never trained on: the learner only reads wm_v1's grasps and online rounds on them).

    python world_model/scripts/eval/make_combo_grasps.py --grasps $Q/data/wm_v1 --out $Q/data/test_combos_2026-11-19 --n 65
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from wm.data.splits import OOD_VAL_SHAPES, TEST_PORTS  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402

PITCH_STRIP, POS_STRIP_MM = (-42.0, -38.0), (-21.0, -19.0)
PITCH_TRAIN = ((-46.0, -42.0), (-38.0, -34.0))           # seen values next to the strip (inside the "train" bands)
POS_TRAIN_MM = ((-23.0, -21.0), (-19.0, -17.0))
MARGIN = 0.2                                              # deg / mm from every band edge
SEED0 = 9_000_000
CELLS = ("none", "shape", "angle", "position", "shape+angle", "shape+position", "angle+position", "shape+angle+position")


def uniform_in(rng, bands) -> float:
    """Uniform over the union of bands, each shrunk by MARGIN at both ends."""
    b = [(lo + MARGIN, hi - MARGIN) for lo, hi in bands]
    w = np.array([hi - lo for lo, hi in b])
    u = float(rng.uniform(0.0, w.sum()))
    for (lo, hi), wi in zip(b, w):
        if u < wi:
            return lo + u
        u -= wi
    return b[-1][1]


def make(train_ports: list[str], n: int, seed: int) -> list[dict]:
    if not 0 < n <= 1000:
        raise ValueError("n must be 1..1000 (3-digit pair index in the names)")
    rng = np.random.default_rng(seed)
    ood = sorted(OOD_VAL_SHAPES)
    draws = [{"T": train_ports[(i * len(train_ports)) // n] if n <= len(train_ports) else train_ports[i % len(train_ports)],
              "O": ood[i % len(ood)], "p": uniform_in(rng, PITCH_TRAIN), "ps": uniform_in(rng, (PITCH_STRIP,)),
              "x": uniform_in(rng, POS_TRAIN_MM), "xs": uniform_in(rng, (POS_STRIP_MM,)), "s": SEED0 + 1000 * i} for i in range(n)]
    out = []
    for ci, cell in enumerate(CELLS):
        f = set(cell.split("+")) if cell != "none" else set()
        for i, d in enumerate(draws):
            port = d["O"] if "shape" in f else d["T"]
            pitch = d["ps"] if "angle" in f else d["p"]
            pos_mm = d["xs"] if "position" in f else d["x"]
            g = {"grasp": f"{port}_c{ci}{i:03d}", "port": port, "condition": cell, "pair_index": i, "pitch_deg": pitch,
                 "grasp_pos_m": pos_mm / 1000.0, "episode_seed": d["s"], "reference_ok": True, "axis": None}
            sp = D.split320(port, pitch, g["grasp_pos_m"], g["grasp"])
            if (cell == "none") != (sp in ("train", "select")) or sp == "test":
                raise RuntimeError(f"{g} has split {sp}")
            out.append(g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True, help="wm_v1 (its ports and splits)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=65, help="grasps per cell (a multiple of 13 spreads the held-out shapes evenly)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    every = [json.loads(Path(f).read_text()) for f in glob.glob(f"{args.grasps}/*/*/grasp.json")]
    train_ports = sorted({g["port"] for g in every if g.get("reference_ok")
                          and D.split320(g["port"], g["pitch_deg"], g["grasp_pos_m"], g["grasp"]) == "train"})
    train_ports = [p for p in train_ports if p not in OOD_VAL_SHAPES and p not in TEST_PORTS]
    gs = make(train_ports, args.n, args.seed)
    for g in gs:
        f = out / g["port"] / g["grasp"] / "grasp.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(g))
    (out / "README.json").write_text(json.dumps({"made_by": "make_combo_grasps.py", "n": args.n, "seed": args.seed, "cells": CELLS,
                                                 "train_ports": len(train_ports), "grasps": len(gs)}))
    print(f"{len(gs)} grasps ({args.n} per cell x {len(CELLS)} cells, {len(train_ports)} training shapes) in {out}")


if __name__ == "__main__":
    main()
