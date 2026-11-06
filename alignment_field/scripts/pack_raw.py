"""Pack raw-RGB-D shards into memory-mapped arrays for fast random access
(the full set does not fit in RAM: ~330 kB per view).

python scripts/pack_raw.py --data "data/train_v4/*.npz" --out data/packed_v4
Writes rgb.npy (N,256,256,3) uint8, depth.npy (N,256,256) int16, K/cam_T_world/H/R_P
float32, u.npy (N,6) sigma units, push.npy, kind.npy, grasp.npy, ports.json.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from paf import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+", help="one or more globs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    files = []
    n = 0
    for f in sorted(sum((glob.glob(g) for g in args.data), [])):
        d = np.load(f, allow_pickle=False)
        if "depth_full" not in d.files or len(d["q"]) == 0:
            continue
        files.append(f)
        n += len(d["q"])
    if n == 0:
        raise SystemExit(f"no shards with raw RGB-D found in {args.data} (collect with --store-raw)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d0 = np.load(files[0], allow_pickle=False)
    hw = d0["depth_full"].shape[1:]
    mm = {
        "rgb": np.lib.format.open_memmap(out / "rgb.npy", "w+", np.uint8, (n, *hw, 3)),
        "depth": np.lib.format.open_memmap(out / "depth.npy", "w+", np.int16, (n, *hw)),
    }
    small = {k: [] for k in ("K", "cam_T_world", "H", "R_P", "u", "push", "kind", "grasp")}
    ports, k0 = [], 0
    for gi, f in enumerate(files):
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        m = len(d["q"])
        mm["rgb"][k0:k0 + m] = d["rgb_full"]
        mm["depth"][k0:k0 + m] = d["depth_full"]
        small["K"].append(d["K"])
        small["cam_T_world"].append(d["cam_T_world"])
        small["H"].append(d["H"])
        small["R_P"].append(np.repeat(d["R_P_est"][None].astype(np.float32), m, 0))
        small["u"].append((d["q"] / C.SIGMA).astype(np.float32))
        small["push"].append(d["push"])
        small["kind"].append(d["kind"])
        small["grasp"].append(np.full(m, gi))
        ports.append(meta["port"])
        k0 += m
    for k, v in small.items():
        np.save(out / f"{k}.npy", np.concatenate(v))
    (out / "ports.json").write_text(json.dumps({"grasp_port": ports, "files": files}))
    for a in mm.values():
        a.flush()
    print(f"packed {len(files)} grasps, {n} views into {out}")


if __name__ == "__main__":
    main()
