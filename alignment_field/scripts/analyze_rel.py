"""Offline diagnostics of a relational net on packed views (default: A-D hand-over views).

python scripts/analyze_rel.py --model runs_v6/rel/model.pt --packed data/packed_v6_abcd --kinds 3
Per port: lateral error (median, p90), mean error vector (systematic bias), t_x and rotation
error; and how many board ("port") points the net sees near the pin tips.
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from paf.relnet import RelNet, backproject_torch, to_port_frame, BOARD_BAND, BOX_YZ
from paf.train_rel import Packed, to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", default="data/packed_v6_abcd")
    ap.add_argument("--kinds", type=int, nargs="+", default=[3])
    ap.add_argument("--kw", default="{}", help="RelNet constructor kwargs as JSON")
    ap.add_argument("--points", action="store_true", help="also count board points near the pin tips")
    ap.add_argument("--ports", nargs="+", default=None, help="only these ports (e.g. validation shapes)")
    args = ap.parse_args()
    dev = "cuda"
    m = RelNet(**json.loads(args.kw)).to(dev).eval()
    m.load_state_dict(torch.load(args.model, map_location=dev))
    ds = Packed(args.packed)
    sel = np.isin(ds.small["kind"][ds.idx], args.kinds)
    if args.ports:
        sel &= np.isin(ds.port[ds.idx], args.ports)
    ds.idx = ds.idx[sel]
    E, T, P, cnt = [], [], [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in DataLoader(ds, batch_size=32, num_workers=2):
            rgb, depth, K, Tc, H, R, u, _ = to_dev(b, dev)
            E.append(m(rgb, depth, K.float(), Tc.float(), H.float(), R.float())["q_hat"].float().cpu())
            T.append(u.float().cpu())
            if args.points:
                with torch.autocast("cuda", enabled=False):
                    pc, valid = backproject_torch(depth, K.float())
                    p = to_port_frame(pc, Tc.float(), H.float(), R.float()).reshape(len(u), -1, 3)
                valid = valid.reshape(len(u), -1)
                lat = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
                for i in range(len(u)):
                    pi, vi = p[i], valid[i] & lat[i]
                    x = pi[vi, 0]
                    if len(x) < 50:
                        cnt.append((0, 0, 0, np.nan, np.nan)); continue
                    xb = torch.quantile(x[x > 0], 0.8) if (x > 0).any() else torch.tensor(0.05)
                    port = vi & ((pi[:, 0] - xb).abs() < BOARD_BAND)
                    plug = vi & (pi[:, 0] < xb - BOARD_BAND) & (pi[:, 0] > -0.02)
                    if plug.sum() < 5:
                        cnt.append((int(port.sum()), int(plug.sum()), 0, np.nan, np.nan)); continue
                    px = pi[plug]
                    tips = px[px[:, 0] > px[:, 0].max() - 0.004]           # plug points within 4 mm of the front
                    c = tips[:, 1:].median(0).values
                    near = port & ((pi[:, 1:] - c).norm(dim=1) < 0.008)       # board points within 8 mm of the pin tips
                    cnt.append((int(port.sum()), int(plug.sum()), int(near.sum()), float(c[0]) * 1000, float(c[1]) * 1000))
    e, t = torch.cat(E).numpy(), torch.cat(T).numpy()
    err = e - t
    port = np.array(ds.grasp_port)[ds.small["grasp"][ds.idx]]
    for p in sorted(set(port)):
        s = port == p
        le = np.linalg.norm(err[s][:, 1:3], axis=1)
        print(f"{p:4s} n{s.sum():4d} lateral err med {np.median(le):.2f} p90 {np.percentile(le, 90):.2f} mm | "
              f"mean bias y {err[s][:, 1].mean():+.2f} z {err[s][:, 2].mean():+.2f} mm | t_x err med {np.median(np.abs(err[s][:, 0])):.2f} | "
              f"rot err med {np.median(np.linalg.norm(err[s][:, 3:], axis=1)):.2f} deg")
    names = ["t_x", "t_y", "t_z", "r_x", "r_y", "r_z"]
    print("per axis (mean bias / median |err|): " + "  ".join(f"{n} {err[:, i].mean():+.2f}/{np.median(np.abs(err[:, i])):.2f}" for i, n in enumerate(names)))
    for i in (2, 4):
        k = np.polyfit(t[:, i], e[:, i], 1)[0]
        print(f"  {names[i]}: estimate = {k:.2f} x truth + {np.mean(e[:, i] - k * t[:, i]):+.2f}; truth spread (std) {t[:, i].std():.2f}")
    le = np.linalg.norm(err[:, 1:3], axis=1)
    print(f"ALL  n{len(le)} lateral err med {np.median(le):.2f} p90 {np.percentile(le, 90):.2f} mm, rot med "
          f"{np.median(np.linalg.norm(err[:, 3:], axis=1)):.2f} deg, t_x med {np.median(np.abs(err[:, 0])):.2f} mm")
    if cnt:
        c = np.array(cnt, float)
        print(f"points per view: board {np.median(c[:, 0]):.0f}, plug {np.median(c[:, 1]):.0f}, board within 8 mm of pin tips "
              f"{np.median(c[:, 2]):.0f} ({100 * np.median(c[:, 2] / np.maximum(c[:, 0], 1)):.1f}% of board points); "
              f"pin-tip centre y,z median {np.nanmedian(c[:, 3]):.1f}, {np.nanmedian(c[:, 4]):.1f} mm (5-95%: y {np.nanpercentile(c[:, 3], 5):.1f}..{np.nanpercentile(c[:, 3], 95):.1f}, z {np.nanpercentile(c[:, 4], 5):.1f}..{np.nanpercentile(c[:, 4], 95):.1f})")


if __name__ == "__main__":
    main()
