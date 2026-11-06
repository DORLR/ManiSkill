"""Offline check: can the pin direction be read from the plug points (depth only)?
Fit y = a + b x and z = c + d x (port axes, hand-centred) to the plug points in the front
`--front` mm (pins only, the housing face is >= 14 mm behind the tips) and compare the slopes
with the label tilt (r_y, r_z) of each view.
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from paf.relnet import BOARD_BAND, BOX_YZ, backproject_torch, to_port_frame
from paf.train_rel import Packed, to_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="data/packed_v6_abcd")
    ap.add_argument("--kinds", type=int, nargs="+", default=[3])
    ap.add_argument("--ports", nargs="+", default=None)
    ap.add_argument("--front", type=float, default=8.0)
    args = ap.parse_args()
    ds = Packed(args.packed)
    sel = np.isin(ds.small["kind"][ds.idx], args.kinds)
    if args.ports:
        sel &= np.isin(ds.port[ds.idx], args.ports)
    ds.idx = ds.idx[sel]
    G, T = [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, num_workers=2):
            rgb, depth, K, Tc, H, R, u, _ = to_dev(b, "cuda")
            pc, valid = backproject_torch(depth, K.float())
            p = to_port_frame(pc, Tc.float(), H.float(), R.float()).reshape(len(u), -1, 3)
            valid = valid.reshape(len(u), -1)
            lat = (p[..., 1].abs() < BOX_YZ) & (p[..., 2].abs() < BOX_YZ)
            for i in range(len(u)):
                pi, vi = p[i].double(), valid[i] & lat[i]
                x = pi[vi, 0]
                xb = torch.quantile(x[x > 0].float(), 0.8).double() if (x > 0).sum() > 10 else torch.tensor(0.05)
                plug = vi & (pi[:, 0] < xb - BOARD_BAND) & (pi[:, 0] > -0.02)
                if plug.sum() < 20:
                    G.append([np.nan, np.nan]); T.append(u[i].cpu().numpy()); continue
                q = pi[plug]
                q = q[q[:, 0] > q[:, 0].max() - args.front / 1000]
                A = torch.stack([torch.ones_like(q[:, 0]), q[:, 0]], 1)
                sol = torch.linalg.lstsq(A, q[:, 1:]).solution            # (2,2): intercepts, slopes
                G.append([float(sol[1, 0]), float(sol[1, 1])])          # dy/dx, dz/dx
                T.append(u[i].cpu().numpy())
    G, T = np.array(G), np.array(T)
    ok = ~np.isnan(G[:, 0])
    gz = np.degrees(np.arctan(G[ok, 0]))   # yaw-like slope (deg)
    gy = np.degrees(np.arctan(G[ok, 1]))   # pitch-like slope (deg)
    for name, g, j in (("slope dy/dx vs r_z", gz, 5), ("slope dz/dx vs r_y", gy, 4)):
        t = T[ok, j]
        for s in (+1, -1):
            e = s * g - t
            print(f"{name} sign {s:+d}: corr {np.corrcoef(s * g, t)[0, 1]:+.3f}  bias {e.mean():+.2f}  median |err| {np.median(np.abs(e - e.mean())):.2f} deg (after removing bias), raw {np.median(np.abs(e)):.2f}")
    print(f"views {ok.sum()}/{len(ok)}")


if __name__ == "__main__":
    main()
