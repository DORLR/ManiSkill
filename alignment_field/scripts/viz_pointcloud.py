"""Picture of what the method sees: RGB crop, depth (clean / strong noise) and
the point cloud in hand-centred port axes (clean / strong noise), one row per
port, hand 5 mm behind the pre-insertion pose.

python scripts/viz_pointcloud.py --ports A D G03 --out results/viz/pointcloud_compare.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from paf import noise as N  # noqa: E402
from paf.collect import establish_reference  # noqa: E402
from paf.config import CROP, PC_BOX  # noqa: E402
from paf.env_adapter import Robot  # noqa: E402
from paf.geometry import pose_from_state  # noqa: E402
from paf.perception import backproject, port_axes_from_obs, rgb_crop, to_world  # noqa: E402


def cloud(obs, R_P):
    pts, valid = backproject(obs.depth_mm, obs.K)
    rgb = obs.rgb[..., :3][valid] / 255.0
    p = (to_world(pts, obs.cam_T_world) - obs.H[:3, 3]) @ R_P
    (x0, x1), (y0, y1), (z0, z1) = PC_BOX
    k = (p[:, 0] > x0) & (p[:, 0] < x1) & (np.abs(p[:, 1]) < 0.04) & (np.abs(p[:, 2]) < 0.04)
    return p[k] * 1000, rgb[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", default=["A", "D", "G03"])
    ap.add_argument("--out", default="results/viz/pointcloud_compare.png")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for port in args.ports:
        r = Robot(port)
        for seed in range(777, 787):  # the expert occasionally fails; try another episode
            r.reset(seed=seed, pitch_deg=-45.0, grasp_pos_m=-0.020)
            H_pre, R_P, _ = establish_reference(r)
            if H_pre is not None:
                break
        r.move_to(pose_from_state(np.array([-0.005, 0, 0, 0, 0, 0]), H_pre, R_P))
        obs = r.observe()
        noisy = N.apply(obs, rng, "strong")
        R_est, _ = port_axes_from_obs(obs, rng)
        R_est_n, _ = port_axes_from_obs(noisy, rng)
        rows.append((port, obs, noisy, R_est, R_est_n))
        r.env.close()
    fig, ax = plt.subplots(len(rows), 7, figsize=(28, 4.2 * len(rows)))
    r0, r1, c0, c1 = CROP
    for i, (port, obs, noisy, R_est, R_est_n) in enumerate(rows):
        ax[i, 0].imshow(rgb_crop(obs))
        ax[i, 0].set_title(f"{port}: RGB crop (native px)")
        for j, (o, name) in enumerate(((obs, "clean"), (noisy, "strong noise"))):
            d = o.depth_mm[r0:r1, c0:c1].astype(float)
            d[d == 0] = np.nan
            ax[i, 1 + j].imshow(d, cmap="viridis", vmin=np.nanpercentile(d, 2), vmax=np.nanpercentile(d, 98))
            miss = np.mean(o.depth_mm[r0:r1, c0:c1] == 0) * 100
            ax[i, 1 + j].set_title(f"depth {name} (missing {miss:.0f}%)")
        for j, (o, R, name) in enumerate(((obs, R_est, "clean"), (noisy, R_est_n, "strong noise"))):
            p, c = cloud(o, R)
            # side view: insertion axis x (towards board) vs vertical z
            ax[i, 3 + 2 * j].scatter(p[:, 0], p[:, 2], c=c, s=1.5)
            ax[i, 3 + 2 * j].set_title(f"cloud {name}: side (x into board, z up), mm")
            # front view: looking along the insertion axis, y vs z, points beyond the pin roots
            k = p[:, 0] > 20
            ax[i, 4 + 2 * j].scatter(-p[k, 1], p[k, 2], c=c[k], s=1.5)
            ax[i, 4 + 2 * j].set_title(f"cloud {name}: front (y, z), x > 20 mm")
            for a in (ax[i, 3 + 2 * j], ax[i, 4 + 2 * j]):
                a.set_aspect("equal")
                a.grid(alpha=0.3)
        for a in ax[i, :3]:
            a.axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("saved", args.out)


if __name__ == "__main__":
    main()
