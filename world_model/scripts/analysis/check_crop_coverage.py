"""Do the stored crops contain the true pin tips and the true socket openings? (truth only for this check)

    python world_model/scripts/analysis/check_crop_coverage.py --data $Q/data/wm_v1x --crops $Q/data/wm_v1x_crops \
        --geometry $Q/data/wm_v1_labelgeom --out $Q/runs/checks_v4/explore/crops --n 600
Per distance to full depth: share of frames whose plug crop contains every pin tip and whose opening crop
contains every opening corner (frames whose openings are outside the camera image cannot be cropped and are
counted separately). Exit code 1 if, within 15 mm, pin tips < 99.5 % or openings < 99 % inside.
Also writes crop images near contact (true tips red, true openings green) for a visual check.
"""
import argparse
import json
import sys
from collections import Counter

import cv2
import numpy as np

from wm.data import index, store
from wm.data.datasets import crops_path
from wm.data.outcome import diagnosis
from wm.perception.crops import CROP
from wm.perception.labels import load_geometry, opening_corners, pose7_to_T, project

ap = argparse.ArgumentParser()
for a in ("--data", "--crops", "--geometry", "--out"):
    ap.add_argument(a, required=True)
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--seed", type=int, default=2)
args = ap.parse_args()
E = [e for e in index.load(args.data) if e["n_images"] and e["split"] in ("train", "ood_val")]
rng = np.random.default_rng(args.seed)
pick = [E[i] for i in rng.choice(len(E), min(args.n, len(E)), replace=False)]
rows, tiles, off_image = [], [], 0
for e in pick:
    arr, _ = store.load(e["path"])
    g = load_geometry(args.geometry, args.data, e["path"])
    with np.load(crops_path(args.crops, args.data, e["path"])) as f:
        pu, hu = f["plug_uv0"], f["hole_uv0"]
    goal = np.asarray(g["goal_world"])
    L2 = 2 * g["pin_half_length_m"]
    tips = np.array([[L2, p["center_yz_m"][0], p["center_yz_m"][1]] for p in g["pins"]])
    opens = np.concatenate(opening_corners(g))
    H, W = arr["rgb"].shape[1:3]
    for j in range(len(arr["img_step"])):
        s = int(arr["img_step"][j])
        K, cam = arr["K"].astype(np.float64), arr["cam_T_world"][j].astype(np.float64)
        ch = goal @ pose7_to_T(arr["eval_rel"][s])
        tp, _ = project(tips @ ch[:3, :3].T + ch[:3, 3], cam, K)
        op, zc = project(opens @ goal[:3, :3].T + goal[:3, 3], cam, K)
        m = lambda px, uv0: float(np.minimum(px - uv0, uv0 + CROP - px).min())
        d = float(diagnosis(arr["eval_rel"][s])["depth_left_m"] * 1000)
        visible = bool((zc > 0).all() and (op >= 0).all() and (op[:, 0] < W).all() and (op[:, 1] < H).all())
        off_image += not visible
        rows.append((d, m(tp, pu[j]), m(op, hu[j]) if visible else np.nan, e["kind"], e["split"]))
        if d < 16 and len(tiles) < 24 and rng.random() < 0.03:
            img = arr["rgb"][j].copy()
            a = img[pu[j][1]:pu[j][1] + CROP, pu[j][0]:pu[j][0] + CROP].copy()
            b = img[hu[j][1]:hu[j][1] + CROP, hu[j][0]:hu[j][0] + CROP].copy()
            for (x, y) in tp - pu[j]:
                cv2.circle(a, (int(round(x)), int(round(y))), 2, (255, 0, 0), -1)
            for poly in (op - hu[j]).reshape(-1, 4, 2):
                cv2.polylines(b, [np.round(poly).astype(np.int32)], True, (0, 255, 0), 1)
            t = np.concatenate([cv2.resize(img, (128, 128)), a, b], 1)
            cv2.putText(t, "%.1fmm %s" % (d, e["kind"]), (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            tiles.append(t)
bins = [(15, 1e9, "far >15mm"), (5, 15, "5-15mm"), (2, 5, "2-5mm"), (1, 2, "1-2mm"), (-1e9, 1, "<1mm at depth")]
R = np.array([(r[0], r[1], r[2]) for r in rows])
out, fail = {"frames": len(R), "openings_outside_image": int(off_image)}, []
for k, name, thr in ((1, "pin tips in plug crop", 99.5), (2, "openings in opening crop", 99.0)):
    print("==", name)
    for lo, hi, lab in bins:
        mm = R[(R[:, 0] >= lo) & (R[:, 0] < hi), k]
        mm = mm[~np.isnan(mm)]
        if not len(mm):
            continue
        pct = 100 * (mm >= 0).mean()
        print("%-15s frames %6d  inside %6.2f%%  min %7.1f  p1 %6.1f  median %6.1f" % (lab, len(mm), pct, mm.min(), np.percentile(mm, 1), np.median(mm)))
        out[f"{name}/{lab}"] = {"frames": int(len(mm)), "inside_pct": round(float(pct), 3)}
        if lab != "far >15mm" and pct < thr:
            fail.append((name, lab, round(float(pct), 2)))
print("frames with openings outside the camera image (not croppable, excluded):", off_image, "of", len(R))
if tiles:
    sheet = np.concatenate([np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles) - 1, 2)], 0)
    cv2.imwrite(args.out + ".jpg", sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
out["gate"] = "FAIL" if fail else "PASS"
out["failed"] = fail
json.dump(out, open(args.out + ".json", "w"), indent=1)
print("GATE", out["gate"], fail)
sys.exit(1 if fail else 0)
