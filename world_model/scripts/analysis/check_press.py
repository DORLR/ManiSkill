"""Smoke check of the press set: outcome balance, does the requested command keep going while the hand is blocked,
free-space tracking, depth of the blocked hand, action range vs the NWM stats, and a picture strip."""
import glob, json, sys
from pathlib import Path
import numpy as np, yaml
from PIL import Image
sys.path.insert(0, "/mnt/storage2/qingxin/code/ManiSkill/world_model/src")
from wm.data import store

root, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
st = yaml.safe_load(open("/mnt/storage2/qingxin/code/ManiSkill/world_model/nwm_asis/plug_data_config.yaml"))
sp, amin, amax = st["metric_waypoint_spacing"], np.array(st["action_stats"]["min"]), np.array(st["action_stats"]["max"])
rows, strips = [], {"inserted": [], "not_inserted": []}
for f in sorted(glob.glob(root + "/*/*/t*_press.npz")):
    A, m = store.load(f)
    a = np.asarray(m["axis"]); a /= np.linalg.norm(a); Hs = np.asarray(m["H_star"])[:3, 3]; L = m["pin_length_m"]
    hand = A["hand"][:, :3, 3]; req = A["req_pos"]
    dh = (hand - Hs) @ a * 1000; dr = (req - Hs) @ a * 1000              # depth vs inserted, mm (negative = in front)
    board = -L * 1000                                                     # hand depth when the pin tips touch the board
    free = dr < board - 0.5
    lag_free = (dr[free] - dh[1:][free]) if free.any() else np.zeros(1)
    post = dr > board + 1.0
    steps = A["img_step"].astype(int); s = np.clip(steps, 0, len(req) - 1)
    P = req[s]; d = np.diff(P, axis=0) @ np.eye(3)                       # per-image requested step (world; frame ~ rotation only)
    E = np.load(f.replace(root, root + "_ee", 1))["hand_cmd"]
    R = E[s][:, :3, :3]
    dl = np.einsum("ni,nij->nj", np.diff(P, axis=0), R[:-1]) / sp
    rows.append(dict(f=f.split("/")[-3] + "/" + f.split("/")[-2] + "/" + f.split("/")[-1], outcome=m["outcome"], start_lat_mm=float(np.linalg.norm(m["start"]["lateral_m"]) * 1000),
                     start_rot=float(np.linalg.norm(m["start"]["rotvec_deg"])), kinds=[p["kind"] for p in m["log"]], stopped=m["stopped"],
                     req_past_board_mm=float(dr.max() - board), hand_past_board_mm=float(dh.max() - board), hand_end_mm=float(dh[-1] - board),
                     req_past_hand_end_mm=float(dr[-1] - dh[-1]), free_lag_med_mm=float(np.median(lag_free)), free_lag_max_mm=float(lag_free.max()),
                     limited_steps=m["limited_steps"], ik_missed=m["ik_missed_steps"], depth_left_mm=m["depth_left_mm"],
                     act_min=dl.min(0).round(2).tolist(), act_max=dl.max(0).round(2).tolist(), seconds=len(A["action"]) * 0.05))
    if m["outcome"] in strips and len(strips[m["outcome"]]) < 3:
        rgb = A["rgb"]; idx = np.linspace(0, len(rgb) - 1, 7).astype(int)
        strips[m["outcome"]].append(np.concatenate([rgb[i] for i in idx], 1))
for k, v in strips.items():
    if v: Image.fromarray(np.concatenate(v, 0)).resize((7 * 160, len(v) * 160)).save(out / f"strip_{k}.png")
json.dump(rows, open(out / "rows.json", "w"), indent=1)
import collections
print("trajectories", len(rows), collections.Counter(r["outcome"] for r in rows))
for o in ("inserted", "not_inserted"):
    R_ = [r for r in rows if r["outcome"] == o]
    if not R_: continue
    q = lambda k: np.round(np.percentile([r[k] for r in R_], [0, 50, 100]), 2).tolist()
    print(o, "n", len(R_), "| requested past board mm", q("req_past_board_mm"), "| hand past board mm (max)", q("hand_past_board_mm"),
          "| hand at end", q("hand_end_mm"), "| requested ahead of hand at end", q("req_past_hand_end_mm"))
print("free-space lag requested-hand mm median / max:", np.round(np.percentile([r["free_lag_med_mm"] for r in rows], [50, 100]), 2).tolist(), np.round(np.percentile([r["free_lag_max_mm"] for r in rows], [50, 100]), 2).tolist())
print("per-image requested step / spacing: min", np.min([r["act_min"] for r in rows], 0).round(2).tolist(), "max", np.max([r["act_max"] for r in rows], 0).round(2).tolist(), "| NWM stats min", amin.round(2).tolist(), "max", amax.round(2).tolist())
print("stopped", collections.Counter(r["stopped"] for r in rows), "| ik missed >0:", sum(r["ik_missed"] > 0 for r in rows), "| seconds median", np.median([r["seconds"] for r in rows]))
print("start lateral mm by outcome:", {o: np.round(sorted(r["start_lat_mm"] for r in rows if r["outcome"] == o), 2).tolist() for o in set(r["outcome"] for r in rows)})
