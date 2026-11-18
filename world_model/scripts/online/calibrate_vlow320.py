"""Back-off threshold of ONE world-model checkpoint (online320.sh runs it on every round's model.pt).

The agent backs off a blocked hand only when the value without the shaping term is below v_low. Its scale changes
during training and the probe that removes the shaping has its own error, so v_low is taken from the checkpoint
itself: on the contact pushes of the
SELECTION grasps, at the decision just before the pins reach the board face (exactly the windows of the learner's
validation "auc_V_before_face_contact": wm.tdmpc.wm320.face_items, all of them), the value the agent compares
(agent320.unshaped_value: mean over the heads with the shaping term removed) at that observation, split by how the
push ended (seated by FK or not); v_low = midway between the two medians.
Writes JSON: value ("unshaped": the quantity compared; files without it are from an older version and are
recomputed by online320.sh), jammed_median, inserted_median, v_low (null when the medians are not ordered: inserted
<= jammed, or a kind of push is missing, with the reason), n_jammed, n_inserted, shaped medians, the probe's mean
error on the potential at these windows (mm + deg, and times shaping in value units), step, ckpt. A non-finite value or no window at all
(a data error): no file, exit 1.
Batches of 4 windows (eval mode: group norm and the transformer work per sample, so the values do not depend on the
batch), so that it stays under ~2 GB of GPU memory.

    python world_model/scripts/online/calibrate_vlow320.py --data $Q/data/wm_v1_fast --ckpt <model.pt> --out <v_low.json>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import train_wm320 as T  # noqa: E402
from wm.tdmpc import demo320 as D  # noqa: E402
from wm.tdmpc import wmdata320 as WD  # noqa: E402
from wm.tdmpc.wm320 import WM320, WindowData, face_items  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if ck.get("labels") != WD.LABELS:
        raise SystemExit(f"{args.ckpt} was trained with labels {ck.get('labels')}, this code has {WD.LABELS}")
    m = WM320(ck["stem"], ck["stats"], ck.get("shaping", 0.0), **ck["cfg"])
    m.load_state_dict(ck["state"])
    m.to(dev).eval()
    shaping, gamma = float(ck.get("shaping", 0.0)), float(m.cfg.gamma)
    ents = [e for e in D.load_entries(args.data) if e["split"] == "select" and e["steps"] >= WD.EXECUTE]
    items = face_items(ents, args.data, shaping, gamma)
    v0, end, vs, err = [], [], [], []
    with torch.no_grad():
        for W in (T.loader(WindowData(args.data, ents, items, ck["stats"], shaping, gamma), args.batch, args.workers, 0, False) if len(items) else []):
            W = T.to(W, dev)
            z0 = m.observe(W["rgb"], W["depth"], W["q"])[:, 0]          # observation 0 of the window (as validate's _v0)
            v0.append(m.unshaped_value(z0).float().cpu().numpy())
            vs.append(m.scalar(m.V(z0))[..., 0].mean(0).float().cpu().numpy())
            pr = W["probe"][:, 0].cpu().numpy()                          # FK labels of observation 0 (along, lx, ly, angle)
            err.append(np.abs(m.phi_hat(z0).float().cpu().numpy() + (np.abs(pr[:, 0]) + np.hypot(pr[:, 1], pr[:, 2]) + pr[:, 3])))
            end.append(W["end_seated"].cpu().numpy() > 0.5)
    v = np.concatenate(v0) if v0 else np.zeros(0)
    e = np.concatenate(end) if end else np.zeros(0, bool)
    if not len(v):
        raise SystemExit(f"no contact push of the selection grasps in {args.data}: no threshold")
    if not np.isfinite(v).all():
        raise SystemExit(f"non-finite values in {(~np.isfinite(v)).sum()} of {len(v)} windows: no threshold")
    vsh, pe = np.concatenate(vs), np.concatenate(err)
    rec = {"value": "unshaped", "ckpt": args.ckpt, "step": int(ck.get("step", -1)), "n_jammed": int((~e).sum()), "n_inserted": int(e.sum()),
           "shaped_jammed_median": float(np.median(vsh[~e])) if (~e).any() else None,
           "shaped_inserted_median": float(np.median(vsh[e])) if e.any() else None,
           "phi_probe_abs_err": float(pe.mean()), "phi_probe_abs_err_value_units": float(pe.mean() * shaping),
           "jammed_median": float(np.median(v[~e])) if (~e).any() else None,
           "inserted_median": float(np.median(v[e])) if e.any() else None, "v_low": None}
    if rec["jammed_median"] is None or rec["inserted_median"] is None:
        rec["reason"] = "one of the two kinds of push is missing"
    elif rec["inserted_median"] <= rec["jammed_median"]:
        rec["reason"] = "the value does not rank inserted pushes above jammed ones"
    else:
        rec["v_low"] = (rec["jammed_median"] + rec["inserted_median"]) / 2
    tmp = args.out + ".tmp"
    Path(tmp).write_text(json.dumps(rec, indent=1))
    os.replace(tmp, args.out)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
