"""v6: does the world model use the images when it predicts the hand's motion and the contact state? (review of 2026-11-13)

    python world_model/scripts/analysis/check_wm6_image_use.py --ckpt $Q/runs/checks_v6/joint_v6c_50000.pt --out ...
One prediction at +0.4 s from the real history (as the training validation), on held-out windows (ood_val), repeated on
the same samples and the same sampling noise with
    images       everything as recorded
    no_images    no image in any slot (all vision hidden; proprioception, commands and state as recorded)
    swapped      the images of another sample of the batch (proprioception, commands and state of this one)
    bit0 / bit1  images as recorded, the "pressed now" bit of the state forced to 0 / 1
Read from the extra outputs: the hand's advance along the axis in 0.4 s (mm), p(pressed) and p(inserted) at the target.

Why: the closed-loop checks (push, approach) show the imagined hand following the commands through the board whenever
the pressed bit is 0, at every checkpoint; no measurement so far shows that the hand output depends on the images at
all. Onset decisions are the test: the bit is 0, the hand advanced over the previous 0.4 s, the command leads by more
than 0.3 mm -- and the hand then really stops (< 0.1 mm) or really advances (> 0.3 mm). What tells the two apart in
proprioception and commands is limited (the previous decision advanced in both); the board is in the image. The score is
the AUC of the predicted advance for "advances" against "stops", with images and without.
Also counted from the labels alone: how often "pressed" at the target equals "pressed now" (what copying the bit gives),
and how often a truly held hand loses the label (label flicker).
"""
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from wm.data.frame_history import RES_SCALE, FrameHistoryDataset
from wm.models.history_cdit import create_diffusion
from wm.training.common import limit_gpu_memory, to_device
from wm.training.joint import encode_history, load_joint

ARMS = ["images", "no_images", "swapped", "bit0", "bit1"]


class WithPrevious(FrameHistoryDataset):
    """+ prev_adv_mm: the hand's advance along the axis over the 0.4 s before now (from the recorded hand poses)."""

    def __getitem__(self, k):
        out = super().__getitem__(k)
        i, j = self.items[k]
        e = self.entries[i]
        with np.load(e["path"]) as f:
            steps, hand = f["img_step"].astype(int), f["hand"]
        s, axis = int(steps[j]), np.asarray(e["R_P"], np.float64)[:, 0]
        prev = float((hand[s, :3, 3].astype(np.float64) - hand[s - 8, :3, 3]) @ axis) * 1000 if s >= 8 else float("nan")
        out["prev_adv_mm"] = torch.tensor(prev, dtype=torch.float32)
        return out


def auc(score, pos):
    """P(score of a positive > score of a negative)."""
    pos = pos.astype(bool)
    if pos.sum() < 5 or (~pos).sum() < 5:
        return None
    r = np.empty(len(score))
    r[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return round(float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())), 3)


mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--data", nargs="+", default=["/mnt/storage2/qingxin/data/wm_v1c", "/mnt/storage2/qingxin/data/wm_v1x", "/mnt/storage2/qingxin/data/wm_v1a"])
ap.add_argument("--batches", type=int, default=16)
ap.add_argument("--batch", type=int, default=48)
ap.add_argument("--sampling-steps", type=int, default=20)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
limit_gpu_memory()
dev = torch.device("cuda")
enc, head, wm, ck = load_joint(args.ckpt, dev)
sampler = create_diffusion(timestep_respacing=str(args.sampling_steps))


@torch.no_grad()
def predict(kw, shape, seed):
    torch.manual_seed(seed)
    sampler.p_sample_loop(wm.forward, shape, torch.randn(shape, device=dev), clip_denoised=False, model_kwargs=kw, progress=False, device=dev)
    a = wm.aux.float()
    return (a[:, 9] / RES_SCALE * 10).cpu().numpy(), torch.sigmoid(a[:, 19]).cpu().numpy(), torch.sigmoid(a[:, 18]).cpu().numpy()


rep = {"checkpoint": args.ckpt, "checkpoint_step": int(ck["step"]), "trained_on": ck["cfg"]["data"], "arms": ARMS}
raw = {}
for d in args.data:
    name = d.rstrip("/").split("/")[-1]
    ds = WithPrevious(d, d + "_ee", "ood_val", mask=None, max_frames=ck["cfg"]["max_frames"], fixed_shift=2)
    D = {}
    add = lambda k, v: D.setdefault(k, []).append(np.asarray(v))
    for i, b in enumerate(DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
                                     generator=torch.Generator().manual_seed(11))):
        if i >= args.batches:
            break
        b = to_device(b, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z, f_new, x = encode_history(enc, enc, b, grad=False)
        z = z.float()
        base = dict(path=b["path"], rel_t=b["rel_t"], z_hist=z, seen=b["seen"], proprio=b["proprio"], dt=b["dt"], avail=b["avail"], state=b["state"])
        perm = torch.roll(torch.arange(z.shape[0], device=dev), 1)
        seen_sw = b["seen"][perm] & b["avail"]
        s0, s1 = b["state"].clone(), b["state"].clone()
        s0[:, 0], s1[:, 0] = 0.0, 1.0
        arms = {"images": base,
                "no_images": {**base, "z_hist": torch.zeros_like(z), "seen": torch.zeros_like(b["seen"])},
                "swapped": {**base, "z_hist": z[perm] * seen_sw[..., None, None, None], "seen": seen_sw},
                "bit0": {**base, "state": s0}, "bit1": {**base, "state": s1}}
        for arm in ARMS:
            adv, pp, pi = predict(arms[arm], x.shape, 1000 + i)
            add("adv_" + arm, adv), add("pp_" + arm, pp), add("pi_" + arm, pi)
        add("res_ok", b["res_ok"].cpu().numpy()), add("adv_true", (b["res"][:, 9] / RES_SCALE * 10).cpu().numpy())
        add("lead", (b["path"][:, 13 * 9] * 10).cpu().numpy()), add("prev", b["prev_adv_mm"].cpu().numpy())
        add("state_pressed", b["state"][:, 0].cpu().numpy()), add("state_inserted", b["state"][:, 1].cpu().numpy())
        add("t_pressed", b["t_pressed"].cpu().numpy()), add("t_inserted", b["t_inserted"].cpu().numpy())
        add("n_seen", b["seen"].sum(1).cpu().numpy())
    D = {k: np.concatenate(v) for k, v in D.items()}
    raw.update({f"{name}/{k}": v for k, v in D.items()})
    ok, adv, lead, prev = D["res_ok"].astype(bool), D["adv_true"], D["lead"], D["prev"]
    sp, tp, si, ti = D["state_pressed"] > 0.5, D["t_pressed"].astype(bool), D["state_inserted"] > 0.5, D["t_inserted"].astype(bool)
    onset = ok & ~sp & (prev > 0.3) & (lead > 0.3)
    stops, goes = onset & (adv < 0.1), onset & (adv > 0.3)
    held = ok & (np.abs(adv) < 0.05) & (lead > 0.1)
    r = {"samples": int(len(ok)), "with_hand_target": int(ok.sum()),
         "labels_only": {"pressed_at_target_equals_pressed_now": mean(tp == sp), "pressed_changes": int((tp != sp).sum()),
                         "p_pressed_at_target_given_pressed_now": mean(tp[sp]), "p_pressed_at_target_given_not_pressed_now": mean(tp[~sp]),
                         "truly_held_with_bit_1": int((held & sp).sum()), "truly_held_with_bit_0": int((held & ~sp).sum()),
                         "truly_held_bit_1_keeps_label": mean(tp[held & sp]),
                         "inserted_at_target_equals_inserted_now": mean(ti == si), "inserted_changes": int((ti != si).sum())},
         "onset": {"really_stops": int(stops.sum()), "really_advances": int(goes.sum()),
                   "true_advance_mm": (mean(adv[stops]), mean(adv[goes])), "lead_mm": (mean(lead[stops]), mean(lead[goes])),
                   "previous_advance_mm": (mean(prev[stops]), mean(prev[goes])),
                   "auc_by_lead_alone": auc(lead[stops | goes], goes[stops | goes]),
                   "auc_by_previous_advance_alone": auc(prev[stops | goes], goes[stops | goes])}}
    for arm in ARMS:
        p, pp, pi = D["adv_" + arm], D["pp_" + arm] > 0.5, D["pi_" + arm] > 0.5
        ch_p, ch_i = tp != sp, ti != si
        r[arm] = {"onset_auc_advances_vs_stops": auc(p[stops | goes], goes[stops | goes]),
                  "onset_predicted_advance_mm_stops_advances": (mean(p[stops]), mean(p[goes])),
                  "onset_says_pressed_stops_advances": (mean(pp[stops]), mean(pp[goes])),
                  "held_predicted_advance_mm_bit1_bit0": (mean(p[held & sp]), mean(p[held & ~sp])),
                  "held_abs_predicted_advance_mm_bit1_bit0": (mean(np.abs(p[held & sp])), mean(np.abs(p[held & ~sp]))),
                  "advance_error_mm": mean(np.abs(p - adv)[ok]),
                  "advance_corr": round(float(np.corrcoef(p[ok], adv[ok])[0, 1]), 3) if ok.sum() > 2 else None,
                  "pressed_acc": mean(pp == tp), "pressed_acc_where_it_changes": mean((pp == tp)[ch_p]),
                  "says_pressed_given_bit_1_bit_0": (mean(pp[sp]), mean(pp[~sp])),
                  "inserted_acc": mean(pi == ti), "inserted_acc_where_it_changes": mean((pi == ti)[ch_i])}
    rep[name] = r
    print(name, json.dumps(r), flush=True)
json.dump(rep, open(args.out, "w"), indent=1)
np.savez_compressed(args.out.replace(".json", "_cases.npz"), **raw)
