"""Check a branch set (collect/branch.py): outcomes per branch kind, the retry upper bound, the FK label against the
simulator's truth, how far the branches of a start spread, and when a blocked branch could be stopped.

    python world_model/scripts/analysis/check_branch.py /mnt/storage3/qingxin/data/wm_v1b_smoke [out.json]
"""

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wm.data.outcome import diagnosis, inserted

root = Path(sys.argv[1])
rep = {}
gs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(f"{root}/*/*/grasp.json"))]
rep["grasps"] = len(gs)
rep["grasp_errors"] = Counter(str(g.get("error"))[:60] for g in gs if g.get("error"))
starts = [(g, s) for g in gs for s in g["starts"] if s.get("branches")]
rep["starts"] = len(starts)
rep["placement_failed"] = sum(1 for g in gs for s in g["starts"] if s.get("outcome") == "placement_failed")
br = [(g, s, b) for g, s in starts for b in s["branches"]]
rep["branches"] = len(br)
rep["seconds_per_grasp_median"] = float(np.median([g["seconds"] for g in gs]))
rep["bytes_per_branch_MB"] = float(sum(g.get("bytes", 0) for g in gs) / max(len(br), 1) / 1e6)
rep["restore_dev_mm_max"] = float(max(b["restore_dev_mm_deg"][0] for _, _, b in br))
rep["ik_missed_steps"] = int(sum(b.get("ik_missed_steps", 0) for _, _, b in br))

# ---- outcomes: FK stop reason against the simulator's truth
rep["stop_vs_truth"] = {f"{a}/{t}": n for (a, t), n in Counter((b["stopped"], b["outcome"]) for _, _, b in br).items()}
ok = lambda b: b["outcome"] == "inserted"
by_sigma = defaultdict(list)
for _, _, b in br:
    by_sigma[round(b["noise"]["sigma_m"] * 1000, 2)].append(ok(b))
rep["inserted_by_bias_sigma_mm"] = {str(k): [int(sum(v)), len(v)] for k, v in sorted(by_sigma.items())}
pure = lambda s: [b for b in s["branches"] if b["noise"]["sigma_m"] == 0]
two = [s for _, s in starts if len(pure(s)) >= 2]
rep["pure_pairs_both_one_none"] = [sum(ok(pure(s)[0]) and ok(pure(s)[1]) for s in two),
                                   sum(ok(pure(s)[0]) != ok(pure(s)[1]) for s in two),
                                   sum(not ok(pure(s)[0]) and not ok(pure(s)[1]) for s in two)]
rep["any_pure_branch"] = [int(sum(any(ok(b) for b in pure(s)) for _, s in starts)), len(starts)]
first = [ok(s["branches"][0]) for _, s in starts]
anyb = [any(ok(b) for b in s["branches"]) for _, s in starts]
rep["policy_alone_first_branch"] = [int(sum(first)), len(first)]
rep["any_of_the_branches"] = [int(sum(anyb)), len(anyb)]
rep["branches_inserted_per_start"] = dict(sorted(Counter(sum(ok(b) for b in s["branches"]) for _, s in starts).items()))
rep["decisions_of_inserted_pct_10_50_90"] = np.percentile([b["decisions"] for _, _, b in br if ok(b)] or [0], [10, 50, 90]).tolist()

# ---- per-step analysis from the files
depth_err, seat_agree, spread, stall = [], Counter(), [], []
for g, s in starts:
    ends = []
    for b in s["branches"]:
        f = root / g["port"] / g["grasp"] / f"s{s['idx']:02d}_b{b['idx']}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            meta = json.loads(str(z["meta"]))
            hand, rel, dfk, dec = z["hand"], z["eval_rel"], z["depth_fk_m"], z["dec_step"]
        H, axis, L = np.asarray(meta["H_star"]), np.asarray(meta["axis"]), meta["pin_length_m"]
        d = diagnosis(rel)
        true_depth = np.clip(L - d["depth_left_m"], 0, L)
        true_depth[d["lateral_m"] > 0.002] = 0.0                       # beside the hole: not in it
        depth_err.append(np.abs(dfk - true_depth).max() * 1000)
        seat_fk = (np.linalg.norm(hand[:, :3, 3] - H[:3, 3], axis=1) < 0.001)
        seat_agree[(bool(seat_fk[-1]), bool(inserted(rel[-1])))] += 1
        # where the hand is across the axis when the pins first reach the board face (FK: along >= -(L + 0.5 mm))
        relp = (hand[:, :3, 3] - H[:3, 3])
        along = relp @ axis
        across = relp - np.outer(along, axis)
        at_face = np.nonzero(along >= -(L + 0.0005))[0]
        if len(at_face):
            ends.append(across[at_face[0]])
        # stall: decisions after which the hand has moved less than 0.2 mm in the last 5 decisions (2 s)
        hp = hand[dec, :3, 3]
        moved = np.array([np.linalg.norm(hp[k] - hp[k - 5]) for k in range(5, len(hp))])
        st = np.nonzero(moved < 0.0002)[0]
        stall.append({"inserted": bool(inserted(rel[-1])), "decisions": int(len(dec)),
                      "first_stall": int(st[0] + 5) if len(st) else None,
                      "reached_face": int(np.searchsorted(dec, at_face[0])) if len(at_face) else None})
    if len(ends) >= 2:
        e = np.asarray(ends)
        spread.append(np.linalg.norm(e - e.mean(0), axis=1).max() * 1000)
rep["depth_label_max_abs_error_mm_pct_50_90_100"] = np.percentile(depth_err, [50, 90, 100]).tolist() if depth_err else None
rep["seated_fk_vs_truth_inserted"] = {f"fk={a},truth={t}": n for (a, t), n in seat_agree.items()}
rep["across_spread_at_face_mm_pct_10_50_90"] = np.percentile(spread, [10, 50, 90]).tolist() if spread else None
ins = [x for x in stall if x["inserted"]]
rep["inserted_after_a_stall"] = [sum(x["first_stall"] is not None and x["first_stall"] < x["decisions"] - 1 for x in ins), len(ins)]
rep["inserted_decisions_after_reaching_face_pct_50_90_100"] = np.percentile(
    [x["decisions"] - x["reached_face"] for x in ins if x["reached_face"] is not None] or [0], [50, 90, 100]).tolist()
fail = [x for x in stall if not x["inserted"]]
rep["failed_first_stall_decision_pct_10_50_90"] = np.percentile([x["first_stall"] for x in fail if x["first_stall"] is not None] or [0], [10, 50, 90]).tolist()
rep["failed_never_stalled"] = [sum(x["first_stall"] is None for x in fail), len(fail)]
rep["failed_never_reached_face"] = [sum(x["reached_face"] is None for x in fail), len(fail)]
print(json.dumps(rep, indent=1))
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_text(json.dumps(rep, indent=1))
