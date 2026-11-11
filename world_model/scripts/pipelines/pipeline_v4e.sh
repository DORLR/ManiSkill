#!/bin/bash
# v4e (2026-11-11 09:10): the autoencoder gate of pipeline_v4d failed -- the crop autoencoder, trained on
# demonstrations only, read exploration frames much worse (pins at the board face, within 3 mm of the axis:
# 0.84 mm across the axis vs 0.12 mm on demonstrations). So: crop autoencoder from scratch on demonstrations +
# exploration -> gate: on exploration it must be clearly better than the old one and not worse on demonstrations
# -> latents of both sources -> FM, world model and success classifier from scratch on the new latents.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
X=$Q/data/wm_v1x
CK=$Q/runs/checks_v4/explore
SM=$Q/cache/smoke_v4e
AE=$Q/runs/crop_ae_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4e_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4e_$name.log) -- pipeline stopped"; exit 1; fi
}
rm -rf $SM; mkdir -p $SM
python - <<PY
import json
c = json.load(open("$C/crop_ae.json")); c.update(workers=4, eval_batches=2, batch_size=8); json.dump(c, open("$SM/crop_ae.json", "w"))
PY
run smoke_crop_ae 0.08 python $S/train/train_crop_ae.py --config $SM/crop_ae.json --run $SM/crop_ae --steps 30
[ -f $AE/done ] || { run crop_ae 0.14 python $S/train/train_crop_ae.py --config $C/crop_ae.json --run $AE && touch $AE/done; }
run ae_readout 0.06 python $S/analysis/check_ae_on_exploration.py --ae $AE/best.pt --demos $Q/data/wm_v1 --demo-crops $Q/data/wm_v1_crops --explore $X --explore-crops ${X}_crops --out $CK/ae_readout_v4e.json --no-gate
python - <<PY >> $LOG
import json, sys
old, new = json.load(open("$CK/ae_readout.json")), json.load(open("$CK/ae_readout_v4e.json"))
face, near = "13-17 mm (pins at the board face) | within 3 mm of the axis", "<5 mm left | within 3 mm of the axis"
g = lambda r, s, k: r[s][k]["across_axis_err_mm"]
rows = {"exploration at the face": (g(old, "exploration", face), g(new, "exploration", face)),
        "exploration near the hole": (g(old, "exploration", near), g(new, "exploration", near)),
        "demonstrations at the face": (g(old, "demonstrations", face), g(new, "demonstrations", face)),
        "demonstrations near the hole": (g(old, "demonstrations", near), g(new, "demonstrations", near))}
ok = rows["exploration at the face"][1] <= 0.7 * rows["exploration at the face"][0] and \
     rows["demonstrations near the hole"][1] <= max(2 * rows["demonstrations near the hole"][0], 0.2)
print("[autoencoder v4e vs v4b, error across the axis mm (old -> new)] " + "; ".join("%s %.3f -> %.3f" % (k, a, b) for k, (a, b) in rows.items()) + " | GATE " + ("PASS" if ok else "FAIL"))
PY
tail -1 $LOG | grep -q "GATE PASS" || { stamp "FAILED: autoencoder v4e not clearly better on exploration frames (or worse on demonstrations) -- stopped"; exit 1; }
( run encode_demos 0.10 python $S/train/encode_latents_v4.py --ae $AE/best.pt --data $Q/data/wm_v1 --ee $Q/data/wm_v1_ee --crops $Q/data/wm_v1_crops --out $Q/data/wm_v4e_latents --workers 12 ) &
P1=$!
( run encode_explore 0.06 python $S/train/encode_latents_v4.py --ae $AE/best.pt --data $X --ee ${X}_ee --crops ${X}_crops --out $Q/data/wm_v4ex_latents --workers 8 ) &
P2=$!
wait $P1 || exit 1
wait $P2 || exit 1
python - <<PY >> $LOG
import glob, json, numpy as np
for name in ("wm_v4e_latents", "wm_v4ex_latents"):
    fs = glob.glob(f"$Q/data/{name}/**/*.npz", recursive=True)
    bad = sum(int(not np.isfinite(np.load(f)["mean"].astype(np.float32)).all()) for f in fs[::25])
    print("[latents %s] files %d, scale %.4f, sampled files with non-finite latents: %d" % (name, len(fs), json.load(open(f"$Q/data/{name}/stats.json"))["scale"], bad))
PY
[ "$(tail -2 $LOG | grep -c 'non-finite latents: 0')" = "2" ] || { stamp "FAILED: latents"; exit 1; }
python - <<PY
import json
for f, extra in (("wm", dict(workers=2, batch_size=8, accum=1, eval_batches=1, eval_sampling_steps=5, eval_every=20, save_every=10)),
                 ("success", dict(workers=2, batch_size=16, eval_batches=2, eval_every=20, save_every=10)),
                 ("policy", dict(workers=2, batch_size=16, eval_batches=2, eval_every=20))):
    c = json.load(open(f"$C/{f}.json")); c.update(extra); json.dump(c, open(f"$SM/{f}.json", "w"), indent=1)
PY
run smoke_wm 0.10 python $S/train/train_wm4.py --config $SM/wm.json --run $SM/wm --steps 20
run smoke_success 0.05 python $S/train/train_success.py --config $SM/success.json --run $SM/success --steps 20
run smoke_policy 0.05 python $S/train/train_policy.py --config $SM/policy.json --run $SM/policy --max-steps 20
rm -rf $SM
for r in wm_v4e success_v4e policy_v4e; do [ -e $Q/runs/$r ] && { stamp "FAILED: $Q/runs/$r already exists"; exit 1; }; done
tmux new-session -d -s qingxin-v4e-wm "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.26 nice -n 10 python $S/train/train_wm4.py --config $C/wm.json --run $Q/runs/wm_v4e > $Q/logs/v4e_wm.log 2>&1"
tmux new-session -d -s qingxin-v4e-success "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/train/train_success.py --config $C/success.json --run $Q/runs/success_v4e > $Q/logs/v4e_success.log 2>&1"
tmux new-session -d -s qingxin-v4e-policy "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v4e > $Q/logs/v4e_policy.log 2>&1"
stamp "training started: world model (runs/wm_v4e), success classifier (runs/success_v4e), FM (runs/policy_v4e); latents of crop_ae_v4e; demonstrations + exploration; from scratch"
