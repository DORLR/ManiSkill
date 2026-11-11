#!/bin/bash
# v4d (2026-11-11): policy-independent world model. After the contact-exploration collection (tmux
# qingxin-explore): index -> commanded hand poses -> crops -> crop coverage gate -> exploration data gate
# (physics sanity, no truth in commands) -> latents -> smoke tests -> world model + success classifier from
# scratch on demonstrations + exploration (no policy data). Stops at the first failure.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
C=world_model/configs/v4
X=$Q/data/wm_v1x
CK=$Q/runs/checks_v4/explore
SM=$Q/cache/smoke_v4d
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4d_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4d_$name.log) -- pipeline stopped"; exit 1; fi
}
mkdir -p $CK; rm -rf $SM; mkdir -p $SM
while tmux has-session -t qingxin-explore 2>/dev/null; do sleep 60; done
grep -q "contact exploration: collection finished" $LOG || { stamp "FAILED: exploration collection did not finish"; exit 1; }
run index 0.02 python $S/data/build_index.py $X --workers 16
stamp "exploration index: $(tail -1 $Q/logs/v4d_index.log | cut -c1-200)"
run ee 0.02 python $S/data/compute_ee.py $X --out ${X}_ee --workers 8
stamp "commanded hand poses: $(tail -1 $Q/logs/v4d_ee.log | cut -c1-200)"
run crops 0.08 python $S/data/place_crops.py --data $X --unet $Q/runs/unet_v4/best.pt --out ${X}_crops --workers 12
run crop_gate 0.02 python $S/analysis/check_crop_coverage.py --data $X --crops ${X}_crops --geometry $Q/data/wm_v1_labelgeom --out $CK/crops --n 800
run data_gate 0.02 python $S/analysis/check_exploration.py --data $X --ee ${X}_ee --policy-episodes $Q/runs/checks_v4/imagination/episodes --out $CK/report.json
python $CK/explore_viz.py $X $CK/full_sheet > /dev/null 2>&1
run encode 0.10 python $S/train/encode_latents_v4.py --ae $Q/runs/crop_ae_v4b/best.pt --data $X --ee ${X}_ee --crops ${X}_crops --out $Q/data/wm_v4x_latents --workers 12
python - <<PY >> $LOG
import glob, numpy as np
fs = glob.glob("$Q/data/wm_v4x_latents/**/*.npz", recursive=True)
bad = sum(int(not np.isfinite(np.load(f)["mean"].astype(np.float32)).all()) for f in fs[::20])
ins = np.mean([np.load(f)["inserted"].any() for f in fs[::5]])
print("[exploration latents] files %d, sampled files with non-finite latents: %d, trajectories ending inserted (sample) %.1f%%" % (len(fs), bad, 100 * ins))
PY
grep -q "exploration latents.*non-finite latents: 0," $LOG || { stamp "FAILED: exploration latents"; exit 1; }
# smoke: both trainers with the real configs (two sources, new label), a few steps
python - <<PY
import json
for f, extra in (("wm", dict(workers=2, batch_size=8, accum=1, eval_batches=1, eval_sampling_steps=5, eval_every=20, save_every=10)),
                 ("success", dict(workers=2, batch_size=16, eval_batches=2, eval_every=20, save_every=10))):
    c = json.load(open(f"$C/{f}.json")); c.update(extra); json.dump(c, open(f"$SM/{f}.json", "w"), indent=1)
PY
run smoke_wm 0.10 python $S/train/train_wm4.py --config $SM/wm.json --run $SM/wm --steps 20
run smoke_success 0.05 python $S/train/train_success.py --config $SM/success.json --run $SM/success --steps 20
stamp "smoke: $(grep -o '"val_selection_auc": [0-9.]*' $Q/logs/v4d_smoke_success.log | tail -1); wm val keys $(grep -o '"val_[^"]*pred_score"' $Q/logs/v4d_smoke_wm.log | sort -u | wc -l)"
rm -rf $SM
for r in wm_v4d success_v4d; do [ -e $Q/runs/$r ] && { stamp "FAILED: $Q/runs/$r already exists"; exit 1; }; done
tmux new-session -d -s qingxin-v4d-wm "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.26 nice -n 10 python $S/train/train_wm4.py --config $C/wm.json --run $Q/runs/wm_v4d > $Q/logs/v4d_wm.log 2>&1"
tmux new-session -d -s qingxin-v4d-success "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.06 nice -n 10 python $S/train/train_success.py --config $C/success.json --run $Q/runs/success_v4d > $Q/logs/v4d_success.log 2>&1"
stamp "training started: world model (runs/wm_v4d) and success classifier (runs/success_v4d), demonstrations + exploration, from scratch"
