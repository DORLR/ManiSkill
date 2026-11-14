#!/bin/bash
# eval_both2.sh: eval_both.sh with the review's evaluation arms (2026-11-13). Replaces eval_at.sh + eval_at_model.sh (review 2026-11-13: the second one re-read latest.pth.tar after the first had
# run for ~30 min, so it could load a later or half-written checkpoint). Here: as soon as a save is seen, the checkpoint
# is copied once to the archive disk as ckpt_<step>.pth.tar, its step checked, and both the EMA weights (what NWM's
# inference loads) and the trained weights are evaluated from that copy, one process at a time.
#   usage: eval_both.sh <steps ...>
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
RUN=$Q/runs/nwm_asis/plug_cdit_b
KEEP=/mnt/storage3/qingxin/runs/nwm_asis_eval_checkpoints
LOG=$Q/logs/pipeline_v4.log
mkdir -p $KEEP
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
saved() { sed 's/\x1b\[[0-9;]*m//g' $RUN/log.txt 2>/dev/null | grep "Saved checkpoint" | wc -l; }
for K in "$@"; do
    while [ $(( $(saved) * 5000 )) -lt $K ]; do
        tmux has-session -t qingxin-nwm 2>/dev/null || { stamp "NWM as it is: the training ended before step $K"; exit 1; }
        sleep 30
    done
    C=$KEEP/ckpt_$K.pth.tar
    cp $RUN/checkpoints/latest.pth.tar $C.part && mv $C.part $C
    S=$(python -c "import torch; print(torch.load('$C', map_location='cpu', weights_only=False)['train_steps'])" 2>/dev/null || echo -1)
    [ "$S" -eq "$K" ] || { stamp "NWM as it is: the copy for step $K holds step $S -- not evaluated"; continue; }
    WL="model ema"; [ $K -le 20000 ] && WL=model        # at 20k the EMA weights are still 13.5 % random start
    for WTS in $WL; do
        f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
        [ "$f" -gt 15000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] || { stamp "NWM as it is: $WTS evaluation at $K skipped (GPU0 free $f MiB, RAM $a GB, pressure $p %)"; continue; }
        echo "$(date +%H:%M) evaluating $WTS weights at $K: GPU0 free $f MiB, RAM available $a GB, pressure $p %"
        O=$Q/runs/nwm_asis/eval_${K}_$WTS
        if CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.12 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/nwm_asis/eval_plug.py --weights $WTS --ckpt $C --out $O > $Q/logs/nwm_asis_eval_${K}_$WTS.log 2>&1; then
            stamp "NWM as it is, step $K, $WTS weights (pictures in runs/nwm_asis/eval_${K}_$WTS; the push is an out-of-range probe, not a prediction error): $(python -c "
import json; r = json.load(open('$O/metrics.json'))
t = {k: (v['lpips_predicted'], v['lpips_last_frame_again']) for k, v in r['time'].items()}
ro = {k: (v['lpips_predicted'], v['lpips_first_frame_again']) for k, v in r['rollout_recorded_commands'].items() if k.endswith('+8s')}
pu = {k: {a: v[a]['picture_change_0_to_8s'] for a in v if a[:2] in ('A_', 'B_', 'C_', 'D_')} for k, v in r['push_arms'].items()}
au = {k: (v['action_use']['lpips_true_vs_shuffled_prediction'], v['action_use']['lpips_true_vs_no_motion_prediction'], v['lpips_vae_floor']) for k, v in r['time'].items() if 'action_use' in v}
print('trained steps', r['train_steps'], '| LPIPS predicted vs last-frame-again, one prediction:', t, '| rollout at 8 s:', ro, '| action use (true vs shuffled, true vs no motion, vae floor):', au, '| push arms, picture change 0-8 s:', pu)")"
        else
            stamp "FAILED: NWM as it is, $WTS evaluation at $K (logs/nwm_asis_eval_${K}_$WTS.log)"
        fi
    done
done
