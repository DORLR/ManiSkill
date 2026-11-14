#!/bin/bash
# After each scheduled evaluation with the EMA weights (eval_at.sh), the same evaluation with the trained weights
# themselves: early in training the EMA weights are still mostly the random start (decay 0.9999), so their pictures
# say nothing about what has been learned. Runs only when no other evaluation is running.
#   usage: eval_at_model.sh <steps ...>
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
RUN=$Q/runs/nwm_asis/plug_cdit_b
LOG=$Q/logs/pipeline_v4.log
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
for K in "$@"; do
    until [ -f $Q/runs/nwm_asis/eval_$K/metrics.json ] || grep -q "evaluation at $K" $LOG; do
        tmux has-session -t qingxin-nwm-eval 2>/dev/null || break
        sleep 60
    done
    while pgrep -u $(id -u) -f eval_plug.py > /dev/null; do sleep 30; done
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 15000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] || { stamp "NWM as it is: trained-weights evaluation at $K skipped (GPU0 free $f MiB, RAM $a GB, pressure $p %)"; continue; }
    echo "$(date +%H:%M) evaluating trained weights at $K: GPU0 free $f MiB, RAM available $a GB, pressure $p %"
    O=$Q/runs/nwm_asis/eval_${K}_model
    if CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.12 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/nwm_asis/eval_plug.py --weights model --ckpt $RUN/checkpoints/latest.pth.tar --out $O > $Q/logs/nwm_asis_eval_${K}_model.log 2>&1; then
        stamp "NWM as it is, trained weights (not EMA), scheduled for step $K (pictures in runs/nwm_asis/eval_${K}_model): $(python -c "
import json; r = json.load(open('$O/metrics.json'))
t = {k: (v['lpips_predicted'], v['lpips_last_frame_again']) for k, v in r['time'].items()}
ro = {k: (v['lpips_predicted'], v['lpips_first_frame_again']) for k, v in r['rollout_recorded_commands'].items() if k.endswith('+8s')}
print('trained steps', r['train_steps'], '| LPIPS predicted vs last-frame-again, one prediction:', t, '| rollout at 8 s:', ro, '| push:', r['push_command_keeps_advancing'])")"
    else
        stamp "FAILED: NWM as it is, trained-weights evaluation at $K (logs/nwm_asis_eval_${K}_model.log)"
    fi
done
