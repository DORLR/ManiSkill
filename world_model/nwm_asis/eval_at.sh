#!/bin/bash
# Evaluate the running NWM-as-is training when it passes the given steps (reads checkpoints/latest.pth.tar right after it
# is written; no copies -- the disk is nearly full). One process, after a look at the machine.
#   usage: eval_at.sh <steps ...>
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
RUN=$Q/runs/nwm_asis/plug_cdit_b
LOG=$Q/logs/pipeline_v4.log
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
saved() { sed 's/\x1b\[[0-9;]*m//g' $RUN/log.txt 2>/dev/null | grep "Saved checkpoint" | wc -l; }
for K in "$@"; do
    while [ $(( $(saved) * 5000 )) -lt $K ]; do
        tmux has-session -t qingxin-nwm 2>/dev/null || { stamp "NWM as it is: the training ended before step $K"; exit 1; }
        sleep 60
    done
    sleep 20
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 15000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] || { stamp "NWM as it is: evaluation at $K skipped (GPU0 free $f MiB, RAM $a GB, pressure $p %)"; continue; }
    echo "$(date +%H:%M) evaluating at $K: GPU0 free $f MiB, RAM available $a GB, pressure $p %"
    O=$Q/runs/nwm_asis/eval_$K
    if CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.12 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/nwm_asis/eval_plug.py --ckpt $RUN/checkpoints/latest.pth.tar --out $O > $Q/logs/nwm_asis_eval_$K.log 2>&1; then
        stamp "NWM as it is, step $K (pictures in runs/nwm_asis/eval_$K): $(python -c "
import json; r = json.load(open('$O/metrics.json'))
t = {k: (v['lpips_predicted'], v['lpips_last_frame_again']) for k, v in r['time'].items()}
ro = {k: (v['lpips_predicted'], v['lpips_first_frame_again']) for k, v in r['rollout_recorded_commands'].items() if k.endswith('+8s')}
print('trained steps', r['train_steps'], '| LPIPS predicted vs last-frame-again, one prediction:', t, '| rollout at 8 s:', ro, '| push:', r['push_command_keeps_advancing'])")"
    else
        stamp "FAILED: NWM as it is, evaluation at $K (logs/nwm_asis_eval_$K.log)"
    fi
done
