#!/bin/bash
# Full CDiT training after the trial of pipeline_v1.sh, on the shared GPU 1 with a memory cap.
#   tmux new-session -d -s qingxin-wm "bash world_model/scripts/pipeline_wm_full_v1.sh"
set -euo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
export WM_GPU_MEM_FRACTION=0.2
LOG=$Q/logs/pipeline_wm_full_v1.log
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }

stamp "waiting for the CDiT trial"
until grep -q "stopping after the CDiT trial\|STOP" "$Q/logs/pipeline_v1.log" 2>/dev/null; do sleep 60; done
grep -q "CDiT trial ok" "$Q/logs/pipeline_v1.log" || { stamp "STOP: CDiT trial failed"; exit 1; }
if grep -q "stop_nonfinite\|NaN" "$Q/logs/train_wm_trial.log"; then stamp "STOP: non-finite values in the trial"; exit 1; fi
free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
[ "$free_mib" -gt 30000 ] || { stamp "STOP: only ${free_mib} MiB free on GPU 1"; exit 1; }
stamp "trial ok, ${free_mib} MiB free on GPU 1; CDiT training (memory cap 20 %)"
nice -n 10 python world_model/scripts/train_wm.py --config world_model/configs/wm_v1.json \
    --run "$Q/runs/wm_v1" > "$Q/logs/train_wm_v1.log" 2>&1
stamp "CDiT training finished"
