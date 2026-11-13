#!/bin/bash
# The approach check (scripts/analysis/check_wm6_approach.py) on a checkpoint; one line in the pipeline log.
#   usage: v6_approach.sh <checkpoint> <tag>
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
G=$1; TAG=$2
if WM_GPU_MEM_FRACTION=0.08 nice -n 10 python world_model/scripts/analysis/check_wm6_approach.py --ckpt $G --out $Q/runs/checks_v6/approach_$TAG.json > $Q/logs/v6_approach_$TAG.log 2>&1; then
    echo "[$(date +%m-%d\ %H:%M)] v6 approach ($TAG; push-only held-out starts in free space, a script pushes 1.2 mm ahead of the imagined hand for 16 s): $(python -c "
import json; r = json.load(open('$Q/runs/checks_v6/approach_$TAG.json'))
print({k: v for k, v in r.items() if k in ('really_blocked', 'really_inserted') or k.startswith('auc') or k.startswith('imagined_free')})")" >> $Q/logs/pipeline_v4.log
else
    echo "[$(date +%m-%d\ %H:%M)] FAILED: v6 approach ($TAG; logs/v6_approach_$TAG.log)" >> $Q/logs/pipeline_v4.log
fi
