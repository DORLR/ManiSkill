#!/bin/bash
# 2026-11-15: smoke run of the branch set (collect/branch.py): every 20th train grasp, 2 starts x 4 branches, the full 60 s,
# 640 x 640 pictures (WM_CAM_SCALE=2). Purpose: sizes, speed, how branches differ, when to stop a blocked branch.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_branch.log
X=/mnt/storage3/qingxin/data/wm_v1b_smoke
W=${WORKERS:-6}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) ))
p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
a=$(free -g | awk 'NR==2{print $7}')
if ! { [ "$f" -gt 30000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ]; }; then stamp "branch smoke NOT started: GPU1 free $f MiB, RAM $a GB, pressure $p %"; exit 1; fi
stamp "start branch smoke ($W workers; GPU1 free $f MiB, RAM $a GB, pressure $p %)"
if WM_CAM_SCALE=2 WM_GPU_MEM_FRACTION=0.03 CUDA_VISIBLE_DEVICES=1 systemd-run --user --scope -q -p MemoryMax=40G nice -n 10 \
    python world_model/scripts/data/collect_branch.py --grasps $Q/data/wm_v1 --out $X --split train --every 20 --starts 2 --branches 4 \
    --workers $W --vae /mnt/storage3/qingxin/runs/vae_v3g/best.pt --policy /mnt/storage3/qingxin/runs/policy_v3g/best.pt \
    > "$Q/logs/branch_smoke.log" 2>&1; then
    stamp "done branch smoke: $(find $X -name 's*_b*.npz' | wc -l) branches, $(du -sh $X | cut -f1)"
else stamp "FAILED branch smoke (logs/branch_smoke.log)"; fi
