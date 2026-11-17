#!/bin/bash
# 2026-11-17 night: the two "goal" checks the user approved after the fourth review. Reads recorded branches only
# (no simulation): one scoring pass of the finished model (check_goal_reuse.py), then the analysis on the saved
# features (analyse_goal_reuse.py, CPU + a little GPU for small heads).
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
R=/mnt/storage3/qingxin/runs
O=$R/branch_wm_v1s4/goal_check
p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed "s/.*avg10=\([0-9.]*\).*/\1/" | cut -d. -f1)
line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1); g=${line%%,*}; f=${line##*, }
[ "$p" -lt 5 ] && [ "$f" -gt 25000 ] || { echo "[$(date +%m-%d\ %H:%M)] goal check not started: pressure $p %, best GPU free $f MiB" >> $Q/logs/pipeline_tdmpc.log; exit 1; }
echo "[$(date +%m-%d\ %H:%M)] start goal check features (GPU $g)" >> $Q/logs/pipeline_tdmpc.log
WM_GPU_MEM_FRACTION=0.12 CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q -p MemoryMax=40G nice -n 10 python world_model/scripts/analysis/check_goal_reuse.py \
  --ckpt $R/branch_wm_v1s4/last.pt --data /mnt/storage3/qingxin/data/wm_v1b --out $O --workers 8 > $Q/logs/goal_check_features.log 2>&1
c=$?
echo "[$(date +%m-%d\ %H:%M)] goal check features ended with code $c" >> $Q/logs/pipeline_tdmpc.log
[ $c -eq 0 ] || exit 1
WM_GPU_MEM_FRACTION=0.04 CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/scripts/analysis/analyse_goal_reuse.py \
  --data /mnt/storage3/qingxin/data/wm_v1b --out $O > $Q/logs/goal_check_analysis.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] goal check analysis ended with code $?" >> $Q/logs/pipeline_tdmpc.log
