#!/bin/bash
# 2026-11-16 evening: offline measurement asked for by the user after the review (scripts/analysis/probe_lateral_direction.py):
# size of the lateral error at the board face, whether the policy keeps a lateral difference, whether V / the latent
# know which way the hole is. Reads recorded branches only; one scoring pass of the finished model.
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
R=/mnt/storage3/qingxin/runs
p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed "s/.*avg10=\([0-9.]*\).*/\1/" | cut -d. -f1)
line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1); g=${line%%,*}; f=${line##*, }
[ "$p" -lt 5 ] && [ "$f" -gt 25000 ] || { echo "[$(date +%m-%d\ %H:%M)] lateral probe not started: pressure $p %, best GPU free $f MiB" >> $Q/logs/pipeline_tdmpc.log; exit 1; }
echo "[$(date +%m-%d\ %H:%M)] start lateral probe (GPU $g)" >> $Q/logs/pipeline_tdmpc.log
WM_GPU_MEM_FRACTION=0.12 CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q -p MemoryMax=40G nice -n 10 python world_model/scripts/analysis/probe_lateral_direction.py \
  --ckpt $R/branch_wm_v1s4/last.pt --data /mnt/storage3/qingxin/data/wm_v1b --out $R/branch_wm_v1s4/lateral_probe --workers 8 > $Q/logs/lateral_probe.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] lateral probe ended with code $?" >> $Q/logs/pipeline_tdmpc.log
