#!/bin/bash
# 2026-11-16: pilot of the agreed model (wm/tdmpc/model.py) on the HELD-OUT part of the branch set, while the training
# part is still being collected. Purpose: see that the losses fall and the latent does not collapse before the long
# run. The model is thrown away (it has seen the held-out data); nothing from it is a result.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
WM_GPU_MEM_FRACTION=0.30 CUDA_VISIBLE_DEVICES=0 systemd-run --user --scope -q -p MemoryMax=48G nice -n 10 \
  python world_model/scripts/train/train_branch_wm.py --data /mnt/storage3/qingxin/data/wm_v1b \
  --out /mnt/storage3/qingxin/runs/branch_wm_pilot_on_val --train-split ood_val --steps 3000 --files 16 --per-file 2 \
  --workers 10 --dev-every 1000 --dev-branches 80 --save-every 500 > $Q/logs/branch_wm_pilot.log 2>&1
echo "[$(date +%m-%d\ %H:%M)] pilot ended with code $?" >> $Q/logs/pipeline_tdmpc.log
