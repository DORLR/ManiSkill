#!/bin/bash
# v4b (2026-11-10 afternoon): crops re-placed with the fixed plug crop and the gated, box-centred opening crop;
# checked on a fresh sample (stop if any stage within 15 mm has < 99.5 % of pin tips / openings inside);
# then the crop autoencoder from scratch (200 validation batches, patience 10). Later stages are started
# by hand after the autoencoder is checked.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/v4b_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/v4b_$name.log) -- pipeline stopped"; exit 1; fi
}
rm -rf $Q/data/wm_v1_crops
run crops 0.08 python $S/data/place_crops.py --data $Q/data/wm_v1 --unet $Q/runs/unet_v4/best.pt --out $Q/data/wm_v1_crops --workers 12
run crop_check 0.04 python $Q/runs/checks_v4/crop_files_check.py
stamp "stored crops: $(grep -A6 'openings in opening crop' $Q/logs/v4b_crop_check.log | tr '\n' ' ' | cut -c1-600)"
rm -rf $Q/data/wm_v1_crops_old_pinsdrift
run crop_ae 0.12 python $S/train/train_crop_ae.py --config world_model/configs/v4/crop_ae.json --run $Q/runs/crop_ae_v4b
