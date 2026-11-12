#!/bin/bash
# 2026-11-12: is the image resolution the limit of reading the pin tip's offset? (probe_pixels.py reached about one
# pixel, 0.22 mm, at 320 x 320.) A small exploration sample is recorded at twice the pixel density (WM_CAM_SCALE=2:
# 640 x 640, same camera and field of view), every third training grasp and the held-out shapes, and
# probe_resolution.py compares a reader of the 640 frames with the same reader of the frames reduced to 320.
# A diagnosis: the sample is not training data and is deleted afterwards if the answer is no.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=0
LOG=$Q/logs/pipeline_v4.log
X=$Q/data/wm_hr_test
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
stamp "resolution test: recording the 640 x 640 sample (8 workers) -> $X"
WM_CAM_SCALE=2 nice -n 10 python world_model/scripts/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split train --per-grasp 8 --every 3 --workers 8 > $Q/logs/hr_collect_train.log 2>&1 || { stamp "FAILED: resolution test, train sample"; exit 1; }
WM_CAM_SCALE=2 nice -n 10 python world_model/scripts/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split ood_val --per-grasp 4 --workers 8 --only G46 G47 G48 G55 G65 G75 G85 G95 G105 G115 G125 G135 G145 > $Q/logs/hr_collect_val.log 2>&1 || { stamp "FAILED: resolution test, held-out sample"; exit 1; }
stamp "resolution test: sample recorded: $(find $X -name 't*.npz' | wc -l) trajectories, $(du -sh $X | cut -f1); probe started (GPU 0)"
if WM_GPU_MEM_FRACTION=0.12 nice -n 10 python world_model/scripts/analysis/probe_resolution.py --data $X --out $Q/runs/checks_v5/probe_resolution.json > $Q/logs/v5_probe_resolution.log 2>&1; then
    stamp "resolution test (held-out shapes; tip offset error mm: blind | 320 | 640): $(python -c "
import json; r = json.load(open('$Q/runs/checks_v5/probe_resolution.json'))
print({k: (v['blind_error_mm'], v['low_320_error_mm'], v['high_640_error_mm'], v['n']) for k, v in r.items() if isinstance(v, dict)})")"
else
    stamp "FAILED: probe_resolution (logs/v5_probe_resolution.log)"
fi
