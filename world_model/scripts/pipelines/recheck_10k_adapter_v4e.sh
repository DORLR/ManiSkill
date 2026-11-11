#!/bin/bash
# 2026-11-11: imagination vs simulator at the 13k-step world model again, with the joint adapter retrained on
# demonstrations + exploration (online.json now points to it); same branch episodes, same classifier.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
I=$Q/runs/checks_v4/imagination_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python world_model/scripts/analysis/check_imagination4.py compare --config world_model/configs/v4/online.json --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --out "$I/report_10k_adapter_v4e.json" --imagined 4 --batch 64 > "$Q/logs/check4e_compare_10k_adapter_v4e.log" 2>&1; then
    stamp "imagination vs simulator (10k, retrained adapter): $(python -c "
import json; r = json.load(open('$I/report_10k_adapter_v4e.json')); print(r['3_closed_loop']['all'])")"
else
    stamp "FAILED: imagination vs simulator with the retrained adapter (logs/check4e_compare_10k_adapter_v4e.log)"
fi
