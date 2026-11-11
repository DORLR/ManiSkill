#!/bin/bash
# 2026-11-11: which part makes imagination optimistic (classifier / world model / joints), world model step 13000.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
I=$Q/runs/checks_v4/imagination_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python world_model/scripts/analysis/check_imagination_parts.py --config world_model/configs/v4/online.json --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --adapters "$Q/runs/adapter_v4e/best.pt" --out "$I/parts_10k.json" > "$Q/logs/check4e_parts_10k.log" 2>&1; then
    stamp "imagination parts (10k), called inserted at some decision %, by simulator outcome: $(python -c "
import json; r = json.load(open('$I/parts_10k.json'))
print({n: {o: v['called_inserted_at_some_decision_pct'] for o, v in r[n].items()} for n in ('real', 'wm_real_joints', 'wm_adapter_v4e')})")"
else
    stamp "FAILED: check_imagination_parts (logs/check4e_parts_10k.log)"
fi
