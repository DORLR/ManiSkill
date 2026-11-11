#!/bin/bash
# 2026-11-11: vision-conditioned joint adapter (decided with the user): train from scratch, check on held-out
# exploration whether it knows advance / held before the tracking history shows it, then repeat the
# imagination-vs-simulator comparison of the 13k-step world model with it (same branch episodes, classifier).
#   tmux new-session -d -s qingxin-adapter-vision "bash world_model/scripts/pipelines/adapter_vision.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
I=$Q/runs/checks_v4/imagination_v4e
R=$Q/runs/adapter_vision_v4e
S=world_model/scripts/analysis
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
stamp "vision-conditioned joint adapter: training started (30000 steps)"
if ! WM_GPU_MEM_FRACTION=0.10 python world_model/scripts/train/train_adapter_vision.py --config world_model/configs/v4/adapter_vision.json --run "$R" > "$Q/logs/adapter_vision_v4e.log" 2>&1 || [ ! -f "$R/best.pt" ]; then
    stamp "FAILED: vision adapter training (logs/adapter_vision_v4e.log)"; exit 1
fi
stamp "vision adapter trained: best at step $(python -c "import torch;print(torch.load('$R/best.pt',map_location='cpu',weights_only=False)['step'])"), $(grep val_selection "$R/log.jsonl" | tail -n 1)"
if WM_GPU_MEM_FRACTION=0.06 python $S/check_adapter_decisions.py --config world_model/configs/v4/adapter_vision.json --adapters "$Q/runs/adapter_v4e/best.pt" "$R/best.pt" --out "$Q/runs/checks_v4/explore/adapter_decisions.json" > "$Q/logs/check_adapter_decisions.log" 2>&1; then
    stamp "adapter decisions on held-out exploration (AUC advanced vs held | mean predicted fraction when really held): $(python -c "
import json; r = json.load(open('$Q/runs/checks_v4/explore/adapter_decisions.json'))
print({a: {b: (v[b]['auc_advanced_vs_held'], v[b]['mean_predicted_fraction_held']) for b in ('lag<0.3mm', 'lag0.3-1mm')} for a, v in r.items() if isinstance(v, dict) and 'lag<0.3mm' in v})")"
else
    stamp "FAILED: check_adapter_decisions (logs/check_adapter_decisions.log)"; exit 1
fi
python -c "
import json; c = json.load(open('world_model/configs/v4/online.json')); c['actor']['adapter'] = '$R/best.pt'
json.dump(c, open('$I/online_adapter_vision.json', 'w'), indent=1)"
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python $S/check_imagination4.py compare --config "$I/online_adapter_vision.json" --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --out "$I/report_10k_adapter_vision.json" --imagined 4 --batch 64 > "$Q/logs/check4e_compare_10k_adapter_vision.log" 2>&1; then
    stamp "imagination vs simulator (10k, vision adapter): $(python -c "
import json; r = json.load(open('$I/report_10k_adapter_vision.json')); print(r['3_closed_loop']['all'])")"
else
    stamp "FAILED: imagination vs simulator with the vision adapter (logs/check4e_compare_10k_adapter_vision.log)"
fi
