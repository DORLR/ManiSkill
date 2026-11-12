#!/bin/bash
# 2026-11-11 evening: joint adapter conditioned on the success classifier's decision (decided with the user):
# train from scratch (12000 steps), check held windows, the hand creep in stuck states and the full
# imagination-vs-simulator comparison (world model step 13000, same branch episodes and classifier as before).
#   tmux new-session -d -s qingxin-adapter-outcome "bash world_model/scripts/pipelines/adapter_outcome.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
I=$Q/runs/checks_v4/imagination_v4e
R=$Q/runs/adapter_outcome_v4e
S=world_model/scripts/analysis
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
stamp "outcome-conditioned joint adapter: training started (12000 steps)"
if ! WM_GPU_MEM_FRACTION=0.10 python world_model/scripts/train/train_adapter_vision.py --config world_model/configs/v4/adapter_outcome.json --run "$R" > "$Q/logs/adapter_outcome_v4e.log" 2>&1 || [ ! -f "$R/best.pt" ]; then
    stamp "FAILED: outcome adapter training (logs/adapter_outcome_v4e.log)"; exit 1
fi
stamp "outcome adapter trained: best at step $(python -c "import torch;print(torch.load('$R/best.pt',map_location='cpu',weights_only=False)['step'])"), $(grep val_selection "$R/log.jsonl" | tail -n 1)"
python -c "
import json; c = json.load(open('world_model/configs/v4/online.json')); c['actor']['adapter'] = '$R/best.pt'
json.dump(c, open('$I/online_adapter_outcome.json', 'w'), indent=1)"
if WM_GPU_MEM_FRACTION=0.10 python $S/check_imagination_creep.py --config "$I/online_adapter_outcome.json" --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --out "$I/creep_10k_outcome.json" > "$Q/logs/check4e_creep_outcome.log" 2>&1; then
    stamp "hand creep in stuck states, outcome adapter (imagined mm along the axis | real | called inserted so far % | decided goes in %): $(python -c "
import json; r = json.load(open('$I/creep_10k_outcome.json'))
print({k: (v['imagined_displacement_mm_median'][0], v['real_displacement_mm_median'][0], v['imagined_called_inserted_so_far_pct'], v['decided_goes_in_pct']) for k, v in r['by_decision'].items()})")"
else
    stamp "FAILED: creep check with the outcome adapter (logs/check4e_creep_outcome.log)"
fi
if WM_GPU_MEM_FRACTION=0.06 python $S/check_adapter_decisions.py --config world_model/configs/v4/adapter_outcome.json --adapters "$Q/runs/adapter_vision_v4e/best.pt" "$R/best.pt" --out "$Q/runs/checks_v4/explore/adapter_decisions_outcome.json" > "$Q/logs/check_adapter_decisions_outcome.log" 2>&1; then
    stamp "adapter decisions with the true outcome given (AUC advanced vs held | mean predicted fraction when really held): $(python -c "
import json; r = json.load(open('$Q/runs/checks_v4/explore/adapter_decisions_outcome.json'))
print({a: {b: (v[b]['auc_advanced_vs_held'], v[b]['mean_predicted_fraction_held']) for b in ('lag<0.3mm', 'lag0.3-1mm')} for a, v in r.items() if isinstance(v, dict) and 'lag<0.3mm' in v})")"
else
    stamp "FAILED: check_adapter_decisions with the outcome adapter (logs/check_adapter_decisions_outcome.log)"
fi
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python $S/check_imagination4.py compare --config "$I/online_adapter_outcome.json" --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --out "$I/report_10k_adapter_outcome.json" --imagined 4 --batch 64 > "$Q/logs/check4e_compare_10k_adapter_outcome.log" 2>&1; then
    stamp "imagination vs simulator (10k, outcome adapter): $(python -c "
import json; r = json.load(open('$I/report_10k_adapter_outcome.json')); print(r['3_closed_loop']['all'])")"
else
    stamp "FAILED: imagination vs simulator with the outcome adapter (logs/check4e_compare_10k_adapter_outcome.log)"
fi
