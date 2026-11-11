#!/bin/bash
# 2026-11-11: checks of the world model v4e at ~10k steps (result wanted by 21:00).
#  1. when the FM v4e is trained: record its branch episodes on held-out shapes (the 2026-11-10 recording used the
#     old autoencoder and FM and cannot be reused). The FM is not used to train the world model; its rollouts are
#     the test of whether imagination agrees with the simulator.
#  2. when the world model has passed step 10000:
#     (a) does it foresee the outcome of the executed actions on held-out contact exploration
#     (b) imagination vs simulator on the FM's own rollouts
#   tmux new-session -d -s qingxin-checks-10k "bash world_model/scripts/pipelines/checks_v4e_10k.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts/analysis
C=world_model/configs/v4/online.json
I=$Q/runs/checks_v4/imagination_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
mkdir -p "$I"
until tmux has-session -t qingxin-v4e-policy 2>/dev/null; do sleep 30; done
while tmux has-session -t qingxin-v4e-policy 2>/dev/null; do sleep 60; done
[ -f "$Q/runs/policy_v4e/best.pt" ] || { stamp "FAILED: FM v4e has no checkpoint -- 10k checks stopped"; exit 1; }
stamp "FM v4e trained ($(grep -c val_action_err_mrad "$Q/logs/v4e_policy.log") evaluations, best action error $(grep -o '"val_action_err_mrad": [0-9.]*' "$Q/logs/v4e_policy.log" | cut -d' ' -f2 | sort -n | head -1) mrad); recording branch episodes (15 workers)"
if WM_GPU_MEM_FRACTION=0.03 nice -n 10 python $S/check_imagination4.py record --config $C --out "$I/episodes" --workers 15 > "$Q/logs/check4e_record.log" 2>&1; then
    stamp "branch episodes recorded: $(ls "$I"/episodes/*.npz | wc -l) episodes"
else
    stamp "FAILED: branch recording (logs/check4e_record.log)"; exit 1
fi
until python -c "import torch,sys; sys.exit(0 if torch.load('$Q/runs/wm_v4e/current.pt',map_location='cpu',weights_only=False)['step']>=10000 else 1)" 2>/dev/null; do sleep 120; done
cp "$Q/runs/wm_v4e/current.pt" "$I/wm_10k.pt"
cp "$Q/runs/success_v4e/best.pt" "$I/success.pt"
stamp "world model at step $(python -c "import torch;print(torch.load('$I/wm_10k.pt',map_location='cpu',weights_only=False)['step'])"): checks started"
if WM_GPU_MEM_FRACTION=0.10 nice -n 5 python $S/check_wm_on_exploration.py --config $C --wm "$I/wm_10k.pt" --success "$I/success.pt" --data "$Q/data/wm_v1x" --latents "$Q/data/wm_v4ex_latents" --out "$Q/runs/checks_v4/explore/wm_on_exploration_10k.json" > "$Q/logs/check4e_explore_10k.log" 2>&1; then
    stamp "world model on held-out exploration (10k), outcome AUC: $(python -c "
import json; r = json.load(open('$Q/runs/checks_v4/explore/wm_on_exploration_10k.json'))
print({k: r[k]['auc'] for k in ('all', 'push_only', 'with_maneuver')})")"
else
    stamp "FAILED: check_wm_on_exploration (logs/check4e_explore_10k.log)"
fi
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python $S/check_imagination4.py compare --config $C --episodes "$I/episodes" --wm "$I/wm_10k.pt" --success "$I/success.pt" --out "$I/report_10k.json" --imagined 4 --batch 64 > "$Q/logs/check4e_compare_10k.log" 2>&1; then
    stamp "imagination vs simulator (10k): $(python -c "
import json; r = json.load(open('$I/report_10k.json')); print(r['3_closed_loop']['all'], '| classifier', r['1_classifier'])")"
else
    stamp "FAILED: imagination vs simulator (logs/check4e_compare_10k.log)"
fi
