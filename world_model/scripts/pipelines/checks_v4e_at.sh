#!/bin/bash
# Repeat the two world-model checks of checks_v4e_10k.sh at a later training step (argument: the step), with the
# branch episodes already recorded there.
#   tmux new-session -d -s qingxin-checks-30k "bash world_model/scripts/pipelines/checks_v4e_at.sh 30000"
set -uo pipefail
STEP=$1
TAG=$((STEP / 1000))k
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts/analysis
C=world_model/configs/v4/online.json
I=$Q/runs/checks_v4/imagination_v4e
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
until python -c "import torch,sys; sys.exit(0 if torch.load('$Q/runs/wm_v4e/current.pt',map_location='cpu',weights_only=False)['step']>=$STEP else 1)" 2>/dev/null; do sleep 120; done
while tmux has-session -t qingxin-checks-10k 2>/dev/null; do sleep 60; done          # the recording and the 10k checks first
ls "$I"/episodes/*.npz > /dev/null 2>&1 || { stamp "FAILED: no branch episodes for the $TAG checks"; exit 1; }
cp "$Q/runs/wm_v4e/current.pt" "$I/wm_$TAG.pt"
cp "$Q/runs/success_v4e/best.pt" "$I/success.pt"
stamp "world model at step $(python -c "import torch;print(torch.load('$I/wm_$TAG.pt',map_location='cpu',weights_only=False)['step'])"): checks started"
if WM_GPU_MEM_FRACTION=0.10 nice -n 5 python $S/check_wm_on_exploration.py --config $C --wm "$I/wm_$TAG.pt" --success "$I/success.pt" --data "$Q/data/wm_v1x" --latents "$Q/data/wm_v4ex_latents" --out "$Q/runs/checks_v4/explore/wm_on_exploration_$TAG.json" > "$Q/logs/check4e_explore_$TAG.log" 2>&1; then
    stamp "world model on held-out exploration ($TAG), outcome AUC: $(python -c "
import json; r = json.load(open('$Q/runs/checks_v4/explore/wm_on_exploration_$TAG.json'))
print({k: r[k]['auc'] for k in ('all', 'push_only', 'with_maneuver')})")"
else
    stamp "FAILED: check_wm_on_exploration (logs/check4e_explore_$TAG.log)"
fi
if WM_GPU_MEM_FRACTION=0.12 nice -n 5 python $S/check_imagination4.py compare --config $C --episodes "$I/episodes" --wm "$I/wm_$TAG.pt" --success "$I/success.pt" --out "$I/report_$TAG.json" --imagined 4 --batch 64 > "$Q/logs/check4e_compare_$TAG.log" 2>&1; then
    stamp "imagination vs simulator ($TAG): $(python -c "
import json; r = json.load(open('$I/report_$TAG.json')); print(r['3_closed_loop']['all'], '| classifier', r['1_classifier'])")"
else
    stamp "FAILED: imagination vs simulator (logs/check4e_compare_$TAG.log)"
fi
