#!/bin/bash
# FM policy baseline: wait for the latents, a short trial, then the full run. GPU 1 only.
#   tmux new-session -d -s qingxin-policy "bash world_model/scripts/pipeline_policy_v1.sh"
set -euo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
export WM_GPU_MEM_FRACTION=0.1   # shared GPU: this process can never take more than ~10 GB
LOG=$Q/logs/pipeline_policy_v1.log
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }

stamp "waiting for the latents ($Q/data/wm_v1_latents/stats.json)"
until [ -f "$Q/data/wm_v1_latents/stats.json" ]; do sleep 60; done

python - <<'EOF'
import json
c = json.load(open("world_model/configs/policy_v1.json"))
c.update(steps=300, eval_every=150, log_every=50, eval_batches=4)
json.dump(c, open("/mnt/storage2/qingxin/runs/policy_trial_config.json", "w"), indent=1)
EOF
rm -rf "$Q/runs/policy_trial"
stamp "policy trial (300 steps)"
nice -n 10 python world_model/scripts/train_policy.py --config "$Q/runs/policy_trial_config.json" \
    --run "$Q/runs/policy_trial" > "$Q/logs/train_policy_trial.log" 2>&1
test -f "$Q/runs/policy_trial/best.pt" || { stamp "STOP: policy trial wrote no checkpoint"; exit 1; }
stamp "policy trial ok: $(grep val_ "$Q/logs/train_policy_trial.log" | tail -1 | cut -c1-160)"

stamp "policy training"
nice -n 10 python world_model/scripts/train_policy.py --config world_model/configs/policy_v1.json \
    --run "$Q/runs/policy_v1" > "$Q/logs/train_policy_v1.log" 2>&1
stamp "policy training finished"
