#!/bin/bash
# wm_v1 after the VAE: wait for the VAE run, encode latents, CDiT trial, CDiT training.
# Runs on GPU 1 only, low priority. Start in its own tmux session:
#   tmux new-session -d -s qingxin-pipeline "bash world_model/scripts/pipeline_v1.sh"
set -euo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v1.log
stamp() { echo "[$(date +%H:%M)] $*" >> "$LOG"; }

stamp "waiting for the VAE run (tmux qingxin-train)"
while tmux has-session -t qingxin-train 2>/dev/null; do sleep 60; done
test -f "$Q/runs/vae_v1/best.pt" || { stamp "STOP: no VAE checkpoint"; exit 1; }
stamp "VAE done: $(grep -c val_ "$Q/logs/train_vae_v1.log") evaluations"

stamp "encoding latents"
nice -n 10 python world_model/scripts/encode_latents.py --vae "$Q/runs/vae_v1/best.pt" \
    --data "$Q/data/wm_v1" --out "$Q/data/wm_v1_latents" > "$Q/logs/encode_v1.log" 2>&1
stamp "latents: $(tail -1 "$Q/logs/encode_v1.log")"

python - <<'EOF'
import json
c = json.load(open("world_model/configs/wm_v1.json"))
c.update(steps=300, eval_every=150, log_every=20, eval_batches=2)
json.dump(c, open("/mnt/storage2/qingxin/runs/wm_trial_config.json", "w"), indent=1)
EOF
rm -rf "$Q/runs/wm_trial"
stamp "CDiT trial (300 steps)"
nice -n 10 python world_model/scripts/train_wm.py --config "$Q/runs/wm_trial_config.json" \
    --run "$Q/runs/wm_trial" > "$Q/logs/train_wm_trial.log" 2>&1
test -f "$Q/runs/wm_trial/last.pt" || { stamp "STOP: CDiT trial wrote no checkpoint"; exit 1; }
stamp "CDiT trial ok: $(grep val_ "$Q/logs/train_wm_trial.log" | tail -1 | cut -c1-160)"

stamp "stopping after the CDiT trial: full CDiT training needs a decision on sharing GPU 1"
