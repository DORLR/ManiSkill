#!/bin/bash
# v3g training phase without the latent gate. The gate failed only on pitch (pose-head pitch R^2 -2.02;
# docs/2026-11-08_v3_runs.md); the user decided on 2026-11-09 to train the world models and the latent
# policy on the v3g latents anyway while the pitch question is investigated.
#   in parallel: CDiT-S 128x2 (cap 0.19), CDiT-B 64x4 (cap 0.16), latent FM (cap 0.07)
#   tmux new-session -d -s qingxin-v3g-train "bash world_model/scripts/pipelines/pipeline_v3g_train.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v3g.log
S=world_model/scripts
C=world_model/configs/v3
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (memory cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/$name.log" 2>&1; then stamp "done $name"; return 0
    else stamp "FAILED $name (see logs/$name.log)"; return 1; fi
}
stamp "training phase started without the gate (user decision 2026-11-09; pitch unresolved)"
( run train_wm_v3g_s 0.19 python $S/train/train_wm.py --config $C/wm_s.json --run $Q/runs/wm_v3g_s \
  && run check_wm_v3g_s 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3g_latents \
         --vae $Q/runs/vae_v3g/best.pt --wm $Q/runs/wm_v3g_s/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3g --tag s ) &
( run train_wm_v3g_b 0.16 python $S/train/train_wm.py --config $C/wm_b.json --run $Q/runs/wm_v3g_b \
  && run check_wm_v3g_b 0.06 python $S/analysis/check_world_model.py --latents $Q/data/wm_v3g_latents \
         --vae $Q/runs/vae_v3g/best.pt --wm $Q/runs/wm_v3g_b/best.pt --adapter $Q/runs/adapter_v3/best.pt \
         --out $Q/runs/checks_v3g --tag b ) &
( run train_policy_v3g 0.07 python $S/train/train_policy.py --config $C/policy.json --run $Q/runs/policy_v3g \
  && run eval_policy_v3g 0.03 python $S/eval/eval_policy.py --vae $Q/runs/vae_v3g/best.pt \
         --policy $Q/runs/policy_v3g/best.pt --split ood_val --out $Q/runs/eval_policy_v3g_ood --workers 3 ) &
wait
stamp "v3g training phase finished"
