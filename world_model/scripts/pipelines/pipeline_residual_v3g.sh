#!/bin/bash
# Residual PPO on the frozen latent FM, trained in the world model's imagination, judged in simulation.
#   training (cap 0.20)                         runs/residual_v3g
#   mid-run check: when residual_it200.pt exists, closed loop on 2 held-out shapes (G46, G95), the same
#   episodes as the FM evaluation, so the two can be compared before the run ends
#   final: full closed loop, ood_val, v3 protocol (542 episodes)
#   tmux new-session -d -s qingxin-residual "bash world_model/scripts/pipelines/pipeline_residual_v3g.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_residual_v3g.log
S=world_model/scripts
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (memory cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/$name.log" 2>&1; then stamp "done $name"; return 0
    else stamp "FAILED $name (see logs/$name.log)"; return 1; fi
}
EV="--vae $Q/runs/vae_v3g/best.pt --policy $Q/runs/policy_v3g/best.pt --split ood_val"
( run train_residual_v3g 0.20 python $S/train/train_residual.py --config world_model/configs/v3/residual.json \
      --run $Q/runs/residual_v3g ) &
P=$!
( until [ -f $Q/runs/residual_v3g/residual_it200.pt ] || ! kill -0 $P 2>/dev/null; do sleep 60; done
  [ -f $Q/runs/residual_v3g/residual_it200.pt ] && run eval_residual_v3g_it200_dev 0.03 python $S/eval/eval_policy.py $EV \
      --residual $Q/runs/residual_v3g/residual_it200.pt --out $Q/runs/eval_residual_v3g_it200_dev --workers 3 --only G46 G95 ) &
wait $P || { stamp "STOP: training failed"; wait; exit 1; }
run eval_residual_v3g 0.03 python $S/eval/eval_policy.py $EV --residual $Q/runs/residual_v3g/last.pt \
    --out $Q/runs/eval_residual_v3g_ood --workers 6
wait
stamp "residual pipeline finished"
