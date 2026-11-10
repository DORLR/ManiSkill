#!/bin/bash
# Residual v2 in the world model (fence on lateral hand travel, 2 mrad bound, drift penalty, 4 s episodes);
# judged with the fixed success criterion (depth <= 1 mm, <= 2 mm from the hole axis, <= 2 deg).
#   training (cap 0.20) -> runs/residual_v3g_v2
#   iteration 200: closed loop on G46 + G95 (seed 0); final: closed loop, 3 seeds
#   tmux new-session -d -s qingxin-residual2 "bash world_model/scripts/pipelines/pipeline_residual_v2.sh"
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_residual_v2.log
S=world_model/scripts
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (memory cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/$name.log" 2>&1; then stamp "done $name"; return 0
    else stamp "FAILED $name (see logs/$name.log)"; return 1; fi
}
EV="--vae $Q/runs/vae_v3g/best.pt --policy $Q/runs/policy_v3g/best.pt --split ood_val"
RUN=$Q/runs/residual_v3g_v2
( run train_residual_v2 0.20 python $S/train/train_residual.py --config world_model/configs/v3/residual_v2.json --run $RUN ) &
P=$!
( until [ -f $RUN/residual_it200.pt ] || ! kill -0 $P 2>/dev/null; do sleep 60; done
  [ -f $RUN/residual_it200.pt ] && run eval_residual_v2_it200_dev 0.03 python $S/eval/eval_policy.py $EV \
      --residual $RUN/residual_it200.pt --out $Q/runs/eval_residual_v2_it200_dev --workers 3 --only G46 G95 ) &
wait $P || { stamp "STOP: training failed"; wait; exit 1; }
for s in 0 1 2; do
  ( run eval_residual_v2_seed$s 0.03 python $S/eval/eval_policy.py $EV --residual $RUN/last.pt --seed $s \
        --out $Q/runs/evalfix_residual_v2_seed$s --workers 3 ) &
done
wait
stamp "residual v2 pipeline finished"
