#!/bin/bash
# Smoke runs of the multi-GPU learner, the online collector and its tests, and the PPO baseline, each small, each
# through lib320.sh's budget (so the GPU-list claims and guards are exercised too). Output under $Q/runs/smoke320_<tag>/.
#   smoke320_online.sh <tag> <wm ckpt for the agent> <collector GPU> <two learner GPUs, e.g. 6,7>
# Run from a frozen copy of the code. Every step logs "smoke <name>: code <rc>" to the pipeline log; nothing here is
# a result.
source "$(dirname "$(readlink -f "$0")")/lib320.sh"
CODE=$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)
tag=${1:?tag}; ck=${2:?checkpoint}; cg=${3:?collector GPU}; lg=${4:?two learner GPUs}
q320_gpus_ok "$cg" && q320_gpus_ok "$lg" || { echo "bad GPUs"; exit 1; }
cd "$CODE" || exit 1
export PYTHONPATH=$CODE/world_model/src${PYTHONPATH:+:$PYTHONPATH}
S=$Q/runs/smoke320_$tag; mkdir -p "$S"; cp "$ck" "$S/model.pt" || exit 1
step() { local name=$1; shift; q320_run_when_fits "qx320-smoke-$name" "$@"; local rc=$?; q320_log "smoke $name: code $rc"; echo "$name $rc" >> "$S/codes"; }
ATT=world_model/scripts/online/run_wm320_attempts.py
TR=world_model/scripts/train/train_wm320.py
DDP=(python -m torch.distributed.run --standalone --nproc_per_node=2)
frac=$(q320_fraction 18800 "$lg") || exit 1

# 1. collector: 3 training grasps x 2 starts, 3 workers on one GPU (v_low 12: a value between the jammed and inserted medians measured on the sample)
step collect 40 "$cg" 9000 test 1 "$S/collect.log" python $ATT --grasps "$Q/data/wm_v1" --ckpt "$S/model.pt" --mode collect --split train \
  --pool "$S/pool" --round 1 --grasps-n 3 --starts 2 --workers 3 --gpus "$cg" --torch-mib 1500 --v-low 12
# 2. tests of the agent on 2 selection grasps x 1 start
for m in act wm; do
  step "test-$m" 40 "$cg" 9000 test 1 "$S/test_$m.log" python $ATT --grasps "$Q/data/wm_v1" --ckpt "$S/model.pt" --mode $m --split select \
    --out "$S/select_$m" --grasps-n 2 --starts 1 --workers 2 --gpus "$cg" --torch-mib 1500 --v-low 12
done
# 3. learner on two GPUs: 260 steps (one 250-step log line with the all-reduced losses, a validation at 260), then 20 more
#    steps resumed with the pool of step 1 (a 'data' record with the online windows)
step ddp 140 "$lg" 20000 train "$frac" "$S/ddp.log" "${DDP[@]}" $TR --data "$Q/data/wm_v1_fast" --out "$S/learner" --stem standard \
  --shift-px 4 --shaping 0.3 --wm-batch 16 --bc-batch 32 --accum 2 --workers-wm 8 --workers-bc 8 --steps 260 --eval-every 260 --keep-every 260 --save-every 130
rm -f "$S/learner/DONE.json"
step ddp-online 140 "$lg" 20000 train "$frac" "$S/ddp_online.log" "${DDP[@]}" $TR --data "$Q/data/wm_v1_fast" --out "$S/learner" --stem standard \
  --shift-px 4 --shaping 0.3 --wm-batch 16 --bc-batch 32 --accum 2 --workers-wm 8 --workers-bc 8 --steps 280 --eval-every 1000 --keep-every 1000 \
  --save-every 10 --online "$S/pool" --resume
# 4. PPO: 4 workers, 3,072 steps; then its test on 2 selection grasps x 1 start
step ppo 40 - 0 cpu 1 "$S/ppo.log" python world_model/scripts/train/train_ppo_state.py --grasps "$Q/data/wm_v1" --data "$Q/data/wm_v1_fast" \
  --out "$S/ppo" --n-envs 4 --total-steps 3072 --n-steps 256 --batch 512 --shaping 0.3 --save-every 1024
step ppo-test 20 - 0 cpu 1 "$S/ppo_test.log" python world_model/scripts/eval/eval_ppo_state.py --grasps "$Q/data/wm_v1" --split select \
  --model "$S/ppo/final.zip" --vecnorm "$S/ppo/final_vecnormalize.pkl" --scale "$S/ppo/action_scale.json" --out "$S/ppo_select" \
  --grasps-n 2 --starts 1 --workers 2
q320_log "smoke $tag: all steps ran"
