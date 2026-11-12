#!/bin/bash
# v6: the replay check (scripts/analysis/check_wm6_replay.py) on intermediate checkpoints of the joint training.
#   usage: v6_checks.sh <run, e.g. joint_v6> <tmux session of the training> [steps ...]
# At each step: held-out contact-set and exploration trajectories replayed in imagination (script commands only, nothing
# real fed back), compared with the simulator: hand position, creep where the real hand is held, "inserted", read-out.
set -uo pipefail
RUN=$1; SESSION=$2; shift 2
STEPS=${*:-"10000 20000 30000"}
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$Q/logs/pipeline_v4.log
CK=$Q/runs/checks_v6
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
step_now() { python -c "import torch; print(torch.load('$Q/runs/$RUN/current.pt', map_location='cpu', weights_only=False)['step'])" 2>/dev/null || echo 0; }
mkdir -p $CK
for K in $STEPS; do
    while [ "$(step_now)" -lt $K ]; do
        tmux has-session -t $SESSION 2>/dev/null || { stamp "v6 checks: the training $RUN ended before step $K"; exit 1; }
        sleep 180
    done
    G=$CK/${RUN}_$K.pt
    cp $Q/runs/$RUN/current.pt $G
    for D in wm_v1c wm_v1x; do
        ST=30; [ $D = wm_v1x ] && ST=8
        if WM_GPU_MEM_FRACTION=0.08 nice -n 10 python world_model/scripts/analysis/check_wm6_replay.py --ckpt $G --data $Q/data/$D --steps $ST --out $CK/replay_${RUN}_${K}_$D.json > $Q/logs/v6_replay_${RUN}_${K}_$D.log 2>&1; then
            stamp "v6 replay ($RUN step $K, $D, held-out): $(python -c "
import json; r = json.load(open('$CK/replay_${RUN}_${K}_$D.json'))
f = lambda v: {'hand_err_mm': v['hand_along_axis_imagined_minus_real_mm']['median_abs'], 'creep_where_real_held_mm': (v['real_hand_moved_under_0.3mm']['imagined_moved_mm_mean'], v['real_hand_moved_under_0.3mm']['trajectories']), 'inserted_real_called_auc': (v['inserted']['really_pct'], v['inserted']['called_pct'], v['inserted']['auc']), 'read_out_imagined_real_blind': (v['read_out_across_axis_mm_frames_within_3mm']['imagined_features'], v['read_out_across_axis_mm_frames_within_3mm']['real_features'], v['read_out_across_axis_mm_frames_within_3mm']['blind'])}
print({k: f(v) for k, v in r.items() if k.startswith('+')}, '| held decisions', r['held_decisions'])")"
        else
            stamp "FAILED: v6 replay ($RUN step $K, $D; logs/v6_replay_${RUN}_${K}_$D.log)"
        fi
    done
done
