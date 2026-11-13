#!/bin/bash
# v6: world-model checks on intermediate checkpoints of a joint training (v6_checks.sh + the closed-loop push).
#   usage: v6_checks2.sh <run> <tmux session of the training> [steps ...]
# replay   recorded script commands replayed in imagination, nothing real fed back (check_wm6_replay.py)
# push     a script that keeps pushing 0.6 mm ahead of the IMAGINED hand, in states where the real hand is pressed on the
#          board (check_wm6_push.py): the closed loop in which v4e and the first v6 run crept 14-16 mm in 16 s
set -uo pipefail
RUN=$1; SESSION=$2; shift 2
STEPS=${*:-"5000 10000 20000"}
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
LOG=$Q/logs/pipeline_v4.log
CK=$Q/runs/checks_v6
A=world_model/scripts/analysis
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
step_now() { python -c "import torch; print(torch.load('$Q/runs/$RUN/current.pt', map_location='cpu', weights_only=False)['step'])" 2>/dev/null || echo 0; }
mkdir -p $CK
for K in $STEPS; do
    while [ "$(step_now)" -lt $K ]; do
        tmux has-session -t $SESSION 2>/dev/null || { stamp "v6 checks: the training $RUN ended before step $K"; exit 1; }
        sleep 120
    done
    G=$CK/${RUN}_$K.pt
    cp $Q/runs/$RUN/current.pt $G
    if WM_GPU_MEM_FRACTION=0.08 nice -n 10 python $A/check_wm6_push.py --ckpt $G --data $Q/data/wm_v1c --out $CK/push_${RUN}_$K.json > $Q/logs/v6_push_${RUN}_$K.log 2>&1; then
        stamp "v6 push ($RUN step $K; 128 pressed states, script pushes 0.6 mm ahead of the imagined hand; the real hand moves < 0.3 mm): $(python -c "
import json; r = json.load(open('$CK/push_${RUN}_$K.json'))
print({k: (v['imagined_hand_moved_mm']['median'], v['imagined_hand_moved_mm']['p90'], v['says_inserted_pct']) for k, v in r.items() if k.startswith('+')}, '| per decision', r['advance_per_decision_mm'])")"
    else
        stamp "FAILED: v6 push ($RUN step $K; logs/v6_push_${RUN}_$K.log)"
    fi
    for D in wm_v1c wm_v1x; do
        ST=30; [ $D = wm_v1x ] && ST=8
        if WM_GPU_MEM_FRACTION=0.08 nice -n 10 python $A/check_wm6_replay.py --ckpt $G --data $Q/data/$D --steps $ST --out $CK/replay_${RUN}_${K}_$D.json > $Q/logs/v6_replay_${RUN}_${K}_$D.log 2>&1; then
            stamp "v6 replay ($RUN step $K, $D, held-out): $(python -c "
import json; r = json.load(open('$CK/replay_${RUN}_${K}_$D.json'))
f = lambda v: {'hand_err_mm': v['hand_along_axis_imagined_minus_real_mm']['median_abs'], 'where_real_held_mm': (v['real_hand_moved_under_0.3mm']['imagined_moved_mm_mean'], v['real_hand_moved_under_0.3mm']['trajectories']), 'inserted_real_called_auc': (v['inserted']['really_pct'], v['inserted']['called_pct'], v['inserted']['auc']), 'read_out_imagined_real_blind': (v['read_out_across_axis_mm_frames_within_3mm']['imagined_features'], v['read_out_across_axis_mm_frames_within_3mm']['real_features'], v['read_out_across_axis_mm_frames_within_3mm']['blind'])}
print({k: f(v) for k, v in r.items() if k.startswith('+')}, '| held decisions', r['held_decisions'])")"
        else
            stamp "FAILED: v6 replay ($RUN step $K, $D; logs/v6_replay_${RUN}_${K}_$D.log)"
        fi
    done
done
