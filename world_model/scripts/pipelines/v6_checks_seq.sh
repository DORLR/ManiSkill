#!/bin/bash
# v6: all checks of the joint runs in ONE queue, one process at a time (after the incident of 2026-11-13: seven
# sessions started at once pushed the user service over systemd-oomd's memory-pressure limit).
#   usage: v6_checks_seq.sh <run>:<session>:<step>:<checks> ...     checks: letters of  a approach, p push, r replay,
#          h direct prediction, i image use;  e.g.  joint_v6d:qingxin-v6d-joint:5000:ap
# Before every process: GPU 0 must have > 20 GB free, the machine > 60 GB of RAM available, and the user service's
# memory pressure must be < 10 %; otherwise wait (up to 30 min), then skip that check with a line in the log.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=0
LOG=$Q/logs/pipeline_v4.log
CK=$Q/runs/checks_v6
A=world_model/scripts/analysis
LIM="systemd-run --user --scope -q -p MemoryMax=24G"
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
state() { echo "GPU0 free $(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )) MiB, RAM available $(free -g | awk 'NR==2{print $7}') GB, pressure $(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/') %"; }
guard() {
    for n in $(seq 1 30); do
        local f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) ))
        local p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
        local a=$(free -g | awk 'NR==2{print $7}')
        [ "$f" -gt 20000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] && { echo "$(date +%H:%M) start $1: $(state)"; return 0; }
        sleep 60
    done
    stamp "v6 checks: $1 SKIPPED, the machine stayed busy for 30 min ($(state))"; return 1
}
step_log() { grep '"loss"' $Q/runs/$1/log.jsonl 2>/dev/null | tail -n 1 | sed 's/.*"step": \([0-9]*\).*/\1/'; }
for item in "$@"; do
    IFS=: read RUN SESSION K WHAT <<< "$item"
    G=$CK/${RUN}_$K.pt
    if [ ! -f $G ]; then
        while [ "$(step_log $RUN)" -lt $K ] 2>/dev/null; do
            tmux has-session -t $SESSION 2>/dev/null || break
            sleep 60
        done
        sleep 30
        S=$(python -c "import torch; print(torch.load('$Q/runs/$RUN/current.pt', map_location='cpu', weights_only=False)['step'])" 2>/dev/null || echo 0)
        [ "$S" -ge $K ] || { stamp "v6 checks: $RUN did not reach step $K (current.pt is at $S)"; continue; }
        cp $Q/runs/$RUN/current.pt $G
        echo "$(date +%H:%M) copied $RUN current.pt (step $S) as ${RUN}_$K.pt"
    fi
    T=${RUN}_$K
    if [[ $WHAT == *a* ]] && guard "approach $T"; then
        GPU=0 $LIM bash $Q/code/ManiSkill/world_model/scripts/pipelines/v6_approach.sh $G $T
    fi
    if [[ $WHAT == *i* ]] && guard "image use $T"; then
        if WM_GPU_MEM_FRACTION=0.08 $LIM nice -n 10 python $A/check_wm6_image_use.py --ckpt $G --batches 12 --out $CK/image_use_$T.json > $Q/logs/v6_image_use_$T.log 2>&1; then
            stamp "v6 image use ($T; one prediction at +0.4 s on held-out windows; onset = pressed bit 0, the hand advanced before, the command leads; predicted advance mm where the hand really stops | really advances, with the recorded images / another sample's images / no images): $(python -c "
import json; r = json.load(open('$CK/image_use_$T.json'))
print({k: {'n': (v['onset']['really_stops'], v['onset']['really_advances']), 'true': v['onset']['true_advance_mm'], **{a: v[a]['onset_predicted_advance_mm_stops_advances'] for a in ('images', 'swapped', 'no_images')}, 'error_all_mm': {a: v[a]['advance_error_mm'] for a in ('images', 'swapped', 'no_images')}} for k, v in r.items() if isinstance(v, dict)})")"
        else stamp "FAILED: v6 image use ($T; logs/v6_image_use_$T.log)"; fi
    fi
    if [[ $WHAT == *p* ]] && guard "push $T"; then
        if WM_GPU_MEM_FRACTION=0.08 $LIM nice -n 10 python $A/check_wm6_push.py --ckpt $G --data $Q/data/wm_v1c --out $CK/push_$T.json > $Q/logs/v6_push_$T.log 2>&1; then
            stamp "v6 push ($RUN step $K; 128 start states where the real hand is blocked, script pushes 0.6 mm ahead of the imagined hand; median mm, p90 mm, says pressed %): $(python -c "
import json; r = json.load(open('$CK/push_$T.json'))
print({k: (v['imagined_hand_moved_mm']['median'], v['imagined_hand_moved_mm']['p90'], v.get('says_pressed_pct')) for k, v in r.items() if k.startswith('+')}, '| per decision', r['advance_per_decision_mm'])")"
        else stamp "FAILED: v6 push ($T; logs/v6_push_$T.log)"; fi
    fi
    if [[ $WHAT == *r* ]] && guard "replay $T"; then
        if WM_GPU_MEM_FRACTION=0.08 $LIM nice -n 10 python $A/check_wm6_replay.py --ckpt $G --data $Q/data/wm_v1c --steps 30 --out $CK/replay_${T}_wm_v1c.json > $Q/logs/v6_replay_${T}_wm_v1c.log 2>&1; then
            stamp "v6 replay ($RUN step $K, wm_v1c, held-out): $(python -c "
import json; r = json.load(open('$CK/replay_${T}_wm_v1c.json'))
f = lambda v: {'hand_err_mm': v['hand_along_axis_imagined_minus_real_mm']['median_abs'], 'inserted_real_called_auc': (v['inserted']['really_pct'], v['inserted']['called_pct'], v['inserted']['auc']), 'read_out_imagined_real_blind': (v['read_out_across_axis_mm_frames_within_3mm']['imagined_features'], v['read_out_across_axis_mm_frames_within_3mm']['real_features'], v['read_out_across_axis_mm_frames_within_3mm']['blind'])}
print({k: f(v) for k, v in r.items() if k.startswith('+')}, '| held decisions', r['held_decisions'])")"
        else stamp "FAILED: v6 replay ($T; logs/v6_replay_${T}_wm_v1c.log)"; fi
    fi
    if [[ $WHAT == *h* ]] && guard "direct prediction $T"; then
        if WM_GPU_MEM_FRACTION=0.10 $LIM nice -n 10 python $A/check_wm6_horizons.py --ckpt $G --shifts 2 10 20 --batches 4 --samplers ddpm20 --out $CK/horizons_$T.json > $Q/logs/v6_horizons_$T.log 2>&1; then
            stamp "v6 direct prediction ($RUN step $K; feature mse predicted | nothing changes; read-out mm predicted | real | nothing changes): $(python -c "
import json; r = json.load(open('$CK/horizons_$T.json'))
print({k: (v['feature_mse_ddpm20'], v['feature_mse_nothing_changes'], v['read_out_ddpm20_mm'], v['read_out_real_mm'], v['read_out_nothing_changes_mm']) for k, v in r.items() if isinstance(v, dict)})")"
        else stamp "FAILED: v6 direct prediction ($T; logs/v6_horizons_$T.log)"; fi
    fi
done
stamp "v6 checks queue finished"
