#!/bin/bash
# eval_press.sh (2026-11-14): eval_both3.sh for the continued run plug_cdit_b_press (from plug_cdit_b at 60 000 steps + the
# press set). Its first save is at 65 000: the step of the n-th save is 60 000 + 5 000 n. Same evaluation, GPU 1, same limits.
# Memory on GPU 1: torch per-process fraction 0.12 = 11.7 GB hard cap of our allocator (+ ~0.5 GB CUDA context);
# measured footprint of one evaluation 9.0 GB. Started only if GPU 1 has > 20 GB free (need + 8 GB margin) and
# the user service's memory pressure < 10 % and RAM available > 60 GB; otherwise it waits (re-checks every 5 min,
# up to 3 h) instead of starting blind, then skips with a log line.
#   usage: eval_press.sh <steps ...>
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
RUN=$Q/runs/nwm_asis/plug_cdit_b_press
START=60000
TRAIN_TMUX=qingxin-nwm-press
KEEP=/mnt/storage3/qingxin/runs/nwm_asis_press_eval_checkpoints
LOG=$Q/logs/pipeline_v4.log
GPU=1
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
saved() { sed 's/\x1b\[[0-9;]*m//g' $RUN/log.txt 2>/dev/null | grep "Saved checkpoint" | wc -l; }
mkdir -p $KEEP
for K in "$@"; do
    while [ $(( START + $(saved) * 5000 )) -lt $K ]; do
        tmux has-session -t $TRAIN_TMUX 2>/dev/null || { stamp "NWM + press set: the training ended before step $K"; exit 1; }
        sleep 30
    done
    C=$KEEP/ckpt_$K.pth.tar
    cp $RUN/checkpoints/latest.pth.tar $C.part && mv $C.part $C
    S=$(python -c "import torch; print(torch.load('$C', map_location='cpu', weights_only=False)['train_steps'])" 2>/dev/null || echo -1)
    [ "$S" -eq "$K" ] || { stamp "NWM + press set: the copy for step $K holds step $S -- not evaluated"; continue; }
    for WTS in model ema; do
        ok=0
        for try in $(seq 1 36); do
            f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
            if [ "$f" -gt 20000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ]; then ok=1; break; fi
            echo "$(date +%H:%M) waiting: GPU$GPU free $f MiB, RAM $a GB, pressure $p %"; sleep 300
        done
        [ $ok -eq 1 ] || { stamp "NWM + press set: $WTS evaluation at $K skipped after 3 h of waiting (GPU$GPU free $f MiB, RAM $a GB, pressure $p %)"; continue; }
        echo "$(date +%H:%M) evaluating $WTS weights at $K on GPU$GPU: free $f MiB, RAM available $a GB, pressure $p %"
        O=$Q/runs/nwm_asis/press_eval_${K}_$WTS
        if CUDA_VISIBLE_DEVICES=$GPU WM_GPU_MEM_FRACTION=0.12 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/nwm_asis/eval_plug.py --weights $WTS --ckpt $C --out $O > $Q/logs/nwm_press_eval_${K}_$WTS.log 2>&1; then
            stamp "NWM + press set, step $K, $WTS weights (GPU $GPU; pictures in runs/nwm_asis/press_eval_${K}_$WTS; compare with plug_cdit_b at 60000): $(python -c "
import json; r = json.load(open('$O/metrics.json'))
t = {k: (v['lpips_predicted'], v['lpips_last_frame_again']) for k, v in r['time'].items()}
ro = {k: (v['lpips_predicted'], v['lpips_first_frame_again']) for k, v in r['rollout_recorded_commands'].items() if k.endswith('+8s')}
pu = {k: {a: v[a]['picture_change_0_to_8s'] for a in v if a[:2] in ('A_', 'B_', 'C_', 'D_')} for k, v in r['push_arms'].items()}
au = {k: (v['action_use']['lpips_true_vs_shuffled_prediction'], v['action_use']['lpips_true_vs_no_motion_prediction'], v['lpips_vae_floor']) for k, v in r['time'].items() if 'action_use' in v}
print('trained steps', r['train_steps'], '| LPIPS predicted vs last-frame-again, one prediction:', t, '| rollout at 8 s:', ro, '| action use (true vs shuffled, true vs no motion, vae floor):', au, '| push arms, picture change 0-8 s:', pu)")"
        else
            stamp "FAILED: NWM + press set, $WTS evaluation at $K on GPU $GPU (logs/nwm_press_eval_${K}_$WTS.log)"
        fi
    done
done
