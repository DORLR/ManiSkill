#!/bin/bash
# 2026-11-16 night: the policy on the world model's ResNet (the user: the policy in use was still the DINOv2 one).
#   1. tokens of the seated attempts of the branch set (frozen FrameNet of runs/branch_wm_v1s4)  -> data/wm_v1b_tok
#   2. flow-matching policy on them (models/policy.py unchanged)                                 -> runs/policy_resnet_v1
#   3. closed loop, policy alone, the 298 starts of the earlier test                             -> <run>/closed_loop_ood_val
# Safety as the other pipelines: GPU memory cap per process, RAM cap per job, niced, start only with room, our GPU
# memory under 55 GB, a watchdog stops the job if the shared user service comes under memory pressure.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_tdmpc.log
S=world_model/scripts
X=/mnt/storage3/qingxin/data/wm_v1b
T=${X}_tok
R=/mnt/storage3/qingxin/runs
WMCK=$R/branch_wm_v1s4/last.pt
P=$R/policy_resnet_v1
PRESS=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
pressure() { grep full $PRESS | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1; }
ours_gpu_mb() {
    local sum=0 pid mem
    while IFS=', ' read -r pid mem; do
        [ -n "$pid" ] && [[ "$(readlink /proc/$pid/exe 2>/dev/null)" == $Q/* ]] && sum=$((sum + mem))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    echo $sum
}
start_ok() {
    local need=$1 line g f p a o
    for i in $(seq 1 36); do
        line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
        g=${line%%,*}; f=${line##*, }; p=$(pressure); a=$(free -g | awk 'NR==2{print $7}'); o=$(ours_gpu_mb)
        if [ "$f" -gt $((need + 10000)) ] && [ "$p" -lt 5 ] && [ "$a" -gt 80 ] && [ $((o + need)) -lt 55000 ]; then
            echo "$g"; stamp "room: GPU $g free $f MiB, ours $o MiB, RAM $a GB, pressure $p %"; return 0
        fi
        stamp "waiting for room ($need MiB): best GPU $g free $f MiB, ours $o MiB, RAM $a GB, pressure $p %"; sleep 300
    done
    return 1
}
job() {           # name, GPU MiB of the whole job, per-process fraction, RAM cap, command...
    local name=$1 need=$2 frac=$3 ram=$4 g; shift 4
    for attempt in 1 2 3; do
        g=$(start_ok $need) || { stamp "FAILED: $name not started, no room for 3 h"; return 1; }
        systemctl --user reset-failed qingxin-policy.scope 2>/dev/null
        stamp "start $name (attempt $attempt, GPU $g)"
        if WM_CAM_SCALE=2 WM_GPU_MEM_FRACTION=$frac CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q --unit qingxin-policy \
                -p MemoryMax=$ram nice -n 10 "$@" >> "$Q/logs/$name.log" 2>&1; then
            stamp "done $name"; return 0
        fi
        if [ -f "$Q/logs/qingxin-policy.stopped_by_watchdog" ]; then
            rm -f "$Q/logs/qingxin-policy.stopped_by_watchdog"; stamp "$name was stopped by the watchdog; resumes when calm"; sleep 900
        else
            stamp "FAILED $name (logs/$name.log)"; return 1
        fi
    done
    return 1
}
watchdog() {
    while true; do
        if [ "$(pressure)" -ge 20 ] && systemctl --user is-active -q qingxin-policy.scope; then
            touch "$Q/logs/qingxin-policy.stopped_by_watchdog"
            systemctl --user stop qingxin-policy.scope; stamp "watchdog: pressure $(pressure) %, stopped qingxin-policy"; sleep 600
        fi
        sleep 20
    done
}
watchdog &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

stamp "policy-on-ResNet pipeline started"
job policy_resnet_tokens 9000 0.08 40G python $S/data/encode_branch_tokens.py --ckpt $WMCK --data $X --out $T --workers 10 || exit 1
stamp "tokens: $(find $T -name '*.npz' | wc -l) attempts, $(du -sh $T | cut -f1)"
job policy_resnet_train 15000 0.15 48G python $S/train/train_branch_policy.py --tokens $T --out $P --steps ${POLICY_STEPS:-40000} || exit 1
job policy_resnet_closed_loop $((8 * 4900)) 0.05 48G python $S/eval/eval_branch_policy.py --grasps $Q/data/wm_v1 --split ood_val \
    --wm $WMCK --policy $P/best.pt --out $P/closed_loop_ood_val --old $R/branch_wm_v1s4/closed_loop_ood_val --workers 8 || exit 1
stamp "policy-on-ResNet pipeline finished: $(python3 -c "import json; r=json.load(open('$P/closed_loop_ood_val/summary.json')); print(r['resnet_policy'], r.get('against_old_policy_same_starts'))" 2>/dev/null | cut -c1-400)"
