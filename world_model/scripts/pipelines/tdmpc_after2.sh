#!/bin/bash
# 2026-11-16 14:55 (replaces the waiting tdmpc_after.sh): the stem-stride-4 model is finished, so its checks and its
# closed-loop test start NOW, next to the stride-2 training that still runs (7 simulator workers, so that our GPU
# memory stays under 55 GB); the stride-2 model follows when its training and report are done.
# Original header: after the trainings of tdmpc_set3.sh ("tdmpc pipeline finished" in logs/pipeline_tdmpc.log):
#   1. the four offline checks + the early-stop calibration (scripts/analysis/diagnose_branch_wm.py) of both agreed
#      models                                                                      -> <run>/diagnosis.json
#   2. the closed-loop test in the simulator (scripts/eval/eval_branch_closed_loop.py; 150 held-out grasps x 2 fresh
#      starts x 6 modes) of the stem-stride-4 model, and of the stride-2 model if its final report is at least as
#      good (v0 AUC at the board face on ood_val)                                  -> <run>/closed_loop_ood_val/summary.json
# Same safety as tdmpc_set3.sh: GPU memory cap per process, RAM cap per job, niced, a job starts only with room,
# our GPU memory in total under 55 GB, a watchdog stops our jobs if the shared user service comes under pressure.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_tdmpc.log
S=world_model/scripts
X=/mnt/storage3/qingxin/data/wm_v1b
R=/mnt/storage3/qingxin/runs
PRESS=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure
CK="--vae $R/vae_v3g/best.pt --policy $R/policy_v3g/best.pt"
WORKERS=${WORKERS:-8}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
pressure() { grep full $PRESS | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1; }
ours_gpu_mb() {
    local sum=0 pid mem
    while IFS=', ' read -r pid mem; do
        [ -n "$pid" ] && [[ "$(readlink /proc/$pid/exe 2>/dev/null)" == $Q/* ]] && sum=$((sum + mem))
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    echo $sum
}
start_ok() {      # $1 = GPU memory the job may take [MiB]; prints the GPU to use; waits up to 6 h for room
    local need=$1 line g f p a o
    for i in $(seq 1 72); do
        line=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
        g=${line%%,*}; f=${line##*, }; p=$(pressure); a=$(free -g | awk 'NR==2{print $7}'); o=$(ours_gpu_mb)
        if [ "$f" -gt $((need + 10000)) ] && [ "$p" -lt 5 ] && [ "$a" -gt 80 ] && [ $((o + need)) -lt 55000 ]; then
            echo "$g"; stamp "room: GPU $g free $f MiB, ours $o MiB, RAM $a GB, pressure $p %"; return 0
        fi
        stamp "waiting for room ($need MiB): best GPU $g free $f MiB, ours $o MiB, RAM $a GB, pressure $p %"; sleep 300
    done
    return 1
}
job() {           # name, GPU memory the whole job may take [MiB], per-process fraction, RAM cap, command...
    local name=$1 need=$2 frac=$3 ram=$4 g; shift 4
    for attempt in 1 2 3 4; do
        g=$(start_ok $need) || { stamp "FAILED: $name not started, no room for 6 h"; return 1; }
        systemctl --user reset-failed qingxin-after.scope 2>/dev/null
        stamp "start $name (attempt $attempt, GPU $g)"
        if WM_CAM_SCALE=2 WM_GPU_MEM_FRACTION=$frac CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q --unit qingxin-after \
                -p MemoryMax=$ram nice -n 10 "$@" >> "$Q/logs/$name.log" 2>&1; then
            stamp "done $name"; return 0
        fi
        if [ -f "$Q/logs/qingxin-after.stopped_by_watchdog" ]; then
            rm -f "$Q/logs/qingxin-after.stopped_by_watchdog"; stamp "$name was stopped by the watchdog; resumes when calm"; sleep 900
        else
            stamp "FAILED $name (logs/$name.log)"; return 1
        fi
    done
    return 1
}
watchdog() {
    while true; do
        if [ "$(pressure)" -ge 20 ] && systemctl --user is-active -q qingxin-after.scope; then
            touch "$Q/logs/qingxin-after.stopped_by_watchdog"
            systemctl --user stop qingxin-after.scope; stamp "watchdog: pressure $(pressure) %, stopped qingxin-after"; sleep 600
        fi
        sleep 20
    done
}

watchdog &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT
auc() { python3 -c "import json,sys; print(json.load(open('$R/$1/report_final.json'))['ood_val']['face+0']['v0']['auc_inserted'])" 2>/dev/null || echo 0; }
diagnose() {
    job diagnosis_$1 12000 0.12 40G python $S/analysis/diagnose_branch_wm.py --ckpt $R/$1/last.pt --data $X \
        --splits dev,ood_val --calibrate-abort dev --workers 8 --out $R/$1/diagnosis.json
}
closed_loop() {   # run, workers, GPU fraction per worker, MiB counted per worker
    local run=$1 w=$2 frac=$3 per=$4 thr
    thr=$(python3 -c "import json; print(json.load(open('$R/$run/diagnosis.json'))['abort_rule']['threshold'])" 2>/dev/null)
    [ -n "$thr" ] || { stamp "FAILED: no early-stop threshold in $R/$run/diagnosis.json; closed loop of $run skipped"; return 1; }
    job closed_loop_$run $((w * per)) $frac 48G python $S/eval/eval_branch_closed_loop.py --grasps $Q/data/wm_v1 --split ood_val \
        --ckpt $R/$run/last.pt --abort-below $thr --out $R/$run/closed_loop_ood_val --grasps-n 150 --starts 2 --workers $w $CK
}

stamp "after-pipeline 2: stride-4 model now, stride-2 model after its training"
diagnose branch_wm_v1s4 && closed_loop branch_wm_v1s4 7 0.035 3500

while tmux has-session -t qingxin-tdmpc 2>/dev/null; do sleep 120; done
if [ -f $R/branch_wm_v1/report_final.json ]; then
    diagnose branch_wm_v1
    if python3 -c "import sys; sys.exit(0 if float('$(auc branch_wm_v1)') >= float('$(auc branch_wm_v1s4)') > 0 else 1)"; then
        stamp "stride 2 is at least as good offline ($(auc branch_wm_v1) vs $(auc branch_wm_v1s4)): its closed loop follows"
        closed_loop branch_wm_v1 8 0.05 4900
    else
        stamp "stride 2 is not better offline ($(auc branch_wm_v1) vs $(auc branch_wm_v1s4)): no closed loop for it"
    fi
else
    stamp "branch_wm_v1 has no final report; skipped"
fi
stamp "after-pipeline 2 finished"
