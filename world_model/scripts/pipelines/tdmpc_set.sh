#!/bin/bash
# 2026-11-16: after the branch-set collection (branch_set.sh, tmux qingxin-branch) has finished:
#   1. 64 x 64 copies of the pictures (CPU)                                   -> /mnt/storage3/qingxin/data/wm_v1b_c64
#   2. TD-MPC2 unchanged, the control run (wm/tdmpc/original.py)              -> runs/branch_tdmpc2_orig
#   3. the agreed model (wm/tdmpc/model.py), 640 x 640 pictures + depth       -> runs/branch_wm_v1
#   4. the offline report of both on dev and ood_val (wm/tdmpc/report.py)     -> <run>/report_final.json
# Safety on the shared machine: every job has a GPU memory cap (WM_GPU_MEM_FRACTION) and a RAM cap (MemoryMax) and
# runs niced; a job starts only when the machine has room (start_ok); our GPU memory in total stays under 55 GB;
# a watchdog stops our training jobs if the memory pressure of the shared user service rises (they resume from
# their last checkpoint once it is calm again). Nothing here touches anybody else's processes.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_tdmpc.log
S=world_model/scripts
X=/mnt/storage3/qingxin/data/wm_v1b
C=${X}_c64
R=/mnt/storage3/qingxin/runs
PRESS=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure
WM_STEPS=${WM_STEPS:-20000}
ORIG_STEPS=${ORIG_STEPS:-100000}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
pressure() { grep full $PRESS | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1; }
ours_gpu_mb() {   # GPU memory of the processes that run from our own environment
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
train() {         # name, unit, GPU memory fraction, RAM cap, command... ; restarts (resume) after a watchdog stop
    local name=$1 unit=$2 frac=$3 ram=$4; shift 4
    local need=$(awk -v f=$frac 'BEGIN{printf "%d", f * 97887}') g
    for attempt in 1 2 3 4 5 6; do
        g=$(start_ok $need) || { stamp "FAILED: $name not started, no room for 6 h"; return 1; }
        systemctl --user reset-failed $unit.scope 2>/dev/null
        stamp "start $name (attempt $attempt, GPU $g, cap $need MiB, RAM $ram)"
        if WM_GPU_MEM_FRACTION=$frac CUDA_VISIBLE_DEVICES=$g systemd-run --user --scope -q --unit $unit -p MemoryMax=$ram \
                nice -n 10 "$@" >> "$Q/logs/$name.log" 2>&1; then
            stamp "done $name"; return 0
        fi
        if [ -f "$Q/logs/$unit.stopped_by_watchdog" ]; then
            rm -f "$Q/logs/$unit.stopped_by_watchdog"; stamp "$name was stopped by the watchdog; resumes when calm"; sleep 900
        else
            stamp "FAILED $name (logs/$name.log)"; return 1
        fi
    done
    stamp "FAILED $name: stopped by the watchdog 6 times"; return 1
}
watchdog() {      # stop our training scopes if the shared user service comes under memory pressure
    while true; do
        if [ "$(pressure)" -ge 20 ]; then
            for u in qingxin-tdmpc2-train qingxin-wm-train; do
                if systemctl --user is-active -q $u.scope; then
                    touch "$Q/logs/$u.stopped_by_watchdog"
                    systemctl --user stop $u.scope; stamp "watchdog: pressure $(pressure) %, stopped $u"
                fi
            done
            sleep 600
        fi
        sleep 20
    done
}

stamp "tdmpc pipeline waits for the collection"
while tmux has-session -t qingxin-branch 2>/dev/null || tmux has-session -t qingxin-wmpilot 2>/dev/null; do sleep 120; done
grep -q "done branch_collect_train" $Q/logs/pipeline_branch.log || { stamp "FAILED: the collection did not finish; nothing trained"; exit 1; }

stamp "start cache64"
if CUDA_VISIBLE_DEVICES= systemd-run --user --scope -q -p MemoryMax=16G nice -n 12 python $S/data/cache_branch64.py \
        --data $X --out $C --workers 8 > $Q/logs/branch_cache64.log 2>&1; then
    stamp "done cache64: $(tail -1 $Q/logs/branch_cache64.log)"
else stamp "FAILED cache64 (logs/branch_cache64.log); the control run is skipped"; fi

watchdog &
WD=$!
trap 'kill $WD 2>/dev/null' EXIT

( train branch_tdmpc2_orig qingxin-tdmpc2-train 0.10 48G python $S/train/train_branch_tdmpc2.py --data $X --cache $C \
      --out $R/branch_tdmpc2_orig --steps $ORIG_STEPS --no-compile \
  && train branch_tdmpc2_orig_report qingxin-tdmpc2-train 0.10 48G python $S/analysis/eval_branch_value.py --kind tdmpc2 \
      --ckpt $R/branch_tdmpc2_orig/last.pt --data $X --cache $C --out $R/branch_tdmpc2_orig/report_final.json ) &
ORIG=$!
sleep 600      # one start at a time
train branch_wm_v1 qingxin-wm-train 0.30 64G python $S/train/train_branch_wm.py --data $X --out $R/branch_wm_v1 \
      --steps $WM_STEPS --files 16 --per-file 2 --workers 10 --dev-every 2000 --dev-branches 160 \
  && train branch_wm_v1_report qingxin-wm-train 0.30 64G python $S/analysis/eval_branch_value.py --kind wm \
      --ckpt $R/branch_wm_v1/last.pt --data $X --workers 8 --out $R/branch_wm_v1/report_final.json
wait $ORIG
stamp "tdmpc pipeline finished"
