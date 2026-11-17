#!/bin/bash
# 2026-11-14: the press set (collect/press.py; why: its docstring), train split 4 per grasp, ood_val 1 per grasp (checks
# only), index, smoke-style check. Data on /mnt/storage3 (storage2 is nearly full). The requested command poses are
# written by the collector itself into <out>_ee (no compute_ee).
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
X=/mnt/storage3/qingxin/data/wm_v1p
W=${WORKERS:-8}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
guard() {   # GPU 1 (renderer) free > 10 GB, user-service memory pressure < 10 %, RAM available > 60 GB
    local f p a
    for i in $(seq 1 36); do
        f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) ))
        p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
        a=$(free -g | awk 'NR==2{print $7}')
        [ "$f" -gt 10000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] && { echo "guard ok: GPU1 free $f MiB, RAM $a GB, pressure $p %"; return 0; }
        echo "waiting: GPU1 free $f MiB, RAM $a GB, pressure $p %"; sleep 300
    done
    return 1
}
run() {
    local name=$1; shift
    guard || { stamp "FAILED: press set, $name not started (machine busy for 3 h)"; exit 1; }
    stamp "start press_$name"
    if CUDA_VISIBLE_DEVICES=1 systemd-run --user --scope -q -p MemoryMax=40G nice -n 10 "$@" > "$Q/logs/press_$name.log" 2>&1; then stamp "done press_$name"
    else stamp "FAILED press_$name (logs/press_$name.log) -- press-set pipeline stopped"; exit 1; fi
}
run collect_train python $S/data/collect_press.py --grasps $Q/data/wm_v1 --out $X --split train --per-grasp 4 --workers $W --seed 8200000
run collect_val python $S/data/collect_press.py --grasps $Q/data/wm_v1 --out $X --split ood_val --per-grasp 1 --workers $W --seed 8300000
stamp "press set: collection finished: $(find $X -name 't*_press.npz' | wc -l) trajectories, $(du -sh $X | cut -f1)"
run index python $S/data/build_index.py $X --workers 8
mkdir -p /mnt/storage3/qingxin/runs/press_set_check
nice python $Q/cache/tmp/check_press.py $X /mnt/storage3/qingxin/runs/press_set_check > $Q/logs/press_set_check.log 2>&1
stamp "press set checked: $(head -4 $Q/logs/press_set_check.log | tr '\n' ' ')"
