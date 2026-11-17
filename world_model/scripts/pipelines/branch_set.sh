#!/bin/bash
# 2026-11-15: the branch set (collect/branch.py; why: its docstring). Train split: 6 starts x 4 branches per grasp
# (499 grasps); ood_val: 2 starts x 4 branches per grasp (503 grasps; for checking only). 640 x 640 pictures
# (WM_CAM_SCALE=2). Data on /mnt/storage3. Then the check (scripts/analysis/check_branch.py) on both splits.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_branch.log
S=world_model/scripts
X=/mnt/storage3/qingxin/data/wm_v1b
W=${WORKERS:-8}
CK="--vae /mnt/storage3/qingxin/runs/vae_v3g/best.pt --policy /mnt/storage3/qingxin/runs/policy_v3g/best.pt"
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
guard() {   # GPU 1 free > 30 GB, user-service memory pressure < 10 %, RAM available > 60 GB, disk > 150 GB
    local f p a d
    for i in $(seq 1 36); do
        f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) ))
        p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
        a=$(free -g | awk 'NR==2{print $7}')
        d=$(df -BG --output=avail /mnt/storage3 | tail -1 | tr -dc 0-9)
        [ "$d" -gt 150 ] || { echo "disk: only $d GB free on storage3"; return 1; }
        [ "$f" -gt 30000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] && { echo "guard ok: GPU1 free $f MiB, RAM $a GB, pressure $p %, disk $d GB"; return 0; }
        echo "waiting: GPU1 free $f MiB, RAM $a GB, pressure $p %"; sleep 300
    done
    return 1
}
run() {
    local name=$1; shift
    guard || { stamp "FAILED: branch set, $name not started (machine busy for 3 h or disk low)"; exit 1; }
    stamp "start branch_$name"
    if WM_CAM_SCALE=2 WM_GPU_MEM_FRACTION=0.03 CUDA_VISIBLE_DEVICES=1 systemd-run --user --scope -q -p MemoryMax=48G nice -n 10 "$@" > "$Q/logs/branch_$name.log" 2>&1; then
        stamp "done branch_$name: $(find $X -name 's*_b*.npz' | wc -l) branches, $(du -sh $X | cut -f1)"
    else stamp "FAILED branch_$name (logs/branch_$name.log) -- branch-set pipeline stopped"; exit 1; fi
}
# held-out grasps first: the retry upper bound on unseen shapes and grasps is wanted before any training
run collect_val python $S/data/collect_branch.py --grasps $Q/data/wm_v1 --out $X --split ood_val --starts 2 --branches 4 --workers $W --seed 8700000 $CK
run collect_train python $S/data/collect_branch.py --grasps $Q/data/wm_v1 --out $X --split train --starts 6 --branches 4 --workers $W --seed 8500000 $CK
mkdir -p /mnt/storage3/qingxin/runs/branch_set_check
systemd-run --user --scope -q -p MemoryMax=16G nice python $S/analysis/check_branch.py $X /mnt/storage3/qingxin/runs/branch_set_check/report.json > $Q/logs/branch_set_check.log 2>&1
stamp "branch set checked (logs/branch_set_check.log)"
