#!/bin/bash
# The onset probe on one or more checkpoints, each started only in a gap between checks and after a measurement.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=0
for TAG in "$@"; do
    while pgrep -u $(id -u) -f "scripts/analysis/check_wm6" > /dev/null; do sleep 15; done
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 20000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] || { echo "$(date +%H:%M) not started $TAG: GPU0 free $f MiB, RAM $a GB, pressure $p %"; break; }
    echo "$(date +%H:%M) start probe $TAG: GPU0 free $f MiB, RAM available $a GB, pressure $p %"
    WM_GPU_MEM_FRACTION=0.08 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/scripts/analysis/probe_onset.py --ckpt $Q/runs/checks_v6/$TAG.pt --workers 4 --out $Q/runs/checks_v6/probe_onset_$TAG.json > $Q/logs/v6_probe_onset_$TAG.log 2>&1 || echo "FAILED $TAG"
    tail -n 6 $Q/logs/v6_probe_onset_$TAG.log
done
echo "$(date +%H:%M) done"
