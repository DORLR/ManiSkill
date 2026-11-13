#!/bin/bash
# Push check with per-case output and control arms, one process at a time; stops when joint_v6d is close to its next
# checkpoint so that it does not overlap with the queue (v6_checks_seq.sh).
#   usage: push_arms.sh <stop when joint_v6d passes this step> <ckpt tag>:<name>:<extra args> ...
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=0
CK=$Q/runs/checks_v6
STOP=$1; shift
for item in "$@"; do
    IFS=: read TAG NAME EXTRA <<< "$item"
    S=$(grep '"loss"' $Q/runs/joint_v6d/log.jsonl | tail -n 1 | sed 's/.*"step": \([0-9]*\).*/\1/')
    [ "$S" -lt $STOP ] || { echo "$(date +%H:%M) stopped before $TAG $NAME: joint_v6d at $S"; break; }
    while pgrep -u $(id -u) -f "scripts/analysis/check_wm6" > /dev/null; do sleep 20; done
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) )); p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1); a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 20000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 60 ] || { echo "$(date +%H:%M) not started $TAG $NAME: GPU0 free $f MiB, RAM $a GB, pressure $p %"; break; }
    echo "$(date +%H:%M) start $TAG $NAME: GPU0 free $f MiB, RAM available $a GB, pressure $p %"
    WM_GPU_MEM_FRACTION=0.08 systemd-run --user --scope -q -p MemoryMax=24G nice -n 10 python world_model/scripts/analysis/check_wm6_push.py --ckpt $CK/$TAG.pt --data $Q/data/wm_v1c --out $CK/push_${TAG}_$NAME.json $EXTRA > $Q/logs/v6_push_${TAG}_$NAME.log 2>&1 || echo "FAILED $TAG $NAME"
    python - <<PY
import json
r = json.load(open("$CK/push_${TAG}_$NAME.json"))
print("$TAG $NAME", {k: (v["imagined_hand_moved_mm"]["median"], v["imagined_hand_moved_mm"]["p90"], v["says_pressed_pct"]) for k, v in r.items() if k.startswith("+")}, "| per decision", r["advance_per_decision_mm"], "|", r["by_start_bit"])
PY
done
echo "$(date +%H:%M) done"
