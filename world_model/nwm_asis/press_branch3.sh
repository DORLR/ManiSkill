#!/bin/bash
# 2026-11-14 12:15: press_branch2.sh again after the resume fix in run_plug.py (the first start died in torch.load).
# check `pgrep -f plug_cdit_b.yaml` matched the tmux server, whose command line is the first session's command. Here the
# check is by pid only. Warms the page cache, checks GPU 0, starts plug_cdit_b_press, then its evaluation watcher.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_v4.log
X=/mnt/storage3/qingxin/data/wm_v1p
NEW=$Q/runs/nwm_asis/plug_cdit_b_press
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
kill -0 1482776 2>/dev/null && { stamp "press branch 3: baseline pid 1482776 still alive -- not started"; exit 1; }
[ -e $NEW ] && { stamp "press branch 3: $NEW exists -- not started"; exit 1; }
stamp "press branch 3: retry after the resume fix in run_plug.py (torch.load map_location int)"
find $X ${X}_ee -name '*.npz' -print0 | xargs -0 cat > /dev/null
for i in $(seq 1 36); do
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) ))
    p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
    a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 45000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 80 ] && break
    echo "waiting: GPU0 free $f MiB, RAM $a GB, pressure $p %"; sleep 300
done
[ "$f" -gt 45000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 80 ] || { stamp "press branch 3: GPU0 / RAM busy for 3 h -- not started"; exit 1; }
stamp "press branch 3: starting plug_cdit_b_press on GPU 0 (free $f MiB, RAM $a GB, pressure $p %)"
tmux new-session -d -s qingxin-nwm-press "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.4 systemd-run --user --scope -q -p MemoryMax=60G nice -n 10 python world_model/nwm_asis/run_plug.py --config world_model/nwm_asis/plug_cdit_b_press.yaml --ckpt-every 5000 --eval-every 100000000 --bfloat16 1 --epochs 300 --torch-compile 0 > $Q/logs/nwm_press_train.log 2>&1"
sleep 900
L=$(sed 's/\x1b\[[0-9;]*m//g' $NEW/log.txt 2>/dev/null | grep "step=" | tail -1)
grep -q "Loading model from" $Q/logs/nwm_press_train.log && [ -n "$L" ] || { stamp "FAILED: press branch training did not start properly (logs/nwm_press_train.log)"; exit 1; }
stamp "press branch running: $(grep -m1 'Dataset: wm_v1p (train)' $Q/logs/nwm_press_train.log) | $(grep -m1 'Dataset contains' $NEW/log.txt | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-120) | $L"
tmux new-session -d -s qingxin-nwm-press-eval "bash $Q/code/ManiSkill/world_model/nwm_asis/eval_press.sh 65000 70000 75000 80000 90000 100000 2>&1 | tee $Q/logs/nwm_press_eval.log"
