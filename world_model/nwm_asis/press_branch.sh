#!/bin/bash
# 2026-11-14: branch plug_cdit_b at 60 000 steps into plug_cdit_b_press (+ the press set), as agreed with the user.
# Waits for (1) the press set to be indexed, (2) the verified copy of the 60 000-step checkpoint (made by eval_both3.sh);
# then stops the baseline training (our tmux qingxin-nwm; two trainings would exceed our 55 GB), warms the page cache with
# the press set (it lives on a rotating disk), checks GPU 0, starts the continued run with the baseline's exact command
# and limits, and starts its evaluation watcher (eval_press.sh, GPU 1).
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_v4.log
X=/mnt/storage3/qingxin/data/wm_v1p
C60=/mnt/storage3/qingxin/runs/nwm_asis_eval_checkpoints/ckpt_60000.pth.tar
NEW=$Q/runs/nwm_asis/plug_cdit_b_press
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
until grep -q "done press_index" $LOG; do grep -q "FAILED.*press" $LOG && { stamp "press branch: not started (press-set pipeline failed)"; exit 1; }; sleep 60; done
until [ -f $C60 ]; do tmux has-session -t qingxin-nwm || { stamp "press branch: baseline ended before 60000"; exit 1; }; sleep 60; done
S=$(python -c "import torch; print(torch.load('$C60', map_location='cpu', weights_only=False)['train_steps'])" 2>/dev/null || echo -1)
[ "$S" -eq 60000 ] || { stamp "press branch: the 60000 copy holds step $S -- not started"; exit 1; }
[ -e $NEW ] && { stamp "press branch: $NEW exists -- not started"; exit 1; }
P=$(tmux list-panes -t qingxin-nwm -F '#{pane_pid}')
TP=$(pgrep -P $P -f run_plug.py | head -1)          # the pane shell runs python directly (systemd-run --scope execs)
echo "baseline: pane $P, training pid ${TP:-?}"
tmux kill-session -t qingxin-nwm
sleep 60
for pid in $TP; do kill -0 $pid 2>/dev/null && { echo "still alive: $pid, sending TERM"; kill -TERM $pid; sleep 30; }; done
pgrep -f "plug_cdit_b.yaml" > /dev/null && { stamp "press branch: the baseline training is still running after kill -- not started"; exit 1; }
stamp "NWM as it is: baseline plug_cdit_b stopped after its 60000-step checkpoint (branch point; it can be resumed from checkpoints/latest.pth.tar)"
find $X ${X}_ee -name '*.npz' -print0 | xargs -0 cat > /dev/null
for i in $(seq 1 36); do
    f=$(( 97887 - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0) ))
    p=$(grep full /sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/memory.pressure | sed 's/.*avg10=\([0-9.]*\).*/\1/' | cut -d. -f1)
    a=$(free -g | awk 'NR==2{print $7}')
    [ "$f" -gt 45000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 80 ] && break
    echo "waiting: GPU0 free $f MiB, RAM $a GB, pressure $p %"; sleep 300
done
[ "$f" -gt 45000 ] && [ "$p" -lt 10 ] && [ "$a" -gt 80 ] || { stamp "press branch: GPU0 / RAM busy for 3 h -- not started"; exit 1; }
stamp "press branch: starting plug_cdit_b_press on GPU 0 (free $f MiB, RAM $a GB, pressure $p %)"
tmux new-session -d -s qingxin-nwm-press "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=0 WM_GPU_MEM_FRACTION=0.4 systemd-run --user --scope -q -p MemoryMax=60G nice -n 10 python world_model/nwm_asis/run_plug.py --config world_model/nwm_asis/plug_cdit_b_press.yaml --ckpt-every 5000 --eval-every 100000000 --bfloat16 1 --epochs 300 --torch-compile 0 > $Q/logs/nwm_press_train.log 2>&1"
sleep 900
L=$(sed 's/\x1b\[[0-9;]*m//g' $NEW/log.txt 2>/dev/null | grep "step=" | tail -1)
grep -q "Loading model from" $Q/logs/nwm_press_train.log && [ -n "$L" ] || { stamp "FAILED: press branch training did not start properly (logs/nwm_press_train.log)"; exit 1; }
stamp "press branch running: $(grep -m1 'Dataset: wm_v1p (train)' $Q/logs/nwm_press_train.log) | $(grep -m1 'Dataset contains' $NEW/log.txt | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-120) | $L"
tmux new-session -d -s qingxin-nwm-press-eval "bash $Q/code/ManiSkill/world_model/nwm_asis/eval_press.sh 65000 70000 75000 80000 90000 100000 2>&1 | tee $Q/logs/nwm_press_eval.log"
