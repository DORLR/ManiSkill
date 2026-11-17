#!/bin/bash
# 2026-11-16: both agreed-model runs landed on GPU 1 (it was empty at 07:00); with other users back, GPU 1 reached
# 94.9 of 97.9 GB at 11:11 while GPU 0 had 68 GB free. Move the larger run (branch_wm_v1, 27.6 GB) off GPU 1: right
# after its checkpoint at step 10000, stop it through the pipeline own stop-and-resume path (the marker makes
# tdmpc_set3.sh wait 15 min and restart it from last.pt on the GPU with the most free memory).
source /mnt/storage2/qingxin/env.sh
R=/mnt/storage3/qingxin/runs/branch_wm_v1
until grep -q "\"step\": 10000," $R/log.jsonl; do sleep 20; done
sleep 60
if [ $(( $(date +%s) - $(stat -c %Y $R/last.pt) )) -lt 300 ]; then
    touch $Q/logs/qingxin-wm-train.stopped_by_watchdog
    systemctl --user stop qingxin-wm-train.scope
    echo "[$(date +%m-%d\ %H:%M)] branch_wm_v1 stopped by me after its step-10000 checkpoint, to move it off the full GPU 1" >> $Q/logs/pipeline_tdmpc.log
else
    echo "[$(date +%m-%d\ %H:%M)] move of branch_wm_v1 not done: no fresh checkpoint" >> $Q/logs/pipeline_tdmpc.log
fi
