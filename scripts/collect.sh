#!/bin/bash
# Launch training-data collection in the background (resumable).
# usage: bash scripts/collect.sh [workers]
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs
nohup nice -n 10 python scripts/collect.py --out data/train --per-port 60 --workers "${1:-15}" >> logs/collect_train.log 2>&1 &
echo "collector pid $!"
