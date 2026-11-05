#!/bin/bash
# v4 data (resumable): generated shapes with raw RGB-D sweeps + insertion demos;
# then A-D raw sweeps for offline monitoring only (never trained on).
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs
W=${1:-11}
GEN=$(python -c "print(' '.join(f'G{i:02d}' for i in range(50)))")
nice -n 10 python scripts/collect.py --out data/v4_train --ports $GEN --per-port 8 --seed0 300000 \
  --store-raw --n-sweep 120 --n-push 0 --n-demo 6 --workers $W >> logs/collect_v4_train.log 2>&1
nice -n 10 python scripts/collect.py --out data/v4_abcd --ports A B C D --per-port 8 --seed0 400000 \
  --store-raw --n-sweep 120 --n-push 0 --n-demo 0 --workers $W >> logs/collect_v4_abcd.log 2>&1
echo "[$(date +%T)] v4 collection finished" >> logs/collect_v4_train.log
