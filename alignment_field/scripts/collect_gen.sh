#!/bin/bash
# v3 training data: 8 grasps on each of the 50 generated shapes G00..G49
# (A-D are never used for training in v3). Resumable.
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs
PORTS=$(python -c "print(' '.join(f'G{i:02d}' for i in range(50)))")
nohup nice -n 10 python scripts/collect.py --out data/train_gen --ports $PORTS --per-port 8 \
  --seed0 200000 --workers "${1:-15}" >> logs/collect_gen.log 2>&1 &
echo "collector pid $!"
