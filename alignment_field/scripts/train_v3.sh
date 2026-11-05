#!/bin/bash
# v3: train on generated shapes G00-G44, select on G45-G49 (shape-held-out
# validation), A-D are logged every 20 epochs for the record only.
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs runs_v3
TR=$(python -c "print(' '.join(f'G{i:02d}' for i in range(45)))")
nohup python -m paf.train --data "data/train_gen/*.npz" --train-ports $TR --val-ports G45 G46 G47 G48 G49 \
  --monitor-data "data/train/*.npz" --monitor-every 20 --epochs "${1:-40}" --out runs_v3/main \
  > logs/train_v3.log 2>&1 &
echo "v3 training pid $!"
