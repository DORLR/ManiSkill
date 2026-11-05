#!/bin/bash
# Train the four leave-one-shape-out folds concurrently on the one GPU.
# usage: bash scripts/train.sh [epochs]
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs runs
for f in A B C D; do
  nohup python -m paf.train --held-out $f --out runs/fold_$f --epochs "${1:-60}" > logs/train_$f.log 2>&1 &
  echo "fold $f pid $!"
done
