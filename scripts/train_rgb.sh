#!/bin/bash
# Ablation: RGB only (no depth), same leave-one-shape-out folds and data as v1.
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs runs_rgb
for f in A B C D; do
  nohup python -m paf.train --held-out $f --rgb-only --epochs 60 --out runs_rgb/fold_$f > logs/train_rgb_$f.log 2>&1 &
  echo "fold $f pid $!"
done
