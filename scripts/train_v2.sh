#!/bin/bash
# v2 training: rotating two-train / one-check shapes each epoch
# for each leave-one-shape-out fold.
# The held-out shape is evaluated every 20 epochs for the record only; the
# final epoch is the model. usage: bash scripts/train_v2.sh [epochs]
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs runs_v2
for f in A B C D; do
  nohup python -m paf.train --held-out $f --rotate --monitor-every 20 --epochs "${1:-40}" \
    --out runs_v2/fold_$f > logs/train_v2_$f.log 2>&1 &
  echo "fold $f pid $!"
done
