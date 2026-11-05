#!/bin/bash
# Nested leave-one-shape-out model selection (added 2026-11-05, see docs/plan.md
# change log). For test shape X, train on two of the other shapes and validate
# on the third, for two variants (plain, --reg). The test shape X is never
# loaded. Two waves of four runs (GPU memory is shared with the evaluation).
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs runs_nested
declare -A VAL=( [A]=B [B]=C [C]=D [D]=A )
for variant in plain reg; do
  flag=""; [ $variant = reg ] && flag="--reg"
  for x in A B C D; do
    v=${VAL[$x]}
    python -m paf.train --held-out $x --val-port $v $flag --epochs 40 \
      --out runs_nested/test${x}_val${v}_$variant > logs/nested_test${x}_val${v}_$variant.log 2>&1 &
  done
  wait
done
