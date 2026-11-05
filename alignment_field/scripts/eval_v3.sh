#!/bin/bash
# v3 closed loop: one model, test shapes A-D (all never trained on), 6 grasp
# conditions x 8 grasps x 4 OOD starts; methods field / oracle / none.
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs results
nohup sh -c "nice -n 10 python scripts/evaluate.py --single-run runs_v3/main --out results/eval_v3 \
  --methods field oracle none --n-unseen 8 --workers ${1:-15} \
  && python scripts/analyze.py --eval results/eval_v3 --single-run runs_v3/main --out results/summary_v3.json" \
  > logs/eval_v3.log 2>&1 &
echo "v3 eval pid $!"
