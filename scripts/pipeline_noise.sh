#!/bin/bash
# After pipeline_v3.sh: closed-loop test of the v3 model with strong depth noise
# (paf/noise.py) on the same A-D test grasps; field only (oracle/none do not use vision).
source /root/autodl-tmp/port_align_field/scripts/env.sh
while [ "$(ps -eo cmd | grep -c "[p]ipeline_v3.sh")" -gt 0 ]; do sleep 30; done
echo "[$(date +%T)] starting v3 strong-depth-noise evaluation"
nice -n 10 python scripts/evaluate.py --single-run runs_v3/main --out results/eval_v3_noise --methods field \
  --depth-noise strong --n-unseen 8 --workers 15 > logs/eval_v3_noise.log 2>&1
python scripts/analyze.py --eval results/eval_v3_noise --single-run runs_v3/main \
  --out results/summary_v3_noise.json --n-perm 100 >> logs/eval_v3_noise.log 2>&1
echo "[$(date +%T)] v3 noise evaluation finished"
