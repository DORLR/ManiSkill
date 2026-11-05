#!/bin/bash
# Hands-off chain so neither CPU nor GPU waits for a human:
#   1. wait for the v3 collection (generated shapes) to finish
#   2. GPU: v3 training            | CPU: RGB-only closed-loop evaluation (v1 test grasps)
#   3. CPU: v3 closed-loop evaluation once v3 training is done
# usage: nohup bash scripts/pipeline_v3.sh > logs/pipeline_v3.log 2>&1 &
source /root/autodl-tmp/port_align_field/scripts/env.sh
wait_for() { while [ "$(ps -eo cmd | grep -c "$1")" -gt 0 ]; do sleep 30; done; }
stamp() { echo "[$(date +%T)] $*"; }

wait_for "[s]cripts/collect.py --out data/train_gen"
stamp "collection finished: $(ls data/train_gen/*.npz | wc -l) shards"
bash scripts/train_v3.sh

wait_for "[p]af.train --held-out [A-D] --rgb-only"
stamp "RGB-only training finished; starting RGB-only evaluation"
nice -n 10 python scripts/evaluate.py --runs runs --rgb-runs runs_rgb --out results/eval_rgb \
  --methods field_rgb --workers 15 > logs/eval_rgb.log 2>&1
python scripts/analyze.py --eval results/eval_rgb --runs runs_rgb --out results/summary_rgb.json --n-perm 100 \
  >> logs/eval_rgb.log 2>&1
stamp "RGB-only evaluation finished"

wait_for "[p]af.train --data data/train_gen"
stamp "v3 training finished; starting v3 evaluation"
nice -n 10 python scripts/evaluate.py --single-run runs_v3/main --out results/eval_v3 \
  --methods field oracle none --n-unseen 8 --workers 15 > logs/eval_v3.log 2>&1
python scripts/analyze.py --eval results/eval_v3 --single-run runs_v3/main --out results/summary_v3.json \
  --n-perm 100 >> logs/eval_v3.log 2>&1
stamp "v3 evaluation finished"
