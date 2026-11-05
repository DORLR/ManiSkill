#!/bin/bash
# Closed-loop evaluation of all folds (resumable), then the summary table.
# usage: bash scripts/eval.sh [workers]
source /root/autodl-tmp/port_align_field/scripts/env.sh
mkdir -p logs results
nohup sh -c "nice -n 10 python scripts/evaluate.py --runs runs --out results/eval --workers ${1:-15} --roles seen \
  && nice -n 10 python scripts/evaluate.py --runs runs --out results/eval --workers ${1:-15} --roles unseen \
  && python scripts/analyze.py --eval results/eval --runs runs --out results/summary.json" \
  > logs/eval.log 2>&1 &
echo "eval pid $!"
