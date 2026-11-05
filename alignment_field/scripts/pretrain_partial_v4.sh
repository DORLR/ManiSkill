#!/bin/bash
# Use the GPU while collection is still running: once 200 training grasps exist,
# train the relational net and the FM policy on that half; the final runs in
# pipeline_v4.sh warm-start from these weights (fewer epochs after collection).
source /root/autodl-tmp/port_align_field/scripts/env.sh
stamp() { echo "[$(date +%T)] $*"; }
while [ "$(ls data/v4_train/*.npz 2>/dev/null | wc -l)" -lt 200 ]; do sleep 60; done
mkdir -p data/v4_partial && for f in $(ls data/v4_train/*.npz | head -200); do ln -sf "$(readlink -f $f)" data/v4_partial/; done
stamp "partial set: $(ls data/v4_partial | wc -l) grasps"
python scripts/pack_raw.py --data "data/v4_partial/*.npz" --out data/packed_v4_partial
python -m paf.train_rel --packed data/packed_v4_partial --val-ports G46 G47 G48 --epochs 10 --workers 3 \
  --out runs_v4/rel_partial > logs/train_v4_rel_partial.log 2>&1 &
python -m paf.fm_policy --data "data/v4_partial/*.npz" --head bc --out runs_v4/policy_bc_partial --epochs 30 \
  > logs/train_v4_policy_bc_partial.log 2>&1 &
python -m paf.fm_policy --data "data/v4_partial/*.npz" --head fm --out runs_v4/policy_partial --epochs 30 \
  > logs/train_v4_policy_partial.log 2>&1
wait
stamp "partial training finished"
