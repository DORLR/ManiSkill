#!/bin/bash
# v7: RelNet over-fits the 43 training shapes (pitch bias +1.0 deg on unseen G46-48, +1.5 deg on A-D,
# ~0 on training shapes). Add 100 generated shapes G50..G149 (views only; IL unchanged), retrain
# RelNet on 143 shapes from the v6 weights, compare offline, then closed loop (field and fused field).
cd /root/autodl-tmp/port_align_field
source scripts/env.sh >/dev/null 2>&1
export PATH=/root/autodl-tmp/paf_venv/bin:$PATH
stamp() { echo "[$(date +%H:%M:%S)] $*" >> logs/pipeline_v7.log; }
stamp "start"
EXTRA=$(python -c "from paf.ports_ext import extra_specs; print(' '.join(extra_specs()))" 2>/dev/null)
python scripts/collect.py --out data/v7_views --ports $EXTRA --per-port 2 --seed0 1500000 --store-raw \
  --mix handoff=25,approach=40,wide_v6=15 --workers ${WORKERS:-6} > logs/collect_v7_views.log 2>&1
stamp "views: $(ls data/v7_views/*.npz | wc -l) grasps, $(grep -c "'expert_ok': True" logs/collect_v7_views.log) ok"
python scripts/pack_raw.py --data "data/v7_views/*.npz" --out data/packed_v7 > logs/pack_v7.log 2>&1 || { stamp "STOP: pack failed"; exit 1; }
stamp "packed: $(tail -1 logs/pack_v7.log)"
python -m paf.train_rel --packed data/packed_v6 data/packed_v7 --val-ports G46 G47 G48 --monitor-packed data/packed_v6_abcd \
  --epochs 12 --workers 4 --init runs_v6/rel/model.pt --out runs_v7/rel_143 > logs/train_v7_rel_143.log 2>&1 || { stamp "STOP: RelNet failed"; exit 1; }
stamp "RelNet (143 shapes) trained"
for M in runs_v6/rel runs_v7/rel_143; do
  for P in "data/packed_v6 --ports G46 G47 G48" "data/packed_v6_abcd"; do
    echo "== $M on $P" >> results/v7_offline.txt
    python scripts/analyze_rel.py --model $M/model.pt --kinds 3 4 --packed $P 2>&1 | grep -E "^[A-DG]|per axis|estimate =|ALL" >> results/v7_offline.txt
  done
done
stamp "offline comparison written (results/v7_offline.txt)"
python scripts/fit_sid_field.py --handoff-box --rel runs_v7/rel_143 --out runs_v7/fieldrel_143 > logs/field_v7.log 2>&1 || { stamp "STOP: field failed"; exit 1; }
while [ ! -f logs/eval_v6_fuse.done ]; do sleep 30; done
python scripts/eval_v4.py --policies fm=runs_v6/policy bc=runs_v6/policy_bc --fieldrels fm=runs_v7/fieldrel_143 bc=runs_v7/fieldrel_143 \
  --kinds field field_fuse --out results/eval_v7 --workers 10 > logs/eval_v7.log 2>&1
stamp "v7 evaluation finished"
