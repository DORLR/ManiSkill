#!/bin/bash
# v5: one hand-over region for everything (the IL's familiar start: t_x -2 mm, +-0.5 mm, +-0.5 deg).
# 1 collect RelNet views in the hand-over region and on the way to it (training + validation shapes)
# 2 pack with the v4 views, 3 fine-tune RelNet, 4 field with the hand-over region as target, 5 evaluate.
cd /root/autodl-tmp/port_align_field
source scripts/env.sh >/dev/null 2>&1
export PATH=/root/autodl-tmp/paf_venv/bin:$PATH
stamp() { echo "[$(date +%H:%M:%S)] $*" >> logs/pipeline_handoff.log; }
PORTS=$(python -c "from paf.ports_ext import generated_specs; print(' '.join(p for p in generated_specs() if p not in ('G19','G42','G45','G49')))" 2>/dev/null)
python scripts/collect.py --out data/v5_handoff --ports $PORTS --per-port 3 --seed0 1100000 --store-raw \
  --mix handoff=30,approach=50 --workers 11 > logs/collect_v5_handoff.log 2>&1
stamp "views collected: $(ls data/v5_handoff/*.npz | wc -l) grasps"
python scripts/pack_raw.py --data "data/v4_train/*.npz" "data/v5_handoff/*.npz" --out data/packed_v5 > logs/pack_v5.log 2>&1 || { stamp "STOP: pack failed"; exit 1; }
stamp "packed: $(tail -1 logs/pack_v5.log)"
python -m paf.train_rel --packed data/packed_v5 --val-ports G46 G47 G48 --monitor-packed data/packed_v4_abcd \
  --init runs_v4/rel/model.pt --epochs 6 --out runs_v5/rel > logs/train_v5_rel.log 2>&1 || { stamp "STOP: RelNet fine-tune failed"; exit 1; }
stamp "RelNet fine-tuned"
python scripts/fit_sid_field.py --handoff-box --rel runs_v5/rel --out runs_v5/fieldrel_handoff > logs/field_v5.log 2>&1 || { stamp "STOP: field failed"; exit 1; }
stamp "field fitted"
python scripts/eval_v4.py --policies fm=runs_v4/policy bc=runs_v4/policy_bc --fieldrels fm=runs_v5/fieldrel_handoff bc=runs_v5/fieldrel_handoff \
  --kinds field oracle --out results/eval_v5_handoff --workers 11 > logs/eval_v5_handoff.log 2>&1
python scripts/summarize_v4.py --eval results/eval_v4_il results/eval_v5_handoff --out results/summary_v5_handoff.json > results/summary_v5_handoff.txt 2>&1
stamp "v5 evaluation finished"
