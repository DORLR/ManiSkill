#!/bin/bash
# 2026-11-12, decided with the user: the first exploration set (wm_v1x) ends 0.4-1.6 s after the plug is blocked, so
# the data have neither long pressed states nor what any action does from them; closed-loop policies sit there for
# many seconds, and the imagined hand crept forward there (16 mm in 16 s; real 0.05 mm). This collects the contact
# set (collect/explore.py --contact): the same policy-independent exploration with 4-16 s more at the board.
# collection (train 6 per grasp; ood_val 1 per grasp, for checks only) -> index -> commanded hand poses -> crops
# -> crop coverage -> checks of the set. Stops at the first failure. Nothing is trained here.
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
export CUDA_VISIBLE_DEVICES=1
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
X=$Q/data/wm_v1c
CK=$Q/runs/checks_v5/contact
W=${WORKERS:-12}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1 cap=$2; shift 2
    stamp "start $name (cap $cap)"
    if WM_GPU_MEM_FRACTION=$cap nice -n 10 "$@" > "$Q/logs/contact_$name.log" 2>&1; then stamp "done $name"
    else stamp "FAILED $name (logs/contact_$name.log) -- contact-set pipeline stopped"; exit 1; fi
}
mkdir -p $CK
stamp "contact set: collecting train split (6 per grasp, $W workers) -> $X"
run collect_train 0.02 python $S/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split train --per-grasp 6 --workers $W --contact
stamp "contact set: collecting ood_val split (1 per grasp, for checks only)"
run collect_val 0.02 python $S/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split ood_val --per-grasp 1 --workers $W --contact
stamp "contact set: collection finished: $(find $X -name 't*.npz' | wc -l) trajectories, $(du -sh $X | cut -f1)"
run index 0.02 python $S/data/build_index.py $X --workers 12
run ee 0.02 python $S/data/compute_ee.py $X --out ${X}_ee --workers 8
run crops 0.08 python $S/data/place_crops.py --data $X --unet $Q/runs/unet_v4/best.pt --out ${X}_crops --workers 12
run crop_gate 0.02 python $S/analysis/check_crop_coverage.py --data $X --crops ${X}_crops --geometry $Q/data/wm_v1_labelgeom --out $CK/crops --n 800
run checks 0.02 python $S/analysis/check_contact_set.py --data $X --ee ${X}_ee --out $CK/report.json
python - <<PY >> $LOG
import json
r = json.load(open("$CK/report.json"))
print("[contact set] %s | phase %s | decisions by lead %s | %s | GATE %s" % (r["A_collection"]["outcome"], {k: r["B_contact_phase"][k] for k in ("trajectories_with_phase", "seconds_median", "seconds_total_hours", "hand_excursion_across_axis_mm", "phase_steps_by_true_offset_pct", "inserted_at_phase_start_to_end")}, r["C_decisions_by_lead_along_axis"], r["D_no_truth_in_commands"], r["gate"]))
PY
stamp "contact set: ready ($X, ${X}_ee, ${X}_crops)"
