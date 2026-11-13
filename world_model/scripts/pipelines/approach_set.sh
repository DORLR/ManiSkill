#!/bin/bash
# 2026-11-12 night: the approach set (collect/explore.py --approach; why: sample_approach's docstring), then a joint run
# with it from scratch next to the running one, so that by morning "more training" (joint_v6c at 20k / 30k) and "this
# data" (joint_v6d at 5k / 10k / 20k) can be compared on the same checks (push, replay, approach).
set -uo pipefail
source /mnt/storage2/qingxin/env.sh
cd "$Q/code/ManiSkill"
LOG=$Q/logs/pipeline_v4.log
S=world_model/scripts
X=$Q/data/wm_v1a
W=${WORKERS:-12}
stamp() { echo "[$(date +%m-%d\ %H:%M)] $*" >> "$LOG"; }
run() {
    local name=$1; shift
    stamp "start approach_$name"
    if nice -n 10 "$@" > "$Q/logs/approach_$name.log" 2>&1; then stamp "done approach_$name"
    else stamp "FAILED approach_$name (logs/approach_$name.log) -- approach-set pipeline stopped"; exit 1; fi
}
free_gb() { df --output=avail -BG /mnt/storage2 | tail -1 | tr -dc 0-9; }
[ "$(free_gb)" -ge 25 ] || { stamp "FAILED: only $(free_gb) GB free on /mnt/storage2 -- approach set not collected"; exit 1; }
stamp "approach set: collecting train split (4 per grasp, $W workers) -> $X"
run collect_train python $S/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split train --per-grasp 4 --workers $W --approach --seed 7000000
stamp "approach set: collecting ood_val split (1 per grasp, for checks only)"
run collect_val python $S/data/collect_explore.py --grasps $Q/data/wm_v1 --out $X --split ood_val --per-grasp 1 --workers $W --approach --seed 7000000
stamp "approach set: collection finished: $(find $X -name 't*.npz' | wc -l) trajectories, $(du -sh $X | cut -f1); $(free_gb) GB free"
run index python $S/data/build_index.py $X --workers 12
run ee python $S/data/compute_ee.py $X --out ${X}_ee --workers 8
mkdir -p $Q/runs/checks_v6
nice -n 10 python $S/analysis/check_contact_set.py --data $X --ee ${X}_ee --out $Q/runs/checks_v6/approach_set.json > $Q/logs/approach_checks.log 2>&1
python - <<PY >> $LOG
import json
r = json.load(open("$Q/runs/checks_v6/approach_set.json"))
print("[approach set] %s | phase %s | decisions by lead %s | %s | GATE %s %s" % (r["A_collection"]["outcome"], {k: r["B_contact_phase"][k] for k in ("trajectories_with_phase", "seconds_median", "seconds_total_hours", "hand_excursion_across_axis_mm", "inserted_at_phase_start_to_end")}, r["C_decisions_by_lead_along_axis"], r["D_no_truth_in_commands"], r["gate"], r["failed"]))
PY
grep -q '"auc_outcome_from_first_commands"' $Q/runs/checks_v6/approach_set.json || { stamp "FAILED: approach-set checks did not run"; exit 1; }
python - <<PY || { stamp "FAILED: the approach set's commands carry the truth (check D) -- not used for training"; exit 1; }
import json, sys
d = json.load(open("$Q/runs/checks_v6/approach_set.json"))["D_no_truth_in_commands"]
bad = any(v > 0.1 for k, v in d.items() if k.startswith("r2_")) or abs(d["auc_outcome_from_first_commands"] - 0.5) > 0.15
sys.exit(1 if bad else 0)
PY
[ -e $Q/runs/joint_v6d ] && { stamp "FAILED: runs/joint_v6d already exists"; exit 1; }
tmux new-session -d -s qingxin-v6d-joint "source $Q/env.sh; cd $Q/code/ManiSkill; CUDA_VISIBLE_DEVICES=1 WM_GPU_MEM_FRACTION=0.25 nice -n 10 python $S/train/train_joint.py --config world_model/configs/v6/joint_d.json --run $Q/runs/joint_v6d > $Q/logs/v6d_joint.log 2>&1"
tmux new-session -d -s qingxin-v6d-checks "GPU=0 bash $Q/code/ManiSkill/world_model/scripts/pipelines/v6_checks2.sh joint_v6d qingxin-v6d-joint 5000 10000 20000 > $Q/logs/v6d_checks.log 2>&1"
tmux new-session -d -s qingxin-v6d-approach-at "GPU=0 bash $Q/code/ManiSkill/world_model/scripts/pipelines/v6_approach_at.sh joint_v6d qingxin-v6d-checks 5000 10000 20000 > $Q/logs/v6d_approach_at.log 2>&1"
stamp "joint_v6d started from scratch (GPU 1, next to joint_v6c): v6c + the approach set as a fourth source (weights 0.3 / 0.2 / 0.25 / 0.25); checks at 5000 / 10000 / 20000: push, replay, approach"
