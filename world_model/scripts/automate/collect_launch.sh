#!/bin/bash
# Start a collection on rs2 (GPU 1 only):   bash collect_launch.sh <out dir> <config json> <N workers>
# 1. Freezes the code: this script's world_model tree is copied read-only to <out>/code (hash in <out>/code.sha256);
#    a resumed collection must find <out>/code unchanged. The first launch (no <out>/code yet) is refused, before <out>
#    is created, unless that tree's hash equals <B>/code.sha256 of the config's first m1b_dirs folder B (collect_lib.sh
#    cl_batch_code_guard, r11-ops-02): start it as bash <B>/code/scripts/automate/collect_launch.sh ..., so production
#    runs the code the batch and the collector smoke ran. Right after the first launcher lines, still check
#    cmp <smoke out>/code.sha256 <out>/code.sha256 (and run collect_stop.sh if they differ).
# 2. Freezes the inputs (config, milestone-1b results, assets) in <out>/inputs and runs the CPU preflight in a capped
#    scope (collect_prep.sh): the config must name every production key (physics device, grasps per assembly, split,
#    expected_grasps; wm.sim_automate.collect.PRODUCTION_KEYS), the plan must have expected_grasps grasps, and the
#    worker resources come from the config or from the measured defaults per device (16G CPU PhysX, refused below
#    14G; 18G GPU PhysX, refused below 18G; 500 %; GPU-1 GB worker_gpu_gb, default 7, an integer 1..48, r7-ops-05).
#    Any refusal stops here, before tmux or Isaac.
# 3. Refuses N workers that could never fit the caps together, starts the watcher (qx_watch.py) if none runs, and
#    starts N restart loops (collect_worker.sh from <out>/code) in tmux -L qingxin sessions
#    qingxin-collect-<out name>-<index>. The starts themselves are serialised by caps.sh qx_run (lock, 90 s apart,
#    each checked against our caps when it happens).
# Stop: collect_stop.sh. Add a worker: collect_add_worker.sh. Re-allow a skipped assembly: collect_unblock.sh.
# Worker count (lead resolution r3-ops-12): start 4 workers, measure for 1 h (memory.peak, GPU-1 memory and seconds
# per grasp in grasp.json), then add workers with collect_add_worker.sh up to 6 (6 x 16G / 500 % / 7 GB fit the caps);
# re-estimate the end from that hour: T_h = 15 x N_incl x t_g / (n x 3600), t_g = mean grasp.json seconds, n workers.
# While it runs (e.g. hourly): bash <out>/code/scripts/automate/draws_check.sh <out> prints every flag of
# check_expert_data.draw_report (row-22 condition: per-assembly draw-loss, t00-refusal, placement_failed,
# random_truncated and redraw shares above 5 %, starts already satisfying AutoMate's rule; grasp.json only, seconds
# of CPU) for the user.
# First production hour (r6-expert-02 c): run draws_check.sh once and show the user, for the assemblies whose batch
# end height was <= 0.01 mm in (i), (ii) or (iii) (listed after batch_r3_cpu), refused_completed_with_h_le_0,
# t00_refused and label_failed from <out>/logs/draws/draws.json (AutoMate's h > 0 is strict; no rule change).
# A worker loop starts nothing while the watcher is not running (collect_worker.sh).
# Run plan up to production (r12-ops-01 / r12-ops-02; procedure, nothing here enforces it). Strictly one step at a
# time, each started only after the previous one has ended:
#   1. CPU batch: parallel_batch.sh into /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu (this freezes the code),
#      then report_m1b.py (table_r3_cpu.md).
#   2. GPU batch: gpu_batch_r3.sh (batch_r3_gpu, decision 4 of 2026-11-22).
#   3. Encoder check: train_automate.sh ... scripts/train/gpu_check_encoders320.py (40G / 800 % / 36 GB of GPU 1).
#   4. smoke_env: run_on_rs2.sh ... smoke_env.py --assembly 00015 --checks fk,b,g,s,e from <batch>/code (the command
#      is in smoke/collect_smoke.sh's header).
#   5. Collector smoke: <batch>/code/scripts/automate/smoke/collect_smoke.sh. When it is shown to the user
#      (acceptance.json show_user), give the decision-7 numbers of 00422, 00581, 00186 and 00741 next to it from the
#      geometry probe /mnt/storage3/qingxin/audit_am_r13/fix/probe/d7.json (hand_finger_command_clearance counts by
#      kind and waypoint redraws): none of the four is in the smoke's collect_subset, so acceptance.json cannot show
#      them (its 9 ids show 0, as the probe predicts); do not report that as 0 for them. Production's draw_report
#      (check_expert_data.hand_finger_rows) gives the real counts per assembly.
#   6. Production: <batch>/code/scripts/automate/collect_launch.sh (this script).
#   7. Afterwards prep_automate_data.sh, then train_automate.sh runs.
# Disk (r13-ops-01): rs2's /mnt/storage2 is full (28 GB free, 100 %, 2026-11-23); it holds the rs2 working tree and the
# HF / torch caches. Write nothing new there. Every path chosen by hand goes under /mnt/storage3/qingxin/automate/...:
# the fast dir of prep_automate_data.sh (e.g. /mnt/storage3/qingxin/automate/fast/<collection>) and every run dir of
# train_automate.sh and of the encoder check (e.g. /mnt/storage3/qingxin/automate/train/<name>). Before
# prep_automate_data.sh run `df -h /mnt/storage3` and go on only with >= 2 x the collection size free (about 62 GB for
# a ~31 GB collection): the collector's disk check (DISK_FREE_MIN_CLAIM) covers only <out>, not the fast copy or
# training checkpoints.
# Before each step, confirm that no other run of ours is active: `tmux -L qingxin ls` and
# `systemctl --user list-units 'qingxin-*'` may show only the watcher (qx_watch.py), no qingxin-batch*,
# qingxin-smokeenv, qingxin-smoke*, qingxin-collect-* session and no other qingxin-* unit. Only collect_smoke.sh
# checks this itself, and only for qingxin-batch* sessions; parallel_batch.sh, gpu_batch_r3.sh, train_automate.sh and
# this script do not. Two of our runs at once stay inside the caps, but the result is wasted or wrong: the smoke's
# loop 2 is refused for hours ("loop 2 never claimed a grasp"), its seconds per grasp (and the production estimate
# from it) are measured under contention, and production workers wait on refusals with ALERT lines.
# After step 1, before step 2: check in table_r3_cpu.md that each of the 9 collector-smoke ids (collect_subset of
# smoke/collect_smoke_r3.json: 00015 00117 00133 00444 00470 00537 00755 00863 01053) is "included": (ii) pass,
# including S_end (AutoMate's strict h > 0); (iii) pass, or skipped with margin >= 3 mm; nothing stale. Watch 00470
# (batch2 (ii) end at +9.8 um), 00015, 00117 and 00755 (+12 to +13 um). If an id is excluded, do NOT edit the tree
# (collect_smoke_r3.json is in it: the tree hash changes and cl_batch_code_guard then refuses every smoke and
# production start from it). Write a config outside the tree, e.g.
# /mnt/storage3/qingxin/automate/configs/collect_smoke_r3b.json, with substitutes of the same hard property and
# expected_grasps = 4 x the subset size, show the substitution to the user, and pass that file as collect_smoke.sh's
# second argument. About 10 min after the smoke starts, read <out>/smoke/smoke.log and <dirname out>/
# smoke_refusals.log (a refusal of the subset ends the smoke "collect_launch.sh refused" within minutes).
set -u
[ $# -eq 3 ] || { echo "usage: bash collect_launch.sh <out dir> <config json> <N workers>"; exit 1; }
OUT=$(readlink -f "$1"); CFG=$(readlink -f "$2"); N=$3
[[ "$N" =~ ^[1-9][0-9]*$ ]] || { echo "refused: N '$N'"; exit 1; }
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
[ -e "$OUT/STOP" ] && { echo "refused: $OUT/STOP exists (remove it to resume)"; exit 1; }
[ -f "$CFG" ] || { echo "refused: config $CFG missing"; exit 1; }
cl_batch_code_guard "$LIVE" "$OUT" "$CFG" || exit 1            # first launch: the batch's frozen code only (r11-ops-02)
mkdir -p "$OUT/logs"
L=$OUT/logs/launcher.log
cl_freeze_code "$LIVE" "$OUT" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
cl_prep "$OUT" "$CFG" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
read -r MEM CPU GPU TIMEOUT < "$OUT/logs/resources.txt"
cl_static_budget "$N" "$MEM" "$CPU" "$GPU" | tee -a "$L"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
echo "$(date '+%F %T') launch: $N workers, MemoryMax $MEM, CPUQuota $CPU, GPU $GPU GB, hang timeout $TIMEOUT s" | tee -a "$L"
cl_watch_ensure "$OUT" "$OUT/code" | tee -a "$L"
for w in $(seq 0 $((N - 1))); do
  sess=qingxin-collect-$(basename "$OUT")-$w
  if $CL_TMUX has-session -t "=$sess" 2>/dev/null; then
    echo "$(date '+%F %T') worker $w NOT started: tmux session $sess exists" | tee -a "$L"; continue
  fi
  if $CL_TMUX new -d -s "$sess" "bash $OUT/code/scripts/automate/collect_worker.sh $OUT $w $N $sess"; then
    echo "$(date '+%F %T') worker $w: restart loop in tmux -L qingxin session $sess" | tee -a "$L"
  else
    echo "$(date '+%F %T') worker $w NOT started: tmux failed" | tee -a "$L"
  fi
done
