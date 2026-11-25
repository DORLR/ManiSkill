#!/bin/bash
# The CPU-vs-GPU PhysX check of decision 4 of 2026-11-22 ("CPU PhysX, with a CPU-vs-GPU comparison on hard
# assemblies"): milestone 1b (i, ii, iii) with GPU PhysX on a given list of assemblies into
# /mnt/storage3/qingxin/automate/m1b/batch_r3_gpu, the same procedure as the CPU re-run batch_r3_cpu (lead decision of
# round 7: in the collector with cameras GPU PhysX was measured 27-38 x slower per control step than CPU PhysX, so
# not all assemblies; the camera-free GPU batch of 2026-11-21, older code, took 32-50 s per assembly).
# Prepared, NOT started by anyone but the lead, and only after the CPU batch has finished.
#   [BATCH_TIMEOUT_S=21600] bash gpu_batch_r3.sh [N=6] [MEMMAX=18G] [--ids <a,b,...>] [--retry-crashes]
# --ids: the assemblies to run ("all": every assembly); default GPU_IDS, the 9 assemblies of the collector smoke (00015 00117 00133 00444 00470
# 00537 00755 00863 01053: low starts, IK-unreachable poses, end heights near 0, the asymmetric spin box; same list as
# collect_smoke_r3.json). Passed on to parallel_batch.sh --ids (each must be an assembly id). With 9 assemblies and
# N=6, a process runs at most 2 of them.
# BATCH_TIMEOUT_S: the per-process timeout of parallel_batch.sh, default here 21600 s (6 h; r7-ops-01). Not measured
# for this code without cameras; even at 30 x the CPU batch's median of 90 s per assembly, 2 assemblies take about
# 1.5 h, so 6 h leaves room for a slower step rate and a slow Kit start or shutdown. In-assembly hangs are still caught by the 1020 s heartbeat. An overrun ends with exit 124,
# keeps the assembly's <id>.started marker and books it as a crash; set a larger value at the first launch rather than
# restarting. If a restart is still needed (r8-ops-chk-01: the rule of run_batch.sh release_own_markers; the heartbeat
# files cannot be used, the watcher deletes a dead pid's heartbeat within its 60 s period, qx_watch.py):
#   1. touch <out>/STOP. BEFORE stopping anything, record for each loop p<w> whether the unit named in its newest unit
#      file m1/logs/batch_r3_gpu_<stamp>_p<w>_*.unit (largest unix time in the name) is active:
#      systemctl --user is-active <that unit> (r9-ops-chk-01). Then systemctl --user stop the exact units named in
#      m1/logs/batch_r3_gpu_<stamp>_p*_*.unit; wait until `tmux -L qingxin ls` shows no qingxin-batch* session.
#   2. ONLY for each loop p<w> whose newest unit was active in step 1 (its process was really stopped by us; a loop
#      whose newest process had already ended by itself, a crash or exit 124, and that was sleeping before its next
#      start when STOP was touched, keeps every marker: its crash / hang marker must be booked as such):
#      t = the unix time in the name of its newest log m1/logs/batch_r3_gpu_<stamp>_p<w>_<k>_<t>.log or
#      ..._<t>_<n>.log (largest t; the start of the stopped process); its ids = the "ids" of its "worker <w> started ...
#      log tag batch_r3_gpu_<stamp>_p<w> ..." line in m1/logs/parallel_launcher.log; remove each
#      <out>/<id>.started of those ids whose mtime (stat -c %Y) is >= t - 5. Keep older markers: they are real hangs or
#      crashes of earlier processes, to be booked as such.
#   3. move STOP to <out>/STOP.restart_<time>; launch again with the larger BATCH_TIMEOUT_S.
# What differs from the CPU batch: the switch WM_AUTOMATE_DEVICE=cuda:0, frozen in <out>/switches.env before
# parallel_batch.sh runs (batch_assemblies.py takes config.PHYSICS_DEVICE as its device, so every result records
# physics_device "cuda:0" and switches.physics_device_default "cuda:0"), the assembly list and the timeout. Every other
# switch must be config.APPROVED (cl_write_switches refuses otherwise). "cuda:0" is physical GPU 1: caps.sh qx_run
# starts every GPU job with CUDA_VISIBLE_DEVICES=1 (GPU 0 is never visible; the GPU-0 rule and the watcher are
# unchanged). MemoryMax 18G per process (collect.WORKER_RESOURCES["cuda:0"], measured GPU PhysX peaks 13-15 GB); GPU
# GB declared per process: 7 (run_on_rs2.sh default; GPU PhysX without cameras not measured yet: the watcher stops our
# newest unit if the measured GPU-1 sum passes our GPU-1 cap (72 GB since 2026-11-25)).
# Afterwards (not here): compare per assembly with batch_r3_cpu (compare_physics_device.py --batch-a batch_r3_gpu
# --batch-b batch_r3_cpu: (i) / (ii) / (iii) pass, depth_left, h_end, success_end, success_any; the assemblies only in
# the CPU batch are listed as such) and show the comparison to the user.
# The GPU collector smoke is OPTIONAL: run it only if this batch comparison shows differences, and ask the user first.
# It needs more than this batch: the collector's gate includes only assemblies with a (ii) result of the config's
# device, and the split automate_v1 is computed over the included assemblies, so with this batch's 9 assemblies its
# plan (split, grasp values) differs from the CPU smoke's and the trajectories cannot be compared grasp by grasp. Before
# it, extend this batch to every assembly (bash gpu_batch_r3.sh --ids all; up to 30 x the CPU batch's time) or
# agree another comparison with the user. The command (config collect_smoke_r3_gpu.json, the device in the tmux
# command itself: the tmux server does not pass the launching shell's environment on; r5-ops-04; hours 24: GPU PhysX
# with cameras measured ~1200-3500 s per grasp, expert_smoke/compare_gpu_cpu.json, 36 grasps over 2 workers, r7-ops-02):
#   tmux -L qingxin new -d -s qingxin-smoke-gpu "WM_AUTOMATE_DEVICE=cuda:0 SMOKE_ENV_RESULT=<json> bash \
#       <world_model>/scripts/automate/smoke/collect_smoke.sh <out smoke_gpu_*> \
#       <world_model>/scripts/automate/smoke/collect_smoke_r3_gpu.json 24"
# (<json>: smoke_env.py --checks fk,b,g,s,e --assembly 00015 run with QX_SWITCHES=batch_r3_gpu/switches.env after this
# batch from batch_r3_gpu/code, r4-expert-06; start it exactly as the collect_smoke.sh header shows (watcher
# registration, then the retrying run_on_rs2.sh loop in tmux, r10-ops-01), with B = batch_r3_gpu, the session name
# qingxin-smokeenv-gpu, the log names smoke_env_r3_gpu_\$i and --out $M/m1b/smoke_env_r3_gpu, never directly
# and never into m1/ (--out is required since r9-expert-02); r8-ops-02) (its plan's split must equal the CPU plan's
# split; otherwise compare by grasp name only and say so).
# Code (r6-ops-03): the first start must run from the CPU batch's frozen copy (or a tree with the same cl_code_hash):
#   bash /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu/code/scripts/automate/gpu_batch_r3.sh
# otherwise it is refused, so both batches run the same code, not only the same physics files. A later start
# (--retry-crashes; <out>/code exists) goes through cl_freeze_code's own unchanged-copy check instead. After the run,
# still check: cmp batch_r3_cpu/code.sha256 batch_r3_gpu/code.sha256
set -u
GPU_IDS=00015,00117,00133,00444,00470,00537,00755,00863,01053
ARGS=(); IDS=""; prev=""
for a in "$@"; do
  if [ "$prev" = "--ids" ]; then IDS=$a
  elif [ "$a" != "--ids" ]; then ARGS+=("$a"); fi
  prev=$a
done
[ "$prev" = "--ids" ] && { echo "refused: --ids needs a list"; exit 1; }
IDS=${IDS:-$GPU_IDS}
[ "$IDS" = all ] && PASS_IDS=() || PASS_IDS=(--ids "$IDS")
export BATCH_TIMEOUT_S=${BATCH_TIMEOUT_S:-21600}
OUT=/mnt/storage3/qingxin/automate/m1b/batch_r3_gpu
CPU_CODE_SHA=/mnt/storage3/qingxin/automate/m1b/batch_r3_cpu/code.sha256
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
[ -e "$OUT/STOP" ] && { echo "refused: $OUT/STOP exists ($(tail -1 "$OUT/STOP"))"; exit 1; }
if [ ! -e "$OUT/code" ]; then
  [ -f "$CPU_CODE_SHA" ] || { echo "refused: $CPU_CODE_SHA missing (the CPU batch has not frozen its code)"; exit 1; }
  h=$(cl_code_hash "$LIVE") || { echo "refused: cannot hash $LIVE"; exit 1; }
  [ "$h" = "$(cat "$CPU_CODE_SHA")" ] || { echo "refused: $LIVE (sha256 $h) differs from the CPU batch's frozen code" \
    "($CPU_CODE_SHA: $(cat "$CPU_CODE_SHA")); run this script from $(dirname "$CPU_CODE_SHA")/code"; exit 1; }
fi
mkdir -p "$OUT"
cl_freeze_code "$LIVE" "$OUT" || exit 1
if [ ! -e "$OUT/switches.env" ]; then
  ( export WM_AUTOMATE_DEVICE=cuda:0; cl_write_switches "$OUT/switches.env" "$OUT/code" ) && chmod a-w "$OUT/switches.env" ||
    { echo "refused: cannot write $OUT/switches.env"; exit 1; }
fi
grep -qx 'WM_AUTOMATE_DEVICE=cuda:0' "$OUT/switches.env" ||
  { echo "refused: $OUT/switches.env does not say WM_AUTOMATE_DEVICE=cuda:0: $(tr '\n' ' ' < "$OUT/switches.env")"; exit 1; }
echo "GPU batch: assemblies $IDS, timeout $BATCH_TIMEOUT_S s per process"
exec bash "$LIVE/scripts/automate/parallel_batch.sh" "$OUT" "${ARGS[@]}" "${PASS_IDS[@]}"
