#!/bin/bash
# Pre-launch smoke of the production collector on rs2 (finding r3-ops-01; NOT RUN YET: the lead starts it after the
# round-4 audit and the milestone-1b re-run into /mnt/storage3/qingxin/automate/m1b/batch_r3_cpu).
#   tmux -L qingxin new -d -s qingxin-smoke "SMOKE_ENV_RESULT=<smoke_env json> bash <world_model>/scripts/automate/smoke/collect_smoke.sh <out> [config] [hours=6]"
# SMOKE_ENV_RESULT (required, r4-expert-06): the result json of smoke_env.py --assembly 00015 --checks fk,b,g,s,e run
# AFTER the milestone-1b batch of the config's m1b_dirs from that batch's frozen code (<batch>/code) on GPU 1 with the
# batch's switches (QX_SWITCHES=<batch>/switches.env); checked by smoke_env_gate.py before anything starts: s1-s6 and
# e pass, physics_code_sha256, code_files_sha256 and switches equal to the batch results', code_changed_during_run
# False, started after the newest batch result had ended, no live batch heartbeat. Before the gate this script also
# refuses while a tmux -L qingxin session qingxin-batch* exists (a batch loop still running, as parallel_batch.sh
# checks; r7-integration-01). Set SMOKE_ENV_RESULT inside the tmux command (the tmux server does not pass the
# launching shell's environment on).
# How to start that smoke_env run (r8-ops-02), after the batch has ended. In the launching shell set
#   B=/mnt/storage3/qingxin/automate/m1b/batch_r3_cpu      (the batch of the config's m1b_dirs)
#   M=/mnt/storage3/qingxin/automate
# (both commands below use double quotes, so that the launching shell expands $B and $M; r9-ops-02). First register
# the output folder with the watcher, so that a watcher is running (and does not exit idle) during the run:
#   bash -c "source $B/code/scripts/automate/caps.sh; source $B/code/scripts/automate/collect_lib.sh;
#            cl_watch_ensure $M/m1b/smoke_env_r3_cpu $B/code"
# then, inside our caps (run_on_rs2.sh: qx_run scope, MemoryMax, GPU 1 declared, the batch's switches), tried again
# every 60 s, up to 60 tries, while the caps refuse the start (exit 99, e.g. a short memory-pressure burst; any other
# exit ends the loop; r10-ops-01). Each try has its own log m1/logs/smoke_env_r3_cpu_<i>.log (run_on_rs2.sh starts a
# log of the same name afresh): read the one with the highest i. The loop's own variables are escaped (\$i, \$rc,
# \$(seq ...)), so that the tmux shell expands them, while $B and $M are expanded by the launching shell:
#   tmux -L qingxin new -d -s qingxin-smokeenv "for i in \$(seq 1 60); do QX_SWITCHES=$B/switches.env MEMMAX=40G \
#       CPUQUOTA=400% GPU_GB=7 TIMEOUT=7200 bash $B/code/scripts/automate/run_on_rs2.sh smoke_env_r3_cpu_\$i \
#       scripts/automate/smoke_env.py --assembly 00015 --checks fk,b,g,s,e --out $M/m1b/smoke_env_r3_cpu; \
#       rc=\$?; [ \$rc -eq 99 ] || break; sleep 60; done"
# SMOKE_ENV_RESULT is then $M/m1b/smoke_env_r3_cpu/smoke_00015_<mount>_<weld>.json (smoke_env.py names it). Never start
# smoke_env.py directly (python -u smoke_env.py ...): that runs outside the caps (no scope, no MemoryMax, no GPU-1
# declaration). smoke_env.py has no default --out any more (r9-expert-02: it is required); never point it at
# /mnt/storage3/qingxin/automate/m1, which holds the earlier m1/smoke_00015_table_articulation.json and
# m1/smoke_00015_fatal.json.
# Code (r11-ops-02): start this script from the batch's frozen copy, bash <B>/code/scripts/automate/smoke/
# collect_smoke.sh (B = the config's first m1b_dirs folder); a tree whose cl_code_hash differs from <B>/code.sha256 is
# refused before <out> exists (collect_lib.sh cl_batch_code_guard; collect_launch.sh has the same guard). Before showing
# the smoke, still check cmp <B>/code.sha256 <out>/code.sha256.
# <out> must be a new folder whose name starts with smoke_ (e.g. /mnt/storage3/qingxin/automate/smoke/smoke_r3_<date>).
# config: collect_smoke_r3.json next to this script (production-shaped: device cpu, split automate_v1, the gate on
# the new batch folder, grasps_per_assembly 4 as validate() requires; collect_subset 00015, 00117, 00133, 00444 and
# (r5-expert-04) 00470 (low starts: clearance redraws), 00537 (most IK-unreachable poses), 00755 (end height -2.7 um
# in batch2 (i); r6-expert-02), 00863 (asymmetric round spin box), 01053 (end height <= 0 in the old (iii)), so the
# split is the production one and only those 36 grasps are collected). For 00755 and 01053 read t00_refused,
# label_failed and refused_completed_with_h_le_0 in acceptance.json (h) "draws" (AutoMate's h > 0 is strict). Why
# h <= 0 is expected to be rare (r7-expert-01): in (i) / (iii) the plug is placed by move_to (2 mm waypoints at
# 20 mm/s, then settling), which can end up to about 15 um below h = 0 (batch2: 01053 -15.2 um in (iii), 00755 -2.7 um
# in (i)). The expert's 3 mm/s push lags its command (the 0.3 mm insert throttle) and settles onto h = 0 from above
# (gravity off): measured h_end 26-104 um, never below, over the 186 expert_smoke ends (reference, insert and
# perturbed runs) on 00015 / 00083 / 01102, including 01102 whose plug rests on the socket and 00015 with 16 mm free
# below; smoke_env s4 +94 um. Such refusals are still counted (refused_completed_with_h_le_0, this smoke on 00755 /
# 01053, the first-hour draws_check).
# hours (default 6, r5-ops-06): the overall timeout; the 2026-11-23 changes (constant 2.7 mm/s random motion and
# contact offset, 3 mm/s perturbation window, starts from the socket top) make a grasp slower than the old estimate.
# Refusals before <out> exists (wrong name, existing folder, missing config, config device != WM_AUTOMATE_DEVICE of
# this shell, SMOKE_ENV_RESULT missing, a qingxin-batch* session running, code other than the batch's frozen copy, or
# the result refused by smoke_env_gate.py)
# are printed and appended to <dirname out>/smoke_refusals.log (visible when it runs in tmux; r5-ops-04); later
# failures go to <out>/smoke/smoke.log.
# Everything goes through the production path: collect_launch.sh (frozen code + inputs, capped preflight, watcher),
# collect_worker.sh restart loops, caps.sh qx_run scopes, collect_expert.py. This script only launches and samples
# (read-only nvidia-smi / systemctl queries every 60 s into <out>/smoke/samples.txt) and never kills anything itself
# (on its own timeout it calls collect_stop.sh --hard, which stops only our verified worker scopes).
# Phases:
#   1  collect_launch.sh <out> <config> 2
#   2  (g) once >= 4 grasps have grasp.json: a third loop (index 2) whose PATH starts with <out>/smoke/fakebin: its
#      nvidia-smi reports the worker as a compute context on GPU 0 from the second `nvidia-smi -i 0 -q -d PIDS` of
#      that process on (call 1: the app-start check, passes; call 2: the memory reading after the launch,
#      collect_expert.py CO._memory(), recorded only; call 3: the GPU check "after the scene build" in scene.py, for
#      its first claimed grasp, fails with the claim held), so
#      app.assert_off_forbidden_gpu exits 7 through PRE_EXIT_HOOKS. Expected: <out>/STOP written by the hook, every
#      loop ends (w2: "forbidden GPU-0 context (exit 7"; w0 / w1: "STOP file"), the held claim restored (no death)
#   3  STOP moved to <out>/smoke/STOP.g, the .end files to <out>/smoke/; collect_launch.sh <out> <config> 2 again
#      (resume from the frozen plan); wait until both loops end "done"
#   4  collect_smoke_check.py in a capped CPU scope: acceptance (a)-(h), printed and written to
#      <out>/smoke/acceptance.json; exit 0 only if every item passes; the rates it lists under show_user go to the
#      user before the launch. Its item (f) runs the production preparation launcher prep_automate_data.sh (run dir
#      <out>/smoke/check/prep, D_max 0.5 m as a smoke-only value), which starts its own capped CPU scopes
#      (qingxin-prep-prep-<step>-*, 24G / 800 % / GPU 0) one after the other while the checker's 8G scope waits
# Phase 2 waits until loop 2 holds a claim (its heartbeat names one); a loop 2 that ends "done" first fails the smoke
# with that reason. Every sample also lists each heartbeat file's age and stage (checker item (e)).
# GPU PhysX variant: OPTIONAL, not part of the pre-launch smoke (lead decision of round 7): run only if the per-assembly
# comparison of gpu_batch_r3.sh's batch with batch_r3_cpu shows differences, after asking the user, and only once
# batch_r3_gpu has results for every assembly (its default is the 9 smoke assemblies; with fewer included assemblies
# the split and grasp values differ from the CPU smoke's; gpu_batch_r3.sh header), or after another comparison is
# agreed with the user. The config collect_smoke_r3_gpu.json (m1b_dirs = batch_r3_gpu) and WM_AUTOMATE_DEVICE=cuda:0
# inside the tmux command (the tmux server does not inherit the launching shell's environment); hours 24: GPU PhysX with cameras measured ~1200-3500 s
# per grasp (expert_smoke/compare_gpu_cpu.json), 36 grasps over 2 workers (r7-ops-02):
#   tmux -L qingxin new -d -s qingxin-smoke-gpu "WM_AUTOMATE_DEVICE=cuda:0 SMOKE_ENV_RESULT=<smoke_env json of batch_r3_gpu's code and switches> bash <world_model>/scripts/automate/smoke/collect_smoke.sh <out> <world_model>/scripts/automate/smoke/collect_smoke_r3_gpu.json 24"
# The config's "device" must equal ${WM_AUTOMATE_DEVICE:-cpu} (refused otherwise, before anything starts).
# GPU-1 room before the start (r11-ops-01): caps.sh charges each of our processes its declared 7 GB against GPU 1's
# free memory F (rule F - P - G_new >= 12 GB), and phase 2 needs a third worker while loops 0 and 1 run: F >= 31472 MiB
# (11444 + 3 x 6676). Other users' GPU-1 memory moved by about 3 GB within 30 s on 2026-11-23, so start this smoke only
# when F = `nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits` is at least 34500 MiB in each of 5
# samples spread over about 5 minutes. While loops 0 and 1 run phase 1, read <out>/logs/launcher.log for caps refusals
# of loop 2: a run of them until loops 0 and 1 finish the 36 grasps ends the smoke with "loop 2 never claimed a grasp".
# Afterwards: worker_gpu_gb_recommended = ceil(1.3 x the measured GPU-1 maximum) and the time estimate (printed by the
# checker). Production sets worker_gpu_gb in its config to that recommendation, also when it is below the default 7
# (collect.WORKER_GPU_GB; no code change, r7-ops-05, r11-ops-01); N workers x worker_gpu_gb <= 72 (our GPU-1 cap since 2026-11-25), and the workers
# that fit now are n = min(6, floor((F - 12 GB) / worker_gpu_gb)): report the time estimate for that n to the user and
# re-check F before each collect_add_worker.sh.
set -u
[ $# -ge 1 ] || { echo "usage: bash collect_smoke.sh <out dir smoke_*> [config] [hours=6]"; exit 1; }
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
LIVE=$(cd "$HERE/../../.." && pwd)
OUT=$(readlink -m "$1"); CFG=$(readlink -m "${2:-$HERE/collect_smoke_r3.json}"); HOURS=${3:-6}
refuse() {      # before <out> exists: printed and appended to <dirname out>/smoke_refusals.log
  echo "refused: $*"
  mkdir -p "$(dirname "$OUT")" 2>/dev/null &&
    echo "$(date '+%F %T') collect_smoke.sh $OUT: refused: $*" >> "$(dirname "$OUT")/smoke_refusals.log"
  exit 1
}
[[ "$HOURS" =~ ^[1-9][0-9]*$ ]] || refuse "hours '$HOURS' must be a whole number >= 1"
TMAX=$(( HOURS * 3600 ))
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
[[ "$(basename "$OUT")" == smoke_* ]] || refuse "the folder name must start with smoke_"
[ -e "$OUT" ] && refuse "$OUT exists (a smoke needs a new folder)"
[ -f "$CFG" ] || refuse "config $CFG missing"
cdev=$("$CL_PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1])).get("device"))' "$CFG" 2>&1) ||
  refuse "config $CFG unreadable: $cdev"
[ "$cdev" = "${WM_AUTOMATE_DEVICE:-cpu}" ] ||
  refuse "config device '$cdev' differs from WM_AUTOMATE_DEVICE '${WM_AUTOMATE_DEVICE:-cpu}' of this shell (GPU smoke: put WM_AUTOMATE_DEVICE=cuda:0 inside the tmux command)"
[ -n "${SMOKE_ENV_RESULT:-}" ] ||
  refuse "SMOKE_ENV_RESULT is not set: the smoke_env.py result of the batch's frozen code is required (r4-expert-06; header)"
busy=$($CL_TMUX ls -F '#S' 2>/dev/null | grep '^qingxin-batch')
[ -z "$busy" ] || refuse "batch loops still running: $(echo $busy) (the smoke starts after the milestone-1b batch; r7-integration-01)"
guard=$(cl_batch_code_guard "$LIVE" "$OUT" "$CFG") || refuse "${guard#refused: }"
gate=$("$CL_PY" -I "$HERE/smoke_env_gate.py" "$SMOKE_ENV_RESULT" "$CFG" 2>&1) ||
  refuse "smoke_env gate ($SMOKE_ENV_RESULT): $(grep -v '^INFO' <<< "$gate" | tr '\n' ' ' | head -c 1500)"
S=$OUT/smoke
mkdir -p "$S/fakebin" "$S/fakecnt"
T0=$(date +%s)
log() { echo "$(date '+%F %T') $*" | tee -a "$S/smoke.log"; }
fail() { log "SMOKE FAILED: $*"; exit 1; }
REAL_SMI=$(command -v nvidia-smi) || fail "nvidia-smi not found"
log "smoke_env gate: $(tr '\n' ' ' <<< "$gate")"
log "$guard"

sample() {      # one line of read-only readings (no process is touched)
  local alive age us
  alive=$(stat -c %Y "$QX_RUN_DIR/watch.alive" 2>/dev/null || echo 0); age=$(( $(date +%s) - alive ))
  us=$(qx_units 2>/dev/null | tr '\n' ' ')
  { echo "=== $(date '+%F %T') watch.alive_age_s $age heartbeats $(ls "$OUT/heartbeat" 2>/dev/null | grep -c '\.json$')" \
         "grasp_json $(ls "$OUT"/*/*_g*/grasp.json 2>/dev/null | wc -l) units $us"
    for u in $us; do echo "unit $u pids $(qx_unit_pids "$u")"; done
    for f in "$OUT"/heartbeat/*.json; do        # the watcher kills at HANG_TIMEOUT_S; the checker wants < half of it
      [ -e "$f" ] || continue
      echo "hb $(basename "$f") age_s $(( $(date +%s) - $(stat -c %Y "$f" 2>/dev/null || date +%s) ))" \
           "stage $(grep -o '"stage": "[^"]*"' "$f" | sed 's/^"stage": "//; s/"$//')"
    done
    echo "gpu0:"; env -u CUDA_VISIBLE_DEVICES "$REAL_SMI" -i 0 -q -d PIDS | qx_parse_pid_types
    echo "gpu1:"; env -u CUDA_VISIBLE_DEVICES "$REAL_SMI" -i 1 -q -d PIDS | qx_parse_pid_types; } >> "$S/samples.txt" 2>&1
}

wait_until() {  # wait_until <description> <shell condition>: sample every 60 s; overall timeout TMAX
  while ! eval "$2"; do
    sample
    if [ $(( $(date +%s) - T0 )) -gt $TMAX ]; then
      bash "$LIVE/scripts/automate/collect_stop.sh" "$OUT" --hard >> "$S/smoke.log" 2>&1
      fail "timeout ($TMAX s) while waiting for: $1 (collect_stop.sh --hard called)"
    fi
    sleep 60
  done
  sample
}

n_done() { ls "$OUT"/*/*_g*/grasp.json 2>/dev/null | wc -l; }
ended() { [ -e "$OUT/logs/w$1.end" ]; }

# the fake nvidia-smi of phase 2 (only `-i 0 -q -d PIDS` is changed; every other call goes to the real one)
cat > "$S/fakebin/nvidia-smi" <<EOS
#!/bin/bash
if [[ " \$* " == *" -i 0 "* && " \$* " == *" -q "* && " \$* " == *" PIDS "* ]]; then
  f="$S/fakecnt/\$PPID"; n=\$(( \$(cat "\$f" 2>/dev/null || echo 0) + 1 )); echo \$n > "\$f"
  out=\$("$REAL_SMI" "\$@") || exit \$?
  echo "\$out"
  if [ \$n -ge 2 ]; then
    printf '        Process ID                        : %s\n            Type                          : C\n            Used GPU Memory               : 300 MiB\n' "\$PPID"
  fi
  exit 0
fi
exec "$REAL_SMI" "\$@"
EOS
chmod +x "$S/fakebin/nvidia-smi"

log "phase 1: collect_launch.sh $OUT $CFG 2"
bash "$LIVE/scripts/automate/collect_launch.sh" "$OUT" "$CFG" 2 >> "$S/smoke.log" 2>&1 || fail "collect_launch.sh refused (see $S/smoke.log)"
wait_until ">= 4 grasp.json" '[ "$(n_done)" -ge 4 ] || { ended 0 && ended 1; }'
{ ended 0 && ended 1; } && fail "both loops ended before phase 2 ($(cat "$OUT/logs/w0.end") / $(cat "$OUT/logs/w1.end"))"

log "phase 2 (g): loop 2 with the fake nvidia-smi (exit 7 expected after its claim)"
sess=qingxin-collect-$(basename "$OUT")-g
$CL_TMUX new -d -s "$sess" "PATH=$S/fakebin:\$PATH bash $OUT/code/scripts/automate/collect_worker.sh $OUT 2 3 $sess" ||
  fail "tmux failed for $sess"
# loop 2 must claim a grasp before loops 0 / 1 finish the plan (its heartbeat, index 2, names a claim), or end
claimed2() { grep -l '"index": 2,' "$OUT"/heartbeat/*.json 2>/dev/null | xargs -r grep -l '"claim": "' | grep -q .; }
wait_until "loop 2 claimed a grasp (or ended)" 'claimed2 || ended 2'
if ended 2 && grep -q 'done (WORKER DONE' "$OUT/logs/w2.end"; then
  fail "loop 2 never claimed a grasp (it ended done: loops 0 / 1 finished the plan first; launcher.log shows its caps refusals)"
fi
wait_until "loops 0, 1, 2 ended after the exit 7" 'ended 0 && ended 1 && ended 2'
grep -q 'forbidden GPU-0 context (exit 7' "$OUT/logs/w2.end" || fail "loop 2 did not end with exit 7: $(cat "$OUT/logs/w2.end")"
[ -e "$OUT/STOP" ] || fail "no STOP after the exit 7"
mv "$OUT/STOP" "$S/STOP.g"
for w in 0 1 2; do mv "$OUT/logs/w$w.end" "$S/w$w.end.phase2"; done
log "phase 2 done: $(tr '\n' ' ' < "$S/STOP.g")"

log "phase 3: resume with 2 workers"
bash "$LIVE/scripts/automate/collect_launch.sh" "$OUT" "$CFG" 2 >> "$S/smoke.log" 2>&1 || fail "the resume was refused"
wait_until "loops 0 and 1 ended" 'ended 0 && ended 1'
grep -q '^.* done (WORKER DONE' "$OUT/logs/w0.end" && grep -q '^.* done (WORKER DONE' "$OUT/logs/w1.end" ||
  fail "a loop did not end done: $(cat "$OUT/logs/w0.end") / $(cat "$OUT/logs/w1.end")"

log "phase 4: acceptance checks (capped CPU scope)"
( cl_clean_env "$OUT/inputs/switches.env" >/dev/null && export WM_AUTOMATE_ASSETS=$OUT/inputs/assets WM_AUTOMATE_M1B=$OUT/inputs/m1b &&
  qx_run_retry prep "smokecheck" 8G 200% 0 "$CL_PY" -u "$OUT/code/scripts/automate/smoke/collect_smoke_check.py" --out "$OUT" ) \
  >> "$S/smoke.log" 2>&1
rc=$?
tail -40 "$S/smoke.log"
show=$(grep -c 'SHOW THE USER BEFORE LAUNCH' "$S/smoke.log")
[ $rc -eq 0 ] && log "SMOKE PASSED ($show placement / rejection rates to show the user before the launch: $S/acceptance.json show_user)" ||
  fail "acceptance checks failed (exit $rc; $S/acceptance.json)"
