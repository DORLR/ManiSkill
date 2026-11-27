#!/bin/bash
# Online rounds of the 320 px world model on AutoMate (rs2, GPU 1 only), next to its learner, as
# scripts/pipelines/online320.sh does for ManiSkill (the user's decision 10 of 2026-11-25):
#   bash online_automate.sh <learner run dir> <learner out> <pool> <grasps collection> <workers> <torch MiB> \
#        <MemoryMax> <CPUQuota> <GPU GB> <last round> <min_step> [starts=3]
# <learner run dir>: train_automate.sh's run dir (its unit name in <run dir>/logs/train_unit_*); <learner out>: the
# learner's --out (log.jsonl, last.pt, DONE.json); the learner must have been started with --online <pool>.
# <grasps collection>: the AutoMate collection the learner's fast copy was made from (plan.json, split_automate_v1.json):
# the rounds collect on its training grasps, the tests use its selection grasps. Resources per worker are the caller's
# (no defaults; from the smoke T7: MemoryMax >= the measured scope peak, GPU GB >= Kit's measured GPU-1 peak +
# <torch MiB> + 1 GB); every start is checked against the live sums of all our units, the learner's included
# (caps.sh: 180 GB / 4800 % / 72 GB of GPU 1 since 2026-11-25, the user's plan B; before 125 GB / 3200 % / 48 GB). If the planned workers do not fit beside the learner, the starts wait
# (caps refusals, ALERT after 30 min): show the numbers to the user, never shrink silently.
# 1. Code: the rounds run with the learner's frozen code <learner run dir>/code (unchanged since train_automate.sh froze
#    it: cl_code_hash = <learner run dir>/code.sha256) when every file the rounds call (ROUND_FILES: the online
#    scripts, run_wm320_attempts.py, eval_policy320.py, online_job.sh / online_worker.sh / online_automate.sh, caps.sh,
#    collect_lib.sh, qx_watch.py) exists there with the same sha256 as in this script's world_model tree; otherwise
#    (a file missing or different, each named in the log) this tree is frozen to <pool>/code (cl_freeze_code; a
#    restart must find it unchanged) and the rounds are refused unless its src/wm hash equals the learner's code's
#    src/wm hash. A pool whose <pool>/code exists from an earlier start keeps it (a restart does not switch code). job.json and every grasp result record the code (tree hash, physics code identity).
#    Switches: the grasps collection's frozen <collection>/inputs/switches.env (no new switches file; prepare refuses
#    switches or a physics device other than plan.json's); assets: the collection's frozen inputs (online_job.sh).
# 2. Round R = 1..<last round>; finished parts are skipped, so a restart resumes. Before collecting round R, waits (as
#    online320.sh) until the learner
#      R = 1: reads this pool (its newest start / resume record has --online <pool>) and logged step >= <min_step> + 500;
#      R > 1: has read round R-1 (first data record of the newest unbroken run listing it) and logged 1000 more steps;
#    every 5 minutes: DONE.json of the learner -> the rounds end (exit 0); its unit not running at two checks in a row
#    -> the rounds end (exit 1). Then the learner's newest last.pt is copied to <pool>/r<R>/model.pt and the round is
#    collected on every training grasp x <starts> (online_job.sh: prepare --mode collect --split train --pool <pool>
#    --round R --min-step <min_step> (a model below min_step is refused), <workers> capped workers, finish:
#    <pool>/r<R>/manifest.json + DONE). Extra options for the collection and the "wm" test: ATT_EXTRA (e.g.
#    "--inside-mm 3"); the "act_done" test gets ATT_EXTRA without "--inside-mm <v>" (act_done refuses it) plus
#    ACT_EXTRA (default none), so that both tests share every other option. --max-retries is 0 (decision 5; the
#    AutoMate prepare refuses anything else).
# 3. Then the same model is tested on the selection grasps (all x 4 starts): mode act_done (the action head alone with
#    the done stop, decision 7) and mode wm, into <pool>/r<R>/select_<mode>. Two rounds in a row without a better
#    "wm" success (AutoMate's rule, groups.all.success) than the best before end the rounds (exit 0). The two tests
#    must be paired: when both have a job.json, they agree on seed, starts and the grasp list, else the rounds end
#    (exit 1).
#    A job (collection or test) whose finish reports a start_success stop (exit 4: a placed start already satisfied
#    AutoMate's rule, plan item 5) writes <pool>/STOP and the rounds end (exit 1).
# Log: <pool>/logs/online.log. Stop: date > <pool>/STOP (checked between steps) and date > <job>/STOP for a running job.
set -u
[ $# -ge 11 ] || { echo "usage: bash online_automate.sh <learner run dir> <learner out> <pool> <grasps> <workers> <torch MiB> <MemoryMax> <CPUQuota> <GPU GB> <last round> <min_step> [starts]"; exit 1; }
LRUN=$(readlink -f "$1"); LOUT=$(readlink -f "$2"); P=$(readlink -m "$3"); GR=$(readlink -f "$4")
workers=$5; tmib=$6; MEM=$7; CPU=$8; GPU=$9; r1=${10}; min_step=${11}; starts=${12:-3}
att_extra=(${ATT_EXTRA:-}); act_extra=(${ACT_EXTRA:-})
LIVE=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
source "$LIVE/scripts/automate/caps.sh"
source "$LIVE/scripts/automate/collect_lib.sh"
for v in "$workers" "$tmib" "$GPU" "$r1" "$min_step" "$starts"; do [[ "$v" =~ ^[0-9]+$ ]] || { echo "not a whole number: $v"; exit 1; }; done
[ "$r1" -ge 1 ] && [ "$workers" -ge 1 ] || { echo "need at least one round and one worker"; exit 1; }
[ $(( GPU * 1000000000 )) -ge $(( tmib * 1048576 + 1000000000 )) ] ||
  { echo "refused: GPU GB $GPU < torch MiB $tmib + 1 GB (CUDA context), before Kit's own GPU memory"; exit 1; }
[ -f "$GR/plan.json" ] || { echo "refused: $GR has no plan.json"; exit 1; }
SW=$GR/inputs/switches.env
[ -f "$SW" ] || { echo "refused: $SW missing (the collection's frozen switches)"; exit 1; }
cl_static_budget "$workers" "$MEM" "$CPU" "$GPU" || exit 1
mkdir -p "$P/logs"
LOG=$P/logs/online.log
olog() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
LCODE=$LRUN/code
[ -f "$LRUN/code.sha256" ] && [ "$(cl_code_hash "$LCODE")" = "$(cat "$LRUN/code.sha256")" ] ||
  { olog "refused: the learner's frozen code $LCODE is missing or changed since it was frozen"; exit 1; }
ROUND_FILES="scripts/online/run_wm320_automate.py scripts/online/run_wm320_attempts.py scripts/eval/eval_policy320.py
scripts/automate/online_job.sh scripts/automate/online_worker.sh scripts/automate/online_automate.sh
scripts/automate/caps.sh scripts/automate/collect_lib.sh scripts/automate/qx_watch.py"
round_files_differ() {   # <code tree> <code tree>: prints the ROUND_FILES missing in either or with another sha256
  local f
  for f in $ROUND_FILES; do
    if [ ! -f "$1/$f" ] || [ ! -f "$2/$f" ]; then printf ' %s(missing)' "$f"
    elif [ "$(sha256sum < "$1/$f")" != "$(sha256sum < "$2/$f")" ]; then printf ' %s' "$f"; fi
  done
}
differ=$(round_files_differ "$LCODE" "$LIVE")
[ -z "$differ" ] && [ -d "$P/code" ] && differ=" (none, but $P/code was frozen by an earlier start: kept)"
if [ -z "$differ" ]; then
  CODE=$LCODE
  olog "the rounds run with the learner's frozen code $CODE (sha256 $(cat "$LRUN/code.sha256")); its round files equal $LIVE's"
else
  olog "the learner's code differs from $LIVE in the round files:$differ; freezing $LIVE to $P/code (src/wm must equal the learner's)"
  cl_freeze_code "$LIVE" "$P" | tee -a "$LOG"; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
  CODE=$P/code
  hp=$(cl_code_hash "$CODE/src/wm") && hl=$(cl_code_hash "$LCODE/src/wm") && [ "$hp" = "$hl" ] ||
    { olog "refused: $CODE/src/wm (sha256 ${hp:-?}) differs from the learner's $LCODE/src/wm (sha256 ${hl:-?})"; exit 1; }
  olog "the rounds run with $CODE (sha256 $(cat "$P/code.sha256")), src/wm = the learner's (sha256 $hp)"
fi

learner_state() {    # [round] -> "<reads this pool: 1 | 0> <newest logged step> <step the round was first read, or -1>"
  "$CL_PY" -I - "$LOUT/log.jsonl" "$P" "${1:-}" <<'EOF'
import json, os, sys
ok, step, seen = 0, 0, -1
try:
    for line in open(sys.argv[1]):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        ev = r.get("event")
        if ev in ("start", "resume"):
            on = (r.get("args") or {}).get("online")
            ok = int(bool(on) and os.path.realpath(on) == os.path.realpath(sys.argv[2]))
        if isinstance(r.get("step"), int):
            step = max(step, r["step"])
        if ev == "data" and sys.argv[3]:
            if sys.argv[3] in (r.get("online_rounds") or []):
                seen = r["step"] if seen < 0 else seen
            else:
                seen = -1
except OSError:
    pass
print(ok, step, seen)
EOF
}

learner_running() {  # the unit named by the newest <learner run dir>/logs/train_unit_* is active
  local f u
  f=$(ls -t "$LRUN"/logs/train_unit_* 2>/dev/null | head -1)
  [ -n "$f" ] && u=$(cat "$f") && [ -n "$u" ] || return 1
  case "$u" in *.scope|*.service) ;; *) u="$u.scope" ;; esac   # qx_run writes the scope name without its suffix
  qx_unit_active "$u"
}

wait_learner() {     # <R> (header)
  local down=0 said="" ok step seen running n=0
  while true; do
    read -r ok step seen < <(learner_state "r$(( $1 - 1 ))")
    [ -e "$LOUT/DONE.json" ] && { olog "the learner is finished; rounds end before round $1"; exit 0; }
    [ -e "$P/STOP" ] && { olog "$P/STOP: rounds end before round $1"; exit 0; }
    running=""; learner_running && running=1
    if [ -n "$running" ]; then down=0; else down=$(( down + 1 )); fi
    [ "$down" -lt 2 ] || { olog "the learner's unit is not running; rounds end before round $1"; exit 1; }
    if [ -n "$running" ] && [ "$ok" = 1 ]; then
      if [ "$1" = 1 ]; then [ "$step" -ge $(( min_step + 500 )) ] && [ -e "$LOUT/last.pt" ] && return 0
      else [ "$seen" -ge 0 ] && [ "$step" -ge $(( seen + 1000 )) ] && return 0; fi
    fi
    if [ -z "$said" ] || [ $(( n % 12 )) = 0 ]; then
      olog "round $1 waits for the learner (unit running: ${running:-no}, reads $P: $ok, step $step, round $(( $1 - 1 )) read at step $seen)"
      said=1
    fi
    n=$(( n + 1 ))
    sleep 300
  done
}

success_of() {       # <summary.json> -> the success figure of all selection grasps, or nothing
  "$CL_PY" -I -c "import json,sys
try:
    s = json.load(open(sys.argv[1]))['groups']['all']['success']
    print('' if s is None else s)
except Exception:
    pass" "$1"
}

job() {   # <job dir> <prepare args...>
  bash "$CODE/scripts/automate/online_job.sh" "$CODE" "$SW" "$1" "$workers" "$MEM" "$CPU" "$GPU" "${@:2}"
}

act_done_options() {   # ATT_EXTRA without "--inside-mm <v>" / "--inside-mm=<v>", then ACT_EXTRA (one option per line)
  local a skip=0
  for a in "${att_extra[@]}"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case $a in --inside-mm) skip=1; continue ;; --inside-mm=*) continue ;; esac
    printf '%s\n' "$a"
  done
  for a in "${act_extra[@]}"; do printf '%s\n' "$a"; done
}

paired() {   # <job.json> <job.json>: 0 when they agree on seed, starts and the grasp list; else a message, 1
  "$CL_PY" -I -c 'import json, sys
try:
    a, b = (json.load(open(f)) for f in sys.argv[1:3])
except (OSError, ValueError) as exc:
    sys.exit(f"unreadable job.json: {exc}")
key = lambda j: (j.get("seed"), j.get("starts"), [g.get("grasp") for g in j.get("grasps") or []])
if key(a) != key(b):
    sys.exit("the tests are not paired: seed / starts / grasps " + " vs ".join(
        f"{f}: {k[0]} / {k[1]} / {len(k[2])} grasps" for f, k in zip(sys.argv[1:3], (key(a), key(b)))))' "$1" "$2"
}

stop_on_start_success() {   # <job exit code> <what>: exit 4 = a start_success stop -> <pool>/STOP, the rounds end (exit 1)
  if [ "$1" = 4 ]; then
    echo "$(date '+%F %T') $2: start_success stop (a placed start satisfied AutoMate's rule); the rounds stop" >> "$P/STOP"
    olog "$2: start_success stop (job exit 4); $P/STOP written, the rounds end"
    exit 1
  fi
}

best=-1; worse=0
for R in $(seq 1 "$r1"); do
  D=$P/r$R
  [ -e "$P/STOP" ] && { olog "$P/STOP: rounds end before round $R"; exit 0; }
  if [ ! -e "$D/DONE" ]; then
    mkdir -p "$D"
    if [ ! -e "$D/model.pt" ]; then
      wait_learner "$R"
      cp "$LOUT/last.pt" "$D/model.pt.tmp" && mv "$D/model.pt.tmp" "$D/model.pt" || exit 1
    fi
    olog "round $R collects with $D/model.pt ($workers workers x $MEM / $CPU / $GPU GB; extra options: ${ATT_EXTRA:-none})"
    job "$D" --grasps "$GR" --ckpt "$D/model.pt" --mode collect --split train --pool "$P" --round "$R" --starts "$starts" \
      --torch-mib "$tmib" --max-retries 0 --min-step "$min_step" "${att_extra[@]}" >> "$LOG" 2>&1
    rc=$?
    stop_on_start_success "$rc" "round $R collection"
    [ -e "$D/DONE" ] || { olog "round $R has no DONE (job exit $rc); rounds stop"; exit 1; }
  fi
  for m in act_done wm; do
    O=$D/select_$m
    [ -e "$O/summary.json" ] && continue
    [ -e "$P/STOP" ] && { olog "$P/STOP: rounds end in round $R"; exit 0; }
    if [ "$m" = wm ]; then extra=("${att_extra[@]}"); else mapfile -t extra < <(act_done_options); fi
    job "$O" --grasps "$GR" --ckpt "$D/model.pt" --mode "$m" --split select --out "$O" --starts 4 \
      --torch-mib "$tmib" --max-retries 0 --min-step "$min_step" "${extra[@]}" >> "$LOG" 2>&1
    rc=$?
    stop_on_start_success "$rc" "round $R select test $m"
    [ "$rc" -eq 0 ] || olog "round $R select test $m: job exit $rc"
  done
  if [ -f "$D/select_act_done/job.json" ] && [ -f "$D/select_wm/job.json" ]; then
    why=$(paired "$D/select_act_done/job.json" "$D/select_wm/job.json" 2>&1) ||
      { olog "round $R: ${why:-the select tests are not paired}; rounds stop"; exit 1; }
  fi
  s=$(success_of "$D/select_wm/summary.json")
  olog "round $R done; select success (AutoMate's rule) with the world model ${s:-unknown}, action head alone (act_done) $(success_of "$D/select_act_done/summary.json")"
  [ -n "$s" ] || continue
  if "$CL_PY" -I -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$s" "$best"; then best=$s; worse=0
  else worse=$(( worse + 1 )); fi
  if [ "$worse" -ge 2 ]; then olog "two rounds without improvement; rounds end after round $R"; break; fi
done
exit 0
