#!/bin/bash
# lib320.sh: sourced by the 320 px pipelines on the uoft cluster (gpuserver2, 8x RTX A5000 24 GB, 440 GB RAM,
# Ubuntu 20.04, systemd 245 with cgroup v1). Version of 2026-11-17 for this machine, after its review.
#
# The account is shared with other people: nothing here selects processes by a pattern. Every job of ours runs as a
# NAMED transient scope (qx320-...) registered in $Q/run/qx320/<unit>.state ("<gpus|-> <GPU MiB> <RAM GB>"; gpus is
# one GPU or a comma list such as 3,4,5,6,7 for a data-parallel job, GPU MiB is per GPU) and is stopped only by that name; stopping a scope ends all its processes (DataLoader workers, spawned test workers).
# cgroup v1: systemd's MemoryMax= is NOT enforced for user scopes here (measured: a 1 GB scope allocated 2 GB). So
# every job has a guard: every 5 s it adds up the PSS of the scope's processes (shared pages counted once) and their
# GPU memory, and stops the scope above its RAM cap or its declared GPU memory. watchdog320.sh applies the same caps,
# in case the launching shell is gone.
# One budget check before every start, under a lock: the caps of all our live jobs together <= 352 GB (4/5 of the
# RAM, the researcher's limit); on each chosen GPU, free memory minus what our live jobs there declared but do not use
# yet >= the new job's declaration + a margin; low memory pressure (/proc/pressure/memory, machine-wide); one training
# of ours per GPU; at least 5 minutes since the last start of ours.
source /mnt/vol1/dorlr-qingxin/env.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID      # CUDA's numbering = nvidia-smi's
Q320_RAM_LIMIT_GB=352
Q320_GPU_MARGIN_MIB=2000
Q320_START_GAP_S=300
Q320_STATE=${Q:?}/run/qx320
Q320_LOG=$Q/logs/pipeline_320.log
# closed-loop tests: per worker a torch cap plus the CUDA context and the renderer (to be confirmed by the trial run)
EVAL_WORKERS=${EVAL_WORKERS:-6}; EVAL_TORCH_MIB=${EVAL_TORCH_MIB:-1000}; EVAL_EXTRA_MIB=${EVAL_EXTRA_MIB:-1500}; EVAL_RAM_GB=${EVAL_RAM_GB:-40}
mkdir -p "$Q320_STATE" "$Q/logs"

q320_log() { echo "[$(date '+%m-%d %H:%M')] $*" >> "$Q320_LOG"; }
q320_int() { [[ "$1" =~ ^[0-9]+$ ]]; }
q320_gpus_ok() {     # <gpus>: one GPU 0-7 or a comma list of distinct ones
  [[ "$1" =~ ^[0-7](,[0-7])*$ ]] && [ "$(tr ',' '\n' <<< "$1" | sort | uniq -d)" = "" ]; }
q320_has_gpu() { [[ ",$1," == *",$2,"* ]]; }    # <gpus> <gpu>
q320_watchdog_lists() {    # the RUNNING watchdog loaded a lib320.sh that understands GPU lists (an older one would stop
  # a list job): such a watchdog writes its PID to .watchdog_lists after sourcing this file; that PID must be the
  # service's main process now.
  local pid; pid=$(systemctl --user show -p MainPID --value qx320-watchdog.service 2>/dev/null)
  [ -n "$pid" ] && [ "$pid" != 0 ] && [ "$(cat "${Q320_STATE:?}/.watchdog_lists" 2>/dev/null)" = "$pid" ]; }
q320_pressure() { sed -n 's/^full avg10=\([0-9]*\).*/\1/p' /proc/pressure/memory; }    # whole per cent, machine-wide
q320_gpu_free() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${1:?}" | tr -d ' '; }
q320_fraction() {    # <MiB> <gpus> -> the torch memory fraction (of the smallest GPU listed); fails for a GPU that does not exist
  local t g x; t=""
  for g in ${2//,/ }; do
    x=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
    q320_int "$x" || return 1
    { [ -z "$t" ] || [ "$x" -lt "$t" ]; } && t=$x
  done
  q320_int "$t" && q320_int "${1:?}" || return 1
  awk -v m="$1" -v t="$t" 'BEGIN { printf "%.4f", m / t }'; }
q320_state() { systemctl --user show -p ActiveState --value "${1:?}.scope" 2>/dev/null; }     # empty if unknown
q320_procs() {       # <unit>: pids of the unit's scope (cgroup v1 systemd hierarchy, or unified)
  local cg
  cg=$(systemctl --user show -p ControlGroup --value "${1:?}.scope" 2>/dev/null)
  [ -n "$cg" ] || return 0
  cat "/sys/fs/cgroup/systemd$cg/cgroup.procs" 2>/dev/null || cat "/sys/fs/cgroup/unified$cg/cgroup.procs" 2>/dev/null; }
q320_pss_kb() {      # <unit>: summed PSS of its processes [kB]
  local p kb=0 v
  for p in $(q320_procs "$1"); do
    v=$(awk '/^Pss:/ {print $2}' "/proc/$p/smaps_rollup" 2>/dev/null); kb=$(( kb + ${v:-0} ))
  done
  echo "$kb"; }
q320_gpu_used() {    # <unit> <gpu>: GPU memory of its processes on that GPU [MiB]
  local pids mib=0 p m
  pids=" $(q320_procs "$1" | tr '\n' ' ') "
  while IFS=, read -r p m; do
    p=${p// /}; m=${m// /}
    [[ "$pids" == *" $p "* ]] && q320_int "$m" && mib=$(( mib + m ))
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -i "${2:?}" 2>/dev/null)
  echo "$mib"; }
q320_alive() {       # <state file>: dead only when systemd says inactive / failed / unknown and the file is older than 2 min
  local st; st=$(q320_state "$(basename "$1" .state)")
  case "$st" in inactive|failed|"") [ $(( $(date +%s) - $(stat -c %Y "$1") )) -lt 120 ] ;; *) return 0 ;; esac; }
q320_declared() {    # -> "<RAM GB>" of our live jobs; state files of dead jobs are removed
  local ram=0 f g m r
  for f in "${Q320_STATE:?}"/*.state; do
    [ -e "$f" ] || continue
    if q320_alive "$f"; then read -r g m r < "$f"; ram=$(( ram + r )); else rm -f -- "${f:?}"; fi
  done
  echo "$ram"; }
q320_reserved() {    # <gpu>: MiB our live jobs on that GPU declared but do not use yet
  local f g m r u res=0
  for f in "${Q320_STATE:?}"/*.state; do
    [ -e "$f" ] || continue
    read -r g m r < "$f"
    q320_has_gpu "$g" "$1" && q320_alive "$f" || continue
    u=$(q320_gpu_used "$(basename "$f" .state)" "$1"); [ "$m" -gt "$u" ] && res=$(( res + m - u ))
  done
  echo "$res"; }
q320_claim() {       # <unit> <ram_gb> <gpus|-> <gpu_mib per GPU> <train|test|cpu>: reserve; prints the reason and fails if it does not fit
  local unit=${1:?} ram=${2:?} gpus=${3:?} mib=${4:?} kind=${5:?}
  (
    flock 9
    p=$(q320_pressure); [ "${p:-100}" -lt 5 ] || { echo "memory pressure ${p:-unknown} %"; exit 1; }
    case "$(q320_state "$unit")" in inactive|failed|"") ;; *) echo "a unit named $unit exists"; exit 1 ;; esac
    if [ -e "$Q320_STATE/.last_start" ] && [ $(( $(date +%s) - $(stat -c %Y "$Q320_STATE/.last_start") )) -lt "$Q320_START_GAP_S" ]; then
      echo "a job of ours started less than $Q320_START_GAP_S s ago"; exit 1; fi
    uram=$(q320_declared)
    [ $(( uram + ram )) -le "$Q320_RAM_LIMIT_GB" ] || { echo "memory caps $uram + $ram GB > $Q320_RAM_LIMIT_GB GB"; exit 1; }
    if [ "$gpus" != - ]; then
      for gpu in ${gpus//,/ }; do
        free=$(q320_gpu_free "$gpu"); q320_int "$free" || { echo "GPU $gpu unknown"; exit 1; }
        res=$(q320_reserved "$gpu")
        [ $(( free - res )) -ge $(( mib + Q320_GPU_MARGIN_MIB )) ] || { echo "GPU $gpu: $free MiB free, $res reserved by our starting jobs, $mib + $Q320_GPU_MARGIN_MIB needed"; exit 1; }
        if [ "$kind" = train ]; then
          for f in "${Q320_STATE:?}"/qx320-train-*.state; do
            [ -e "$f" ] || continue
            if q320_alive "$f" && q320_has_gpu "$(cut -d' ' -f1 "$f")" "$gpu"; then echo "another training of ours is on GPU $gpu"; exit 1; fi
          done
        fi
      done
    fi
    echo "$gpus $mib $ram" > "${Q320_STATE:?}/${unit:?}.state"
    touch "$Q320_STATE/.last_start"
  ) 9> "${Q320_STATE:?}/.lock"; }
q320_check() {       # <unit> <ram_gb> <gpus|-> <gpu_mib per GPU>: stop the unit if it is over its RAM cap or its GPU declaration on any GPU
  local kb g gpu
  kb=$(q320_pss_kb "$1")
  if [ "$kb" -gt $(( $2 * 1048576 )) ]; then
    q320_log "$1 uses $(( kb / 1048576 )) GB > its cap $2 GB: stopping it"; systemctl --user stop "$1.scope"; return
  fi
  [ "$3" != - ] || return 0
  for gpu in ${3//,/ }; do
    g=$(q320_gpu_used "$1" "$gpu")
    if [ "$g" -gt "$4" ]; then q320_log "$1 uses $g MiB on GPU $gpu > its declared $4 MiB: stopping it"; systemctl --user stop "$1.scope"; return; fi
  done; }
q320_run() {         # <unit> <ram_gb> <gpus|-> <gpu_mib per GPU> <train|test|cpu> <torch fraction> <output file> <command...>; 90 = did not fit
  local unit=${1:?} ram=${2:?} gpu=${3:?} mib=${4:?} kind=${5:?} frac=${6:?} out=${7:?} why rc pid dev
  shift 7
  q320_int "$ram" && q320_int "$mib" && { [ "$gpu" = - ] || q320_gpus_ok "$gpu"; } || { q320_log "$unit not started: bad arguments"; return 1; }
  why=$(q320_claim "$unit" "$ram" "$gpu" "$mib" "$kind") || { [ -n "${Q320_QUIET:-}" ] || q320_log "$unit not started: $why"; return 90; }
  q320_log "$unit starts (memory cap $ram GB, GPU $gpu, $mib MiB declared)"
  dev=$gpu; [ "$gpu" = - ] && dev=""
  WM_GPU_MEM_FRACTION=$frac CUDA_VISIBLE_DEVICES=$dev env -u WM_CAM_SCALE \
    systemd-run --user --scope --collect -q --unit="$unit" nice -n 10 "$@" >> "$out" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do q320_check "$unit" "$ram" "$gpu" "$mib"; sleep 5; done
  wait "$pid"; rc=$?
  case "$(q320_state "$unit")" in inactive|failed|"") ;; *)    # the main process is gone but workers are left: end them
    q320_log "$unit: main process ended (code $rc) with processes left in the scope; stopping it"; systemctl --user stop "$unit.scope" ;; esac
  rm -f -- "${Q320_STATE:?}/${unit:?}.state"
  q320_log "$unit ended with code $rc"
  return $rc; }
q320_run_when_fits() {    # as q320_run, but waits up to 24 h for the budget (logs the reason once)
  local i rc
  for i in $(seq 1 720); do
    if [ "$i" -eq 1 ]; then q320_run "$@"; else Q320_QUIET=1 q320_run "$@"; fi
    rc=$?
    [ $rc -ne 90 ] && return $rc
    sleep 120
  done
  return 90; }
