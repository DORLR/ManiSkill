#!/bin/bash
# Resource caps for every process we start on rs2 (source this file; it defines constants and functions only).
#
# Rule: every one of our processes on rs2 (collection, training, diagnosis, tests, audits, the watcher) is started
# through qx_run, so that it runs in a scope named qingxin-<kind>-* inside our slice qingxin.slice. Anything started
# otherwise is invisible to these sums. Our processes are identified by their scope (unit name and cgroup) and, where
# a single process is meant, by PID and process start time; never by a process-name pattern.
#
# Caps (all our jobs together; fixed here, the environment cannot change them):
# Caps raised on 2026-11-25 by the user's decision (plan B, for the AutoMate online rounds beside the learner;
# before: 125 GB / 3200 % / 48 GB):
#   RAM  sum_i M_i + M_new <= 180 GB = 180e9 bytes            M_i: MemoryMax of every active qingxin-* scope;
#        qingxin.slice itself has MemoryMax=180e9 bytes and MemorySwapMax=0 (a hard limit enforced by the kernel)
#   CPU  sum_i C_i + C_new <= 4800 %                           C_i: CPUQuota; qingxin.slice has CPUQuota=4800 %
#   GPU1 sum_u max(D_u, G_u) + G_new <= 72 GB = 72e9 bytes      D_u: GPU GB declared at the start of unit u (kept
#        in $QX_RUN_DIR/qx/<unit>.gpu while u runs), G_u: GPU-1 memory of the PIDs in u's cgroup from the full
#        nvidia-smi process table (compute AND graphics processes, nvidia-smi -q -d PIDS). Checked at every start
#        and every 60 s by qx_watch.py, which stops our newest GPU unit when the sum passes 72 GB.
#   GPU1 F - P - G_new >= 12 GB (12e9 bytes) stay free for other users   (only if G_new > 0); F = memory.free now,
#        P = sum_u max(0, D_u - G_u): what our running units declared but have not allocated yet (a Kit start takes
#        12 s to 5.5 min, starts are 90 s apart)
#   GPU0 each of our PIDs on GPU 0 may hold ONLY graphics contexts (nvidia-smi type G) of at most 50 MiB in total
#        (QX_GPU0_G_MAX_MIB; the user's decision 2026-11-22: Isaac's idle 8 MiB renderer context is accepted) and no
#        compute context (type C or C+G, or any other type); an entry without a readable type or memory counts as a
#        violation. Read per PID from `nvidia-smi -i <GPU 0> -q -d PIDS`. Refused at the start; qx_watch.py stops
#        such a unit; the process itself exits 7 (app.assert_off_forbidden_gpu)
#   memory pressure some avg10 <= 10 and full avg10 <= 1 in each of /proc/pressure/memory (machine),
#        user@<uid>.service/memory.pressure (the cgroup systemd-oomd watches; shared account) and
#        qingxin.slice/memory.pressure (ours)        [not for kind "watch": 256M, it must start under pressure]
#   free memory MemAvailable - M_new >= 20 GB       [not for kind "watch"]
#   kind "watch" (GPU GB 0) is checked against the RAM and CPU sums only: no GPU-0 / GPU-1 reading can refuse it (it
#   is what stops a GPU-0 violation; r4-ops-N2)
#   the caller's CUDA_VISIBLE_DEVICES is unset, empty or exactly 1
# Inputs are validated: MemoryMax ^[0-9]+[GM]$ (GiB / MiB, as systemd reads it), CPUQuota ^[0-9]+%$, GPU GB
# ^[0-9]+$. A failing systemctl or nvidia-smi query refuses; it never counts as 0.
#
#   qx_check <kind> <MemoryMax e.g. 18G> <CPUQuota e.g. 400%> <GPU GB, 0 = no GPU>
#       0 when the start is allowed, else prints "refused: ..." and returns 1.
#   qx_run <kind> <name> <MemoryMax> <CPUQuota> <GPU GB> <command...>
#       Takes the start lock (flock -w 600 $QX_RUN_DIR/qx.lock), waits until QX_START_GAP_S have passed since the
#       last GPU start (GPU starts only), runs qx_check, records the declared GPU GB, starts the command in scope
#       qingxin-<kind>-<name>-<unix time> (slice qingxin.slice, MemoryMax, MemorySwapMax=0, CPUQuota, --collect;
#       CUDA_VISIBLE_DEVICES=1 and CUDA_DEVICE_ORDER=PCI_BUS_ID for a GPU job, CUDA_VISIBLE_DEVICES empty for a CPU
#       job; QX_GPU_GB=<GPU GB declared> for the job itself, which caps its own allocator at it, r4-data-05) and
#       releases the lock only once the scope is active. Prints "unit <name>" first; if QX_UNIT_FILE is
#       set the unit name is written there, if QX_REGISTRY is set one JSON line per start is appended there.
#       Returns the command's exit code, or 99 when the caps (or the lock) refuse the start.
#   qx_run_retry <same arguments as qx_run>
#       qx_run, started again every 60 s while it returns 99 (the caps or the lock refused the start; r9-ops-01),
#       for up to QX_RETRY_MAX_S seconds (default 3600) after the first try. Any other exit code is returned at once.
#       Each retry prints one line "caps refused <name> (exit 99); retry in 60 s ..." to stdout and to stderr; after
#       QX_RETRY_MAX_S it prints a "giving up" line and returns 99. The start lock is not held while it waits.
#       Used for one-shot starts (cl_prep, cl_train_preflight, the smoke's checker, prep_automate_data.sh steps,
#       the parallel_batch.sh listing, draws_check.sh, train_automate.sh); collect_worker.sh and run_on_rs2.sh keep
#       plain qx_run, because their loops count 99 themselves.
#   qx_budget   prints our active units with MemoryMax, CPUQuota, declared and measured GPU-1 memory, and the sums.
if [[ "$(declare -p QX_CAPS_LOADED 2>/dev/null)" != "declare -r"* ]]; then   # once per shell; the environment cannot preset them
  readonly QX_CAPS_LOADED=1
  readonly QX_CAP_MEM_BYTES=180000000000      # 180 GB (was 125 GB; raised 2026-11-25, the user's plan B)
  readonly QX_CAP_CPU_PCT=4800                # 48 threads (was 32)
  readonly QX_CAP_GPU_OURS_MIB=68664          # 72e9 bytes / 2^20 (was 48e9)
  readonly QX_GPU_FREE_MIN_MIB=11444          # 12e9 bytes / 2^20
  readonly QX_MEMAVAIL_MIN_BYTES=20000000000
  readonly QX_PSI_SOME_MAX=10 QX_PSI_FULL_MAX=1
  readonly QX_START_GAP_S=90
  readonly QX_SLICE=qingxin.slice
  readonly QX_GPU0_G_MAX_MIB=50
fi
# Paths: plain variables set here (not read from the environment), so that the CPU test can point them at fake files
# after sourcing.
QX_RUN_DIR=/mnt/storage3/qingxin/run
QX_PROC=/proc
QX_CGROOT=/sys/fs/cgroup
QX_USER_SERVICE=/user.slice/user-$(id -u).slice/user@$(id -u).service

qx_units() {      # our active scopes, one per line; non-zero if systemctl fails
  ( set -o pipefail
    systemctl --user list-units --type=scope --state=active --no-legend --plain 'qingxin-*' | awk '{print $1}' )
}

qx_to_bytes() {   # 18G / 6144M -> bytes (GiB / MiB, as systemd reads MemoryMax); anything else -> non-zero
  local v=$1
  case $v in
    *G) [[ ${v%G} =~ ^[0-9]+$ ]] && echo $(( ${v%G} * 1073741824 )) ;;
    *M) [[ ${v%M} =~ ^[0-9]+$ ]] && echo $(( ${v%M} * 1048576 )) ;;
    *) return 1 ;;
  esac
}

qx_quota_pct() {  # systemd CPUQuotaPerSecUSec (4s, 500ms, 1.500000s, 1min 30s, infinity) -> percent, -1 for infinity
  awk -v v="$1" 'BEGIN { if (v == "infinity" || v == "") { print -1; exit }
    n = split(v, part, " "); s = 0
    for (i = 1; i <= n; i++) { x = part[i]; t = x + 0
      if (x ~ /us$/) t /= 1000000; else if (x ~ /ms$/) t /= 1000; else if (x ~ /min$/) t *= 60; else if (x ~ /h$/) t *= 3600
      s += t }
    printf "%d\n", s * 100 + 0.5 }'
}

qx_gpu_uuid() {   # qx_gpu_uuid <index> -> UUID
  ( set -o pipefail
    env -u CUDA_VISIBLE_DEVICES nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1 == i {print $2}' )
}

qx_parse_pids() {   # stdin: nvidia-smi -q -d PIDS text -> "pid MiB" per process (C and G entries of one pid summed)
  awk -F': *' '/Process ID/ {pid = $2 + 0} /Used GPU Memory/ { if (pid) m[pid] += $2 + 0 }
    END { for (p in m) print p, m[p] }'
}

qx_parse_pid_types() {  # stdin: nvidia-smi -q -d PIDS text -> "pid types MiB" per pid: types of its entries joined
                        # by "," ("?" when an entry has no Type line), MiB summed ("?" when any entry has no number)
  awk -F': *' '
    /Process ID/ { pid = $2 + 0; if (pid) { n[pid]++; if (!(pid in ty)) ty[pid] = ""; cur = pid } next }
    /^[ \t]*Type[ \t]*:/ { if (cur) { t = $2; gsub(/[ \t]/, "", t); if (t == "") t = "?"
                                       ty[cur] = (ty[cur] == "" ? t : ty[cur] "," t); nt[cur]++ } next }
    /Used GPU Memory/ { if (cur) { v = $2; if (v ~ /^[0-9.]+ *GiB/) m[cur] += 1024 * v; else if (v ~ /^[0-9.]+ *MiB/) m[cur] += v + 0
                                   else bad[cur] = 1 } next }
    END { for (p in ty) { t = ty[p]; if (nt[p] < n[p]) t = (t == "" ? "?" : t ",?")
                          print p, t, ((p in bad) ? "?" : m[p] + 0) } }'
}

qx_gpu0_violations() {  # stdin: "pid types MiB" rows of GPU 0; $1: our pids -> "pid(types MiB)" per violating pid
  awk -v p="$1" -v lim="$QX_GPU0_G_MAX_MIB" 'BEGIN { n = split(p, a, " "); for (i = 1; i <= n; i++) o[a[i]] = 1 }
    ($1 in o) { k = split($2, t, ","); ok = 1
      for (i = 1; i <= k; i++) if (t[i] != "G") ok = 0
      if ($3 == "?" || $3 + 0 > lim) ok = 0
      if (!ok) printf "%s(%s %s MiB) ", $1, $2, $3 }'
}

qx_gpu_pid_mib() {  # qx_gpu_pid_mib <uuid> -> "pid MiB" per process on that GPU (compute and graphics)
  ( set -o pipefail
    env -u CUDA_VISIBLE_DEVICES nvidia-smi -i "$1" -q -d PIDS | qx_parse_pids )
}

qx_psi() {          # qx_psi <memory.pressure file> -> "some full" (avg10)
  awk '/^some/ {split($2, a, "="); s = a[2]} /^full/ {split($2, a, "="); f = a[2]}
    END { if (s == "" || f == "") exit 1; print s, f }' "$1"
}

qx_unit_pids() {    # pids in a unit's cgroup (space separated)
  local cg
  cg=$(systemctl --user show -p ControlGroup --value "$1") || return 1
  if [ -n "$cg" ] && [ -f "$QX_CGROOT$cg/cgroup.procs" ]; then tr '\n' ' ' < "$QX_CGROOT$cg/cgroup.procs"; fi
  return 0
}

qx_gpu_mib_of() {   # qx_gpu_mib_of "<pid list>" "<pid MiB table>" -> MiB sum
  awk -v p="$1" 'BEGIN { n = split(p, a, " "); for (i = 1; i <= n; i++) ours[a[i]] = 1 }
    ($1 in ours) { s += $2 } END { print s + 0 }' <<< "$2"
}

qx_declared_mib() { # declared GPU GB of a unit (its reservation file) -> MiB
  local f=$QX_RUN_DIR/qx/${1%.scope}.gpu g=0        # qx_run writes qx/<name>.gpu; units are listed as <name>.scope
  [ -f "$f" ] && g=$(cat "$f")
  [[ $g =~ ^[0-9]+$ ]] || g=0
  echo $(( g * 1000000000 / 1048576 ))
}

qx_check() (        # a subshell: pipefail stays inside
  set -o pipefail
  local kind=$1 mem=$2 cpu=$3 gpu_gb=$4
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "1" ]; then
    echo "refused: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (GPU 1 only; GPU 0 is forbidden)"; return 1
  fi
  [[ "$kind" =~ ^[a-z0-9]+$ ]] || { echo "refused: kind '$kind' must be [a-z0-9]+"; return 1; }
  [[ "$mem" =~ ^[0-9]+[GM]$ ]] || { echo "refused: MemoryMax '$mem' must be <integer>G or <integer>M"; return 1; }
  [[ "$cpu" =~ ^[0-9]+%$ ]] || { echo "refused: CPUQuota '$cpu' must be <integer>%"; return 1; }
  [[ "$gpu_gb" =~ ^[0-9]+$ ]] || { echo "refused: GPU GB '$gpu_gb' must be an integer"; return 1; }
  local mem_new cpu_new=${cpu%\%} units u v mem_sum=0 cpu_sum=0 allpids="" gpu_rows=()
  mem_new=$(qx_to_bytes "$mem") || { echo "refused: MemoryMax '$mem' unreadable"; return 1; }
  [ "$mem_new" -gt 0 ] && [ "$cpu_new" -gt 0 ] || { echo "refused: MemoryMax and CPUQuota must be > 0"; return 1; }
  units=$(qx_units) || { echo "refused: systemctl list-units failed"; return 1; }
  for u in $units; do
    v=$(systemctl --user show -p MemoryMax --value "$u") || { echo "refused: systemctl show $u failed"; return 1; }
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "refused: our scope $u has no MemoryMax ($v)"; return 1; }
    mem_sum=$(( mem_sum + v ))
    v=$(systemctl --user show -p CPUQuotaPerSecUSec --value "$u") || { echo "refused: systemctl show $u failed"; return 1; }
    v=$(qx_quota_pct "$v")
    [ "$v" -lt 0 ] && { echo "refused: our scope $u has no CPUQuota"; return 1; }
    cpu_sum=$(( cpu_sum + v ))
    v=$(qx_unit_pids "$u") || { echo "refused: cgroup of $u unreadable"; return 1; }
    allpids="$allpids $v"
    gpu_rows+=("$u|$v")
  done
  if [ $(( mem_sum + mem_new )) -gt $QX_CAP_MEM_BYTES ]; then
    echo "refused: our MemoryMax sum ${mem_sum} B + $mem ($mem_new B) > ${QX_CAP_MEM_BYTES} B (180 GB)"; return 1
  fi
  if [ $(( cpu_sum + cpu_new )) -gt $QX_CAP_CPU_PCT ]; then
    echo "refused: our CPUQuota sum ${cpu_sum} % + ${cpu_new} % > ${QX_CAP_CPU_PCT} %"; return 1
  fi
  # the watcher (256M, 5 %, no GPU) must be able to start exactly when it is needed most: under memory pressure, while
  # one of our pids breaks the GPU-0 rule (it stops that unit) or while nvidia-smi fails for a moment (r4-ops-N2)
  if [ "$kind" = watch ]; then
    if [ "$gpu_gb" != 0 ]; then echo "refused: kind watch declares no GPU memory"; return 1; fi
    echo "caps ok (watch): ours MemoryMax ${mem_sum} B + $mem, CPUQuota ${cpu_sum} % + ${cpu_new} %"
    return 0
  fi
  # GPU 0: only a graphics context of <= 50 MiB per pid of ours, no compute context (header); GPU 1: our sum per unit
  # as max(declared, measured), and what is declared but not yet allocated (pend) counts against the free memory
  local uuid0 uuid1 tab0 tab1 ours0 ours1=0 pend=0 row pids meas decl free_mib new_mib
  uuid0=$(qx_gpu_uuid 0) && uuid1=$(qx_gpu_uuid 1) || { echo "refused: nvidia-smi --query-gpu failed"; return 1; }
  [ -n "$uuid0" ] && [ -n "$uuid1" ] || { echo "refused: GPU 0 / GPU 1 not found"; return 1; }
  tab0=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi -i "$uuid0" -q -d PIDS | qx_parse_pid_types) &&
    tab1=$(qx_gpu_pid_mib "$uuid1") || { echo "refused: nvidia-smi -q -d PIDS failed"; return 1; }
  ours0=$(qx_gpu0_violations "$allpids" <<< "$tab0")
  [ -z "$ours0" ] || { echo "refused: our PIDs on GPU 0 beyond a graphics context of <= ${QX_GPU0_G_MAX_MIB} MiB: ${ours0}(forbidden)"; return 1; }
  for row in "${gpu_rows[@]}"; do
    u=${row%%|*}; pids=${row#*|}
    meas=$(qx_gpu_mib_of "$pids" "$tab1"); decl=$(qx_declared_mib "$u")
    ours1=$(( ours1 + (meas > decl ? meas : decl) ))
    pend=$(( pend + (decl > meas ? decl - meas : 0) ))
  done
  if [ "$gpu_gb" -gt 0 ]; then
    new_mib=$(( gpu_gb * 1000000000 / 1048576 ))
    free_mib=$(env -u CUDA_VISIBLE_DEVICES nvidia-smi --id="$uuid1" --query-gpu=memory.free --format=csv,noheader,nounits) ||
      { echo "refused: nvidia-smi memory.free failed"; return 1; }
    [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "refused: GPU 1 free memory unreadable ($free_mib)"; return 1; }
    if [ $(( ours1 + new_mib )) -gt $QX_CAP_GPU_OURS_MIB ]; then
      echo "refused: our GPU-1 memory ${ours1} MiB + ${gpu_gb} GB > 72 GB (${QX_CAP_GPU_OURS_MIB} MiB)"; return 1
    fi
    if [ $(( free_mib - pend - new_mib )) -lt $QX_GPU_FREE_MIN_MIB ]; then
      echo "refused: GPU 1 has ${free_mib} MiB free, ${pend} MiB of it declared by our starting units; after ${gpu_gb} GB" \
           "fewer than 12 GB would stay free"; return 1
    fi
  fi
  # memory pressure (machine, the shared user service that systemd-oomd watches, our slice) and free memory
  local f sf name psi_txt="" avail
  for f in "machine:$QX_PROC/pressure/memory" "user-service:$QX_CGROOT$QX_USER_SERVICE/memory.pressure" \
           "ours:$QX_CGROOT$QX_USER_SERVICE/$QX_SLICE/memory.pressure"; do
    name=${f%%:*}; f=${f#*:}
    if [ ! -f "$f" ]; then
      [ "$name" = ours ] && continue                         # our slice's cgroup exists only while a scope of ours runs
      echo "refused: $f missing"; return 1
    fi
    sf=$(qx_psi "$f") || { echo "refused: $f unreadable"; return 1; }
    psi_txt="$psi_txt $name ${sf/ //}"
    if awk -v s="${sf% *}" -v f="${sf#* }" -v S=$QX_PSI_SOME_MAX -v F=$QX_PSI_FULL_MAX 'BEGIN { exit !(s > S || f > F) }'; then
      echo "refused: memory pressure ($name) some/full avg10 = ${sf/ //} (limits $QX_PSI_SOME_MAX / $QX_PSI_FULL_MAX)"; return 1
    fi
  done
  avail=$(awk '/^MemAvailable:/ {printf "%d\n", $2 * 1024}' "$QX_PROC/meminfo")
  [[ "$avail" =~ ^[0-9]+$ ]] || { echo "refused: MemAvailable unreadable"; return 1; }
  if [ $(( avail - mem_new )) -lt $QX_MEMAVAIL_MIN_BYTES ]; then
    echo "refused: MemAvailable ${avail} B - $mem < 20 GB"; return 1
  fi
  echo "caps ok: ours MemoryMax ${mem_sum} B + $mem, CPUQuota ${cpu_sum} % + ${cpu_new} %, GPU-1 ${ours1} MiB + ${gpu_gb} GB;" \
       "pressure${psi_txt}; MemAvailable ${avail} B"
  return 0
)

qx_slice_caps() {  # the hard limits of our slice (a runtime drop-in of qingxin.slice only)
  systemctl --user set-property --runtime "$QX_SLICE" MemoryMax=$QX_CAP_MEM_BYTES MemorySwapMax=0 \
    CPUQuota=${QX_CAP_CPU_PCT}% >/dev/null 2>&1
}

qx_run() {
  local kind=$1 name=$2 mem=$3 cpu=$4 gpu_gb=$5; shift 5
  local lk msg unit cvd=1 now last pid rc i f u m
  mkdir -p "$QX_RUN_DIR/qx" || { echo "refused: $QX_RUN_DIR not writable"; return 99; }
  exec {lk}>>"$QX_RUN_DIR/qx.lock"
  if ! flock -w 600 "$lk"; then echo "refused: start lock busy for 600 s"; exec {lk}>&-; return 99; fi
  for f in "$QX_RUN_DIR"/qx/*.gpu; do              # reservations of units that are gone (older than 5 min)
    [ -e "$f" ] || continue
    u=$(basename "$f" .gpu)
    systemctl --user is-active --quiet "$u.scope" && continue   # the file holds the scope name without its suffix
    m=$(stat -c %Y "$f" 2>/dev/null) || continue               # removed meanwhile by its own qx_run
    [ $(( $(date +%s) - m )) -lt 300 ] || rm -f "$f"
  done
  if [[ "$gpu_gb" =~ ^[0-9]+$ ]] && [ "$gpu_gb" -gt 0 ]; then
    last=$(cat "$QX_RUN_DIR/.last_gpu_start" 2>/dev/null); [[ "$last" =~ ^[0-9]+$ ]] || last=0
    now=$(date +%s)
    [ $(( now - last )) -lt $QX_START_GAP_S ] && sleep $(( QX_START_GAP_S - (now - last) ))
  fi
  if ! msg=$(qx_check "$kind" "$mem" "$cpu" "$gpu_gb"); then
    echo "$msg"; exec {lk}>&-; return 99
  fi
  echo "$msg"
  if ! qx_slice_caps; then echo "refused: cannot set the caps of $QX_SLICE"; exec {lk}>&-; return 99; fi
  unit="qingxin-${kind}-$(echo -n "$name" | tr -c 'a-zA-Z0-9_.-' '_')-$(date +%s)"
  echo "unit $unit"
  [ -n "${QX_UNIT_FILE:-}" ] && echo "$unit" > "$QX_UNIT_FILE"
  [ -n "${QX_REGISTRY:-}" ] && printf '{"time": "%s", "unit": "%s", "kind": "%s", "memory_max": "%s", "cpu_quota": "%s", "gpu_gb": %s, "launcher_pid": %s}\n' \
    "$(date '+%F %T')" "$unit" "$kind" "$mem" "$cpu" "$gpu_gb" "$$" >> "$QX_REGISTRY"
  echo "$gpu_gb" > "$QX_RUN_DIR/qx/$unit.gpu"
  [ "$gpu_gb" = "0" ] && cvd=""
  systemd-run --user --scope -q --collect --slice="$QX_SLICE" --unit="$unit" -p MemoryMax="$mem" -p MemorySwapMax=0 \
    -p CPUQuota="$cpu" env CUDA_VISIBLE_DEVICES=$cvd CUDA_DEVICE_ORDER=PCI_BUS_ID QX_GPU_GB="$gpu_gb" "$@" {lk}>&- &
  pid=$!
  for i in $(seq 1 120); do                         # hold the lock until the scope is active (or the command ended)
    systemctl --user is-active --quiet "$unit.scope" && break   # systemd-run --scope --unit=<name> creates <name>.scope
    kill -0 $pid 2>/dev/null || break
    sleep 0.5
  done
  [ "$gpu_gb" != "0" ] && date +%s > "$QX_RUN_DIR/.last_gpu_start"
  exec {lk}>&-
  wait $pid; rc=$?
  rm -f "$QX_RUN_DIR/qx/$unit.gpu"
  return $rc
}

qx_run_retry() {   # qx_run, retried every 60 s on exit 99 (a caps refusal) for up to QX_RETRY_MAX_S (header)
  local max=${QX_RETRY_MAX_S:-3600} t0 rc waited msg
  [[ "$max" =~ ^[0-9]+$ ]] || { echo "refused: QX_RETRY_MAX_S=$max is not a whole number of seconds"; return 99; }
  t0=$(date +%s)
  while true; do
    qx_run "$@"; rc=$?
    [ $rc -eq 99 ] || return $rc
    waited=$(( $(date +%s) - t0 ))
    if [ $waited -ge "$max" ]; then
      msg="caps refused $2 (exit 99) for ${waited} s; giving up (QX_RETRY_MAX_S=${max})"
      echo "$msg"; echo "$msg" >&2
      return 99
    fi
    msg="caps refused $2 (exit 99); retry in 60 s ($(date '+%F %T'), ${waited} s of QX_RETRY_MAX_S=${max} used)"
    echo "$msg"; echo "$msg" >&2
    sleep 60
  done
}

qx_budget() {      # our units: MemoryMax, CPUQuota, declared / measured GPU-1 memory, and the sums against the caps
  local u m c pids meas decl uuid1 tab1 sm=0 sc=0 sg=0
  uuid1=$(qx_gpu_uuid 1) && tab1=$(qx_gpu_pid_mib "$uuid1") || { echo "nvidia-smi failed"; return 1; }
  printf '%-64s %14s %6s %11s %11s\n' unit MemoryMax_B CPU_% GPUdecl_MiB GPUmeas_MiB
  for u in $(qx_units); do
    m=$(systemctl --user show -p MemoryMax --value "$u")
    c=$(qx_quota_pct "$(systemctl --user show -p CPUQuotaPerSecUSec --value "$u")")
    pids=$(qx_unit_pids "$u"); meas=$(qx_gpu_mib_of "$pids" "$tab1"); decl=$(qx_declared_mib "$u")
    printf '%-64s %14s %6s %11s %11s\n' "$u" "$m" "$c" "$decl" "$meas"
    [[ $m =~ ^[0-9]+$ ]] && sm=$(( sm + m ))
    sc=$(( sc + c )); sg=$(( sg + (meas > decl ? meas : decl) ))
  done
  echo "sum: MemoryMax $sm B of $QX_CAP_MEM_BYTES; CPUQuota $sc % of $QX_CAP_CPU_PCT; GPU-1 $sg MiB of $QX_CAP_GPU_OURS_MIB"
}
