#!/bin/bash
# AutoDL backend of caps.sh / collect_lib.sh (sourced at the end of both when /root/autodl-tmp exists). The AutoDL
# container is ours alone and has no systemd: the functions below keep the interfaces of caps.sh / collect_lib.sh
# (same arguments, output lines, exit codes 0 / 99, unit names, QX_UNIT_FILE / QX_REGISTRY, marker folders) so that
# online_automate.sh, online_job.sh and online_worker.sh run unchanged. Differences from rs2:
#   - no systemd scope: a "unit" is a process group started with setsid; $QX_RUN_DIR/units/<unit>.pid holds its pid
#     while it runs (qx_unit_active, qx_units read it); no per-unit MemoryMax / CPUQuota (the container's own limits
#     apply: 124 GiB, 32 CPUs); the declared sizes are still recorded and summed (qx_budget).
#   - GPU jobs see one GPU: CUDA_VISIBLE_DEVICES=$QX_WORKER_GPU (default 1, as on rs2, so the collection switches
#     kit_gpu 1 / forbidden_gpu 0 stay those of plan.json); the learner runs on GPU 0 (train_autodl.sh).
#   - qx_check: the worker GPU keeps >= 1 GiB free after the new declaration; the container keeps >= 8 GB free memory
#     after the new MemoryMax. No GPU-0 or co-tenant rules (nobody else is here).
#   - the watcher is autodl_watch.py: only rule 4 of qx_watch.py (heartbeat hang -> mark the claim, write .hung, kill
#     the worker's process group), reusing qx_watch.check_heartbeats.
QX_RUN_DIR=/root/autodl-tmp/run
QX_AUTODL=1
QX_WORKER_GPU=${QX_WORKER_GPU:-1}
QX_AUTODL_START_GAP_S=${QX_AUTODL_START_GAP_S:-30}
mkdir -p "$QX_RUN_DIR/units" "$QX_RUN_DIR/qx" 2>/dev/null

qx_pid_start() {   # <pid> -> its start time (field 22 of /proc/<pid>/stat); non-zero when the pid is gone
  local st
  st=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
  st=${st##*) }; set -- $st; echo "${20}"
}

qx_unit_active() {  # <unit, with or without .scope>: the process its pid file names still runs (same start time)
  local u=${1%.scope} p s
  read -r p s < "$QX_RUN_DIR/units/$u.pid" 2>/dev/null || return 1
  [ -n "$p" ] && [ "$(qx_pid_start "$p")" = "$s" ]
}

qx_units() {
  local f u
  for f in "$QX_RUN_DIR"/units/*.pid; do
    [ -e "$f" ] || continue
    u=$(basename "$f" .pid)
    qx_unit_active "$u" && echo "$u.scope"
  done
  return 0
}

QX_AUTODL_MEM_CAP=${QX_AUTODL_MEM_CAP:-118000000000}   # sum of declared memory (workers + learner) <= 118 GB of 124 GiB

qx_mem_used_bytes() {   # container memory in use without reclaimable page cache: memory.current - inactive_file
  local cur ina
  cur=$(cat /sys/fs/cgroup/memory.current) && ina=$(awk '/^inactive_file /{print $2}' /sys/fs/cgroup/memory.stat) || return 1
  echo $(( cur - ina ))
}

qx_mem_free_bytes() {   # container memory.max - qx_mem_used_bytes
  local mx used
  mx=$(cat /sys/fs/cgroup/memory.max) && used=$(qx_mem_used_bytes) || return 1
  [ "$mx" = max ] && mx=$(awk '/MemTotal/{print $2 * 1024}' /proc/meminfo)
  echo $(( mx - used ))
}

qx_declared_sum() {     # declared memory of our running units (their $QX_RUN_DIR/qx/<unit>.gpu: "gpu mem cpu") in bytes
  local u f g m c b sum=0
  for u in $(qx_units); do
    f=$QX_RUN_DIR/qx/${u%.scope}.gpu
    read -r g m c < "$f" 2>/dev/null || continue
    b=$(qx_to_bytes "$m" 2>/dev/null) || b=0
    sum=$(( sum + b ))
  done
  echo $sum
}

qx_check() (
  set -o pipefail
  kind=$1 mem=$2 cpu=$3 gpu=$4
  b=$(qx_to_bytes "$mem") || { echo "refused: MemoryMax '$mem' unreadable"; exit 1; }
  [[ "$cpu" =~ ^[0-9]+%$ ]] || { echo "refused: CPUQuota '$cpu' unreadable"; exit 1; }
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "refused: GPU GB '$gpu' unreadable"; exit 1; }
  fb=$(qx_mem_free_bytes) || { echo "refused: container memory unreadable"; exit 1; }
  ds=$(qx_declared_sum)
  [ $(( ds + b )) -le $QX_AUTODL_MEM_CAP ] ||
    { echo "refused: declared memory of our units $ds B + $b B > $QX_AUTODL_MEM_CAP B"; exit 1; }
  [ "$kind" = watch ] || [ $(( fb - b )) -ge 8000000000 ] ||
    { echo "refused: container free memory $fb B (without page cache) - MemoryMax $b B < 8 GB"; exit 1; }
  if [ "$gpu" -gt 0 ]; then
    free=$(nvidia-smi -i "$QX_WORKER_GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') &&
      [[ "$free" =~ ^[0-9]+$ ]] || { echo "refused: nvidia-smi unreadable"; exit 1; }
    need=$(( gpu * 1000000000 / 1048576 + 1024 ))
    [ "$free" -ge "$need" ] || { echo "refused: GPU $QX_WORKER_GPU has $free MiB free < $gpu GB + 1 GiB"; exit 1; }
  fi
  echo "caps ok (AutoDL): container free $fb B, worker GPU $QX_WORKER_GPU${free:+ free $free MiB}"
)

qx_run() {
  local kind=$1 name=$2 mem=$3 cpu=$4 gpu_gb=$5; shift 5
  local lk msg unit cvd=$QX_WORKER_GPU now last pid rc
  mkdir -p "$QX_RUN_DIR/qx" "$QX_RUN_DIR/units" || { echo "refused: $QX_RUN_DIR not writable"; return 99; }
  exec {lk}>>"$QX_RUN_DIR/qx.lock"
  if ! flock -w 600 "$lk"; then echo "refused: start lock busy for 600 s"; exec {lk}>&-; return 99; fi
  if [[ "$gpu_gb" =~ ^[0-9]+$ ]] && [ "$gpu_gb" -gt 0 ]; then
    last=$(cat "$QX_RUN_DIR/.last_gpu_start" 2>/dev/null); [[ "$last" =~ ^[0-9]+$ ]] || last=0
    now=$(date +%s)
    [ $(( now - last )) -lt $QX_AUTODL_START_GAP_S ] && sleep $(( QX_AUTODL_START_GAP_S - (now - last) ))
  fi
  if ! msg=$(qx_check "$kind" "$mem" "$cpu" "$gpu_gb"); then echo "$msg"; exec {lk}>&-; return 99; fi
  echo "$msg"
  unit="qingxin-${kind}-$(echo -n "$name" | tr -c 'a-zA-Z0-9_.-' '_')-$(date +%s)"
  echo "unit $unit"
  [ -n "${QX_UNIT_FILE:-}" ] && echo "$unit" > "$QX_UNIT_FILE"
  [ -n "${QX_REGISTRY:-}" ] && printf '{"time": "%s", "unit": "%s", "kind": "%s", "memory_max": "%s", "cpu_quota": "%s", "gpu_gb": %s, "launcher_pid": %s}\n' \
    "$(date '+%F %T')" "$unit" "$kind" "$mem" "$cpu" "$gpu_gb" "$$" >> "$QX_REGISTRY"
  echo "$gpu_gb $mem $cpu" > "$QX_RUN_DIR/qx/$unit.gpu"
  [ "$gpu_gb" = "0" ] && cvd=""
  setsid env CUDA_VISIBLE_DEVICES=$cvd CUDA_DEVICE_ORDER=PCI_BUS_ID QX_GPU_GB="$gpu_gb" QX_UNIT="$unit" "$@" {lk}>&- &
  pid=$!
  echo "$pid $(qx_pid_start "$pid")" > "$QX_RUN_DIR/units/$unit.pid"
  [ "$gpu_gb" != "0" ] && date +%s > "$QX_RUN_DIR/.last_gpu_start"
  exec {lk}>&-
  wait $pid; rc=$?
  rm -f "$QX_RUN_DIR/qx/$unit.gpu" "$QX_RUN_DIR/units/$unit.pid"
  return $rc
}

qx_budget() {
  local u f sm=0 sc=0 sg=0 g m c mb
  printf '%-64s %8s %6s %8s\n' unit Memory CPU_% GPU_GB
  for u in $(qx_units); do
    f=$QX_RUN_DIR/qx/${u%.scope}.gpu
    read -r g m c < "$f" 2>/dev/null || { g=0; m=0G; c=0%; }
    printf '%-64s %8s %6s %8s\n' "$u" "$m" "${c%\%}" "$g"
    mb=$(qx_to_bytes "$m" 2>/dev/null) || mb=0
    sm=$(( sm + mb )); sc=$(( sc + ${c%\%} )); sg=$(( sg + g ))
  done
  echo "sum (AutoDL, declared): memory $sm B; CPU $sc %; GPU $sg GB; container free $(qx_mem_free_bytes) B"
}

# ---- collect_lib.sh parts
CL_PY=/root/autodl-tmp/envs/isaaclab/bin/python
CL_TMUX=""
CL_HF_HOME=/root/autodl-tmp/cache/hf
CL_TORCH_HOME=/root/autodl-tmp/cache/torch
CL_TRAIN_WEIGHTS="$CL_TORCH_HOME/hub/checkpoints/resnet34-b627a593.pth $CL_HF_HOME/hub/models--nvidia--segformer-b1-finetuned-ade-512-512"

cl_isaac_env() {
  mkdir -p /root/autodl-tmp/isaac/home /root/autodl-tmp/isaac/cache /root/autodl-tmp/isaac/tmp
  export HOME=/root/autodl-tmp/isaac/home XDG_CACHE_HOME=/root/autodl-tmp/isaac/cache
  export TMPDIR=/root/autodl-tmp/isaac/tmp OMNI_KIT_ACCEPT_EULA=YES NUMPY_MADVISE_HUGEPAGE=0
  export PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=$CL_TORCH_HOME
  # the container image ICD (libGLX_nvidia) cannot create a Vulkan instance headless; the EGL entry point can (2026-11-26)
  export VK_ICD_FILENAMES=/root/autodl-tmp/isaac/vk/nvidia_egl_icd.json
}

cl_train_env() {
  local code=$1
  export PYTHONPATH="$code/src${CL_EXTRA_SITE:+:$CL_EXTRA_SITE}" HF_HOME=$CL_HF_HOME TORCH_HOME=$CL_TORCH_HOME
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
  export TDMPC2_DIR=/root/autodl-tmp/code/ext/tdmpc2_orig/tdmpc2 TDMPC2_EXTRA=/root/autodl-tmp/code/ext/tdmpc2_extra
}

cl_static_budget() {    # N workers x each must fit the worker GPU (32 GiB card) and the container memory together
  local n=$1 mem=$2 cpu=$3 gpu=$4 b tot
  b=$(qx_to_bytes "$mem") && [[ "$cpu" =~ ^[0-9]+%$ ]] && [[ "$gpu" =~ ^[0-9]+$ ]] ||
    { echo "refused: resources '$mem $cpu $gpu' unreadable"; return 1; }
  tot=$(nvidia-smi -i "$QX_WORKER_GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
  if [ $(( n * gpu * 1000000000 / 1048576 )) -gt "${tot:-0}" ] || [ $(( n * b )) -gt $QX_AUTODL_MEM_CAP ]; then
    echo "refused: $n workers x ($mem, $gpu GB) exceed GPU $QX_WORKER_GPU (${tot:-?} MiB) or $QX_AUTODL_MEM_CAP B"; return 1
  fi
}

cl_watch_ok() {
  local m
  m=$(stat -c %Y "$QX_RUN_DIR/watch.alive" 2>/dev/null) || return 1
  [ $(( $(date +%s) - m )) -le 180 ]
}

cl_watch_code() { return 0; }

cl_watch_ensure() {   # registers <dir>; starts autodl_watch.py (screen session qingxin-watch) if none is alive
  local dir=$1 code=$2 key i
  mkdir -p "$QX_RUN_DIR/watch"
  key=$(echo -n "$dir" | sha256sum | cut -c1-12)
  echo "$dir" > "$QX_RUN_DIR/watch/$key.dir"
  cl_watch_ok && return 0
  screen -ls 2>/dev/null | grep -q '[.]qingxin-watch[[:space:]]' ||
    screen -dmS qingxin-watch bash -c "$CL_PY -u $code/scripts/automate/autodl_watch.py $code >> $QX_RUN_DIR/watch.log 2>&1"
  for i in $(seq 1 30); do cl_watch_ok && { echo "watcher running (screen qingxin-watch, log $QX_RUN_DIR/watch.log)"; return 0; }; sleep 2; done
  echo "WARNING: autodl_watch.py not alive after 60 s (log $QX_RUN_DIR/watch.log)"; return 1
}
