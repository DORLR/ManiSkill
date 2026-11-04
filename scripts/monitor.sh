#!/bin/bash
# Sample container CPU (cgroup), memory and GPU once per second for N seconds.
# usage: bash scripts/monitor.sh [seconds] [outfile]
N=${1:-20}; OUT=${2:-/dev/stdout}
prev=$(sed -n 1p /sys/fs/cgroup/cpu.stat | cut -d" " -f2)
for i in $(seq 1 $N); do
  sleep 1
  cur=$(sed -n 1p /sys/fs/cgroup/cpu.stat | cut -d" " -f2)
  g=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits)
  echo "$(date +%T) cores=$(awk -v d=$((cur-prev)) 'BEGIN{printf "%.2f", d/1e6}') mem_GB=$(( $(cat /sys/fs/cgroup/memory.current)/1000000000 )) gpu=$g" >> $OUT
  prev=$cur
done
