#!/bin/bash
# Queue F (milestone-1b diagnosis): rerun of simC3 (OOM-killed under its 14G cap in the one-shot penetration query),
# started only when our caps allow it (caps.sh qx_check).
D=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
source "$D/../caps.sh"
RD="bash $D/run_diag.sh"
S=scripts/automate/diag/sim_diag.py
O=/mnt/storage3/qingxin/automate/m1b/diag/sim/rerun
slot() { until qx_check diag 22G 400% 7 >/dev/null; do sleep 45; done; }   # caps.sh, not a process-name count
sleep 20; slot; $RD simF1 22G $S --out $O --ids 00141,00360,00755 --exp i,ii,teleport
echo "QUEUE F DONE $(date)" >> /mnt/storage3/qingxin/automate/m1b/diag/logs/groups.log
