#!/bin/bash
# Milestone-1b diagnosis runs on rs2, one group per tmux session (each step is one memory-capped Isaac process).
#   bash run_groups.sh A|B|C
G=$1
D=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
source "$D/../caps.sh"
RD="bash $D/run_diag.sh"
S=scripts/automate/diag/sim_diag.py
O=/mnt/storage3/qingxin/automate/m1b/diag/sim
case $G in
A) $RD simA1 14G $S --out $O --ids 00271 --exp i,i_open,ii,ii_open,teleport,ii_dp-2.0
   $RD simA2 14G $S --out $O --ids 00681 --exp i,i_open,ii,ii_open,teleport,ii_dp-6.5
   $RD simA3 14G $S --out $O --ids 01092 --exp i,i_open,ii,ii_open,teleport,ii_dp-5.5 ;;
B) $RD simB1 14G $S --out $O --ids 00855,00597 --exp i,i_slow,i_open,ii,ii_slow,ii_open,teleport ;;
C) $RD simC1 14G $S --out $O --ids 01026 --exp i,i_open,ii,ii_open,teleport,ii_dp-2.0
   $RD simC2 14G $S --out $O --ids 00175 --exp i,i_open,ii,teleport,ii_dp-1.5
   $RD simC3 14G $S --out $O --ids 00141,00360,00755 --exp i,ii,teleport
 ;;
E) slot() { until qx_check diag 22G 400% 7 >/dev/null; do sleep 45; done; }   # caps.sh, not a process-name count
   slot; $RD simE1 22G $S --out $O/rerun --ids 00681 --exp i,i_open,ii,ii_open,teleport,ii_dp-6.5
   slot; $RD simE2 22G $S --out $O/rerun --ids 01026 --exp i,i_open,ii,ii_open,teleport,ii_dp-2.0
   slot; $RD simE3 22G $S --out $O/rerun --ids 00855 --exp ii_shift0_0.534,ii_shift0_0.3,ii_shift0_-0.534
   slot; $RD batch_rerun_01125 24G scripts/automate/batch_assemblies.py --out /mnt/storage3/qingxin/automate/m1b/diag/batch_rerun --ids 01125 ;;
D) $RD simD1 22G $S --out $O --ids 01026 --exp i,i_open,ii,ii_open,teleport,ii_dp-2.0 ;;
esac
echo "GROUP $G DONE $(date)" >> /mnt/storage3/qingxin/automate/m1b/diag/logs/groups.log
# group D (added after simC1 was killed by its own 14G scope limit inside the penetration computation of 01026):
#   bash run_groups.sh D   -> 01026 again with a 22G cap
