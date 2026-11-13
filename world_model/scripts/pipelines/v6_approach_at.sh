#!/bin/bash
# The approach check on later checkpoints of a joint run, after v6_checks2.sh has copied them.
#   usage: v6_approach_at.sh <run> <tmux session of v6_checks2.sh> <steps ...>
set -uo pipefail
RUN=$1; SESSION=$2; shift 2
Q=/mnt/storage2/qingxin
for K in "$@"; do
    until [ -f $Q/runs/checks_v6/${RUN}_$K.pt ]; do
        tmux has-session -t $SESSION 2>/dev/null || exit 1
        sleep 120
    done
    sleep 30
    GPU=${GPU:-0} bash $Q/code/ManiSkill/world_model/scripts/pipelines/v6_approach.sh $Q/runs/checks_v6/${RUN}_$K.pt ${RUN}_$K
done
