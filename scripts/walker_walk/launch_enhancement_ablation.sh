#!/usr/bin/env bash
set -euo pipefail

# General launcher for Walker-Walk enhancement ablation
# Default: 2R2N1A experts (teacher_betas=[1, 1, 0, 0, -1]), FB=3000, reversed order

CONTAINER_NAME="mixture_pbrl_container"

# If running on the host machine, forward execution into the docker container
if [ ! -f /.dockerenv ]; then
    echo "Running on host. Forwarding execution into '$CONTAINER_NAME'..."
    exec docker exec -w /workspace "$CONTAINER_NAME" /workspace/scripts/walker_walk/$(basename "$0") "$@"
fi

LOG_FILE="/workspace/exp_pebble_mixture_ablation/walker_walk/ablation_2R2N1A_fb3000.log"
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================================================"
echo "Starting Walker-Walk Enhancement Ablation (2R2N1A, FB=3000)"
echo "Date: $(date)"
echo "Log: $LOG_FILE"
echo "========================================================================"

python scripts/walker_walk/run_enhancement_ablation.py \
    --teacher-betas 1 1 0 0 -1 \
    --variants tanh_maxn tanh raw \
    --max-feedback 3000 \
    --seeds 12345 23451 34512 45123 51234 \
    --skip-existing \
    "$@" \
    2>&1 | tee -a "$LOG_FILE"
