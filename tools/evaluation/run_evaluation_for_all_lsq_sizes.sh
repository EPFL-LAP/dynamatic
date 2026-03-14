#!/usr/bin/env bash
# This script run evaluation for all LSQ sizes: 2, 4, 6, 8, 12, 16, 24
# Usage: ./run_evaluation_for_all_lsq_sizes.sh <results_dir>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR=$1
# minimum LSQ size of 4 required for some kernels
LSQ_SIZES=(4 6 8 10 12 16 20)

mkdir -p "$RESULTS_DIR"

for LSQ_SIZE in "${LSQ_SIZES[@]}"; do
    echo "Running evaluation for LSQ size: $LSQ_SIZE"
    export LSQ_NUM_LDQ_ENTRIES=$LSQ_SIZE
    export LSQ_NUM_STQ_ENTRIES=$LSQ_SIZE
    export LSQ_PIPE_COMP_EN=0
    export LSQ_PIPE0_EN=0
    export LSQ_PIPE1_EN=0
    export LSQ_HEAD_LAG_EN=0
    # Allow errors when running evaluation script:
    # Required due to failures if LSQ is too small to hold a full group.
    python3 "$SCRIPT_DIR/run_evaluation.py" \
        -j 24 --synth-lsqs \
        --json "$RESULTS_DIR/lsq_${LSQ_SIZE}_${LSQ_SIZE}.json"
done
