#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_EVALUATION_PY="${SCRIPT_DIR}/../tools/evaluation/run_evaluation.py"
GIT_REVISION="$(git rev-parse --short HEAD)"

CLOCK_PERIOD=(20 15 10 8 7 6 5 4 3)
OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/01_dynamatic_target_clock_period_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF > "${OUTPUT_DIR}/README.txt"
Experiment 1: Dynamatic Target Clock Period
- git revision: ${GIT_REVISION}
- no synthesis, only simulation
- default LSQ size = (16,16)
- no pipelining
EOF

for CP in "${CLOCK_PERIOD[@]}"; do
    echo "Clock period: ${CP} ns"

    export LSQ_NUM_LDQ_ENTRIES=16
    export LSQ_NUM_STQ_ENTRIES=16
    export LSQ_PIPE_COMP_EN=0
    export LSQ_PIPE0_EN=0
    export LSQ_PIPE1_EN=0
    export LSQ_HEAD_LAG_EN=0

    "$RUN_EVALUATION_PY" --no-synth --clock-period "${CP}" -j 16 \
        --json "${OUTPUT_DIR}/${CP}ns.json"
done
