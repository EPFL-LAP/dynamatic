#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_EVALUATION_PY="${SCRIPT_DIR}/../tools/evaluation/run_evaluation.py"
GIT_REVISION="$(git rev-parse --short HEAD)"

OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/00_simple_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF >"${OUTPUT_DIR}/README.txt"
Experiment 0: Simple run with defaults
- git revision: ${GIT_REVISION}
EOF

OUTPUT_SUBDIR="${OUTPUT_DIR}/data"
mkdir -p "${OUTPUT_SUBDIR}"

echo "Output subdirectory: ${OUTPUT_SUBDIR}"

echo "Running evaluation..."

export SYNTHESIS_CLOCK_PERIOD_NS="2.5"

"$RUN_EVALUATION_PY" --synth-lsqs -j 16 --json "${OUTPUT_SUBDIR}/output.json"
