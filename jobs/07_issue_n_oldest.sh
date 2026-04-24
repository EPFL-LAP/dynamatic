#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_EVALUATION_PY="${SCRIPT_DIR}/../tools/evaluation/run_evaluation.py"
GIT_REVISION="$(git rev-parse --short HEAD)"

LSQ_SIZE=20
NUM_OLDEST_LOADS_VALUES=(1 2 3 4 6 8 20)
PIPELINE_CONFIGS=(
	"headlag"
	"headlag_pipe0"
)

export SYNTHESIS_CLOCK_PERIOD_NS="2.5"
export LSQ_NO_BYPASS=1
export LSQ_NUM_LDQ_ENTRIES=$LSQ_SIZE
export LSQ_NUM_STQ_ENTRIES=$LSQ_SIZE

OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/07_issue_n_oldest_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF >"${OUTPUT_DIR}/README.txt"
Experiment 7: Issue N oldest loads
- synthesis target clock period: ${SYNTHESIS_CLOCK_PERIOD_NS} ns
- pipeline configurations: ${PIPELINE_CONFIGS[*]}
- LSQ size: ${LSQ_SIZE} entries
- LSQ_ISSUE_OLDEST_LOADS values: ${NUM_OLDEST_LOADS_VALUES[*]}
- git revision: ${GIT_REVISION}
EOF

for PIPELINE_CONFIG in "${PIPELINE_CONFIGS[@]}"; do
	OUTPUT_SUBDIR="${OUTPUT_DIR}/${PIPELINE_CONFIG}"
	echo "Output directory for pipeline configuration = ${PIPELINE_CONFIG}: ${OUTPUT_SUBDIR}"
	mkdir -p "${OUTPUT_SUBDIR}"

	export LSQ_PIPE_COMP_EN=0
	export LSQ_PIPE0_EN=0
	export LSQ_PIPE1_EN=0
	export LSQ_HEAD_LAG_EN=0
	if [[ "$PIPELINE_CONFIG" == *"pipecomp"* ]]; then
		export LSQ_PIPE_COMP_EN=1
	fi
	if [[ "$PIPELINE_CONFIG" == *"pipe0"* ]]; then
		export LSQ_PIPE0_EN=1
	fi
	if [[ "$PIPELINE_CONFIG" == *"pipe1"* ]]; then
		export LSQ_PIPE1_EN=1
	fi
	if [[ "$PIPELINE_CONFIG" == *"headlag"* ]]; then
		export LSQ_HEAD_LAG_EN=1
	fi

	for NUM_OLDEST_LOADS in "${NUM_OLDEST_LOADS_VALUES[@]}"; do
		export LSQ_ISSUE_OLDEST_LOADS=$NUM_OLDEST_LOADS

		echo "Running evaluation with pipeline configuration = ${PIPELINE_CONFIG}; LSQ_ISSUE_OLDEST_LOADS = ${NUM_OLDEST_LOADS}"
		"$RUN_EVALUATION_PY" --no-synth -j 16 \
			--json "${OUTPUT_SUBDIR}/oldest_loads_${NUM_OLDEST_LOADS}.json"
	done
done
