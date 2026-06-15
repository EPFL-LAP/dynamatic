#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_EVALUATION_PY="${SCRIPT_DIR}/../tools/evaluation/run_evaluation.py"
GIT_REVISION="$(git rev-parse --short HEAD)"

LSQ_SIZES=(
	12
	16
)

# k, log2(m)
BLOOM_FILTER_HASH_COUNTS_WIDTHS=(
	"1 2"
	"1 3"
	"2 4"
	"3 5"
	"3 6"
)

PIPELINE_CONFIGS=(
	"headlag"
	"headlag_pipe0"
)

export SYNTHESIS_CLOCK_PERIOD_NS="2.5"
export LSQ_NO_BYPASS=1
export LSQ_BLOOM_FILTER_LOAD=1
export LSQ_ST_ISSUE_NO_COMPARE=1
export LSQ_BLOOM_FILTER_SEQUENTIAL=1
export LSQ_BLOOM_FILTER_SEED=1

EXPERIMENT_NAME="$(basename "$0" .sh)"
OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/${EXPERIMENT_NAME}_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF >"${OUTPUT_DIR}/README.txt"
Experiment 12: Bloom filters (all kernels)
- Bloom filter type: sequential, conservative store issue
- synthesis target clock period: ${SYNTHESIS_CLOCK_PERIOD_NS} ns
- LSQ sizes: ${LSQ_SIZE[*]}
- pipeline configurations: ${PIPELINE_CONFIGS[*]}
- bloom filter hash counts and widths: ${BLOOM_FILTER_HASH_COUNTS_WIDTHS[*]}
- git revision: ${GIT_REVISION}
EOF

for PIPELINE_CONFIG in "${PIPELINE_CONFIGS[@]}"; do
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

	for HASH_COUNT_WIDTH in "${BLOOM_FILTER_HASH_COUNTS_WIDTHS[@]}"; do
		set -- $HASH_COUNT_WIDTH
		HASH_COUNT=$1
		HASH_WIDTH=$2
		export LSQ_BLOOM_FILTER_HASH_COUNT=$HASH_COUNT
		export LSQ_BLOOM_FILTER_HASH_WIDTH=$HASH_WIDTH
		FILTER_WIDTH=$((2 ** HASH_WIDTH))

		OUTPUT_SUBDIR="${OUTPUT_DIR}/${PIPELINE_CONFIG}/bf_m${FILTER_WIDTH}_k${HASH_COUNT}"
		echo "Output directory for pipeline configuration = ${PIPELINE_CONFIG}, m=${FILTER_WIDTH}, k=${HASH_COUNT}: ${OUTPUT_SUBDIR}"
		mkdir -p "${OUTPUT_SUBDIR}"

		for LSQ_SIZE in "${LSQ_SIZES[@]}"; do
			export LSQ_NUM_LDQ_ENTRIES=$LSQ_SIZE
			export LSQ_NUM_STQ_ENTRIES=$LSQ_SIZE

			echo "Running evaluation LSQ size ${LSQ_SIZE}..."
			"$RUN_EVALUATION_PY" --synth-lsqs -j 8 \
				--json "${OUTPUT_SUBDIR}/lsq_${LSQ_SIZE}_${LSQ_SIZE}.json"
		done
	done
done
