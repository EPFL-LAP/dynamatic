#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_ROOT="$(realpath "${SCRIPT_DIR}/..")"
RUN_EVALUATION_PY="${REPO_ROOT}/tools/evaluation/run_evaluation.py"
LSQ_GENERATOR_PY="${REPO_ROOT}/tools/backend/lsq-generator-python/lsq-generator.py"
SYNTHESIZE_SH="${REPO_ROOT}/tools/dynamatic/scripts/synthesize.sh"
GIT_REVISION="$(git rev-parse --short HEAD)"
KERNEL="histogram"
KERNEL_OUTPUT_DIR="${REPO_ROOT}/integration-test/histogram/out"
KERNEL_HDL_DIR="${KERNEL_OUTPUT_DIR}/hdl"
LSQ_JSON="${KERNEL_HDL_DIR}/handshake_lsq_lsq1.json"

# sweep parameters
LSQ_SIZES=(
  4
  6
  8
  10
  12
  16
  20
  24
  28
  32
  40
  48
  56
  64
)

ADDR_WIDTHS=(
  4
  6
  8
  10
  12
  64
  20
  24
  28
  32
  40
  48
  56
  64
)

PIPELINE_CONFIGS=(
	"headlag"
	# "headlag_pipe0"
)

# k, log2(m)
BLOOM_FILTER_HASH_COUNTS_WIDTHS=(
	"1 3"
	"2 4"
	"3 5"
	"4 6"
)

# fixed parameters
export LSQ_NO_BYPASS=1
export LSQ_ST_ISSUE_NO_COMPARE=1
export LSQ_BLOOM_FILTER_LOAD=1
export LSQ_BLOOM_FILTER_SEQUENTIAL=1
export LSQ_BLOOM_FILTER_SEED=1

EXPERIMENT_NAME="$(basename "$0" .sh)"
OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/${EXPERIMENT_NAME}_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF >"${OUTPUT_DIR}/README.txt"
Experiment 13: Bloom filter sweep with varying address widths and LSQ sizes
- Bloom filter type: sequential, conservative store issue
- sweep parameters:
  - LSQ sizes: ${LSQ_SIZES[*]} entries
  - address widths: ${ADDR_WIDTHS[*]} bits
- pipeline configurations: ${PIPELINE_CONFIGS[*]}
- bloom filter hash counts and widths: ${BLOOM_FILTER_HASH_COUNTS_WIDTHS[*]}
- git revision: ${GIT_REVISION}
EOF

echo "Running evaluation (to generate LSQ JSON)..."
"$RUN_EVALUATION_PY" --no-synth -j 8 \
	--kernel "${KERNEL}" \
	--json /dev/null

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

    for ADDR_WIDTH in "${ADDR_WIDTHS[@]}"; do
      # Change the address width in the LSQ configuration
      echo "Changing address width to ${ADDR_WIDTH} in LSQ JSON configuration..."
      python3 - "$LSQ_JSON" "$ADDR_WIDTH" <<EOF
import json
import sys

lsq_json = sys.argv[1]
addr_width = int(sys.argv[2])

with open(lsq_json, 'r') as f:
    lsq_config = json.load(f)
lsq_config['addrWidth'] = addr_width
with open(lsq_json, 'w') as f:
    json.dump(lsq_config, f)
EOF

      for LSQ_SIZE in "${LSQ_SIZES[@]}"; do
        OUTPUT_SUBDIR="${OUTPUT_DIR}/${PIPELINE_CONFIG}/bf_m${FILTER_WIDTH}_k${HASH_COUNT}/addr${ADDR_WIDTH}/lsq_${LSQ_SIZE}"

        echo "Output directory for"
        echo "  - pipeline configuration: ${PIPELINE_CONFIG}"
        echo "  - bloom filter:           m=${FILTER_WIDTH}, k=${HASH_COUNT}"
        echo "  - address width:          ${ADDR_WIDTH}"
        echo "  - LSQ size:               ${LSQ_SIZE}"
        echo "  => ${OUTPUT_SUBDIR}"

        mkdir -p "${OUTPUT_SUBDIR}"

        export LSQ_NUM_LDQ_ENTRIES=$LSQ_SIZE
        export LSQ_NUM_STQ_ENTRIES=$LSQ_SIZE

        echo "Running the LSQ generator with modified address width..."
        python3 "$LSQ_GENERATOR_PY" \
          -o "$KERNEL_HDL_DIR" \
          -c "$LSQ_JSON"

        echo "Running LSQ synthesis..."
        SYNTH_DIR="${KERNEL_OUTPUT_DIR}/synth_sweep"
        "$SYNTHESIZE_SH" \
          "$REPO_ROOT" \
          "$KERNEL_OUTPUT_DIR" \
          "handshake_lsq_lsq1_core" \
          "4.000" \
          "2.000" \
          "$SYNTH_DIR"

        echo "Copying artifacts..."
        cp "$SYNTH_DIR/report.txt" "$OUTPUT_SUBDIR"
        cp "$SYNTH_DIR/"*.rpt "$OUTPUT_SUBDIR"
        ls "$OUTPUT_SUBDIR"
      done
    done
	done
done
