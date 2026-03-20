#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
RUN_EVALUATION_PY="${SCRIPT_DIR}/../tools/evaluation/run_evaluation.py"
GIT_REVISION="$(git rev-parse --short HEAD)"

# HACK: only run remaining
# CLOCK_PERIODS=(6 5 4 3 2.5 2)
CLOCK_PERIODS=(2.5 2)
LSQ_SIZES=(4 6 8 10 12 16 20)

OUTPUT_DIR="${SCRIPT_DIR}/../eval_results/02_synthesis_target_clock_period_${GIT_REVISION}"
mkdir -p "${OUTPUT_DIR}"
cat <<EOF >"${OUTPUT_DIR}/README.txt"
Experiment 2: Synthesis Target Clock Period
- remaining run only: 2.5ns with size 20, 2ns full run
- git revision: ${GIT_REVISION}
- full pipelining
EOF

for CP in "${CLOCK_PERIODS[@]}"; do
    CP_FORMATTED="${CP//./p}"
	OUTPUT_DIR_CP="${OUTPUT_DIR}/${CP_FORMATTED}ns"
	echo "Output directory for clock period = ${CP} ns: ${OUTPUT_DIR_CP}"
	mkdir -p "${OUTPUT_DIR_CP}"

	for LSQ_SIZE in "${LSQ_SIZES[@]}"; do
    	if [[ "$CP" == "2.5" && "$LSQ_SIZE" != "20" ]]; then
            # HACK: only run remaining
    		continue
    	fi

		echo "Running evaluation with clock period = ${CP} ns; LSQ size = $LSQ_SIZE"
		export SYNTHESIS_CLOCK_PERIOD_NS="${CP}"
		export LSQ_NUM_LDQ_ENTRIES=$LSQ_SIZE
		export LSQ_NUM_STQ_ENTRIES=$LSQ_SIZE
		export LSQ_PIPE_COMP_EN=1
		export LSQ_PIPE0_EN=1
		export LSQ_PIPE1_EN=1
		export LSQ_HEAD_LAG_EN=1

		"$RUN_EVALUATION_PY" --synth-lsqs -j 16 \
		    --json "${OUTPUT_DIR_CP}/lsq_${LSQ_SIZE}_${LSQ_SIZE}.json"
	done
done
