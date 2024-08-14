#!/bin/bash

# ============================================================================ #
# Function Definitions
# ============================================================================ #

# Prints some information to stdout.
#   $1: the text to print
echo_info() {
    echo "[INFO] $1"
}

# Prints a fatal error message to stdout.
#   $1: the text to print
echo_fatal() {
    echo "[FATAL] $1"
}


# Exits the script with a fatal error message if the last command that was
# called before this function failed, otherwise optionally prints an information
# message.
#   $1: fatal error message
#   $2: [optional] information message
exit_on_fail() {
    if [[ $? -ne 0 ]]; then
        if [[ ! -z $1 ]]; then
            echo_fatal "$1"
            exit 1
        fi
        echo_fatal "Failed!"
        exit 1
    else
        if [[ ! -z $2 ]]; then
            echo_info "$2"
        fi
    fi
}

echo_section() {
    echo ""
    echo "# ===----------------------------------------------------------------------=== #"
    echo "# $1"
    echo "# ===----------------------------------------------------------------------=== #"
    echo ""
}

# ============================================================================ #
# Variable definitions
# ============================================================================ #
# Script arguments
KERNEL_NAME=$1
DYNAMATIC_DIR="/home/jianliu/Projects/Dynamatic_adds_on/SwitchingEstimation/dynamatic"
SRC_DIR="./integration-test/$KERNEL_NAME"
OUTPUT_DIR="$SRC_DIR/out/comp"
DATA_LOG="$OUTPUT_DIR/profiling.log"
TRACE_LOG="$OUTPUT_DIR/trace.log"

DATA_PROFILER_BIN="$DYNAMATIC_DIR/bin/data-profiler"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"

F_CF_DYN_TRANSFORMED="$OUTPUT_DIR/cf_dyn_transformed.mlir"
F_HANDSHAKE_EXPORT="$OUTPUT_DIR/handshake_export.mlir"
F_HANDSHAKE_SWITCH="$OUTPUT_DIR/handshake_switch_test.mlir"


# ============================================================================ #
# Switching Estimation Flow
# ============================================================================ #

# Check the existence of the handshake_export.mlir file
if [ -e $F_HANDSHAKE_EXPORT ]; then
  echo "The handshake_export.mlir for $KERNEL_NAME exists"
else
  echo "[ERROR] The handshake_export.mlir file doesn't exist"
  exit 1
fi

# Run the data profiler
echo_section "[Step 1] Runing Data Profiler for ${KERNEL_NAME}"
"$DATA_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
  --top-level-function="$KERNEL_NAME" \
  --input-args-file="$OUTPUT_DIR/profiler-inputs.txt" \
  --trace-log-file="$TRACE_LOG" >/dev/null
echo_info "Data Profiling Finished"

# Run the switching estimation pass
echo_section "[Step 2] Running Switching Estiamtion Pass for ${KERNEL_NAME}"
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" \
  --switching-estimation="result-folder-path=$OUTPUT_DIR timing-models=$DYNAMATIC_DIR/data/components.json" \
 2>&1 | tee "$F_HANDSHAKE_SWITCH"



