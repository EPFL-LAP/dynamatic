#!/bin/bash

source "$1"/tools/dynamatic/scripts/utils.sh

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script arguments
DYNAMATIC_DIR=$1
SRC_DIR=$2
OUTPUT_DIR=$3
KERNEL_NAME=$4
BUFFER_ALGORITHM=$5
TARGET_CP=$6
POLYGEIST_PATH=$7
USE_SHARING=$8

POLYGEIST_CLANG_BIN="$DYNAMATIC_DIR/bin/cgeist"
CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_PROFILER_BIN="$DYNAMATIC_DIR/bin/exp-frequency-profiler"
DYNAMATIC_EXPORT_DOT_BIN="$DYNAMATIC_DIR/bin/export-dot"
DYNAMATIC_EXPORT_CFG_BIN="$DYNAMATIC_DIR/bin/export-cfg"

# Generated directories/files
COMP_DIR="$OUTPUT_DIR/comp"
F_AFFINE="$COMP_DIR/affine.mlir"
F_AFFINE_MEM="$COMP_DIR/affine_mem.mlir"
F_SCF="$COMP_DIR/scf.mlir"
F_CF="$COMP_DIR/cf.mlir"
F_CF_TRANFORMED="$COMP_DIR/cf_transformed.mlir"
F_CF_DYN_TRANSFORMED="$COMP_DIR/cf_dyn_transformed.mlir"
F_PROFILER_BIN="$COMP_DIR/$KERNEL_NAME-profile"
F_PROFILER_INPUTS="$COMP_DIR/profiler-inputs.txt"
F_HANDSHAKE="$COMP_DIR/handshake.mlir"
F_HANDSHAKE_TRANSFORMED="$COMP_DIR/handshake_transformed.mlir"
F_HANDSHAKE_BUFFERED="$COMP_DIR/handshake_buffered.mlir"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_export.mlir"
F_HW="$COMP_DIR/hw.mlir"
F_FREQUENCIES="$COMP_DIR/frequencies.csv"

# ============================================================================ #
# Helper funtions
# ============================================================================ #

# Exports Handshake-level IR to DOT using Dynamatic, then converts the DOT to
# a PNG using dot.
#   $1: input handshake-level IR filename
#   $1: output filename, without extension (will use .dot and .png)
export_dot() {
  local f_handshake="$1"
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"

  # Export to DOT
  "$DYNAMATIC_EXPORT_DOT_BIN" "$f_handshake" "--edge-style=spline" \
    > "$f_dot"
  exit_on_fail "Failed to create $2 DOT" "Created $2 DOT"

  # Convert DOT graph to PNG
  dot -Tpng "$f_dot" > "$f_png"
  exit_on_fail "Failed to convert $2 DOT to PNG" "Converted $2 DOT to PNG"
  return 0
}

export_cfg() {
  local f_cf="$1"
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"


  # Export to DOT
  "$DYNAMATIC_EXPORT_CFG_BIN" "$f_cf" \
    > "$f_dot"
  exit_on_fail "Failed to create $2 DOT" "Created $2 DOT"

  # Convert DOT graph to PNG
  dot -Tpng "$f_dot" > "$f_png"
  exit_on_fail "Failed to convert $2 DOT to PNG" "Converted $2 DOT to PNG"
  return 0
}

# ============================================================================ #
# Compilation flow
# ============================================================================ #

# Reset output directory
rm -rf "$COMP_DIR" && mkdir -p "$COMP_DIR"

# source -> affine level
"$POLYGEIST_CLANG_BIN" "$SRC_DIR/$KERNEL_NAME.c" --function="$KERNEL_NAME" \
  -I "$POLYGEIST_PATH/llvm-project/clang/lib/Headers" \
  -I "$DYNAMATIC_DIR/include" \
  -S -O3 --memref-fullrank --raise-scf-to-affine \
  > "$F_AFFINE"
exit_on_fail "Failed to compile source to affine" "Compiled source to affine"

# affine level -> pre-processing and memory analysis
"$DYNAMATIC_OPT_BIN" "$F_AFFINE" --allow-unregistered-dialect \
  --remove-polygeist-attributes \
  --func-set-arg-names="source=$SRC_DIR/$KERNEL_NAME.c" \
  --mark-memory-dependencies \
  > "$F_AFFINE_MEM"
exit_on_fail "Failed to run memory analysis" "Ran memory analysis"

# affine level -> scf level
"$DYNAMATIC_OPT_BIN" "$F_AFFINE_MEM" --lower-affine-to-scf \
  --flatten-memref-row-major --scf-simple-if-to-select \
  --scf-rotate-for-loops \
  > "$F_SCF"
exit_on_fail "Failed to compile affine to scf" "Compiled affine to scf"

# scf level -> cf level
"$DYNAMATIC_OPT_BIN" "$F_SCF" --lower-scf-to-cf > "$F_CF"
exit_on_fail "Failed to compile scf to cf" "Compiled scf to cf"

# cf transformations (standard)
"$DYNAMATIC_OPT_BIN" "$F_CF" --canonicalize --cse --sccp --symbol-dce \
    --control-flow-sink --loop-invariant-code-motion --canonicalize \
    > "$F_CF_TRANFORMED"
exit_on_fail "Failed to apply standard transformations to cf" \
  "Applied standard transformations to cf"

# cf transformations (dynamatic)
"$DYNAMATIC_OPT_BIN" "$F_CF_TRANFORMED" \
  --arith-reduce-strength="max-adder-depth-mul=1" --push-constants \
  --mark-memory-interfaces \
  > "$F_CF_DYN_TRANSFORMED"
exit_on_fail "Failed to apply Dynamatic transformations to cf" \
  "Applied Dynamatic transformations to cf"

# cf level -> handshake level
"$DYNAMATIC_OPT_BIN" "$F_CF_DYN_TRANSFORMED" --lower-cf-to-handshake \
  > "$F_HANDSHAKE"
exit_on_fail "Failed to compile cf to handshake" "Compiled cf to handshake"

# handshake transformations
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE" \
  --handshake-analyze-lsq-usage --handshake-replace-memory-interfaces \
  --handshake-minimize-cst-width --handshake-optimize-bitwidths \
  --handshake-materialize --handshake-infer-basic-blocks \
  > "$F_HANDSHAKE_TRANSFORMED"
exit_on_fail "Failed to apply transformations to handshake" \
  "Applied transformations to handshake"


# Credit-based sharing
if [[ $USE_SHARING -ne 0 ]]; then
  BUFFER_PLACEMENT_PASS="credit-based-sharing"
  echo_info "Set to apply credit-based sharing after buffer placement."
else
  BUFFER_PLACEMENT_PASS="handshake-place-buffers"
fi

# Buffer placement
if [[ "$BUFFER_ALGORITHM" == "on-merges" ]]; then
  # Simple buffer placement
  echo_info "Running simple buffer placement (on-merges)."
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-set-buffering-properties="version=fpga20" \
    --$BUFFER_PLACEMENT_PASS="algorithm=$BUFFER_ALGORITHM timing-models=$DYNAMATIC_DIR/data/components.json" \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place simple buffers" "Placed simple buffers"
else
  # Compile kernel's main function to extract profiling information
  "$CLANGXX_BIN" "$SRC_DIR/$KERNEL_NAME.c" -D PRINT_PROFILING_INFO -I \
    "$DYNAMATIC_DIR/include" -Wno-deprecated -o "$F_PROFILER_BIN"
  exit_on_fail "Failed to build kernel for profiling" "Built kernel for profiling"

  "$F_PROFILER_BIN" > "$F_PROFILER_INPUTS"
  exit_on_fail "Failed to kernel for profiling" "Ran kernel for profiling"

  # cf-level profiler
  "$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
    --top-level-function="$KERNEL_NAME" --input-args-file="$F_PROFILER_INPUTS" \
    > $F_FREQUENCIES
  exit_on_fail "Failed to profile cf-level" "Profiled cf-level"

  # Smart buffer placement
  echo_info "Running smart buffer placement with CP = $TARGET_CP and algorithm = '$BUFFER_ALGORITHM'"
  cd "$COMP_DIR"
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-set-buffering-properties="version=fpga20" \
    --$BUFFER_PLACEMENT_PASS="algorithm=$BUFFER_ALGORITHM frequencies=$F_FREQUENCIES timing-models=$DYNAMATIC_DIR/data/components.json target-period=$TARGET_CP timeout=300 dump-logs blif-files=$DYNAMATIC_DIR/data/aig/" \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place smart buffers" "Placed smart buffers"
  cd - > /dev/null
fi

# handshake canonicalization
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" \
  --handshake-canonicalize \
  --handshake-hoist-ext-instances \
  > "$F_HANDSHAKE_EXPORT"
exit_on_fail "Failed to canonicalize Handshake" "Canonicalized handshake"

# Export to DOT
export_dot "$F_HANDSHAKE_EXPORT" "$KERNEL_NAME"
export_cfg "$F_CF_DYN_TRANSFORMED" "${KERNEL_NAME}_CFG"

# handshake level -> hw level
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_EXPORT" --lower-handshake-to-hw \
  > "$F_HW"
exit_on_fail "Failed to lower to HW" "Lowered to HW"

echo_info "Compilation succeeded"
