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
USE_SIMPLE_BUFFERS=$5

# Binaries used during compilation
POLYGEIST_PATH="$DYNAMATIC_DIR/polygeist/llvm-project/clang/lib/Headers/"
POLYGEIST_CLANG_BIN="$DYNAMATIC_DIR/bin/cgeist"
CLANGXX_BIN="$DYNAMATIC_DIR/bin/clang++"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_PROFILER_BIN="$DYNAMATIC_DIR/bin/exp-frequency-profiler"
DYNAMATIC_EXPORT_DOT_BIN="$DYNAMATIC_DIR/bin/export-dot"

# Generated directories/files
COMP_DIR="$OUTPUT_DIR/comp"
F_AFFINE="$COMP_DIR/affine.mlir"
F_AFFINE_MEM="$COMP_DIR/affine_mem.mlir"
F_SCF="$COMP_DIR/scf.mlir"
F_CF="$COMP_DIR/std.mlir"
F_CF_TRANFORMED="$COMP_DIR/std_transformed.mlir"
F_CF_DYN_TRANSFORMED="$COMP_DIR/std_dyn_transformed.mlir"
F_PROFILER_BIN="$COMP_DIR/$KERNEL_NAME-profile"
F_PROFILER_INPUTS="$COMP_DIR/profiler-inputs.txt"
F_HANDSHAKE="$COMP_DIR/handshake.mlir"
F_HANDSHAKE_TRANSFORMED="$COMP_DIR/handshake_transformed.mlir"
F_HANDSHAKE_BUFFERED="$COMP_DIR/handshake_buffered.mlir"
F_HANDSHAKE_EXPORT="$COMP_DIR/handshake_export.mlir"
F_FREQUENCIES="$COMP_DIR/frequencies.csv"

# ============================================================================ #
# Helper funtions
# ============================================================================ #

# Exports Handshake-level IR to DOT using Dynamatic, then converts the DOT to
# a PNG using dot.
#   $1: mode to run the tool in; options are "visual", "legacy", "legacy-buffers"
#   $2: output filename, without extension (will use .dot and .png)
export_dot() {
  local mode=$1
  local f_dot="$COMP_DIR/$2.dot"
  local f_png="$COMP_DIR/$2.png"

  # Export to DOT
  "$DYNAMATIC_EXPORT_DOT_BIN" "$F_HANDSHAKE_EXPORT" "--mode=$mode" \
      "--edge-style=spline" \
      "--timing-models=$DYNAMATIC_DIR/data/components.json" \
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
"$POLYGEIST_CLANG_BIN" "$SRC_DIR/$KERNEL_NAME.c" -I \
  "$POLYGEIST_PATH/llvm-project/clang/lib/Headers/" --function="$KERNEL_NAME" \
  -S -O3 --memref-fullrank --raise-scf-to-affine \
  > "$F_AFFINE" 2>/dev/null
exit_on_fail "Failed to compile source to affine" "Compiled source to affine"

# affine level -> pre-processing and memory analysis
"$DYNAMATIC_OPT_BIN" "$F_AFFINE" --allow-unregistered-dialect \
  --remove-polygeist-attributes --mark-memory-dependencies \
  --mark-memory-interfaces \
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
"$DYNAMATIC_OPT_BIN" "$F_CF_TRANFORMED" --flatten-memref-calls \
  --arith-reduce-strength="max-adder-depth-mul=1" --push-constants \
  > "$F_CF_DYN_TRANSFORMED"
exit_on_fail "Failed to apply Dynamatic transformations to cf" \
  "Applied Dynamatic transformations to cf"

# cf level -> handshake level
"$DYNAMATIC_OPT_BIN" "$F_CF_DYN_TRANSFORMED" --lower-std-to-handshake-fpga18 \
  --handshake-fix-arg-names="source=$SRC_DIR/$KERNEL_NAME.c" \
  > "$F_HANDSHAKE"
exit_on_fail "Failed to compile cf to handshake" "Compiled cf to handshake"

# handshake transformations
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE" \
  --handshake-concretize-index-type="width=32" \
  --handshake-minimize-cst-width --handshake-optimize-bitwidths="legacy" \
  --handshake-materialize-forks-sinks --handshake-infer-basic-blocks \
  > "$F_HANDSHAKE_TRANSFORMED"    
exit_on_fail "Failed to apply transformations to handshake" \
  "Applied transformations to handshake"

# Buffer placement
if [[ $USE_SIMPLE_BUFFERS -ne 0 ]]; then
  # Simple buffer placement
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-place-buffers="algorithm=on-merges" \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place simple buffers" "Placed simple buffers"
else
  # Compile kernel's main function to extract profiling information
  "$CLANGXX_BIN" "$SRC_DIR/$KERNEL_NAME.c" -D PRINT_PROFILING_INFO \
    -Wno-deprecated -o "$F_PROFILER_BIN"
  exit_on_fail "Failed to build kernel for profiling" "Built kernel for profiling" 

  "$F_PROFILER_BIN" > "$F_PROFILER_INPUTS"
  exit_on_fail "Failed to kernel for profiling" "Ran kernel for profiling" 

  # cf-level profiler
  "$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
    --top-level-function="$KERNEL_NAME" --input-args-file="$F_PROFILER_INPUTS" \
    > $F_FREQUENCIES 
  exit_on_fail "Failed to profile cf-level" "Profiled cf-level"

  # Smart buffer placement
  echo_info "Running smart buffer placement"
  cd "$COMP_DIR"
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --handshake-set-buffering-properties="version=fpga20" \
    --handshake-place-buffers="algorithm=fpga20-legacy frequencies=$F_FREQUENCIES timing-models=$DYNAMATIC_DIR/data/components.json timeout=300 dump-logs" \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place smart buffers" "Placed smart buffers"
  cd - > /dev/null
fi

# handshake canonicalization
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" --handshake-canonicalize \
  > "$F_HANDSHAKE_EXPORT"
exit_on_fail "Failed to canonicalize Handshake" "Canonicalized handshake"

# Export to DOT (one clean for viewing and one compatible with legacy)
export_dot "visual" "visual"
export_dot "legacy" "$KERNEL_NAME"
echo_info "Compilation succeeded"
