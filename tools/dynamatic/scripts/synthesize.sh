#!/bin/bash

# ============================================================================ #
# Variable definitions
# ============================================================================ #

# Script variables
DYNAMATIC_DIR=$1
SRC_DIR=$2
OUTPUT_DIR=$3
KERNEL_NAME=$4
USE_SIMPLE_BUFFERS=$5

# Binaries used during synthesis
POLYGEIST_PATH="$DYNAMATIC_DIR/polygeist/llvm-project/clang/lib/Headers/"
POLYGEIST_CLANG_BIN="$DYNAMATIC_DIR/bin/cgeist"
MLIR_OPT_BIN="$DYNAMATIC_DIR/bin/mlir-opt"
DYNAMATIC_OPT_BIN="$DYNAMATIC_DIR/bin/dynamatic-opt"
DYNAMATIC_PROFILER_BIN="$DYNAMATIC_DIR/bin/exp-frequency-profiler"
DYNAMATIC_EXPORT_DOT_BIN="$DYNAMATIC_DIR/bin/export-dot"

# Generated files
F_AFFINE="$OUTPUT_DIR/affine.mlir"
F_SCF="$OUTPUT_DIR/scf.mlir"
F_CF="$OUTPUT_DIR/std.mlir"
F_CF_TRANFORMED="$OUTPUT_DIR/std_transformed.mlir"
F_CF_DYN_TRANSFORMED="$OUTPUT_DIR/std_dyn_transformed.mlir"
F_HANDSHAKE="$OUTPUT_DIR/handshake.mlir"
F_HANDSHAKE_TRANSFORMED="$OUTPUT_DIR/handshake_transformed.mlir"
F_HANDSHAKE_BUFFERED="$OUTPUT_DIR/handshake_buffered.mlir"
F_HANDSHAKE_EXPORT="$OUTPUT_DIR/handshake_export.mlir"
F_FREQUENCIES="$OUTPUT_DIR/frequencies.csv"

# ============================================================================ #
# Helper funtions
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

# Exports Handshake-level IR to DOT using Dynamatic, then converts the DOT to
# a PNG using dot.
#   $1: mode to run the tool in; options are "visual", "legacy", "legacy-buffers"
#   $2: output filename, without extension (will use .dot and .png)
export_dot() {
  local mode=$1
  local f_dot="$OUTPUT_DIR/$2.dot"
  local f_png="$OUTPUT_DIR/$2.png"

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
# Synthesis flow
# ============================================================================ #

# Reset output directory
rm -rf "$OUTPUT_DIR" && mkdir -p "$OUTPUT_DIR"

# source -> affine level
"$POLYGEIST_CLANG_BIN" "$SRC_DIR/$KERNEL_NAME.c" -I \
  "$POLYGEIST_PATH/llvm-project/clang/lib/Headers/" --function="$KERNEL_NAME" \
  -S -O3 --memref-fullrank --raise-scf-to-affine \
  > "$F_AFFINE" 2>/dev/null
exit_on_fail "Failed to compile source to affine" "Compiled source to affine"
    
# affine level -> scf level
"$DYNAMATIC_OPT_BIN" "$F_AFFINE" --allow-unregistered-dialect \
  --name-memory-ops --analyze-memory-accesses --lower-affine-to-scf \
  --scf-simple-if-to-select --scf-rotate-for-loops \
  > "$F_SCF"
exit_on_fail "Failed to compile affine to scf" "Compiled affine to scf"

# scf level -> cf level
"$DYNAMATIC_OPT_BIN" "$F_SCF" --allow-unregistered-dialect \
  --lower-scf-to-cf > "$F_CF"
exit_on_fail "Failed to compile scf to cf" "Compiled scf to cf"

# cf transformations (standard)
"$MLIR_OPT_BIN" "$F_CF" --allow-unregistered-dialect --canonicalize --cse \
    --sccp --symbol-dce --control-flow-sink --loop-invariant-code-motion \
    --canonicalize \
    > "$F_CF_TRANFORMED"
exit_on_fail "Failed to apply standard transformations to cf" \
  "Applied standard transformations to cf"

# cf transformations (dynamatic) 
"$DYNAMATIC_OPT_BIN" "$F_CF_TRANFORMED" --allow-unregistered-dialect \
  --flatten-memref-row-major --flatten-memref-calls --arith-reduce-strength \
  --push-constants \
  > "$F_CF_DYN_TRANSFORMED"
exit_on_fail "Failed to apply Dynamatic transformations to cf" \
  "Applied Dynamatic transformations to cf"

# cf level -> handshake level
"$DYNAMATIC_OPT_BIN" "$F_CF_DYN_TRANSFORMED" --allow-unregistered-dialect \
  --lower-std-to-handshake-fpga18="id-basic-blocks" \
  --handshake-fix-arg-names="source=$SRC_DIR/$KERNEL_NAME.c" \
  > "$F_HANDSHAKE"
exit_on_fail "Failed to compile cf to handshake" "Compiled cf to handshake"

# handshake transformations
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE" --allow-unregistered-dialect \
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
    --allow-unregistered-dialect \
    --handshake-insert-buffers="buffer-size=2 strategy=cycles" \
    --handshake-infer-basic-blocks \
    > "$F_HANDSHAKE_BUFFERED"
  exit_on_fail "Failed to place simple buffers" "Placed simple buffers"
else
  # cf-level profiler
  "$DYNAMATIC_PROFILER_BIN" "$F_CF_DYN_TRANSFORMED" \
    --top-level-function="$KERNEL_NAME" \
    --input-args-file="$SRC_DIR/inputs.txt" \
    > $F_FREQUENCIES 
  exit_on_fail "Failed to profile cf-level" "Profiled cf-level"

  # Smart buffer placement
  echo_info "Running smart buffer placement"
  "$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_TRANSFORMED" \
    --allow-unregistered-dialect \
    --handshake-set-buffering-properties="version=fpga20" \
    --handshake-place-buffers="algorithm=fpga20-legacy frequencies=$OUTPUT_DIR/frequencies.csv timing-models=$DYNAMATIC_DIR/data/components.json dump-logs" \
    > "$F_HANDSHAKE_BUFFERED"
  RET=$?
  mv buffer-placement "$OUTPUT_DIR" > /dev/null 2>&1 
  if [[ $RET -ne 0 ]]; then
      echo_fatal "Failed to place smart buffers"
      exit 1
  fi
  echo_info "Placed smart buffers"
fi

# handshake canonicalization
"$DYNAMATIC_OPT_BIN" "$F_HANDSHAKE_BUFFERED" \
  --allow-unregistered-dialect --handshake-canonicalize \
  > "$F_HANDSHAKE_EXPORT"
exit_on_fail "Failed to canonicalize Handshake" "Canonicalized handshake"

# Export to DOT (one clean for viewing and one compatible with legacy)
export_dot "visual" "visual"
export_dot "legacy" "$KERNEL_NAME"

echo_info "All done!"
echo ""
